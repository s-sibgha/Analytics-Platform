from __future__ import annotations

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
frontend_dir = Path(__file__).resolve().parent
for _p in (root_dir, frontend_dir):
    _p_str = str(_p)
    if _p_str in sys.path:
        sys.path.remove(_p_str)
    sys.path.insert(0, _p_str)




import hashlib
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import os
import re
import tempfile
import threading
import uuid

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from utils.error_logging import log_exception
from core.themes import (
    THEMES,
    DEFAULT_THEME_KEY,
    Theme,
    inject_static_theme_link,
    sync_theme_state_marker,
    apply_theme_now,
    bootstrap_static_js,
)
from core.settings import (
    APP_TITLE,
    APP_ICON,
    SUPPORTED_UPLOAD_TYPES,
    READINESS_BANDS,
    MIN_AUTO_CONFIDENCE,
    APP_LOGO_FILENAME,
    APP_ICON_LOGO_FILENAME,
    ENTERPRISE_COPY_MAP,
)
from core.column_registry import ColumnRegistry
from core.type_inference import infer_dataframe
from engine.cleaner import SafeCleaningEngine, CleaningResult
from core.schema_models import ColumnProfile
from core.roles import (
    ROLE_RECORD_ID,
    ROLE_REGISTRATION_DATE,
    ROLE_STATUS,
    ROLE_ZONE,
    ROLE_CIRCLE,
    ROLE_DIVISION,
    ROLE_SUBDIVISION
)


from core.parquet_safety import (
    read_uploaded_file,
    prepare_df_for_parquet,
    sanitize_for_parquet,
    write_parquet_safely,
)
import gc
from engine.domain_detection import detect_domain, DOMAIN_UNKNOWN
from utils.audit_log import AuditLog

try:
    from FRONTEND.components.sidebar import render as render_sidebar
except ImportError:
    render_sidebar = None


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE CONTRACT
# ══════════════════════════════════════════════════════════════════════════════

_REQUIRED_SESSION_STATE_KEYS: Tuple[str, ...] = (
    "uploaded_dataframe",
    "filtered_dataframe",
    "column_registry",
    "active_filters",
    "selected_chart",
    "dashboard_state",
    "analytics_results",
    "visualization_cache",
    "domain_detection",
    "audit_results",
    "theme",
    "active_tab",
    "drill_breadcrumbs",
)


def _default_session_state() -> Dict[str, Any]:
    """Returns the complete default payload for every platform-wide
    session_state key. Called fresh on every initialize_session_state()
    invocation so defaults are never stale references shared across reruns."""
    return {
        "uploaded_dataframe": None,
        "filtered_dataframe": None,
        "column_registry": None,
        "column_type_overrides": {},
        "active_filters": {},
        "selected_chart": None,
        "dashboard_state": {},
        "analytics_results": {},
        "visualization_cache": {},
        "domain_detection": (DOMAIN_UNKNOWN, 0.0),
        "audit_results": {},
        "theme": DEFAULT_THEME_KEY,
        "active_tab": "landing",
        "drill_breadcrumbs": [],
        "workspace_name": "Default Workspace",
        "workspace_profiles": {},
        "cleaned_dataframe": None,
        "analytics_ready_dataframe": None,
        "column_profiles": [],
        "ingestion_warnings": [],
        "file_fingerprint": None,
        "active_filename": None,
        "uploaded_files_store": {},
        "readiness_score": 0,
        "readiness_band": "Critical",
        "readiness_recommendations": [],
        "notifications": [],
        "pinned_kpis": [],
        "pinned_charts": [],
        "bookmarked_dashboards": [],
        "favorite_reports": [],
        "last_ingestion_timestamp": None,
        # NEW — Milestone 11: one-time bootstrap guard for the static JS
        # loader. Additive-only key; harmless if absent for any consumer
        # that predates this remediation.
        "_static_js_bootstrapped": False,
        # NEW — one-time bootstrap guard for the static CSS loader,
        # mirroring the JS guard above. See main()'s static-asset-bridge
        # comment for why this must only ever fire once per session.
        "_static_css_bootstrapped": False,
        # NEW — hard cache-invalidation epoch, bumped by
        # schema_mapping._commit_pending_mappings on every applied
        # mapping batch. Consumed by 1_dashboard.py's cache-key builder
        # so a registry.version collision can never mask a mapping
        # change served from Streamlit's process-wide st.cache_data store.
        "_registry_mutation_epoch": 0,
    }
    



def initialize_session_state() -> None:
    """
    Idempotently initializes every platform-wide session_state key. Safe to
    call on every rerun — setdefault semantics ensure existing state is
    never clobbered by a subsequent call.
    """
    defaults = _default_session_state()
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    for key in _REQUIRED_SESSION_STATE_KEYS:
        if key not in st.session_state:
            st.session_state[key] = defaults.get(key)


def _clear_active_dataset() -> None:
    """Resets every dataset-derived session_state key back to its default
    while preserving user-level preferences (theme, workspace name,
    static-JS bootstrap flag)."""
    defaults = _default_session_state()
    preserve: set = {"theme", "workspace_name", "_static_js_bootstrapped", "_static_css_bootstrapped", "_registry_mutation_epoch",}
    for key, value in defaults.items():
        if key not in preserve:
            st.session_state[key] = value
    _register_active_parquet(None)
    _cleanup_stale_parquets(keep_fingerprint=None)

# ══════════════════════════════════════════════════════════════════════════════
# INGESTION PIPELINE — CACHED PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

_PARQUET_TEMP_DIR = Path(tempfile.gettempdir()) / "kesco_parquet_cache"
_PARQUET_TEMP_DIR.mkdir(parents=True, exist_ok=True)

_HEADER_SANITIZE_RE = re.compile(r"[^\w]+")


def _sanitize_header(raw: str, seen: Dict[str, int]) -> str:
    """Strips whitespace and replaces non-alphanumeric characters with `_`
    so downstream DuckDB SQL identifier construction (duckdb_executor.py's
    _sanitize_identifier) never sees a raw header with slashes/spaces/
    symbols. Disambiguates collisions produced by sanitization itself."""
    cleaned = _HEADER_SANITIZE_RE.sub("_", str(raw).strip()).strip("_")
    if not cleaned:
        cleaned = "column"
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    seen[cleaned] = seen.get(cleaned, 0) + 1
    return cleaned if seen[cleaned] == 1 else f"{cleaned}_{seen[cleaned]}"

def _compute_file_fingerprint(file_bytes: bytes, filename: str) -> str:
    """Deterministic SHA-256 fingerprint of an uploaded file's content plus
    name, used as the caching key across the ingestion/inference/cleaning
    pipeline and as the change-detection signal in the upload widget."""
    hasher = hashlib.sha256()
    hasher.update(filename.encode("utf-8", errors="ignore"))
    hasher.update(file_bytes)
    return hasher.hexdigest()
class _BytesUploadShim:
    """Minimal adapter so read_uploaded_file() can accept raw bytes +
    filename (already extracted upstream by app.py's caching layer)
    without needing the original Streamlit UploadedFile object."""
    __slots__ = ("name", "_data")

    def __init__(self, data: bytes, filename: str) -> None:
        self.name = filename
        self._data = data

    def getvalue(self) -> bytes:
        return self._data

# def convert_upload_to_parquet(file_bytes: bytes, filename: str, fingerprint: str) -> Dict[str, Any]:
#     """
#     Converts raw uploaded bytes to a compressed Parquet file on disk, keyed
#     by content fingerprint. Every read/sanitize/write step below is wrapped
#     in its own isolated try/except so a failure at any single stage is
#     caught, logged, and surfaced as a structured `{"error": ...}` result —
#     never as an unhandled exception, and never with a local variable left
#     unbound for a later `finally`/cleanup step to trip over. Never raises.
#     """
#     suffix = Path(filename).suffix.lower().lstrip(".")
#     # Defensive re-creation — /tmp on Streamlit Cloud is ephemeral and can
#     # be reclaimed mid-session; import-time mkdir() alone is not durable.
#     _PARQUET_TEMP_DIR.mkdir(parents=True, exist_ok=True)
#     parquet_path = str(_PARQUET_TEMP_DIR / f"{fingerprint}.parquet")

#     if os.path.exists(parquet_path):
#         try:
#             meta = pq.read_metadata(parquet_path)
#             return {
#                 "parquet_path": parquet_path,
#                 "row_count": meta.num_rows,
#                 "column_names": [f.name for f in pq.read_schema(parquet_path)],
#                 "warnings": [],
#             }
#         except Exception:  # noqa: BLE001 — corrupted/partial cache hit, fall through and reconvert
#             pass

#     warnings: List[str] = []
#     tmp_source_path: Optional[str] = None
#     # CRITICAL: initialized unconditionally, before any branch or try block,
#     # so a `finally`/cleanup reference below can NEVER raise UnboundLocalError
#     # regardless of which branch executes or where an exception originates.
#     df_working: Optional[pd.DataFrame] = None

#     if suffix not in ("csv", "xlsx", "xls"):
#         return {
#             "error": f"Unsupported file extension '.{suffix}'. "
#                      f"Supported types: {', '.join(SUPPORTED_UPLOAD_TYPES)}."
#         }

#     try:
#         if suffix == "csv":
#             # DuckDB's native CSV -> Parquet pushdown remains the fastest,
#             # lowest-RAM path for well-formed CSVs (zero pandas
#             # materialization). Tried FIRST; only the fallback branch
#             # (irregular/ragged CSVs DuckDB's sniffer rejects) routes
#             # through the mixed-type-safe pandas pipeline.
#             tmp_fd, tmp_source_path = tempfile.mkstemp(suffix=".csv", dir=str(_PARQUET_TEMP_DIR))
#             with os.fdopen(tmp_fd, "wb") as fh:
#                 fh.write(file_bytes)

#             con = duckdb.connect(database=":memory:")
#             try:
#                 con.execute(
#                     "COPY (SELECT * FROM read_csv_auto(?, HEADER=TRUE, UNION_BY_NAME=TRUE, "
#                     "IGNORE_ERRORS=TRUE)) TO ? (FORMAT PARQUET, COMPRESSION 'ZSTD', "
#                     "ROW_GROUP_SIZE 128000)",
#                     [tmp_source_path, parquet_path],
#                 )
#             except Exception as duckdb_exc:  # noqa: BLE001
#                 log_exception(
#                     "app.convert_upload_to_parquet.duckdb_csv", duckdb_exc,
#                     context={"filename": filename},
#                 )
#                 warnings.append("CSV required a fallback parser (irregular formatting detected).")

#                 # ── Isolated read step ──────────────────────────────────
#                 try:
#                     df_working = read_uploaded_file(uploaded_file=_BytesUploadShim(file_bytes, filename))
#                 except Exception as read_exc:  # noqa: BLE001
#                     log_exception(
#                         "app.convert_upload_to_parquet.csv_fallback_read", read_exc,
#                         context={"filename": filename},
#                     )
#                     return {"error": f"Failed to parse CSV '{filename}' after fallback: {read_exc}"}

#                 # ── Isolated sanitize step (explicit, per requirement) ──
#                 try:
#                     df_working = sanitize_for_parquet(df_working)
#                 except Exception as sanitize_exc:  # noqa: BLE001
#                     log_exception(
#                         "app.convert_upload_to_parquet.csv_fallback_sanitize", sanitize_exc,
#                         context={"filename": filename},
#                     )
#                     warnings.append(
#                         "Data sanitization encountered an issue; proceeding with best-effort cleanup."
#                     )

#                 # ── Isolated write step (sanitize already applied above,
#                 # so skip the redundant internal sanitize pass) ─────────
#                 try:
#                     write_result = write_parquet_safely(
#                         df_working, parquet_path,
#                         compression="zstd", row_group_size=128_000, use_dictionary=True,
#                         sanitize=False,
#                     )
#                     if write_result["engine_used"] == "fastparquet":
#                         warnings.append("Parquet write used the fastparquet fallback engine.")
#                 except Exception as write_exc:  # noqa: BLE001
#                     log_exception(
#                         "app.convert_upload_to_parquet.csv_fallback_write", write_exc,
#                         context={"filename": filename},
#                     )
#                     return {"error": f"Failed to write Parquet for '{filename}': {write_exc}"}
#             finally:
#                 con.close()
def convert_upload_to_parquet(file_bytes: bytes, filename: str, fingerprint: str) -> Dict[str, Any]:
    suffix = Path(filename).suffix.lower().lstrip(".")
    _PARQUET_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = str(_PARQUET_TEMP_DIR / f"{fingerprint}.parquet")

    if os.path.exists(parquet_path):
        try:
            meta = pq.read_metadata(parquet_path)
            return {
                "parquet_path": parquet_path,
                "row_count": meta.num_rows,
                "column_names": [f.name for f in pq.read_schema(parquet_path)],
                "warnings": [],
            }
        except Exception:  # noqa: BLE001
            pass

    warnings: List[str] = []
    tmp_source_path: Optional[str] = None
    df_working: Optional[pd.DataFrame] = None

    if suffix not in ("csv", "xlsx", "xls"):
        return {
            "error": f"Unsupported file extension '.{suffix}'. "
                     f"Supported types: {', '.join(SUPPORTED_UPLOAD_TYPES)}."
        }

    try:
        if suffix == "csv":
            tmp_fd, tmp_source_path = tempfile.mkstemp(suffix=".csv", dir=str(_PARQUET_TEMP_DIR))
            with os.fdopen(tmp_fd, "wb") as fh:
                fh.write(file_bytes)

            con = duckdb.connect(database=":memory:")
            try:
                # FIX: Place 'parquet_path' directly into SQL query string rather than binding via '?'
                con.execute(
                    f"COPY (SELECT * FROM read_csv_auto(?, HEADER=TRUE, UNION_BY_NAME=TRUE, "
                    f"IGNORE_ERRORS=TRUE)) TO '{parquet_path}' (FORMAT PARQUET, COMPRESSION 'ZSTD', "
                    f"ROW_GROUP_SIZE 128000)",
                    [tmp_source_path],
                )
            except Exception as duckdb_exc:  # noqa: BLE001
                log_exception(
                    "app.convert_upload_to_parquet.duckdb_csv", duckdb_exc,
                    context={"filename": filename},
                )
                warnings.append("CSV required a fallback parser (irregular formatting detected).")

                try:
                    df_working = read_uploaded_file(uploaded_file=_BytesUploadShim(file_bytes, filename))
                except Exception as read_exc:  # noqa: BLE001
                    log_exception(
                        "app.convert_upload_to_parquet.csv_fallback_read", read_exc,
                        context={"filename": filename},
                    )
                    return {"error": f"Failed to parse CSV '{filename}' after fallback: {read_exc}"}

                try:
                    df_working = sanitize_for_parquet(df_working)
                except Exception as sanitize_exc:  # noqa: BLE001
                    log_exception(
                        "app.convert_upload_to_parquet.csv_fallback_sanitize", sanitize_exc,
                        context={"filename": filename},
                    )
                    warnings.append(
                        "Data sanitization encountered an issue; proceeding with best-effort cleanup."
                    )

                try:
                    write_result = write_parquet_safely(
                        df_working, parquet_path,
                        compression="zstd", row_group_size=128_000, use_dictionary=True,
                        sanitize=False,
                    )
                    if write_result.get("engine_used") == "fastparquet":
                        warnings.append("Parquet write used the fastparquet fallback engine.")
                except Exception as write_exc:  # noqa: BLE001
                    log_exception(
                        "app.convert_upload_to_parquet.csv_fallback_write", write_exc,
                        context={"filename": filename},
                    )
                    return {"error": f"Failed to write Parquet for '{filename}': {write_exc}"}
            finally:
                con.close()

    finally:
        # Guarantee tmp_source_path cleanup regardless of success or exception
        if tmp_source_path and os.path.exists(tmp_source_path):
            try:
                os.remove(tmp_source_path)
            except OSError:
                pass
        elif suffix in ("xlsx", "xls"):
            # ── Isolated read step ───────────────────────────────────────
            try:
                df_working = read_uploaded_file(uploaded_file=_BytesUploadShim(file_bytes, filename))
            except Exception as read_exc:  # noqa: BLE001
                log_exception(
                    "app.convert_upload_to_parquet.excel_read", read_exc,
                    context={"filename": filename},
                )
                return {"error": f"Failed to parse Excel file '{filename}': {read_exc}"}

            # ── Isolated sanitize step — handles messy REMARKS/COMMENTS
            # columns mixing text and bare numeric codes without raising
            # on mixed dtypes. ────────────────────────────────────────────
            try:
                df_working = sanitize_for_parquet(df_working)
            except Exception as sanitize_exc:  # noqa: BLE001
                log_exception(
                    "app.convert_upload_to_parquet.excel_sanitize", sanitize_exc,
                    context={"filename": filename},
                )
                warnings.append(
                    "Data sanitization encountered an issue; proceeding with best-effort cleanup."
                )

            # ── Isolated write step (sanitize already applied above) ────
            try:
                write_result = write_parquet_safely(
                    df_working, parquet_path,
                    compression="zstd", row_group_size=128_000, use_dictionary=True,
                    sanitize=False,
                )
                if write_result["engine_used"] == "fastparquet":
                    warnings.append("Parquet write used the fastparquet fallback engine.")
            except Exception as write_exc:  # noqa: BLE001
                log_exception(
                    "app.convert_upload_to_parquet.excel_write", write_exc,
                    context={"filename": filename},
                )
                return {"error": f"Failed to write Parquet for '{filename}': {write_exc}"}

        # ── Immediate RAM cleanup — the DataFrame is fully persisted to
        # disk at this point; holding the in-memory reference any longer
        # only risks a Streamlit session memory spike on large uploads.
        if df_working is not None:
            del df_working
            df_working = None
            gc.collect()

        # ── Header sanitization — schema-only rewrite ───────────────────
        try:
            schema = pq.read_schema(parquet_path)
            seen: Dict[str, int] = {}
            sanitized = [_sanitize_header(name, seen) for name in schema.names]
            if sanitized != schema.names:
                table = pq.read_table(parquet_path).rename_columns(sanitized)
                pq.write_table(
                    table, parquet_path,
                    compression="zstd", row_group_size=128_000, use_dictionary=True,
                )
                del table
                gc.collect()
                warnings.append("Column headers were sanitized for SQL/DuckDB compatibility.")
        except Exception as header_exc:  # noqa: BLE001
            log_exception(
                "app.convert_upload_to_parquet.header_sanitize", header_exc,
                context={"filename": filename},
            )
            return {"error": f"Failed to sanitize Parquet schema headers for '{filename}': {header_exc}"}

        if not os.path.exists(parquet_path) or os.path.getsize(parquet_path) == 0:
            return {
                "error": f"Parquet write for '{filename}' produced an empty or missing output "
                         f"file (possible disk-full condition or interrupted write on the "
                         f"cloud filesystem). Please retry the upload."
            }

        meta = pq.read_metadata(parquet_path)
        if meta.num_rows == 0:
            return {"error": f"'{filename}' converted successfully but contains zero data rows."}

        return {
            "parquet_path": parquet_path,
            "row_count": meta.num_rows,
            "column_names": sanitized,
            "warnings": warnings,
        }

    except Exception as exc:  # noqa: BLE001 — absolute final safety net
        log_exception("app.convert_upload_to_parquet", exc, context={"filename": filename})
        return {"error": f"Parquet conversion failed: {exc}"}

    finally:
        # Defensive `locals()` guard: even if a future edit moves the
        # `df_working = None` initialization out of its unconditional
        # top-of-function position, this cleanup can never raise
        # UnboundLocalError again.
        _pending_df = locals().get("df_working")
        if _pending_df is not None:
            del _pending_df
            gc.collect()
        if tmp_source_path and os.path.exists(tmp_source_path):
            try:
                os.remove(tmp_source_path)
            except OSError:
                pass


@st.cache_data(show_spinner=False)
def _load_uploaded_file(
    file_bytes: bytes, filename: str, fingerprint: str
) -> Tuple[Optional[pd.DataFrame], List[str], Optional[str]]:
    """
    Converts the upload straight to Parquet on disk, then reads the Parquet
    back through DuckDB for the existing pandas-based Inference -> Registry
    -> Cleaning stages, which still expect a DataFrame. Returns
    (dataframe_or_None, warnings, parquet_path_or_None). Never raises —
    every failure path (conversion error, unreadable Parquet, empty result)
    degrades to a structured `(None, warnings, ...)` tuple so Streamlit's
    `@st.cache_data` layer never caches a partially-constructed or crashed
    result.
    """
    warnings: List[str] = []

    try:
        conversion = convert_upload_to_parquet(file_bytes, filename, fingerprint)
    except Exception as exc:  # noqa: BLE001 — belt-and-suspenders; convert_upload_to_parquet
                              # already never raises, but caching layers must never propagate.
        log_exception("app._load_uploaded_file.convert", exc, context={"filename": filename})
        return None, [f"Unexpected error during Parquet conversion: {exc}"], None

    if not isinstance(conversion, dict) or "error" in conversion:
        error_text = conversion.get("error", "Unknown conversion failure.") if isinstance(conversion, dict) else "Unknown conversion failure."
        warnings.append(error_text)
        return None, warnings, None

    parquet_path = conversion.get("parquet_path")
    if not parquet_path:
        warnings.append("Parquet conversion did not return a valid file path.")
        return None, warnings, None
    warnings.extend(conversion.get("warnings", []))

    df: Optional[pd.DataFrame] = None
    try:
        con = duckdb.connect(database=":memory:")
        try:
            df = con.execute("SELECT * FROM read_parquet(?)", [parquet_path]).df()
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001
        log_exception("app._load_uploaded_file.read_parquet", exc, context={"filename": filename})
        warnings.append(f"Failed to read back converted Parquet file: {exc}")
        return None, warnings, None

    if df is None or df.empty:
        warnings.append("The uploaded file was parsed but contains no data rows.")
        return None, warnings, parquet_path

    try:
        duplicate_cols = df.columns[df.columns.duplicated()].unique().tolist()
        if duplicate_cols:
            warnings.append(f"Duplicate header(s) detected and disambiguated: {', '.join(duplicate_cols)}.")
    except Exception:  # noqa: BLE001
        pass

    return df, warnings, parquet_path


_active_parquet_registry_lock = threading.Lock()
_active_parquet_registry: Dict[str, str] = {}  # per-session UUID -> fingerprint currently in use


def _get_session_uuid() -> str:
    """Returns a stable per-browser-session UUID, generated once and cached
    in st.session_state (session-scoped, survives reruns, unique per
    concurrent user). Deliberately avoids Streamlit's internal/private
    script-run-context APIs, which are not a stable public contract across
    versions."""
    if "_kesco_session_uuid" not in st.session_state:
        st.session_state["_kesco_session_uuid"] = str(uuid.uuid4())
    return st.session_state["_kesco_session_uuid"]


def _register_active_parquet(fingerprint: Optional[str]) -> None:
    """Publishes this session's currently-active parquet fingerprint into a
    process-wide registry before any cleanup pass runs. duckdb_executor.py's
    `_shared_connection` is a single DuckDB connection shared across every
    concurrent Streamlit session in this server process — without this
    registry, one session's upload could delete a Parquet file another live
    session still has open against that shared connection (Finding C1)."""
    session_uuid = _get_session_uuid()
    with _active_parquet_registry_lock:
        if fingerprint:
            _active_parquet_registry[session_uuid] = fingerprint
        else:
            _active_parquet_registry.pop(session_uuid, None)


def _cleanup_stale_parquets(keep_fingerprint: Optional[str] = None, max_age_hours: int = 6) -> None:
    """
    Removes temp Parquet files that are no longer referenced by ANY live
    session. A file is only evicted when it is (a) not this session's own
    fingerprint, (b) not claimed by any other session currently registered
    in _active_parquet_registry, AND (c) older than max_age_hours. The
    age check is the safety net for orphaned registrations left behind by
    a session that ended without a teardown hook (Streamlit has no
    reliable public "session closed" callback) — it is not the primary
    eviction mechanism for files still in active use. Never raises.
    """
    try:
        if keep_fingerprint:
            _register_active_parquet(keep_fingerprint)

        with _active_parquet_registry_lock:
            live_fingerprints = set(_active_parquet_registry.values())
        if keep_fingerprint:
            live_fingerprints.add(keep_fingerprint)

        cutoff = datetime.now().timestamp() - (max_age_hours * 3600)
        for path in _PARQUET_TEMP_DIR.glob("*.parquet"):
            try:
                if path.stem in live_fingerprints:
                    continue
                if path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
            except OSError:
                continue
    except Exception as exc:  # noqa: BLE001
        log_exception("app._cleanup_stale_parquets", exc, severity="info")


# @st.cache_data(show_spinner=False)
# def _load_uploaded_file(
#     file_bytes: bytes, filename: str, fingerprint: str
# ) -> Tuple[Optional[pd.DataFrame], List[str], Optional[str]]:
#     """
#     Converts the upload straight to Parquet on disk (Finding C2 — no
#     upfront full pandas materialization via pd.read_csv/read_excel), then
#     reads the Parquet back through DuckDB for the existing pandas-based
#     Inference -> Registry -> Cleaning stages, which still expect a
#     DataFrame. Returns (dataframe_or_None, warnings, parquet_path_or_None).
#     Never raises.
#     """
#     warnings: List[str] = []
#     conversion = convert_upload_to_parquet(file_bytes, filename, fingerprint)
#     if "error" in conversion:
#         warnings.append(conversion["error"])
#         return None, warnings, None

#     parquet_path = conversion["parquet_path"]
#     warnings.extend(conversion.get("warnings", []))

#     try:
#         con = duckdb.connect(database=":memory:")
#         try:
#             df = con.execute("SELECT * FROM read_parquet(?)", [parquet_path]).df()
#         finally:
#             con.close()
#     except Exception as exc:  # noqa: BLE001
#         log_exception("app._load_uploaded_file.read_parquet", exc, context={"filename": filename})
#         warnings.append(f"Failed to read back converted Parquet file: {exc}")
#         return None, warnings, None

#     if df is None or df.empty:
#         warnings.append("The uploaded file was parsed but contains no data rows.")
#         return None, warnings, parquet_path

#     duplicate_cols = df.columns[df.columns.duplicated()].unique().tolist()
#     if duplicate_cols:
#         warnings.append(f"Duplicate header(s) detected and disambiguated: {', '.join(duplicate_cols)}.")

#     return df, warnings, parquet_path


@st.cache_data(show_spinner=False)
def _run_type_inference_cached(_df: pd.DataFrame, fingerprint: str) -> List[ColumnProfile]:
    """Cached wrapper around core.type_inference.infer_dataframe."""
    return infer_dataframe(_df)


@st.cache_data(show_spinner=False)
def _run_domain_detection_cached(
    _registry: ColumnRegistry, filename: str, registry_version: int
) -> Tuple[str, float]:
    """Cached wrapper around engine.domain_detection.detect_domain."""
    return detect_domain(_registry, filename)


@st.cache_data(show_spinner=False)
def _run_safe_cleaning_cached(
    _df: pd.DataFrame,
    _registry: ColumnRegistry,
    _profiles: List[ColumnProfile],
    fingerprint: str,
    registry_version: int,
) -> CleaningResult:
    """Cached wrapper around core.cleaner.SafeCleaningEngine.clean."""
    engine = SafeCleaningEngine(_registry, _profiles)
    return engine.clean(_df)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS READINESS SCORE & DATA QUALITY SCORE
# ══════════════════════════════════════════════════════════════════════════════

def _compute_analytics_readiness_score(
    registry: ColumnRegistry,
    profiles: List[ColumnProfile],
    df: pd.DataFrame,
) -> Tuple[int, str, List[str]]:
    """
    Computes the 0-100 Analytics Readiness Score. Returns
    (score, band_label, recommendations). Never raises.
    """
    try:
        if df is None or df.empty or not profiles:
            return 0, "Critical", ["Upload a non-empty dataset to compute readiness."]

        recommendations: List[str] = []

        total_columns = max(len(profiles), 1)
        reviewed_columns = sum(1 for p in profiles if not p.needs_manual_review)
        schema_completeness = (reviewed_columns / total_columns) * 100.0

        confirmed_mappings = sum(1 for m in registry.mappings.values() if m.confirmed)
        role_mapping_completeness = (
            (confirmed_mappings / max(len(registry.mappings), 1)) * 100.0
            if registry.mappings else 0.0
        )

        avg_null_pct = float(np.mean([p.null_pct for p in profiles])) if profiles else 100.0
        data_quality = max(0.0, 100.0 - avg_null_pct)

        required_field_hits = sum(
            1 for r in (ROLE_RECORD_ID, ROLE_REGISTRATION_DATE, ROLE_STATUS) if registry.has_role(r)
        )
        required_field_availability = (required_field_hits / 3.0) * 100.0

        hierarchy_hits = sum(
            1 for r in (ROLE_ZONE, ROLE_CIRCLE, ROLE_DIVISION, ROLE_SUBDIVISION) if registry.has_role(r)
        )
        hierarchy_detection = 100.0 if hierarchy_hits > 0 else 0.0

        date_availability = 100.0 if registry.has_role(ROLE_REGISTRATION_DATE) else 0.0

        weighted_score = (
            schema_completeness * 0.20
            + role_mapping_completeness * 0.25
            + data_quality * 0.20
            + required_field_availability * 0.20
            + hierarchy_detection * 0.075
            + date_availability * 0.075
        )
        score = int(round(max(0.0, min(100.0, weighted_score))))

        band = "Critical"
        for lower, upper, label in READINESS_BANDS:
            if lower <= score <= upper:
                band = label
                break

        if role_mapping_completeness < 100.0:
            recommendations.append(
                "Confirm remaining role mappings in the Schema Mapping Studio to raise readiness."
            )
        if required_field_availability < 100.0:
            recommendations.append(
                "Map Record ID, Registration Date, and Status roles — these unlock the core KPI library."
            )
        if hierarchy_detection == 0.0:
            recommendations.append(
                "Map at least one geographic hierarchy role (Zone/Circle/Division/Subdivision) "
                "to enable drill-down and risk-hierarchy analytics."
            )
        if date_availability == 0.0:
            recommendations.append(
                "Map a Registration Date role to enable trend, pending-age, and executive reporting."
            )
        if data_quality < 70.0:
            recommendations.append(
                "Review columns with high null rates in the Metadata Explorer before running analytics."
            )
        if not recommendations:
            recommendations.append("Dataset is fully analytics-ready across all evaluated dimensions.")

        return score, band, recommendations
    except Exception as exc:  # noqa: BLE001
        return 0, "Critical", [f"Readiness scoring could not be completed: {exc}"]


def _compute_data_quality_score(df: pd.DataFrame, profiles: List[ColumnProfile]) -> int:
    """
    Computes a deterministic 0-100 Data Quality Score. Never raises;
    returns 0 on any failure or empty dataset.
    """
    try:
        if df is None or df.empty:
            return 0
        total_cells = max(df.shape[0] * df.shape[1], 1)
        missing_cells = int(df.isna().sum().sum())
        missing_ratio = missing_cells / total_cells

        duplicate_ratio = float(df.duplicated(keep="first").mean()) if len(df) else 0.0

        review_ratio = (
            sum(1 for p in profiles if p.needs_manual_review) / max(len(profiles), 1)
            if profiles else 0.0
        )

        penalty = (missing_ratio * 45.0) + (duplicate_ratio * 25.0) + (review_ratio * 30.0)
        score = max(0.0, min(100.0, 100.0 - penalty))
        return int(round(score))
    except Exception:  # noqa: BLE001
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# NOTIFICATION CENTER & AUDIT FORMATTING
# ══════════════════════════════════════════════════════════════════════════════

def _audit_entries_to_table(entries: List[Any]) -> List[Dict[str, Any]]:
    """
    Formats a List[AuditEntry] into the AuditLog.as_table() schema. Never
    raises; returns an empty list on any formatting failure.
    """
    table: List[Dict[str, Any]] = []
    try:
        for entry in entries:
            table.append({
                "Timestamp": entry.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "Type": entry.action_type,
                "Description": entry.description,
                "Rows Affected": entry.rows_affected if entry.rows_affected is not None else "—",
                "Details": "; ".join(f"{k}={v}" for k, v in entry.details.items()) if getattr(entry, "details", None) else "—",
            })
    except Exception:  # noqa: BLE001
        return []
    return table


def _build_notifications(
    profiles: List[ColumnProfile],
    cleaning_result: CleaningResult,
    registry: ColumnRegistry,
    readiness_band: str,
    domain_confidence: float,
) -> List[Dict[str, str]]:
    """Builds the Notification Center payload. Never raises."""
    notifications: List[Dict[str, str]] = []
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if readiness_band in ("Critical", "Poor"):
            notifications.append({
                "severity": "critical" if readiness_band == "Critical" else "warning",
                "category": "Analytics Readiness",
                "message": f"Analytics Readiness Score is currently rated '{readiness_band}'. "
                           "Review the Schema Mapping Studio to raise data readiness before analysis.",
                "timestamp": timestamp,
            })

        unmapped_high_value = [
            p.original_name for p in profiles
            if p.needs_manual_review and p.confidence < MIN_AUTO_CONFIDENCE
        ]
        if unmapped_high_value:
            notifications.append({
                "severity": "warning",
                "category": "Missing Schema Mappings",
                "message": f"{len(unmapped_high_value)} column(s) require manual review before they "
                           "can be safely mapped: " + ", ".join(unmapped_high_value[:5])
                           + (" …" if len(unmapped_high_value) > 5 else ""),
                "timestamp": timestamp,
            })

        if cleaning_result.flagged_business_key_duplicates:
            notifications.append({
                "severity": "warning",
                "category": "Data Quality Issues",
                "message": f"{len(cleaning_result.flagged_business_key_duplicates)} business-key "
                           "duplicate group(s) were flagged (not removed) — review for repeat-activity patterns.",
                "timestamp": timestamp,
            })

        if cleaning_result.flagged_outlier_summaries:
            notifications.append({
                "severity": "info",
                "category": "Data Quality Issues",
                "message": f"{len(cleaning_result.flagged_outlier_summaries)} numeric column(s) contain "
                           "statistical outliers — retained per non-destructive cleaning policy.",
                "timestamp": timestamp,
            })

        if domain_confidence < 0.4:
            notifications.append({
                "severity": "info",
                "category": "High Priority Insights",
                "message": "Business domain detection returned low confidence. Verify role mappings "
                           "reflect the intended dataset type.",
                "timestamp": timestamp,
            })

        if not notifications:
            notifications.append({
                "severity": "info",
                "category": "System",
                "message": "No critical issues detected. Dataset ingested and cleaned successfully.",
                "timestamp": timestamp,
            })

        return notifications
    except Exception:  # noqa: BLE001
        return []


def _build_notifications_from_state() -> List[Dict[str, str]]:
    """
    Rebuilds the Notification Center payload purely from already-persisted
    st.session_state artifacts. Never raises.
    """
    notifications: List[Dict[str, str]] = []
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        registry: Optional[ColumnRegistry] = st.session_state.get("column_registry")
        profiles: List[ColumnProfile] = st.session_state.get("column_profiles", [])
        audit_results: Dict[str, Any] = st.session_state.get("audit_results", {})
        readiness_band: str = st.session_state.get("readiness_band", "Critical")
        _domain_label, domain_confidence = st.session_state.get("domain_detection", (DOMAIN_UNKNOWN, 0.0))

        if readiness_band in ("Critical", "Poor"):
            notifications.append({
                "severity": "critical" if readiness_band == "Critical" else "warning",
                "category": "Analytics Readiness",
                "message": f"Analytics Readiness Score is currently rated '{readiness_band}'. "
                           "Review the Schema Mapping Studio to raise data readiness before analysis.",
                "timestamp": timestamp,
            })

        unmapped_high_value = [
            p.original_name for p in profiles
            if p.needs_manual_review and p.confidence < MIN_AUTO_CONFIDENCE
        ]
        if unmapped_high_value:
            notifications.append({
                "severity": "warning",
                "category": "Type Inference Review",
                "message": f"{len(unmapped_high_value)} column(s) require manual type-inference "
                           "review (low detection confidence): " + ", ".join(unmapped_high_value[:5])
                           + (" …" if len(unmapped_high_value) > 5 else ""),
                "timestamp": timestamp,
            })

        if registry is not None:
            confirmed_count = sum(1 for m in registry.mappings.values() if m.confirmed)
            total_tracked = len(registry.mappings)
            core_roles_missing = registry.missing_roles(
                [ROLE_RECORD_ID, ROLE_REGISTRATION_DATE, ROLE_STATUS]
            )
            if core_roles_missing:
                notifications.append({
                    "severity": "warning",
                    "category": "Missing Schema Mappings",
                    "message": f"{len(core_roles_missing)} core role(s) are not yet confirmed in the "
                               "Schema Mapping Studio: " + ", ".join(
                                   registry.display_name(r) for r in core_roles_missing
                               ) + ". Core KPIs and executive reporting will remain disabled until "
                               "these are mapped.",
                    "timestamp": timestamp,
                })
            elif total_tracked and confirmed_count < total_tracked:
                notifications.append({
                    "severity": "info",
                    "category": "Missing Schema Mappings",
                    "message": f"{total_tracked - confirmed_count} of {total_tracked} tracked role(s) "
                               "remain unconfirmed. Some advanced KPIs and charts may be disabled "
                               "until they are mapped or explicitly cleared.",
                    "timestamp": timestamp,
                })

        if int(audit_results.get("flagged_business_key_duplicates", 0) or 0) > 0:
            notifications.append({
                "severity": "warning",
                "category": "Data Quality Issues",
                "message": f"{audit_results.get('flagged_business_key_duplicates', 0)} business-key "
                           "duplicate group(s) were flagged (not removed) — review for repeat-activity "
                           "patterns.",
                "timestamp": timestamp,
            })

        if int(audit_results.get("flagged_outlier_summaries", 0) or 0) > 0:
            notifications.append({
                "severity": "info",
                "category": "Data Quality Issues",
                "message": f"{audit_results.get('flagged_outlier_summaries', 0)} numeric column(s) "
                           "contain statistical outliers — retained per non-destructive cleaning policy.",
                "timestamp": timestamp,
            })

        if domain_confidence < 0.4:
            notifications.append({
                "severity": "info",
                "category": "High Priority Insights",
                "message": "Business domain detection returned low confidence. Verify role mappings "
                           "reflect the intended dataset type.",
                "timestamp": timestamp,
            })

        if not notifications:
            notifications.append({
                "severity": "info",
                "category": "System",
                "message": "No critical issues detected for the current role-mapping state.",
                "timestamp": timestamp,
            })

        return notifications
    except Exception:  # noqa: BLE001
        return []


def sync_notifications_with_registry() -> None:
    """
    Public invalidation checkpoint for the Notification Center. Never raises.
    """
    try:
        registry: Optional[ColumnRegistry] = st.session_state.get("column_registry")
        if registry is None:
            return
        last_synced_version = st.session_state.get("_notifications_synced_registry_version")
        if last_synced_version == registry.version:
            return
        st.session_state["notifications"] = _build_notifications_from_state()
        st.session_state["_notifications_synced_registry_version"] = registry.version
    except Exception:  # noqa: BLE001
        return


# ══════════════════════════════════════════════════════════════════════════════
# INGESTION ORCHESTRATION — PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def process_uploaded_file(
    file_bytes: bytes,
    filename: str,
    workspace_name: str,
) -> Dict[str, Any]:
    """
    Executes the full Ingestion -> Type Inference -> Registry Bootstrap ->
    Domain Detection -> Safe Cleaning pipeline. Never raises.
    """
    result: Dict[str, Any] = {
        "status": "failed",
        "filename": filename,
        "errors": [],
        "warnings": [],
    }
    try:
        fingerprint = _compute_file_fingerprint(file_bytes, filename)
        _cleanup_stale_parquets(keep_fingerprint=fingerprint)
        raw_df, ingestion_warnings, parquet_path = _load_uploaded_file(file_bytes, filename, fingerprint)
        result["warnings"].extend(ingestion_warnings)

        if raw_df is None:
            result["errors"].append(
                "The uploaded file could not be parsed into tabular data. "
                "Verify the file is a valid CSV/XLSX/XLS export and is not corrupted or password-protected."
            )
            return result

        profiles = _run_type_inference_cached(raw_df, fingerprint)
        registry = ColumnRegistry(workspace_name=workspace_name or "Default Workspace")
        registry.bootstrap_from_profiles(profiles)
        domain_label, domain_confidence = _run_domain_detection_cached(registry, filename, registry.version)
        cleaning_result = _run_safe_cleaning_cached(raw_df, registry, profiles, fingerprint, registry.version)

        readiness_score, readiness_band, readiness_recommendations = _compute_analytics_readiness_score(
            registry=registry, profiles=profiles, df=cleaning_result.cleaned_df,
        )
        data_quality_score = _compute_data_quality_score(cleaning_result.cleaned_df, profiles)

        notifications = _build_notifications(
            profiles=profiles,
            cleaning_result=cleaning_result,
            registry=registry,
            readiness_band=readiness_band,
            domain_confidence=domain_confidence,
        )

        st.session_state["uploaded_dataframe"] = raw_df
        st.session_state["cleaned_dataframe"] = cleaning_result.cleaned_df
        st.session_state["analytics_ready_dataframe"] = cleaning_result.analytics_df
        st.session_state["filtered_dataframe"] = cleaning_result.analytics_df
        st.session_state["column_registry"] = registry
        st.session_state["column_profiles"] = profiles
        st.session_state["domain_detection"] = (domain_label, domain_confidence)
        st.session_state["audit_results"] = {
            "cleaning_summary": cleaning_result.cleaning_summary,
            "audit_entries": _audit_entries_to_table(cleaning_result.audit_entries),
            "flagged_business_key_duplicates": len(cleaning_result.flagged_business_key_duplicates),
            "flagged_outlier_summaries": len(cleaning_result.flagged_outlier_summaries),
            "rows_original": cleaning_result.rows_original,
            "rows_cleaned": cleaning_result.rows_cleaned,
            "rows_removed": cleaning_result.rows_removed,
            "data_quality_score": data_quality_score,
        }
        st.session_state["ingestion_warnings"] = result["warnings"]
        st.session_state["file_fingerprint"] = fingerprint
        st.session_state["parquet_path"] = parquet_path
        st.session_state["active_filename"] = filename
        st.session_state["readiness_score"] = readiness_score
        st.session_state["readiness_band"] = readiness_band
        st.session_state["readiness_recommendations"] = readiness_recommendations
        st.session_state["notifications"] = notifications
        st.session_state["_notifications_synced_registry_version"] = registry.version
        st.session_state["workspace_name"] = workspace_name or "Default Workspace"
        st.session_state["last_ingestion_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["active_filters"] = {}
        st.session_state["dashboard_state"] = {}
        st.session_state["analytics_results"] = {}
        st.session_state["visualization_cache"] = {}
        st.session_state["drill_breadcrumbs"] = []
        st.session_state["active_tab"] = "landing"
        # Prevent stale, unapplied Schema Mapping Studio selections — staged
        # against the PREVIOUS dataset's column headers — from silently
        # persisting into the newly ingested dataset's rerun.
        st.session_state["_schema_pending_mappings"] = {}
        st.session_state["_schema_mapping_pending_warnings"] = []

        result["status"] = "success"
        result["rows"] = int(len(raw_df))
        result["columns"] = int(len(raw_df.columns))
        result["domain"] = domain_label
        result["readiness_score"] = readiness_score
        return result

    except Exception as exc:  # noqa: BLE001 — absolute final safety net
        incident_id = log_exception(
            "app.process_uploaded_file",
            exc,
            context={"filename": filename, "workspace_name": workspace_name},
        )
        result["errors"].append(
            f"An unexpected error occurred during ingestion (Reference ID: {incident_id})."
        )
        return result


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK MASTER CONTROL SIDEBAR (used only until components/sidebar.py exists)
# ══════════════════════════════════════════════════════════════════════════════

def _sync_workspace_name() -> None:
    st.session_state["workspace_name"] = st.session_state.get("_workspace_name_input", "Default Workspace")


def _render_fallback_sidebar() -> None:
    """
    Minimal, fully functional Master Control sidebar rendered only when
    components/sidebar.py has not yet been generated or fails to
    import/execute. Never raises.
    """
    with st.sidebar:
        st.markdown(f"### {APP_ICON} {APP_TITLE}")
        st.caption("Master Control · Unified Ingestion")

        st.divider()
        uploaded_file = st.file_uploader(
            "Upload Dataset",
            type=SUPPORTED_UPLOAD_TYPES,
            accept_multiple_files=False,
            key="_primary_file_uploader",
            help=f"Supported formats: {', '.join(SUPPORTED_UPLOAD_TYPES)}",
        )
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            fingerprint = _compute_file_fingerprint(file_bytes, uploaded_file.name)
            if st.session_state.get("file_fingerprint") != fingerprint:
                with st.spinner(
                    ENTERPRISE_COPY_MAP.get(
                        "Detecting Schema Topology and Resolving Domain Dictionaries...",
                        "Detecting Schema Topology and Resolving Domain Dictionaries...",
                    )
                ):
                    outcome = process_uploaded_file(
                        file_bytes=file_bytes,
                        filename=uploaded_file.name,
                        workspace_name=st.session_state.get("workspace_name", "Default Workspace"),
                    )
                if outcome["status"] == "success":
                    st.success(
                        f"Ingested '{outcome['filename']}' — {outcome['rows']:,} rows, "
                        f"{outcome['columns']} columns. Domain: {outcome['domain']}."
                    )
                    st.rerun()
                else:
                    for err in outcome["errors"]:
                        st.error(err)
                    for warn in outcome["warnings"]:
                        st.warning(warn)

        st.divider()
        st.text_input(
            "Workspace Name",
            value=st.session_state.get("workspace_name", "Default Workspace"),
            key="_workspace_name_input",
            on_change=_sync_workspace_name,
        )

        nav_pages: List[Any] = st.session_state.get("_nav_pages_registry", [])
        if nav_pages:
            st.divider()
            st.caption("Pages")
            for page in nav_pages:
                try:
                    st.page_link(page)
                except Exception:  # noqa: BLE001
                    continue

        st.divider()
        theme_options = list(THEMES.keys())
        current_theme = st.session_state.get("theme", DEFAULT_THEME_KEY)
        selected_theme = st.selectbox(
            "Theme",
            options=theme_options,
            index=theme_options.index(current_theme) if current_theme in theme_options else 0,
            format_func=lambda k: THEMES[k]["name"],
            key="_theme_selector",
        )
        if selected_theme != st.session_state.get("theme"):
            st.session_state["theme"] = selected_theme
            st.rerun()

        if st.session_state.get("uploaded_dataframe") is not None:
            st.divider()
            st.caption("Active Dataset")
            st.write(f"**File:** {st.session_state.get('active_filename', '—')}")
            domain_label, domain_conf = st.session_state.get("domain_detection", (DOMAIN_UNKNOWN, 0.0))
            st.write(f"**Domain:** {domain_label} ({domain_conf:.0%} confidence)")
            if st.button("Clear Active Dataset", use_container_width=True):
                _clear_active_dataset()
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SMART LANDING PAGE (fallback default route until pages/1_dashboard.py exists)
# ══════════════════════════════════════════════════════════════════════════════

def _render_landing_page() -> None:
    """
    Renders the Smart Landing Page. No pandas aggregation or business
    calculation occurs in this function.
    """
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.caption("KESCO Grid Operational Intelligence Workspace")
    st.divider()

    df = st.session_state.get("analytics_ready_dataframe")
    registry: Optional[ColumnRegistry] = st.session_state.get("column_registry")

    if df is None or registry is None:
        st.info(
            "No dataset is currently active. Use the sidebar to upload a CSV or Excel file "
            f"(supported: {', '.join(SUPPORTED_UPLOAD_TYPES)}) to begin analysis."
        )
        return

    domain_label, domain_confidence = st.session_state.get("domain_detection", (DOMAIN_UNKNOWN, 0.0))
    audit_results = st.session_state.get("audit_results", {})
    readiness_score = st.session_state.get("readiness_score", 0)
    readiness_band = st.session_state.get("readiness_band", "Critical")
    notifications = st.session_state.get("notifications", [])

    st.markdown('<div class="kesco-section-title">Dataset Summary</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Total Columns", f"{df.shape[1]:,}")
    col3.metric("Business Domain", domain_label)
    col4.metric("Domain Confidence", f"{domain_confidence:.0%}")

    st.markdown('<div class="kesco-section-title">Readiness & Data Quality</div>', unsafe_allow_html=True)
    col5, col6, col7 = st.columns(3)
    with col5:
        st.metric("Analytics Readiness Score", f"{readiness_score}/100", help=readiness_band)
        st.progress(min(max(readiness_score, 0), 100) / 100.0)
    with col6:
        st.metric("Data Quality Score", f"{audit_results.get('data_quality_score', 0)}/100")
    with col7:
        st.metric("Rows Removed (Safe Cleaning)", f"{audit_results.get('rows_removed', 0):,}")

    st.markdown('<div class="kesco-section-title">Recommended Next Steps</div>', unsafe_allow_html=True)
    for rec in st.session_state.get("readiness_recommendations", []):
        st.markdown(f"- {rec}")

    st.markdown('<div class="kesco-section-title">Notification Center</div>', unsafe_allow_html=True)
    severity_badge_map: Dict[str, str] = {
        "critical": "kesco-badge-critical",
        "warning": "kesco-badge-warning",
        "info": "kesco-badge-info",
    }
    _landing_theme: Theme = THEMES.get(st.session_state.get("theme", DEFAULT_THEME_KEY), THEMES[DEFAULT_THEME_KEY])
    for note in notifications:
        badge_class = severity_badge_map.get(note.get("severity", "info"), "kesco-badge-info")
        st.markdown(
            f'<div class="kesco-card">'
            f'<span class="kesco-badge {badge_class}">{note.get("severity", "info").upper()}</span>'
            f'&nbsp;&nbsp;<strong>{note.get("category", "")}</strong>'
            f'<br/>{note.get("message", "")}'
            f'<br/><span style="font-size:0.7rem;color:{_landing_theme["secondary"]};">{note.get("timestamp", "")}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with st.expander("Recent Audit Activity", expanded=False):
        audit_table = audit_results.get("audit_entries", [])
        if audit_table:
            st.dataframe(pd.DataFrame(audit_table), use_container_width=True, hide_index=True)
        else:
            st.caption("No audit activity recorded yet.")

    st.markdown(
        f'<div class="kesco-footer">{APP_TITLE} · Workspace: '
        f'{st.session_state.get("workspace_name", "Default Workspace")} · '
        f'Last Ingested: {st.session_state.get("last_ingestion_timestamp") or "—"}</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ROUTING MESH
# ══════════════════════════════════════════════════════════════════════════════

def _build_nav_pages() -> List[Any]:
    """
    Constructs and returns the ordered list of st.Page objects, WITHOUT
    invoking st.navigation() itself, so it can be published to
    st.session_state["_nav_pages_registry"] and consumed by
    FRONTEND.components.sidebar.render() via st.page_link() BEFORE the
    actual st.navigation(..., position="hidden") call happens in main().
    Never raises in a way that leaves nav_pages empty.
    """
    pages_dir = Path(__file__).resolve().parent / "pages"
    dashboard_path = pages_dir / "1_dashboard.py"
    schema_mapping_path = pages_dir / "3_schema_mapping.py"
    audit_path = pages_dir / "2_audit.py"
    self_service_path = pages_dir / "4_self_service.py"

    nav_pages: List[Any] = []
    if dashboard_path.exists():
        nav_pages.append(st.Page(str(dashboard_path), title="Grid Operational Dashboard", icon="📊", default=True))
    if schema_mapping_path.exists():
        nav_pages.append(st.Page(str(schema_mapping_path), title="Schema Mapping Studio", icon="🧭"))
    if self_service_path.exists():
        nav_pages.append(st.Page(str(self_service_path), title="Self-Service Analytics Builder", icon="🛠️"))
    if audit_path.exists():
        nav_pages.append(st.Page(str(audit_path), title="System Diagnostics & Audit", icon="🧾"))
    if not nav_pages:
        nav_pages.append(st.Page(_render_landing_page, title="Home", icon="⚡", default=True))
    return nav_pages


# ══════════════════════════════════════════════════════════════════════════════
# APPLICATION ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
def bootstrap_ui_engine() -> None:
    """
    Atomic UI Bootstrap — Streamlit 1.59.0 race-condition fix.

    INTENTIONALLY re-executes on EVERY rerun (no session_state guard).
    Removing the guard is required: Streamlit's frontend reconciler streams
    st.html()/st.markdown() deltas independently, not in strict source
    order, so a single-shot injector can lose the "style before paint"
    race on session start and never gets a second chance. Re-injecting
    every rerun means even a lost race on rerun N is won on rerun N+1,
    and the MutationObserver below closes the gap within a single rerun.

    Idempotent: uses fixed element IDs + getElementById-replace, so
    re-running never accumulates duplicate <style>/<script> nodes.
    """
    st.html(r"""
<style id="keds-bootstrap-style">
html[data-theme="kesco_corporate"] {
  --keds-primary:#1D4ED8; --keds-secondary:#5B6472; --keds-background:#F1F5F9;
  --keds-surface:#FFFFFF; --keds-text:#0F172A; --keds-success:#15803D;
  --keds-warning:#B45309; --keds-danger:#B91C1C;
  --text-color:#0F172A; --background-color:#F1F5F9;
  --secondary-background-color:#FFFFFF; --primary-color:#1D4ED8;
}
html[data-theme="executive_dark"] {
  --keds-primary:#60A5FA; --keds-secondary:#94A3B8; --keds-background:#0B1329;
  --keds-surface:#1C2541; --keds-text:#E4E4E7; --keds-success:#4ADE80;
  --keds-warning:#FBBF24; --keds-danger:#F87171;
  --text-color:#E4E4E7; --background-color:#0B1329;
  --secondary-background-color:#1C2541; --primary-color:#60A5FA;
}
html[data-theme="professional_light"] {
  --keds-primary:#334155; --keds-secondary:#64748B; --keds-background:#F8FAFC;
  --keds-surface:#FFFFFF; --keds-text:#1E293B; --keds-success:#166534;
  --keds-warning:#92400E; --keds-danger:#991B1B;
  --text-color:#1E293B; --background-color:#F8FAFC;
  --secondary-background-color:#FFFFFF; --primary-color:#334155;
}
html[data-theme="government_blue"] {
  --keds-primary:#1E3A5F; --keds-secondary:#51677D; --keds-background:#EEF2F6;
  --keds-surface:#FFFFFF; --keds-text:#16202A; --keds-success:#1F6B44;
  --keds-warning:#8A5A00; --keds-danger:#8E2A2E;
  --text-color:#16202A; --background-color:#EEF2F6;
  --secondary-background-color:#FFFFFF; --primary-color:#1E3A5F;
}
html[data-theme="high_contrast"] {
  --keds-primary:#000000; --keds-secondary:#2B2B2B; --keds-background:#FFFFFF;
  --keds-surface:#FFFFFF; --keds-text:#000000; --keds-success:#0B5A1E;
  --keds-warning:#7A4B00; --keds-danger:#7A0C0C;
  --text-color:#000000; --background-color:#FFFFFF;
  --secondary-background-color:#FFFFFF; --primary-color:#000000;
}
/* No-theme-yet default: Executive Dark, so first paint never shows
   Streamlit's own navy/white default before data-theme lands. */
html:not([data-theme]),
html:not([data-theme]) body,
html:not([data-theme]) .stApp,
html:not([data-theme]) [data-testid="stAppViewContainer"] {
  --keds-primary:#60A5FA; --keds-secondary:#94A3B8; --keds-background:#0B1329;
  --keds-surface:#1C2541; --keds-text:#E4E4E7;
  background-color:#0B1329 !important; color:#E4E4E7 !important;
}

.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], body {
  background-color: var(--keds-background) !important;
  color: var(--keds-text) !important;
}
[data-testid="stMetricValue"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span, label {
  color: var(--keds-text) !important;
}

/* ══════ 3D KPI FLIP CARD — GLOBAL ROOT ENFORCEMENT ══════
   Targets .stApp / stAppViewContainer explicitly per requirement #3,
   not just the card class alone, so specificity always beats any
   Streamlit-reconciled wrapper computed style. */
.stApp .keds-kpi-card-container,
[data-testid="stAppViewContainer"] .keds-kpi-card-container {
  width: 100% !important; height: 152px !important;
  perspective: 1200px !important; -webkit-perspective: 1200px !important;
  margin-bottom: 12px !important; overflow: visible !important;
  isolation: isolate !important; display: block !important;
}
.keds-kpi-card-inner {
  position: relative; width: 100%; height: 100%;
  transition: transform 0.65s cubic-bezier(0.4,0,0.2,1);
  transform-style: preserve-3d !important; -webkit-transform-style: preserve-3d !important;
  transform: translateZ(0) rotateY(0deg);
}
.keds-kpi-card-container:hover .keds-kpi-card-inner,
.keds-kpi-card-container:focus-within .keds-kpi-card-inner {
  transform: translateZ(0) rotateY(180deg);
}
.keds-kpi-flip-front, .keds-kpi-flip-back {
  position: absolute !important; inset: 0 !important;
  backface-visibility: hidden !important; -webkit-backface-visibility: hidden !important;
  box-sizing: border-box; border-radius: 10px;
  border: 1px solid rgba(255,255,255,0.08);
  background-color: var(--keds-surface); padding: 16px 18px; overflow: hidden;
  margin: 0 !important; z-index: 1;
}
.keds-kpi-flip-front {
  transform: rotateY(0deg) translateZ(1px);
  border-left: 4px solid var(--keds-primary);
  display: flex; flex-direction: column; justify-content: space-between;
}
.keds-kpi-flip-front.keds-accent-success { border-left-color: var(--keds-success); }
.keds-kpi-flip-front.keds-accent-warning { border-left-color: var(--keds-warning); }
.keds-kpi-flip-front.keds-accent-danger  { border-left-color: var(--keds-danger); }
.keds-kpi-flip-back {
  transform: rotateY(180deg) translateZ(1px);
  border-left: 4px solid var(--keds-secondary);
  display: flex; flex-direction: column; justify-content: center;
}
.keds-kpi-label { font-size:0.68rem; font-weight:700; text-transform:uppercase;
  letter-spacing:0.04em; color: var(--keds-secondary); margin-bottom:2px; }
.keds-kpi-value { font-size:1.55rem; font-weight:800; color: var(--keds-text); line-height:1.1; }
.keds-kpi-flip-back-formula {
  font-family: 'JetBrains Mono', monospace; font-size:0.74rem; line-height:1.5;
  color: var(--keds-text); background: rgba(0,0,0,0.15); border-radius:6px;
  padding:10px 12px; white-space:pre-wrap;
}

/* Global-root override: force every Streamlit reconciled wrapper
   ancestor of a flip card to never clip it — matched at .stApp scope
   per requirement #3, not just locally. */
.stApp div[data-testid="stMarkdownContainer"]:has(.keds-kpi-card-container),
.stApp div[data-testid="element-container"]:has(.keds-kpi-card-container),
.stApp div[data-testid="stVerticalBlock"]:has(.keds-kpi-card-container),
.stApp div[data-testid="stVerticalBlockBorderWrapper"]:has(.keds-kpi-card-container),
.stApp div[data-testid="column"]:has(.keds-kpi-card-container),
.stApp div[data-testid="stHorizontalBlock"]:has(.keds-kpi-card-container) {
  overflow: visible !important; transform: none !important;
  contain: none !important; perspective: none !important;
}

/* ══════ PLOTLY TITLE/PLOT-AREA COLLISION FIX ══════
   Targets the SVG title node directly plus its component parent, with
   !important buffer, so a re-measuring ResizeObserver can never
   collapse the gap chart_factory._apply_layout() already reserves. */
div[data-testid="stPlotlyChart"] {
  padding-top: 34px !important;
  overflow: visible !important;
}
div[data-testid="stPlotlyChart"] > div {
  padding-top: 6px !important;
}
div[data-testid="stPlotlyChart"] .main-svg .gtitle {
  dominant-baseline: hanging !important;
  transform: translateY(4px) !important;
}

/* No-flash guard: any element carrying our marker class before JS has
   finished attaching real styles is hidden rather than shown unstyled. */
.keds-pending-style { visibility: hidden !important; }
</style>

<script id="keds-bootstrap-script">
(function () {
  var GUARD_SELECTORS = [
    '[data-testid="stMarkdownContainer"]', '[data-testid="element-container"]',
    '[data-testid="stVerticalBlock"]', '[data-testid="stVerticalBlockBorderWrapper"]',
    '[data-testid="column"]', '[data-testid="stHorizontalBlock"]'
  ];

  function enforceFlipCards(root) {
    try {
      var cards = root.querySelectorAll('.keds-kpi-card-container');
      for (var i = 0; i < cards.length; i++) {
        var c = cards[i];
        c.style.setProperty('overflow', 'visible', 'important');
        c.style.setProperty('perspective', '1200px', 'important');
        c.style.setProperty('display', 'block', 'important');
        var node = c.parentElement, hops = 0;
        while (node && hops < 8) {
          for (var s = 0; s < GUARD_SELECTORS.length; s++) {
            if (node.matches && node.matches(GUARD_SELECTORS[s])) {
              node.style.setProperty('overflow', 'visible', 'important');
            }
          }
          node = node.parentElement; hops++;
        }
      }
    } catch (e) { /* never break the page */ }
  }

  function enforcePlotlyBuffer(root) {
    try {
      var charts = root.querySelectorAll('div[data-testid="stPlotlyChart"]');
      for (var i = 0; i < charts.length; i++) {
        charts[i].style.setProperty('padding-top', '34px', 'important');
        charts[i].style.setProperty('overflow', 'visible', 'important');
      }
    } catch (e) { /* never break the page */ }
  }

  function applyTheme() {
    try {
      // Priority 1: sessionStorage (instant, set by the sidebar toggle,
      // survives a rerun's first paint with zero flash).
      var stored = window.sessionStorage.getItem('keds_theme');
      var marker = document.getElementById('keds-theme-engine-marker');
      var fromMarker = marker ? marker.getAttribute('data-active-theme') : null;
      var theme = stored || fromMarker;
      if (theme) {
        document.documentElement.setAttribute('data-theme', theme);
        if (document.body) document.body.setAttribute('data-theme', theme);
        if (!stored) { window.sessionStorage.setItem('keds_theme', theme); }
      }
    } catch (e) { /* never break the page */ }
  }

  function unhidePending(root) {
    try {
      var pending = root.querySelectorAll('.keds-pending-style');
      for (var i = 0; i < pending.length; i++) {
        pending[i].classList.remove('keds-pending-style');
      }
    } catch (e) { /* never break the page */ }
  }

  function runAll() {
    applyTheme();
    enforceFlipCards(document);
    enforcePlotlyBuffer(document);
    unhidePending(document);
  }

  // Run immediately (this script tag executes synchronously as inserted).
  runAll();

  // No-flash MutationObserver: re-run against every newly-inserted node,
  // not just at load — this is what actually closes the progressive-
  // streaming race described in the diagnostics report.
  if (window.__kescoObserver) {
    try { window.__kescoObserver.disconnect(); } catch (e) {}
  }
  var observer = new MutationObserver(function (mutations) {
    runAll();
  });
  observer.observe(document.documentElement, {
    childList: true, subtree: true, attributes: true,
    attributeFilter: ['data-active-theme', 'class', 'style']
  });
  window.__kescoObserver = observer;

  // Safety-net poll — covers React's occasional atomic subtree swap
  // that MutationObserver's callback batch can momentarily miss.
  var attempts = 30;
  (function poll() {
    runAll();
    attempts -= 1;
    if (attempts > 0) { setTimeout(poll, 100); }
  })();
})();
</script>
""")


def main() -> None:
    """
    Application entry point.

    Milestone 11 remediation:
      1. inject_static_theme_link() — unconditional every rerun, loads
         static/style.css via a stable <link> tag.
      2. bootstrap_static_js() — ONE-TIME per session (guarded by
         st.session_state["_static_js_bootstrapped"]).
      3. sync_theme_state_marker() — unconditional every rerun, writes the
         current theme value into the hidden marker div (backup signal).
      4. apply_theme_now() — unconditional every rerun, the PRIMARY,
         deterministic mechanism that actually applies `data-theme` to the
         real document via a real script execution context.
      5. Sidebar renders unconditionally every rerun — Presentation Mode
         (and its visibility guard) has been removed entirely.
      6. st.navigation(..., position="hidden") — the sidebar's manual
         "Pages" section (FRONTEND.components.sidebar) is the ONLY page
         list rendered; Streamlit's own native nav block is suppressed to
         eliminate the duplicate page listing.
    """
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    initialize_session_state()
    bootstrap_ui_engine() 
    from utils.layout_guard import inject_chart_block_css
    inject_chart_block_css()
   # ── Static asset bridge ──────────────────────────────────────────
    # inject_static_theme_link() injects the FULL stylesheet as a DOM
    # node-removal/recreation <script>. Its content is theme-agnostic
    # (every palette lives in the same stylesheet, switched purely via
    # the [data-theme] attribute selector — see core/themes.py), so
    # re-running this DOM mutation on EVERY rerun (i.e. every widget
    # interaction anywhere on the page, including schema mapping role
    # selects and chart-type dropdowns) is pure wasted work. Gate it
    # behind the same one-time-bootstrap pattern already used for JS.
    if not st.session_state.get("_static_css_bootstrapped"):
        inject_static_theme_link()
        st.session_state["_static_css_bootstrapped"] = True

    if not st.session_state.get("_static_js_bootstrapped"):
        bootstrap_static_js()
        st.session_state["_static_js_bootstrapped"] = True 

    _active_theme_key = st.session_state.get("theme", DEFAULT_THEME_KEY)
    sync_theme_state_marker(_active_theme_key)
    apply_theme_now(_active_theme_key)

    try:
        _assets_dir = Path(__file__).resolve().parent / "assets"
        _logo_path = _assets_dir / APP_LOGO_FILENAME
        _icon_path = _assets_dir / APP_ICON_LOGO_FILENAME
        if _logo_path.exists():
            st.logo(
                image=str(_logo_path),
                icon_image=str(_icon_path) if _icon_path.exists() else str(_logo_path),
            )
    except Exception as exc:  # noqa: BLE001
        log_exception("app.main.st_logo", exc, severity="info")

    # Page list must be published to session_state BEFORE render_sidebar()
    # executes, so the sidebar's "Pages" section can render st.page_link()s
    # against it at the correct, hardcoded position.
    _nav_pages = _build_nav_pages()
    st.session_state["_nav_pages_registry"] = _nav_pages

    # Sidebar/Header rendering — unconditional (Presentation Mode removed).
    if render_sidebar is not None:
        try:
            render_sidebar()
        except Exception as exc:  # noqa: BLE001
            incident_id = log_exception("app.main.render_sidebar", exc)
            st.sidebar.error(f"Sidebar component encountered an issue (Reference ID: {incident_id}).")
            _render_fallback_sidebar()
    else:
        _render_fallback_sidebar()

    # position="hidden" is REQUIRED — Streamlit's own native page-nav block
    # would otherwise render a second, duplicate page list at the top of
    # the sidebar alongside the manual "Pages" section rendered above.
    navigation = st.navigation(_nav_pages, position="hidden")
    navigation.run()





if __name__ == "__main__":
    main()
