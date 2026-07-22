"""
components/sidebar.py — Master Control, Unified Ingestion, & Global Report
Dispatch (FILE 2 / 6)

Implements the Universal Component Interface (render / validate / refresh /
export / metadata) for the platform's single, persistent sidebar surface.

Milestone 11 / Presentation-Mode-Removal remediation: the Presentation Mode
toggle (and its session-state key import) has been removed entirely from
the "Settings" section. The platform now relies exclusively on Streamlit's
native sidebar collapse control and browser zoom for presentation-style
viewing.

Section order remains HARDCODED and immutable across every page/rerun:
    [Logo/Title] -> [Upload Section] -> [Workspace] -> [Pages] ->
    [Reports] -> [Settings]
"Pages" is rendered via st.page_link() against the st.Page registry
app.py publishes into st.session_state["_nav_pages_registry"] BEFORE this
render() call. app.py calls st.navigation(..., position="hidden") so this
is the ONLY place pages are ever listed — no duplicate nav block.

This module performs NO pandas aggregation, NO KPI computation, and NO
chart rendering.
"""
from __future__ import annotations

import io
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from core.settings import (
    APP_TITLE,
    APP_ICON,
    SUPPORTED_UPLOAD_TYPES,
    ENTERPRISE_COPY_MAP,
)
from core.themes import THEMES, DEFAULT_THEME_KEY
from core.column_registry import ColumnRegistry
from core.schema_models import ColumnRegistrySnapshot
from engine.domain_detection import DOMAIN_UNKNOWN
from visualization.executive_summary import generate_executive_narrative, generate_executive_html_report
from utils.error_logging import log_exception

_COMPONENT_NAME: str = "sidebar"
_MAX_STAGED_FILES: int = 10


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS — WORKSPACE PROFILE PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def _current_workspace_snapshot_payload() -> Dict[str, Any]:
    """Builds a serializable snapshot of the active workspace's state. Never raises."""
    try:
        registry: Optional[ColumnRegistry] = st.session_state.get("column_registry")
        registry_snapshot: Optional[ColumnRegistrySnapshot] = registry.snapshot() if registry else None
        return {
            "registry_snapshot": registry_snapshot,
            "theme": st.session_state.get("theme", DEFAULT_THEME_KEY),
            "active_filters": dict(st.session_state.get("active_filters", {})),
            "dashboard_state": dict(st.session_state.get("dashboard_state", {})),
            "pinned_kpis": list(st.session_state.get("pinned_kpis", [])),
            "pinned_charts": list(st.session_state.get("pinned_charts", [])),
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception:  # noqa: BLE001
        return {
            "registry_snapshot": None,
            "theme": DEFAULT_THEME_KEY,
            "active_filters": {},
            "dashboard_state": {},
            "pinned_kpis": [],
            "pinned_charts": [],
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


def _save_workspace_profile(workspace_name: str) -> bool:
    """Persists the current workspace's mapping profile. Returns True on success."""
    if not workspace_name or not workspace_name.strip():
        return False
    try:
        profiles: Dict[str, Any] = st.session_state.setdefault("workspace_profiles", {})
        profiles[workspace_name.strip()] = _current_workspace_snapshot_payload()
        st.session_state["workspace_profiles"] = profiles
        return True
    except Exception:  # noqa: BLE001
        return False


def _load_workspace_profile(workspace_name: str) -> bool:
    """Restores a previously saved workspace mapping profile. Returns True on success."""
    try:
        profiles: Dict[str, Any] = st.session_state.get("workspace_profiles", {})
        payload = profiles.get(workspace_name)
        if payload is None:
            return False

        st.session_state["theme"] = payload.get("theme", DEFAULT_THEME_KEY)
        st.session_state["active_filters"] = dict(payload.get("active_filters", {}))
        st.session_state["dashboard_state"] = dict(payload.get("dashboard_state", {}))
        st.session_state["pinned_kpis"] = list(payload.get("pinned_kpis", []))
        st.session_state["pinned_charts"] = list(payload.get("pinned_charts", []))

        registry_snapshot: Optional[ColumnRegistrySnapshot] = payload.get("registry_snapshot")
        active_registry: Optional[ColumnRegistry] = st.session_state.get("column_registry")
        if registry_snapshot is not None and active_registry is not None:
            active_registry.restore(registry_snapshot)
            st.session_state["column_registry"] = active_registry
            st.session_state["visualization_cache"] = {}
            st.session_state["analytics_results"] = {}

        st.session_state["workspace_name"] = workspace_name
        return True
    except Exception:  # noqa: BLE001
        return False


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS — STAGED MULTI-FILE INGESTION
# ══════════════════════════════════════════════════════════════════════════════

def _stage_uploaded_files(uploaded_files: List[Any]) -> None:
    """Stages uploaded files into session_state, enforcing a max staged count. Never raises."""
    try:
        store: Dict[str, bytes] = st.session_state.setdefault("uploaded_files_store", {})
        for uploaded_file in uploaded_files:
            try:
                file_bytes = uploaded_file.getvalue()
            except Exception:  # noqa: BLE001
                # Covers ClientDisconnect / interrupted browser upload mid-transfer.
                continue
            if not file_bytes:
                # Zero-byte upload (dropped connection, empty file) — skip
                # rather than staging something that will fail downstream
                # with an unhelpful DuckDB/pandas parse error.
                continue
            store[uploaded_file.name] = file_bytes
        if len(store) > _MAX_STAGED_FILES:
            overflow = len(store) - _MAX_STAGED_FILES
            for stale_key in list(store.keys())[:overflow]:
                store.pop(stale_key, None)
        st.session_state["uploaded_files_store"] = store
    except Exception:  # noqa: BLE001
        return


def _activate_staged_file(filename: str, workspace_name: str) -> Dict[str, Any]:
    """Dispatches a staged file into the app.py ingestion pipeline (lazy import breaks circularity).

    STABILITY HARDENING: guarded by a one-shot in-flight flag
    ("_ingestion_in_flight") so a duplicate rerun landing on this call
    while a Parquet conversion is already executing can never launch a
    second, overlapping ingestion pass (which would itself trigger a
    second st.rerun() and reintroduce DOM-instability symptoms). The flag
    is always cleared in the `finally` block regardless of outcome, so a
    failed/errored ingestion never permanently locks out subsequent
    upload attempts.
    """
    if st.session_state.get("_ingestion_in_flight", False):
        return {
            "status": "failed",
            "filename": filename,
            "errors": ["An ingestion is already in progress. Please wait for it to complete."],
            "warnings": [],
        }

    from FRONTEND.app import process_uploaded_file  # local import: breaks circular import

    store: Dict[str, bytes] = st.session_state.get("uploaded_files_store", {})
    file_bytes = store.get(filename)
    if file_bytes is None:
        return {
            "status": "failed",
            "filename": filename,
            "errors": [f"Staged file '{filename}' could not be found. Please re-upload."],
            "warnings": [],
        }

    st.session_state["_ingestion_in_flight"] = True
    try:
        return process_uploaded_file(file_bytes=file_bytes, filename=filename, workspace_name=workspace_name)
    finally:
        st.session_state["_ingestion_in_flight"] = False
# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS — GLOBAL REPORT DISPATCH (EXPORT FORMATTING ONLY)
# ══════════════════════════════════════════════════════════════════════════════

def _prepare_df_for_excel_export(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Converts any residual ArrowDtype-backed column to a numpy-backed
    equivalent before handing the frame to openpyxl, since not every
    pandas/openpyxl version pair reliably serializes ArrowDtype extension
    arrays. A defensive no-op for the normal case (analytics_ready_dataframe
    is already numpy-backed via cleaner.py). Never raises."""
    if df is None or df.empty:
        return df
    working = df.copy(deep=False)
    for col in working.columns:
        try:
            pyarrow_dtype = getattr(working[col].dtype, "pyarrow_dtype", None)
            if pyarrow_dtype is None:
                continue
            working[col] = working[col].astype(object)
        except Exception:  # noqa: BLE001
            continue
    return working

def _export_dataframe_csv(df: Optional[pd.DataFrame]) -> Optional[bytes]:
    """Serializes a dataframe to CSV bytes. Never raises."""
    if df is None or df.empty:
        return None
    try:
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        return buffer.getvalue().encode("utf-8")
    except Exception:  # noqa: BLE001
        return None


def _export_dataframe_excel(df: Optional[pd.DataFrame], sheet_name: str = "Data") -> Optional[bytes]:
    """Serializes a dataframe to XLSX bytes via pandas' ExcelWriter."""
    if df is None or df.empty:
        return None
    df = _prepare_df_for_excel_export(df)
    try:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name[:31] or "Data")
        return buffer.getvalue()
    except Exception as exc:  # noqa: BLE001
        log_exception("sidebar._export_dataframe_excel", exc, severity="warning", context={"sheet_name": sheet_name})
        return None


def _export_executive_markdown() -> Optional[str]:
    """Generates the Executive Narrative Markdown report."""
    df: Optional[pd.DataFrame] = st.session_state.get("analytics_ready_dataframe")
    registry: Optional[ColumnRegistry] = st.session_state.get("column_registry")
    if df is None or df.empty or registry is None:
        return None
    try:
        markdown_report, _json_payload = generate_executive_narrative(df, registry)
        return markdown_report
    except Exception as exc:  # noqa: BLE001
        log_exception("sidebar._export_executive_markdown", exc)
        return None
def _export_executive_html() -> Optional[str]:
    """Generates the self-contained HTML5/CSS3 Executive Report string,
    mirroring _export_executive_markdown()'s exact structure and never-raise
    contract. Never raises."""
    df: Optional[pd.DataFrame] = st.session_state.get("analytics_ready_dataframe")
    registry: Optional[ColumnRegistry] = st.session_state.get("column_registry")
    if df is None or df.empty or registry is None:
        return None
    try:
        return generate_executive_html_report(df, registry)
    except Exception as exc:  # noqa: BLE001
        log_exception("sidebar._export_executive_html", exc)
        return None



def _export_audit_trace_csv() -> Optional[bytes]:
    """Serializes the current Audit Trace Log into CSV bytes."""
    audit_results: Dict[str, Any] = st.session_state.get("audit_results", {})
    entries: List[Dict[str, Any]] = audit_results.get("audit_entries", [])
    if not entries:
        return None
    return _export_dataframe_csv(pd.DataFrame(entries))


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC UNIVERSAL COMPONENT INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def validate() -> Dict[str, Any]:
    """Verifies the sidebar's structural preconditions. Never raises."""
    try:
        df: Optional[pd.DataFrame] = st.session_state.get("analytics_ready_dataframe")
        registry: Optional[ColumnRegistry] = st.session_state.get("column_registry")
        has_dataset = df is not None and not df.empty
        has_registry = registry is not None
        confirmed_mappings = (
            sum(1 for m in registry.mappings.values() if m.confirmed) if has_registry else 0
        )
        return {
            "component": _COMPONENT_NAME,
            "has_dataset": has_dataset,
            "has_registry": has_registry,
            "confirmed_mapping_count": confirmed_mappings,
            "is_ready": has_dataset and has_registry and confirmed_mappings > 0,
            "workspace_name": st.session_state.get("workspace_name", "Default Workspace"),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "component": _COMPONENT_NAME,
            "has_dataset": False,
            "has_registry": False,
            "confirmed_mapping_count": 0,
            "is_ready": False,
            "workspace_name": "Default Workspace",
            "error": str(exc),
        }


def refresh() -> None:
    """Purges localized execution caches. Never raises."""
    try:
        st.session_state["visualization_cache"] = {}
        st.session_state["analytics_results"] = {}
        st.session_state["dashboard_state"] = {}
    except Exception:  # noqa: BLE001
        return


def export(export_format: str) -> Optional[Any]:
    """Standalone programmatic export entry point. Never raises."""
    try:
        fmt = export_format.strip().lower()
        if fmt == "executive_markdown":
            return _export_executive_markdown()
        if fmt == "executive_html":
            return _export_executive_html()
        if fmt == "original_csv":
            return _export_dataframe_csv(st.session_state.get("uploaded_dataframe"))
        if fmt == "cleaned_csv":
            return _export_dataframe_csv(st.session_state.get("cleaned_dataframe"))
        if fmt == "analytics_csv":
            return _export_dataframe_csv(st.session_state.get("analytics_ready_dataframe"))
        if fmt == "analytics_excel":
            return _export_dataframe_excel(
                st.session_state.get("analytics_ready_dataframe"), sheet_name="Analytics Ready"
            )
        if fmt == "audit_csv":
            return _export_audit_trace_csv()
        return None
    except Exception:  # noqa: BLE001
        return None


def metadata() -> Dict[str, Any]:
    """Returns the sidebar component's capability descriptor. Never raises."""
    try:
        domain_label, domain_confidence = st.session_state.get("domain_detection", (DOMAIN_UNKNOWN, 0.0))
        return {
            "component": _COMPONENT_NAME,
            "supported_export_formats": [
                "executive_html", "original_csv",
                "cleaned_csv", "analytics_csv", "analytics_excel", "audit_csv",
            ],
            "supported_upload_types": list(SUPPORTED_UPLOAD_TYPES),
            "available_themes": list(THEMES.keys()),
            "active_domain": domain_label,
            "active_domain_confidence": domain_confidence,
            "max_staged_files": _MAX_STAGED_FILES,
        }
    except Exception as exc:  # noqa: BLE001
        return {"component": _COMPONENT_NAME, "error": str(exc)}


def render(**kwargs: Any) -> None:
    """
    Renders the complete Master Control sidebar in a HARDCODED, immutable
    section order: [Logo/Title] -> [Upload Section] -> [Workspace] ->
    [Pages] -> [Reports] -> [Settings]. No CSS is injected here —
    static/style.css (loaded once by app.py::main()) already covers the
    file-uploader overflow fix and every other rule this sidebar depends
    on. Never raises.
    """
    try:
        from FRONTEND.app import sync_notifications_with_registry
        sync_notifications_with_registry()
    except Exception:  # noqa: BLE001
        pass

    from core.themes import render_sidebar_section_header  # local import to avoid circular churn

    with st.sidebar:
        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
              st.image("D:/KESCO_ANALYTICS_PLATFORM/FRONTEND/components/LOGO_KESCO.jpg",width = "stretch", link= "https://kesco.org.in")
            try:
                _sidebar_dir = Path(__file__).resolve().parent
                logo_path = _sidebar_dir / "LOGO_KESCO.jpg"
                st.image("D:/KESCO_ANALYTICS_PLATFORM/FRONTEND/components/LOGO_KESCO.jpg",width = "stretch", link= "https://kesco.org.in")
    # 1. Official Company Logo (Global Brand Mark)
        # col1, col2, col3 = st.columns([1, 4, 1])
        # with col2:
        #     try:
        #         _sidebar_dir = Path(__file__).resolve().parent
        #         logo_path = _sidebar_dir / "LOGO_KESCO.jpg"
            
        #         if not logo_path.exists():
        #             # Linux (Streamlit Cloud) is case-sensitive — fall back
        #             # to a case-insensitive scan so a dev-machine-only
        #             # casing mismatch never crashes the sidebar in prod.
        #             _match = next(
        #                 (p for p in _sidebar_dir.glob("*")
        #                  if p.is_file() and p.name.lower() == "logo_kesco.jpg"),
        #                 None,
        #             )
        #             logo_path = _match if _match is not None else logo_path

        #         if logo_path.exists():
        #             st.image(str(logo_path), width="stretch", link="https://kesco.org.in")
        #         else:
        #             st.caption("KESCO")
        #     except Exception as exc:  # noqa: BLE001
        #         log_exception("sidebar.render.logo_image", exc, severity="info")
        #         st.caption("KESCO")
    # OR simply use standard Streamlit image rendering:
    # st.image(str(logo_path), use_container_width=True)
            
            #st.image("D:/KESCO_ANALYTICS_PLATFORM/FRONTEND/components/LOGO_KESCO.jpg",width = "stretch", link= "https://kesco.org.in")
               
    
        # ── LOGO / TITLE ──────────────────────────────────────────────
        st.markdown(f"## {APP_ICON} {APP_TITLE}")
        st.caption("Master Control · Unified Ingestion · Global Report Dispatch")
        st.divider()

        # ── 1. UPLOAD SECTION (Data Management) ──────────────────────────
        st.markdown(render_sidebar_section_header("📥", "Upload Dataset"), unsafe_allow_html=True)
        newly_uploaded = st.file_uploader(
            "Upload Dataset(s)", type=SUPPORTED_UPLOAD_TYPES, accept_multiple_files=True,
            key="_sidebar_multi_file_uploader",
            help=f"Supported formats: {', '.join(SUPPORTED_UPLOAD_TYPES)}. "
                 f"Up to {_MAX_STAGED_FILES} files may be staged at once.",
        )
        if newly_uploaded:
            _stage_uploaded_files(newly_uploaded)

        staged_files: Dict[str, bytes] = st.session_state.get("uploaded_files_store", {})
        if staged_files:
            staged_names = sorted(staged_files.keys())
            active_filename = st.session_state.get("active_filename")
            default_index = staged_names.index(active_filename) if active_filename in staged_names else 0
            selected_staged_file = st.selectbox(
                "Staged Files", options=staged_names, index=default_index,
                key="_sidebar_staged_file_select",
            )
            col_activate, col_remove = st.columns(2)
            with col_activate:
                if st.button("Activate", key="_sidebar_activate_file_btn", width="stretch"):
                    with st.spinner(
                        ENTERPRISE_COPY_MAP.get(
                            "Detecting Schema Topology, Resolving Domain Dictionaries, "
                            "Assembling KESCO Grid Reports...",
                            "Detecting Schema Topology, Resolving Domain Dictionaries, "
                            "Assembling KESCO Grid Reports...",
                        )
                    ):
                        outcome = _activate_staged_file(
                            selected_staged_file, st.session_state.get("workspace_name", "Default Workspace"),
                        )
                    if outcome["status"] == "success":
                        st.success(
                            f"Ingested '{outcome['filename']}' — {outcome['rows']:,} rows, "
                            f"{outcome['columns']} columns. Domain: {outcome['domain']} "
                            f"(Readiness: {outcome['readiness_score']}/100)."
                        )
                        st.rerun()
                    else:
                        for err in outcome.get("errors", []):
                            st.error(err)
                        for warn in outcome.get("warnings", []):
                            st.warning(warn)
            with col_remove:
                if st.button("Remove", key="_sidebar_remove_file_btn", width="stretch"):
                    staged_files.pop(selected_staged_file, None)
                    st.session_state["uploaded_files_store"] = staged_files
                    st.rerun()

        if st.session_state.get("uploaded_dataframe") is not None:
            st.caption("Active Dataset")
            domain_label, domain_conf = st.session_state.get("domain_detection", (DOMAIN_UNKNOWN, 0.0))
            st.write(f"**File:** {st.session_state.get('active_filename', '—')}")
            st.write(f"**Domain:** {domain_label} ({domain_conf:.0%} confidence)")
            st.write(f"**Readiness:** {st.session_state.get('readiness_score', 0)}/100")
            if st.button("Clear Active Dataset", key="_sidebar_clear_dataset_btn", width="stretch"):
                from FRONTEND.app import _clear_active_dataset
                _clear_active_dataset()
                st.rerun()

        # ── 2. WORKSPACE ──────────────────────────────────────────────────
        st.markdown(render_sidebar_section_header("🗂️", "Workspace"), unsafe_allow_html=True)
        workspace_profiles: Dict[str, Any] = st.session_state.get("workspace_profiles", {})
        current_workspace: str = st.session_state.get("workspace_name", "Default Workspace")
        workspace_input = st.text_input(
            "Workspace Name", value=current_workspace,
            key="_sidebar_workspace_name_input",
            help="Example workspaces: KESCO Complaints, Supply Operations, Revenue Analytics, "
                 "Asset Management, HR Analytics, Custom Projects.",
        )
        if workspace_input.strip() and workspace_input.strip() != current_workspace:
            st.session_state["workspace_name"] = workspace_input.strip()

        saved_workspace_names: List[str] = sorted(workspace_profiles.keys())
        if saved_workspace_names:
            col_load, col_save = st.columns(2)
            with col_load:
                selected_saved_workspace = st.selectbox(
                    "Load Saved Workspace", options=["—"] + saved_workspace_names,
                    key="_sidebar_workspace_load_select",
                )
                if selected_saved_workspace != "—" and st.button(
                    "Apply", key="_sidebar_apply_workspace_btn", width="stretch"
                ):
                    if _load_workspace_profile(selected_saved_workspace):
                        st.success(f"Workspace '{selected_saved_workspace}' restored.")
                        st.rerun()
                    else:
                        st.error(f"Could not restore workspace '{selected_saved_workspace}'.")
            with col_save:
                st.write("")
                st.write("")
                if st.button("Save Current Workspace", key="_sidebar_save_workspace_btn", width="stretch"):
                    target_name = st.session_state.get("workspace_name", "Default Workspace")
                    if _save_workspace_profile(target_name):
                        st.success(f"Workspace '{target_name}' saved.")
                    else:
                        st.error("Unable to save the current workspace profile.")
        else:
            if st.button("Save Current Workspace", key="_sidebar_save_workspace_btn_initial", width="stretch"):
                target_name = st.session_state.get("workspace_name", "Default Workspace")
                if _save_workspace_profile(target_name):
                    st.success(f"Workspace '{target_name}' saved.")
                else:
                    st.error("Unable to save the current workspace profile.")

        # ── 3. PAGES (single, authoritative page listing) ────────────────
        nav_pages: List[Any] = st.session_state.get("_nav_pages_registry", [])
        st.markdown(render_sidebar_section_header("🧭", "Pages"), unsafe_allow_html=True)
        # if nav_pages:                    COMMENTED OUT TO HIDE SELF SERVICE ANALYTICS PAGE
        #     for page in nav_pages:
        #         try:
        #             st.page_link(page)
        #         except Exception as exc:  # noqa: BLE001
        #             log_exception("sidebar.render.page_link", exc)
        #              continue

        if nav_pages:                         # 467 TO 480 LINES ARE TEMPORARY TO HIDE THAT PAGE
            for page in nav_pages:
                try:
                    # SUBMISSION MODE (temporary): hide Self-Service Analytics
                    # Builder from nav. Underlying page module is untouched.
                    _page_label = str(
                        getattr(page, "title", "") or getattr(page, "url_path", "")
                    ).lower()
                    if "self_service" in _page_label or "self-service" in _page_label:
                        continue
                    st.page_link(page)
                except Exception as exc:  # noqa: BLE001
                    log_exception("sidebar.render.page_link", exc)
                    continue
        else:
            st.caption("No pages are currently registered.")

        # ── 4. REPORTS (Report Center) ───────────────────────────────────
        st.markdown(render_sidebar_section_header("📤", "Reports"), unsafe_allow_html=True)
        validation_report = validate()
        if not validation_report.get("is_ready", False):
            st.caption(
                "Reports become available once a dataset is ingested and at least one "
                "role mapping is confirmed."
            )
        else:
            active_filename_stem = (st.session_state.get("active_filename") or "dataset").rsplit(".", 1)[0]
            # executive_html = _export_executive_html()
            # if executive_html:
            #     st.download_button(
            #         "Export Executive Report (HTML)", data=executive_html.encode("utf-8"),
            #         file_name=f"{active_filename_stem}_executive_report.html", mime="text/html",
            #         key="_sidebar_export_exec_html", width="stretch",
            #     )
            executive_html = _export_executive_html()
            if executive_html:
                st.sidebar.download_button(
                    label="📥 Download Executive HTML Report",
                    data=executive_html.encode("utf-8"),
                    file_name=f"{active_filename_stem}_executive_report.html",
                    mime="text/html",
                    key="_sidebar_export_exec_html",
                    width="stretch",
                )
            analytics_csv = _export_dataframe_csv(st.session_state.get("analytics_ready_dataframe"))
            if analytics_csv:
                st.download_button(
                    "Export Analytics-Ready Dataset (CSV)", data=analytics_csv,
                    file_name=f"{active_filename_stem}_analytics_ready.csv", mime="text/csv",
                    key="_sidebar_export_analytics_csv", width="stretch",
                )
            analytics_excel = _export_dataframe_excel(
                st.session_state.get("analytics_ready_dataframe"), sheet_name="Analytics Ready"
            )
            if analytics_excel:
                st.download_button(
                    "Export Analytics-Ready Dataset (Excel)", data=analytics_excel,
                    file_name=f"{active_filename_stem}_analytics_ready.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="_sidebar_export_analytics_xlsx", width="stretch",
                )
            else:
                st.caption("Excel export is currently unavailable (missing 'openpyxl' engine or empty dataset).")
            original_csv = _export_dataframe_csv(st.session_state.get("uploaded_dataframe"))
            if original_csv:
                st.download_button(
                    "Export Original Dataset (CSV)", data=original_csv,
                    file_name=f"{active_filename_stem}_original.csv", mime="text/csv",
                    key="_sidebar_export_original_csv", width="stretch",
                )
            audit_csv = _export_audit_trace_csv()
            if audit_csv:
                st.download_button(
                    "Export Audit Trace Log (CSV)", data=audit_csv,
                    file_name=f"{active_filename_stem}_audit_trace_log.csv", mime="text/csv",
                    key="_sidebar_export_audit_csv", width="stretch",
                )

        # ── 5. SETTINGS ───────────────────────────────────────────────────
        st.markdown(render_sidebar_section_header("⚙️", "Settings"), unsafe_allow_html=True)

        # theme_options: List[str] = list(THEMES.keys())
        # current_theme: str = st.session_state.get("theme", DEFAULT_THEME_KEY)
        # selected_theme: str = st.selectbox(
        #     "Enterprise Theme", options=theme_options,
        #     index=theme_options.index(current_theme) if current_theme in theme_options else 0,
        #     format_func=lambda k: THEMES[k]["name"],
        #     key="_sidebar_theme_select",
        # )
        #if selected_theme != st.session_state.get("theme"):
        #     st.session_state["theme"] = selected_theme
        # COMMENTED OUT FOR TEMPORARY HIDE THEME SELECTION 539 TO 548  AND ADD 552 TO 558 FOR HIDE THEME SELECTION

       # SUBMISSION MODE (temporary): hide theme selector, force Executive Dark only.
        _exec_theme_key = next(
                (k for k, v in THEMES.items() if v.get("name", "").strip().lower() == "executive dark"),
            DEFAULT_THEME_KEY,
        )
        _theme_actually_changed = st.session_state.get("theme") != _exec_theme_key
        if _theme_actually_changed:
            st.session_state["theme"] = _exec_theme_key
        selected_theme: str = _exec_theme_key
            # Synchronous, pre-rerun DOM toggle + sessionStorage persistence.
            # sessionStorage is what bootstrap_ui_engine()'s applyTheme()
            # reads FIRST on the next rerun's very first paint, so the
            # correct theme is already active before Streamlit's own
            # reconciler repaints anything — no flash, no wait on
            # st.rerun()'s round trip.
        st.html(f"""<script>
                (function() {{
                    try {{
                        window.sessionStorage.setItem('keds_theme', '{selected_theme}');
                        document.documentElement.setAttribute('data-theme', '{selected_theme}');
                        if (document.body) document.body.setAttribute('data-theme', '{selected_theme}');
                    }} catch (e) {{}}
                }})();
            </script>""")

        # ── RERUN-LOOP / 3D-FLIP-CARD STABILITY FIX ─────────────────────
        # CRITICAL: st.rerun() must NEVER fire unconditionally inside a
        # component that renders on every page load (this sidebar renders
        # before every st.navigation() dispatch). The previous code called
        # st.rerun() on every single render() invocation regardless of
        # whether the theme actually changed, which aborted the script
        # mid-execution and forced an immediate re-run — every single
        # time, on every widget interaction anywhere in the app (file
        # staging, Parquet "Activate" click, spinner completion, KPI
        # widget clicks, etc). That perpetual double/triple-execution
        # cycle never let the browser settle a single stable paint of the
        # .keds-kpi-card-container DOM subtree, which is what collapsed
        # the CSS perspective / transform-style: preserve-3d / backface-
        # visibility 3D flip into a flat 2D card.
        #
        # Fix: only force a rerun the FIRST time the theme is actually
        # changed this session (cold start), guarded by a one-shot
        # session_state flag so it can never fire again afterward. Every
        # subsequent render() call updates the DOM synchronously via the
        # st.html() script above (which already sets data-theme and
        # sessionStorage instantly, pre-rerun) — no rerun is needed for
        # that to take visual effect.
        if _theme_actually_changed and not st.session_state.get("_theme_force_rerun_done", False):
            st.session_state["_theme_force_rerun_done"] = True
            st.rerun()

        _readiness = st.session_state.get("readiness_score", 0)
        _readiness_band = st.session_state.get("readiness_band", "Critical")
        _dq_score = st.session_state.get("audit_results", {}).get("data_quality_score", 0)
        status_col_a, status_col_b = st.columns(2)
        status_col_a.metric("Readiness", f"{_readiness}", help=_readiness_band)
        status_col_b.metric("Data Quality", f"{_dq_score}")

        st.markdown(
            f'<div class="kesco-footer">Workspace: '
            f'{st.session_state.get("workspace_name", "Default Workspace")}</div>',
            unsafe_allow_html=True,
        )
