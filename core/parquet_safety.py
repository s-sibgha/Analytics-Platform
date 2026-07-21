"""
core/parquet_safety.py — Mixed-Type Ingestion Safety Layer for the
KESCO DuckDB + Parquet pushdown pipeline.

Root cause this module exists to close:
    PyArrow's `Table.from_pandas()` commits to a single Arrow leaf type
    per `object`-dtype column after sampling it, then raises
    `ArrowTypeError: Expected bytes, got a 'int' object` (or the inverse)
    the moment a later row doesn't match that committed type. Real-world
    utility complaint exports routinely produce exactly this shape:
    REMARKS mixing free text and bare numeric codes; SUBSTATION/
    SUBDIVISION mixing numeric IDs, alphanumeric names, and leading-zero
    strings; COMPLAINT_NO/CONSUMER_NO/CONSUMER_NAME mixing IDs and
    mistakenly-numeric entries.

This module is a drop-in, standalone layer — it does not alter, import,
or depend on `engine.duckdb_executor`, and every public function is a
pure function or an explicit disk-write with no hidden global state, so
it is safe to import directly into `app.py` alongside the existing
`convert_upload_to_parquet` pipeline without touching any existing class
structure.

Public API:
    read_uploaded_file(uploaded_file)        -> pd.DataFrame
    prepare_df_for_parquet(df, ...)          -> pd.DataFrame
    write_parquet_safely(df, path, ...)      -> Dict[str, Any]
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    _HAS_PYARROW = True
except ImportError:  # pragma: no cover — optional dependency
    pa = None  # type: ignore
    pq = None  # type: ignore
    _HAS_PYARROW = False

logger = logging.getLogger("kesco_platform.parquet_safety")

# ── Null-token normalization ────────────────────────────────────────────
# Every one of these, after stripping and lowercasing, is treated as a
# true null rather than a literal string value.
_NULL_TOKENS: frozenset = frozenset({
    "nan", "none", "null", "na", "n/a", "n.a.", "n.a", "nil", "nat",
    "<na>", "-", "--", "?", "unknown", "",
})

# Columns that are always treated as categorical/ID-like even if their
# name doesn't otherwise trigger the object/category dtype branch — kept
# here purely as a whitespace-trim priority list for the caller-facing
# docstring; the actual trimming below applies to EVERY object/category
# column, not just this list, since any of them could legitimately carry
# a utility dataset's ID/name columns under a different header spelling.
DEFAULT_CATEGORICAL_TRIM_HINTS: Set[str] = {
    "REMARKS", "SUBSTATION", "SUBDIVISION", "CONSUMER_NAME",
    "COMPLAINT_NO", "CONSUMER_NO", "DIVISION", "CIRCLE", "ZONE",
    "FEEDER", "TRANSFORMER", "OFFICER", "STATUS", "CATEGORY",
}

_PYARROW_STRING_DTYPE = "string[pyarrow]" if _HAS_PYARROW else "string"


# ══════════════════════════════════════════════════════════════════════
# 1. ROBUST STREAMLIT FILE HANDLER
# ══════════════════════════════════════════════════════════════════════

def read_uploaded_file(uploaded_file: Any) -> pd.DataFrame:
    """
    Safely reads a Streamlit `UploadedFile` (or any file-like object
    exposing `.name` and `.getvalue()`/`.read()`) into a pandas
    DataFrame, regardless of extension (.csv, .xlsx, .xls).

    Engine resolution order:
        CSV:  pandas' C engine first (fast); on any parse failure,
              retries with the Python engine and `on_bad_lines="skip"`
              so a single malformed row can never abort the whole read.
        XLSX: python-calamine (fastest Rust-backed reader) -> openpyxl
              fallback if calamine is not installed or errors.
        XLS:  xlrd (the only engine that reads legacy .xls) -> openpyxl
              as a last-resort attempt (covers mislabeled .xls files
              that are actually .xlsx under the hood).

    Never silently returns an empty frame on a real parse failure — it
    raises the last encountered exception once every fallback has been
    exhausted, so the caller (Streamlit UI) can surface an accurate
    error to the user instead of silently proceeding with no data.

    Args:
        uploaded_file: A Streamlit `UploadedFile` instance (or
            file-like object with `.name` and byte-content access).

    Returns:
        The parsed pandas DataFrame (never None).

    Raises:
        ValueError: if the file extension is unsupported.
        Exception: the last engine-specific exception if every
            available engine fails to parse the file.
    """
    filename: str = getattr(uploaded_file, "name", "uploaded_file")
    suffix = Path(filename).suffix.lower().lstrip(".")

    try:
        raw_bytes: bytes = uploaded_file.getvalue()
    except AttributeError:
        raw_bytes = uploaded_file.read()

    if suffix == "csv":
        return _read_csv_bytes(raw_bytes, filename)
    if suffix == "xlsx":
        return _read_excel_bytes(raw_bytes, filename, engines=("calamine", "openpyxl"))
    if suffix == "xls":
        return _read_excel_bytes(raw_bytes, filename, engines=("xlrd", "openpyxl", "calamine"))

    raise ValueError(
        f"Unsupported file extension '.{suffix}' for '{filename}'. "
        "Supported types: csv, xlsx, xls."
    )


def _read_csv_bytes(raw_bytes: bytes, filename: str) -> pd.DataFrame:
    """CSV read with a C-engine-first, Python-engine-fallback strategy."""
    last_exc: Optional[Exception] = None

    # Attempt 1 — fast C engine, UTF-8.
    try:
        return pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        last_exc = exc
        logger.warning("CSV C-engine (utf-8) read failed for '%s': %s", filename, exc)

    # Attempt 2 — C engine, latin-1 (tolerates byte sequences UTF-8 rejects).
    try:
        return pd.read_csv(io.BytesIO(raw_bytes), encoding="latin-1")
    except Exception as exc:  # noqa: BLE001
        last_exc = exc
        logger.warning("CSV C-engine (latin-1) read failed for '%s': %s", filename, exc)

    # Attempt 3 — Python engine, tolerant of ragged/malformed rows.
    try:
        return pd.read_csv(
            io.BytesIO(raw_bytes),
            engine="python",
            encoding="utf-8",
            on_bad_lines="skip",
            sep=None,  # sniff the delimiter defensively
        )
    except Exception as exc:  # noqa: BLE001
        last_exc = exc
        logger.error("CSV Python-engine fallback failed for '%s': %s", filename, exc)

    assert last_exc is not None
    raise last_exc


def _read_excel_bytes(
    raw_bytes: bytes, filename: str, engines: Iterable[str]
) -> pd.DataFrame:
    """Excel read trying each engine in `engines` order until one succeeds."""
    last_exc: Optional[Exception] = None
    for engine in engines:
        try:
            return pd.read_excel(io.BytesIO(raw_bytes), engine=engine)
        except ImportError as exc:
            # Engine package not installed — try the next one silently.
            last_exc = exc
            continue
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "Excel engine '%s' failed for '%s': %s", engine, filename, exc
            )
            continue

    assert last_exc is not None
    raise last_exc


# ══════════════════════════════════════════════════════════════════════
# 2. MIXED-TYPE / OBJECT-COLUMN SANITIZER
# ══════════════════════════════════════════════════════════════════════

def _normalize_scalar(value: Any) -> Optional[str]:
    """
    Cleans a single cell value into either `None` (true null) or a
    clean string representation, per these rules:
        1. Any pandas/numpy null (`NaN`, `NaT`, `None`, `pd.NA`) -> None.
        2. A whole-number float (e.g. 11.0) or any int/numpy-int
           -> its clean integer string ("11"), never "11.0".
        3. A non-whole float (e.g. 11.5) -> Python's default str repr,
           trimmed.
        4. A bool -> "True"/"False" (kept explicit rather than "1"/"0"
           so a boolean column never collides with a numeric-ID column
           after stringification).
        5. Anything else -> str(value), whitespace-trimmed; if the
           trimmed, lowercased result matches a known null token
           (see `_NULL_TOKENS`), returns None instead of the literal
           string "nan"/"null"/etc.
    Never raises: any unexpected type falls through to str(value).
    """
    if value is None:
        return None
    try:
        if isinstance(value, float) and np.isnan(value):
            return None
    except TypeError:
        pass
    if value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, bool):
        return "True" if value else "False"

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        fval = float(value)
        if not np.isfinite(fval):
            return None
        if fval.is_integer():
            return str(int(fval))
        # Trim trailing zeros from a normal float repr without losing
        # precision (e.g. 11.50 -> "11.5", 11.0001 -> "11.0001").
        text = f"{fval:.10f}".rstrip("0").rstrip(".")
        return text if text else "0"

    try:
        text = str(value).strip()
    except Exception:  # noqa: BLE001
        return None

    if not text:
        return None
    if text.strip().lower() in _NULL_TOKENS:
        return None
    return text


def _column_has_mixed_types(series: pd.Series) -> bool:
    """
    Returns True if `series` (an object/category-dtype column) contains
    more than one Python type bucket among its non-null values, where
    buckets are: bool / int / float / str / other. This is a diagnostic
    signal only — `prepare_df_for_parquet` normalizes ALL object/category
    columns uniformly regardless of this flag, since a uniformly-typed
    object column can still carry unclean nulls, leading-zero-losing
    floats, or untrimmed whitespace that would otherwise silently corrupt
    downstream SQL group-bys.
    """
    non_null = series.dropna()
    if non_null.empty:
        return False
    buckets: Set[str] = set()
    for v in non_null.head(5000):  # bounded sample — cheap, representative
        if isinstance(v, bool):
            buckets.add("bool")
        elif isinstance(v, (int, np.integer)):
            buckets.add("int")
        elif isinstance(v, (float, np.floating)):
            buckets.add("float")
        elif isinstance(v, str):
            buckets.add("str")
        else:
            buckets.add("other")
        if len(buckets) > 1:
            return True
    return len(buckets) > 1


def prepare_df_for_parquet(
    df: pd.DataFrame,
    *,
    extra_trim_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Sanitizes every `object`/`category`-dtype column in `df` so it can be
    handed to `pyarrow.Table.from_pandas()` without raising a mixed-type
    `ArrowTypeError`. Returns a NEW DataFrame — the caller's original
    reference is never mutated in place.

    Transformation steps applied to every object/category column:
        1. Null-token normalization: string spellings of null
           ("nan", "NaN", "None", "null", "N/A", "-", "", etc.) are
           coerced back to true `np.nan` / `None`, never left as
           literal text.
        2. Whole-number float cleanup: a float cell like `11.0`
           (the typical shape produced by an Excel numeric-ID column
           read into pandas) becomes the clean string `"11"`, never
           `"11.0"`.
        3. Uniform stringification: every remaining non-null cell
           (int, float, bool, str, or any other Python object) is
           coerced to a single consistent string representation, so a
           column mixing `"Meter burnt"` and `104294` becomes a
           homogeneous string column PyArrow can serialize deterministically.
        4. Whitespace trimming: leading/trailing whitespace is
           stripped from every value (this is what step 3's
           `_normalize_scalar` already does), preventing
           `"Kalyanpur "` and `"Kalyanpur"` from silently fragmenting
           into two distinct SQL GROUP BY keys downstream.
        5. Explicit PyArrow-string casting: the fully-cleaned column is
           cast to `"string[pyarrow]"` (falling back to pandas'
           `"string"` dtype if the `pyarrow` package/backend is
           unavailable), guaranteeing PyArrow serializes it as a single
           `utf8`/`large_utf8` Arrow leaf type — never re-inferring a
           mixed schema from raw Python objects.

    Numeric (`int64`/`float64`), boolean, and datetime64 dtype columns
    are left completely untouched — PyArrow already serializes those
    natively and correctly; touching them here would only add risk for
    zero benefit.

    Args:
        df: The DataFrame to sanitize (not mutated).
        extra_trim_columns: Optional iterable of additional column names
            to guarantee are treated as categorical/trim-priority even
            if their dtype inference is ambiguous. Purely documentary —
            every object/category column is already fully sanitized
            regardless of whether it appears in this set.

    Returns:
        A new, PyArrow-safe DataFrame. Never raises: any single-column
        sanitization failure is logged and that column is defensively
        cast to plain `str` as an absolute fallback rather than
        propagating an exception that would abort the whole ingestion.
    """
    working = df.copy(deep=True)
    trim_priority: Set[str] = set(DEFAULT_CATEGORICAL_TRIM_HINTS)
    if extra_trim_columns:
        trim_priority.update(extra_trim_columns)

    for col in working.columns:
        try:
            series = working[col]
            dtype = series.dtype

            is_object = dtype == object
            is_category = isinstance(dtype, pd.CategoricalDtype)
            if not (is_object or is_category):
                continue

            if is_category:
                series = series.astype(object)

            had_mixed_types = _column_has_mixed_types(series)
            if had_mixed_types:
                logger.info(
                    "Column '%s' contains mixed Python types — normalizing "
                    "to a uniform PyArrow-safe string column.", col,
                )

            cleaned = series.map(_normalize_scalar)

            try:
                working[col] = cleaned.astype(_PYARROW_STRING_DTYPE)
            except (TypeError, ValueError):
                # Extremely defensive fallback: if the pyarrow-backed
                # StringDtype construction itself fails for any reason
                # (e.g. an unexpected pandas/pyarrow version mismatch),
                # degrade to plain pandas "string" dtype rather than
                # leaving the column unsanitized.
                working[col] = cleaned.astype("string")

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Sanitization failed for column '%s': %s — applying "
                "absolute str() fallback.", col, exc,
            )
            try:
                working[col] = working[col].apply(
                    lambda v: None if pd.isna(v) else str(v).strip()
                ).astype("string")
            except Exception as fallback_exc:  # noqa: BLE001
                logger.critical(
                    "Absolute fallback also failed for column '%s': %s — "
                    "column left unmodified; parquet write may still fail.",
                    col, fallback_exc,
                )

    return working

def sanitize_for_parquet(
    df: pd.DataFrame,
    *,
    extra_trim_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Stable, explicitly-named public entry point for pre-Parquet dataframe
    sanitization — wired directly into FRONTEND.app.convert_upload_to_parquet
    immediately after the raw upload is parsed into a DataFrame and BEFORE
    any PyArrow/Parquet write is attempted.

    This is a thin, zero-duplicate-logic wrapper around
    `prepare_df_for_parquet` — every actual transformation (null-token
    normalization, whole-number float cleanup, uniform stringification,
    whitespace trimming, PyArrow-string casting) lives exclusively in that
    function. This wrapper exists solely so callers have a single, stable,
    intention-revealing name to depend on for the "sanitize this DataFrame
    for Parquet" step of the pipeline, independent of that function's
    internal naming.

    Args:
        df: The raw DataFrame to sanitize (never mutated in place — a new
            DataFrame is always returned).
        extra_trim_columns: Optional additional column names to flag as
            categorical/trim-priority (see `prepare_df_for_parquet`).

    Returns:
        A new, PyArrow-safe DataFrame with every object/category column
        normalized to a uniform `string[pyarrow]` (or `string`) dtype.
        Never raises — delegates entirely to `prepare_df_for_parquet`'s
        own per-column exception containment, which degrades to a plain
        `str()`-cast fallback rather than propagating on any single
        column's sanitization failure.
    """
    return prepare_df_for_parquet(df, extra_trim_columns=extra_trim_columns)
# ══════════════════════════════════════════════════════════════════════
# 3. SAFE PARQUET WRITER (PyArrow primary, fastparquet fallback)
# ══════════════════════════════════════════════════════════════════════

def write_parquet_safely(
    df: pd.DataFrame,
    path: str,
    *,
    compression: str = "zstd",
    sanitize: bool = True,
    row_group_size: int = 128_000,
    use_dictionary: bool = True,
) -> Dict[str, Any]:
    """
    (docstring unchanged above this point — additive params only)

    row_group_size: Target rows per Parquet row group. 128,000 balances
        DuckDB's parallel-scan granularity (more row groups = more
        parallelizable scan units) against per-group metadata overhead.
        Lower it (e.g. 32,000) for very wide tables; raise it for very
        narrow, high-row-count tables.
    use_dictionary: Enables Parquet dictionary encoding, which is highly
        effective for the platform's low-cardinality categorical columns
        (STATUS, ZONE, CIRCLE, DIVISION, SUBSTATION names, etc.) —
        typically a 3-8x size/read-speed win on those columns with zero
        downside for high-cardinality columns (Parquet auto-falls-back
        to plain encoding per-column when dictionary size exceeds a
        threshold, so this is safe to leave on globally).
    """
    working_df = prepare_df_for_parquet(df) if sanitize else df

    pyarrow_error: Optional[Exception] = None
    if _HAS_PYARROW:
        try:
            table = pa.Table.from_pandas(working_df, preserve_index=False)
            pq.write_table(
                table,
                path,
                compression=compression,
                row_group_size=row_group_size,
                use_dictionary=use_dictionary,
                write_statistics=True,  # enables DuckDB row-group pruning (predicate pushdown)
            )
            return {
                "path": path,
                "engine_used": "pyarrow",
                "row_count": int(len(working_df)),
                "column_count": int(working_df.shape[1]),
                "sanitized": sanitize,
                "compression": compression,
                "row_group_size": row_group_size,
            }
        except Exception as exc:  # noqa: BLE001
            pyarrow_error = exc
            logger.error(
                "PyArrow parquet write failed for '%s' even after sanitization: %s",
                path, exc,
            )
    else:
        pyarrow_error = ImportError("pyarrow is not installed")

    # ── Fallback: fastparquet ────────────────────────────────────────
    try:
        working_df.to_parquet(path, engine="fastparquet", compression=compression)
        logger.warning(
            "Parquet write for '%s' succeeded via fastparquet fallback "
            "after a PyArrow failure: %s", path, pyarrow_error,
        )
        return {
            "path": path,
            "engine_used": "fastparquet",
            "row_count": int(len(working_df)),
            "column_count": int(working_df.shape[1]),
            "sanitized": sanitize,
        }
    except Exception as fastparquet_error:  # noqa: BLE001
        raise RuntimeError(
            f"Parquet write failed on BOTH engines for '{path}'.\n"
            f"PyArrow error: {pyarrow_error}\n"
            f"fastparquet error: {fastparquet_error}"
        ) from fastparquet_error