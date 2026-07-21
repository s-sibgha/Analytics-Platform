"""
utils/grid_utils.py — Enterprise Data Grid Rendering Utility

Milestone 14 / Height-Crash, Arrow-Type-Crash & Width-Deprecation
remediation:
  • HEIGHT CRASH FIX: `height=None` is no longer forwarded verbatim to
    `st.data_editor` / `st.dataframe` on the code paths that previously
    did so implicitly. `height` is now resolved to a guaranteed integer
    fallback (`_DEFAULT_GRID_HEIGHT = 400`) whenever the caller does not
    supply one, eliminating the runtime crash observed when the modern
    Streamlit layout engine received an unresolved `None` height during
    its internal size-calculation pass.
  • ARROW TYPE CRASH FIX: every dataframe is now passed through
    `sanitize_dataframe_for_display()` before being handed to
    `st.dataframe` / `st.data_editor`. This coerces any object-dtype
    column containing a heterogeneous mix of Python types (the classic
    failure case: a "Remarks"/"Comments"/"Notes" column holding both `int`
    and `str` values) into a uniform pandas `string` dtype, preventing
    PyArrow's schema inference from raising
    `ArrowTypeError: Expected bytes, got a 'int' object` (or its inverse)
    mid-render.
  • WIDTH DEPRECATION FIX: this module's own public signature
    (`use_container_width: bool`) is preserved byte-for-byte so every
    existing call site across the platform (1_dashboard.py, 2_audit.py,
    schema_mapping.py) continues to compile with zero changes. Internally,
    however, the boolean is now translated to the modern `width=` keyword
    (`"stretch"` / `"content"`) before being forwarded to
    `st.dataframe` / `st.data_editor`, so the deprecated
    `use_container_width` keyword itself is never actually passed to a
    live Streamlit widget call from this module again.

This module still performs NO business logic, NO aggregation, and NO
data mutation beyond the display-safety coercions described above — it
only infers a display column_config from already-computed dataframe
dtypes/column-name heuristics and renders the frame. Never raises: any
internal failure degrades to a plain, unconfigured st.dataframe render
(or a neutral caption) rather than crashing the host page.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# Column-name substring heuristics used ONLY for display-format inference
# (NumberColumn/DatetimeColumn/TextColumn choice) — never for business
# logic, role resolution, or aggregation. A column's role/meaning is never
# reinterpreted here; only its *rendering* is refined.
_PERCENT_NAME_HINTS: tuple = ("%", "rate", "pct", "percent", "compliance", "confidence")
_CURRENCY_NAME_HINTS: tuple = ("amount", "revenue", "collected", "target", "bill")

# Column-name substring heuristics that force string coercion regardless
# of observed type-homogeneity, since free-text narrative columns are the
# single most common source of Arrow mixed-type schema-inference failures
# (numeric-looking remarks like "0" or "12" sitting alongside prose).
_FORCE_STRING_NAME_HINTS: tuple = ("remark", "comment", "note", "description", "reason")

# Guaranteed fallback height (pixels) used whenever a caller does not
# supply an explicit height, preventing `height=None` from ever reaching
# st.dataframe/st.data_editor on this code path.
_DEFAULT_GRID_HEIGHT: int = 400


def _resolve_width_kwarg(use_container_width: bool) -> str:
    """
    Translates the module's legacy `use_container_width: bool` public
    parameter into the modern Streamlit `width=` keyword value. Never
    raises. `"stretch"` mirrors the old `use_container_width=True`
    behavior (fill the available column width); `"content"` mirrors
    `use_container_width=False` (size to the widget's natural content
    width).
    """
    return "stretch" if use_container_width else "content"


def _resolve_grid_height(height: Optional[int]) -> int:
    """
    Guarantees a concrete integer height is always resolved before being
    forwarded to st.dataframe/st.data_editor, eliminating the
    `height=None` layout-calculation crash observed on modern Streamlit
    runtimes. Never raises; a non-positive or invalid height defensively
    falls back to `_DEFAULT_GRID_HEIGHT`.
    """
    try:
        if height is None:
            return _DEFAULT_GRID_HEIGHT
        resolved = int(height)
        return resolved if resolved > 0 else _DEFAULT_GRID_HEIGHT
    except (TypeError, ValueError):
        return _DEFAULT_GRID_HEIGHT


# def sanitize_dataframe_for_display(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
#     """
#     Coerces columns with heterogeneous Python types — the classic case
#     being a "Remarks"/"Comments"/"Notes" column that mixes `int`, `float`,
#     and `str` values within the same object-dtype column — into a uniform
#     pandas `string` dtype before the frame is handed to Streamlit's
#     Arrow-backed `st.dataframe` / `st.data_editor` renderers.

#     PyArrow's schema inference raises
#     `ArrowTypeError: Expected bytes, got a 'int' object` (or the inverse,
#     depending on which type it samples first) whenever an object-dtype
#     column contains more than one distinct Python type among its non-null
#     values. This function eliminates that entire class of crash without
#     touching numeric, datetime, or boolean columns, which Arrow already
#     handles natively and correctly.

#     Exposed as a public, non-underscore function so pages that construct
#     their own raw `st.dataframe` / `st.data_editor` calls (bypassing
#     `render_enterprise_grid`, e.g. for folium-map companion tables) can
#     apply the exact same safety pass explicitly.

#     Never raises: any per-column inspection failure defensively casts
#     that column to string as the safest fallback; a `None` or empty input
#     is returned unchanged.
#     """
#     if df is None or df.empty:
#         return df
#     working = df.copy(deep=True)
#     for col in working.columns:
#         try:
#             series = working[col]
#             if (
#                 pd.api.types.is_numeric_dtype(series)
#                 or pd.api.types.is_datetime64_any_dtype(series)
#                 or pd.api.types.is_bool_dtype(series)
#             ):
#                 continue

#             col_l = str(col).lower()
#             forced_string = any(hint in col_l for hint in _FORCE_STRING_NAME_HINTS)

#             if series.dtype == object or pd.api.types.is_categorical_dtype(series):
#                 non_null = series.dropna()
#                 distinct_types = {type(v) for v in non_null.tolist()} if len(non_null) else set()
#                 mixed_types = len(distinct_types) > 1

#                 if forced_string or mixed_types:
#                     working[col] = series.apply(
#                         lambda v: str(v) if pd.notna(v) else v
#                     ).astype("string")
#         except Exception:  # noqa: BLE001
#             try:
#                 working[col] = working[col].astype(str)
#             except Exception:  # noqa: BLE001
#                 continue
#     return working


def sanitize_dataframe_for_display(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Coerces columns with heterogeneous Python types — the classic case
    being a "Remarks"/"Comments"/"Notes" column that mixes `int`, `float`,
    and `str` values within the same object-dtype column — into a uniform
    pandas `string` dtype before the frame is handed to Streamlit's
    Arrow-backed `st.dataframe` / `st.data_editor` renderers.

    W2 fix: replaced the deprecated `pd.api.types.is_categorical_dtype`
    call with `isinstance(series.dtype, pd.CategoricalDtype)` — the former
    is slated for removal in a future pandas release and already emits
    FutureWarning on current versions.

    Never raises: any per-column inspection failure defensively casts
    that column to string as the safest fallback; a `None` or empty input
    is returned unchanged.
    """
    if df is None or df.empty:
        return df
    working = df.copy(deep=True)
    for col in working.columns:
        try:
            series = working[col]
            if (
                pd.api.types.is_numeric_dtype(series)
                or pd.api.types.is_datetime64_any_dtype(series)
                or pd.api.types.is_bool_dtype(series)
            ):
                continue

            col_l = str(col).lower()
            forced_string = any(hint in col_l for hint in _FORCE_STRING_NAME_HINTS)

            if series.dtype == object or isinstance(series.dtype, pd.CategoricalDtype):
                non_null = series.dropna()
                distinct_types = {type(v) for v in non_null.tolist()} if len(non_null) else set()
                mixed_types = len(distinct_types) > 1

                if forced_string or mixed_types:
                    working[col] = series.apply(
                        lambda v: str(v) if pd.notna(v) else v
                    ).astype("string")
        except Exception:  # noqa: BLE001
            try:
                working[col] = working[col].astype(str)
            except Exception:  # noqa: BLE001
                continue
    return working

# def _infer_column_config(df: pd.DataFrame) -> Dict[str, Any]:
#     """
#     Infers a per-column st.column_config.* mapping purely from each
#     column's pandas dtype and column-name heuristics. Never raises: any
#     per-column inference failure is skipped (that column falls back to
#     Streamlit's own default rendering) rather than aborting the whole
#     grid's configuration.
#     """
#     config: Dict[str, Any] = {}
#     try:
#         for col in df.columns:
#             col_l = str(col).lower()
#             try:
#                 series = df[col]
#                 if pd.api.types.is_datetime64_any_dtype(series):
#                     config[col] = st.column_config.DatetimeColumn(str(col))
#                     continue
#                 if pd.api.types.is_bool_dtype(series):
#                     config[col] = st.column_config.CheckboxColumn(str(col), disabled=True)
#                     continue
#                 if pd.api.types.is_numeric_dtype(series):
#                     if any(hint in col_l for hint in _PERCENT_NAME_HINTS):
#                         config[col] = st.column_config.NumberColumn(str(col), format="%.2f%%")
#                     elif any(hint in col_l for hint in _CURRENCY_NAME_HINTS):
#                         config[col] = st.column_config.NumberColumn(str(col), format="₹%.2f")
#                     else:
#                         config[col] = st.column_config.NumberColumn(str(col), format="%.2f")
#                     continue
#                 config[col] = st.column_config.TextColumn(str(col))
#             except Exception:  # noqa: BLE001
#                 continue
#     except Exception:  # noqa: BLE001
#         return {}
#     return config

def _infer_column_config(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Infers a per-column st.column_config.* mapping purely from each
    column's pandas dtype and column-name heuristics. Never raises: any
    per-column inference failure is skipped (that column falls back to
    Streamlit's own default rendering) rather than aborting the whole
    grid's configuration.

    W3 fix: categorical-dtype columns (e.g. cleaner.py's `_build_analytics_df`
    status canonicalization, or a Schema Mapping Studio CATEGORICAL
    override) previously fell through every branch to the generic
    TextColumn — losing the dropdown-style rendering AG-Grid/Streamlit
    would otherwise give them. Now explicitly mapped to SelectboxColumn.
    """
    config: Dict[str, Any] = {}
    try:
        for col in df.columns:
            col_l = str(col).lower()
            try:
                series = df[col]
                if pd.api.types.is_datetime64_any_dtype(series):
                    config[col] = st.column_config.DatetimeColumn(str(col))
                    continue
                if pd.api.types.is_bool_dtype(series):
                    config[col] = st.column_config.CheckboxColumn(str(col), disabled=True)
                    continue
                if isinstance(series.dtype, pd.CategoricalDtype):
                    try:
                        category_options = [str(c) for c in series.cat.categories.tolist()]
                    except Exception:  # noqa: BLE001
                        category_options = []
                    config[col] = st.column_config.SelectboxColumn(
                        str(col), options=category_options, disabled=True,
                    )
                    continue
                if pd.api.types.is_numeric_dtype(series):
                    if any(hint in col_l for hint in _PERCENT_NAME_HINTS):
                        config[col] = st.column_config.NumberColumn(str(col), format="%.2f%%")
                    elif any(hint in col_l for hint in _CURRENCY_NAME_HINTS):
                        config[col] = st.column_config.NumberColumn(str(col), format="₹%.2f")
                    else:
                        config[col] = st.column_config.NumberColumn(str(col), format="%.2f")
                    continue
                config[col] = st.column_config.TextColumn(str(col))
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        return {}
    return config

def render_enterprise_grid(
    df: Optional[pd.DataFrame],
    key: str,
    *,
    use_container_width: bool = True,
    hide_index: bool = True,
    editable: bool = False,
    height: Optional[int] = None,
    search_columns: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Shared enterprise-grid rendering primitive. Applies a consistently
    inferred column_config (numeric/percentage/currency/date/boolean
    formatting) in place of a bare st.dataframe/st.data_editor call.
    Native per-column sort and Streamlit's built-in toolbar search icon
    are provided by st.dataframe/st.data_editor itself — this function
    does not reimplement them.

    Milestone 14: every dataframe now passes through
    `sanitize_dataframe_for_display()` before rendering (Arrow mixed-type
    crash fix), `height` is always resolved to a concrete integer before
    being forwarded to the underlying widget (height=None crash fix), and
    the legacy `use_container_width` boolean is translated internally to
    the modern `width=` keyword rather than being forwarded verbatim
    (deprecation-warning fix). The public signature is otherwise
    unchanged — every existing call site continues to work with zero
    modification.

    If `search_columns` is provided, an optional free-text filter input is
    rendered above the grid, narrowing rows via a case-insensitive
    substring match across the named columns.

    Args:
        df: The dataframe to render. A None or empty dataframe renders a
            neutral caption instead of an empty grid.
        key: Unique Streamlit widget key for this grid instance.
        use_container_width: Legacy sizing flag, translated internally to
            `width="stretch"` (True) or `width="content"` (False).
        hide_index: Passed through to st.dataframe/st.data_editor.
        editable: If True, renders via st.data_editor(disabled=True)
            (matching the platform's existing "editable widget, but
            disabled for read-only display" convention); if False, renders
            via the lighter-weight st.dataframe.
        height: Optional fixed pixel height. Resolved to
            `_DEFAULT_GRID_HEIGHT` (400) when not supplied.
        search_columns: Optional list of column names to expose a
            free-text filter over.

    Returns:
        The (possibly search-filtered, always Arrow-safe) dataframe that
        was rendered, or None if nothing was rendered. Never raises.
    """
    if df is None or df.empty:
        st.caption("No data is currently available for this table.")
        return None

    working_df = sanitize_dataframe_for_display(df)
    if working_df is None:
        st.caption("No data is currently available for this table.")
        return None

    try:
        if search_columns:
            valid_search_cols = [c for c in search_columns if c in working_df.columns]
            if valid_search_cols:
                st.markdown('<div class="keds-grid-toolbar-shell">', unsafe_allow_html=True)
                search_text = st.text_input(
                    "Search",
                    key=f"{key}_search",
                    placeholder=f"Search across {', '.join(valid_search_cols)}…",
                    label_visibility="collapsed",
                )
                st.markdown('</div>', unsafe_allow_html=True)
                if search_text.strip():
                    needle = search_text.strip().lower()
                    mask = pd.Series(False, index=working_df.index)
                    for col in valid_search_cols:
                        mask |= working_df[col].astype(str).str.lower().str.contains(needle, na=False)
                    working_df = working_df.loc[mask]
    except Exception:  # noqa: BLE001
        working_df = sanitize_dataframe_for_display(df)

    resolved_width = _resolve_width_kwarg(use_container_width)
    resolved_height = _resolve_grid_height(height)

    try:
        column_config = _infer_column_config(working_df)
        if editable:
            st.data_editor(
                working_df,
                width=resolved_width,
                hide_index=hide_index,
                key=key,
                disabled=True,
                column_config=column_config,
                height=resolved_height,
            )
        else:
            st.dataframe(
                working_df,
                width=resolved_width,
                hide_index=hide_index,
                key=key,
                column_config=column_config,
                height=resolved_height,
            )
    except Exception:  # noqa: BLE001
        try:
            st.dataframe(
                working_df,
                width=resolved_width,
                hide_index=hide_index,
                key=f"{key}_fallback",
                height=resolved_height,
            )
        except Exception:  # noqa: BLE001
            st.caption("This table could not be rendered.")
            return None

    if working_df.empty and search_columns:
        st.caption("No rows match the current search.")

    return working_df