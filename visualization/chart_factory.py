from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from engine.duckdb_executor import (
    should_use_duckdb,
    duckdb_group_aggregate,
    duckdb_group_aggregate_from_parquet,
)
from core.themes import DEFAULT_THEME_KEY, THEMES, Theme, get_categorical_palette
# AFTER
from core.roles import (
    ROLE_AMOUNT,
    ROLE_CATEGORY,
    ROLE_CIRCLE,
    ROLE_CLOSING_DATE,
    ROLE_CONSUMER_ID,
    ROLE_DIVISION,
    ROLE_FEEDER,
    ROLE_OFFICER,
    ROLE_PRIORITY,
    ROLE_RECORD_ID,
    ROLE_REGISTRATION_DATE,
    ROLE_STATUS, 
    ROLE_REOPEN_FLAG, 
    ROLE_SLA_DEADLINE,
    ROLE_SUBCATEGORY,
    ROLE_SUBDIVISION,
    ROLE_SUBSTATION,
    ROLE_TRANSFORMER,
    ROLE_ZONE,
    ROLE_LATITUDE,
    ROLE_LONGITUDE,
    CANONICAL_GEO_HIERARCHY,
)
 
from core.themes import DEFAULT_THEME_KEY, THEMES, Theme
from core.column_registry import ColumnRegistry
import folium
from folium.plugins import (
    MarkerCluster,
    HeatMap,
    Fullscreen,
    MiniMap,
    MeasureControl,
    MousePosition,
    Search,
)
# ── Type alias ─────────────────────────────────────────────────────────────────
ChartReturn = Tuple[Optional[go.Figure], Optional[pd.DataFrame], Dict[str, Any]]

# ── Layout constants ────────────────────────────────────────────────────────────
_FONT_FAMILY: str = "Segoe UI, Inter, -apple-system, Arial, sans-serif"
_DEFAULT_HEIGHT: int = 420
_SPARKLINE_HEIGHT: int = 52
_SPARKLINE_WIDTH: int = 200
_MAX_SCATTER_ROWS: int = 5_000
_MAX_HORIZONTAL_BAR_CATEGORIES: int = 100
_DEFAULT_TOP_N: int = 0  # 0 = no limit

# ── Colour palettes ─────────────────────────────────────────────────────────────
_CATEGORICAL_15: List[str] = [
    "#003B73", "#E65100", "#1B5E20", "#4A148C", "#006064",
    "#BF360C", "#37474F", "#F57F17", "#880E4F", "#1565C0",
    "#558B2F", "#0277BD", "#AD1457", "#00695C", "#25329A",
]
_SEQUENTIAL_BLUES: List[str] = [
    "#E3F2FD", "#90CAF9", "#42A5F5", "#1E88E5", "#1565C0",
    "#003B73",
]
_STATUS_COLOR_MAP: Dict[str, str] = {
    "CLOSED": "#16A34A",
    "PENDING": "#D97706",
    "REOPENED": "#DC2626",
}
_BENCHMARK_COLOR_MAP: Dict[str, str] = {
    "excellent": "#16A34A",
    "good": "#2563EB",
    "fair": "#D97706",
    "critical": "#DC2626",
    "warning": "#D97706",
    "na": "#64748B",
}
_SPARKLINE_COLOR_MAP: Dict[str, str] = {
    "up_good": "#16A34A",
    "up_bad": "#DC2626",
    "down_good": "#16A34A",
    "down_bad": "#DC2626",
    "neutral": "#64748B",
    "none": "#64748B",
}
_VALID_AGGREGATIONS: Tuple[str, ...] = (
    "count", "sum", "mean", "median", "min", "max", "nunique"
)

# ── Private helpers ─────────────────────────────────────────────────────────────

def _get_theme(theme_key: str) -> Theme:
    """
    Resolves the active Theme dict. theme_key is the explicit per-call
    argument every render_* function already threads through from its
    own kwargs — this function's signature is unchanged. Internally, it
    now also cross-checks st.session_state["active_theme"] as an
    additive fallback (bridging any caller that reads the newer
    session-state key name instead of "theme"), without altering the
    resolution order any existing caller already relies on.
    """
    resolved_key = theme_key
    try:
        import streamlit as st  # local import — avoids a hard module-level
                                  # dependency change to this file's imports
        if resolved_key not in THEMES:
            resolved_key = st.session_state.get("active_theme", resolved_key)
    except Exception:  # noqa: BLE001 — never let theme resolution crash a render
        pass
    return THEMES.get(resolved_key, THEMES[DEFAULT_THEME_KEY])



def _color_seq(n: int, theme_key: str = DEFAULT_THEME_KEY) -> List[str]:
    palette = get_categorical_palette(theme_key)
    if n <= 0:
        return palette[:1]
    base = palette * (math.ceil(n / len(palette)))
    return base[:n]



def _resolve_col(
    role_or_col: Optional[str],
    df: pd.DataFrame,
    registry: ColumnRegistry,
) -> Optional[str]:
    if not role_or_col:
        return None
    resolved = registry.resolve(role_or_col)
    if resolved and resolved in df.columns:
        return resolved
    if role_or_col in df.columns:
        return role_or_col
    return None


def _col_label(role_or_col: Optional[str], registry: ColumnRegistry) -> str:
    if not role_or_col:
        return ""
    return registry.display_name(role_or_col)


def _tz_naive_series(series: pd.Series) -> pd.Series:
    """
    Refactor Phase 3 / Audit Finding B3 remediation — normalizes an
    already-parsed datetime series to tz-naive, mirroring
    engine.analytics._safe_tz_naive's exact tz-stripping semantics
    locally (this module is intentionally kept free of a hard import
    dependency on engine.analytics to avoid a circular import). Applied
    unconditionally wherever a parsed date column is about to be compared
    against, subtracted from, or assigned alongside another datetime
    series/scalar, so a tz-aware Parquet-origin column can never raise
    "Cannot subtract tz-naive and tz-aware datetime-like objects" or
    silently upcast a sibling series to object dtype. Never raises.
    """
    try:
        if getattr(series.dt, "tz", None) is not None:
            return series.dt.tz_localize(None)
    except (TypeError, AttributeError):  # noqa: BLE001
        pass
    return series


def _tz_naive_scalar(ts: Optional[Any]) -> Optional[Any]:
    """
    Refactor Phase 3 / Audit Finding B3 remediation — normalizes a single
    externally-supplied timestamp/date bound (e.g. a UI `date_input`-
    derived `pd.Timestamp`, or an explicit `reference_date` override) to
    tz-naive before it is ever compared against a `_tz_naive_series`-
    normalized column. Never raises; a `None` input passes through
    unchanged, and any coercion failure returns the original value rather
    than propagating an exception.
    """
    if ts is None:
        return None
    try:
        pts = pd.Timestamp(ts)
    except Exception:  # noqa: BLE001
        return ts
    try:
        if getattr(pts, "tzinfo", None) is not None:
            pts = pts.tz_localize(None)
    except (TypeError, AttributeError):  # noqa: BLE001
        pass
    return pts

def _ensure_numpy_backed_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Refactor Phase 4 / Force-DuckDB mandate — Plotly/UI compatibility guard.
    Converts any pandas ArrowDtype-backed column (produced by
    engine.duckdb_executor._fetch_df_arrow_backed, now reached on every
    aggregation regardless of row count) back to its conventional
    numpy/object-backed pandas dtype, since Plotly Express does not
    uniformly support ArrowDtype extension arrays across all chart
    primitives and pandas/plotly version combinations — unlike Streamlit's
    own Arrow-native data-grid transport, which handles ArrowDtype natively.
    Applied once, immediately after _aggregate() resolves a DuckDB-sourced
    result, so every render_* function downstream receives the exact same
    dtype shape it would have received from the pandas-only aggregation
    path. A no-op for any dataframe already fully numpy-backed. Never
    raises: any per-column conversion failure leaves that column exactly
    as returned by DuckDB rather than aborting the render.
    """
    if df is None or df.empty:
        return df
    for col in df.columns:
        try:
            dtype = df[col].dtype
            pyarrow_dtype = getattr(dtype, "pyarrow_dtype", None)
            if pyarrow_dtype is None:
                continue
            if pd.api.types.is_integer_dtype(dtype):
                df[col] = df[col].astype("int64")
            elif pd.api.types.is_float_dtype(dtype):
                df[col] = df[col].astype("float64")
            elif pd.api.types.is_bool_dtype(dtype):
                df[col] = df[col].astype(object)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                converted = pd.to_datetime(df[col].astype(object), errors="coerce")
                if getattr(converted.dt, "tz", None) is not None:
                    converted = converted.dt.tz_localize(None)
                df[col] = converted
            else:
                mask = df[col].notna()
                df[col] = df[col].astype(object).where(mask, np.nan)
        except Exception:  # noqa: BLE001
            continue
    return df


def _build_metadata(
    required_roles: List[str],
    optional_roles: List[str],
    supported_domains: List[str],
    title: str,
    status: str = "Eligible",
    reason: str = "",
) -> Dict[str, Any]:
    return {
        "required_roles": required_roles,
        "optional_roles": optional_roles,
        "supported_domains": supported_domains,
        "title": title,
        "status": status,
        "reason": reason,
    }


def _ineligible(
    required_roles: List[str],
    optional_roles: List[str],
    supported_domains: List[str],
    chart_name: str,
    reason: str,
) -> ChartReturn:
    return (
        None,
        None,
        _build_metadata(
            required_roles, optional_roles, supported_domains,
            chart_name, "Ineligible", reason,
        ),
    )


def _apply_date_range(
    df: pd.DataFrame,
    date_col: Optional[str],
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    if not date_col or date_range is None or date_col not in df.columns:
        return df
    start, end = date_range
    # Refactor Phase 3 / Audit Finding B3 remediation — both the parsed
    # source series and the incoming start/end bounds are normalized to
    # tz-naive before the comparison mask is built. This is the highest-
    # blast-radius fix in this module: _apply_date_range is the first
    # data-touching operation in essentially every render_* function, so
    # a tz mismatch here previously disabled every chart on the dashboard
    # with a generic "Render error" the instant a date filter was applied
    # against a tz-aware column.
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        dt = df[date_col]
    else:
        try:
            normalized = df[date_col].astype(str).str.strip().str.replace("_", "-", regex=False)
            try:
                dt = pd.to_datetime(normalized, errors="coerce", format="mixed")
            except (ValueError, TypeError):
                dt = pd.to_datetime(normalized, errors="coerce")
        except Exception:  # noqa: BLE001
            dt = pd.to_datetime(df[date_col], errors="coerce")
    dt = _tz_naive_series(dt)
    start = _tz_naive_scalar(start)
    end = _tz_naive_scalar(end)
    mask = pd.Series([True] * len(df), index=df.index)
    if start is not None:
        mask &= dt >= start
    if end is not None:
        mask &= dt <= end
    return df[mask].copy()


def _aggregate(
    df: pd.DataFrame,
    x_col: str,
    y_col: Optional[str],
    aggregation: str,
    group_col: Optional[str],
    facet_col: Optional[str],
    parquet_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, str]:
    agg = aggregation if aggregation in _VALID_AGGREGATIONS else "count"
    group_keys: List[str] = [x_col]
    if group_col and group_col != x_col:
        group_keys.append(group_col)
    if facet_col and facet_col not in group_keys:
        group_keys.append(facet_col)
    group_keys = [k for k in group_keys if k in df.columns]

    # ── Milestone 6 / Issue 17 remediation — Execution Substrate Switch ──
    # Above DUCKDB_ROW_THRESHOLD rows, attempt the out-of-core DuckDB
    # aggregation path first via engine.duckdb_executor. This is a pure
    # execution-strategy branch: duckdb_group_aggregate returns the exact
    # same (group columns + value column) shape the pandas path below
    # already produces, so every downstream caller of _aggregate (every
    # chart_factory.py render_* function) is completely unaffected by
    # which engine actually computed the result. Any DuckDB unavailability
    # or execution failure returns None and falls through unconditionally
    # to the untouched, original pandas implementation below — this is
    # strictly additive and never changes behavior for datasets at or
    # below the threshold, or in environments without the optional
    # `duckdb` package installed.
    #
    # ── Refactor Phase 3 / Audit Findings C2-C3 — Parquet-native routing ──
    # When `parquet_path` is supplied AND `should_use_duckdb(df)` passes,
    # the Parquet-native aggregation is attempted FIRST, executing
    # entirely against the on-disk Parquet file via DuckDB's native
    # `read_parquet()` scan with ZERO pandas materialization of the raw
    # rows for this computation. Column-name/type validation on that path
    # is performed against the Parquet file's own schema (see
    # engine.duckdb_executor.duckdb_group_aggregate_from_parquet), so any
    # mismatch (missing column, nested Arrow type) safely returns None and
    # this function falls through unconditionally to the existing
    # in-memory DuckDB replacement-scan branch immediately below, and
    # ultimately to the original, untouched pandas implementation.
    # `parquet_path` can therefore only ever change WHICH engine computed
    # an already-equivalent aggregated result — never the result itself,
    # and never a caller's behavior when the argument is omitted (it
    # defaults to None, so every pre-Phase-3 call site is unaffected).
    # Refactor Phase 4: should_use_duckdb no longer gates on row count —
    # this branch is now taken for every non-empty group_keys request.
    if group_keys and should_use_duckdb(df):
        value_col_for_duckdb = None if (y_col is None or agg == "count") else y_col
        if parquet_path:
            duckdb_parquet_result = duckdb_group_aggregate_from_parquet(
                parquet_path, group_keys, value_col_for_duckdb, agg
            )
            if duckdb_parquet_result is not None:
                duckdb_value_label = "Count" if (y_col is None or agg == "count") else y_col
                return _ensure_numpy_backed_df(duckdb_parquet_result), duckdb_value_label
        duckdb_result = duckdb_group_aggregate(df, group_keys, value_col_for_duckdb, agg)
        if duckdb_result is not None:
            duckdb_value_label = "Count" if (y_col is None or agg == "count") else y_col
            return _ensure_numpy_backed_df(duckdb_result), duckdb_value_label

    value_label: str
    if y_col is None or agg == "count":
        result = (
            df.groupby(group_keys, dropna=True)
            .size()
            .reset_index(name="Count")
        )
        value_label = "Count"
    else:
        try:
            numeric_y = pd.to_numeric(df[y_col], errors="coerce")
            tmp = df[group_keys].copy()
            tmp["__y"] = numeric_y
            agg_fn = agg if agg != "nunique" else pd.Series.nunique
            result = (
                tmp.groupby(group_keys, dropna=True)["__y"]
                .agg(agg)
                .reset_index()
            )
            result.rename(columns={"__y": y_col}, inplace=True)
            value_label = y_col
        except Exception:
            result = (
                df.groupby(group_keys, dropna=True)
                .size()
                .reset_index(name="Count")
            )
            value_label = "Count"
    return result, value_label


def _apply_top_n_sort(
    df: pd.DataFrame,
    value_col: str,
    top_n: int,
    sort_order: str,
) -> pd.DataFrame:
    if value_col not in df.columns:
        return df
    if sort_order == "desc":
        df = df.sort_values(value_col, ascending=False)
    elif sort_order == "asc":
        df = df.sort_values(value_col, ascending=True)
    if top_n > 0:
        df = df.head(top_n)
    return df.reset_index(drop=True)


# ADDITIVE — shared dynamic top-margin estimator
_TITLE_MARGIN_BASE_PX: int = 64
_TITLE_MARGIN_PER_WRAP_PX: int = 18
_TITLE_WRAP_CHAR_ESTIMATE: int = 70  # approx chars per line at 14px bold, default figure width


def _dynamic_top_margin(title: str, base: int = _TITLE_MARGIN_BASE_PX) -> int:
    """Estimates title wrap lines from character count and reserves
    proportional top margin, so long auto-generated titles never bleed
    into the plot area. Never raises; degrades to `base` on any failure."""
    try:
        clean_len = len(title or "")
        est_lines = max(1, math.ceil(clean_len / _TITLE_WRAP_CHAR_ESTIMATE))
        return base + (est_lines - 1) * _TITLE_MARGIN_PER_WRAP_PX
    except Exception:  # noqa: BLE001
        return base

def _apply_layout(
    fig: go.Figure,
    title: str,
    theme: Theme,
    height: int = _DEFAULT_HEIGHT,
    showlegend: bool = True,
    margin: Optional[Dict[str, int]] = None,
    xaxis_title: str = "",
    yaxis_title: str = "",
) -> go.Figure:
    m = margin or {"l": 20, "r": 20, "t": _dynamic_top_margin(title, base=72), "b": 30}
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=14, color=theme["text"], family=_FONT_FAMILY),
            x=0.0,
            xanchor="left",
            y=0.98,
            yanchor="top",
            pad={"l": 0, "t": 6, "b": 10},
            automargin=True,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=_FONT_FAMILY, color=theme["text"], size=12),
        height=height,
        showlegend=showlegend,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            font=dict(size=11, family=_FONT_FAMILY),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
        margin=m,
        hoverlabel=dict(
            bgcolor=theme["surface"],
            font_size=12,
            font_family=_FONT_FAMILY,
            bordercolor=theme["secondary"],
        ),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )
    fig.update_xaxes(
        gridcolor="rgba(100,116,139,0.12)",
        linecolor="rgba(100,116,139,0.25)",
        tickfont=dict(size=11, color=theme["secondary"], family=_FONT_FAMILY),
        title_font=dict(size=12, color=theme["secondary"]),
        zeroline=False,
    )
    fig.update_yaxes(
        gridcolor="rgba(100,116,139,0.12)",
        linecolor="rgba(100,116,139,0.25)",
        tickfont=dict(size=11, color=theme["secondary"], family=_FONT_FAMILY),
        title_font=dict(size=12, color=theme["secondary"]),
        zeroline=False,
    )

    # ── Final forced high-contrast override pass (Milestone 16) ─────────
    try:
        fig.update_layout(
            legend=dict(font=dict(color=theme["text"])),
            font=dict(color=theme["text"]),
        )
        if hasattr(fig, "layout") and getattr(fig.layout, "annotations", None):
            for ann in fig.layout.annotations:
                if ann.font is None or ann.font.color is None:
                    ann.update(font=dict(color=theme["text"]))
    except Exception:  # noqa: BLE001 — never let a cosmetic override crash a render
        pass

    return fig


def _auto_title(
    chart_type: str,
    x_label: str,
    y_label: str,
    aggregation: str,
    group_label: str = "",
    facet_label: str = "",  # Added facet support to handle subplots cleanly
) -> str:
    # 1. Standardize text components
    x_lbl = x_label.strip()
    y_lbl = y_label.strip() if y_label else ""
    grp_lbl = group_label.strip() if group_label else ""
    fct_lbl = facet_label.strip() if facet_label else ""

    # 2. Map Aggregations intelligently
    agg_map = {
        "count": "Volume",
        "sum": "Total",
        "mean": "Average",
        "median": "Median",
        "min": "Minimum",
        "max": "Maximum",
        "nunique": "Unique Count of",
    }
    agg_prefix = agg_map.get(aggregation.lower(), aggregation.title())

    # 3. Formulate the core Metric Title (Preventing "Volume Count")
    if aggregation.lower() == "count" or not y_lbl or y_lbl.lower() == "count":
        # If the metric is a raw count, name it "Volume" or "Record Count"
        y_part = "Volume" if y_lbl.lower() != "count" else "Record Count"
    else:
        # Avoid double-naming if y_lbl already starts with the prefix phrase
        if y_lbl.lower().startswith(agg_prefix.lower()):
            y_part = y_lbl
        else:
            y_part = f"{agg_prefix} {y_lbl}"

    # 4. Generate core structural chart titles
    c_type = chart_type.lower()
    if c_type in ("bar", "bar_horizontal", "bar_grouped", "bar_stacked", "vertical bar chart"):
        title = f"{y_part} by {x_lbl}"
    elif c_type == "line":
        title = f"{y_part} Trend over {x_lbl}"
    elif c_type == "area":
        title = f"{y_part} Distribution over {x_lbl}"
    elif c_type == "scatter":
        title = f"{x_lbl} vs {y_lbl}"
    elif c_type == "bubble":
        title = f"{x_lbl} vs {y_lbl}"
    elif c_type in ("pie", "donut"):
        title = f"{y_part} Breakdown by {x_lbl}"
    elif c_type == "histogram":
        title = f"Distribution of {x_lbl}"
    elif c_type == "box":
        title = f"Distribution of {y_lbl}" + (f" by {x_lbl}" if x_lbl else "")
    else:
        title = f"{y_part} by {x_lbl}"

    # 5. Handle Group-By Modifiers dynamically (Prevents duplicate structural text)
    if grp_lbl and grp_lbl.lower() != x_lbl.lower():
        if c_type == "bubble":
            title += f" (Size: {grp_lbl})"
        elif "stacked" in c_type or "grouped" in c_type:
            title += f" (Segmented by {grp_lbl})"
        else:
            title += f" by {grp_lbl}"

    # 6. Handle Facets (Subplots)
    if fct_lbl and fct_lbl.lower() != x_lbl.lower() and fct_lbl.lower() != grp_lbl.lower():
        title += f" [Split by {fct_lbl}]"

    return title



def _rename_agg_cols(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    value_col: str,
    value_label: str,
    group_col: Optional[str] = None,
    group_label: Optional[str] = None,
    facet_col: Optional[str] = None,
    facet_label: Optional[str] = None,
) -> Tuple[pd.DataFrame, str, str, Optional[str], Optional[str]]:
    rename_map: Dict[str, str] = {}
    new_x, new_val = x_col, value_col
    new_grp: Optional[str] = group_col
    new_fct: Optional[str] = facet_col
    if x_col != x_label:
        rename_map[x_col] = x_label
        new_x = x_label
    if value_col != value_label and value_col in df.columns:
        rename_map[value_col] = value_label
        new_val = value_label
    if group_col and group_label and group_col != group_label and group_col in df.columns:
        rename_map[group_col] = group_label
        new_grp = group_label
    if facet_col and facet_label and facet_col != facet_label and facet_col in df.columns:
        rename_map[facet_col] = facet_label
        new_fct = facet_label
    if rename_map:
        df = df.rename(columns=rename_map)
    return df, new_x, new_val, new_grp, new_fct


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC RENDER FUNCTIONS — Part 1: Standard Metric Visuals
# ══════════════════════════════════════════════════════════════════════════════

def render_bar(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    **kwargs: Any,
) -> ChartReturn:
    _REQUIRED: List[str] = []
    _OPTIONAL: List[str] = [ROLE_CATEGORY, ROLE_RECORD_ID, ROLE_AMOUNT, ROLE_STATUS]
    _DOMAINS: List[str] = ["all"]
    _CHART = "Vertical Bar Chart"

    x_role: Optional[str] = kwargs.get("x_role") or ROLE_CATEGORY
    y_role: Optional[str] = kwargs.get("y_role")
    group_by: Optional[str] = kwargs.get("group_by") or kwargs.get("color_role")
    facet_role: Optional[str] = kwargs.get("facet_role")
    aggregation: str = str(kwargs.get("aggregation", "count"))
    top_n: int = int(kwargs.get("top_n", _DEFAULT_TOP_N))
    sort_order: str = str(kwargs.get("sort_order", "desc"))
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = kwargs.get("date_range")
    theme_key: str = str(kwargs.get("theme_key", DEFAULT_THEME_KEY))
    height: int = int(kwargs.get("height", _DEFAULT_HEIGHT))
    show_values: bool = bool(kwargs.get("show_values", False))
    date_role: Optional[str] = kwargs.get("date_role") or ROLE_REGISTRATION_DATE

    try:
        theme = _get_theme(theme_key)
        date_col = _resolve_col(date_role, df, registry)
        df = _apply_date_range(df, date_col, date_range)
        if df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "No records remain after applying date filter.")

        x_col = _resolve_col(x_role, df, registry)
        if not x_col:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                               f"x_role '{x_role}' could not be resolved to a column.")

        y_col = _resolve_col(y_role, df, registry) if y_role else None
        group_col = _resolve_col(group_by, df, registry) if group_by else None
        facet_col = _resolve_col(facet_role, df, registry) if facet_role else None

        x_label = _col_label(x_role, registry)
        y_label = (_col_label(y_role, registry) if y_role else "Count")
        group_label = _col_label(group_by, registry) if group_by else ""
       

        agg_df, value_col = _aggregate(df, x_col, y_col, aggregation, group_col, facet_col)
        agg_df = _apply_top_n_sort(agg_df, value_col, top_n, sort_order)
        if agg_df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "Aggregation produced an empty result.")

        #title: str = str(kwargs.get("title") or _auto_title("bar", x_label, y_label, aggregation, group_label))
        
        # --- AFTER (EXACT REPLACEMENT) ---
        title: str = str( kwargs.get("title") or _auto_title(chart_type="bar", 
        x_label=x_label, 
        y_label=y_label, 
        aggregation=aggregation, 
        group_label=group_label or ""
    )
)

        export_df, ex, ev, eg, ef = _rename_agg_cols(
            agg_df.copy(), x_col, x_label, value_col,
            f"{aggregation.title()} {y_label}" if y_label != "Count" else "Count",
            group_col, group_label or None, facet_col, 
        )

        color_seq = _color_seq(agg_df[group_col].nunique() if group_col else 1)

        fig = px.bar(
            agg_df,
            x=x_col,
            y=value_col,
            color=group_col,
            facet_col=facet_col,
            color_discrete_sequence=color_seq,
            template=theme["plotly_template"],
        )
        if show_values:
            fig.update_traces(
                texttemplate="%{y:,.0f}",
                textposition="outside",
                textfont=dict(size=10, family=_FONT_FAMILY),
            )
        fig.update_traces(marker_line_width=0)
        _apply_layout(fig, "" , theme, height=height, xaxis_title=x_label, yaxis_title=y_label)

        meta = _build_metadata(_REQUIRED, _OPTIONAL, _DOMAINS, title)
        return fig, export_df, meta

    except Exception as exc:
        return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, f"Render error: {exc}")



def render_bar_horizontal(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    **kwargs: Any,
) -> ChartReturn:
    _REQUIRED: List[str] = []
    _OPTIONAL: List[str] = [ROLE_CATEGORY, ROLE_RECORD_ID, ROLE_AMOUNT, ROLE_OFFICER]
    _DOMAINS: List[str] = ["all"]
    _CHART = "Horizontal Bar Chart"

    x_role: Optional[str] = kwargs.get("x_role") or ROLE_CATEGORY
    y_role: Optional[str] = kwargs.get("y_role")
    group_by: Optional[str] = kwargs.get("group_by") or kwargs.get("color_role")
    aggregation: str = str(kwargs.get("aggregation", "count"))
    facet_role: Optional[str] = kwargs.get("facet_role")
    top_n: int = int(kwargs.get("top_n", _DEFAULT_TOP_N))
    sort_order: str = str(kwargs.get("sort_order", "desc"))
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = kwargs.get("date_range")
    theme_key: str = str(kwargs.get("theme_key", DEFAULT_THEME_KEY))
    height: int = int(kwargs.get("height", _DEFAULT_HEIGHT))
    show_values: bool = bool(kwargs.get("show_values", True))
    date_role: Optional[str] = kwargs.get("date_role") or ROLE_REGISTRATION_DATE


    try:
        theme = _get_theme(theme_key)
        date_col = _resolve_col(date_role, df, registry)
        df = _apply_date_range(df, date_col, date_range)
        if df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "No records after date filter.")

        x_col = _resolve_col(x_role, df, registry)
        if not x_col:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                               f"x_role '{x_role}' could not be resolved.")

        y_col = _resolve_col(y_role, df, registry) if y_role else None
        group_col = _resolve_col(group_by, df, registry) if group_by else None

        x_label = _col_label(x_role, registry)
        y_label = _col_label(y_role, registry) if y_role else "Count"
        group_label = _col_label(group_by, registry) if group_by else ""
        facet_label = _col_label(facet_role, registry) if facet_role else ""

        agg_df, value_col = _aggregate(df, x_col, y_col, aggregation, group_col, None)
        agg_df = _apply_top_n_sort(agg_df, value_col, top_n, sort_order)
        if agg_df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "Empty aggregation.")
        truncated_for_render_safety = len(agg_df) > _MAX_HORIZONTAL_BAR_CATEGORIES
        if truncated_for_render_safety:
            agg_df = agg_df.sort_values(value_col, ascending=False).head(_MAX_HORIZONTAL_BAR_CATEGORIES)
        n_cats = agg_df[x_col].nunique()
        dynamic_height = max(height, 36 * n_cats + 80)
        #title: str = str(kwargs.get("title") or _auto_title("bar_horizontal", x_label, y_label, aggregation))
        title: str = str( kwargs.get("title") or _auto_title(chart_type="bar_horizontal", 
        x_label=x_label, 
        y_label=y_label, 
        aggregation=aggregation, 
        group_label=group_label or "",
        facet_label=facet_label or "",
        
           )
)
        color_seq = _color_seq(agg_df[group_col].nunique() if group_col else 1)

        fig = px.bar(
            agg_df,
            x=value_col,
            y=x_col,
            color=group_col,
            orientation="h",
            color_discrete_sequence=color_seq,
            template=theme["plotly_template"],
        )
        if show_values:
            fig.update_traces(
                texttemplate="%{x:,.0f}",
                textposition="outside",
                textfont=dict(size=10, family=_FONT_FAMILY),
            )
        fig.update_traces(marker_line_width=0)
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        _apply_layout(
            fig, "" , theme, height=dynamic_height,
            xaxis_title=y_label, yaxis_title=x_label,
        )

        export_df, *_ = _rename_agg_cols(
            agg_df.copy(), x_col, x_label, value_col,
            f"{aggregation.title()} {y_label}" if y_label != "Count" else "Count",
            group_col, group_label or None,
        )
        safety_reason = (
            f"Display capped at the top {_MAX_HORIZONTAL_BAR_CATEGORIES:,} of {x_label} categories "
            f"by volume to protect rendering performance; set a smaller explicit top_n to narrow "
            f"further, or increase the cap if the full breakdown is required."
            if truncated_for_render_safety else ""
        )
        meta = _build_metadata(_REQUIRED, _OPTIONAL, _DOMAINS, title, reason=safety_reason)
        return fig, export_df, meta

    except Exception as exc:
        return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, f"Render error: {exc}")


def render_bar_grouped(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    **kwargs: Any,
) -> ChartReturn:
    _REQUIRED: List[str] = []
    _OPTIONAL: List[str] = [ROLE_CATEGORY, ROLE_STATUS, ROLE_ZONE, ROLE_OFFICER]
    _DOMAINS: List[str] = ["all"]
    _CHART = "Grouped Bar Chart"

    x_role: Optional[str] = kwargs.get("x_role") or ROLE_CATEGORY
    y_role: Optional[str] = kwargs.get("y_role")
    group_by: Optional[str] = kwargs.get("group_by") or kwargs.get("color_role") or ROLE_STATUS
    aggregation: str = str(kwargs.get("aggregation", "count"))
    top_n: int = int(kwargs.get("top_n", _DEFAULT_TOP_N))
    sort_order: str = str(kwargs.get("sort_order", "desc"))
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = kwargs.get("date_range")
    theme_key: str = str(kwargs.get("theme_key", DEFAULT_THEME_KEY))
    height: int = int(kwargs.get("height", _DEFAULT_HEIGHT))
    date_role: Optional[str] = kwargs.get("date_role") or ROLE_REGISTRATION_DATE

    try:
        theme = _get_theme(theme_key)
        date_col = _resolve_col(date_role, df, registry)
        df = _apply_date_range(df, date_col, date_range)
        if df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "No records after date filter.")

        x_col = _resolve_col(x_role, df, registry)
        if not x_col:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                               f"x_role '{x_role}' could not be resolved.")

        group_col = _resolve_col(group_by, df, registry) if group_by else None
        if not group_col:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                               "Grouped bar requires a group_by role. "
                               "Map a categorical role (e.g. Status, Zone) to group_by.")

        y_col = _resolve_col(y_role, df, registry) if y_role else None
        x_label = _col_label(x_role, registry)
        y_label = _col_label(y_role, registry) if y_role else "Count"
        group_label = _col_label(group_by, registry)

        agg_df, value_col = _aggregate(df, x_col, y_col, aggregation, group_col, None)

        total_per_x = agg_df.groupby(x_col)[value_col].sum().nlargest(top_n if top_n > 0 else len(agg_df))
        if top_n > 0:
            agg_df = agg_df[agg_df[x_col].isin(total_per_x.index)]

        if agg_df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "Empty aggregation.")

        # ── Professional, executive-readable title (replaces the terse
        # "X by Y and Z" auto-title with a cleaner grouped-comparison
        # phrasing) — only applied when the caller hasn't supplied an
        # explicit title override via kwargs["title"].
        agg_label_map: Dict[str, str] = {
            "count": "Volume", "sum": "Total", "mean": "Average", "median": "Median",
            "min": "Minimum", "max": "Maximum", "nunique": "Unique Count of",
        }
        measure_phrase = (
            f"{agg_label_map.get(aggregation, 'Value of')} {y_label}"
            if y_label != "Count" else agg_label_map.get(aggregation, "Volume")
        )
        default_title = f"{measure_phrase} by {x_label}, Grouped by {group_label}"
        title: str = str(kwargs.get("title") or default_title)

        n_groups = agg_df[group_col].nunique()
        color_seq = _color_seq(n_groups)

        fig = px.bar(
            agg_df,
            x=x_col,
            y=value_col,
            color=group_col,
            barmode="group",
            color_discrete_sequence=color_seq,
            template=theme["plotly_template"],
        )
        fig.update_traces(marker_line_width=0)

        _apply_layout(fig, "" , theme, height=height, xaxis_title=x_label, yaxis_title=y_label)

        export_df, *_ = _rename_agg_cols(
            agg_df.copy(), x_col, x_label, value_col,
            "Count" if y_label == "Count" else f"{aggregation.title()} {y_label}",
            group_col, group_label,
        )
        meta = _build_metadata(_REQUIRED, _OPTIONAL, _DOMAINS, title)
        return fig, export_df, meta

    except Exception as exc:
        return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, f"Render error: {exc}")
    
def render_bar_stacked(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    **kwargs: Any,
) -> ChartReturn:
    _REQUIRED: List[str] = []
    _OPTIONAL: List[str] = [ROLE_CATEGORY, ROLE_STATUS, ROLE_ZONE]
    _DOMAINS: List[str] = ["all"]
    _CHART = "Stacked Bar Chart"

    x_role: Optional[str] = kwargs.get("x_role") or ROLE_ZONE
    y_role: Optional[str] = kwargs.get("y_role")
    group_by: Optional[str] = kwargs.get("group_by") or kwargs.get("color_role") or ROLE_STATUS
    aggregation: str = str(kwargs.get("aggregation", "count"))
    facet_role: Optional[str] = kwargs.get("facet_role")
    top_n: int = int(kwargs.get("top_n", _DEFAULT_TOP_N))
    sort_order: str = str(kwargs.get("sort_order", "desc"))
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = kwargs.get("date_range")
    theme_key: str = str(kwargs.get("theme_key", DEFAULT_THEME_KEY))
    height: int = int(kwargs.get("height", _DEFAULT_HEIGHT))
    normalize: bool = bool(kwargs.get("normalize", False))
    date_role: Optional[str] = kwargs.get("date_role") or ROLE_REGISTRATION_DATE

    try:
        theme = _get_theme(theme_key)
        date_col = _resolve_col(date_role, df, registry)
        df = _apply_date_range(df, date_col, date_range)
        if df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "No records after date filter.")

        x_col = _resolve_col(x_role, df, registry)
        if not x_col:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                               f"x_role '{x_role}' could not be resolved.")

        group_col = _resolve_col(group_by, df, registry) if group_by else None
        if not group_col:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                               "Stacked bar requires a group_by role for the stack dimension.")

        y_col = _resolve_col(y_role, df, registry) if y_role else None
        x_label = _col_label(x_role, registry)
        y_label = _col_label(y_role, registry) if y_role else "Count"
        group_label = _col_label(group_by, registry)
        facet_label = _col_label(facet_role, registry) if facet_role else ""
        agg_df, value_col = _aggregate(df, x_col, y_col, aggregation, group_col, None)

        total_per_x = agg_df.groupby(x_col)[value_col].sum().nlargest(top_n if top_n > 0 else len(agg_df))
        if top_n > 0:
            agg_df = agg_df[agg_df[x_col].isin(total_per_x.index)]

        if normalize:
            agg_df = agg_df.copy()
            totals = agg_df.groupby(x_col)[value_col].transform("sum")
            agg_df[value_col] = (agg_df[value_col] / totals.replace(0, np.nan) * 100).round(2)
            y_label = f"{y_label} (%)"

        if agg_df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "Empty aggregation.")

        #title: str = str(kwargs.get("title") or _auto_title("bar_stacked", x_label, y_label, aggregation, group_label))
        title: str = str( kwargs.get("title") or _auto_title(chart_type="bar_stacked", 
        x_label=x_label, 
        y_label=y_label, 
        aggregation=aggregation, 
        group_label=group_label or "",
        facet_label=facet_label or "",
        
    )
)
        n_groups = agg_df[group_col].nunique()
        color_seq = _color_seq(n_groups)

        fig = px.bar(
            agg_df,
            x=x_col,
            y=value_col,
            color=group_col,
            barmode="stack",
            color_discrete_sequence=color_seq,
            template=theme["plotly_template"],
        )
        fig.update_traces(marker_line_width=0)
        _apply_layout(fig, "" , theme, height=height, xaxis_title=x_label, yaxis_title=y_label)

        export_df, *_ = _rename_agg_cols(
            agg_df.copy(), x_col, x_label, value_col,
            "Count" if y_label == "Count" else f"{aggregation.title()} {y_label}",
            group_col, group_label,
        )
        meta = _build_metadata(_REQUIRED, _OPTIONAL, _DOMAINS, title)
        return fig, export_df, meta

    except Exception as exc:
        return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, f"Render error: {exc}")




def render_line(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    **kwargs: Any,
) -> ChartReturn:
    _REQUIRED: List[str] = [ROLE_REGISTRATION_DATE]
    _OPTIONAL: List[str] = [ROLE_RECORD_ID, ROLE_AMOUNT, ROLE_STATUS]
    _DOMAINS: List[str] = ["all"]
    _CHART = "Line Chart"

    x_role: Optional[str] = kwargs.get("x_role") or ROLE_REGISTRATION_DATE
    y_role: Optional[str] = kwargs.get("y_role")
    group_by: Optional[str] = kwargs.get("group_by") or kwargs.get("color_role")
    aggregation: str = str(kwargs.get("aggregation", "count"))
    facet_role: Optional[str] = kwargs.get("facet_role")
    date_freq: str = str(kwargs.get("date_freq", "M"))
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = kwargs.get("date_range")
    theme_key: str = str(kwargs.get("theme_key", DEFAULT_THEME_KEY))
    height: int = int(kwargs.get("height", _DEFAULT_HEIGHT))
    smooth: bool = bool(kwargs.get("smooth", False))
    show_markers: bool = bool(kwargs.get("show_markers", True))

    try:
        theme = _get_theme(theme_key)
        x_col = _resolve_col(x_role, df, registry)
        if not x_col:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                               f"x_role '{x_role}' could not be resolved. "
                               "Map a date or ordered column to x_role.")

        df = _apply_date_range(df, x_col, date_range)
        if df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "No records after date filter.")

        x_is_date = pd.api.types.is_datetime64_any_dtype(df[x_col])
        if x_is_date:
            df = df.copy()
            df["__period"] = pd.to_datetime(df[x_col], errors="coerce").dt.to_period(date_freq).astype(str)
            effective_x_col = "__period"
        else:
            effective_x_col = x_col

        y_col = _resolve_col(y_role, df, registry) if y_role else None
        group_col = _resolve_col(group_by, df, registry) if group_by else None
        x_label = _col_label(x_role, registry)
        y_label = _col_label(y_role, registry) if y_role else "Count"
        group_label = _col_label(group_by, registry) if group_by else ""
        facet_label = _col_label(facet_role, registry) if facet_role else ""

        agg_df, value_col = _aggregate(df, effective_x_col, y_col, aggregation, group_col, None)
        if agg_df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "Empty aggregation.")

        #title: str = str(kwargs.get("title") or _auto_title("line", x_label, y_label, aggregation))
        title: str = str( kwargs.get("title") or _auto_title(chart_type="line", 
        x_label=x_label, 
        y_label=y_label, 
        aggregation=aggregation, 
        group_label=group_label or "",
        facet_label=facet_label or "",
    )
)
        n_lines = agg_df[group_col].nunique() if group_col else 1
        color_seq = _color_seq(n_lines)
        line_shape = "spline" if smooth else "linear"

        fig = px.line(
            agg_df,
            x=effective_x_col,
            y=value_col,
            color=group_col,
            line_shape=line_shape,
            markers=show_markers,
            color_discrete_sequence=color_seq,
            template=theme["plotly_template"],
        )
        fig.update_traces(
            line_width=2,
            marker_size=5,
        )
        _apply_layout(fig, "" , theme, height=height, xaxis_title=x_label, yaxis_title=y_label)
        fig.update_xaxes(tickangle=-30)

        export_df, *_ = _rename_agg_cols(
            agg_df.copy(), effective_x_col, x_label, value_col,
            "Count" if y_label == "Count" else f"{aggregation.title()} {y_label}",
            group_col, group_label or None,
        )
        meta = _build_metadata(_REQUIRED, _OPTIONAL, _DOMAINS, title)
        return fig, export_df, meta

    except Exception as exc:
        return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, f"Render error: {exc}")


def render_area(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    **kwargs: Any,
) -> ChartReturn:
    _REQUIRED: List[str] = [ROLE_REGISTRATION_DATE]
    _OPTIONAL: List[str] = [ROLE_RECORD_ID, ROLE_AMOUNT, ROLE_STATUS]
    _DOMAINS: List[str] = ["all"]
    _CHART = "Area Chart"

    x_role: Optional[str] = kwargs.get("x_role") or ROLE_REGISTRATION_DATE
    y_role: Optional[str] = kwargs.get("y_role")
    group_by: Optional[str] = kwargs.get("group_by") or kwargs.get("color_role")
    aggregation: str = str(kwargs.get("aggregation", "count"))
    facet_role: Optional[str] = kwargs.get("facet_role")
    date_freq: str = str(kwargs.get("date_freq", "M"))
    stacked: bool = bool(kwargs.get("stacked", False))
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = kwargs.get("date_range")
    theme_key: str = str(kwargs.get("theme_key", DEFAULT_THEME_KEY))
    height: int = int(kwargs.get("height", _DEFAULT_HEIGHT))

    try:
        theme = _get_theme(theme_key)
        x_col = _resolve_col(x_role, df, registry)
        if not x_col:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                               f"x_role '{x_role}' could not be resolved.")

        df = _apply_date_range(df, x_col, date_range)
        if df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "No records after date filter.")

        x_is_date = pd.api.types.is_datetime64_any_dtype(df[x_col])
        if x_is_date:
            df = df.copy()
            df["__period"] = pd.to_datetime(df[x_col], errors="coerce").dt.to_period(date_freq).astype(str)
            effective_x_col = "__period"
        else:
            effective_x_col = x_col

        y_col = _resolve_col(y_role, df, registry) if y_role else None
        group_col = _resolve_col(group_by, df, registry) if group_by else None
        x_label = _col_label(x_role, registry)
        y_label = _col_label(y_role, registry) if y_role else "Count"
        group_label = _col_label(group_by, registry) if group_by else ""
        facet_label = _col_label(facet_role, registry) if facet_role else ""

        agg_df, value_col = _aggregate(df, effective_x_col, y_col, aggregation, group_col, None)
        if agg_df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "Empty aggregation.")

        #title: str = str(kwargs.get("title") or _auto_title("area", x_label, y_label, aggregation))
        title: str = str( kwargs.get("title") or _auto_title(chart_type="area", 
        x_label=x_label, 
        y_label=y_label, 
        aggregation=aggregation, 
        group_label=group_label or "",
        facet_label=facet_label or "",
    )
)

        n_areas = agg_df[group_col].nunique() if group_col else 1
        color_seq = _color_seq(n_areas)

        groupnorm = "percent" if stacked and group_col else None

        fig = px.area(
            agg_df,
            x=effective_x_col,
            y=value_col,
            color=group_col,
            line_shape="spline",
            color_discrete_sequence=color_seq,
            groupnorm=groupnorm,
            template=theme["plotly_template"],
        )
        fig.update_traces(line_width=1.5)
        _apply_layout(fig, "" , theme, height=height, xaxis_title=x_label, yaxis_title=y_label)
        fig.update_xaxes(tickangle=-30)

        export_df, *_ = _rename_agg_cols(
            agg_df.copy(), effective_x_col, x_label, value_col,
            "Count" if y_label == "Count" else f"{aggregation.title()} {y_label}",
            group_col, group_label or None,
        )
        meta = _build_metadata(_REQUIRED, _OPTIONAL, _DOMAINS, title)
        return fig, export_df, meta

    except Exception as exc:
        return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, f"Render error: {exc}")


def render_scatter(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    **kwargs: Any,
) -> ChartReturn:
    _REQUIRED: List[str] = []
    _OPTIONAL: List[str] = [ROLE_CATEGORY, ROLE_AMOUNT, ROLE_STATUS, ROLE_OFFICER]
    _DOMAINS: List[str] = ["all"]
    _CHART = "Scatter Plot"

    x_role: Optional[str] = kwargs.get("x_role")
    y_role: Optional[str] = kwargs.get("y_role")
    color_role: Optional[str] = kwargs.get("color_role") or kwargs.get("group_by")
    facet_role: Optional[str] = kwargs.get("facet_role")
    group_by: Optional[str] = kwargs.get("group_by")
    aggregation: Optional[str] = kwargs.get("aggregation")
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = kwargs.get("date_range")
    theme_key: str = str(kwargs.get("theme_key", DEFAULT_THEME_KEY))
    height: int = int(kwargs.get("height", _DEFAULT_HEIGHT))
    date_role: Optional[str] = kwargs.get("date_role") or ROLE_REGISTRATION_DATE
    opacity: float = float(kwargs.get("opacity", 0.72))
    trendline: Optional[str] = kwargs.get("trendline")

    try:
        theme = _get_theme(theme_key)
        date_col = _resolve_col(date_role, df, registry)
        df = _apply_date_range(df, date_col, date_range)
        if df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "No records after date filter.")

        x_col = _resolve_col(x_role, df, registry) if x_role else None
        y_col = _resolve_col(y_role, df, registry) if y_role else None

        if not x_col or not y_col:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                               "Scatter requires both x_role and y_role to be mapped to numeric columns.")

        color_col = _resolve_col(color_role, df, registry) if color_role else None
        facet_col = _resolve_col(facet_role, df, registry) if facet_role else None
        group_col = _resolve_col(group_by, df, registry) if group_by else None
        x_label = _col_label(x_role, registry)
        y_label = _col_label(y_role, registry)
        #color_label = _col_label(color_role, registry) if color_role else ""

        if aggregation and aggregation in _VALID_AGGREGATIONS and group_col:
            agg_df, x_agg_col = _aggregate(df, group_col, x_col, aggregation, None, None)
            agg_df2, y_agg_col = _aggregate(df, group_col, y_col, aggregation, None, None)
            plot_df = agg_df.merge(agg_df2, on=group_col)
            plot_df = plot_df.dropna(subset=[x_agg_col, y_agg_col])
            px_x, px_y = x_agg_col, y_agg_col
            px_color = group_col
            export_df = plot_df.rename(columns={group_col: _col_label(group_by, registry),
                                                 x_agg_col: x_label,
                                                 y_agg_col: y_label})
        else:
            plot_df = df[[c for c in [x_col, y_col, color_col, facet_col] if c and c in df.columns]].copy()
            plot_df = plot_df.dropna(subset=[x_col, y_col])
            if len(plot_df) > _MAX_SCATTER_ROWS:
                plot_df = plot_df.sample(_MAX_SCATTER_ROWS, random_state=42)
            px_x, px_y = x_col, y_col
            px_color = color_col
            export_df = plot_df.rename(columns={x_col: x_label, y_col: y_label})

        if plot_df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "No valid numeric rows to plot.")

        #title: str = str(kwargs.get("title") or _auto_title("scatter", x_label, y_label, aggregation or "none"))
        title: str = str( kwargs.get("title") or _auto_title(chart_type="scatter", 
        x_label=x_label, 
        y_label=y_label, 
        aggregation=aggregation or "none", 
    ))

        n_colors = plot_df[px_color].nunique() if px_color else 1
        color_seq = _color_seq(n_colors)

        fig_kwargs: Dict[str, Any] = dict(
            data_frame=plot_df,
            x=px_x,
            y=px_y,
            color=px_color,
            facet_col=facet_col,
            opacity=opacity,
            color_discrete_sequence=color_seq,
            template=theme["plotly_template"],
        )
        if trendline in ("ols", "lowess", "expanding", "rolling"):
            fig_kwargs["trendline"] = trendline

        fig = px.scatter(**fig_kwargs)
        fig.update_traces(marker_size=7, marker_line_width=0.5,
                          marker_line_color="rgba(255,255,255,0.4)")
        _apply_layout(fig, "" , theme, height=height, xaxis_title=x_label, yaxis_title=y_label)

        meta = _build_metadata(_REQUIRED, _OPTIONAL, _DOMAINS, title)
        return fig, export_df, meta

    except Exception as exc:
        return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, f"Render error: {exc}")


def render_bubble(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    **kwargs: Any,
) -> ChartReturn:
    _REQUIRED: List[str] = []
    _OPTIONAL: List[str] = [ROLE_CATEGORY, ROLE_AMOUNT, ROLE_STATUS, ROLE_OFFICER, ROLE_RECORD_ID]
    _DOMAINS: List[str] = ["all"]
    _CHART = "Bubble Chart"

    x_role: Optional[str] = kwargs.get("x_role")
    y_role: Optional[str] = kwargs.get("y_role")
    size_role: Optional[str] = kwargs.get("size_role") or kwargs.get("group_by")
    color_role: Optional[str] = kwargs.get("color_role")
    group_by: Optional[str] = kwargs.get("group_by")
    aggregation: str = str(kwargs.get("aggregation", "count"))
    facet_role: Optional[str] = kwargs.get("facet_role")
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = kwargs.get("date_range")
    theme_key: str = str(kwargs.get("theme_key", DEFAULT_THEME_KEY))
    height: int = int(kwargs.get("height", _DEFAULT_HEIGHT))
    max_bubble_size: int = int(kwargs.get("max_bubble_size", 60))
    date_role: Optional[str] = kwargs.get("date_role") or ROLE_REGISTRATION_DATE

    try:
        theme = _get_theme(theme_key)
        date_col = _resolve_col(date_role, df, registry)
        df = _apply_date_range(df, date_col, date_range)
        if df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "No records after date filter.")

        x_col = _resolve_col(x_role, df, registry) if x_role else None
        y_col = _resolve_col(y_role, df, registry) if y_role else None
        size_col = _resolve_col(size_role, df, registry) if size_role else None
        color_col = _resolve_col(color_role, df, registry) if color_role else None
        group_col = _resolve_col(group_by, df, registry) if group_by else None
        #facet_col = _resolve_col(facet_role, registry) if facet_role else None

        if not x_col or not y_col:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                               "Bubble chart requires x_role and y_role. "
                               "For officer-level views, map officer/category to group_by and "
                               "resolution metrics to x_role and y_role.")

        x_label = _col_label(x_role, registry)
        y_label = _col_label(y_role, registry)
        size_label = _col_label(size_role, registry) if size_role else "Count"
        facet_label = _col_label(facet_role, registry) if facet_role else ""
        # color_label = _col_label(color_role, registry) if color_role else ""

        if group_col:
            x_agg, _ = _aggregate(df, group_col, x_col, aggregation, None, None)
            y_agg, _ = _aggregate(df, group_col, y_col, aggregation, None, None)
            plot_df = x_agg.merge(y_agg, on=group_col)
            if size_col:
                s_agg, _ = _aggregate(df, group_col, size_col, "sum", None, None)
                plot_df = plot_df.merge(s_agg, on=group_col)
            else:
                cnt_agg, _ = _aggregate(df, group_col, None, "count", None, None)
                plot_df = plot_df.merge(cnt_agg, on=group_col)
            cols_check = [c for c in plot_df.columns if c != group_col]
            plot_df = plot_df.dropna(subset=cols_check[:2])
            px_x = [c for c in plot_df.columns if c != group_col][0]
            px_y = [c for c in plot_df.columns if c != group_col][1]
            px_size = [c for c in plot_df.columns if c != group_col and c != px_x and c != px_y]
            px_size = px_size[0] if px_size else None
            px_color = group_col if not color_col else color_col
            px_text = group_col
        else:
            numeric_cols = [x_col, y_col] + ([size_col] if size_col else []) + ([color_col] if color_col else [])
            plot_df = df[[c for c in numeric_cols if c in df.columns]].copy()
            plot_df = plot_df.dropna(subset=[x_col, y_col])
            if len(plot_df) > _MAX_SCATTER_ROWS:
                plot_df = plot_df.sample(_MAX_SCATTER_ROWS, random_state=42)
            px_x, px_y = x_col, y_col
            px_size = size_col
            px_color = color_col
            px_text = None

        if plot_df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "No plottable rows after aggregation.")

        #title: str = str(kwargs.get("title") or _auto_title("bubble", x_label, y_label, aggregation, size_label))
        title: str = str( kwargs.get("title") or _auto_title(chart_type="bubble", 
        x_label=x_label, 
        y_label=y_label, 
        aggregation=aggregation or "none", 
        size_label=size_label or "",
        facet_label=facet_label or "",
    ))
        n_colors = plot_df[px_color].nunique() if px_color and px_color in plot_df.columns else 1
        color_seq = _color_seq(n_colors)

        fig = px.scatter(
            plot_df,
            x=px_x,
            y=px_y,
            size=px_size,
            color=px_color,
            text=px_text,
            size_max=max_bubble_size,
            color_discrete_sequence=color_seq,
            template=theme["plotly_template"],
        )
        fig.update_traces(
            marker_opacity=0.78,
            marker_line_width=1,
            marker_line_color="rgba(255,255,255,0.5)",
            textfont=dict(size=9, family=_FONT_FAMILY, color=theme["text"]),
            textposition="top center",
        )
        _apply_layout(fig, "" , theme, height=height, xaxis_title=x_label, yaxis_title=y_label)

        export_df = plot_df.copy()
        meta = _build_metadata(_REQUIRED, _OPTIONAL, _DOMAINS, title)
        return fig, export_df, meta

    except Exception as exc:
        return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, f"Render error: {exc}")


def render_pie(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    **kwargs: Any,
) -> ChartReturn:
    _REQUIRED: List[str] = []
    _OPTIONAL: List[str] = [ROLE_CATEGORY, ROLE_STATUS, ROLE_ZONE, ROLE_PRIORITY]
    _DOMAINS: List[str] = ["all"]
    _CHART = "Pie Chart"

    x_role: Optional[str] = kwargs.get("x_role") or ROLE_CATEGORY
    y_role: Optional[str] = kwargs.get("y_role")
    aggregation: str = str(kwargs.get("aggregation", "count"))
    top_n: int = int(kwargs.get("top_n", 10))
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = kwargs.get("date_range")
    theme_key: str = str(kwargs.get("theme_key", DEFAULT_THEME_KEY))
    height: int = int(kwargs.get("height", _DEFAULT_HEIGHT))
    date_role: Optional[str] = kwargs.get("date_role") or ROLE_REGISTRATION_DATE

    try:
        theme = _get_theme(theme_key)
        date_col = _resolve_col(date_role, df, registry)
        df = _apply_date_range(df, date_col, date_range)
        if df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "No records after date filter.")

        x_col = _resolve_col(x_role, df, registry)
        if not x_col:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                               f"x_role '{x_role}' could not be resolved for pie slice names.")

        y_col = _resolve_col(y_role, df, registry) if y_role else None
        x_label = _col_label(x_role, registry)
        y_label = _col_label(y_role, registry) if y_role else "Count"

        agg_df, value_col = _aggregate(df, x_col, y_col, aggregation, None, None)
        agg_df = _apply_top_n_sort(agg_df, value_col, top_n, "desc")

        if agg_df.empty or agg_df[value_col].sum() == 0:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "All aggregated values are zero or empty.")

        title: str = str(kwargs.get("title") or _auto_title("pie", x_label, y_label, aggregation))

        is_status_col = (x_col == registry.resolve(ROLE_STATUS))
        if is_status_col and agg_df[x_col].isin(_STATUS_COLOR_MAP).any():
            color_map = {k: _STATUS_COLOR_MAP.get(k, "#64748B") for k in agg_df[x_col].unique()}
            fig = px.pie(
                agg_df, names=x_col, values=value_col,
                color=x_col, color_discrete_map=color_map,
                template=theme["plotly_template"],
            )
        else:
            n_slices = len(agg_df)
            color_seq = _color_seq(n_slices)
            fig = px.pie(
                agg_df, names=x_col, values=value_col,
                color_discrete_sequence=color_seq,
                template=theme["plotly_template"],
            )

    #    title=dict(
    #             text=f"<b>{title}</b>",
    #             font=dict(size=14, color=theme["text"], family=_FONT_FAMILY),
    #             x=0.0, xanchor="left",
    #             y=0.97, yanchor="top",
    #             automargin=True,
    #         ),
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family=_FONT_FAMILY, color=theme["text"], size=12),
            height=height,
            title= "",
            margin={"l": 10, "r": 10, "t": _dynamic_top_margin(title, base=60), "b": 10},
            legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=11, color=theme["text"])),
            hoverlabel=dict(bgcolor=theme["surface"], font_size=12,
                            font_family=_FONT_FAMILY, bordercolor=theme["secondary"]),
        )
        fig.update_traces(
            textinfo="percent+label",
            textfont=dict(size=11, family=_FONT_FAMILY),
            pull=[0.03] + [0.0] * (len(agg_df) - 1),
            marker_line=dict(color=theme["surface"], width=2),
        )
        

        export_df = agg_df.rename(columns={x_col: x_label, value_col: y_label})
        meta = _build_metadata(_REQUIRED, _OPTIONAL, _DOMAINS, title)
        return fig, export_df, meta

    except Exception as exc:
        return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, f"Render error: {exc}")


def render_donut(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    **kwargs: Any,
) -> ChartReturn:
    _REQUIRED: List[str] = []
    _OPTIONAL: List[str] = [ROLE_CATEGORY, ROLE_STATUS, ROLE_ZONE, ROLE_PRIORITY]
    _DOMAINS: List[str] = ["all"]
    _CHART = "Donut Chart"

    x_role: Optional[str] = kwargs.get("x_role") or ROLE_STATUS
    y_role: Optional[str] = kwargs.get("y_role")
    aggregation: str = str(kwargs.get("aggregation", "count"))
    top_n: int = int(kwargs.get("top_n", 10))
    hole: float = float(kwargs.get("hole", 0.52))
    center_text: Optional[str] = kwargs.get("center_text")
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = kwargs.get("date_range")
    theme_key: str = str(kwargs.get("theme_key", DEFAULT_THEME_KEY))
    height: int = int(kwargs.get("height", _DEFAULT_HEIGHT))
    date_role: Optional[str] = kwargs.get("date_role") or ROLE_REGISTRATION_DATE

    try:
        theme = _get_theme(theme_key)
        date_col = _resolve_col(date_role, df, registry)
        df = _apply_date_range(df, date_col, date_range)
        if df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "No records after date filter.")

        x_col = _resolve_col(x_role, df, registry)
        if not x_col:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                               f"x_role '{x_role}' could not be resolved for donut segment names.")

        y_col = _resolve_col(y_role, df, registry) if y_role else None
        x_label = _col_label(x_role, registry)
        y_label = _col_label(y_role, registry) if y_role else "Count"

        agg_df, value_col = _aggregate(df, x_col, y_col, aggregation, None, None)
        agg_df = _apply_top_n_sort(agg_df, value_col, top_n, "desc")

        if agg_df.empty or agg_df[value_col].sum() == 0:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "All aggregated values are zero or empty.")

        title: str = str(kwargs.get("title") or _auto_title("donut", x_label, y_label, aggregation))

        is_status_col = (x_col == registry.resolve(ROLE_STATUS))
        if is_status_col and agg_df[x_col].isin(_STATUS_COLOR_MAP).any():
            color_map = {k: _STATUS_COLOR_MAP.get(k, "#64748B") for k in agg_df[x_col].unique()}
            fig = px.pie(
                agg_df, names=x_col, values=value_col, hole=hole,
                color=x_col, color_discrete_map=color_map,
                template=theme["plotly_template"],
            )
        else:
            color_seq = _color_seq(len(agg_df))
            fig = px.pie(
                agg_df, names=x_col, values=value_col, hole=hole,
                color_discrete_sequence=color_seq,
                template=theme["plotly_template"],
            )

        fig.update_traces(
            textinfo="percent+label",
            textfont=dict(size=11, family=_FONT_FAMILY),
            marker_line=dict(color=theme["surface"], width=2.5),
        )

        if center_text is not None:
            total_val = int(agg_df[value_col].sum())
            display_center = center_text if center_text else f"<b>{total_val:,}</b><br>Total"
            fig.add_annotation(
                text=display_center,
                x=0.5, y=0.5,
                font_size=16, font_family=_FONT_FAMILY,
                font_color=theme["text"],
                showarrow=False,
            )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family=_FONT_FAMILY, color=theme["text"], size=12),
            height=height,
            title="",
            margin={"l": 10, "r": 10, "t": _dynamic_top_margin(title, base=60), "b": 10},
            legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=11, color=theme["text"])),
            hoverlabel=dict(bgcolor=theme["surface"], font_size=12,
                            font_family=_FONT_FAMILY, bordercolor=theme["secondary"]),
        )

        export_df = agg_df.rename(columns={x_col: x_label, value_col: y_label})
        meta = _build_metadata(_REQUIRED, _OPTIONAL, _DOMAINS, title)
        return fig, export_df, meta

    except Exception as exc:
        return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, f"Render error: {exc}")


def render_histogram(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    **kwargs: Any,
) -> ChartReturn:
    _REQUIRED: List[str] = []
    _OPTIONAL: List[str] = [ROLE_AMOUNT, ROLE_CATEGORY, ROLE_STATUS]
    _DOMAINS: List[str] = ["all"]
    _CHART = "Histogram"

    x_role: Optional[str] = kwargs.get("x_role") or ROLE_AMOUNT
    color_role: Optional[str] = kwargs.get("color_role") or kwargs.get("group_by")
    facet_role: Optional[str] = kwargs.get("facet_role")
    nbins: int = int(kwargs.get("nbins", 0))
    histnorm: Optional[str] = kwargs.get("histnorm")
    barmode: str = str(kwargs.get("barmode", "overlay"))
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = kwargs.get("date_range")
    theme_key: str = str(kwargs.get("theme_key", DEFAULT_THEME_KEY))
    height: int = int(kwargs.get("height", _DEFAULT_HEIGHT))
    date_role: Optional[str] = kwargs.get("date_role") or ROLE_REGISTRATION_DATE
    outlier_method: Optional[str] = kwargs.get("outlier_method")

    try:
        theme = _get_theme(theme_key)
        date_col = _resolve_col(date_role, df, registry)
        df = _apply_date_range(df, date_col, date_range)
        if df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "No records after date filter.")

        x_col = _resolve_col(x_role, df, registry)
        if not x_col:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                               f"x_role '{x_role}' could not be resolved. "
                               "Map a numeric column to x_role for histogram distribution.")

        x_numeric = pd.to_numeric(df[x_col], errors="coerce")
        valid_ratio = x_numeric.notna().mean()
        if valid_ratio < 0.1:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                               f"Column '{x_col}' has fewer than 10% parseable numeric values "
                               f"(valid={valid_ratio:.0%}). Use a numeric role for histogram.")

        color_col = _resolve_col(color_role, df, registry) if color_role else None
        facet_col = _resolve_col(facet_role, df, registry) if facet_role else None
        x_label = _col_label(x_role, registry)
        color_label = _col_label(color_role, registry) if color_role else ""

        plot_df = df[[c for c in [x_col, color_col, facet_col] if c and c in df.columns]].copy()
        plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
        plot_df = plot_df.dropna(subset=[x_col])

        outlier_note = ""
        if outlier_method:
            keep_mask = _apply_outlier_filter(plot_df[x_col], outlier_method)
            pre_filter_rows = len(plot_df)
            plot_df = plot_df.loc[keep_mask].copy()
            removed = pre_filter_rows - len(plot_df)
            if removed > 0:
                outlier_note = f" {removed:,} extreme value(s) excluded via '{outlier_method}' filtering."

        if plot_df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                               "No numeric rows remain after outlier filtering.")

        title: str = str(kwargs.get("title") or _auto_title("histogram", x_label, "", "count"))

        n_colors = plot_df[color_col].nunique() if color_col else 1
        color_seq = _color_seq(n_colors)

        hist_kwargs: Dict[str, Any] = dict(
            data_frame=plot_df,
            x=x_col,
            color=color_col,
            facet_col=facet_col,
            barmode=barmode,
            opacity=0.80,
            color_discrete_sequence=color_seq,
            template=theme["plotly_template"],
        )
        if nbins > 0:
            hist_kwargs["nbins"] = nbins
        if histnorm:
            hist_kwargs["histnorm"] = histnorm

        fig = px.histogram(**hist_kwargs)
        fig.update_traces(marker_line_width=0.5, marker_line_color=theme["surface"])
        _apply_layout(fig, "" , theme, height=height, xaxis_title=x_label, yaxis_title="Frequency")

        desc = plot_df[x_col].describe().round(2)
        export_df = pd.DataFrame({
            "Statistic": desc.index,
            x_label: desc.values,
        })
        meta = _build_metadata(_REQUIRED, _OPTIONAL, _DOMAINS, title, reason=outlier_note.strip())
        return fig, export_df, meta

    except Exception as exc:
        return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, f"Render error: {exc}")


def render_box(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    **kwargs: Any,
) -> ChartReturn:
    _REQUIRED: List[str] = []
    _OPTIONAL: List[str] = [ROLE_AMOUNT, ROLE_CATEGORY, ROLE_STATUS, ROLE_OFFICER]
    _DOMAINS: List[str] = ["all"]
    _CHART = "Box Plot"

    y_role: Optional[str] = kwargs.get("y_role") or ROLE_AMOUNT
    x_role: Optional[str] = kwargs.get("x_role") or kwargs.get("group_by")
    color_role: Optional[str] = kwargs.get("color_role")
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = kwargs.get("date_range")
    theme_key: str = str(kwargs.get("theme_key", DEFAULT_THEME_KEY))
    height: int = int(kwargs.get("height", _DEFAULT_HEIGHT))
    points: Union[str, bool] = kwargs.get("points", "outliers")
    notched: bool = bool(kwargs.get("notched", False))
    date_role: Optional[str] = kwargs.get("date_role") or ROLE_REGISTRATION_DATE
    outlier_method: Optional[str] = kwargs.get("outlier_method")

    try:
        theme = _get_theme(theme_key)
        date_col = _resolve_col(date_role, df, registry)
        df = _apply_date_range(df, date_col, date_range)
        if df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "No records after date filter.")

        y_col = _resolve_col(y_role, df, registry)
        if not y_col:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                               f"y_role '{y_role}' could not be resolved. "
                               "Map a numeric column to y_role for box plot distribution.")

        y_numeric = pd.to_numeric(df[y_col], errors="coerce")
        if y_numeric.notna().mean() < 0.1:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                               f"Column '{y_col}' has fewer than 10% parseable numeric values.")

        x_col = _resolve_col(x_role, df, registry) if x_role else None
        color_col = _resolve_col(color_role, df, registry) if color_role else None
        y_label = _col_label(y_role, registry)
        x_label = _col_label(x_role, registry) if x_role else ""
        color_label = _col_label(color_role, registry) if color_role else ""

        plot_cols = [c for c in [y_col, x_col, color_col] if c and c in df.columns]
        plot_df = df[plot_cols].copy()
        plot_df[y_col] = y_numeric
        plot_df = plot_df.dropna(subset=[y_col])

        plot_cols = [c for c in [y_col, x_col, color_col] if c and c in df.columns]
        plot_df = df[plot_cols].copy()
        plot_df[y_col] = y_numeric
        plot_df = plot_df.dropna(subset=[y_col])

        if outlier_method:
            keep_mask = _apply_outlier_filter(plot_df[y_col], outlier_method)
            plot_df = plot_df.loc[keep_mask].copy()

        if plot_df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                               "No numeric rows remain after outlier filtering.")

        title: str = str(kwargs.get("title") or _auto_title("box", x_label, y_label, "none"))

        n_colors = plot_df[color_col].nunique() if color_col else (plot_df[x_col].nunique() if x_col else 1)
        color_seq = _color_seq(max(n_colors, 1))

        fig = px.box(
            plot_df,
            x=x_col,
            y=y_col,
            color=color_col or x_col,
            points=points,
            notched=notched,
            color_discrete_sequence=color_seq,
            template=theme["plotly_template"],
        )
        fig.update_traces(
            marker_size=3.5,
            line_width=1.5,
            boxmean="sd",
        )
        _apply_layout(fig, "" , theme, height=height, xaxis_title=x_label, yaxis_title=y_label)

        summary_rows = []
        group_col_for_summary = x_col or (color_col if color_col else None)
        if group_col_for_summary and group_col_for_summary in plot_df.columns:
            for gval, grp in plot_df.groupby(group_col_for_summary):
                s = grp[y_col].describe().round(2)
                row = {"Group": gval}
                row.update(s.to_dict())
                summary_rows.append(row)
            export_df = pd.DataFrame(summary_rows)
        else:
            s = plot_df[y_col].describe().round(2)
            export_df = pd.DataFrame({"Statistic": s.index, y_label: s.values})

        meta = _build_metadata(_REQUIRED, _OPTIONAL, _DOMAINS, title)
        return fig, export_df, meta

    except Exception as exc:
        return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, f"Render error: {exc}")


def render_sparkline(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    **kwargs: Any,
) -> ChartReturn:
    _REQUIRED: List[str] = []
    _OPTIONAL: List[str] = [ROLE_REGISTRATION_DATE, ROLE_RECORD_ID]
    _DOMAINS: List[str] = ["all"]
    _CHART = "KPI Trend Sparkline"

    values_direct: Optional[List[float]] = kwargs.get("values")
    values_role: Optional[str] = kwargs.get("values_role") or kwargs.get("y_role")
    period_role: Optional[str] = kwargs.get("period_role") or kwargs.get("x_role") or ROLE_REGISTRATION_DATE
    date_freq: str = str(kwargs.get("date_freq", "M"))
    trend_direction: str = str(kwargs.get("trend_direction", "none"))
    trend_is_positive: Optional[bool] = kwargs.get("trend_is_positive")
    color_override: Optional[str] = kwargs.get("color")
    theme_key: str = str(kwargs.get("theme_key", DEFAULT_THEME_KEY))
    width: int = int(kwargs.get("width", _SPARKLINE_WIDTH))
    height: int = int(kwargs.get("height", _SPARKLINE_HEIGHT))
    fill: bool = bool(kwargs.get("fill", True))

    try:
        theme = _get_theme(theme_key)

        if color_override:
            line_color = color_override
        else:
            if trend_direction == "none" or trend_direction == "neutral":
                line_color = _SPARKLINE_COLOR_MAP["neutral"]
            elif trend_is_positive is None:
                line_color = _SPARKLINE_COLOR_MAP["neutral"]
            elif trend_direction == "up" and trend_is_positive:
                line_color = _SPARKLINE_COLOR_MAP["up_good"]
            elif trend_direction == "up" and not trend_is_positive:
                line_color = _SPARKLINE_COLOR_MAP["up_bad"]
            elif trend_direction == "down" and trend_is_positive:
                line_color = _SPARKLINE_COLOR_MAP["down_good"]
            elif trend_direction == "down" and not trend_is_positive:
                line_color = _SPARKLINE_COLOR_MAP["down_bad"]
            else:
                line_color = _SPARKLINE_COLOR_MAP["neutral"]

        if values_direct is not None:
            y_vals = [float(v) for v in values_direct if v is not None and not math.isnan(float(v))]
            x_vals = list(range(len(y_vals)))
        elif not df.empty:
            period_col = _resolve_col(period_role, df, registry)
            val_col = _resolve_col(values_role, df, registry) if values_role else None

            if period_col and pd.api.types.is_datetime64_any_dtype(df[period_col]):
                tmp = df.copy()
                tmp["__period"] = pd.to_datetime(tmp[period_col], errors="coerce").dt.to_period(date_freq)
                if val_col:
                    series = tmp.groupby("__period")[val_col].mean().sort_index()
                else:
                    series = tmp.groupby("__period").size().sort_index()
                y_vals = [float(v) for v in series.values if not math.isnan(float(v))]
                x_vals = list(range(len(y_vals)))
            elif period_col:
                if val_col:
                    series = df.groupby(period_col)[val_col].mean().sort_index()
                else:
                    series = df.groupby(period_col).size().sort_index()
                y_vals = [float(v) for v in series.values if not math.isnan(float(v))]
                x_vals = list(range(len(y_vals)))
            else:
                return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                                   "Sparkline requires either 'values' kwarg (List[float]) or a "
                                   "resolvable period_role/x_role mapped to a date column.")
        else:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                               "No data: pass values=[...] or a non-empty DataFrame with a date role.")

        if len(y_vals) < 2:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                               "Sparkline requires at least 2 data points.")

        fill_color = (
            f"rgba({int(line_color[1:3], 16)},{int(line_color[3:5], 16)},"
            f"{int(line_color[5:7], 16)},0.18)"
        ) if fill else "none"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            line=dict(color=line_color, width=1.8, shape="spline"),
            fill="tozeroy" if fill else "none",
            fillcolor=fill_color,
            hoverinfo="skip",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            width=width,
            height=height,
            showlegend=False,
            xaxis=dict(visible=False, fixedrange=True),
            yaxis=dict(visible=False, fixedrange=True),
        )
        fig.update_xaxes(showgrid=False, showline=False, zeroline=False)
        fig.update_yaxes(showgrid=False, showline=False, zeroline=False)

        export_df = pd.DataFrame({"Period Index": x_vals, "Value": y_vals})
        meta = _build_metadata(_REQUIRED, _OPTIONAL, _DOMAINS, "KPI Trend Sparkline")
        return fig, export_df, meta

    except Exception as exc:
        return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, f"Render error: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC RENDER FUNCTIONS — Part 2: Advanced Matrix & Network Flows
# ══════════════════════════════════════════════════════════════════════════════

_HIERARCHY_ROLES_DEFAULT: List[str] = [
    role for role, _label in CANONICAL_GEO_HIERARCHY
] + [ROLE_CATEGORY]
_CAL_COLORSCALE: List[List[Any]] = [
    [0.0, "#F5F7FA"], [0.2, "#BBDEFB"], [0.45, "#64B5F6"],
    [0.70, "#1976D2"], [1.0, "#003B73"],
]
_DAY_ABBR: List[str] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_MONTH_ABBR: List[str] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def _resolve_hierarchy(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    hierarchy_roles: List[str],
) -> Tuple[List[str], List[str]]:
    cols, labels = [], []
    for role in hierarchy_roles:
        col = _resolve_col(role, df, registry)
        if col and df[col].notna().any():
            cols.append(col)
            labels.append(_col_label(role, registry))
    return cols, labels


def _build_hierarchy_agg(
    df: pd.DataFrame,
    path_cols: List[str],
    y_col: Optional[str],
    aggregation: str,
) -> Tuple[pd.DataFrame, str]:
    if not path_cols:
        return pd.DataFrame(), "__value"
    if y_col is None or aggregation == "count":
        agg_df = (
            df.groupby(path_cols, dropna=True)
            .size()
            .reset_index(name="__value")
        )
    else:
        tmp = df[path_cols].copy()
        tmp["__value"] = pd.to_numeric(df[y_col], errors="coerce")
        agg_df = (
            tmp.groupby(path_cols, dropna=True)["__value"]
            .agg(aggregation)
            .reset_index()
        )
    agg_df = agg_df[agg_df["__value"].notna() & (agg_df["__value"] > 0)]
    return agg_df, "__value"


def _clip_to_top_n_per_level(
    df: pd.DataFrame,
    cols: List[str],
    top_n: int,
) -> pd.DataFrame:
    if top_n <= 0:
        return df
    result = df.copy()
    for col in cols:
        if col not in result.columns:
            continue
        top_vals: set = set(result[col].value_counts().nlargest(top_n).index)
        result[col] = result[col].apply(lambda v: v if v in top_vals else "(Other)")
    return result


def render_treemap(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    **kwargs: Any,
) -> ChartReturn:
    _REQUIRED: List[str] = []
    _OPTIONAL: List[str] = _HIERARCHY_ROLES_DEFAULT
    _DOMAINS: List[str] = ["Complaint Management", "Supply Operations", "Revenue Analytics", "all"]
    _CHART = "Treemap — Nested Hierarchy"

    hierarchy_roles: List[str] = kwargs.get("hierarchy_roles", _HIERARCHY_ROLES_DEFAULT)
    y_role: Optional[str] = kwargs.get("y_role")
    aggregation: str = str(kwargs.get("aggregation", "count"))
    color_mode: str = str(kwargs.get("color_mode", "level"))
    top_n: int = int(kwargs.get("top_n", 0))
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = kwargs.get("date_range")
    date_role: Optional[str] = kwargs.get("date_role") or ROLE_REGISTRATION_DATE
    theme_key: str = str(kwargs.get("theme_key", DEFAULT_THEME_KEY))
    height: int = int(kwargs.get("height", 500))
    max_depth: int = int(kwargs.get("max_depth", -1))

    try:
        theme = _get_theme(theme_key)
        date_col = _resolve_col(date_role, df, registry)
        df = _apply_date_range(df, date_col, date_range)
        if df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "No records after date filter.")

        path_cols, path_labels = _resolve_hierarchy(df, registry, hierarchy_roles)
        if not path_cols:
            return _ineligible(
                _REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                "No hierarchy roles are mapped. Map at least one of: "
                + ", ".join(_col_label(r, registry) for r in _HIERARCHY_ROLES_DEFAULT)
                + " in the Schema Mapping Studio.",
            )

        if top_n > 0:
            df = _clip_to_top_n_per_level(df, path_cols, top_n)

        y_col = _resolve_col(y_role, df, registry) if y_role else None
        y_label = _col_label(y_role, registry) if y_role else "Count"
        agg_df, value_col = _build_hierarchy_agg(df, path_cols, y_col, aggregation)

        if agg_df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "Hierarchy aggregation produced no rows.")

        depth_label = " › ".join(path_labels)
        title: str = str(kwargs.get("title") or f"{y_label} by {depth_label}")

        n_top = agg_df[path_cols[0]].nunique() if path_cols else 1
        color_seq = _color_seq(n_top)

        if color_mode == "value":
            fig = px.treemap(
                agg_df, path=path_cols, values=value_col,
                color=value_col,
                color_continuous_scale=_SEQUENTIAL_BLUES,
                template=theme["plotly_template"],
            )
            fig.update_coloraxes(
                colorbar=dict(
                    thickness=12, len=0.6,
                    title=dict(text=y_label, font=dict(size=11, family=_FONT_FAMILY)),
                    tickfont=dict(size=10, family=_FONT_FAMILY),
                )
            )
        else:
            fig = px.treemap(
                agg_df, path=path_cols, values=value_col,
                color=path_cols[0],
                color_discrete_sequence=color_seq,
                template=theme["plotly_template"],
            )

        fig.update_traces(
            textinfo="label+value+percent root",
            textfont=dict(size=11, family=_FONT_FAMILY),
            marker_line=dict(width=1.5, color=theme["surface"]),
            hovertemplate="<b>%{label}</b><br>Value: %{value:,}<br>Share: %{percentRoot:.1%}<extra></extra>",
        )
        if max_depth > 0:
            fig.update_traces(maxdepth=max_depth)
            # title=dict(
            #     text=f"<b>{title}</b>",
            #     font=dict(size=14, color=theme["text"], family=_FONT_FAMILY),
            #     x=0.0, xanchor="left",
            #     y=0.97, yanchor="top",
            #     automargin=True,
            # ),

        fig.update_layout( 
            title= "", 
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family=_FONT_FAMILY, color=theme["text"], size=12),
            height=height,
            margin={"l": 10, "r": 10, "t": _dynamic_top_margin(title, base=60), "b": 10},
            hoverlabel=dict(bgcolor=theme["surface"], font_size=12,
                            font_family=_FONT_FAMILY, bordercolor=theme["secondary"]),
        )

        export_df = agg_df.rename(columns={value_col: y_label})
        for i, (col, lbl) in enumerate(zip(path_cols, path_labels)):
            if col != lbl:
                export_df = export_df.rename(columns={col: lbl})

        meta = _build_metadata(_REQUIRED, _OPTIONAL, _DOMAINS, title)
        return fig, export_df, meta

    except Exception as exc:
        return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, f"Render error: {exc}")


def render_sunburst(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    **kwargs: Any,
) -> ChartReturn:
    _REQUIRED: List[str] = []
    _OPTIONAL: List[str] = _HIERARCHY_ROLES_DEFAULT
    _DOMAINS: List[str] = ["Complaint Management", "Supply Operations", "all"]
    _CHART = "Sunburst — Hierarchical Distribution"

    hierarchy_roles: List[str] = kwargs.get("hierarchy_roles", _HIERARCHY_ROLES_DEFAULT)
    y_role: Optional[str] = kwargs.get("y_role")
    aggregation: str = str(kwargs.get("aggregation", "count"))
    top_n: int = int(kwargs.get("top_n", 0))
    max_depth: int = int(kwargs.get("max_depth", 3))
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = kwargs.get("date_range")
    date_role: Optional[str] = kwargs.get("date_role") or ROLE_REGISTRATION_DATE
    theme_key: str = str(kwargs.get("theme_key", DEFAULT_THEME_KEY))
    height: int = int(kwargs.get("height", 500))

    try:
        theme = _get_theme(theme_key)
        date_col = _resolve_col(date_role, df, registry)
        df = _apply_date_range(df, date_col, date_range)
        if df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "No records after date filter.")

        path_cols, path_labels = _resolve_hierarchy(df, registry, hierarchy_roles)
        if not path_cols:
            return _ineligible(
                _REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                "No hierarchy roles are mapped. Map Zone, Division, Category, or similar roles.",
            )

        if top_n > 0:
            df = _clip_to_top_n_per_level(df, path_cols, top_n)

        y_col = _resolve_col(y_role, df, registry) if y_role else None
        y_label = _col_label(y_role, registry) if y_role else "Count"
        agg_df, value_col = _build_hierarchy_agg(df, path_cols, y_col, aggregation)

        if agg_df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "Hierarchy aggregation produced no rows.")

        depth_label = " › ".join(path_labels)
        title: str = str(kwargs.get("title") or f"{y_label} Distribution — {depth_label}")

        n_top = agg_df[path_cols[0]].nunique() if path_cols else 1
        color_seq = _color_seq(n_top)

        fig = px.sunburst(
            agg_df,
            path=path_cols,
            values=value_col,
            color=path_cols[0],
            color_discrete_sequence=color_seq,
            template=theme["plotly_template"],
        )
        fig.update_traces(
            textinfo="label+percent root",
            textfont=dict(size=11, family=_FONT_FAMILY),
            insidetextorientation="radial",
            marker_line=dict(width=1.0, color=theme["surface"]),
            hovertemplate="<b>%{label}</b><br>Value: %{value:,}<br>Share of Total: %{percentRoot:.1%}<extra></extra>",
            maxdepth=max_depth,
        )
        fig.update_layout(
            title="",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family=_FONT_FAMILY, color=theme["text"], size=12),
            height=height,
            margin={"l": 10, "r": 10, "t": _dynamic_top_margin(title, base=60), "b": 10},
            hoverlabel=dict(bgcolor=theme["surface"], font_size=12,
                            font_family=_FONT_FAMILY, bordercolor=theme["secondary"]),
        )

        export_df = agg_df.rename(columns={value_col: y_label})
        for col, lbl in zip(path_cols, path_labels):
            if col != lbl:
                export_df = export_df.rename(columns={col: lbl})

        meta = _build_metadata(_REQUIRED, _OPTIONAL, _DOMAINS, title)
        return fig, export_df, meta

    except Exception as exc:
        return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, f"Render error: {exc}")


def render_heatmap(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    **kwargs: Any,
) -> ChartReturn:
    _REQUIRED: List[str] = []
    _OPTIONAL: List[str] = [ROLE_CATEGORY, ROLE_STATUS, ROLE_OFFICER, ROLE_ZONE, ROLE_AMOUNT]
    _DOMAINS: List[str] = ["all"]
    _CHART = "Heatmap — 2D Density / Correlation Matrix"

    mode: str = str(kwargs.get("mode", "auto"))
    x_role: Optional[str] = kwargs.get("x_role")
    y_role: Optional[str] = kwargs.get("y_role")
    value_role: Optional[str] = kwargs.get("value_role") or kwargs.get("z_role")
    aggregation: str = str(kwargs.get("aggregation", "count"))
    colorscale: Any = kwargs.get("colorscale", "Blues")
    annotate: bool = bool(kwargs.get("annotate", True))
    annotation_threshold: int = int(kwargs.get("annotation_threshold", 200))
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = kwargs.get("date_range")
    date_role: Optional[str] = kwargs.get("date_role") or ROLE_REGISTRATION_DATE
    theme_key: str = str(kwargs.get("theme_key", DEFAULT_THEME_KEY))
    height: int = int(kwargs.get("height", _DEFAULT_HEIGHT))

    try:
        theme = _get_theme(theme_key)
        date_col = _resolve_col(date_role, df, registry)
        df = _apply_date_range(df, date_col, date_range)
        if df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "No records after date filter.")

        x_col = _resolve_col(x_role, df, registry) if x_role else None
        y_col = _resolve_col(y_role, df, registry) if y_role else None

        use_correlation = (
            mode == "correlation"
            or (mode == "auto" and (x_col is None or y_col is None))
        )

        if use_correlation:
            # Refactor Phase 3 / Audit Finding B5 remediation —
            # df.select_dtypes(include=[np.number]) has documented,
            # version-dependent unreliability against ArrowDtype-backed
            # numeric columns (pd.ArrowDtype(pa.float64())) across pandas
            # 2.0-2.2. pd.api.types.is_numeric_dtype reliably matches
            # ArrowDtype numeric columns where select_dtypes does not.
            numeric_cols_b5 = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            numeric_df = df[numeric_cols_b5]
            if numeric_df.shape[1] < 2:
                return _ineligible(
                    _REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                    "Correlation heatmap requires at least 2 numeric columns. "
                    "Map numeric roles (Amount, Units Consumed, etc.) or provide x_role and y_role "
                    "for a categorical density heatmap.",
                )
            corr = numeric_df.corr(method="pearson").round(3)
            z_values = corr.values
            x_labels = corr.columns.tolist()
            y_labels = corr.index.tolist()
            colorscale_used = "RdBu_r"
            zmin, zmax = -1.0, 1.0
            hover_fmt = ".3f"
            title: str = str(kwargs.get("title") or "Numeric Column Correlation Matrix (Pearson)")
            export_df = corr.reset_index().rename(columns={"index": "Column"})
        else:
            if not x_col or not y_col:
                return _ineligible(
                    _REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                    "Crosstab heatmap requires both x_role and y_role to be mapped to categorical columns.",
                )
            val_col = _resolve_col(value_role, df, registry) if value_role else None
            x_label = _col_label(x_role, registry)
            y_label = _col_label(y_role, registry)
            val_label = _col_label(value_role, registry) if value_role else "Count"

            if val_col is None or aggregation == "count":
                pivot = (
                    df.groupby([y_col, x_col], dropna=True)
                    .size()
                    .unstack(fill_value=0)
                )
            else:
                numeric_val = pd.to_numeric(df[val_col], errors="coerce")
                tmp = df[[x_col, y_col]].copy()
                tmp["__v"] = numeric_val
                pivot = tmp.groupby([y_col, x_col], dropna=True)["__v"].agg(aggregation).unstack(fill_value=0)

            pivot = pivot.fillna(0)
            z_values = pivot.values
            x_labels = [str(c) for c in pivot.columns.tolist()]
            y_labels = [str(r) for r in pivot.index.tolist()]
            colorscale_used = colorscale
            zmin, zmax = None, None
            hover_fmt = ",.1f"
            title = str(kwargs.get("title") or f"{val_label} — {y_label} × {x_label}")
            export_df = pivot.copy()
            export_df.index.name = y_label
            export_df.columns.name = x_label
            export_df = export_df.reset_index()

        n_cells = z_values.size
        show_text = annotate and n_cells <= annotation_threshold
        text_matrix: Optional[List[List[str]]] = None
        if show_text:
            text_matrix = [
                [f"{v:.3f}" if use_correlation else f"{v:,.0f}" for v in row]
                for row in z_values
            ]

        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale_used,
            zmin=zmin,
            zmax=zmax,
            text=text_matrix,
            texttemplate="%{text}" if show_text else None,
            textfont=dict(size=9, family=_FONT_FAMILY,
                          color=theme["surface"] if not use_correlation else None),
            hovertemplate=f"<b>%{{x}}</b> × <b>%{{y}}</b><br>Value: %{{z:{hover_fmt}}}<extra></extra>",
            colorbar=dict(
                thickness=14, len=0.85,
                tickfont=dict(size=10, family=_FONT_FAMILY, color=theme["text"]),
                title=dict(
                    text="r" if use_correlation else "Value",
                    font=dict(size=11, family=_FONT_FAMILY, color=theme["secondary"]),
                ),
            ),
        ))

        _apply_layout(
            fig, "" , theme, height=height,
            showlegend=False,
            margin={"l": 80, "r": 20, "t": 64, "b": 80},
        )
        fig.update_xaxes(tickangle=-40, side="bottom", tickfont=dict(size=10))
        fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))

        meta = _build_metadata(_REQUIRED, _OPTIONAL, _DOMAINS, title)
        return fig, export_df, meta

    except Exception as exc:
        return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, f"Render error: {exc}")


def render_calendar_heatmap(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    **kwargs: Any,
) -> ChartReturn:
    _REQUIRED: List[str] = [ROLE_REGISTRATION_DATE]
    _OPTIONAL: List[str] = [ROLE_RECORD_ID, ROLE_AMOUNT]
    _DOMAINS: List[str] = ["Complaint Management", "Revenue Analytics", "all"]
    _CHART = "Calendar Heatmap — Daily Activity Grid"

    date_role: str = str(kwargs.get("date_role") or ROLE_REGISTRATION_DATE)
    y_role: Optional[str] = kwargs.get("y_role")
    aggregation: str = str(kwargs.get("aggregation", "count"))
    year: Optional[int] = kwargs.get("year")
    colorscale: Any = kwargs.get("colorscale", _CAL_COLORSCALE)
    theme_key: str = str(kwargs.get("theme_key", DEFAULT_THEME_KEY))
    height: int = int(kwargs.get("height", 220))
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = kwargs.get("date_range")

    try:
        theme = _get_theme(theme_key)
        date_col = _resolve_col(date_role, df, registry)
        if not date_col:
            return _ineligible(
                _REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                f"Date role '{date_role}' is not mapped. Map a date column to Registration Date.",
            )

        df = _apply_date_range(df, date_col, date_range)
        if df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "No records after date filter.")

        date_series = pd.to_datetime(df[date_col], errors="coerce")
        valid_mask = date_series.notna()
        if valid_mask.sum() < 7:
            return _ineligible(
                _REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                f"Fewer than 7 valid dates found in '{date_col}'. "
                "Calendar heatmap requires at least one week of dated records.",
            )

        y_col = _resolve_col(y_role, df, registry) if y_role else None
        y_label = _col_label(y_role, registry) if y_role else "Count"

        tmp = df[valid_mask].copy()
        tmp["__date"] = date_series[valid_mask].dt.normalize()

        if y_col is None or aggregation == "count":
            daily = tmp.groupby("__date").size()
            daily.name = "__val"
        else:
            tmp["__val"] = pd.to_numeric(tmp[y_col], errors="coerce")
            daily = tmp.groupby("__date")["__val"].agg(aggregation)

        daily.index = pd.to_datetime(daily.index)

        target_year: int = int(year) if year else int(daily.index.year.max())
        year_data = daily[daily.index.year == target_year]

        if year_data.empty:
            return _ineligible(
                _REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                f"No data found for year {target_year}.",
            )

        full_range = pd.date_range(
            start=f"{target_year}-01-01",
            end=f"{target_year}-12-31",
            freq="D",
        )
        year_series = year_data.reindex(full_range, fill_value=0).astype(float)

        iso_cal = year_series.index.isocalendar()
        frame = pd.DataFrame({
            "date": year_series.index,
            "value": year_series.values,
            "week": iso_cal.week.astype(int),
            "weekday": year_series.index.dayofweek,
            "month": year_series.index.month,
        })
        frame.loc[
            (frame["month"] == 1) & (frame["week"] > 50), "week"
        ] = 1
        frame.loc[
            (frame["month"] == 12) & (frame["week"] == 1), "week"
        ] = 53

        pivot = (
            frame.pivot_table(index="weekday", columns="week", values="value", aggfunc="sum")
            .reindex(index=range(7))
        )
        pivot = pivot.fillna(0)

        month_tick_map: Dict[int, int] = {}
        for _, row in frame.iterrows():
            m = int(row["month"])
            w = int(row["week"])
            if m not in month_tick_map:
                month_tick_map[m] = w

        x_tickvals = list(month_tick_map.values())
        x_ticktext = [_MONTH_ABBR[m - 1] for m in month_tick_map.keys()]

        hover_matrix = [
            [
                frame.loc[
                    (frame["weekday"] == wd) & (frame["week"] == wk), "date"
                ].dt.strftime("%d %b %Y").values[0]
                if not frame.loc[
                    (frame["weekday"] == wd) & (frame["week"] == wk)
                ].empty else ""
                for wk in pivot.columns
            ]
            for wd in range(7)
        ]

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=list(range(7)),
            colorscale=colorscale,
            showscale=True,
            text=hover_matrix,
            hovertemplate="%{text}<br>" + y_label + ": %{z:,.0f}<extra></extra>",
            xgap=2,
            ygap=2,
            colorbar=dict(
                thickness=10, len=0.9,
                tickfont=dict(size=9, family=_FONT_FAMILY, color=theme["text"]),
                title=dict(
                    text=y_label,
                    font=dict(size=10, family=_FONT_FAMILY, color=theme["secondary"]),
                ),
            ),
        ))

        title: str = str(kwargs.get("title") or f"Daily {y_label} — {target_year}")

        fig.update_layout(
            title="",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family=_FONT_FAMILY, color=theme["text"], size=11),
            height=height,
            margin={"l": 50, "r": 20, "t": _dynamic_top_margin(title, base=60), "b": 20},
            hoverlabel=dict(bgcolor=theme["surface"], font_size=11,
                            font_family=_FONT_FAMILY, bordercolor=theme["secondary"]),
            xaxis=dict(
                tickmode="array",
                tickvals=x_tickvals,
                ticktext=x_ticktext,
                showgrid=False,
                zeroline=False,
                tickfont=dict(size=10, color=theme["secondary"]),
            ),
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(7)),
                ticktext=_DAY_ABBR,
                showgrid=False,
                zeroline=False,
                autorange="reversed",
                tickfont=dict(size=10, color=theme["secondary"]),
            ),
        )

        export_df = pd.DataFrame({
            "Date": year_series.index.strftime("%Y-%m-%d"),
            "Day": year_series.index.day_name(),
            y_label: year_series.values,
        })

        meta = _build_metadata(_REQUIRED, _OPTIONAL, _DOMAINS, title)
        return fig, export_df, meta

    except Exception as exc:
        return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, f"Render error: {exc}")


def render_pareto_diagram(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    **kwargs: Any,
) -> ChartReturn:
    _REQUIRED: List[str] = []
    _OPTIONAL: List[str] = [ROLE_CATEGORY, ROLE_OFFICER, ROLE_ZONE, ROLE_AMOUNT, ROLE_RECORD_ID]
    _DOMAINS: List[str] = ["all"]
    _CHART = "Pareto Diagram — 80/20 Bottleneck Breakdown"

    x_role: Optional[str] = kwargs.get("x_role") or ROLE_CATEGORY
    y_role: Optional[str] = kwargs.get("y_role")
    aggregation: str = str(kwargs.get("aggregation", "count"))
    top_n: int = int(kwargs.get("top_n", 20))
    threshold_pct: float = float(kwargs.get("threshold_pct", 80.0))
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = kwargs.get("date_range")
    date_role: Optional[str] = kwargs.get("date_role") or ROLE_REGISTRATION_DATE
    theme_key: str = str(kwargs.get("theme_key", DEFAULT_THEME_KEY))
    height: int = int(kwargs.get("height", _DEFAULT_HEIGHT))

    try:
        theme = _get_theme(theme_key)
        date_col = _resolve_col(date_role, df, registry)
        df = _apply_date_range(df, date_col, date_range)
        if df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "No records after date filter.")

        x_col = _resolve_col(x_role, df, registry)
        if not x_col:
            return _ineligible(
                _REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                f"x_role '{x_role}' could not be resolved. Map a categorical column for the Pareto X axis.",
            )

        y_col = _resolve_col(y_role, df, registry) if y_role else None
        x_label = _col_label(x_role, registry)
        y_label = _col_label(y_role, registry) if y_role else "Count"

        agg_df, value_col = _aggregate(df, x_col, y_col, aggregation, None, None)
        agg_df = agg_df.sort_values(value_col, ascending=False)
        if top_n > 0:
            agg_df = agg_df.head(top_n)
        agg_df = agg_df.reset_index(drop=True)

        if agg_df.empty or agg_df[value_col].sum() == 0:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "Aggregation produced zero total — cannot compute Pareto.")

        total = float(agg_df[value_col].sum())
        agg_df["__cum_pct"] = (agg_df[value_col].cumsum() / total * 100).round(2)

        threshold_idx: Optional[int] = None
        for i, pct in enumerate(agg_df["__cum_pct"]):
            if pct >= threshold_pct:
                threshold_idx = i
                break
        vital_few = threshold_idx + 1 if threshold_idx is not None else len(agg_df)

        title: str = str(kwargs.get("title") or f"Pareto — {y_label} by {x_label}")

        bar_colors = [theme["primary"]] * vital_few + [theme["secondary"]] * (len(agg_df) - vital_few)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Bar(
                x=agg_df[x_col],
                y=agg_df[value_col],
                name=y_label,
                marker_color=bar_colors,
                marker_line_width=0,
                hovertemplate=f"<b>%{{x}}</b><br>{y_label}: %{{y:,}}<extra></extra>",
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=agg_df[x_col],
                y=agg_df["__cum_pct"],
                name="Cumulative %",
                mode="lines+markers",
                line=dict(color="#DC2626", width=2.2, shape="spline"),
                marker=dict(size=5, color="#DC2626"),
                hovertemplate="<b>%{x}</b><br>Cumulative: %{y:.1f}%<extra></extra>",
            ),
            secondary_y=True,
        )

        fig.add_hline(
            y=threshold_pct,
            secondary_y=True,
            line_dash="dash",
            line_color="#D97706",
            line_width=1.5,
            annotation_text=f"{threshold_pct:.0f}% threshold",
            annotation_position="top right",
            annotation_font=dict(size=10, color="#D97706", family=_FONT_FAMILY),
        )

        fig.update_layout(
            title="",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family=_FONT_FAMILY, color=theme["text"], size=12),
            height=height,
            legend=dict(
                bgcolor="rgba(0,0,0,0)", borderwidth=0,
                font=dict(size=11, family=_FONT_FAMILY, color=theme["text"]),
                orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
            ),
            margin={"l": 20, "r": 60, "t": 64, "b": 60},
            hoverlabel=dict(bgcolor=theme["surface"], font_size=12,
                            font_family=_FONT_FAMILY, bordercolor=theme["secondary"]),
            barmode="overlay",
        )
        fig.update_xaxes(
            tickangle=-35,
            gridcolor="rgba(100,116,139,0.12)",
            linecolor="rgba(100,116,139,0.25)",
            tickfont=dict(size=10, color=theme["secondary"], family=_FONT_FAMILY),
            title_text=x_label,
            title_font=dict(size=12, color=theme["secondary"]),
            zeroline=False,
        )
        fig.update_yaxes(
            title_text=y_label,
            gridcolor="rgba(100,116,139,0.12)",
            linecolor="rgba(100,116,139,0.25)",
            tickfont=dict(size=11, color=theme["secondary"], family=_FONT_FAMILY),
            title_font=dict(size=12, color=theme["secondary"]),
            zeroline=False,
            secondary_y=False,
        )
        fig.update_yaxes(
            title_text="Cumulative %",
            range=[0, 105],
            ticksuffix="%",
            showgrid=False,
            linecolor="rgba(100,116,139,0.25)",
            tickfont=dict(size=11, color="#DC2626", family=_FONT_FAMILY),
            title_font=dict(size=12, color="#DC2626"),
            zeroline=False,
            secondary_y=True,
        )

        export_df = agg_df[[x_col, value_col, "__cum_pct"]].rename(columns={
            x_col: x_label,
            value_col: f"{aggregation.title()} {y_label}" if y_label != "Count" else "Count",
            "__cum_pct": "Cumulative %",
        })
        export_df.insert(
            len(export_df.columns),
            "Pareto Class",
            ["Vital Few"] * vital_few + ["Trivial Many"] * (len(agg_df) - vital_few),
        )

        meta = _build_metadata(_REQUIRED, _OPTIONAL, _DOMAINS, title)
        return fig, export_df, meta

    except Exception as exc:
        return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, f"Render error: {exc}")


def render_sankey_diagram(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    **kwargs: Any,
) -> ChartReturn:
    _REQUIRED: List[str] = []
    _OPTIONAL: List[str] = [ROLE_CATEGORY, ROLE_OFFICER, ROLE_STATUS, ROLE_ZONE, ROLE_DIVISION]
    _DOMAINS: List[str] = ["Complaint Management", "Supply Operations", "all"]
    _CHART = "Sankey Diagram — Workflow Path Flow"

    flow_roles: List[str] = kwargs.get(
        "flow_roles",
        [ROLE_CATEGORY, ROLE_OFFICER, ROLE_STATUS],
    )
    y_role: Optional[str] = kwargs.get("y_role")
    aggregation: str = str(kwargs.get("aggregation", "count"))
    min_flow: float = float(kwargs.get("min_flow", 1.0))
    top_n_per_level: int = int(kwargs.get("top_n_per_level", 15))
    link_opacity: float = float(kwargs.get("link_opacity", 0.38))
    node_pad: int = int(kwargs.get("node_pad", 18))
    node_thickness: int = int(kwargs.get("node_thickness", 22))
    date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = kwargs.get("date_range")
    date_role: Optional[str] = kwargs.get("date_role") or ROLE_REGISTRATION_DATE
    theme_key: str = str(kwargs.get("theme_key", DEFAULT_THEME_KEY))
    height: int = int(kwargs.get("height", 520))

    try:
        theme = _get_theme(theme_key)
        date_col = _resolve_col(date_role, df, registry)
        df = _apply_date_range(df, date_col, date_range)
        if df.empty:
            return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, "No records after date filter.")

        flow_cols: List[str] = []
        flow_labels: List[str] = []
        for role in flow_roles:
            col = _resolve_col(role, df, registry)
            if col and df[col].notna().any():
                flow_cols.append(col)
                flow_labels.append(_col_label(role, registry))

        if len(flow_cols) < 2:
            resolved = [_col_label(r, registry) for r in flow_roles]
            return _ineligible(
                _REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                f"Sankey requires at least 2 resolvable, non-empty flow roles. "
                f"Could only resolve {len(flow_cols)} of: {', '.join(resolved)}. "
                "Map roles such as Category, Officer, and Status.",
            )

        work_df = _clip_to_top_n_per_level(df, flow_cols, top_n_per_level)

        for col in flow_cols:
            work_df[col] = work_df[col].fillna("(Unknown)").astype(str).str.strip()

        y_col = _resolve_col(y_role, work_df, registry) if y_role else None

        all_node_labels: List[str] = []
        node_level_map: Dict[str, int] = {}

        for level_idx, col in enumerate(flow_cols):
            for val in work_df[col].unique():
                label = str(val)
                if label not in node_level_map:
                    all_node_labels.append(label)
                    node_level_map[label] = level_idx

        node_index: Dict[str, int] = {lbl: i for i, lbl in enumerate(all_node_labels)}

        link_sources: List[int] = []
        link_targets: List[int] = []
        link_values: List[float] = []
        link_hover: List[str] = []
        flow_records: List[Dict[str, Any]] = []

        for i in range(len(flow_cols) - 1):
            src_col = flow_cols[i]
            tgt_col = flow_cols[i + 1]
            src_label_name = flow_labels[i]
            tgt_label_name = flow_labels[i + 1]

            if y_col and y_col in work_df.columns and aggregation != "count":
                work_df["__val"] = pd.to_numeric(work_df[y_col], errors="coerce")
                agg_df = (
                    work_df.groupby([src_col, tgt_col], dropna=True)["__val"]
                    .agg(aggregation)
                    .reset_index()
                    .rename(columns={"__val": "__flow"})
                )
            else:
                agg_df = (
                    work_df.groupby([src_col, tgt_col], dropna=True)
                    .size()
                    .reset_index(name="__flow")
                )

            agg_df = agg_df[agg_df["__flow"] >= min_flow]

            for _, row in agg_df.iterrows():
                src_val = str(row[src_col])
                tgt_val = str(row[tgt_col])
                flow_val = float(row["__flow"])

                if src_val not in node_index or tgt_val not in node_index:
                    continue

                link_sources.append(node_index[src_val])
                link_targets.append(node_index[tgt_val])
                link_values.append(flow_val)
                link_hover.append(
                    f"{src_label_name}: {src_val} → {tgt_label_name}: {tgt_val}: {int(flow_val):,}"
                )
                flow_records.append({
                    "Source Level": src_label_name,
                    "Source": src_val,
                    "Target Level": tgt_label_name,
                    "Target": tgt_val,
                    "Flow": flow_val,
                })

        if not link_sources:
            return _ineligible(
                _REQUIRED, _OPTIONAL, _DOMAINS, _CHART,
                f"No flows met the minimum flow threshold of {min_flow}. "
                "Reduce min_flow or check that the mapped columns have overlapping records.",
            )

        level_palette = _color_seq(len(flow_cols))

        def _hex_to_rgba(hex_color: str, alpha: float) -> str:
            h = hex_color.lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"

        node_colors = [
            level_palette[node_level_map.get(lbl, 0) % len(level_palette)]
            for lbl in all_node_labels
        ]
        link_colors = [
            _hex_to_rgba(
                level_palette[node_level_map.get(all_node_labels[src_idx], 0) % len(level_palette)],
                link_opacity,
            )
            for src_idx in link_sources
        ]

        flow_path = " → ".join(flow_labels)
        title: str = str(kwargs.get("title") or f"Workflow Flow — {flow_path}")

        fig = go.Figure(data=go.Sankey(
            arrangement="snap",
            node=dict(
                pad=node_pad,
                thickness=node_thickness,
                line=dict(color=theme["surface"], width=0.5),
                label=all_node_labels,
                color=node_colors,
                hovertemplate="<b>%{label}</b><br>Total flow: %{value:,}<extra></extra>",
            ),
            link=dict(
                source=link_sources,
                target=link_targets,
                value=link_values,
                color=link_colors,
                label=link_hover,
                hovertemplate="%{label}<extra></extra>",
            ),
        ))

        level_annotations = []
        for i, lbl in enumerate(flow_labels):
            level_annotations.append(dict(
                x=i / max(len(flow_labels) - 1, 1),
                y=1.06,
                xref="paper",
                yref="paper",
                text=f"<b>{lbl}</b>",
                showarrow=False,
                font=dict(size=11, color=theme["secondary"], family=_FONT_FAMILY),
                align="center",
            ))

        fig.update_layout(
            title="",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family=_FONT_FAMILY, color=theme["text"], size=11),
            height=height,
            margin={"l": 10, "r": 10, "t": _dynamic_top_margin(title, base=80), "b": 10},
            hoverlabel=dict(bgcolor=theme["surface"], font_size=11,
                            font_family=_FONT_FAMILY, bordercolor=theme["secondary"]),
            annotations=level_annotations,
        )

        export_df = pd.DataFrame(flow_records)
        meta = _build_metadata(_REQUIRED, _OPTIONAL, _DOMAINS, title)
        return fig, export_df, meta

    except Exception as exc:
        return _ineligible(_REQUIRED, _OPTIONAL, _DOMAINS, _CHART, f"Render error: {exc}")


# ── Dispatch registry for self-service builder ──────────────────────────────────
CHART_REGISTRY: Dict[str, Any] = {
    "bar": render_bar,
    "bar_horizontal": render_bar_horizontal,
    "bar_grouped": render_bar_grouped,
    "bar_stacked": render_bar_stacked,
    "line": render_line,
    "area": render_area,
    "scatter": render_scatter,
    "bubble": render_bubble,
    "pie": render_pie,
    "donut": render_donut,
    "histogram": render_histogram,
    "box": render_box,
    "sparkline": render_sparkline,
    "treemap": render_treemap,
    "sunburst": render_sunburst,
    "heatmap": render_heatmap,
    "calendar_heatmap": render_calendar_heatmap,
    "pareto": render_pareto_diagram,
    "sankey": render_sankey_diagram,
}


def render(
    chart_type: str,
    df: pd.DataFrame,
    registry: ColumnRegistry,
    **kwargs: Any,
) -> ChartReturn:
    fn = CHART_REGISTRY.get(chart_type)
    if fn is None:
        return _ineligible(
            [], [], ["all"], f"Unknown chart type: {chart_type}",
            f"'{chart_type}' is not registered. "
            f"Available types: {', '.join(sorted(CHART_REGISTRY))}",
        )
    try:
        return fn(df, registry, **kwargs)
    except Exception as exc:
        return _ineligible([], [], ["all"], chart_type, f"Unhandled dispatch error: {exc}")
    
    # ══════════════════════════════════════════════════════════════════════════════
# PART 3 — SHARED SUPPORT UTILITIES: Lifecycle, Latency, Weighting, Trend & Risk
# ══════════════════════════════════════════════════════════════════════════════

_DURATION_DIVISORS: Dict[str, float] = {"hours": 3600.0, "days": 86400.0, "weeks": 604800.0}

_LIFECYCLE_ACTIVE_TOKENS: frozenset = frozenset(
    {"pending", "open", "active", "in progress", "inprogress"}
)
_LIFECYCLE_CLOSED_TOKENS: frozenset = frozenset(
    {"closed", "completed", "resolved", "finished"}
)
_LIFECYCLE_EXCLUDED_TOKENS: frozenset = frozenset({"cancelled", "canceled", "rejected"})

_SEVERITY_WEIGHT_MAP: Dict[str, float] = {
    "critical": 4.0, "high": 3.0, "severe": 3.0, "major": 3.0,
    "medium": 2.0, "moderate": 2.0, "normal": 2.0,
    "low": 1.0, "minor": 1.0, "trivial": 0.5,
}

_REOPEN_TRUE_TOKENS: frozenset = frozenset(
    {"1", "true", "yes", "y", "reopened", "re-opened", "escalated"}
)

_GRANULARITY_FREQ_MAP: Dict[str, str] = {
    "daily": "D", "weekly": "W", "monthly": "M", "quarterly": "Q", "yearly": "A",
}
_GRANULARITY_DEFAULT_WINDOW: Dict[str, int] = {
    "daily": 30, "weekly": 4, "monthly": 3, "quarterly": 2, "yearly": 2,
}
_GROWTH_METHOD_FREQ_MAP: Dict[str, str] = {"mom": "M", "qoq": "Q", "yoy": "A"}

_RISK_METHOD_ORDER: Tuple[str, ...] = ("enterprise", "advanced", "standard", "minimum")
_ROLE_BUSINESS_CRITICALITY: str = "business_criticality"


def _build_universal_metadata(
    chart_type: str,
    required_roles: List[str],
    optional_roles: List[str],
    supported_domains: List[str],
    recommended_filters: List[str],
    supported_aggregations: List[str],
    supported_time_granularities: List[str],
    export_filename: str,
    title: str,
    status: str = "Eligible",
    reason: str = "",
) -> Dict[str, Any]:
    """Builds the Phase 3 universal export metadata contract shared by all
    Part 3 rendering functions."""
    return {
        "chart_type": chart_type,
        "supported_roles": sorted(set(required_roles) | set(optional_roles)),
        "required_roles": required_roles,
        "optional_roles": optional_roles,
        "supported_domains": supported_domains,
        "recommended_filters": recommended_filters,
        "supported_aggregations": supported_aggregations,
        "supported_time_granularities": supported_time_granularities,
        "export_filename": export_filename,
        "status": status,
        "reason": reason,
        "title": title,
    }


def _universal_ineligible(
    chart_type: str,
    required_roles: List[str],
    optional_roles: List[str],
    supported_domains: List[str],
    recommended_filters: List[str],
    supported_aggregations: List[str],
    supported_time_granularities: List[str],
    export_filename: str,
    reason: str,
) -> ChartReturn:
    """Standard fault-tolerant short-circuit return for Part 3 rendering functions."""
    meta = _build_universal_metadata(
        chart_type, required_roles, optional_roles, supported_domains,
        recommended_filters, supported_aggregations, supported_time_granularities,
        export_filename, title=chart_type, status="Ineligible",
        reason=reason or "Missing required core roles",
    )
    return None, None, meta


def _resolve_lifecycle_state(
    df: pd.DataFrame,
    registry: ColumnRegistry,
) -> Tuple[Optional[pd.Series], Optional[pd.Series], str]:
    """
    Resolves per-row lifecycle state via the mandated priority cascade:
    Priority 1 — Status column, case-insensitive token mapping to Active/Closed,
                 with Cancelled/Rejected rows flagged for exclusion.
    Priority 2 — Closing Date presence/absence (Closed if present, Active if NaT).
    Priority 3 — Unresolved (returns None, None, "unresolved").

    Returns (lifecycle_series, exclude_mask, method). lifecycle_series values are
    'Active' / 'Closed' / NaN (indeterminate — never guessed). exclude_mask is True
    for rows that should be dropped prior to further analysis.
    """
    status_col = _resolve_col(ROLE_STATUS, df, registry)
    if status_col:
        raw = df[status_col].astype(str).str.strip().str.lower()
        exclude_mask = raw.isin(_LIFECYCLE_EXCLUDED_TOKENS)
        lifecycle = pd.Series(np.nan, index=df.index, dtype=object)
        lifecycle = lifecycle.mask(raw.isin(_LIFECYCLE_ACTIVE_TOKENS), "Active")
        lifecycle = lifecycle.mask(raw.isin(_LIFECYCLE_CLOSED_TOKENS), "Closed")
        return lifecycle, exclude_mask, "status_column"

    closing_col = _resolve_col(ROLE_CLOSING_DATE, df, registry)
    if closing_col:
        closing_dt = pd.to_datetime(df[closing_col], errors="coerce")
        lifecycle = pd.Series(
            np.where(closing_dt.notna(), "Closed", "Active"), index=df.index, dtype=object
        )
        exclude_mask = pd.Series(False, index=df.index)
        return lifecycle, exclude_mask, "closing_date_presence"

    return None, None, "unresolved"


def _compute_latency_series(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    lifecycle: pd.Series,
    reference_date: Optional[pd.Timestamp],
    duration_unit: str,
) -> Tuple[Optional[pd.Series], Optional[pd.Timestamp]]:
    """
    Vectorized latency computation. Closed rows: Closing Date − Registration Date.
    Active rows: Reference Date − Registration Date. Indeterminate-lifecycle rows
    (NaN) are left as NaN latency rather than guessed. Reference date resolution:
    explicit override -> max(registration_date) -> pd.Timestamp.now(). Timedeltas
    are always converted via .dt.total_seconds() before unit division — raw
    Timedeltas are never passed downstream.
    """
    reg_col = _resolve_col(ROLE_REGISTRATION_DATE, df, registry)
    if not reg_col:
        return None, None

    # Refactor Phase 3 / Audit Finding B4 remediation — both source date
    # series are tz-normalized immediately after parsing, BEFORE either is
    # ever assigned into end_dt via .mask(). Previously, end_dt was
    # hardcoded to tz-naive dtype ahead of knowing whether close_dt would
    # arrive tz-aware; a tz-aware close_dt then either silently upcast
    # end_dt to object dtype (mixing tz-naive NaT with tz-aware Timestamp
    # values) or raised outright, depending on pandas version, with the
    # failure only surfacing two lines later at `delta = end_dt - reg_dt`.
    reg_dt = _tz_naive_series(pd.to_datetime(df[reg_col], errors="coerce"))
    closing_col = _resolve_col(ROLE_CLOSING_DATE, df, registry)
    close_dt = (
        _tz_naive_series(pd.to_datetime(df[closing_col], errors="coerce"))
        if closing_col else pd.Series(pd.NaT, index=df.index, dtype=reg_dt.dtype)
    )

    reference_date = _tz_naive_scalar(reference_date)
    if reference_date is None or pd.isna(reference_date):
        valid_reg = reg_dt.dropna()
        reference_date = valid_reg.max() if len(valid_reg) else pd.Timestamp.now()
    if pd.isna(reference_date):
        reference_date = pd.Timestamp.now()
    reference_date = _tz_naive_scalar(reference_date)

    is_closed = (lifecycle == "Closed")
    is_active = (lifecycle == "Active")

    # Refactor Phase 3 / Audit Finding B4 remediation — end_dt's dtype is
    # no longer a hardcoded "datetime64[ns]" string literal declared ahead
    # of knowing the source tz-state; it is derived dynamically from the
    # now-guaranteed-tz-naive reg_dt series, so it can never mismatch
    # reg_dt's/close_dt's dtype during the subsequent .mask() assignments.
    end_dt = pd.Series(pd.NaT, index=df.index, dtype=reg_dt.dtype)
    end_dt = end_dt.mask(is_closed, close_dt)
    end_dt = end_dt.mask(is_closed & close_dt.isna(), reference_date)
    end_dt = end_dt.mask(is_active, reference_date)

    delta = end_dt - reg_dt
    divisor = _DURATION_DIVISORS.get(duration_unit, 86400.0)
    latency = delta.dt.total_seconds() / divisor
    latency = latency.where(latency >= 0)
    latency = latency.replace([np.inf, -np.inf], np.nan)
    return latency, reference_date


def _resolve_weight_series(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    role: str,
    default_weight: float = 1.0,
) -> Optional[pd.Series]:
    """Resolves a numeric or categorical weighting role (severity, criticality)
    into a numeric Series. Numeric columns are used directly (NaN/inf guarded).
    Categorical columns are mapped via a severity-token lexicon; unrecognized
    tokens fall back to `default_weight` rather than being dropped or guessed."""
    col = _resolve_col(role, df, registry)
    if not col:
        return None
    raw = df[col]
    if pd.api.types.is_numeric_dtype(raw):
        weight = pd.to_numeric(raw, errors="coerce")
        weight = weight.replace([np.inf, -np.inf], np.nan)
        weight = weight.fillna(default_weight).clip(lower=0.0)
        return weight
    normalized = raw.astype(str).str.strip().str.lower()
    weight = normalized.map(_SEVERITY_WEIGHT_MAP)
    weight = weight.fillna(default_weight)
    return weight


def _resolve_reopen_series(df: pd.DataFrame, registry: ColumnRegistry) -> Optional[pd.Series]:
    """Resolves a boolean 'was reopened' Series from Reopen Flag and/or Status."""
    reopen_col = _resolve_col(ROLE_REOPEN_FLAG, df, registry)
    status_col = _resolve_col(ROLE_STATUS, df, registry)
    from_flag: Optional[pd.Series] = None
    from_status: Optional[pd.Series] = None
    if reopen_col:
        from_flag = df[reopen_col].apply(
            lambda v: str(v).strip().lower() in _REOPEN_TRUE_TOKENS if pd.notna(v) else False
        )
    if status_col:
        from_status = df[status_col].astype(str).str.strip().str.lower().isin(
            {"reopened", "re-opened", "escalated"}
        )
    if from_flag is not None and from_status is not None:
        return from_flag | from_status
    return from_flag if from_flag is not None else from_status


def _apply_outlier_filter(
    series: Optional[pd.Series],
    method: Optional[str],
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
) -> pd.Series:
    """Returns a boolean keep-mask for `series` using IQR / Z-Score / Percentile
    Clipping, entirely controlled via kwargs. NaN values are always kept (never
    treated as outliers) so upstream NaN-handling policy remains authoritative."""
    if series is None or not method or series.empty:
        return pd.Series(True, index=series.index if series is not None else [])
    clean = series.dropna()
    if len(clean) < 4:
        return pd.Series(True, index=series.index)
    method_l = method.strip().lower()
    if method_l == "iqr":
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return pd.Series(True, index=series.index)
        lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
        return series.between(lower, upper) | series.isna()
    if method_l in ("zscore", "z_score", "z"):
        mean, std = clean.mean(), clean.std()
        if not std or np.isnan(std) or std == 0:
            return pd.Series(True, index=series.index)
        z = (series - mean) / std
        return (z.abs() <= z_threshold) | series.isna()
    if method_l in ("percentile", "percentile_clip"):
        lo, hi = clean.quantile(lower_pct), clean.quantile(upper_pct)
        return series.between(lo, hi) | series.isna()
    return pd.Series(True, index=series.index)


def _build_rolling_trend(
    df: pd.DataFrame,
    date_col: str,
    y_col: Optional[str],
    group_col: Optional[str],
    aggregation: str,
    freq: str,
    rolling_window: int,
) -> pd.DataFrame:
    """
    Builds a continuous-period trend table (gaps filled with 0) plus a trailing
    rolling-average column, per the mandated windowing algorithm. Daily uses a
    calendar-day reindex; Weekly/Monthly/Quarterly/Yearly use Period-based
    reindexing. Grouped input produces one raw/rolling column pair per group.
    """
    work = df.copy()
    dt = pd.to_datetime(work[date_col], errors="coerce")
    work = work.loc[dt.notna()].copy()
    dt = dt.loc[dt.notna()]
    if work.empty:
        return pd.DataFrame(columns=["__period", "Raw", "Rolling Average"])

    if freq == "D":
        work["__period"] = dt.dt.normalize()
        full_index = pd.date_range(work["__period"].min(), work["__period"].max(), freq="D")
    else:
        work["__period"] = dt.dt.to_period(freq).dt.to_timestamp()
        full_index = pd.period_range(dt.min(), dt.max(), freq=freq).to_timestamp()

    if y_col is None or aggregation == "count":
        if group_col and group_col in work.columns:
            pivot = work.groupby(["__period", group_col]).size().unstack(fill_value=0)
        else:
            pivot = work.groupby("__period").size().to_frame("Raw")
    else:
        numeric_y = pd.to_numeric(work[y_col], errors="coerce")
        work = work.assign(__val=numeric_y)
        agg = aggregation if aggregation in _VALID_AGGREGATIONS else "sum"
        if group_col and group_col in work.columns:
            pivot = work.groupby(["__period", group_col])["__val"].agg(agg).unstack(fill_value=0)
        else:
            pivot = work.groupby("__period")["__val"].agg(agg).to_frame("Raw")

    pivot = pivot.reindex(full_index, fill_value=0.0)
    pivot.index.name = "__period"

    if "Raw" in pivot.columns:
        rolling = pivot[["Raw"]].rolling(window=rolling_window, min_periods=1).mean()
        rolling.columns = ["Rolling Average"]
        result = pivot.join(rolling).reset_index()
    else:
        rolling = pivot.rolling(window=rolling_window, min_periods=1).mean()
        rolling.columns = [f"{c} (Rolling Avg)" for c in rolling.columns]
        result = pivot.join(rolling).reset_index()

    return result


def _pivot_period_category(
    df: pd.DataFrame,
    date_col: str,
    category_col: str,
    y_col: Optional[str],
    aggregation: str,
    freq: str,
) -> pd.DataFrame:
    """Pivots category (rows) × chronologically-sorted period (columns) for the
    growth heatmap engine. Sorting is performed on Period dtype before the
    columns are relabeled to strings, guaranteeing chronological (not
    lexicographic) column order."""
    dt = pd.to_datetime(df[date_col], errors="coerce")
    work = df.loc[dt.notna()].copy()
    dt = dt.loc[dt.notna()]
    if work.empty:
        return pd.DataFrame()

    work = work.assign(__period=dt.dt.to_period(freq))
    if y_col is None or aggregation == "count":
        pivot = work.groupby([category_col, "__period"], observed=True).size().unstack(fill_value=0)
    else:
        numeric_y = pd.to_numeric(work[y_col], errors="coerce")
        work = work.assign(__val=numeric_y)
        agg = aggregation if aggregation in _VALID_AGGREGATIONS else "sum"
        pivot = (
            work.groupby([category_col, "__period"], observed=True)["__val"]
            .agg(agg).unstack(fill_value=0)
        )
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    pivot.columns = [str(c) for c in pivot.columns]
    return pivot


def _select_risk_method(
    requested: Optional[str],
    has_latency: bool,
    has_severity: bool,
    has_criticality: bool,
    has_reopen: bool,
) -> str:
    """Selects the Risk Engine tier: an explicit valid+available request wins,
    otherwise the highest tier whose required roles are all resolvable."""

    def _available(tier: str) -> bool:
        if tier == "minimum":
            return True
        if tier == "standard":
            return has_latency
        if tier == "advanced":
            return has_latency and has_severity
        if tier == "enterprise":
            return has_latency and has_severity and has_criticality and has_reopen
        return False

    if requested:
        req = requested.strip().lower()
        if req in _RISK_METHOD_ORDER and _available(req):
            return req
    for tier in _RISK_METHOD_ORDER:
        if _available(tier):
            return tier
    return "minimum"


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC RENDER FUNCTIONS — Part 3: Lifecycle, Trend & Risk Intelligence Visuals
# ══════════════════════════════════════════════════════════════════════════════

def render_peak_complaint_hour(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    **kwargs: Any,
) -> ChartReturn:
    """Intraday (00:00–23:00) volume distribution highlighting the peak activity hour."""
    _CHART_TYPE = "Peak Complaint Hour — Intraday Volume Distribution"
    _REQUIRED: List[str] = [ROLE_REGISTRATION_DATE]
    _OPTIONAL: List[str] = [ROLE_RECORD_ID, ROLE_CATEGORY, ROLE_ZONE, ROLE_OFFICER, ROLE_AMOUNT]
    _DOMAINS: List[str] = ["Complaint Management", "Supply Operations", "all"]
    _FILTERS: List[str] = ["date_range", "category", "zone", "officer"]
    _AGGS: List[str] = list(_VALID_AGGREGATIONS)
    _GRANULARITIES: List[str] = ["hourly"]
    _EXPORT_FILE = "peak_complaint_hour.csv"

    try:
        reg_role: str = str(kwargs.get("date_role") or ROLE_REGISTRATION_DATE)
        y_role: Optional[str] = kwargs.get("y_role")
        group_by: Optional[str] = kwargs.get("group_by") or kwargs.get("color_role")
        aggregation: str = str(kwargs.get("aggregation", "count"))
        date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = kwargs.get("date_range")
        theme_key: str = str(kwargs.get("theme_key", DEFAULT_THEME_KEY))
        height: int = int(kwargs.get("height", _DEFAULT_HEIGHT))
        show_values: bool = bool(kwargs.get("show_values", True))
        outlier_method: Optional[str] = kwargs.get("outlier_method")

        theme = _get_theme(theme_key)
        reg_col = _resolve_col(reg_role, df, registry)
        if not reg_col:
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE, f"Registration date role '{reg_role}' could not be resolved to a column.",
            )

        work_df = _apply_date_range(df, reg_col, date_range).copy()
        if work_df.empty:
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE, "No records remain after applying the date filter.",
            )

        reg_dt = pd.to_datetime(work_df[reg_col], errors="coerce")
        valid_mask = reg_dt.notna()
        if int(valid_mask.sum()) == 0:
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE, f"No parseable timestamps found in column '{reg_col}'.",
            )

        work_df = work_df.loc[valid_mask].copy()
        work_df["__hour"] = reg_dt.loc[valid_mask].dt.hour

        y_col = _resolve_col(y_role, work_df, registry) if y_role else None
        if outlier_method and y_col:
            numeric_y = pd.to_numeric(work_df[y_col], errors="coerce")
            keep_mask = _apply_outlier_filter(numeric_y, outlier_method)
            work_df = work_df.loc[keep_mask].copy()
            if work_df.empty:
                return _universal_ineligible(
                    _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                    _EXPORT_FILE, "No records remain after outlier filtering.",
                )

        group_col = _resolve_col(group_by, work_df, registry) if group_by else None
        y_label = _col_label(y_role, registry) if y_role else "Count"
        group_label = _col_label(group_by, registry) if group_by else ""

        agg_df, value_col = _aggregate(work_df, "__hour", y_col, aggregation, group_col, None)

        full_hours = pd.DataFrame({"__hour": list(range(24))})
        if group_col and group_col in agg_df.columns:
            distinct_groups = agg_df[group_col].dropna().unique().tolist()
            if distinct_groups:
                full_idx = pd.MultiIndex.from_product(
                    [range(24), distinct_groups], names=["__hour", group_col]
                )
                agg_df = (
                    agg_df.set_index(["__hour", group_col])
                    .reindex(full_idx, fill_value=0)
                    .reset_index()
                )
            else:
                agg_df = full_hours.merge(agg_df, on="__hour", how="left")
                agg_df[value_col] = agg_df[value_col].fillna(0.0)
        else:
            agg_df = full_hours.merge(agg_df, on="__hour", how="left")
            agg_df[value_col] = agg_df[value_col].fillna(0.0)

        agg_df = agg_df.sort_values("__hour").reset_index(drop=True)
        agg_df["__hour_label"] = agg_df["__hour"].apply(lambda h: f"{int(h):02d}:00")

        hour_totals = agg_df.groupby("__hour_label")[value_col].sum()
        if hour_totals.empty or float(hour_totals.max()) <= 0:
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE, "Hourly aggregation produced no non-zero values.",
            )
        peak_hour_label = str(hour_totals.idxmax())
        peak_value = float(hour_totals.max())

        title: str = str(kwargs.get("title") or f"Peak Complaint Hour Analysis — Intraday {y_label} Distribution")

        if group_col and group_col in agg_df.columns:
            n_groups = agg_df[group_col].nunique()
            color_seq = _color_seq(n_groups)
            fig = px.bar(
                agg_df, x="__hour_label", y=value_col, color=group_col,
                color_discrete_sequence=color_seq, template=theme["plotly_template"],
            )
        else:
            bar_colors = np.where(
                agg_df["__hour_label"] == peak_hour_label, theme["danger"], theme["primary"]
            ).tolist()
            fig = go.Figure(data=go.Bar(
                x=agg_df["__hour_label"], y=agg_df[value_col],
                marker_color=bar_colors, marker_line_width=0,
                hovertemplate="Hour: %{x}<br>Value: %{y:,.1f}<extra></extra>",
            ))

        if show_values:
            fig.update_traces(
                texttemplate="%{y:,.0f}", textposition="outside",
                textfont=dict(size=9, family=_FONT_FAMILY),
            )

        fig.add_annotation(
            x=peak_hour_label, y=peak_value, text=f"Peak: {peak_hour_label}",
            showarrow=True, arrowhead=2, arrowcolor=theme["danger"], ax=0, ay=-28,
            font=dict(size=10, color=theme["danger"], family=_FONT_FAMILY),
        )

        _apply_layout(
            fig, "" , theme, height=height,
            xaxis_title="Hour of Day", yaxis_title=y_label, showlegend=bool(group_col),
        )
        fig.update_xaxes(tickangle=-45)

        export_df = agg_df.drop(columns=["__hour"]).rename(
            columns={"__hour_label": "Hour", value_col: (y_label if y_label != "Count" else "Count")}
        )
        if group_col and group_col in export_df.columns and group_label:
            export_df = export_df.rename(columns={group_col: group_label})

        meta = _build_universal_metadata(
            _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
            _EXPORT_FILE, title=title, status="Eligible",
            reason=f"Peak intraday activity detected at {peak_hour_label} "
                   f"({peak_value:,.0f} {y_label.lower()}).",
        )
        return fig, export_df, meta

    except Exception as e:
        return _universal_ineligible(
            "Peak Complaint Hour — Intraday Volume Distribution",
            [ROLE_REGISTRATION_DATE],
            [ROLE_RECORD_ID, ROLE_CATEGORY, ROLE_ZONE, ROLE_OFFICER, ROLE_AMOUNT],
            ["Complaint Management", "Supply Operations", "all"],
            ["date_range", "category", "zone", "officer"],
            list(_VALID_AGGREGATIONS), ["hourly"], "peak_complaint_hour.csv",
            str(e) or "Missing required core roles",
        )


def render_monthly_rolling_average(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    **kwargs: Any,
) -> ChartReturn:
    """Raw volume overlaid with a trailing rolling-average trend line, across a
    configurable granularity (daily/weekly/monthly/quarterly/yearly)."""
    _CHART_TYPE = "Rolling Average Trend — Complaint Volume Momentum"
    _REQUIRED: List[str] = [ROLE_REGISTRATION_DATE]
    _OPTIONAL: List[str] = [ROLE_RECORD_ID, ROLE_AMOUNT, ROLE_STATUS, ROLE_CATEGORY, ROLE_ZONE]
    _DOMAINS: List[str] = ["Complaint Management", "Revenue Analytics", "Supply Operations", "all"]
    _FILTERS: List[str] = ["date_range", "category", "zone", "status"]
    _AGGS: List[str] = list(_VALID_AGGREGATIONS)
    _GRANULARITIES: List[str] = ["daily", "weekly", "monthly", "quarterly", "yearly"]
    _EXPORT_FILE = "rolling_average_trend.csv"

    try:
        date_role: str = str(kwargs.get("date_role") or ROLE_REGISTRATION_DATE)
        y_role: Optional[str] = kwargs.get("y_role")
        group_by: Optional[str] = kwargs.get("group_by") or kwargs.get("color_role")
        aggregation: str = str(kwargs.get("aggregation", "count"))
        aggregation_level: str = str(kwargs.get("aggregation_level", "monthly")).lower()
        date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = kwargs.get("date_range")
        theme_key: str = str(kwargs.get("theme_key", DEFAULT_THEME_KEY))
        height: int = int(kwargs.get("height", _DEFAULT_HEIGHT))
        outlier_method: Optional[str] = kwargs.get("outlier_method")

        if aggregation_level not in _GRANULARITY_FREQ_MAP:
            aggregation_level = "monthly"
        freq = _GRANULARITY_FREQ_MAP[aggregation_level]
        rolling_window: int = max(int(kwargs.get("rolling_window", _GRANULARITY_DEFAULT_WINDOW[aggregation_level])), 1)

        theme = _get_theme(theme_key)
        date_col = _resolve_col(date_role, df, registry)
        if not date_col:
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE, f"Registration date role '{date_role}' could not be resolved.",
            )

        work_df = _apply_date_range(df, date_col, date_range).copy()
        if work_df.empty:
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE, "No records remain after applying the date filter.",
            )

        y_col = _resolve_col(y_role, work_df, registry) if y_role else None
        group_col = _resolve_col(group_by, work_df, registry) if group_by else None
        y_label = _col_label(y_role, registry) if y_role else "Count"

        if outlier_method and y_col:
            numeric_y = pd.to_numeric(work_df[y_col], errors="coerce")
            keep_mask = _apply_outlier_filter(numeric_y, outlier_method)
            work_df = work_df.loc[keep_mask].copy()
            if work_df.empty:
                return _universal_ineligible(
                    _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                    _EXPORT_FILE, "No records remain after outlier filtering.",
                )

        trend_df = _build_rolling_trend(work_df, date_col, y_col, group_col, aggregation, freq, rolling_window)
        if trend_df.empty:
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE, "Unable to build a continuous time series from the resolved date column.",
            )

        title: str = str(
            kwargs.get("title")
            or f"{y_label} Trend — {aggregation_level.title()} with {rolling_window}-Period Rolling Average"
        )
        use_secondary = (aggregation_level == "daily")
        fig = make_subplots(specs=[[{"secondary_y": use_secondary}]])

        if group_col and group_col in work_df.columns:
            raw_cols = [c for c in trend_df.columns if c != "__period" and not str(c).endswith("(Rolling Avg)")]
            roll_cols = [c for c in trend_df.columns if str(c).endswith("(Rolling Avg)")]
            color_seq = _color_seq(len(raw_cols))
            for i, rc in enumerate(raw_cols):
                fig.add_trace(
                    go.Bar(x=trend_df["__period"], y=trend_df[rc], name=str(rc),
                          marker_color=color_seq[i % len(color_seq)], opacity=0.55, marker_line_width=0),
                    secondary_y=False,
                )
            for i, gc in enumerate(roll_cols):
                base_name = gc.replace(" (Rolling Avg)", "")
                fig.add_trace(
                    go.Scatter(x=trend_df["__period"], y=trend_df[gc], name=f"{base_name} — Rolling Avg",
                              mode="lines", line=dict(width=2.2, shape="spline", color=color_seq[i % len(color_seq)])),
                    secondary_y=use_secondary,
                )
            fig.update_layout(barmode="overlay")
        else:
            fig.add_trace(
                go.Bar(x=trend_df["__period"], y=trend_df["Raw"], name=y_label,
                      marker_color=theme["secondary"], opacity=0.45, marker_line_width=0,
                      hovertemplate="%{x}<br>Raw: %{y:,.1f}<extra></extra>"),
                secondary_y=False,
            )
            fig.add_trace(
                go.Scatter(x=trend_df["__period"], y=trend_df["Rolling Average"],
                          name=f"{rolling_window}-Period Rolling Avg", mode="lines",
                          line=dict(color=theme["primary"], width=2.4, shape="spline"),
                          hovertemplate="%{x}<br>Rolling Avg: %{y:,.2f}<extra></extra>"),
                secondary_y=use_secondary,
            )

        fig.update_layout(
            title="",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family=_FONT_FAMILY, color=theme["text"], size=12), height=height,
            legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=11, family=_FONT_FAMILY, color=theme["text"]),
                       orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0),
            margin={"l": 20, "r": 40, "t": 64, "b": 40},
            hoverlabel=dict(bgcolor=theme["surface"], font_size=12, font_family=_FONT_FAMILY,
                           bordercolor=theme["secondary"]),
        )
        fig.update_xaxes(gridcolor="rgba(100,116,139,0.12)", linecolor="rgba(100,116,139,0.25)",
                        tickfont=dict(size=10, color=theme["secondary"]), tickangle=-30, title_text="Period")
        fig.update_yaxes(title_text=y_label, secondary_y=False, gridcolor="rgba(100,116,139,0.12)",
                        tickfont=dict(size=11, color=theme["secondary"]))
        if use_secondary:
            fig.update_yaxes(title_text=f"{rolling_window}-Period Rolling Avg", secondary_y=True,
                            showgrid=False, tickfont=dict(size=11, color=theme["primary"]))

        export_df = trend_df.rename(columns={"__period": "Period"})
        meta = _build_universal_metadata(
            _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
            _EXPORT_FILE, title=title, status="Eligible",
            reason=f"Rolling {rolling_window}-period average computed at {aggregation_level} granularity.",
        )
        return fig, export_df, meta

    except Exception as e:
        return _universal_ineligible(
            "Rolling Average Trend — Complaint Volume Momentum",
            [ROLE_REGISTRATION_DATE],
            [ROLE_RECORD_ID, ROLE_AMOUNT, ROLE_STATUS, ROLE_CATEGORY, ROLE_ZONE],
            ["Complaint Management", "Revenue Analytics", "Supply Operations", "all"],
            ["date_range", "category", "zone", "status"],
            list(_VALID_AGGREGATIONS), ["daily", "weekly", "monthly", "quarterly", "yearly"],
            "rolling_average_trend.csv", str(e) or "Missing required core roles",
        )


def render_complaint_growth_heatmap(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    **kwargs: Any,
) -> ChartReturn:
    """Category × Period growth heatmap (MoM / QoQ / YoY / Rolling Growth), with
    division-by-zero-safe percentage-change vectorization."""
    _CHART_TYPE = "Complaint Growth Heatmap — Period-over-Period % Change"
    _REQUIRED: List[str] = [ROLE_REGISTRATION_DATE, ROLE_CATEGORY]
    _OPTIONAL: List[str] = [ROLE_RECORD_ID, ROLE_AMOUNT, ROLE_ZONE]
    _DOMAINS: List[str] = ["Complaint Management", "Revenue Analytics", "all"]
    _FILTERS: List[str] = ["date_range", "zone", "category"]
    _AGGS: List[str] = list(_VALID_AGGREGATIONS)
    _GRANULARITIES: List[str] = ["MoM", "QoQ", "YoY", "Rolling Growth"]
    _EXPORT_FILE = "complaint_growth_heatmap.csv"

    try:
        date_role: str = str(kwargs.get("date_role") or ROLE_REGISTRATION_DATE)
        category_role: str = str(kwargs.get("category_role") or kwargs.get("y_role") or ROLE_CATEGORY)
        y_role: Optional[str] = kwargs.get("value_role")
        aggregation: str = str(kwargs.get("aggregation", "count"))
        growth_method: str = str(kwargs.get("growth_method", "MoM"))
        rolling_window: int = max(int(kwargs.get("rolling_window", 3)), 1)
        top_n: int = int(kwargs.get("top_n", 15))
        date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = kwargs.get("date_range")
        theme_key: str = str(kwargs.get("theme_key", DEFAULT_THEME_KEY))
        height: int = int(kwargs.get("height", 520))

        theme = _get_theme(theme_key)
        date_col = _resolve_col(date_role, df, registry)
        category_col = _resolve_col(category_role, df, registry)
        if not date_col or not category_col:
            missing = [registry.display_name(r) for r, c in
                      ((date_role, date_col), (category_role, category_col)) if not c]
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE, f"Missing required role mapping(s): {', '.join(missing)}.",
            )

        work_df = _apply_date_range(df, date_col, date_range).copy()
        if work_df.empty:
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE, "No records remain after applying the date filter.",
            )

        if top_n > 0:
            top_categories = work_df[category_col].value_counts().nlargest(top_n).index
            work_df = work_df[work_df[category_col].isin(top_categories)].copy()

        y_col = _resolve_col(y_role, work_df, registry) if y_role else None
        y_label = _col_label(y_role, registry) if y_role else "Count"

        method_key = growth_method.strip().lower().replace(" ", "").replace("-", "")

        if method_key in _GROWTH_METHOD_FREQ_MAP:
            freq = _GROWTH_METHOD_FREQ_MAP[method_key]
            pivot = _pivot_period_category(work_df, date_col, category_col, y_col, aggregation, freq)
            if pivot.empty or pivot.shape[1] < 2:
                return _universal_ineligible(
                    _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                    _EXPORT_FILE, f"Fewer than 2 periods available for {growth_method} growth calculation.",
                )
            growth = (pivot.T.pct_change().T * 100.0).replace([np.inf, -np.inf], np.nan)
            growth = growth.iloc[:, 1:]
        elif method_key in ("rollinggrowth", "rolling"):
            pivot = _pivot_period_category(work_df, date_col, category_col, y_col, aggregation, "M")
            if pivot.empty or pivot.shape[1] < 2:
                return _universal_ineligible(
                    _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                    _EXPORT_FILE, "Fewer than 2 monthly periods available for rolling growth calculation.",
                )
            baseline = pivot.T.shift(1).rolling(window=rolling_window, min_periods=1).mean().T
            growth = ((pivot - baseline) / baseline.replace(0, np.nan) * 100.0)
            growth = growth.replace([np.inf, -np.inf], np.nan).iloc[:, 1:]
        else:
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE,
                f"Unsupported growth_method '{growth_method}'. Supported: MoM, QoQ, YoY, Rolling Growth.",
            )

        if growth.empty or growth.dropna(how="all").empty:
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE, "Growth matrix contains no computable values (insufficient prior-period baselines).",
            )

        category_label = _col_label(category_role, registry)
        title: str = str(kwargs.get("title") or f"{growth_method} Growth Heatmap — {y_label} by {category_label}")

        z_values = growth.values
        x_labels = [str(c) for c in growth.columns]
        y_labels = [str(r) for r in growth.index]

        finite_abs = np.abs(z_values[np.isfinite(z_values)])
        max_abs = float(finite_abs.max()) if finite_abs.size else 1.0
        max_abs = max_abs if max_abs > 0 else 1.0

        show_text = z_values.size <= 250
        text_matrix = (
            [[f"{v:+.1f}%" if not np.isnan(v) else "—" for v in row] for row in z_values]
            if show_text else None
        )

        fig = go.Figure(data=go.Heatmap(
            z=z_values, x=x_labels, y=y_labels,
            colorscale="RdBu_r", zmid=0.0, zmin=-max_abs, zmax=max_abs,
            text=text_matrix, texttemplate="%{text}" if show_text else None,
            textfont=dict(size=9, family=_FONT_FAMILY),
            hovertemplate="<b>%{y}</b> — %{x}<br>Growth: %{z:.1f}%<extra></extra>",
            colorbar=dict(
                thickness=14, len=0.85, ticksuffix="%",
                tickfont=dict(size=10, family=_FONT_FAMILY, color=theme["text"]),
                title=dict(text="Growth %", font=dict(size=11, family=_FONT_FAMILY, color=theme["secondary"])),
            ),
        ))

        _apply_layout(fig, "", theme, height=height, showlegend=False,
                     margin={"l": 120, "r": 20, "t": 64, "b": 70})
        fig.update_xaxes(tickangle=-40, tickfont=dict(size=10))
        fig.update_yaxes(tickfont=dict(size=10))

        export_df = growth.reset_index().rename(columns={category_col: category_label})
        meta = _build_universal_metadata(
            _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
            _EXPORT_FILE, title=title, status="Eligible",
            reason=f"{growth_method} growth computed across {len(x_labels)} period(s) "
                   f"for {len(y_labels)} categor{'y' if len(y_labels) == 1 else 'ies'}.",
        )
        return fig, export_df, meta

    except Exception as e:
        return _universal_ineligible(
            "Complaint Growth Heatmap — Period-over-Period % Change",
            [ROLE_REGISTRATION_DATE, ROLE_CATEGORY],
            [ROLE_RECORD_ID, ROLE_AMOUNT, ROLE_ZONE],
            ["Complaint Management", "Revenue Analytics", "all"],
            ["date_range", "zone", "category"],
            list(_VALID_AGGREGATIONS), ["MoM", "QoQ", "YoY", "Rolling Growth"],
            "complaint_growth_heatmap.csv", str(e) or "Missing required core roles",
        )


def render_officer_performance_matrix(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    **kwargs: Any,
) -> ChartReturn:
    """Multi-metric productivity heatmap (Backlog, Resolved, Resolution Time,
    SLA Compliance, Reopen Rate, Workload, Throughput) per group-by dimension.
    Metrics whose required roles are unavailable are silently omitted rather
    than fabricated."""
    _CHART_TYPE = "Officer Performance Matrix — Multi-Metric Productivity Heatmap"
    _REQUIRED: List[str] = [ROLE_OFFICER]
    _OPTIONAL: List[str] = [
        ROLE_RECORD_ID, ROLE_STATUS, ROLE_CLOSING_DATE, ROLE_REGISTRATION_DATE,
        ROLE_SLA_DEADLINE, ROLE_REOPEN_FLAG,
    ]
    _DOMAINS: List[str] = ["Complaint Management", "Supply Operations", "all"]
    _FILTERS: List[str] = ["date_range", "officer", "zone", "category"]
    _AGGS: List[str] = ["count", "mean", "median"]
    _GRANULARITIES: List[str] = ["overall"]
    _EXPORT_FILE = "officer_performance_matrix.csv"

    try:
        group_role: str = str(kwargs.get("group_by") or ROLE_OFFICER)
        duration_unit: str = str(kwargs.get("duration_unit", "days")).lower()
        analysis_date_raw: Any = kwargs.get("analysis_date")
        date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = kwargs.get("date_range")
        theme_key: str = str(kwargs.get("theme_key", DEFAULT_THEME_KEY))
        height: int = int(kwargs.get("height", 520))
        top_n: int = int(kwargs.get("top_n", 20))
        sort_metric: str = str(kwargs.get("sort_metric", "Total Cases"))

        theme = _get_theme(theme_key)
        group_col = _resolve_col(group_role, df, registry)
        if not group_col:
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE, f"Group-by role '{group_role}' could not be resolved. "
                              "Map Officer (or an equivalent role) via group_by.",
            )

        reg_col = _resolve_col(ROLE_REGISTRATION_DATE, df, registry)
        work_df = _apply_date_range(df, reg_col, date_range).copy()
        if work_df.empty:
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE, "No records remain after applying the date filter.",
            )

        lifecycle, exclude_mask, lifecycle_method = _resolve_lifecycle_state(work_df, registry)
        if exclude_mask is not None:
            keep = ~exclude_mask.reindex(work_df.index).fillna(False)
            work_df = work_df.loc[keep].copy()
            if lifecycle is not None:
                lifecycle = lifecycle.reindex(work_df.index)

        analysis_date: Optional[pd.Timestamp] = None
        if analysis_date_raw is not None:
            analysis_date = pd.to_datetime(analysis_date_raw, errors="coerce")

        latency_days: Optional[pd.Series] = None
        if lifecycle is not None and reg_col:
            latency_days, _ref = _compute_latency_series(work_df, registry, lifecycle, analysis_date, duration_unit)

        sla_col = _resolve_col(ROLE_SLA_DEADLINE, work_df, registry)
        closing_col = _resolve_col(ROLE_CLOSING_DATE, work_df, registry)
        reopen_col = _resolve_col(ROLE_REOPEN_FLAG, work_df, registry)

        grouped = work_df.groupby(group_col, dropna=True)
        matrix = pd.DataFrame(index=grouped.size().index)
        matrix["Total Cases"] = grouped.size()
        available_metrics: List[str] = ["Total Cases"]

        if lifecycle is not None:
            work_df = work_df.assign(__lifecycle=lifecycle)
            resolved_mask = work_df["__lifecycle"] == "Closed"
            active_mask = work_df["__lifecycle"] == "Active"
            matrix["Resolved Count"] = (
                work_df.loc[resolved_mask].groupby(group_col).size().reindex(matrix.index, fill_value=0)
            )
            matrix["Backlog"] = (
                work_df.loc[active_mask].groupby(group_col).size().reindex(matrix.index, fill_value=0)
            )
            matrix["Workload"] = matrix["Backlog"]
            matrix["Throughput (%)"] = np.where(
                matrix["Total Cases"] > 0,
                matrix["Resolved Count"] / matrix["Total Cases"].replace(0, np.nan) * 100.0,
                0.0,
            )
            matrix["Throughput (%)"] = matrix["Throughput (%)"].fillna(0.0).round(2)
            available_metrics += ["Resolved Count", "Backlog", "Workload", "Throughput (%)"]

            if latency_days is not None:
                work_df = work_df.assign(__latency=latency_days)
                resolved_latency = work_df.loc[resolved_mask].groupby(group_col)["__latency"]
                matrix[f"Avg Resolution ({duration_unit})"] = resolved_latency.mean().reindex(matrix.index).round(2)
                matrix[f"Median Resolution ({duration_unit})"] = resolved_latency.median().reindex(matrix.index).round(2)
                available_metrics += [f"Avg Resolution ({duration_unit})", f"Median Resolution ({duration_unit})"]

            if sla_col and closing_col:
                sla_dt = pd.to_datetime(work_df[sla_col], errors="coerce")
                close_dt = pd.to_datetime(work_df[closing_col], errors="coerce")
                valid_sla_mask = resolved_mask & sla_dt.notna() & close_dt.notna()
                compliant_mask = valid_sla_mask & (close_dt <= sla_dt)
                resolved_valid_totals = work_df.loc[valid_sla_mask].groupby(group_col).size().reindex(matrix.index, fill_value=0)
                compliant_totals = work_df.loc[compliant_mask].groupby(group_col).size().reindex(matrix.index, fill_value=0)
                matrix["SLA Compliance (%)"] = np.where(
                    resolved_valid_totals > 0,
                    compliant_totals / resolved_valid_totals.replace(0, np.nan) * 100.0, np.nan,
                )
                matrix["SLA Compliance (%)"] = matrix["SLA Compliance (%)"].round(2)
                available_metrics.append("SLA Compliance (%)")

            if reopen_col:
                reopen_flag_series = work_df[reopen_col].apply(
                    lambda v: str(v).strip().lower() in _REOPEN_TRUE_TOKENS if pd.notna(v) else False
                )
                work_df = work_df.assign(__reopened=reopen_flag_series)
                reopened_totals = work_df.loc[work_df["__reopened"]].groupby(group_col).size().reindex(matrix.index, fill_value=0)
                matrix["Reopen Rate (%)"] = np.where(
                    matrix["Total Cases"] > 0,
                    reopened_totals / matrix["Total Cases"].replace(0, np.nan) * 100.0, 0.0,
                )
                matrix["Reopen Rate (%)"] = matrix["Reopen Rate (%)"].fillna(0.0).round(2)
                available_metrics.append("Reopen Rate (%)")

        matrix = matrix.replace([np.inf, -np.inf], np.nan)
        matrix.index = matrix.index.astype(str)
        matrix.index.name = registry.display_name(group_role)

        sort_col = sort_metric if sort_metric in matrix.columns else "Total Cases"
        matrix = matrix.sort_values(sort_col, ascending=False)
        if top_n > 0:
            matrix = matrix.head(top_n)
        if matrix.empty:
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE, "No officer groups remained after aggregation.",
            )

        heat_cols = [c for c in available_metrics if c in matrix.columns and c != "Total Cases"] or ["Total Cases"]

        normalized = matrix[heat_cols].copy()
        for col in heat_cols:
            series = normalized[col]
            valid = series.dropna()
            if len(valid) < 2 or valid.max() == valid.min():
                normalized[col] = 50.0
            else:
                normalized[col] = (series - valid.min()) / (valid.max() - valid.min()) * 100.0
        normalized = normalized.fillna(0.0)

        z_values = normalized.values
        text_matrix = [
            [f"{matrix[col].iloc[i]:,.1f}" if pd.notna(matrix[col].iloc[i]) else "—" for col in heat_cols]
            for i in range(len(matrix))
        ]

        title: str = str(
            kwargs.get("title")
            or f"Officer Performance Matrix — {registry.display_name(group_role)} Productivity Overview"
        )

        fig = go.Figure(data=go.Heatmap(
            z=z_values, x=heat_cols, y=matrix.index.tolist(),
            colorscale=_SEQUENTIAL_BLUES, zmin=0, zmax=100,
            text=text_matrix, texttemplate="%{text}",
            textfont=dict(size=9, family=_FONT_FAMILY),
            hovertemplate="<b>%{y}</b> — %{x}<br>Raw Value: %{text}<br>Normalized: %{z:.0f}<extra></extra>",
            colorbar=dict(
                thickness=14, len=0.85,
                title=dict(text="Percentile Score", font=dict(size=10, family=_FONT_FAMILY)),
                tickfont=dict(size=10, family=_FONT_FAMILY, color=theme["text"]),
            ),
        ))

        dynamic_height = max(height, 32 * len(matrix) + 120)
        _apply_layout(fig, "" , theme, height=dynamic_height, showlegend=False,
                     margin={"l": 140, "r": 20, "t": 64, "b": 90})
        fig.update_xaxes(tickangle=-30, tickfont=dict(size=10))
        fig.update_yaxes(tickfont=dict(size=10), autorange="reversed")

        export_df = matrix.reset_index()
        meta = _build_universal_metadata(
            _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
            _EXPORT_FILE, title=title, status="Eligible",
            reason=f"Performance matrix computed for {len(matrix)} group(s) across {len(heat_cols)} metric(s) "
                   f"(lifecycle resolved via {lifecycle_method}).",
        )
        return fig, export_df, meta

    except Exception as e:
        return _universal_ineligible(
            "Officer Performance Matrix — Multi-Metric Productivity Heatmap",
            [ROLE_OFFICER],
            [ROLE_RECORD_ID, ROLE_STATUS, ROLE_CLOSING_DATE, ROLE_REGISTRATION_DATE,
             ROLE_SLA_DEADLINE, ROLE_REOPEN_FLAG],
            ["Complaint Management", "Supply Operations", "all"],
            ["date_range", "officer", "zone", "category"],
            ["count", "mean", "median"], ["overall"],
            "officer_performance_matrix.csv", str(e) or "Missing required core roles",
        )


def render_pending_age_distribution(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    **kwargs: Any,
) -> ChartReturn:
    """Distribution (histogram or box) of currently pending/active record ages,
    with mean/median reference markers and optional outlier filtering."""
    _CHART_TYPE = "Pending Age Distribution — Backlog Aging Profile"
    _REQUIRED: List[str] = [ROLE_REGISTRATION_DATE]
    _OPTIONAL: List[str] = [ROLE_STATUS, ROLE_CLOSING_DATE, ROLE_CATEGORY, ROLE_ZONE, ROLE_OFFICER]
    _DOMAINS: List[str] = ["Complaint Management", "Supply Operations", "all"]
    _FILTERS: List[str] = ["date_range", "zone", "category", "officer"]
    _AGGS: List[str] = ["mean", "median", "count"]
    _GRANULARITIES: List[str] = ["overall"]
    _EXPORT_FILE = "pending_age_distribution.csv"

    try:
        duration_unit: str = str(kwargs.get("duration_unit", "days")).lower()
        analysis_date_raw: Any = kwargs.get("analysis_date")
        group_by: Optional[str] = kwargs.get("group_by") or kwargs.get("color_role")
        date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = kwargs.get("date_range")
        theme_key: str = str(kwargs.get("theme_key", DEFAULT_THEME_KEY))
        height: int = int(kwargs.get("height", _DEFAULT_HEIGHT))
        chart_style: str = str(kwargs.get("chart_style", "histogram")).lower()
        outlier_method: Optional[str] = kwargs.get("outlier_method")
        nbins: int = int(kwargs.get("nbins", 0))

        theme = _get_theme(theme_key)
        reg_col = _resolve_col(ROLE_REGISTRATION_DATE, df, registry)
        if not reg_col:
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE, "Registration date role could not be resolved.",
            )

        work_df = _apply_date_range(df, reg_col, date_range).copy()
        if work_df.empty:
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE, "No records remain after applying the date filter.",
            )

        lifecycle, exclude_mask, lifecycle_method = _resolve_lifecycle_state(work_df, registry)
        if lifecycle is None:
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE, "Unable to determine lifecycle state — map a Status or Closing Date role.",
            )
        if exclude_mask is not None:
            keep = ~exclude_mask.reindex(work_df.index).fillna(False)
            work_df = work_df.loc[keep].copy()
            lifecycle = lifecycle.reindex(work_df.index)

        active_mask = (lifecycle == "Active")
        pending_df = work_df.loc[active_mask].copy()
        if pending_df.empty:
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE, "No currently pending/active records found for age-distribution analysis.",
            )

        analysis_date: Optional[pd.Timestamp] = None
        if analysis_date_raw is not None:
            analysis_date = pd.to_datetime(analysis_date_raw, errors="coerce")

        pending_lifecycle = pd.Series("Active", index=pending_df.index)
        latency_days, _ref = _compute_latency_series(
            pending_df, registry, pending_lifecycle, analysis_date, duration_unit
        )
        if latency_days is None:
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE, "Registration date role could not be resolved for latency computation.",
            )

        pending_df = pending_df.assign(__age=latency_days).dropna(subset=["__age"])
        if pending_df.empty:
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE, "No valid pending-age values could be computed (registration dates invalid or in future).",
            )

        if outlier_method:
            keep_mask = _apply_outlier_filter(pending_df["__age"], outlier_method)
            pending_df = pending_df.loc[keep_mask].copy()
            if pending_df.empty:
                return _universal_ineligible(
                    _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                    _EXPORT_FILE, "No records remain after outlier filtering.",
                )

        group_col = _resolve_col(group_by, pending_df, registry) if group_by else None
        group_label = _col_label(group_by, registry) if group_by else ""
        unit_label = duration_unit if duration_unit in _DURATION_DIVISORS else "days"

        title: str = str(kwargs.get("title") or f"Pending Age Distribution — Current Backlog ({unit_label.title()})")
        n_colors = pending_df[group_col].nunique() if group_col else 1
        color_seq = _color_seq(n_colors)

        if chart_style == "box":
            fig = px.box(
                pending_df, x=group_col, y="__age", color=group_col, points="outliers",
                color_discrete_sequence=color_seq, template=theme["plotly_template"],
            )
            fig.update_traces(marker_size=3.5, line_width=1.5, boxmean="sd")
            _apply_layout(fig, "" , theme, height=height, xaxis_title=group_label,
                         yaxis_title=f"Pending Age ({unit_label})", showlegend=bool(group_col))
        else:
            hist_kwargs: Dict[str, Any] = dict(
                data_frame=pending_df, x="__age", color=group_col, opacity=0.80,
                color_discrete_sequence=color_seq, template=theme["plotly_template"], barmode="overlay",
            )
            if nbins > 0:
                hist_kwargs["nbins"] = nbins
            fig = px.histogram(**hist_kwargs)
            fig.update_traces(marker_line_width=0.5, marker_line_color=theme["surface"])
            mean_age = float(pending_df["__age"].mean())
            median_age = float(pending_df["__age"].median())
            fig.add_vline(x=mean_age, line_dash="dash", line_color=theme["danger"], line_width=1.5,
                         annotation_text=f"Mean: {mean_age:,.1f}",
                         annotation_font=dict(size=10, color=theme["danger"], family=_FONT_FAMILY))
            fig.add_vline(x=median_age, line_dash="dot", line_color=theme["primary"], line_width=1.5,
                         annotation_text=f"Median: {median_age:,.1f}", annotation_position="bottom right",
                         annotation_font=dict(size=10, color=theme["primary"], family=_FONT_FAMILY))
            _apply_layout(fig, "" , theme, height=height, xaxis_title=f"Pending Age ({unit_label})",
                         yaxis_title="Frequency", showlegend=bool(group_col))

        if group_col:
            group_stats = pending_df.groupby(group_col)["__age"].describe().round(2).reset_index()
            if group_label and group_col in group_stats.columns:
                group_stats = group_stats.rename(columns={group_col: group_label})
            export_df = group_stats
        else:
            desc = pending_df["__age"].describe().round(2)
            export_df = pd.DataFrame({"Statistic": desc.index, f"Pending Age ({unit_label})": desc.values})

        meta = _build_universal_metadata(
            _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
            _EXPORT_FILE, title=title, status="Eligible",
            reason=f"Pending-age distribution computed for {len(pending_df):,} active record(s) "
                   f"(lifecycle resolved via {lifecycle_method}).",
        )
        return fig, export_df, meta

    except Exception as e:
        return _universal_ineligible(
            "Pending Age Distribution — Backlog Aging Profile",
            [ROLE_REGISTRATION_DATE],
            [ROLE_STATUS, ROLE_CLOSING_DATE, ROLE_CATEGORY, ROLE_ZONE, ROLE_OFFICER],
            ["Complaint Management", "Supply Operations", "all"],
            ["date_range", "zone", "category", "officer"],
            ["mean", "median", "count"], ["overall"],
            "pending_age_distribution.csv", str(e) or "Missing required core roles",
        )


def render_reopened_complaint_analysis(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    **kwargs: Any,
) -> ChartReturn:
    """Reopen rate by dimension (category/officer/zone), division-by-zero-safe,
    with an overall-rate reference line."""
    _CHART_TYPE = "Reopened Complaint Analysis — Reopen Rate by Dimension"
    _REQUIRED: List[str] = []
    _OPTIONAL: List[str] = [ROLE_REOPEN_FLAG, ROLE_STATUS, ROLE_CATEGORY, ROLE_OFFICER, ROLE_ZONE, ROLE_RECORD_ID, ROLE_SUBSTATION ]
    _DOMAINS: List[str] = ["Complaint Management", "all"]
    _FILTERS: List[str] = ["date_range", "category", "officer", "zone","substation", "RECORD_ID"]
    _AGGS: List[str] = ["count"]
    _GRANULARITIES: List[str] = ["overall"]
    _EXPORT_FILE = "reopened_complaint_analysis.csv"

    try:
        group_by: str = str(kwargs.get("group_by") or kwargs.get("x_role") or ROLE_CATEGORY)
        top_n: int = int(kwargs.get("top_n", 15))
        min_group_size: int = int(kwargs.get("min_group_size", 3))
        date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = kwargs.get("date_range")
        date_role: str = str(kwargs.get("date_role") or ROLE_REGISTRATION_DATE)
        theme_key: str = str(kwargs.get("theme_key", DEFAULT_THEME_KEY))
        height: int = int(kwargs.get("height", _DEFAULT_HEIGHT))
        sort_order: str = str(kwargs.get("sort_order", "desc"))

        theme = _get_theme(theme_key)
        date_col = _resolve_col(date_role, df, registry)
        work_df = _apply_date_range(df, date_col, date_range).copy()
        if work_df.empty:
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE, "No records remain after applying the date filter.",
            )

        reopen_series = _resolve_reopen_series(work_df, registry)
        if reopen_series is None:
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE, "Unable to detect reopened complaints — map a Reopen Flag or Status role.",
            )

        group_col = _resolve_col(group_by, work_df, registry)
        if not group_col:
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE, f"group_by role '{group_by}' could not be resolved to a column.",
            )

        work_df = work_df.assign(__reopened=reopen_series.reindex(work_df.index).fillna(False))

        totals = work_df.groupby(group_col).size()
        reopened_totals = work_df.loc[work_df["__reopened"]].groupby(group_col).size().reindex(totals.index, fill_value=0)

        summary = pd.DataFrame({"Total Cases": totals, "Reopened Cases": reopened_totals})
        summary["Reopen Rate (%)"] = np.where(
            summary["Total Cases"] > 0,
            summary["Reopened Cases"] / summary["Total Cases"].replace(0, np.nan) * 100.0, 0.0,
        )
        summary["Reopen Rate (%)"] = summary["Reopen Rate (%)"].fillna(0.0).round(2)
        summary = summary.loc[summary["Total Cases"] >= max(min_group_size, 1)]
        summary.index = summary.index.astype(str)
        summary.index.name = _col_label(group_by, registry)

        if summary.empty:
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE, f"No groups met the minimum group size of {min_group_size} case(s).",
            )

        summary = summary.sort_values("Reopen Rate (%)", ascending=(sort_order == "asc"))
        if top_n > 0:
            summary = summary.head(top_n)

        group_label = _col_label(group_by, registry)
        title: str = str(kwargs.get("title") or f"Reopened Complaint Analysis — Reopen Rate by {group_label}")
        overall_rate = float(work_df["__reopened"].mean() * 100.0) if len(work_df) else 0.0

        bar_colors = [
            theme["danger"] if v >= max(overall_rate, 1e-9) else theme["primary"]
            for v in summary["Reopen Rate (%)"]
        ]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=summary.index.tolist(), y=summary["Reopen Rate (%)"],
            marker_color=bar_colors, marker_line_width=0,
            text=[f"{v:.1f}%" for v in summary["Reopen Rate (%)"]],
            textposition="outside", textfont=dict(size=10, family=_FONT_FAMILY),
            customdata=np.stack([summary["Total Cases"], summary["Reopened Cases"]], axis=-1),
            hovertemplate="<b>%{x}</b><br>Reopen Rate: %{y:.1f}%<br>"
                         "Reopened: %{customdata[1]:,.0f} / Total: %{customdata[0]:,.0f}<extra></extra>",
        ))
        fig.add_hline(
            y=overall_rate, line_dash="dash", line_color=theme["secondary"], line_width=1.5,
            annotation_text=f"Overall: {overall_rate:.1f}%", annotation_position="top left",
            annotation_font=dict(size=10, color=theme["secondary"], family=_FONT_FAMILY),
        )
        _apply_layout(fig, "" , theme, height=height, xaxis_title=group_label,
                     yaxis_title="Reopen Rate (%)", showlegend=False)
        fig.update_xaxes(tickangle=-35)

        export_df = summary.reset_index()
        meta = _build_universal_metadata(
            _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
            _EXPORT_FILE, title=title, status="Eligible",
            reason=f"Reopen rate computed across {len(summary)} group(s); overall reopen rate is "
                   f"{overall_rate:.1f}%.",
        )
        return fig, export_df, meta

    except Exception as e:
        return _universal_ineligible(
            "Reopened Complaint Analysis — Reopen Rate by Dimension", [],
            [ROLE_REOPEN_FLAG, ROLE_STATUS, ROLE_CATEGORY, ROLE_OFFICER, ROLE_ZONE, ROLE_RECORD_ID, ROLE_SUBSTATION],
            ["Complaint Management", "all"],
            ["date_range", "category", "officer", "zone", "substation", "record_id"], ["count"], ["overall"],
            "reopened_complaint_analysis.csv", str(e) or "Missing required core roles",
        )


def render_risk_matrix(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    **kwargs: Any,
) -> ChartReturn:
    """Volume × Median-Latency risk quadrant scatter, bubble-sized by a
    dynamically degrading risk model (Minimum -> Standard -> Advanced ->
    Enterprise) selected by role availability."""
    _CHART_TYPE = "Risk Matrix — Volume × Latency Quadrant Analysis"
    _REQUIRED: List[str] = []
    _OPTIONAL: List[str] = [
        ROLE_CATEGORY, ROLE_ZONE, ROLE_REGISTRATION_DATE, ROLE_STATUS, ROLE_CLOSING_DATE,
        ROLE_PRIORITY, ROLE_REOPEN_FLAG, ROLE_SLA_DEADLINE, ROLE_RECORD_ID,
    ]
    _DOMAINS: List[str] = ["Complaint Management", "Supply Operations", "all"]
    _FILTERS: List[str] = ["date_range", "category", "zone", "officer"]
    _AGGS: List[str] = ["count", "mean", "median"]
    _GRANULARITIES: List[str] = ["overall"]
    _EXPORT_FILE = "risk_matrix.csv"

    try:
        group_by: str = str(kwargs.get("group_by") or kwargs.get("x_role") or ROLE_CATEGORY)
        severity_role: str = str(kwargs.get("severity_role") or ROLE_PRIORITY)
        criticality_role: str = str(kwargs.get("criticality_role") or _ROLE_BUSINESS_CRITICALITY)
        duration_unit: str = str(kwargs.get("duration_unit", "days")).lower()
        analysis_date_raw: Any = kwargs.get("analysis_date")
        requested_method: Optional[str] = kwargs.get("risk_method")
        quadrant_method: str = str(kwargs.get("quadrant_method", "median")).lower()
        quadrant_percentile: float = float(kwargs.get("quadrant_percentile", 75))
        x_threshold_override: Optional[float] = kwargs.get("x_threshold")
        y_threshold_override: Optional[float] = kwargs.get("y_threshold")
        min_group_size: int = int(kwargs.get("min_group_size", 2))
        date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = kwargs.get("date_range")
        theme_key: str = str(kwargs.get("theme_key", DEFAULT_THEME_KEY))
        height: int = int(kwargs.get("height", 560))

        theme = _get_theme(theme_key)
        reg_col = _resolve_col(ROLE_REGISTRATION_DATE, df, registry)
        work_df = _apply_date_range(df, reg_col, date_range).copy()
        if work_df.empty:
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE, "No records remain after applying the date filter.",
            )

        group_col = _resolve_col(group_by, work_df, registry)
        if not group_col:
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE, f"group_by role '{group_by}' could not be resolved. "
                             "Map Category, Zone, or an equivalent grouping role.",
            )

        lifecycle, exclude_mask, lifecycle_method = _resolve_lifecycle_state(work_df, registry)
        if exclude_mask is not None:
            keep = ~exclude_mask.reindex(work_df.index).fillna(False)
            work_df = work_df.loc[keep].copy()
            if lifecycle is not None:
                lifecycle = lifecycle.reindex(work_df.index)

        analysis_date: Optional[pd.Timestamp] = None
        if analysis_date_raw is not None:
            analysis_date = pd.to_datetime(analysis_date_raw, errors="coerce")

        latency_days: Optional[pd.Series] = None
        if lifecycle is not None and reg_col:
            latency_days, _ref = _compute_latency_series(work_df, registry, lifecycle, analysis_date, duration_unit)

        severity_series = _resolve_weight_series(work_df, registry, severity_role, default_weight=2.0)
        criticality_series = _resolve_weight_series(work_df, registry, criticality_role, default_weight=1.0)
        reopen_series = _resolve_reopen_series(work_df, registry)

        has_latency = latency_days is not None
        has_severity = severity_series is not None
        has_criticality = criticality_series is not None
        has_reopen = reopen_series is not None
        risk_method = _select_risk_method(requested_method, has_latency, has_severity, has_criticality, has_reopen)

        work_df = work_df.assign(
            __latency=latency_days if has_latency else np.nan,
            __severity=severity_series if has_severity else np.nan,
            __criticality=criticality_series if has_criticality else np.nan,
            __reopened=(reopen_series.fillna(False) if has_reopen else False),
        )

        grouped = work_df.groupby(group_col, dropna=True)
        summary = pd.DataFrame(index=grouped.size().index)
        summary["Volume"] = grouped.size()
        summary["Median Latency"] = grouped["__latency"].median() if has_latency else 0.0
        summary["Mean Latency"] = grouped["__latency"].mean() if has_latency else 0.0

        if has_severity:
            summary["Avg Severity Weight"] = grouped["__severity"].mean()
        if has_criticality:
            summary["Avg Criticality Weight"] = grouped["__criticality"].mean()
        if has_reopen:
            reopened_totals = work_df.loc[work_df["__reopened"]].groupby(group_col).size().reindex(summary.index, fill_value=0)
            summary["Reopen Rate (%)"] = np.where(
                summary["Volume"] > 0, reopened_totals / summary["Volume"].replace(0, np.nan) * 100.0, 0.0
            )
            summary["Reopen Rate (%)"] = summary["Reopen Rate (%)"].fillna(0.0)

        summary = summary.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        summary = summary.loc[summary["Volume"] >= max(min_group_size, 1)]
        summary.index = summary.index.astype(str)
        summary.index.name = _col_label(group_by, registry)

        if summary.empty:
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE, f"No groups met the minimum group size of {min_group_size} case(s).",
            )

        if risk_method == "minimum":
            summary["Risk Score"] = summary["Volume"].astype(float)
        elif risk_method == "standard":
            summary["Risk Score"] = summary["Volume"] * summary["Mean Latency"].clip(lower=0.0)
        elif risk_method == "advanced":
            summary["Risk Score"] = (
                summary["Volume"] * summary["Mean Latency"].clip(lower=0.0) * summary.get("Avg Severity Weight", 1.0)
            )
        else:
            summary["Risk Score"] = (
                summary["Volume"]
                * summary["Mean Latency"].clip(lower=0.0)
                * summary.get("Avg Severity Weight", 1.0)
                * summary.get("Avg Criticality Weight", 1.0)
                * (1.0 + summary.get("Reopen Rate (%)", 0.0) / 100.0)
            )

        summary["Risk Score"] = summary["Risk Score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        max_risk = float(summary["Risk Score"].max())
        if max_risk <= 0:
            summary["Risk Score"] = summary["Volume"].astype(float)
            max_risk = float(summary["Risk Score"].max())
        max_risk = max_risk if max_risk > 0 else 1.0

        volume_series = summary["Volume"].astype(float)
        latency_series_plot = summary["Median Latency"].astype(float)

        def _threshold(series: pd.Series, method: str, override: Optional[float]) -> float:
            if override is not None:
                return float(override)
            if method == "mean":
                return float(series.mean())
            if method == "percentile":
                return float(series.quantile(min(max(quadrant_percentile, 0.0), 100.0) / 100.0))
            return float(series.median())

        if quadrant_method == "sla" and reg_col:
            sla_col = _resolve_col(ROLE_SLA_DEADLINE, work_df, registry)
            if sla_col:
                sla_dt = pd.to_datetime(work_df[sla_col], errors="coerce")
                reg_dt = pd.to_datetime(work_df[reg_col], errors="coerce")
                allowed = ((sla_dt - reg_dt).dt.total_seconds() / _DURATION_DIVISORS.get(duration_unit, 86400.0))
                allowed = allowed.replace([np.inf, -np.inf], np.nan).dropna()
                y_threshold = float(allowed.mean()) if len(allowed) else _threshold(latency_series_plot, "median", None)
            else:
                y_threshold = _threshold(latency_series_plot, "median", None)
            x_threshold = _threshold(volume_series, "median", x_threshold_override)
        else:
            resolved_qm = quadrant_method if quadrant_method in ("median", "mean", "percentile", "static") else "median"
            x_threshold = _threshold(volume_series, resolved_qm, x_threshold_override)
            y_threshold = _threshold(latency_series_plot, resolved_qm, y_threshold_override)

        group_label = _col_label(group_by, registry)
        title: str = str(
            kwargs.get("title") or f"Risk Matrix ({risk_method.title()} Model) — Volume × Latency by {group_label}"
        )
        sizeref = 2.0 * max_risk / (40.0 ** 2)
        sizeref = sizeref if sizeref > 0 else 1.0

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=volume_series, y=latency_series_plot, mode="markers+text",
            text=summary.index.tolist(), textposition="top center",
            textfont=dict(size=9, family=_FONT_FAMILY, color=theme["text"]),
            marker=dict(
                size=summary["Risk Score"].astype(float), sizemode="area", sizeref=sizeref, sizemin=4,
                color=summary["Risk Score"], colorscale=_SEQUENTIAL_BLUES, showscale=True,
                colorbar=dict(thickness=12, len=0.7,
                             title=dict(text="Risk Score", font=dict(size=10, family=_FONT_FAMILY)),
                             tickfont=dict(size=9, family=_FONT_FAMILY)),
                line=dict(width=1, color="rgba(255,255,255,0.6)"), opacity=0.82,
            ),
            customdata=np.stack([summary["Volume"], summary["Median Latency"], summary["Risk Score"]], axis=-1),
            hovertemplate=(
                "<b>%{text}</b><br>Volume: %{customdata[0]:,.0f}<br>"
                f"Median Latency ({duration_unit}): " + "%{customdata[1]:,.1f}<br>"
                "Risk Score: %{customdata[2]:,.1f}<extra></extra>"
            ),
        ))

        fig.add_vline(x=x_threshold, line_dash="dash", line_color=theme["secondary"], line_width=1.2)
        fig.add_hline(y=y_threshold, line_dash="dash", line_color=theme["secondary"], line_width=1.2)
        fig.add_annotation(x=1, y=1, xref="paper", yref="paper", xanchor="right", yanchor="top",
                          text="High Volume / High Latency", showarrow=False,
                          font=dict(size=9, color=theme["danger"], family=_FONT_FAMILY))
        fig.add_annotation(x=0, y=1, xref="paper", yref="paper", xanchor="left", yanchor="top",
                          text="Low Volume / High Latency", showarrow=False,
                          font=dict(size=9, color=theme["warning"], family=_FONT_FAMILY))
        fig.add_annotation(x=1, y=0, xref="paper", yref="paper", xanchor="right", yanchor="bottom",
                          text="High Volume / Low Latency", showarrow=False,
                          font=dict(size=9, color=theme["warning"], family=_FONT_FAMILY))
        fig.add_annotation(x=0, y=0, xref="paper", yref="paper", xanchor="left", yanchor="bottom",
                          text="Low Volume / Low Latency", showarrow=False,
                          font=dict(size=9, color=theme["success"], family=_FONT_FAMILY))

        _apply_layout(fig, "" , theme, height=height, showlegend=False,
                     xaxis_title="Total Volume (Count)", yaxis_title=f"Median Latency ({duration_unit})")

        export_df = summary.reset_index()
        meta = _build_universal_metadata(
            _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
            _EXPORT_FILE, title=title, status="Eligible",
            reason=f"Risk computed using the '{risk_method}' model across {len(summary)} group(s) "
                   f"(quadrant method: {quadrant_method}).",
        )
        return fig, export_df, meta

    except Exception as e:
        return _universal_ineligible(
            "Risk Matrix — Volume × Latency Quadrant Analysis", [],
            [ROLE_CATEGORY, ROLE_ZONE, ROLE_REGISTRATION_DATE, ROLE_STATUS, ROLE_CLOSING_DATE,
             ROLE_PRIORITY, ROLE_REOPEN_FLAG, ROLE_SLA_DEADLINE, ROLE_RECORD_ID],
            ["Complaint Management", "Supply Operations", "all"],
            ["date_range", "category", "zone", "officer"], ["count", "mean", "median"], ["overall"],
            "risk_matrix.csv", str(e) or "Missing required core roles",
        )


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC RENDER FUNCTION — Part 3 Section B: Enterprise Geospatial Intelligence
# ══════════════════════════════════════════════════════════════════════════════

_GEO_LAT_MIN: float = -90.0
_GEO_LAT_MAX: float = 90.0
_GEO_LON_MIN: float = -180.0
_GEO_LON_MAX: float = 180.0
_GEO_DEFAULT_ZOOM: int = 11
_GEO_DEFAULT_MAX_MARKERS: int = 5000
_GEO_DEFAULT_HEATMAP_RADIUS: int = 18
_GEO_DEFAULT_HEATMAP_BLUR: int = 14


def _geo_valid_coord_mask(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """Vectorized coordinate sanitation: numeric, within WGS-84 bounds, and
    excludes the (0.0, 0.0) null-island sentinel used by many legacy exports."""
    lat_n = pd.to_numeric(lat, errors="coerce")
    lon_n = pd.to_numeric(lon, errors="coerce")
    mask = (
        lat_n.notna()
        & lon_n.notna()
        & lat_n.between(_GEO_LAT_MIN, _GEO_LAT_MAX)
        & lon_n.between(_GEO_LON_MIN, _GEO_LON_MAX)
        & ~((lat_n == 0.0) & (lon_n == 0.0))
    )
    return mask


def _geo_build_tooltip_html(
    row: Any,
    field_map: Dict[str, Optional[str]],
) -> str:
    """Builds a compact HTML tooltip block from a namedtuple row (via
    itertuples) and a {display_label: resolved_column_name_or_None} map."""
    parts: List[str] = []
    for label, col in field_map.items():
        if col is None:
            continue
        try:
            val = getattr(row, col, None)
        except Exception:
            val = None
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        parts.append(f"<b>{label}:</b> {val}")
    if not parts:
        return "<i>No attribute data available</i>"
    return "<br>".join(parts)


def render_geographical_concentration_map(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    **kwargs: Any,
) -> Tuple[Optional[folium.Map], Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Enterprise geospatial concentration map with a four-tier resolution
    fallback (Exact GPS -> Offline Substation Dictionary -> Boundary Centroid
    Approximation -> Graceful Ineligibility). Never raises to the UI; every
    failure path returns structured (None, None, metadata).
    """
    _CHART_TYPE = "Geographical Concentration Map — Density & Heat Distribution"
    _REQUIRED: List[str] = []
    _OPTIONAL: List[str] = [
        ROLE_LATITUDE, ROLE_LONGITUDE, ROLE_FEEDER, ROLE_TRANSFORMER,
        ROLE_DIVISION, ROLE_SUBDIVISION,ROLE_SUBSTATION, ROLE_ZONE, ROLE_CIRCLE,
        ROLE_RECORD_ID, ROLE_STATUS, ROLE_CATEGORY, ROLE_OFFICER,
        ROLE_REGISTRATION_DATE,
    ]
    _DOMAINS: List[str] = ["Complaint Management", "Supply Operations", "Asset Management", "all"]
    _FILTERS: List[str] = ["date_range", "zone", "circle", "division", "category", "status","subdivision","substation","feeder"]
    _AGGS: List[str] = ["count"]
    _GRANULARITIES: List[str] = ["overall"]
    _EXPORT_FILE = "geographical_concentration_map.csv"

    try:
        date_role: str = str(kwargs.get("date_role") or ROLE_REGISTRATION_DATE)
        date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = kwargs.get("date_range")
        theme_key: str = str(kwargs.get("theme_key", DEFAULT_THEME_KEY))
        tile_style: str = str(kwargs.get("tile_style", "cartodbpositron"))
        zoom_start: int = int(kwargs.get("zoom_start", _GEO_DEFAULT_ZOOM))
        max_markers: int = int(kwargs.get("max_markers", _GEO_DEFAULT_MAX_MARKERS))
        heatmap_radius: int = int(kwargs.get("heatmap_radius", _GEO_DEFAULT_HEATMAP_RADIUS))
        heatmap_blur: int = int(kwargs.get("heatmap_blur", _GEO_DEFAULT_HEATMAP_BLUR))
        duration_unit: str = str(kwargs.get("duration_unit", "days")).lower()
        analysis_date_raw: Any = kwargs.get("analysis_date")
        substation_lookup: Optional[Dict[str, Tuple[float, float]]] = kwargs.get("substation_lookup")
        boundary_centroid_lookup: Optional[Dict[str, Tuple[float, float]]] = kwargs.get("boundary_centroid_lookup")
        location_key_role: str = str(kwargs.get("location_key_role") or ROLE_FEEDER)

        theme = _get_theme(theme_key)
        date_col = _resolve_col(date_role, df, registry)
        work_df = _apply_date_range(df, date_col, date_range).copy()
        if work_df.empty:
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE, "No records remain after applying the date filter.",
            )

        lat_col = _resolve_col(ROLE_LATITUDE, work_df, registry)
        lon_col = _resolve_col(ROLE_LONGITUDE, work_df, registry)

        plot_df: Optional[pd.DataFrame] = None
        export_df: Optional[pd.DataFrame] = None
        resolution_method: str = "unresolved"
        marker_mode: str = "row"  # "row" (individual records) or "group" (aggregated centroids)

        # ── Priority 1: Exact GPS mapping ────────────────────────────────
        if lat_col and lon_col:
            lat_series = pd.to_numeric(work_df[lat_col], errors="coerce")
            lon_series = pd.to_numeric(work_df[lon_col], errors="coerce")
            valid_mask = _geo_valid_coord_mask(lat_series, lon_series)

            export_df = work_df.copy()
            export_df["Resolved Latitude"] = lat_series.where(valid_mask)
            export_df["Resolved Longitude"] = lon_series.where(valid_mask)
            export_df["Geolocation Status"] = np.where(
                valid_mask, "Resolved (Exact GPS)", "Dropped — Invalid/Missing Coordinates"
            )

            if valid_mask.any():
                plot_df = work_df.loc[valid_mask].copy()
                plot_df["__lat"] = lat_series.loc[valid_mask]
                plot_df["__lon"] = lon_series.loc[valid_mask]
                resolution_method = "exact_gps"
                marker_mode = "row"

        # ── Priority 2: Offline substation/feeder dictionary lookup ─────
        if plot_df is None and isinstance(substation_lookup, dict) and substation_lookup:
            key_col = (
                _resolve_col(location_key_role, work_df, registry)
                or _resolve_col(ROLE_FEEDER, work_df, registry)
                or _resolve_col(ROLE_TRANSFORMER, work_df, registry)
            )
            if key_col:
                coord_map: Dict[str, Tuple[float, float]] = {
                    str(k).strip(): v for k, v in substation_lookup.items()
                }
                keys_normalized = work_df[key_col].astype(str).str.strip()
                mapped_lat = keys_normalized.map(
                    lambda k: coord_map[k][0] if k in coord_map else np.nan
                )
                mapped_lon = keys_normalized.map(
                    lambda k: coord_map[k][1] if k in coord_map else np.nan
                )
                lat_series = pd.to_numeric(mapped_lat, errors="coerce")
                lon_series = pd.to_numeric(mapped_lon, errors="coerce")
                valid_mask = _geo_valid_coord_mask(lat_series, lon_series)

                export_df = work_df.copy()
                export_df["Lookup Key"] = keys_normalized
                export_df["Resolved Latitude"] = lat_series.where(valid_mask)
                export_df["Resolved Longitude"] = lon_series.where(valid_mask)
                export_df["Geolocation Status"] = np.where(
                    valid_mask,
                    "Resolved (Offline Substation Dictionary)",
                    "Dropped — No Substation/Feeder Coordinate Match",
                )

                if valid_mask.any():
                    plot_df = work_df.loc[valid_mask].copy()
                    plot_df["__lat"] = lat_series.loc[valid_mask]
                    plot_df["__lon"] = lon_series.loc[valid_mask]
                    resolution_method = "offline_dictionary_substation"
                    marker_mode = "row"

        # ── Priority 3: Boundary centroid approximation ─────────────────
        if plot_df is None and isinstance(boundary_centroid_lookup, dict) and boundary_centroid_lookup:
            boundary_col = (
                _resolve_col(ROLE_SUBDIVISION, work_df, registry)
                or _resolve_col(ROLE_DIVISION, work_df, registry)
                or _resolve_col(ROLE_CIRCLE, work_df, registry)
                or _resolve_col(ROLE_ZONE, work_df, registry)
            )
            if boundary_col:
                coord_map = {str(k).strip(): v for k, v in boundary_centroid_lookup.items()}
                keys_normalized = work_df[boundary_col].astype(str).str.strip()

                def _lat_of(k: str) -> float:
                    v = coord_map.get(k)
                    return float(v[0]) if v is not None else np.nan

                def _lon_of(k: str) -> float:
                    v = coord_map.get(k)
                    return float(v[1]) if v is not None else np.nan

                mapped_lat = keys_normalized.map(_lat_of)
                mapped_lon = keys_normalized.map(_lon_of)
                lat_series = pd.to_numeric(mapped_lat, errors="coerce")
                lon_series = pd.to_numeric(mapped_lon, errors="coerce")
                valid_mask = _geo_valid_coord_mask(lat_series, lon_series)

                export_df = work_df.copy()
                export_df["Boundary Group"] = keys_normalized
                export_df["Resolved Latitude"] = lat_series.where(valid_mask)
                export_df["Resolved Longitude"] = lon_series.where(valid_mask)
                export_df["Geolocation Status"] = np.where(
                    valid_mask,
                    "Resolved (Boundary Centroid Approximation)",
                    "Dropped — No Centroid Mapping for Boundary Group",
                )

                if valid_mask.any():
                    valid_work = work_df.loc[valid_mask].copy()
                    valid_work["__lat"] = lat_series.loc[valid_mask]
                    valid_work["__lon"] = lon_series.loc[valid_mask]
                    valid_work["__group"] = keys_normalized.loc[valid_mask]
                    grouped = (
                        valid_work.groupby("__group", dropna=True)
                        .agg(
                            __lat=("__lat", "first"),
                            __lon=("__lon", "first"),
                            __count=("__group", "size"),
                        )
                        .reset_index()
                    )
                    plot_df = grouped
                    resolution_method = "centroid_approximation"
                    marker_mode = "group"

        # ── Priority 4: Graceful ineligibility ───────────────────────────
        if plot_df is None or plot_df.empty:
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE,
                "Geospatial markers unresolvable: no Latitude/Longitude roles are mapped, "
                "no matching entries were found in an offline substation_lookup, and no "
                "boundary_centroid_lookup could resolve Division/Subdivision/Circle/Zone "
                "boundaries to coordinates.",
            )

        if marker_mode == "row" and len(plot_df) > max_markers:
            plot_df = plot_df.sample(max_markers, random_state=42)

        center_lat = float(plot_df["__lat"].mean())
        center_lon = float(plot_df["__lon"].mean())
        if not (np.isfinite(center_lat) and np.isfinite(center_lon)):
            return _universal_ineligible(
                _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
                _EXPORT_FILE, "Resolved coordinate set produced a non-finite map center.",
            )

        fmap = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles=tile_style,
            control_scale=True,
        )

        marker_cluster = MarkerCluster(name="Concentration Markers").add_to(fmap)
        heat_data: List[List[float]] = []
        geojson_features: List[Dict[str, Any]] = []

        if marker_mode == "row":
            id_col = _resolve_col(ROLE_RECORD_ID, plot_df, registry)
            status_col = _resolve_col(ROLE_STATUS, plot_df, registry)
            category_col = _resolve_col(ROLE_CATEGORY, plot_df, registry)
            officer_col = _resolve_col(ROLE_OFFICER, plot_df, registry)
            division_col = _resolve_col(ROLE_DIVISION, plot_df, registry)
            subdivision_col = _resolve_col(ROLE_SUBDIVISION, plot_df, registry)
            feeder_col = _resolve_col(ROLE_FEEDER, plot_df, registry)
            substation_col = _resolve_col(ROLE_TRANSFORMER, plot_df, registry)

            pending_age_col: Optional[str] = None
            reg_col = _resolve_col(ROLE_REGISTRATION_DATE, plot_df, registry)
            if reg_col:
                lifecycle, exclude_mask, _lm = _resolve_lifecycle_state(plot_df, registry)
                if lifecycle is not None:
                    if exclude_mask is not None:
                        keep = ~exclude_mask.reindex(plot_df.index).fillna(False)
                        plot_df = plot_df.loc[keep].copy()
                        lifecycle = lifecycle.reindex(plot_df.index)
                    analysis_date: Optional[pd.Timestamp] = None
                    if analysis_date_raw is not None:
                        analysis_date = pd.to_datetime(analysis_date_raw, errors="coerce")
                    latency_days, _ref = _compute_latency_series(
                        plot_df, registry, lifecycle, analysis_date, duration_unit
                    )
                    if latency_days is not None:
                        plot_df = plot_df.assign(__pending_age=latency_days.round(1))
                        pending_age_col = "__pending_age"

            field_map: Dict[str, Optional[str]] = {
                "Complaint ID": id_col,
                "Status": status_col,
                "Category": category_col,
                "Officer": officer_col,
                f"Pending Age ({duration_unit})": pending_age_col,
                "Division": division_col,
                "Subdivision": subdivision_col,
                "Feeder": feeder_col,
                "Substation": substation_col,
            }

            status_col_for_color = status_col
            for row in plot_df.itertuples(index=False):
                lat_v = float(getattr(row, "__lat"))
                lon_v = float(getattr(row, "__lon"))
                tooltip_html = _geo_build_tooltip_html(row, field_map)

                marker_color = "blue"
                if status_col_for_color:
                    raw_status = str(getattr(row, status_col_for_color, "")).strip().upper()
                    marker_color = _BENCHMARK_COLOR_MAP.get(
                        {"CLOSED": "excellent", "PENDING": "warning", "REOPENED": "critical"}.get(raw_status, "na"),
                        theme["primary"],
                    )

                folium.CircleMarker(
                    location=[lat_v, lon_v],
                    radius=5,
                    color=marker_color,
                    fill=True,
                    fill_color=marker_color,
                    fill_opacity=0.85,
                    weight=1,
                    tooltip=folium.Tooltip(tooltip_html),
                ).add_to(marker_cluster)

                heat_data.append([lat_v, lon_v])

                search_label = (
                    str(getattr(row, id_col)) if id_col and getattr(row, id_col, None) is not None
                    else f"{lat_v:.4f},{lon_v:.4f}"
                )
                geojson_features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon_v, lat_v]},
                    "properties": {"search_label": search_label},
                })

            title: str = str(kwargs.get("title") or "Geographical Concentration Map — Exact Record Density")

        else:
            max_count = float(plot_df["__count"].max()) if len(plot_df) else 1.0
            max_count = max_count if max_count > 0 else 1.0
            for row in plot_df.itertuples(index=False):
                lat_v = float(getattr(row, "__lat"))
                lon_v = float(getattr(row, "__lon"))
                group_name = str(getattr(row, "__group"))
                count_v = int(getattr(row, "__count"))
                radius = 6.0 + 24.0 * (count_v / max_count)
                tooltip_html = f"<b>Boundary:</b> {group_name}<br><b>Total Cases:</b> {count_v:,}"

                folium.CircleMarker(
                    location=[lat_v, lon_v],
                    radius=radius,
                    color=theme["primary"],
                    fill=True,
                    fill_color=theme["primary"],
                    fill_opacity=0.55,
                    weight=1.2,
                    tooltip=folium.Tooltip(tooltip_html),
                ).add_to(marker_cluster)

                heat_data.append([lat_v, lon_v, count_v])

                geojson_features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon_v, lat_v]},
                    "properties": {"search_label": group_name},
                })

            title = str(
                kwargs.get("title")
                or "Geographical Concentration Map — Boundary Centroid Approximation"
            )

        if heat_data:
            HeatMap(
                heat_data,
                name="Heat Density",
                radius=heatmap_radius,
                blur=heatmap_blur,
                show=False,
            ).add_to(fmap)

        if geojson_features:
            search_layer = folium.GeoJson(
                {"type": "FeatureCollection", "features": geojson_features},
                name="Search Index",
                show=False,
                marker=folium.CircleMarker(radius=0, fill_opacity=0.0, opacity=0.0),
            )
            search_layer.add_to(fmap)
            Search(
                layer=search_layer,
                geom_type="Point",
                search_label="search_label",
                placeholder="Search records / boundaries…",
                collapsed=True,
                search_zoom=14,
            ).add_to(fmap)

        Fullscreen(position="topleft").add_to(fmap)
        MiniMap(toggle_display=True, position="bottomleft").add_to(fmap)
        MeasureControl(primary_length_unit="kilometers", position="topleft").add_to(fmap)
        MousePosition(position="bottomright").add_to(fmap)
        folium.LayerControl(collapsed=False).add_to(fmap)

        resolved_count = int(len(plot_df)) if marker_mode == "row" else int(plot_df["__count"].sum())
        total_count = int(len(work_df))
        meta = _build_universal_metadata(
            _CHART_TYPE, _REQUIRED, _OPTIONAL, _DOMAINS, _FILTERS, _AGGS, _GRANULARITIES,
            _EXPORT_FILE, title=title, status="Eligible",
            reason=(
                f"Resolved {resolved_count:,} of {total_count:,} record(s) via '{resolution_method}' "
                f"geospatial resolution strategy."
            ),
        )
        return fmap, export_df, meta

    except Exception as e:
        return _universal_ineligible(
            "Geographical Concentration Map — Density & Heat Distribution",
            [],
            [
                ROLE_LATITUDE, ROLE_LONGITUDE, ROLE_FEEDER, ROLE_TRANSFORMER,
                ROLE_DIVISION, ROLE_SUBDIVISION, ROLE_ZONE, ROLE_CIRCLE,
                ROLE_RECORD_ID, ROLE_STATUS, ROLE_CATEGORY, ROLE_OFFICER,
                ROLE_REGISTRATION_DATE,
            ],
            ["Complaint Management", "Supply Operations", "Asset Management", "all"],
            ["date_range", "zone", "circle", "division", "category", "status"],
            ["count"], ["overall"],
            "geographical_concentration_map.csv", str(e) or "Missing required core roles",
        )


# ══════════════════════════════════════════════════════════════════════════════
# FINAL PLATFORM REGISTRY — Consolidated across Parts 1, 2, and 3 (A + B)
# ══════════════════════════════════════════════════════════════════════════════
CHART_REGISTRY: Dict[str, Any] = {
    "bar": render_bar,
    "bar_horizontal": render_bar_horizontal,
    "bar_grouped": render_bar_grouped,
    "bar_stacked": render_bar_stacked,
    "line": render_line,
    "area": render_area,
    "scatter": render_scatter,
    "bubble": render_bubble,
    "pie": render_pie,
    "donut": render_donut,
    "histogram": render_histogram,
    "box": render_box,
    "sparkline": render_sparkline,
    "treemap": render_treemap,
    "sunburst": render_sunburst,
    "heatmap": render_heatmap,
    "calendar_heatmap": render_calendar_heatmap,
    "pareto": render_pareto_diagram,
    "sankey": render_sankey_diagram,
    "peak_complaint_hour": render_peak_complaint_hour,
    "rolling_average_trend": render_monthly_rolling_average,
    "growth_heatmap": render_complaint_growth_heatmap,
    "officer_performance_matrix": render_officer_performance_matrix,
    "pending_age_distribution": render_pending_age_distribution,
    "reopened_complaint_analysis": render_reopened_complaint_analysis,
    "risk_matrix": render_risk_matrix,
    "geographical_concentration_map": render_geographical_concentration_map,
}


# ══════════════════════════════════════════════════════════════════════════════
# UNIVERSAL DISPATCH WRAPPER — Final Routing Entry Point
# ══════════════════════════════════════════════════════════════════════════════
def render(
    chart_type: str,
    df: pd.DataFrame,
    registry: ColumnRegistry,
    **kwargs: Any,
) -> Union[ChartReturn, Tuple[Optional[folium.Map], Optional[pd.DataFrame], Dict[str, Any]]]:
    """Single entry point for the entire visualization layer. Resolves
    `chart_type` against the consolidated CHART_REGISTRY and dispatches with
    the full **kwargs self-service parameter surface. Never raises — any
    dispatch or execution failure degrades to a structured ineligible tuple."""
    fn = CHART_REGISTRY.get(chart_type)
    if fn is None:
        return _ineligible(
            [], [], ["all"], f"Unknown chart type: {chart_type}",
            f"'{chart_type}' is not registered. "
            f"Available types: {', '.join(sorted(CHART_REGISTRY))}",
        )
    try:
        return fn(df, registry, **kwargs)
    except Exception as exc:
        return _ineligible([], [], ["all"], chart_type, f"Unhandled dispatch error: {exc}")
