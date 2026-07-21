from __future__ import annotations
# """
# FRONTEND/pages/1_dashboard.py — KESCO Grid Operational Intelligence Dashboard
# (FILE 5 / 6)

# Presentation-Layer Orchestrator ONLY. This module owns zero business logic,
# zero KPI computation, zero chart construction, and zero data-quality
# scoring — every one of those responsibilities remains exclusively with the
# immutable production engine modules and with the already-populated
# st.session_state artifacts produced by FRONTEND.app's ingestion pipeline
# and FRONTEND.components.filters' Global Filter Engine + Cascading Drill
# Panel.

# Milestone 11 note: this file contains no Presentation Mode references and
# required no logic changes for the sidebar-duplication / theme-switching /
# KPI-flip-card fixes — those were resolved in app.py, sidebar.py, and
# style.css respectively. Reproduced here unchanged to keep the full file
# set internally consistent.

# Immutable Layout Pipeline (executed sequentially every rerun):
#     1. Initialize Page Session States
#     2. Data Integrity Gate
#     3. Dynamic Schema Resolution (implicit — resolved via ColumnRegistry
#        inside every downstream engine/visualization call)
#     4. Cascading Filter Layer (delegated to FRONTEND.components.filters)
#     5. Analytics Execution (delegated to engine.analytics, cached)
#     6. UI Component Streaming (Cognitive Storytelling Hierarchy, Levels 1-7)
# """
import io
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from utils.error_logging import log_exception
import pandas as pd
import streamlit as st
# ── Milestone 14 import addition (top of file, alongside the existing
# `from utils.grid_utils import render_enterprise_grid` import) ──────────
from utils.grid_utils import render_enterprise_grid, sanitize_dataframe_for_display

from core.themes import inject_fragment_atomic_style  

from core.column_registry import ColumnRegistry
from core.roles import (
    ROLE_CATEGORY,
    ROLE_CIRCLE,
    ROLE_OFFICER,
    ROLE_REGISTRATION_DATE,
    ROLE_STATUS,
    ROLE_ZONE,
    ROLE_SUBCATEGORY,
    ROLE_PRIORITY,
    ROLE_DEPARTMENT,
    CANONICAL_GEO_HIERARCHY,
)
from core.settings import APP_ICON, MAX_PREVIEW_ROWS, ENTERPRISE_COPY_MAP
from utils.grid_utils import render_enterprise_grid
from core.themes import THEMES, DEFAULT_THEME_KEY, Theme
from engine.analytics import UniversalAnalyticsEngine, ComplaintAnalyticsBundle, KPIResult
from engine.domain_detection import DOMAIN_UNKNOWN
from visualization.kpi_cards import generate_metric_cards
from visualization.chart_factory import render as render_chart
from visualization.chart_interpreter import interpret_chart_output
from visualization.executive_summary import generate_executive_narrative
from FRONTEND.components import filters as filters_component

_PAGE_NAME: str = "dashboard"


# ── Presentation-layer correlation map: links kpi_cards.py's opaque
# metric_id back to the richer engine.analytics.KPIResult key that produced
# it, purely for tooltip/definition/formula enrichment.
_METRIC_ID_TO_KPI_KEY: Dict[str, str] = {
    "total_volume": "total_cases",
    "resolution_rate": "closure_rate",
    "avg_processing_latency": "avg_resolution_time",
    "sla_compliance": "sla_compliance_rate",
    "reopen_rate": "reopen_rate",
    "pending_rate": "pending_rate",
    "backlog": "backlog",
    "first_time_resolution_rate": "first_time_resolution_rate",
    "avg_pending_age": "avg_pending_age",
    "median_pending_age": "median_pending_age",
    "p95_pending_age": "p95_pending_age",
    "median_resolution_time": "median_resolution_time",
    "p95_resolution_time": "p95_resolution_time",
    "sla_breach_rate": "sla_breach_rate",
    "unique_consumers": "unique_consumers",
    "repeat_consumer_rate": "repeat_consumer_rate",
    "mom_growth": "mom_growth",
    "qoq_growth": "qoq_growth",
    "yoy_growth": "yoy_growth",
}

_ACCENT_TO_BADGE: Dict[str, str] = {
    "success": "good", "warning": "warning", "danger": "critical", "secondary": "info",
}

_STATUS_TIER_TO_ACCENT: Dict[str, str] = {
    "EXCELLENT": "success", "GOOD": "success", "FAIR": "warning",
    "CRITICAL": "danger", "WARNING": "warning", "NA": "secondary",
}

_ROW1_CHART_TYPE_OPTIONS: Tuple[str, ...] = (
    "bar_grouped", "bar", "bar_horizontal", "bar_stacked", "pie", "donut", "treemap", "sunburst",
)
_ROW2_CHART_TYPE_OPTIONS: Tuple[str, ...] = ("line", "area", "rolling_average_trend")

_ROW3_CHART_TYPE_OPTIONS: Tuple[str, ...] = (
    "risk_matrix", "pareto", "officer_performance_matrix",
)

_ROW1_GROUP_ROLE_CANDIDATES: Tuple[Tuple[str, str], ...] = (
    (ROLE_STATUS, "Status"),
    (ROLE_OFFICER, "Officer"),
    (ROLE_CATEGORY, "Category"),
    (ROLE_SUBCATEGORY, "Subcategory"),
    (ROLE_PRIORITY, "Priority"),
    (ROLE_DEPARTMENT, "Department"),
) + CANONICAL_GEO_HIERARCHY


_GEO_SUBSTATION_LOOKUP_HELP: str = (
    "Optional JSON object mapping a Feeder/Substation identifier (as it appears in the "
    "mapped column) to a [latitude, longitude] pair, e.g. "
    '{"FDR-001": [26.4499, 80.3319]}. Used as the Tier-2 fallback when exact GPS '
    "coordinates are not mapped."
)
_GEO_BOUNDARY_LOOKUP_HELP: str = (
    "Optional JSON object mapping a Division/Subdivision/Circle/Zone name to a "
    '[latitude, longitude] centroid, e.g. {"Kanpur Division": [26.45, 80.33]}. Used as '
    "the Tier-3 fallback when neither exact GPS nor a substation dictionary resolves."
)


def _parse_geo_lookup_json(raw_text: str) -> Optional[Dict[str, Tuple[float, float]]]:
    """
    Parses a user-authored JSON lookup dictionary into the
    Dict[str, Tuple[float, float]] shape expected by
    visualization.chart_factory.render_geographical_concentration_map's
    substation_lookup / boundary_centroid_lookup kwargs. Never raises.
    """
    if not raw_text or not raw_text.strip():
        return None
    try:
        parsed = json.loads(raw_text)
        if not isinstance(parsed, dict):
            return None
        result: Dict[str, Tuple[float, float]] = {}
        for key, value in parsed.items():
            if (
                isinstance(value, (list, tuple))
                and len(value) == 2
                and all(isinstance(v, (int, float)) for v in value)
            ):
                result[str(key)] = (float(value[0]), float(value[1]))
        return result or None
    except Exception:  # noqa: BLE001
        return None


def _render_geospatial_intelligence_row(
    df_view: pd.DataFrame, registry: ColumnRegistry, cache_key: str
) -> bool:
    """
    Renders visualization.chart_factory.render_geographical_concentration_map.
    Folium maps are rendered via Streamlit's built-in st.components.v1.html
    (a legitimate use — folium maps are inherently self-contained HTML
    documents, unlike the platform's own theme JS, so iframe scoping is
    correct and expected here). Never raises.
    """
    from utils.layout_guard import chart_block, apply_fixed_margin_chart_layout
    with chart_block("Geographical Concentration & Density Intelligence"):
      try:
        control_col, _spacer_col = st.columns([5, 7])
        with control_col:
            with st.expander("Geospatial Resolution Fallback Dictionaries (Optional)", expanded=False):
                st.caption(
                    "Exact GPS coordinates are used automatically when Latitude/Longitude roles "
                    "are mapped. If they are not mapped, author either fallback dictionary below "
                    "to enable Tier-2/Tier-3 approximate resolution."
                )
                substation_raw = st.text_area(
                    "Feeder/Substation Coordinate Dictionary (JSON)",
                    value=st.session_state.get("_geo_substation_lookup_raw", ""),
                    key="_dash_geo_substation_lookup_input",
                    help=_GEO_SUBSTATION_LOOKUP_HELP,
                    height=90,
                )
                boundary_raw = st.text_area(
                    "Boundary Centroid Dictionary (JSON)",
                    value=st.session_state.get("_geo_boundary_lookup_raw", ""),
                    key="_dash_geo_boundary_lookup_input",
                    help=_GEO_BOUNDARY_LOOKUP_HELP,
                    height=90,
                )
                st.session_state["_geo_substation_lookup_raw"] = substation_raw
                st.session_state["_geo_boundary_lookup_raw"] = boundary_raw

        substation_lookup = _parse_geo_lookup_json(st.session_state.get("_geo_substation_lookup_raw", ""))
        boundary_lookup = _parse_geo_lookup_json(st.session_state.get("_geo_boundary_lookup_raw", ""))

        render_kwargs: Dict[str, Any] = {"theme_key": st.session_state.get("theme", DEFAULT_THEME_KEY)}
        if substation_lookup:
            render_kwargs["substation_lookup"] = substation_lookup
        if boundary_lookup:
            render_kwargs["boundary_centroid_lookup"] = boundary_lookup

        combined_key = f"{cache_key}|geographical_concentration_map|{_kwargs_signature(render_kwargs)}"
        fmap, export_df, meta = _cached_chart_render(
            "geographical_concentration_map", df_view, registry, combined_key, render_kwargs
        )

        left_col, right_col = st.columns([6, 6])
        with left_col:
            if fmap is not None and hasattr(fmap, "_repr_html_"):
                st.components.v1.html(fmap._repr_html_(), height=520, scrolling=False)
                if export_df is not None and not export_df.empty:
                    _geo_export_df = sanitize_dataframe_for_display(export_df)
                    st.data_editor(
                        _geo_export_df, width="stretch", hide_index=True,
                        key="_dash_grid_geo_row", disabled=True, height=400,
                    )
                    try:
                        resolved_export_name = meta.get("export_filename") or "geographical_concentration_map.csv"
                        st.download_button(
                            "Export This Chart's Data (CSV)",
                            data=export_df.to_csv(index=False).encode("utf-8"),
                            file_name=resolved_export_name,
                            mime="text/csv",
                            key="_dash_export_chart_geo_row",
                            width="stretch",
                        )
                    except Exception:  # noqa: BLE001
                        pass
            else:
                st.info(
                    meta.get("reason", "Geospatial visualization is not currently eligible for the "
                                       "active role mappings and filter scope.")
                )
        with right_col:
            st.markdown('<div class="kesco-narrative-container">', unsafe_allow_html=True)
            if fmap is not None:
                interpretation = interpret_chart_output("geographical_concentration_map", export_df, meta)
                summary = interpretation.get("statistical_summary", {})
                st.markdown(
                    f"**Peak:** {summary.get('peak_coordinate', 'N/A')}  \n"
                    f"**Trough:** {summary.get('trough_coordinate', 'N/A')}"
                )
                severity_badge_map = {"critical": "critical", "warning": "warning", "positive": "good", "info": "info"}
                for insight in interpretation.get("insights", []):
                    severity = insight.get("severity", "info")
                    badge_class = severity_badge_map.get(severity, "info")
                    st.markdown(
                        f'<div class="kesco-card">'
                        f'<span class="kesco-badge kesco-badge-{badge_class}">{severity.upper()}</span>'
                        f'&nbsp;&nbsp;<strong>{insight.get("headline", "")}</strong>'
                        f'<br/>{insight.get("body_text", "")}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("Narrative interpretation becomes available once this visualization is eligible.")
            st.markdown('</div>', unsafe_allow_html=True)

        return fmap is not None
      except Exception as exc:  # noqa: BLE001
        incident_id = log_exception("dashboard._render_geospatial_intelligence_row", exc)
        st.warning(
            f"Geographical Concentration Map encountered an issue and has been degraded gracefully "
            f"(Reference ID: {incident_id})."
        )
        return False


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE & CACHE-KEY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _init_dashboard_session_state() -> None:
    """Idempotently seeds this page's local diagnostic session_state keys."""
    st.session_state.setdefault("_dashboard_cache_key_hits", {})
    st.session_state.setdefault("_dashboard_cache_key_calls", {"count": 0, "hits": 0})




def _compute_cache_key(df: pd.DataFrame, registry: ColumnRegistry) -> str:
    """
    Builds a deterministic cache-invalidation signature from the current
    filtered view's shape, the Column Registry's mutation version, the
    active filter/drill-path state, and the session's hard-invalidation
    epoch (bumped by schema_mapping.py on every applied mapping commit).
    The epoch is the primary safety net against registry.version
    colliding across concurrent sessions in Streamlit's process-wide
    st.cache_data store. Never raises.
    """
    try:
        filters_sig = json.dumps(st.session_state.get("active_filters", {}), sort_keys=True, default=str)
    except Exception:  # noqa: BLE001
        filters_sig = str(st.session_state.get("active_filters", {}))
    try:
        breadcrumbs_sig = json.dumps(st.session_state.get("drill_breadcrumbs", []), sort_keys=True, default=str)
    except Exception:  # noqa: BLE001
        breadcrumbs_sig = str(st.session_state.get("drill_breadcrumbs", []))
    mutation_epoch = st.session_state.get("_registry_mutation_epoch", 0)
    return (
        f"{len(df)}|{df.shape[1]}|{registry.version}|{mutation_epoch}|"
        f"{filters_sig}|{breadcrumbs_sig}"
    )

def _kwargs_signature(kwargs: Dict[str, Any]) -> str:
    """Deterministic, order-independent string signature of a kwargs dict."""
    try:
        return json.dumps(kwargs, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001
        return str(sorted(kwargs.items(), key=lambda kv: kv[0]))


def _track_cache_usage(cache_key: str) -> None:
    """Session-observed cache-reuse heuristic. Never raises."""
    try:
        seen: Dict[str, int] = st.session_state.get("_dashboard_cache_key_hits", {})
        calls: Dict[str, int] = st.session_state.get("_dashboard_cache_key_calls", {"count": 0, "hits": 0})
        calls["count"] = calls.get("count", 0) + 1
        if cache_key in seen:
            calls["hits"] = calls.get("hits", 0) + 1
        seen[cache_key] = seen.get(cache_key, 0) + 1
        st.session_state["_dashboard_cache_key_hits"] = seen
        st.session_state["_dashboard_cache_key_calls"] = calls
    except Exception:  # noqa: BLE001
        return

def _build_duckdb_filter_predicates(
    registry: ColumnRegistry,
) -> List[Tuple[str, str, Any]]:
    """
    Translates the currently active st.session_state['active_filters'] and
    ['drill_breadcrumbs'] into the (column, operator, value) predicate list
    consumed by engine.duckdb_executor.duckdb_group_aggregate_from_parquet_filtered.
    Every column is resolved exclusively through the Column Registry — never
    a literal name. Categorical multiselects become "in" predicates, the
    date range becomes a "between" predicate, and drill breadcrumbs become
    "eq" predicates. Any role that fails to resolve is silently skipped
    (fails safe toward "no predicate for that filter", never toward a
    crash). Never raises.
    """
    predicates: List[Tuple[str, str, Any]] = []
    try:
        active_filters: Dict[str, Any] = st.session_state.get("active_filters", {})
        for role, value in active_filters.items():
            if role == "date_range":
                date_col = registry.resolve(ROLE_REGISTRATION_DATE)
                if date_col and isinstance(value, (list, tuple)) and len(value) == 2:
                    predicates.append((date_col, "between", (str(value[0]), str(value[1]))))
                continue
            col = registry.resolve(role)
            if col and value:
                predicates.append((col, "in", list(value)))

        for crumb in st.session_state.get("drill_breadcrumbs", []):
            role = crumb.get("role", "")
            crumb_value = crumb.get("value", "")
            col = registry.resolve(role)
            if col and crumb_value:
                predicates.append((col, "eq", crumb_value))
    except Exception:  # noqa: BLE001
        return []
    return predicates

# ══════════════════════════════════════════════════════════════════════════════
# CACHED ANALYTICS / CHART EXECUTION WRAPPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _cached_analytics_bundle(
    _df: pd.DataFrame, _registry: ColumnRegistry, cache_key: str, parquet_path: Optional[str] = None
) -> ComplaintAnalyticsBundle:
    """Cached wrapper around engine.analytics.UniversalAnalyticsEngine.run_complaint_analytics.
    parquet_path is threaded through so ComplaintKPIEngine can take the
    DuckDB read_parquet() pushdown path (analytics.py Refactor Phase 2B)
    instead of the in-memory pandas replacement scan. Caller is responsible
    for only supplying a parquet_path when `_df` reflects the FULL,
    unfiltered contents of that file — see call site guard below — since
    the parquet-native accelerator queries the file directly and has no
    visibility into any pandas-side filter/drill mask applied to `_df`."""
    engine = UniversalAnalyticsEngine(_registry, parquet_path=parquet_path)
    return engine.run_complaint_analytics(_df)

@st.cache_data(show_spinner=False)
def _cached_metric_cards(
    _df: pd.DataFrame, _registry: ColumnRegistry, cache_key: str
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Cached wrapper around visualization.kpi_cards.generate_metric_cards."""
    return generate_metric_cards(_df, _registry)


@st.cache_data(show_spinner=False)
def _cached_executive_narrative(
    _df: pd.DataFrame, _registry: ColumnRegistry, cache_key: str
) -> Tuple[str, Dict[str, Any]]:
    """Cached wrapper around visualization.executive_summary.generate_executive_narrative."""
    return generate_executive_narrative(_df, _registry)


@st.cache_data(show_spinner=False)
def _cached_chart_render(
    chart_type: str,
    _df: pd.DataFrame,
    _registry: ColumnRegistry,
    combined_key: str,
    _kwargs: Dict[str, Any],
) -> Tuple[Optional[Any], Optional[pd.DataFrame], Dict[str, Any]]:
    """Cached wrapper around visualization.chart_factory.render."""
    return render_chart(chart_type, _df, _registry, **_kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 1 — EMPTY STATE
# ══════════════════════════════════════════════════════════════════════════════

def _render_empty_state() -> None:
    """Renders the Enterprise Empty State when no active dataset exists."""
    st.markdown(f"# {APP_ICON} KESCO Grid Operational Intelligence Dashboard")
    st.markdown(
        '<p style="font-weight:300; color:#64748B; font-size:0.95rem; margin-top:-8px;">'
        "Enterprise Analytics Platform for Distribution Network Monitoring, Consumer Service "
        "Intelligence, Asset Performance Management &amp; Operational Decision Support.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="kesco-card" style="text-align:center; padding: 48px 24px;">
            <h3>No Active Dataset Detected</h3>
            <p>This workspace has no ingested telemetry to analyze yet. Use the
            <strong>Unified Ingestion</strong> panel in the sidebar to upload a CSV or Excel
            export (complaint logs, feeder telemetry, billing records, HR/asset data). Once
            ingested, the platform will automatically profile the schema, detect the business
            domain, and populate this dashboard.</p>
            <p><strong>Next steps:</strong></p>
            <ol style="text-align:left; display:inline-block;">
                <li>Upload a dataset via the sidebar's Unified Ingestion panel.</li>
                <li>Confirm role mappings in the Schema Mapping Studio (Record ID, Registration
                Date, Status at minimum).</li>
                <li>Return to this page — KPIs, charts, and executive narratives will render
                automatically.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 1 — QUALITY BANNER
# ══════════════════════════════════════════════════════════════════════════════
# def _render_quality_banner(df: pd.DataFrame, registry: ColumnRegistry) -> None:
#     """
#     Data Health Card — Native Component Migration (Visual Elevation Pass).

#     Renders the six top-level dataset health metrics inside a bordered
#     st.container ("card" framing), arranged as two balanced 3-column rows,
#     each metric carrying a native help-icon tooltip (short) plus a
#     st.popover for the fuller Formula / Technical Description breakdown
#     (long-form, density-separated via st.caption). Delta indicators are
#     computed from the same already-derived values (no new calculations)
#     to signal live-tracking status without fabricating trend data that
#     doesn't exist.

#     Zero CSS: every visual affordance below (border, spacing, icon,
#     delta arrows/coloring, popover chrome) is a native Streamlit
#     component argument. This function carries no dependency on the
#     platform's injected stylesheet and is therefore immune to the
#     fragment-sandbox CSS-starvation failure mode. Never raises.
#     """
#     try:
#         domain_label, domain_confidence = st.session_state.get("domain_detection", (DOMAIN_UNKNOWN, 0.0))
#         readiness_score = st.session_state.get("readiness_score", 0)
#         readiness_band = st.session_state.get("readiness_band", "Critical")
#         total_cells = max(df.shape[0] * df.shape[1], 1)
#         missing_cells = int(df.isna().sum().sum())
#         completeness_pct = max(0.0, 100.0 - (missing_cells / total_cells * 100.0))
#         duplicate_rows = int(df.duplicated(keep="first").sum())

#         # Each metric: label, value, formula (short math), description
#         # (long-form technical explanation), delta (display string or
#         # None), delta_color ("normal" | "inverse" | "off"). Delta values
#         # are derived signals from data already computed above — never
#         # fabricated trend history.
#         metrics: List[Dict[str, Any]] = [
#             {
#                 "label": "Rows Loaded",
#                 "value": f"{len(df):,}",
#                 "formula": "COUNT(rows) in the active analytics-ready dataframe.",
#                 "description": "Total number of records currently loaded into the workspace after safe cleaning.",
#                 "delta": "Active dataset",
#                 "delta_color": "off",
#             },
#             {
#                 "label": "Structural Columns",
#                 "value": f"{df.shape[1]}",
#                 "formula": "COUNT(columns) in the active analytics-ready dataframe.",
#                 "description": "Total number of columns detected in the ingested dataset.",
#                 "delta": "Schema fixed",
#                 "delta_color": "off",
#             },
#             {
#                 "label": "Data Completeness",
#                 "value": f"{completeness_pct:.1f}%",
#                 "formula": "100 − (missing_cells / total_cells × 100)",
#                 "description": "Share of non-null cells across the entire dataframe. Lower values indicate higher sparsity.",
#                 "delta": "Healthy" if completeness_pct >= 90 else ("Fair" if completeness_pct >= 70 else "Needs review"),
#                 "delta_color": "normal" if completeness_pct >= 90 else ("off" if completeness_pct >= 70 else "inverse"),
#             },
#             {
#                 "label": "Duplicate Rows Flagged",
#                 "value": f"{duplicate_rows:,}",
#                 "formula": "COUNT(rows WHERE duplicated(keep='first'))",
#                 "description": "Rows that are exact duplicates of an earlier row. Flagged for review — never auto-removed by this display.",
#                 "delta": "None detected" if duplicate_rows == 0 else "Review recommended",
#                 "delta_color": "normal" if duplicate_rows == 0 else "inverse",
#             },
#             {
#                 "label": "Analytics Readiness",
#                 "value": f"{readiness_score}/100",
#                 "formula": "Weighted composite: schema completeness (20%) + role-mapping completeness (25%) "
#                            "+ data quality (20%) + required-field availability (20%) + hierarchy detection "
#                            "(7.5%) + date availability (7.5%).",
#                 "description": "Overall score indicating how ready this dataset is for full analytics computation.",
#                 "delta": readiness_band,
#                 "delta_color": "normal" if readiness_band in ("Excellent", "Good") else (
#                     "off" if readiness_band == "Fair" else "inverse"
#                 ),
#             },
#             {
#                 "label": "Business Domain",
#                 "value": f"{domain_label}",
#                 "formula": "Resolved via engine.domain_detection.detect_domain (role-signature scoring + filename heuristics).",
#                 "description": f"Detection confidence: {domain_confidence:.0%}. Drives KPI library selection and dashboard naming.",
#                 "delta": f"{domain_confidence:.0%} confidence",
#                 "delta_color": "normal" if domain_confidence >= 0.6 else ("off" if domain_confidence >= 0.4 else "inverse"),
#             },
#         ]

#         with st.container(border=True):
#             st.caption("DATASET HEALTH OVERVIEW")

#             # Two balanced rows of three — prevents the 6-metric crowding
#             # a single st.columns(6) row would produce on narrower widths.
#             for row_start in range(0, len(metrics), 3):
#                 row_metrics = metrics[row_start:row_start + 3]
#                 cols = st.columns(len(row_metrics))
#                 for col, m in zip(cols, row_metrics):
#                     with col:
#                         label_col, icon_col = st.columns([10, 1])
#                         with label_col:
#                             st.metric(
#                                 label=m["label"],
#                                 value=m["value"],
#                                 delta=m["delta"],
#                                 delta_color=m["delta_color"],
#                                 help=m["description"],
#                             )
#                         with icon_col:
#                             st.write("")  # vertical alignment spacer
#                             with st.popover("ℹ️", width="content"):
#                                 st.markdown(f"**{m['label']}**")
#                                 st.markdown(m["description"])
#                                 st.caption(f"Formula: `{m['formula']}`")
#     except Exception as exc:  # noqa: BLE001
#         st.caption(f"Quality banner could not be rendered: {exc}")
def _render_quality_banner(df: pd.DataFrame, registry: ColumnRegistry) -> None:
    """
    Executive Minimalist Quality Banner.
    Displays top-level KPIs in a single, clean row. 
    Technical specifications are hidden within a single expander.
    """
    # Calculation Logic
    total_cells = max(df.shape[0] * df.shape[1], 1)
    missing_cells = int(df.isna().sum().sum())
    completeness_pct = max(0.0, 100.0 - (missing_cells / total_cells * 100.0))
    duplicate_rows = int(df.duplicated(keep="first").sum())
    
    # 1. Executive Top-Level View (Clean, uncluttered)
    cols = st.columns(4)
    cols[0].metric("Rows", f"{len(df):,}")
    cols[1].metric("Completeness", f"{completeness_pct:.1f}%")
    cols[2].metric("Duplicates", f"{duplicate_rows:,}")
    cols[3].metric("Readiness", f"{st.session_state.get('readiness_score', 0)}/100")

    # 2. Progressive Disclosure (Hidden technical details)
    with st.expander("View Technical Data Quality Specifications"):
        st.caption("This section contains methodology and field-level metadata.")
        
        # Display as a clean table or list
        data = {
            "Metric": ["Rows", "Columns", "Completeness", "Duplicates", "Readiness"],
            "Formula": ["Count(rows)", "Count(cols)", "100 * (missing/total)", "Count(duplicates)", "Weighted Composite"],
            "Description": ["Total record count", "Total column count", "Non-null density", "Exact duplicate count", "Analytics readiness score"]
        }
        st.table(pd.DataFrame(data))

def _render_alert_center() -> None:
    """
    Critical Operational Alert Center — Native Component Migration.

    Replaces the prior custom-HTML three-tier alert-card layout (raw
    <div class="kesco-alert-card">... markup) with native st.error /
    st.warning / st.info calls inside a bordered, fixed-height
    st.container per severity tier. This is what resolves the 'messy
    text' symptom: native alert widgets carry their own built-in styling
    contract from Streamlit core (icon, background, border, text color)
    that is applied at the component level, not via an external
    stylesheet race. The bounded-height container makes the tier
    gracefully scrollable regardless of how many notifications are
    present, instead of pushing the page layout. Underlying severity-
    tiering logic (session_state → tiers dict) is unchanged. Never raises.
    """
    st.subheader("Critical Operational Alert Center")
    try:
        notifications: List[Dict[str, str]] = st.session_state.get("notifications", [])
        if not notifications:
            st.caption("No active notifications for the current workspace.")
            return

        tiers: Dict[str, List[Dict[str, str]]] = {"critical": [], "warning": [], "info": []}
        for note in notifications:
            severity = note.get("severity", "info")
            tiers.setdefault(severity, tiers["info"]).append(note)

        tier_labels: Dict[str, str] = {
            "critical": "Critical Alerts", "warning": "Warnings", "info": "Informational Logs",
        }
        tier_render_fn = {"critical": st.error, "warning": st.warning, "info": st.info}

        alert_cols = st.columns(3)
        for col, tier_key in zip(alert_cols, ("critical", "warning", "info")):
            with col:
                st.markdown(f"**{tier_labels[tier_key]}**")
                entries = tiers.get(tier_key, [])
                if not entries:
                    st.caption("No entries.")
                    continue
                render_fn = tier_render_fn[tier_key]
                with st.container(height=280, border=True):
                    for note in entries:
                        render_fn(
                            f"**{note.get('category', '')}**\n\n"
                            f"{note.get('message', '')}\n\n"
                            f"_{note.get('timestamp', '')}_"
                        )
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Alert Center encountered an issue and has been suppressed: {exc}")


def _log_theme_css_diagnostic() -> None:
    """Logs current theme/CSS binding state to the browser console for
    manual QA. Purely diagnostic — no DOM mutation. Never raises."""
    try:
        active_theme = st.session_state.get("theme", DEFAULT_THEME_KEY)
        st.html(f"""
        <script>
        (function() {{
            try {{
                var root = document.documentElement;
                var rootStyle = getComputedStyle(root);
                console.groupCollapsed('%c[KESCO Theme Diagnostic]', 'color:#60A5FA;font-weight:bold;');
                console.log('Expected theme (session_state):', '{active_theme}');
                console.log('data-theme attribute:', root.getAttribute('data-theme'));
                console.log('--keds-background:', rootStyle.getPropertyValue('--keds-background'));
                console.log('--keds-text:', rootStyle.getPropertyValue('--keds-text'));
                console.log('keds-theme-engine-style present:', !!document.getElementById('keds-theme-engine-style'));
                console.log('keds-root-enforcement-style present:', !!document.getElementById('keds-root-enforcement-style'));
                console.log('MutationObserver active:', !!window.__kescoObserver);
                console.groupEnd();
            }} catch (e) {{ console.warn('KESCO diagnostic failed:', e); }}
        }})();
        </script>
        """)
    except Exception:  # noqa: BLE001
        pass
# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 2 — CRITICAL OPERATIONAL ALERT CENTER
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 3 — EXECUTIVE KPI CARD ROW
# ══════════════════════════════════════════════════════════════════════════════

def _build_sparkline_series(
    monthly_trend: Optional[pd.DataFrame], value_col: str
) -> Tuple[List[float], str, Optional[bool]]:
    """Extracts a compact numeric trend series from monthly_trend for inline sparklines. Never raises."""
    try:
        if monthly_trend is None or monthly_trend.empty or value_col not in monthly_trend.columns:
            return [], "none", None
        series = pd.to_numeric(monthly_trend[value_col], errors="coerce").dropna().tolist()
        if len(series) < 2:
            return [], "none", None
        if series[-1] > series[0]:
            direction = "up"
        elif series[-1] < series[0]:
            direction = "down"
        else:
            direction = "neutral"
        return series, direction, None
    except Exception:  # noqa: BLE001
        return [], "none", None


_METRIC_ID_TO_ICON: Dict[str, str] = {
    "total_volume": "📊", "resolution_rate": "✅", "avg_processing_latency": "⏱️",
    "sla_compliance": "🛡️", "reopen_rate": "🔁", "pending_rate": "⏳", "backlog": "📥",
    "first_time_resolution_rate": "🎯", "avg_pending_age": "📆", "median_pending_age": "📆",
    "p95_pending_age": "🚨", "median_resolution_time": "⏲️", "p95_resolution_time": "🚨",
    "sla_breach_rate": "⚠️", "unique_consumers": "👥", "repeat_consumer_rate": "🔂",
    "mom_growth": "📈", "qoq_growth": "📈", "yoy_growth": "📈",
}

def _render_single_kpi_card(
    card: Dict[str, Any],
    bundle: ComplaintAnalyticsBundle,
    df_view: pd.DataFrame,
    registry: ColumnRegistry,
    theme_key: str,
) -> None:
    """
    Renders a single Executive KPI card as a CSS-only 3D flip card (Front:
    value/badge/icon | Back: formula), per static/style.css's
    .keds-kpi-card-container / .keds-kpi-card-inner / .keds-kpi-flip-front
    / .keds-kpi-flip-back rules (Milestone 13 hyper-resilient 3-layer
    architecture). Zero inline JS — the flip is pure CSS :hover on
    .keds-kpi-card-container, with FRONTEND/static/js/main.js's DOM Guard
    forcibly unclipping any Streamlit-generated wrapper ancestor at
    runtime so the 3D rotation is never squashed or clipped. Never raises.
    """
    try:
        accent = _STATUS_TIER_TO_ACCENT.get(card.get("status_tier", "NA"), "secondary")
        badge_class = _ACCENT_TO_BADGE.get(accent, "info")
        icon = _METRIC_ID_TO_ICON.get(card["metric_id"], "📌")

        kpi_key = _METRIC_ID_TO_KPI_KEY.get(card["metric_id"])
        kpi_detail: Optional[KPIResult] = bundle.kpis.get(kpi_key) if kpi_key else None

        tooltip_parts: List[str] = []
        hover_metric_html = ""
        if kpi_detail is not None:
            if kpi_detail.definition:
                tooltip_parts.append(f"Definition: {kpi_detail.definition}")
            if kpi_detail.formula:
                tooltip_parts.append(f"Formula: {kpi_detail.formula}")
            if kpi_detail.interpretation:
                tooltip_parts.append(f"Insight: {kpi_detail.interpretation}")
            if kpi_detail.recommendation:
                hover_metric_html = (
                    f'<div class="keds-kpi-hover-metric" style="font-size:0.66rem;'
                    f'color:var(--keds-secondary);margin-top:4px;">{kpi_detail.recommendation}</div>'
                )
        tooltip_text = " | ".join(tooltip_parts) if tooltip_parts else (
            f"{card['display_name']} — no additional detail available for the current scope."
        )
        tooltip_text = tooltip_text.replace('"', "'")

        back_formula_text = (
            kpi_detail.formula if kpi_detail is not None and kpi_detail.formula
            else (kpi_detail.definition if kpi_detail is not None and kpi_detail.definition else "")
        )
        if not back_formula_text:
            back_formula_text = "No formula is available for this metric in the current scope."
        back_formula_text = back_formula_text.replace('"', "'")

        # ── Milestone 13: 3-layer hyper-resilient flip-card structure ──
        # Outer .keds-kpi-card-container establishes the perspective and
        # explicit box; .keds-kpi-card-inner is the sole element that
        # rotates; .keds-kpi-flip-front / .keds-kpi-flip-back are
        # absolutely positioned, backface-hidden siblings inside it. This
        # replaces the prior structure to guarantee the rotation survives
        # Streamlit's own flexbox wrapper divs (see main.js's DOM Guard).
        st.markdown(
            f"""
            <div class="keds-kpi-card-container" title="{tooltip_text}">
             <div class="keds-kpi-card-inner">
                <div class="keds-kpi-flip-front keds-accent-{accent}">
                    <div class="keds-kpi-top-row">
                        <span class="keds-kpi-icon">{icon}</span>
                        <span class="keds-kpi-badge kesco-badge kesco-badge-{badge_class}">{card['status_tier']}</span>
                    </div>
                    <div class="keds-kpi-label">{card['display_name']}</div>
                    <div class="keds-kpi-value">{card['formatted_value']}</div>
                    <div class="keds-kpi-sparkline-slot" id="spark-anchor-{card['metric_id']}"></div>
                    {hover_metric_html}
                </div>
                <div class="keds-kpi-flip-back">
                    <div class="keds-kpi-flip-back-label">Formula — {card['display_name']}</div>
                    <div class="keds-kpi-flip-back-formula">{back_formula_text}</div>
                    <div class="keds-kpi-flip-hint">Hover to flip back</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sparkline_col_map = {"total_volume": "Total", "resolution_rate": "Closure Rate (%)"}
        spark_col = sparkline_col_map.get(card["metric_id"])
        if spark_col:
            series, direction, is_positive = _build_sparkline_series(bundle.monthly_trend, spark_col)
            if series:
                if card["metric_id"] == "resolution_rate":
                    is_positive = direction == "up"
                fig, _export_df, _meta = render_chart(
                    "sparkline", df_view, registry,
                    values=series, trend_direction=direction, trend_is_positive=is_positive,
                    height=40, width=160, theme_key=theme_key,
                )
                if fig is not None:
                    st.plotly_chart(
                        fig, width="content",
                        config={"displayModeBar": False},
                        key=f"_dash_spark_{card['metric_id']}",
                    )
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Unable to render KPI card '{card.get('display_name', '?')}': {exc}")
def _render_kpi_row(
    df_view: pd.DataFrame, registry: ColumnRegistry, cache_key: str, bundle: ComplaintAnalyticsBundle
) -> None:
    """Renders the Executive KPI Command Row. Never raises."""
    st.markdown('<div class="kesco-section-title">Executive KPI Command Row</div>', unsafe_allow_html=True)
    try:
        metric_cards, _metric_meta = _cached_metric_cards(df_view, registry, cache_key)
    except Exception as exc:  # noqa: BLE001
        incident_id = log_exception("dashboard._render_kpi_row", exc, context={"cache_key": cache_key})
        st.warning(f"KPI computation could not complete for the current scope (Reference ID: {incident_id}).")
        return

    if not metric_cards:
        st.info(
            "No KPI metrics could be computed for the current filter scope. Confirm the Record ID, "
            "Status, Registration Date, and SLA-related role mappings in the Schema Mapping Studio."
        )
        return

    theme_key = st.session_state.get("theme", DEFAULT_THEME_KEY)
    rows = [metric_cards[i:i + 4] for i in range(0, len(metric_cards), 4)]
    for row in rows:
        cols = st.columns(len(row))
        for col, card in zip(cols, row):
            with col:
                _render_single_kpi_card(card, bundle, df_view, registry, theme_key)


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 4 & 5 — PRIMARY VISUALIZATIONS & MANDATORY CHART-GRID PAIRING
# ══════════════════════════════════════════════════════════════════════════════

def _render_chart_row(
    row_title: str,
    row_key: str,
    default_chart_type: str,
    chart_type_options: Tuple[str, ...],
    df_view: pd.DataFrame,
    registry: ColumnRegistry,
    cache_key: str,
    base_kwargs: Dict[str, Any],
    group_role_candidates: Optional[Tuple[Tuple[str, str], ...]] = None,
    show_top_n: bool = True,
) -> bool:
    """
    Renders one Chart-Grid Pairing viewport. Returns True if the resulting
    visualization was eligible/rendered, False otherwise. Never raises.

    PERFORMANCE FIX: this function previously called BOTH the cached
    `_cached_chart_render()` wrapper AND the raw, uncached
    `chart_factory.render()` dispatcher for the exact same chart_type/
    kwargs — computing every chart twice on every single rerun (once
    thrown away, once used). That silently defeated the @st.cache_data
    layer on _cached_chart_render and was the actual root cause of the
    "entire dashboard reruns and takes noticeable time" symptom on every
    drill-down/'Apply Filter' click. Now there is exactly ONE render call
    per chart row, and it goes through the cached wrapper.
    """
    from utils.layout_guard import chart_block, apply_fixed_margin_chart_layout
    with chart_block(row_title):
      try:
        control_col, _spacer_col = st.columns([4, 8])
        with control_col:
            chart_type = st.selectbox(
                "Visualization Type",
                options=list(chart_type_options),
                index=list(chart_type_options).index(default_chart_type)
                if default_chart_type in chart_type_options else 0,
                key=f"_dash_chart_type_{row_key}",
            )
            render_kwargs: Dict[str, Any] = dict(base_kwargs)

            if group_role_candidates:
                resolvable_groups = [
                    (role, label) for role, label in group_role_candidates if registry.has_role(role)
                ]
                if resolvable_groups:
                    group_labels = [label for _role, label in resolvable_groups]
                    selected_label = st.selectbox(
                        "Group By", options=group_labels, index=0, key=f"_dash_groupby_{row_key}",
                    )
                    selected_role = next(
                        role for role, label in resolvable_groups if label == selected_label
                    )
                    render_kwargs["group_by"] = selected_role

            if show_top_n:
                render_kwargs["top_n"] = st.slider(
                    "Top N", min_value=5, max_value=50, value=15, step=5, key=f"_dash_topn_{row_key}",
                )

        render_kwargs["theme_key"] = st.session_state.get("theme", DEFAULT_THEME_KEY)
        combined_key = f"{cache_key}|{chart_type}|{_kwargs_signature(render_kwargs)}"

        # ── SINGLE, CACHED render call. Do not add a second uncached
        # render_chart(...) call below — that was the performance bug. ──
        fig, export_df, meta = _cached_chart_render(chart_type, df_view, registry, combined_key, render_kwargs)
        st.session_state.setdefault("_dashboard_chart_figs", {})[row_key] = fig

        left_col, right_col = st.columns([6, 6])
        with left_col:
            if fig is not None and hasattr(fig, "update_layout"):
                with st.container(border=False):
                    st.markdown(
                        f'<div style="margin:0 0 15px 0; padding:0;">'
                        f'<h4 style="margin:0; color:#1E293B; font-weight:700; font-size:1.05rem;">'
                        f'{meta.get("title", "")}'
                        f'</h4>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                fig = apply_fixed_margin_chart_layout(fig)
                st.plotly_chart(fig, width="stretch", key=f"_dash_fig_{row_key}")

                if export_df is not None and not export_df.empty:
                    render_enterprise_grid(
                        export_df, key=f"_dash_grid_{row_key}", editable=True,
                    )
                    try:
                        resolved_export_name = meta.get("export_filename") or f"{row_key}_{chart_type}.csv"
                        st.download_button(
                            "Export This Chart's Data (CSV)",
                            data=export_df.to_csv(index=False).encode("utf-8"),
                            file_name=resolved_export_name,
                            mime="text/csv",
                            key=f"_dash_export_chart_{row_key}",
                            width="stretch",
                        )
                    except Exception:  # noqa: BLE001
                        pass

                with st.expander(
                    f"View Raw Filtered Records Behind This Chart "
                    f"({len(df_view):,} row(s) in active scope)",
                    expanded=False,
                ):
                    try:
                        if df_view is not None and not df_view.empty:
                            preview_df = df_view.head(MAX_PREVIEW_ROWS)
                            render_enterprise_grid(
                                preview_df, key=f"_dash_raw_grid_{row_key}", editable=True,
                            )
                            if len(df_view) > MAX_PREVIEW_ROWS:
                                st.caption(
                                    f"Showing the first {MAX_PREVIEW_ROWS:,} of "
                                    f"{len(df_view):,} raw record(s) in the active filter/drill "
                                    f"scope. Use the export below for the complete set."
                                )
                            st.download_button(
                                "Export Raw Filtered Records for This Chart (CSV)",
                                data=df_view.to_csv(index=False).encode("utf-8"),
                                file_name=f"{row_key}_{chart_type}_raw_records.csv",
                                mime="text/csv",
                                key=f"_dash_export_raw_{row_key}",
                                width="stretch",
                            )
                        else:
                            st.caption("No raw records are available in the active filter/drill scope.")
                    except Exception as exc:  # noqa: BLE001
                        incident_id = log_exception(
                            "dashboard._render_chart_row.raw_grid", exc,
                            context={"row_key": row_key, "chart_type": chart_type},
                        )
                        st.caption(
                            f"Raw record view could not be rendered (Reference ID: {incident_id})."
                        )
            else:
                st.info(
                    meta.get("reason", "This visualization is not currently eligible for the active "
                                       "role mappings and filter scope.")
                )

        with right_col:
            st.markdown('<div class="kesco-narrative-container">', unsafe_allow_html=True)
            if fig is not None:
                interpretation = interpret_chart_output(chart_type, export_df, meta)
                summary = interpretation.get("statistical_summary", {})
                st.markdown(
                    f"**Peak:** {summary.get('peak_coordinate', 'N/A')}  \n"
                    f"**Trough:** {summary.get('trough_coordinate', 'N/A')}"
                )
                severity_badge_map = {"critical": "critical", "warning": "warning", "positive": "good", "info": "info"}
                for insight in interpretation.get("insights", []):
                    severity = insight.get("severity", "info")
                    badge_class = severity_badge_map.get(severity, "info")
                    st.markdown(
                        f'<div class="kesco-card">'
                        f'<span class="kesco-badge kesco-badge-{badge_class}">{severity.upper()}</span>'
                        f'&nbsp;&nbsp;<strong>{insight.get("headline", "")}</strong>'
                        f'<br/>{insight.get("body_text", "")}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("Narrative interpretation becomes available once this visualization is eligible.")
            st.markdown('</div>', unsafe_allow_html=True)

        return fig is not None
      except Exception as exc:  # noqa: BLE001
        incident_id = log_exception(
            "dashboard._render_chart_row",
            exc,
            context={"row_title": row_title, "row_key": row_key},
        )
        st.warning(f"'{row_title}' encountered an issue and has been degraded gracefully (Reference ID: {incident_id}).")
        return False

@st.fragment
def _render_chart_row_isolated(
    row_title: str,
    row_key: str,
    default_chart_type: str,
    chart_type_options: Tuple[str, ...],
    df_view: pd.DataFrame,
    registry: ColumnRegistry,
    cache_key: str,
    base_kwargs: Dict[str, Any],
    group_role_candidates: Optional[Tuple[Tuple[str, str], ...]] = None,
    show_top_n: bool = True,
) -> None:
    """st.fragment isolation wrapper around _render_chart_row. Never raises beyond what it already guards against."""
    inject_fragment_atomic_style(st.session_state.get("theme", DEFAULT_THEME_KEY))
    rendered = _render_chart_row(
        row_title=row_title,
        row_key=row_key,
        default_chart_type=default_chart_type,
        chart_type_options=chart_type_options,
        df_view=df_view,
        registry=registry,
        cache_key=cache_key,
        base_kwargs=base_kwargs,
        group_role_candidates=group_role_candidates,
        show_top_n=show_top_n,
    )
    try:
        flags: Dict[str, bool] = st.session_state.setdefault("_dashboard_active_viz_flags", {})
        flags[row_key] = bool(rendered)
        st.session_state["_dashboard_active_viz_flags"] = flags
    except Exception:  # noqa: BLE001
        pass


@st.fragment
def _render_geospatial_intelligence_row_isolated(
    df_view: pd.DataFrame, registry: ColumnRegistry, cache_key: str
) -> None:
    """st.fragment isolation wrapper for the Geographical Concentration Map row. Never raises."""
    inject_fragment_atomic_style(st.session_state.get("theme", DEFAULT_THEME_KEY))
    rendered = _render_geospatial_intelligence_row(df_view, registry, cache_key)
    try:
        flags: Dict[str, bool] = st.session_state.setdefault("_dashboard_active_viz_flags", {})
        flags["row4_geospatial"] = bool(rendered)
        st.session_state["_dashboard_active_viz_flags"] = flags
    except Exception:  # noqa: BLE001
        pass


@st.fragment
def _render_kpi_row_isolated(
    df_view: pd.DataFrame, registry: ColumnRegistry, cache_key: str, bundle: ComplaintAnalyticsBundle
) -> None:
    """st.fragment isolation wrapper for the Executive KPI Command Row. Never raises."""
    inject_fragment_atomic_style(st.session_state.get("theme", DEFAULT_THEME_KEY))
    _render_kpi_row(df_view, registry, cache_key, bundle)


def _render_primary_visualizations(
    df_view: pd.DataFrame, registry: ColumnRegistry, domain_label: str, cache_key: str
) -> int:
    st.session_state["_dashboard_active_viz_flags"] = {}

    _render_chart_row_isolated(
        row_title=f"Operational Concentration — {domain_label}",
        row_key="row1_concentration",
        default_chart_type="bar_grouped",
        chart_type_options=_ROW1_CHART_TYPE_OPTIONS,
        df_view=df_view,
        registry=registry,
        cache_key=cache_key,
        base_kwargs={"x_role": ROLE_CATEGORY, "aggregation": "count"},
        group_role_candidates=_ROW1_GROUP_ROLE_CANDIDATES,
        show_top_n=True,
    )

    _render_chart_row_isolated(
        row_title="Chronological Volume Trend",
        row_key="row2_trend",
        default_chart_type="line",
        chart_type_options=_ROW2_CHART_TYPE_OPTIONS,
        df_view=df_view,
        registry=registry,
        cache_key=cache_key,
        base_kwargs={
            "x_role": ROLE_REGISTRATION_DATE, "aggregation": "count",
            "date_freq": "M", "aggregation_level": "monthly",
        },
        group_role_candidates=None,
        show_top_n=False,
    )

    _render_chart_row_isolated(
        row_title=ENTERPRISE_COPY_MAP.get(
            "Friction Vector Risk & Root-Cause Analysis",
            "Friction Vector Risk & Root-Cause Analysis",
        ),
        row_key="row3_risk",
        default_chart_type="risk_matrix",
        chart_type_options=_ROW3_CHART_TYPE_OPTIONS,
        df_view=df_view,
        registry=registry,
        cache_key=cache_key,
        base_kwargs={"aggregation": "count"},
        group_role_candidates=_ROW1_GROUP_ROLE_CANDIDATES,
        show_top_n=True,
    )

    _render_geospatial_intelligence_row_isolated(df_view, registry, cache_key)

    flags: Dict[str, bool] = st.session_state.get("_dashboard_active_viz_flags", {})
    return sum(1 for is_rendered in flags.values() if is_rendered)


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 6 — EXECUTIVE ANALYTICS BLOCK & ACTION CENTER
# ══════════════════════════════════════════════════════════════════════════════

def _render_executive_block(df_view: pd.DataFrame, registry: ColumnRegistry, cache_key: str) -> None:
    """Renders the executive-ready Markdown narrative and download triggers. Never raises."""
    st.markdown(
        '<div class="kesco-section-title">Executive Narrative & Decision Support</div>',
        unsafe_allow_html=True,
    )
    try:
        markdown_report, json_payload = _cached_executive_narrative(df_view, registry, cache_key)
    except Exception as exc:  # noqa: BLE001
        incident_id = log_exception("dashboard._render_executive_block", exc, context={"cache_key": cache_key})
        st.warning(f"Executive narrative could not be generated for the current scope (Reference ID: {incident_id}).")
        return

    st.markdown(markdown_report)

    # ── _render_executive_block: dataset/JSON export download buttons ──────
# BEFORE (three st.download_button calls each with use_container_width=True)
# AFTER:
    export_col_a, export_col_b = st.columns(2)
    with export_col_a:
        try:
            if df_view is not None and not df_view.empty:
                csv_bytes = df_view.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Export Filtered Dataset (CSV)", data=csv_bytes,
                    file_name="kesco_filtered_dataset.csv", mime="text/csv",
                    key="_dash_export_csv", width="stretch",
                )
        except Exception:  # noqa: BLE001
            st.caption("CSV export is currently unavailable for this dataset.")
    with export_col_b:
        try:
            if df_view is not None and not df_view.empty:
                parquet_buffer = io.BytesIO()
                df_view.to_parquet(parquet_buffer, index=False)
                st.download_button(
                    "Export Filtered Dataset (Parquet)", data=parquet_buffer.getvalue(),
                    file_name="kesco_filtered_dataset.parquet", mime="application/octet-stream",
                    key="_dash_export_parquet", width="stretch",
                )
        except Exception:  # noqa: BLE001
            st.caption("Parquet export is unavailable in this environment (missing pyarrow/fastparquet engine).")
    

# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 7 — DYNAMIC DIAGNOSTICS & RUNTIME MONITORING FOOTER
# ══════════════════════════════════════════════════════════════════════════════

def _render_diagnostics_footer(
    full_df: pd.DataFrame,
    df_view: pd.DataFrame,
    page_start_time: float,
    analytics_duration: float,
    active_viz_count: int,
) -> None:
    """Renders dynamically computed runtime diagnostics. Never raises."""
    try:
        render_duration = time.perf_counter() - page_start_time
        calls_stats: Dict[str, int] = st.session_state.get("_dashboard_cache_key_calls", {"count": 0, "hits": 0})
        total_calls = calls_stats.get("count", 0)
        cache_ratio = (calls_stats.get("hits", 0) / total_calls * 100.0) if total_calls else 0.0
        active_filters = st.session_state.get("active_filters", {})
        drill_breadcrumbs = st.session_state.get("drill_breadcrumbs", [])

        st.markdown(
            '<div class="kesco-section-title">Runtime Diagnostics & System Telemetry</div>',
            unsafe_allow_html=True,
        )
        diag_row_a = st.columns(4)
        diag_row_a[0].metric("Page Render Duration", f"{render_duration:.2f}s")
        diag_row_a[1].metric("Analytics Execution Duration", f"{analytics_duration:.2f}s")
        diag_row_a[2].metric("Session Cache Reuse Ratio", f"{cache_ratio:.0f}%")
        diag_row_a[3].metric("Active Visualizations", f"{active_viz_count}")

        diag_row_b = st.columns(4)
        diag_row_b[0].metric("Processed Rows (Full Dataset)", f"{len(full_df):,}")
        diag_row_b[1].metric("Filtered Rows (Active Scope)", f"{len(df_view):,}")
        diag_row_b[2].metric("Active Filters", f"{len(active_filters)}")
        diag_row_b[3].metric("Current Drill Level", f"{len(drill_breadcrumbs)}")

        st.caption(
            f"KESCO Grid Operational Intelligence Dashboard · Workspace: "
            f"{st.session_state.get('workspace_name', 'Default Workspace')} · "
            f"Rendered at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Runtime diagnostics could not be fully rendered: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE ORCHESTRATION — IMMUTABLE LAYOUT PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

_page_start_time: float = time.perf_counter()

_init_dashboard_session_state()

_analytics_ready_df: Optional[pd.DataFrame] = st.session_state.get("analytics_ready_dataframe")
_registry: Optional[ColumnRegistry] = st.session_state.get("column_registry")
# Defensive instance-identity guard — catches any future code path that
# accidentally rebinds/copies the registry instead of mutating it in place.
if _registry is not None and id(_registry) != id(st.session_state.get("column_registry")):
    log_exception(
        "dashboard.registry_identity_check",
        RuntimeError("ColumnRegistry instance mismatch between local var and session_state"),
        severity="critical",
    )

if _analytics_ready_df is None or _analytics_ready_df.empty or _registry is None:
    _render_empty_state()
    st.stop()

st.markdown(f"# {APP_ICON} KESCO Grid Operational Intelligence Dashboard")
st.markdown(
    '<p style="font-weight:300; color:#64748B; font-size:0.95rem; margin-top:-8px;">'
    "Enterprise Analytics Platform for Distribution Network Monitoring, Consumer Service "
    "Intelligence, Asset Performance Management &amp; Operational Decision Support.</p>",
    unsafe_allow_html=True,
)
_render_quality_banner(_analytics_ready_df, _registry)
_render_alert_center()

filters_component.render()
_df_view: pd.DataFrame = st.session_state.get("filtered_dataframe")
if _df_view is None:
    _df_view = _analytics_ready_df

_cache_key: str = _compute_cache_key(_df_view, _registry)
_track_cache_usage(_cache_key)

# Parquet-native pushdown is only safe when df_view has NOT been narrowed
# by an active filter or drill-down — the accelerator queries
# read_parquet(parquet_path) directly and does not see any pandas-side
# mask applied to _df_view. Passing it through unconditionally would
# silently compute KPI totals against the full unfiltered file while the
# UI displays a "filtered" scope.
_effective_parquet_path = st.session_state.get("parquet_path")
_effective_duckdb_filters = _build_duckdb_filter_predicates(_registry)

_analytics_start_time = time.perf_counter()
_bundle: ComplaintAnalyticsBundle = _cached_analytics_bundle(
    _df_view, _registry, _cache_key, parquet_path=_effective_parquet_path
)
_analytics_duration: float = time.perf_counter() - _analytics_start_time

_render_kpi_row_isolated(_df_view, _registry, _cache_key, _bundle)

# ══════════════════════════════════════════════════════════════════════
# TEMPORARY RUNTIME VERIFICATION AUDIT — remove after diagnosis
# ══════════════════════════════════════════════════════════════════════
# def _print_registry_truth_diagnostic(
#     registry: ColumnRegistry,
#     bundle: ComplaintAnalyticsBundle,
#     df_view: pd.DataFrame,
# ) -> None:
#     import sys
#     from engine.analytics import _KPI_REQUIRED_ROLES, _FLEXIBLE_GROUP_A_KPIS, _FLEXIBLE_GROUP_B_KPIS

#     print("\n" + "=" * 90, file=sys.stderr)
#     print("REGISTRY TRUTH DIAGNOSTIC", file=sys.stderr)
#     print("=" * 90, file=sys.stderr)
#     print(f"registry id()               : {id(registry)}", file=sys.stderr)
#     print(f"registry.version             : {registry.version}", file=sys.stderr)
#     print(f"session_state registry id()  : {id(st.session_state.get('column_registry'))}", file=sys.stderr)
#     print(f"epoch (_registry_mutation_epoch): {st.session_state.get('_registry_mutation_epoch')}", file=sys.stderr)
#     print(f"df_view columns               : {list(df_view.columns)}", file=sys.stderr)
#     print("-" * 90, file=sys.stderr)
#     print("registry.mappings (role -> RoleMapping):", file=sys.stderr)
#     for role, mapping in sorted(registry.mappings.items()):
#         col_in_df = mapping.column_name in df_view.columns if mapping.column_name else False
#         print(
#             f"  role={role!r:28} column_name={mapping.column_name!r:25} "
#             f"confirmed={mapping.confirmed!s:5} source={mapping.source!r:20} "
#             f"resolve()={registry.resolve(role)!r:20} column_in_df_view={col_in_df}",
#             file=sys.stderr,
#         )
#     print("-" * 90, file=sys.stderr)
#     pending = st.session_state.get("_schema_pending_mappings", {})
#     print(f"UNAPPLIED STAGED MAPPINGS (_schema_pending_mappings): {pending}", file=sys.stderr)
#     print("-" * 90, file=sys.stderr)
#     for kpi_name in ("officer_productivity", "sla_breach_detail", "category_breakdown", "hierarchy_risk"):
#         required = _KPI_REQUIRED_ROLES.get(kpi_name, [])
#         group = (
#             "GROUP_A_FLEXIBLE" if kpi_name in _FLEXIBLE_GROUP_A_KPIS
#             else "GROUP_B_FLEXIBLE" if kpi_name in _FLEXIBLE_GROUP_B_KPIS
#             else "STRICT_AND"
#         )
#         eligible = kpi_name in bundle.eligibility_report.eligible
#         missing = bundle.eligibility_report.ineligible.get(kpi_name, [])
#         print(
#             f"  KPI={kpi_name!r:22} group={group!r:20} required_roles={required} "
#             f"eligible={eligible} missing_roles={missing}",
#             file=sys.stderr,
#         )
#     print(f"bundle.officer_productivity is None : {bundle.officer_productivity is None}", file=sys.stderr)
#     if bundle.officer_productivity is not None:
#         print(f"bundle.officer_productivity.empty   : {bundle.officer_productivity.empty}", file=sys.stderr)
#     print(f"bundle.sla_breach_detail is None     : {bundle.sla_breach_detail is None}", file=sys.stderr)
#     if bundle.sla_breach_detail is not None:
#         print(f"bundle.sla_breach_detail.empty       : {bundle.sla_breach_detail.empty}", file=sys.stderr)
#     print("=" * 90 + "\n", file=sys.stderr)


#_print_registry_truth_diagnostic(_registry, _bundle, _df_view)


def _render_operational_deep_dive_tables(bundle: ComplaintAnalyticsBundle, registry: ColumnRegistry) -> None:
    """Renders the ComplaintAnalyticsBundle dataframe outputs across five tabs. Never raises.

    FIX: ineligibility captions now read the actual missing_roles computed by
    MetricEligibilityEngine (via bundle.eligibility_report) instead of a
    hardcoded per-table string. This makes the true cause of a disabled table
    visible in the UI itself — no more guessing whether it's Officer, Record
    ID, or the Status/Closing-Date OR-pair that's actually unconfirmed.
    """
   # Left-aligned, moderately larger title — still built from the same
    # .kesco-section-title token used by every other section header on
    # this page (same weight/color/left-accent-bar), just bumped up from
    # the base 1.05rem to 1.2rem so it stands out slightly, with tight
    # top/bottom margins so it stays compact rather than wasting vertical
    # screen space.
    st.markdown(
        '<div class="kesco-section-title" '
        'style="margin-top:14px;margin-bottom:8px;text-align:left;font-size:1.2rem;">'
        'Operational Deep-Dive Tables</div>',
        unsafe_allow_html=True,
    )

    def _missing_roles_caption(kpi_name: str, fallback_role_label: str) -> str:
        missing = bundle.eligibility_report.ineligible.get(kpi_name)
        if not missing:
            return (
                f"{fallback_role_label} could not be computed for the current scope "
                f"(no rows matched, or an internal calculation issue occurred). "
                f"Check the Audit Trace Log for details."
            )
        display_missing = ", ".join(registry.display_name(r) for r in missing)
        return f"{fallback_role_label} is unavailable — map the following role(s): {display_missing}."

    try:
        # Full-width tabbed table block — stretches flush from the left
        # margin to the right margin of the main content container, with
        # no side-padding columns narrowing it.
        tab_officer, tab_category, tab_hierarchy, tab_repeat, tab_sla = st.tabs([
            "Officer Productivity", "Category Breakdown", "Hierarchy Risk",
            "Top Repeat Consumers", "SLA Breach Detail",
        ])
        with tab_officer:
            if bundle.officer_productivity is not None and not bundle.officer_productivity.empty:
                render_enterprise_grid(
                    bundle.officer_productivity, key="_dash_dd_officer_grid", search_columns=["Officer"],
                )
            else:
                st.caption(_missing_roles_caption("officer_productivity", "Officer Productivity"))
        with tab_category:
            if bundle.category_breakdown is not None and not bundle.category_breakdown.empty:
                render_enterprise_grid(
                    bundle.category_breakdown, key="_dash_dd_category_grid", search_columns=["Category"],
                )
            else:
                st.caption(_missing_roles_caption("category_breakdown", "Category Breakdown"))
        with tab_hierarchy:
            if bundle.hierarchy_risk is not None and not bundle.hierarchy_risk.empty:
                render_enterprise_grid(
                    bundle.hierarchy_risk, key="_dash_dd_hierarchy_grid", search_columns=["Group"],
                )
            else:
                st.caption(_missing_roles_caption("hierarchy_risk", "Hierarchy Risk"))
        with tab_repeat:
            if bundle.top_repeat_consumers is not None and not bundle.top_repeat_consumers.empty:
                render_enterprise_grid(
                    bundle.top_repeat_consumers, key="_dash_dd_repeat_grid", search_columns=["Consumer ID"],
                )
            else:
                st.caption(_missing_roles_caption("top_repeat_consumers", "Top Repeat Consumers"))
        with tab_sla:
            if bundle.sla_breach_detail is not None and not bundle.sla_breach_detail.empty:
                render_enterprise_grid(bundle.sla_breach_detail, key="_dash_dd_sla_grid")
            else:
                st.caption(_missing_roles_caption("sla_breach_detail", "SLA Breach Detail"))
    except Exception as exc:  # noqa: BLE001
        incident_id = log_exception("dashboard._render_operational_deep_dive_tables", exc)
        st.warning(f"Operational deep-dive tables could not be rendered (Reference ID: {incident_id}).")


_domain_label, _domain_confidence = st.session_state.get("domain_detection", (DOMAIN_UNKNOWN, 0.0))
_active_viz_count: int = _render_primary_visualizations(_df_view, _registry, _domain_label, _cache_key)

_render_operational_deep_dive_tables(_bundle, _registry)

_render_executive_block(_df_view, _registry, _cache_key)

_render_diagnostics_footer(
    full_df=_analytics_ready_df,
    df_view=_df_view,
    page_start_time=_page_start_time,
    analytics_duration=_analytics_duration,
    active_viz_count=_active_viz_count,
)

# def _render_operational_deep_dive_tables(bundle: ComplaintAnalyticsBundle, registry: ColumnRegistry) -> None:
#     """Renders the ComplaintAnalyticsBundle dataframe outputs across five tabs. Never raises.

#     FIX: ineligibility captions now read the actual missing_roles computed by
#     MetricEligibilityEngine (via bundle.eligibility_report) instead of a
#     hardcoded per-table string. This makes the true cause of a disabled table
#     visible in the UI itself — no more guessing whether it's Officer, Record
#     ID, or the Status/Closing-Date OR-pair that's actually unconfirmed.
#     """
#     st.markdown(
#         '<div class="kesco-section-title">Operational Deep-Dive Tables</div>',
#         unsafe_allow_html=True,
#     )

#     def _missing_roles_caption(kpi_name: str, fallback_role_label: str) -> str:
#         missing = bundle.eligibility_report.ineligible.get(kpi_name)
#         if not missing:
#             return (
#                 f"{fallback_role_label} could not be computed for the current scope "
#                 f"(no rows matched, or an internal calculation issue occurred). "
#                 f"Check the Audit Trace Log for details."
#             )
#         display_missing = ", ".join(registry.display_name(r) for r in missing)
#         return f"{fallback_role_label} is unavailable — map the following role(s): {display_missing}."

#     try:
#         tab_officer, tab_category, tab_hierarchy, tab_repeat, tab_sla = st.tabs([
#             "Officer Productivity", "Category Breakdown", "Hierarchy Risk",
#             "Top Repeat Consumers", "SLA Breach Detail",
#         ])
#         with tab_officer:
#             if bundle.officer_productivity is not None and not bundle.officer_productivity.empty:
#                 render_enterprise_grid(
#                     bundle.officer_productivity, key="_dash_dd_officer_grid", search_columns=["Officer"],
#                 )
#             else:
#                 st.caption(_missing_roles_caption("officer_productivity", "Officer Productivity"))
#         with tab_category:
#             if bundle.category_breakdown is not None and not bundle.category_breakdown.empty:
#                 render_enterprise_grid(
#                     bundle.category_breakdown, key="_dash_dd_category_grid", search_columns=["Category"],
#                 )
#             else:
#                 st.caption(_missing_roles_caption("category_breakdown", "Category Breakdown"))
#         with tab_hierarchy:
#             if bundle.hierarchy_risk is not None and not bundle.hierarchy_risk.empty:
#                 render_enterprise_grid(
#                     bundle.hierarchy_risk, key="_dash_dd_hierarchy_grid", search_columns=["Group"],
#                 )
#             else:
#                 st.caption(_missing_roles_caption("hierarchy_risk", "Hierarchy Risk"))
#         with tab_repeat:
#             if bundle.top_repeat_consumers is not None and not bundle.top_repeat_consumers.empty:
#                 render_enterprise_grid(
#                     bundle.top_repeat_consumers, key="_dash_dd_repeat_grid", search_columns=["Consumer ID"],
#                 )
#             else:
#                 st.caption(_missing_roles_caption("top_repeat_consumers", "Top Repeat Consumers"))
#         with tab_sla:
#             if bundle.sla_breach_detail is not None and not bundle.sla_breach_detail.empty:
#                 render_enterprise_grid(bundle.sla_breach_detail, key="_dash_dd_sla_grid")
#             else:
#                 st.caption(_missing_roles_caption("sla_breach_detail", "SLA Breach Detail"))
#     except Exception as exc:  # noqa: BLE001
#         incident_id = log_exception("dashboard._render_operational_deep_dive_tables", exc)
#         st.warning(f"Operational deep-dive tables could not be rendered (Reference ID: {incident_id}).")


# _domain_label, _domain_confidence = st.session_state.get("domain_detection", (DOMAIN_UNKNOWN, 0.0))
# _active_viz_count: int = _render_primary_visualizations(_df_view, _registry, _domain_label, _cache_key)

# _render_operational_deep_dive_tables(_bundle, _registry)

# _render_executive_block(_df_view, _registry, _cache_key)

# _render_diagnostics_footer(
#     full_df=_analytics_ready_df,
#     df_view=_df_view,
#     page_start_time=_page_start_time,
#     analytics_duration=_analytics_duration,
#     active_viz_count=_active_viz_count,
# )