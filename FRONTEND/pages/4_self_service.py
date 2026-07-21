"""
FRONTEND/pages/4_self_service.py — Self-Service / Ad-Hoc Analytics Builder
(Milestone 4 / Issue 9 remediation)

Presentation-Layer Orchestrator ONLY, matching the exact zero-business-logic
pattern established by 1_dashboard.py, 2_audit.py, and 3_schema_mapping.py.
This page introduces NO new business logic, NO new KPI computation, and NO
new chart-rendering logic — it is a freeform, no-code parameterization
surface over the ALREADY-EXISTING, fully self-service-ready
visualization.chart_factory.CHART_REGISTRY / render() dispatcher (19
registered chart types, each already accepting x_role, y_role, group_by,
color_role, size_role, facet_role, aggregation, top_n, sort_order, and
date_range kwargs per the main_prompt.txt SELF-SERVICE ANALYTICS mandate),
and over visualization.chart_interpreter.interpret_chart_output for the
automated narrative panel.

Zero hardcoded column names: every role selector below is populated
exclusively from the active core.column_registry.ColumnRegistry's confirmed
mappings. Every KPI/chart function this page invokes resolves columns
through that same registry — nothing here ever references a literal
DataFrame column name.

The "geographical_concentration_map" chart type is intentionally excluded
from this builder: it returns a folium.Map rather than a go.Figure/px chart
and is handled by its own dedicated dashboard row
(1_dashboard.py::_render_geospatial_intelligence_row) per the Milestone 3
remediation — mixing that return type into this generic Plotly-oriented
builder would require special-casing the render/display path, which is
out of scope for this specific issue.
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, List, Optional, Tuple
from utils.error_logging import log_exception

import pandas as pd
import streamlit as st

from core.column_registry import ColumnRegistry
from core.settings import APP_ICON, MAX_PREVIEW_ROWS
from core.themes import DEFAULT_THEME_KEY
from visualization.chart_factory import CHART_REGISTRY, render as render_chart
from visualization.chart_interpreter import interpret_chart_output

_PAGE_NAME: str = "self_service"

# Chart types excluded from this generic Plotly-oriented builder because
# they return a non-Plotly object (folium.Map) or require a specialized,
# non-role-based input contract (raw `values` list) rather than the
# standard role-parameterized surface this page exposes.
_EXCLUDED_CHART_TYPES: frozenset = frozenset({"geographical_concentration_map", "sparkline"})

_AVAILABLE_CHART_TYPES: List[str] = sorted(
    ct for ct in CHART_REGISTRY.keys() if ct not in _EXCLUDED_CHART_TYPES
)

# Mirrors visualization.chart_factory.py's private `_VALID_AGGREGATIONS`
# constant. Reproduced here (rather than importing a leading-underscore,
# non-public module attribute across a module boundary) since it is a
# static UI-facing enumeration, not business logic — the actual
# aggregation execution remains 100% owned by chart_factory.py's
# `_aggregate()` helper, which this page never re-implements.
_AGGREGATION_OPTIONS: Tuple[str, ...] = (
    "count", "sum", "mean", "median", "min", "max", "nunique"
)
_SORT_ORDER_OPTIONS: Tuple[str, ...] = ("desc", "asc")

_NONE_OPTION: str = "— None —"

_ROLE_KWARG_LABELS: Tuple[Tuple[str, str], ...] = (
    ("x_role", "X-Axis Role"),
    ("y_role", "Y-Axis Role (Measure)"),
    ("group_by", "Group By / Color Role"),
    ("color_role", "Color Role (if distinct from Group By)"),
    ("size_role", "Size Role (Bubble Charts)"),
    ("facet_role", "Facet Role"),
)


def _init_self_service_state() -> None:
    """Idempotently seeds this page's local UI-only session_state keys.
    Never disturbs the platform-wide contract owned by FRONTEND.app. Never
    raises."""
    st.session_state.setdefault("_self_service_last_signature", None)


def _confirmed_role_options(registry: ColumnRegistry) -> List[Tuple[str, str]]:
    """
    Returns every role currently confirmed (i.e., resolvable) in the active
    Column Registry as (role, display_name) pairs, sorted by display name.
    This is the SOLE source of selectable roles in every dropdown below —
    guaranteeing this page can never reference a role that does not
    actually resolve to a live column, per the Column Registry & Fail-Safe
    Contract. Never raises.
    """
    try:
        options: List[Tuple[str, str]] = []
        for role, mapping in registry.mappings.items():
            if mapping.confirmed and mapping.column_name:
                options.append((role, registry.display_name(role)))
        return sorted(options, key=lambda pair: pair[1])
    except Exception:  # noqa: BLE001
        return []


def _role_selectbox(
    kwarg_name: str,
    label: str,
    role_options: List[Tuple[str, str]],
    key_suffix: str,
) -> Optional[str]:
    """Renders a single role-selection dropdown (role -> canonical role
    string, or None if unselected) sourced exclusively from
    `role_options`. Never raises; returns None on any internal failure."""
    try:
        display_labels = [_NONE_OPTION] + [label for _role, label in role_options]
        selected_label = st.selectbox(
            label, options=display_labels, index=0, key=f"_ss_{key_suffix}",
        )
        if selected_label == _NONE_OPTION:
            return None
        for role, disp in role_options:
            if disp == selected_label:
                return role
        return None
    except Exception:  # noqa: BLE001
        return None


def _build_scope_signature(
    df: pd.DataFrame, registry: ColumnRegistry, chart_type: str, render_kwargs: Dict[str, Any]
) -> str:
    try:
        filters_sig = json.dumps(st.session_state.get("active_filters", {}), sort_keys=True, default=str)
    except Exception:  # noqa: BLE001
        filters_sig = str(st.session_state.get("active_filters", {}))
    try:
        breadcrumbs_sig = json.dumps(st.session_state.get("drill_breadcrumbs", []), sort_keys=True, default=str)
    except Exception:  # noqa: BLE001
        breadcrumbs_sig = str(st.session_state.get("drill_breadcrumbs", []))
    try:
        kwargs_sig = json.dumps(render_kwargs, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001
        kwargs_sig = str(sorted(render_kwargs.items(), key=lambda kv: kv[0]))
    mutation_epoch = st.session_state.get("_registry_mutation_epoch", 0)
    return (
        f"{len(df)}|{df.shape[1]}|{registry.version}|{mutation_epoch}|{filters_sig}|"
        f"{breadcrumbs_sig}|{chart_type}|{kwargs_sig}"
    )

@st.cache_data(show_spinner=False)
def _cached_self_service_render(
    chart_type: str,
    _df: pd.DataFrame,
    _registry: ColumnRegistry,
    scope_signature: str,
    _render_kwargs: Dict[str, Any],
) -> Tuple[Optional[Any], Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Cached wrapper around visualization.chart_factory.render, mirroring
    1_dashboard.py::_cached_chart_render's caching contract exactly:
    `scope_signature` is the sole hashed parameter; `_df`, `_registry`, and
    `_render_kwargs` are excluded from hashing via the leading-underscore
    convention. Introduces zero new rendering logic — delegates 100% to
    the already-production chart_factory.render dispatcher.
    """
    return render_chart(chart_type, _df, _registry, **_render_kwargs)


def _render_empty_state() -> None:
    """Renders the Enterprise Empty State when no active dataset exists,
    mirroring the corrective-action pattern established by
    1_dashboard.py::_render_empty_state. Never raises."""
    st.markdown(f"# {APP_ICON} Self-Service Analytics Builder")
    st.markdown(
        '<p style="font-weight:300; color:#64748B; font-size:0.95rem; margin-top:-8px;">'
        "No-code, freeform chart construction over the active dataset — select roles, "
        "aggregation, and chart type to generate a visualization, summary table, and "
        "executive insight instantly.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="kesco-card" style="text-align:center; padding: 48px 24px;">
            <h3>No Active Dataset Detected</h3>
            <p>Upload a dataset via the sidebar's <strong>Unified Ingestion</strong> panel and confirm
            at least one role mapping in the <strong>Schema Mapping Studio</strong> to begin building
            ad-hoc visualizations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_self_service_builder(df_view: pd.DataFrame, registry: ColumnRegistry) -> None:
    """
    Renders the complete freeform chart-builder control strip, invokes the
    selected chart_factory chart type with the assembled kwargs, and
    displays the resulting (figure, aggregated_df, metadata) triple plus
    its automated narrative interpretation — exactly mirroring the display
    pattern already established by 1_dashboard.py::_render_chart_row, but
    with a fully freeform (not per-row-fixed) role/chart-type surface.
    Never raises.
    """
    role_options = _confirmed_role_options(registry)
    if not role_options:
        st.warning(
            "No confirmed role mappings exist for the active dataset. Confirm at least one role "
            "in the Schema Mapping Studio before building an ad-hoc visualization."
        )
        return

    st.markdown('<div class="kesco-section-title">Chart Configuration</div>', unsafe_allow_html=True)

    control_col_a, control_col_b, control_col_c = st.columns(3)
    with control_col_a:
        chart_type = st.selectbox(
            "Chart Type", options=_AVAILABLE_CHART_TYPES,
            index=_AVAILABLE_CHART_TYPES.index("bar") if "bar" in _AVAILABLE_CHART_TYPES else 0,
            key="_ss_chart_type",
        )
    with control_col_b:
        aggregation = st.selectbox("Aggregation", options=list(_AGGREGATION_OPTIONS), index=0, key="_ss_aggregation")
    with control_col_c:
        sort_order = st.selectbox("Sort Order", options=list(_SORT_ORDER_OPTIONS), index=0, key="_ss_sort_order")

    st.markdown("**Role Assignment**")
    role_cols = st.columns(3)
    render_kwargs: Dict[str, Any] = {"aggregation": aggregation, "sort_order": sort_order}
    for idx, (kwarg_name, label) in enumerate(_ROLE_KWARG_LABELS):
        with role_cols[idx % 3]:
            resolved_role = _role_selectbox(kwarg_name, label, role_options, key_suffix=kwarg_name)
            if resolved_role:
                render_kwargs[kwarg_name] = resolved_role

    control_col_d, control_col_e = st.columns(2)
    with control_col_d:
        top_n = st.slider("Top N", min_value=0, max_value=100, value=15, step=5, key="_ss_top_n")
        render_kwargs["top_n"] = top_n
    with control_col_e:
        date_role = next((r for r, _l in role_options if registry.display_name(r) == "Registration / Open Date"), None)
        if date_role:
            date_col = registry.resolve(date_role)
            if date_col and date_col in df_view.columns:
                parsed_dates = pd.to_datetime(df_view[date_col], errors="coerce").dropna()
                if not parsed_dates.empty:
                    min_date = parsed_dates.min().date()
                    max_date = parsed_dates.max().date()
                    selected_range = st.date_input(
                        "Date Range Filter", value=(min_date, max_date),
                        min_value=min_date, max_value=max_date, key="_ss_date_range",
                    )
                    if isinstance(selected_range, tuple) and len(selected_range) == 2:
                        start_ts = pd.Timestamp(selected_range[0])
                        end_ts = pd.Timestamp(selected_range[1])
                        render_kwargs["date_range"] = (start_ts, end_ts)

    render_kwargs["theme_key"] = st.session_state.get("theme", DEFAULT_THEME_KEY)

    scope_signature = _build_scope_signature(df_view, registry, chart_type, render_kwargs)

    try:
        fig, export_df, meta = _cached_self_service_render(
            chart_type, df_view, registry, scope_signature, render_kwargs
        )
    except Exception as exc:  # noqa: BLE001
        incident_id = log_exception(
            "self_service._render_self_service_builder", exc,
            context={"chart_type": chart_type, "render_kwargs": render_kwargs},
        )
        st.warning(f"Chart construction could not complete (Reference ID: {incident_id}).")
        return

    st.divider()
    from utils.layout_guard import chart_block, apply_fixed_margin_chart_layout
    with chart_block("Generated Visualization"):
      left_col, right_col = st.columns([6, 6])
      with left_col:
            if fig is not None and hasattr(fig, "update_layout"):
                fig = apply_fixed_margin_chart_layout(fig)
                st.plotly_chart(fig, width="stretch", key="_ss_fig")
                if export_df is not None and not export_df.empty:
                    st.data_editor(
                        export_df, width="stretch", hide_index=True,
                        key="_ss_grid", disabled=True,
                    )
                    try:
                        resolved_export_name = meta.get("export_filename") or f"self_service_{chart_type}.csv"
                        st.download_button(
                            "Export Summary Table (CSV)",
                            data=export_df.to_csv(index=False).encode("utf-8"),
                            file_name=resolved_export_name,
                            mime="text/csv",
                            key="_ss_export_summary",
                            width="stretch",
                        )
                    except Exception:  # noqa: BLE001
                        pass

            with st.expander(
                f"View Raw Filtered Records Behind This Chart ({len(df_view):,} row(s) in active scope)",
                expanded=False,
            ):
                try:
                    if not df_view.empty:
                        st.data_editor(
                            df_view.head(MAX_PREVIEW_ROWS), width="stretch",
                            hide_index=True, key="_ss_raw_grid", disabled=True,
                        )
                        if len(df_view) > MAX_PREVIEW_ROWS:
                            st.caption(
                                f"Showing the first {MAX_PREVIEW_ROWS:,} of {len(df_view):,} raw record(s)."
                            )
                        st.download_button(
                            "Export Raw Filtered Records (CSV)",
                            data=df_view.to_csv(index=False).encode("utf-8"),
                            file_name=f"self_service_{chart_type}_raw_records.csv",
                            mime="text/csv",
                            key="_ss_export_raw",
                            width="stretch",
                        )
                    else:
                        st.caption("No raw records are available in the active filter/drill scope.")
                except Exception:  # noqa: BLE001
                    st.caption("Raw record view could not be rendered.")
                else:
                    st.info(
                    meta.get("reason", "This visualization is not currently eligible for the selected "
                                   "role assignment. Adjust the role/chart-type configuration above.")
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


# ══════════════════════════════════════════════════════════════════════════════
# PAGE ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

_init_self_service_state()

_analytics_ready_df: Optional[pd.DataFrame] = st.session_state.get("analytics_ready_dataframe")
_registry: Optional[ColumnRegistry] = st.session_state.get("column_registry")

if _analytics_ready_df is None or _analytics_ready_df.empty or _registry is None:
    _render_empty_state()
    st.stop()

st.markdown(f"# {APP_ICON} Self-Service Analytics Builder")
st.markdown(
    '<p style="font-weight:300; color:#64748B; font-size:0.95rem; margin-top:-8px;">'
    "No-code, freeform chart construction over the active dataset — select roles, aggregation, "
    "and chart type to generate a visualization, summary table, and executive insight instantly.</p>",
    unsafe_allow_html=True,
)

_df_view: pd.DataFrame = st.session_state.get("filtered_dataframe")
if _df_view is None or _df_view.empty:
    _df_view = _analytics_ready_df
st.caption(
    f"Operating on {len(_df_view):,} row(s) from the active Global Filter / Drill-Down scope "
    f"(set on the Grid Operational Dashboard page)."
)

_render_self_service_builder(_df_view, _registry)