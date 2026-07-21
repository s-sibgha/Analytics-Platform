"""
components/filters.py — Dynamic Multi-Variable Scoping & Cascading Drill
Panel (FILE 3 / 6)

Implements the Universal Component Interface (render / validate / refresh /
export / metadata) for the platform's Global Filter Engine and Synchronized
Hierarchical Drilling module:

  • Global Filter Engine: date range, status, category, officer, and any
    other resolvable categorical role, applied against the active
    analytics-ready dataset and materialized into
    st.session_state['filtered_dataframe'] as a boolean-masked view (never
    a deep copy) of the immutable analytics_ready_dataframe reference.
  • Cascading Drill Panel: hierarchical Zone -> Circle -> Division ->
    Subdivision -> Feeder -> Transformer -> Consumer navigation, with each
    level's option list dynamically constrained by the parent level's
    current selection. Only levels whose role is actually resolvable via
    the Column Registry are rendered — the panel never fabricates a
    hierarchy level that does not exist in the active dataset.
  • Context Preservation: selected filters and drill path are persisted in
    st.session_state['active_filters'] and st.session_state
    ['drill_breadcrumbs'] respectively, surviving reruns and page
    navigation without forcing the user to re-select scope.

This module performs NO business KPI computation and NO chart rendering —
it exclusively narrows the working dataset via boolean masks and exposes
that narrowed view through the mandated session_state contract for
pages/1_dashboard.py and pages/2_audit.py to consume.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Tuple
from utils.error_logging import log_exception

import numpy as np
import streamlit as st
import pandas as pd
from core.column_registry import ColumnRegistry
from core.roles import (
    ROLE_REGISTRATION_DATE,
    ROLE_STATUS,
    ROLE_CATEGORY,
    ROLE_SUBCATEGORY,
    ROLE_OFFICER,
    ROLE_PRIORITY,
    ROLE_ZONE,
    ROLE_CIRCLE,
    ROLE_DIVISION,
    ROLE_SUBDIVISION,
    ROLE_FEEDER,
    ROLE_TRANSFORMER,
    ROLE_SUBSTATION,
    ROLE_CONSUMER_ID,
    CANONICAL_DRILL_HIERARCHY,
)

_COMPONENT_NAME: str = "filters"

# Ordered hierarchy definition for the Synchronized Hierarchical Drilling
# module: (role, display_label). Only entries whose role resolves to an
# existing, non-empty column in the active dataset are rendered.
#
# MILESTONE 1 / ISSUE 2, 7, 15 REMEDIATION: this tuple is no longer an
# independently-declared local constant. It now references
# core.roles.CANONICAL_DRILL_HIERARCHY — the single source of truth also
# consumed by engine/analytics.py, visualization/chart_factory.py, and
# FRONTEND/components/schema_mapping.py — eliminating the prior
# Feeder/Transformer ordering contradiction and the missing Substation tier.
_DRILL_HIERARCHY: Tuple[Tuple[str, str], ...] = CANONICAL_DRILL_HIERARCHY

# Standard categorical filter roles offered by the Global Filter Engine,
# independent of the geographic drill hierarchy.
_STANDARD_FILTER_ROLES: Tuple[Tuple[str, str], ...] = (
    (ROLE_STATUS, "Status"),
    (ROLE_CATEGORY, "Category"),
    (ROLE_SUBCATEGORY, "Subcategory"),
    (ROLE_OFFICER, "Officer"),
    (ROLE_PRIORITY, "Priority"),
)

_ALL_OPTION: str = "All"
_MAX_MULTISELECT_OPTIONS: int = 500
_HIGH_CARDINALITY_FILTER_THRESHOLD: int = 2000  # above this, degrade the widget to search-mode


_WIDGET_KEY_PREFIX: str = "_filters_"


def _purge_filter_widget_state() -> None:
    """
    Deletes every Streamlit widget-bound session_state key owned by this
    component's render() (identified by the `_filters_` key prefix used
    across every widget instantiated there: date range, categorical
    multiselect/search inputs, drill-hierarchy selectboxes, and the
    Apply/Reset/Clear-Drill-Path action buttons).

    REQUIRED because Streamlit widgets persist their current value under
    their explicit `key=` across reruns independent of the
    `value=`/`default=`/`index=` parameter supplied at construction — that
    parameter is honored only the FIRST time a given key is instantiated.
    Without this purge, refresh() clearing active_filters/drill_breadcrumbs
    is silently undone within the same render pass: render() re-reads each
    still-resident widget key on the next rerun and repopulates
    active_filters/drill_breadcrumbs from that stale value before the
    dataset is ever re-filtered (Milestone 2 / Issue 5 remediation).

    Never raises: iterates over a materialized key list (so mutation
    during iteration is impossible) and defensively swallows any per-key
    deletion failure.
    """
    try:
        stale_keys: List[str] = [
            key for key in list(st.session_state.keys())
            if isinstance(key, str) and key.startswith(_WIDGET_KEY_PREFIX)
        ]
        for key in stale_keys:
            try:
                del st.session_state[key]
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        return
@dataclass
class FilterEngineResult:
    """Structured outcome of applying the Global Filter Engine + Cascading
    Drill Panel to the active dataset."""
    filtered_df: pd.DataFrame
    rows_before: int
    rows_after: int
    active_filters: Dict[str, Any]
    drill_breadcrumbs: List[Dict[str, str]]
    applied_roles: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS — ROLE RESOLUTION & OPTION BUILDING
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_role_column(
    registry: ColumnRegistry, df: pd.DataFrame, role: str
) -> Optional[str]:
    """Resolves a role to a live column name present in `df`. Returns None
    (never raises) if the role is unmapped or the resolved column is
    missing from the current dataframe view."""
    try:
        col = registry.resolve(role)
    except Exception:  # noqa: BLE001
        return None
    if col and col in df.columns:
        return col
    return None



def _safe_unique_options(series: pd.Series, limit: int = _MAX_MULTISELECT_OPTIONS) -> List[str]:
    """
    Returns a sorted, de-duplicated, string-coerced, NaN-free list of
    unique values from `series`, capped at `limit` entries. Never sorts
    more than `limit` values: on high-cardinality columns, sorting the
    FULL unique set on every Streamlit rerun (which fires on ANY widget
    interaction anywhere on the page, not just a filter change) was a
    measured O(n log n) performance bottleneck. The top-`limit` most
    frequent values are now selected via a single vectorized
    value_counts().nlargest() pass, and only that bounded subset is ever
    sorted. Never raises.
    """
    try:
        non_null = series.dropna()
        if non_null.empty:
            return []
        values = non_null.astype(str).str.strip()
        values = values[values != ""]
        if values.empty:
            return []
        if values.nunique() <= limit:
            return sorted(values.unique().tolist())
        top_values = values.value_counts().nlargest(limit).index.tolist()
        return sorted(str(v) for v in top_values)
    except Exception:  # noqa: BLE001
        return []


def _available_drill_levels(
    registry: ColumnRegistry, df: pd.DataFrame
) -> List[Tuple[str, str, str]]:
    """Returns the subset of `_DRILL_HIERARCHY` that is actually resolvable
    and non-empty in the active dataset, as (role, label, column_name)
    triples, preserving the mandated Zone->Circle->...->Consumer order."""
    levels: List[Tuple[str, str, str]] = []
    for role, label in _DRILL_HIERARCHY:
        col = _resolve_role_column(registry, df, role)
        if col is not None and df[col].notna().any():
            levels.append((role, label, col))
    return levels


def _available_standard_filters(
    registry: ColumnRegistry, df: pd.DataFrame
) -> List[Tuple[str, str, str]]:
    """Returns the subset of `_STANDARD_FILTER_ROLES` that is resolvable and
    non-empty in the active dataset, as (role, label, column_name) triples."""
    filters: List[Tuple[str, str, str]] = []
    for role, label in _STANDARD_FILTER_ROLES:
        col = _resolve_role_column(registry, df, role)
        if col is not None and df[col].notna().any():
            filters.append((role, label, col))
    return filters


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS — MASK CONSTRUCTION (VECTORIZED, NON-MUTATING)
# ══════════════════════════════════════════════════════════════════════════════

def _robust_parse_dates(series: pd.Series) -> pd.Series:
    """
    Cloud-hardened, multi-year, multi-format date parser.

    - Bypasses re-parsing entirely for columns that already arrived as
      native datetime64 (e.g. a Parquet-origin column) — avoids the
      pandas 'Could not infer format' UserWarning and unnecessary work.
    - Normalizes non-standard delimiters (e.g. '02_DEC_2024' ->
      '02-DEC-2024') before parsing, since dateutil does not recognize
      underscores as date separators and would otherwise coerce these
      to NaT even under format='mixed'.
    - Uses format='mixed' so a single column may legally contain several
      distinct formats simultaneously (e.g. '2023-01-01' alongside
      '07-JUNE-2026'), which pandas >= 2.0 otherwise rejects with
      "time data ... doesn't match format" unless format='mixed' is set.

    Never raises: any unparseable value degrades to NaT via
    errors='coerce'.
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    try:
        normalized = series.astype(str).str.strip().str.replace("_", "-", regex=False)
        try:
            return pd.to_datetime(normalized, errors="coerce", format="mixed")
        except (ValueError, TypeError):
            return pd.to_datetime(normalized, errors="coerce")
    except Exception:  # noqa: BLE001
        return pd.to_datetime(series, errors="coerce")


def _apply_date_filter_mask(
    df: pd.DataFrame,
    date_col: Optional[str],
    date_range: Optional[Tuple[Optional[date], Optional[date]]],
) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if not date_col or date_col not in df.columns or not date_range:
        return mask
    start, end = date_range
    if start is None and end is None:
        return mask
    try:
        parsed = _robust_parse_dates(df[date_col])
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_localize(None)
        if start is not None:
            mask &= parsed >= pd.Timestamp(start)
        if end is not None:
            mask &= parsed <= (pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
        mask = mask.fillna(False)
        return mask
    except Exception:  # noqa: BLE001
        return pd.Series(True, index=df.index)

def _apply_categorical_filter_mask(
    df: pd.DataFrame,
    col: Optional[str],
    selected_values: Optional[List[str]],
) -> pd.Series:
    """Builds a boolean keep-mask for a multi-select categorical filter.
    Returns an all-True mask when no selection has been made or the column
    is not resolvable. Never raises."""
    mask = pd.Series(True, index=df.index)
    if not col or col not in df.columns or not selected_values:
        return mask
    try:
        normalized = df[col].astype(str).str.strip()
        mask = normalized.isin(selected_values)
        return mask.fillna(False)
    except Exception:  # noqa: BLE001
        return pd.Series(True, index=df.index)


# ══════════════════════════════════════════════════════════════════════════════
# STATIC TOPOLOGY CACHE — pre-built in-memory hierarchy map
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _build_topology_map(
    _df: pd.DataFrame,
    level_cols: Tuple[str, ...],
    cache_key: str,
) -> Dict[str, Any]:
    """
    Builds a fully in-memory nested dictionary topology from the active
    dataset's resolved hierarchy columns, e.g.:

        {"Zone_A": {"Circle_1": {"Division_X": {"Subdiv_Y":
            {"Substation_Z": {"Feeder_1": {"Consumer_1": {}, "Consumer_2": {}}}}}}}}

    Every level — including the deepest — is a uniform dict (leaf values
    map to `{}`), so dropdown option resolution never has to branch on
    "is this a dict or a list": it is always `sorted(node.keys())`.

    Cached on `cache_key` (registry version + mutation epoch + row count +
    the exact tuple of hierarchy columns), so this scans the raw
    dataframe ONCE per ingestion/mapping change — never on a per-dropdown
    interaction. Every selectbox in the drill-down fragment below reads
    exclusively from this structure; it never touches `_df` again.

    Never raises: any internal failure degrades to an empty map (the
    fragment renders "no drill-down levels available" rather than
    crashing the host page).
    """
    try:
        if not level_cols or _df is None or _df.empty:
            return {}

        def _recurse(frame: pd.DataFrame, cols: List[str]) -> Dict[str, Any]:
            if not cols:
                return {}
            col, rest = cols[0], cols[1:]
            normalized = frame[col].astype(str).str.strip()
            node: Dict[str, Any] = {}
            for value in sorted(normalized.dropna().unique()):
                if not value or value.lower() in ("nan", "none", "nat", ""):
                    continue
                sub_frame = frame.loc[normalized == value]
                node[value] = _recurse(sub_frame, rest)
            return node

        return _recurse(_df[list(level_cols)].copy(), list(level_cols))
    except Exception:  # noqa: BLE001
        return {}


def _resolve_topology_node(
    topology: Dict[str, Any],
    level_roles: List[str],
    temp_filters: Dict[str, str],
    upto_index: int,
) -> Optional[Dict[str, Any]]:
    """
    Walks the cached topology map down to the node representing "everything
    selected so far" (levels 0..upto_index-1 in `temp_filters`), so the
    current level's selectbox can list exactly the children that actually
    exist under that path — without ever re-touching the raw dataframe.
    Returns None if a required ancestor selection is missing or the path
    no longer resolves (e.g. an ancestor's value was cleared). Never raises.
    """
    try:
        node: Any = topology
        for j in range(upto_index):
            role_j = level_roles[j]
            value = temp_filters.get(role_j)
            if value is None or not isinstance(node, dict) or value not in node:
                return None
            node = node[value]
        return node if isinstance(node, dict) else None
    except Exception:  # noqa: BLE001
        return None
    
def _on_drill_level_change(role: str, deeper_roles: List[str], widget_key: str) -> None:
    """
    Widget on_change handler for a single drill-tier selectbox. Writes
    ONLY into `st.session_state["temp_filters"]` (the staged/uncommitted
    path) — never into `applied_filters` or `drill_breadcrumbs`, and never
    calls st.rerun(). Because this selectbox lives inside an @st.fragment,
    Streamlit's own widget-interaction rerun is already scoped to just
    that fragment — no explicit rerun call is needed or wanted here.

    Also purges every deeper tier's staged value AND its own widget key,
    so the now-stale child selectboxes reset to "All" on the fragment's
    next render instead of retaining a value that no longer matches the
    newly narrowed path (the same widget-key-persistence pitfall
    documented elsewhere in this file for the categorical filters).
    """
    try:
        temp_filters: Dict[str, str] = dict(st.session_state.get("temp_filters", {}))
        selected = st.session_state.get(widget_key, _ALL_OPTION)
        if selected == _ALL_OPTION:
            temp_filters.pop(role, None)
        else:
            temp_filters[role] = selected
        for deeper_role in deeper_roles:
            temp_filters.pop(deeper_role, None)
            st.session_state.pop(f"_drill_frag_{deeper_role}", None)
        st.session_state["temp_filters"] = temp_filters
    except Exception:  # noqa: BLE001
        pass


def _on_clear_staged_drill_path(level_roles: List[str]) -> None:
    """
    Clears ONLY the staged path (`temp_filters` + every drill widget key).
    Does NOT touch `applied_filters` or `drill_breadcrumbs` — the
    currently active dashboard scope remains untouched until the user
    explicitly clicks Apply Filters again. This is a fragment-scoped
    mutation; no st.rerun() call is required (the button click itself
    already triggers the fragment's own automatic rerun).
    """
    try:
        st.session_state["temp_filters"] = {}
        for role in level_roles:
            st.session_state.pop(f"_drill_frag_{role}", None)
    except Exception:  # noqa: BLE001
        pass


def _on_apply_drill_filters(level_roles_labels: List[Tuple[str, str]]) -> None:
    """
    Commits `st.session_state["temp_filters"]` -> `st.session_state
    ["applied_filters"]`, and translates it into the exact
    `drill_breadcrumbs` schema `apply_filters()` / `_apply_drill_path_mask`
    already expect (role/label/value dicts, in hierarchy order) — so
    every downstream masking function in this file is completely
    untouched by this refactor. Purges the visualization/analytics caches
    so the committed scope is guaranteed fresh on the next chart render.
    Never raises.
    """
    try:
        temp_filters: Dict[str, str] = dict(st.session_state.get("temp_filters", {}))
        st.session_state["applied_filters"] = temp_filters
        st.session_state["drill_breadcrumbs"] = [
            {"role": role, "label": label, "value": temp_filters[role]}
            for role, label in level_roles_labels
            if role in temp_filters
        ]
        st.session_state["visualization_cache"] = {}
        st.session_state["analytics_results"] = {}
    except Exception:  # noqa: BLE001
        pass

# --- Fragment Start ---
@st.fragment
def _render_drill_down_fragment(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    drill_levels: List[Tuple[str, str, str]],
) -> None:
    """
    Complete Container Isolation for the 7-tier hierarchical drill-down
    (Zone → Circle → Division → Subdivision → Substation → Feeder →
    Consumer, or whatever subset actually resolves for the active
    dataset per `_available_drill_levels`).

    Every selectbox here is bound via `on_change` to a callback that
    writes only into `st.session_state["temp_filters"]` — the parent
    dashboard script, its analytics engines, and every chart render
    remain completely frozen while the user navigates all tiers, because
    a widget interaction inside an @st.fragment reruns ONLY this
    fragment's render tree, never the host page.

    The dropdown option lists at every tier are looked up exclusively
    against the pre-built `_build_topology_map` in-memory structure — the
    raw dataframe is never re-scanned during tier-to-tier navigation.

    Nothing takes effect against the live dashboard until "Apply
    Filters" is clicked, which explicitly escalates to a full-app rerun
    (`st.rerun(scope="app")`) so the committed scope propagates to the
    KPI row, every chart, and the executive narrative in one single,
    deliberate refresh. Never raises.
    """
    if not drill_levels:
        st.caption("No hierarchical drill-down levels are resolvable for the active dataset.")
        return

    level_roles: List[str] = [role for role, _label, _col in drill_levels]
    level_cols: Tuple[str, ...] = tuple(col for _role, _label, col in drill_levels)
    level_roles_labels: List[Tuple[str, str]] = [(role, label) for role, label, _col in drill_levels]

    mutation_epoch = st.session_state.get("_registry_mutation_epoch", 0)
    topology_cache_key = f"{registry.version}|{mutation_epoch}|{len(df)}|{'|'.join(level_cols)}"
    topology = _build_topology_map(df, level_cols, topology_cache_key)

    st.markdown("**Hierarchical Drill-Down**")
    st.caption(" → ".join(label for _role, label in level_roles_labels))

    temp_filters: Dict[str, str] = st.session_state.get("temp_filters", {})

    for idx, (role, label) in enumerate(level_roles_labels):
        node = _resolve_topology_node(topology, level_roles, temp_filters, idx)
        options = sorted(node.keys()) if isinstance(node, dict) else []
        widget_key = f"_drill_frag_{role}"

        current_value = temp_filters.get(role)
        default_index = (options.index(current_value) + 1) if current_value in options else 0

        deeper_roles = [r for r in level_roles[idx + 1:]]

        st.selectbox(
            label,
            options=[_ALL_OPTION] + options,
            index=default_index,
            key=widget_key,
            disabled=(idx > 0 and node is None),
            on_change=_on_drill_level_change,
            args=(role, deeper_roles, widget_key),
        )

    if temp_filters:
        staged_trail = " › ".join(
            f"{label}: {temp_filters[role]}" for role, label in level_roles_labels if role in temp_filters
        )
        st.caption(f"Staged path (not yet applied): {staged_trail}")

    btn_col_apply, btn_col_clear = st.columns(2)
    with btn_col_apply:
        if st.button(
            "Apply Filters",
            key="_drill_frag_apply_btn",
            width="stretch",
            type="primary",
            on_click=_on_apply_drill_filters,
            args=(level_roles_labels,),
        ):
            # Executes on the fragment's own rerun immediately following
            # the on_click callback above. Explicit scope="app" is
            # required here: a bare st.rerun() called from inside an
            # @st.fragment only re-executes the fragment itself — this
            # is the one deliberate escape hatch that propagates the
            # newly committed drill path to the full dashboard.
            st.rerun(scope="app")
    with btn_col_clear:
        st.button(
            "Clear Staged Path",
            key="_drill_frag_clear_btn",
            width="stretch",
            on_click=_on_clear_staged_drill_path,
            args=(level_roles,),
        )
# --- Fragment End ---

def _apply_drill_path_mask(
    df: pd.DataFrame,
    drill_levels: List[Tuple[str, str, str]],
    breadcrumbs: List[Dict[str, str]],
) -> pd.Series:
    """Builds a cumulative boolean keep-mask across the active drill path.
    Each breadcrumb entry {'role': ..., 'label': ..., 'value': ...} narrows
    the mask further, matched against the corresponding resolved column in
    `drill_levels`. Never raises; a breadcrumb whose role is no longer
    resolvable in the current dataset is silently skipped."""
    mask = pd.Series(True, index=df.index)
    level_col_by_role: Dict[str, str] = {role: col for role, _label, col in drill_levels}
    try:
        for crumb in breadcrumbs:
            role = crumb.get("role", "")
            value = crumb.get("value", "")
            col = level_col_by_role.get(role)
            if not col or col not in df.columns or not value:
                continue
            normalized = df[col].astype(str).str.strip()
            mask &= normalized.eq(value)
        return mask.fillna(False)
    except Exception:  # noqa: BLE001
        return pd.Series(True, index=df.index)


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC UNIVERSAL COMPONENT INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def validate() -> Dict[str, Any]:
    """
    Verifies the filter panel's structural preconditions: presence of an
    active analytics-ready dataset and Column Registry, and enumerates
    which standard filter roles and drill-hierarchy levels are currently
    resolvable. Never raises.
    """
    try:
        df: Optional[pd.DataFrame] = st.session_state.get("analytics_ready_dataframe")
        registry: Optional[ColumnRegistry] = st.session_state.get("column_registry")
        if df is None or df.empty or registry is None:
            return {
                "component": _COMPONENT_NAME,
                "has_dataset": False,
                "resolvable_filters": [],
                "resolvable_drill_levels": [],
                "is_ready": False,
            }
        standard_filters = _available_standard_filters(registry, df)
        drill_levels = _available_drill_levels(registry, df)
        return {
            "component": _COMPONENT_NAME,
            "has_dataset": True,
            "resolvable_filters": [label for _r, label, _c in standard_filters],
            "resolvable_drill_levels": [label for _r, label, _c in drill_levels],
            "is_ready": True,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "component": _COMPONENT_NAME,
            "has_dataset": False,
            "resolvable_filters": [],
            "resolvable_drill_levels": [],
            "is_ready": False,
            "error": str(exc),
        }


def refresh() -> None:
    """
    Resets the Global Filter Engine and Cascading Drill Panel back to their
    unfiltered state: clears active_filters and drill_breadcrumbs, purges
    every widget-bound key this component owns (Milestone 2 / Issue 5
    remediation — see _purge_filter_widget_state for why this is required),
    and re-points filtered_dataframe back at the full analytics-ready
    dataset reference (no copy). Never raises.
    """
    try:
        st.session_state["active_filters"] = {}
        st.session_state["drill_breadcrumbs"] = []
        st.session_state["temp_filters"] = {}
        st.session_state["applied_filters"] = {}
        _purge_filter_widget_state()
        analytics_df: Optional[pd.DataFrame] = st.session_state.get("analytics_ready_dataframe")
        st.session_state["filtered_dataframe"] = analytics_df
        st.session_state["visualization_cache"] = {}
        st.session_state["analytics_results"] = {}
    except Exception:  # noqa: BLE001
        return


def export(export_format: str) -> Optional[bytes]:
    """
    Exports the currently filtered dataset view. Supported formats:
        "filtered_csv" -> bytes
    Returns None for an unrecognized format, a missing/empty filtered
    dataset, or any internal failure — never raises.
    """
    try:
        fmt = export_format.strip().lower()
        if fmt != "filtered_csv":
            return None
        df: Optional[pd.DataFrame] = st.session_state.get("filtered_dataframe")
        if df is None or df.empty:
            return None
        import io
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        return buffer.getvalue().encode("utf-8")
    except Exception:  # noqa: BLE001
        return None


def metadata() -> Dict[str, Any]:
    """
    Returns the filter component's capability descriptor: the full
    candidate drill hierarchy and standard filter role set it is capable of
    rendering (irrespective of current resolvability), for use by the
    Metadata Explorer and Business Glossary. Never raises.
    """
    return {
        "component": _COMPONENT_NAME,
        "drill_hierarchy_roles": [role for role, _label in _DRILL_HIERARCHY],
        "drill_hierarchy_labels": [label for _role, label in _DRILL_HIERARCHY],
        "standard_filter_roles": [role for role, _label in _STANDARD_FILTER_ROLES],
        "standard_filter_labels": [label for _role, label in _STANDARD_FILTER_ROLES],
        "supported_export_formats": ["filtered_csv"],
    }


def apply_filters(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    active_filters: Dict[str, Any],
    drill_breadcrumbs: List[Dict[str, str]],
) -> FilterEngineResult:
    """
    Pure, stateless application of a given filter/drill configuration
    against `df`, returning a FilterEngineResult. Performs no
    st.session_state reads or writes — callers (render(), pages) own state
    persistence. Operates entirely via vectorized boolean masks against a
    single boolean-indexed view of `df` (never a row-loop, never an
    intermediate deep copy chain). Never raises: any internal failure
    degrades to returning the original, unfiltered `df`.
    """
    rows_before = int(len(df))
    if df.empty:
        return FilterEngineResult(
            filtered_df=df,
            rows_before=rows_before,
            rows_after=rows_before,
            active_filters=dict(active_filters),
            drill_breadcrumbs=list(drill_breadcrumbs),
            applied_roles=[],
        )

    try:
        combined_mask = pd.Series(True, index=df.index)
        applied_roles: List[str] = []

        date_role_col = _resolve_role_column(registry, df, ROLE_REGISTRATION_DATE)
        date_range_value = active_filters.get("date_range")
        if date_range_value:
            date_mask = _apply_date_filter_mask(df, date_role_col, date_range_value)
            combined_mask &= date_mask
            if date_role_col:
                applied_roles.append(ROLE_REGISTRATION_DATE)

        for role, _label, col in _available_standard_filters(registry, df):
            selected = active_filters.get(role)
            if selected:
                combined_mask &= _apply_categorical_filter_mask(df, col, selected)
                applied_roles.append(role)

        drill_levels = _available_drill_levels(registry, df)

        

        if drill_breadcrumbs:
            combined_mask &= _apply_drill_path_mask(df, drill_levels, drill_breadcrumbs)
            applied_roles.extend(
                crumb.get("role", "") for crumb in drill_breadcrumbs if crumb.get("role")
            )

        combined_mask = combined_mask.fillna(False)
        filtered_view = df.loc[combined_mask]

        return FilterEngineResult(
            filtered_df=filtered_view,
            rows_before=rows_before,
            rows_after=int(len(filtered_view)),
            active_filters=dict(active_filters),
            drill_breadcrumbs=list(drill_breadcrumbs),
            applied_roles=sorted(set(applied_roles)),
        )
    except Exception as exc:  # noqa: BLE001
        log_exception(
            "filters.apply_filters",
            exc,
            context={"active_filters": active_filters, "drill_breadcrumbs": drill_breadcrumbs},
        )
        return FilterEngineResult(
            filtered_df=df,
            rows_before=rows_before,
            rows_after=rows_before,
            active_filters=dict(active_filters),
            drill_breadcrumbs=list(drill_breadcrumbs),
            applied_roles=[],
        )


def render(**kwargs: Any) -> None:
    """
    Renders the Global Filter Engine + Synchronized Hierarchical Drilling
    panel and materializes the result into
    st.session_state['filtered_dataframe']. Intended to be called once per
    page (pages/1_dashboard.py, pages/2_audit.py) after
    components.sidebar.render(). Never raises: any internal failure
    degrades to an inline st.warning and an unfiltered dataset view rather
    than crashing the host page.
    """

    df: Optional[pd.DataFrame] = st.session_state.get("analytics_ready_dataframe")
    registry: Optional[ColumnRegistry] = st.session_state.get("column_registry")

    if df is None or df.empty or registry is None:
        st.info(
            "Global filters and drill-down become available once a dataset is ingested "
            "and role mappings are confirmed."
        )
        st.session_state["filtered_dataframe"] = df
        return

    active_filters: Dict[str, Any] = dict(st.session_state.get("active_filters", {}))
    drill_breadcrumbs: List[Dict[str, str]] = list(st.session_state.get("drill_breadcrumbs", []))

    # NEW — high-density scope summary chip row, rendered ABOVE the expander so
    # the current filter scope is visible without opening the panel. Purely
    # derived from already-persisted state; performs no filtering itself.
    _active_scope_chips: List[str] = []
    if active_filters.get("date_range"):
        _active_scope_chips.append("Date Range")
    _active_scope_chips.extend(
        registry.display_name(k) for k in active_filters.keys() if k != "date_range"
    )
    if drill_breadcrumbs:
        _active_scope_chips.append(f"Drill: {len(drill_breadcrumbs)} level(s)")
    if _active_scope_chips:
        st.markdown(
            "".join(
                f'<span class="kesco-badge kesco-badge-info" style="margin-right:6px;">{c}</span>'
                for c in _active_scope_chips
            ),
            unsafe_allow_html=True,
        )

    with st.expander("Global Filters & Drill-Down", expanded=bool(_active_scope_chips)):
        # ... existing code (unchanged: st.form, date range, categorical filters,
        # drill-down cascade, Reset All button, apply logic) ...

        try:
            # ── Milestone 6 / Issue 17 remediation — st.form batching ──
            # Date range and categorical filters are now gated behind a
            # single st.form with an explicit "Apply Filters" submit
            # button, so adjusting a date range or toggling several
            # multiselect options no longer triggers an independent full
            # page rerun per widget interaction — only the single rerun
            # triggered by form submission. The Cascading Drill-Down panel
            # is deliberately kept OUTSIDE this form: its UX is
            # fundamentally progressive-reveal (selecting a Zone must
            # immediately narrow the Circle dropdown's options, and so on
            # down the hierarchy), which requires a live rerun on every
            # level selection. Wrapping it in the same form would freeze
            # the cascade until submission and break the panel entirely.
            # "Reset All" is likewise kept outside the form so a full
            # reset is never deferred behind a pending, unsubmitted form.
            with st.form(key="_filters_form", border=False):
                date_col = _resolve_role_column(registry, df, ROLE_REGISTRATION_DATE)
                if date_col:
                    st.markdown("**Date Range**")
                    parsed_dates = _robust_parse_dates(df[date_col]).dropna()
                    if not parsed_dates.empty:
                        min_date = parsed_dates.min().date()
                        max_date = parsed_dates.max().date()
                        existing_range = active_filters.get("date_range")
                        default_start = existing_range[0] if existing_range and existing_range[0] else min_date
                        default_end = existing_range[1] if existing_range and existing_range[1] else max_date
                        selected_range = st.date_input(
                            "Registration Date Range",
                            value=(default_start, default_end),
                            min_value=min_date,
                            max_value=max_date,
                            key="_filters_date_range_input",
                            label_visibility="collapsed",
                        )
                        if isinstance(selected_range, tuple) and len(selected_range) == 2:
                            active_filters["date_range"] = (selected_range[0], selected_range[1])
                        st.divider()

                standard_filters = _available_standard_filters(registry, df)
                if standard_filters:
                    st.markdown("**Categorical Filters**")
                    filter_cols = st.columns(min(len(standard_filters), 3))
                    for idx, (role, label, col) in enumerate(standard_filters):
                        with filter_cols[idx % len(filter_cols)]:
                            distinct_count = df[col].nunique(dropna=True)
                            if distinct_count > _HIGH_CARDINALITY_FILTER_THRESHOLD:
                                # A dropdown populated with thousands of options is a
                                # frontend rendering cost independent of backend sort
                                # cost — degrade to a bounded substring-search input
                                # rather than building an effectively-unusable
                                # multiselect. isin()-based mask semantics in
                                # _apply_categorical_filter_mask are fully preserved.
                                search_value = st.text_input(
                                    f"{label} (search — {distinct_count:,} distinct values)",
                                    value=(active_filters.get(role, [""]) or [""])[0],
                                    key=f"_filters_search_{role}",
                                )
                                if search_value.strip():
                                    needle = search_value.strip().lower()
                                    normalized = df[col].astype(str).str.strip().str.lower()
                                    matched = sorted(
                                        df.loc[normalized.str.contains(needle, na=False), col]
                                        .astype(str).str.strip().unique().tolist()
                                    )[:_MAX_MULTISELECT_OPTIONS]
                                    if matched:
                                        active_filters[role] = matched
                                    elif role in active_filters:
                                        del active_filters[role]
                                elif role in active_filters:
                                    del active_filters[role]
                                continue
                            options = _safe_unique_options(df[col])
                            existing_selection = [
                                v for v in active_filters.get(role, []) if v in options
                            ]
                            selected = st.multiselect(
                                label,
                                options=options,
                                default=existing_selection,
                                key=f"_filters_multiselect_{role}",
                            )
                            if selected:
                                active_filters[role] = selected
                            elif role in active_filters:
                                del active_filters[role]
                    st.divider()

                apply_clicked = st.form_submit_button(
                    "Apply Filters", width="stretch", type="primary"
                )

            drill_levels = _available_drill_levels(registry, df)
            st.session_state.setdefault("applied_filters", {}) 
            if "temp_filters" not in st.session_state:
                 st.session_state["temp_filters"] = dict(
                     st.session_state.get("applied_filters", {})
                )
            _render_drill_down_fragment(df, registry, drill_levels)

            # --- Fragment Start --- 
            reset_clicked = st.button(
                "Reset All", key="_filters_reset_btn", width="stretch"
            )
            # --- Fragment End --- 
            if reset_clicked:
                refresh()
                st.rerun()

            if apply_clicked or active_filters != st.session_state.get("active_filters", {}) or (
                drill_breadcrumbs != st.session_state.get("drill_breadcrumbs", [])
            ):
                st.session_state["active_filters"] = active_filters
                st.session_state["drill_breadcrumbs"] = drill_breadcrumbs
                st.session_state["visualization_cache"] = {}
                st.session_state["analytics_results"] = {}

        except Exception as exc:  # noqa: BLE001
            incident_id = log_exception("filters.render", exc)
            st.warning(f"Filter panel encountered an issue and has been reset (Reference ID: {incident_id}).")
            active_filters = {}
            drill_breadcrumbs = []
            st.session_state["active_filters"] = {}
            st.session_state["drill_breadcrumbs"] = []

    result = apply_filters(
        df=df,
        registry=registry,
        active_filters=st.session_state.get("active_filters", {}),
        drill_breadcrumbs=st.session_state.get("drill_breadcrumbs", []),
    )
    st.session_state["filtered_dataframe"] = result.filtered_df

    if result.rows_after < result.rows_before:
        st.caption(
            f"Showing {result.rows_after:,} of {result.rows_before:,} rows "
            f"after applying active filters and drill-down scope."
        )
