"""
FRONTEND/components/schema_mapping.py — Raw-to-Domain Metadata Transformation
Deck (FILE 4 / 6)

Implements the Universal Component Interface (render / validate / refresh /
export / metadata) for the platform's Schema Mapping Studio:

  • Role Mapping Deck: groups every canonical business role defined in
    core.roles into cognitively-scoped clusters (Identifiers, Temporal,
    Status & Workflow, Categorization, Personnel & Ownership, Geographic
    Hierarchy, Financial & Consumption, Geospatial) and exposes a
    selectbox-driven binding surface against the active dataset's live
    column headers, resolved and mutated exclusively through the
    core.column_registry.ColumnRegistry public API (resolve / set_mapping /
    has_role / display_name / summary_table / suggestions_for_unmapped).
  • Auto-Suggestion Engine: surfaces the Type Inference Engine's
    (core.type_inference) per-column `suggested_role_scores` to pre-fill
    unmapped roles with their highest-confidence candidate column, and
    offers a one-click "Auto-Suggest Unmapped Roles" action that applies
    every still-unresolved, sufficiently-confident suggestion in one pass
    without disturbing any already-confirmed manual mapping.
  • Column Type Override Studio: lets an analyst re-cast any column's
    detected InferredType (core.schema_models.InferredType) — Datetime,
    Numeric, Currency, Percentage, Boolean, Categorical, Text, ID — via
    vectorized, exception-safe coercion, replacing the
    `analytics_ready_dataframe` / `filtered_dataframe` session pointers
    with a freshly-cast copy rather than ever mutating the original
    reference in place.
  • Unmapped Columns & Fuzzy Suggestions Panel: surfaces every column not
    yet claimed by a confirmed role mapping alongside the Column Registry's
    own fuzzy-matched role candidates, so an analyst can spot orphaned
    headers at a glance.
  • Active Mapping Summary & Export: renders `registry.summary_table()` as
    an interactive data matrix and exposes CSV/JSON export of the mapping
    profile and type-override ledger.

This module performs NO pandas aggregation, NO KPI computation, and NO
chart rendering. Every mutation it performs against the Column Registry
goes exclusively through that registry's own documented public API and
public dataclass attributes (RoleMapping), and every mapping/type-override
action it takes is appended to st.session_state['audit_results']
['audit_entries'] in the same flat schema produced by utils.audit_log
.AuditLog.as_table(), so the Audit Trace Log panel (pages/2_audit.py)
reflects schema-mapping activity alongside cleaning-engine activity with
zero additional wiring.
"""
from __future__ import annotations

import io
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from utils.error_logging import log_exception
from core.themes import inject_fragment_atomic_style  
import numpy as np
import pandas as pd
import streamlit as st

from core.column_registry import ColumnRegistry
from core.roles import (
    ROLE_RECORD_ID,
    ROLE_REGISTRATION_DATE,
    ROLE_CLOSING_DATE,
    ROLE_STATUS,
    ROLE_CATEGORY,
    ROLE_SUBCATEGORY,
    ROLE_OFFICER,
    ROLE_CONSUMER_ID,
    ROLE_CONSUMER_NAME,
    ROLE_ZONE,
    ROLE_CIRCLE,
    ROLE_DIVISION,
    ROLE_SUBDIVISION,
    ROLE_FEEDER,
    ROLE_TRANSFORMER,
    ROLE_SUBSTATION,
    ROLE_AMOUNT,
    ROLE_TARGET_AMOUNT,
    ROLE_COLLECTED_AMOUNT,
    ROLE_UNITS_CONSUMED,
    ROLE_SLA_DEADLINE,
    ROLE_REOPEN_FLAG,
    ROLE_PRIORITY,
    ROLE_LATITUDE,
    ROLE_LONGITUDE,
    ROLE_EMPLOYEE_ID,
    ROLE_DEPARTMENT,
    ROLE_ATTENDANCE_STATUS,
    ROLE_ASSET_ID,
    ROLE_ASSET_HEALTH,
)
from core.schema_models import ColumnProfile, InferredType, RoleMapping
from core.settings import MIN_FUZZY_ROLE_SCORE, MIN_AUTO_CONFIDENCE
from core.themes import DEFAULT_THEME_KEY

from utils.grid_utils import render_enterprise_grid

_COMPONENT_NAME: str = "schema_mapping"

# Cognitively-scoped role groupings for the Role Mapping Deck. Order here
# drives render order; every role constant appearing anywhere in
# core.roles is represented exactly once across these groups.
_ROLE_GROUPS: Dict[str, List[str]] = {
    "Identifiers": [
        ROLE_RECORD_ID, ROLE_CONSUMER_ID, ROLE_CONSUMER_NAME,
        ROLE_EMPLOYEE_ID, ROLE_ASSET_ID,
    ],
    "Temporal": [
        ROLE_REGISTRATION_DATE, ROLE_CLOSING_DATE, ROLE_SLA_DEADLINE,
    ],
    "Status & Workflow": [
        ROLE_STATUS, ROLE_REOPEN_FLAG, ROLE_PRIORITY,
        ROLE_ATTENDANCE_STATUS, ROLE_ASSET_HEALTH,
    ],
    "Categorization": [
        ROLE_CATEGORY, ROLE_SUBCATEGORY,
    ],
    "Personnel & Ownership": [
        ROLE_OFFICER, ROLE_DEPARTMENT,
    ],

    # AFTER
    "Geographic Hierarchy": [
        # MILESTONE 1 / ISSUE 7 REMEDIATION — ROLE_SUBSTATION is now a
        # first-class canonical role (core.roles). This group ordering
        # mirrors core.roles.CANONICAL_GEO_HIERARCHY exactly.
        ROLE_ZONE, ROLE_CIRCLE, ROLE_DIVISION, ROLE_SUBDIVISION,
        ROLE_SUBSTATION, ROLE_FEEDER, ROLE_TRANSFORMER,
    ],
    "Financial & Consumption": [
        ROLE_AMOUNT, ROLE_TARGET_AMOUNT, ROLE_COLLECTED_AMOUNT,
        ROLE_UNITS_CONSUMED,
    ],
    "Geospatial": [
        ROLE_LATITUDE, ROLE_LONGITUDE,
    ],
}

_ALL_ROLES: List[str] = [role for group in _ROLE_GROUPS.values() for role in group]

# Vectorized, exception-safe type-coercion primitives for the Column Type
# Override Studio. Deliberately self-contained (rather than importing
# engine.cleaner's private regex constants) to avoid coupling this
# presentation-layer module to another module's private implementation
# details.
_CURRENCY_STRIP_RE = re.compile(r"[₹\$€£¥,\s]")
_PERCENT_STRIP_RE = re.compile(r"%\s*$")
_BOOLEAN_TRUE_TOKENS: frozenset = frozenset({"true", "yes", "y", "1", "t"})
_BOOLEAN_FALSE_TOKENS: frozenset = frozenset({"false", "no", "n", "0", "f"})

# ── NEW: strict-typed role validation constants ──────────────────────────
_STRICT_NUMERIC_VALIDATION_ROLES: frozenset = frozenset({
    ROLE_AMOUNT, ROLE_TARGET_AMOUNT, ROLE_COLLECTED_AMOUNT, ROLE_UNITS_CONSUMED,
})
_STRICT_DATETIME_VALIDATION_ROLES: frozenset = frozenset({
    ROLE_REGISTRATION_DATE, ROLE_CLOSING_DATE, ROLE_SLA_DEADLINE,
})
_MAPPING_PARSE_FAILURE_WARN_THRESHOLD: float = 0.15  # >15% unparseable triggers a UI warning


def _validate_strict_role_parse_rate(
    role: str, column_name: str, df: pd.DataFrame
) -> Optional[str]:
    """
    Non-destructive, post-mapping data-quality check for roles with strict
    downstream type expectations. Never coerces, mutates, or rejects the
    mapping — the platform's non-destructive compliance policy means the
    user's mapping choice is always honored exactly as requested. This
    function only measures, using the SAME vectorized coercion primitives
    already used throughout core/cleaner.py and engine/analytics.py
    (pd.to_numeric/pd.to_datetime with errors='coerce'), what fraction of
    the newly-mapped column would silently resolve to NaN downstream, and
    surfaces that as an explicit, actionable UI warning rather than letting
    it manifest as an unexplained dip in a KPI three screens later. Never
    raises.
    """
    try:
        if column_name not in df.columns:
            return None
        non_null = df[column_name].dropna()
        if non_null.empty:
            return None

        if role in _STRICT_NUMERIC_VALIDATION_ROLES:
            cleaned = non_null.astype(str).str.replace(r"[₹\$€£¥,\s%]", "", regex=True)
            failure_rate = float(pd.to_numeric(cleaned, errors="coerce").isna().mean())
            if failure_rate > _MAPPING_PARSE_FAILURE_WARN_THRESHOLD:
                return (
                    f"{failure_rate:.0%} of values in '{column_name}' could not be parsed as "
                    f"numeric after mapping to a financial/consumption role. These rows will "
                    f"resolve to null (not zero) in every downstream KPI and chart until the "
                    f"source data or this mapping is corrected."
                )
        elif role in _STRICT_DATETIME_VALIDATION_ROLES:
            failure_rate = float(pd.to_datetime(non_null, errors="coerce").isna().mean())
            if failure_rate > _MAPPING_PARSE_FAILURE_WARN_THRESHOLD:
                return (
                    f"{failure_rate:.0%} of values in '{column_name}' could not be parsed as "
                    f"dates after mapping to a temporal role. These rows will be silently "
                    f"excluded from every date-dependent KPI (resolution time, pending age, "
                    f"trend charts) until the source data or this mapping is corrected."
                )
        return None
    except Exception:  # noqa: BLE001
        return None



# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS — REGISTRY / PROFILE RESOLUTION
# ══════════════════════════════════════════════════════════════════════════════

def _profiles_by_column(profiles: List[ColumnProfile]) -> Dict[str, ColumnProfile]:
    """Indexes a flat ColumnProfile list by original column name. Never
    raises; returns an empty dict for an empty/invalid input list."""
    try:
        return {p.original_name: p for p in profiles}
    except Exception:  # noqa: BLE001
        return {}


def _claimed_columns(registry: ColumnRegistry) -> set:
    """Returns the set of column names already bound to a confirmed role
    mapping, used to prevent the same physical column from being
    double-suggested for two different roles simultaneously."""
    try:
        return {m.column_name for m in registry.mappings.values() if m.confirmed and m.column_name}
    except Exception:  # noqa: BLE001
        return set()


def _best_candidate_column_for_role(
    role: str,
    profiles: List[ColumnProfile],
    claimed_columns: set,
) -> Tuple[Optional[str], float]:
    """Scans every not-yet-claimed ColumnProfile for the highest
    `suggested_role_scores[role]` value, returning (column_name, score).
    Returns (None, 0.0) when no profile carries a score for this role.
    Never raises."""
    best_col: Optional[str] = None
    best_score: float = 0.0
    try:
        for profile in profiles:
            if profile.original_name in claimed_columns:
                continue
            score = profile.suggested_role_scores.get(role, 0.0)
            if score > best_score:
                best_score = score
                best_col = profile.original_name
        return best_col, best_score
    except Exception:  # noqa: BLE001
        return None, 0.0


def _confidence_badge_html(confidence: float, confirmed: bool) -> str:
    """Renders a semantic KESCO badge span (reusing the CSS classes injected
    globally by app.py's design system) representing a role mapping's
    current confidence tier. Never raises."""
    try:
        if not confirmed:
            return '<span class="kesco-badge kesco-badge-info">Suggested — Not Confirmed</span>'
        if confidence >= 0.85:
            tier, label = "kesco-badge-good", f"{confidence:.0%} Confidence"
        elif confidence >= MIN_AUTO_CONFIDENCE:
            tier, label = "kesco-badge-warning", f"{confidence:.0%} Confidence"
        else:
            tier, label = "kesco-badge-critical", f"{confidence:.0%} Confidence"
        return f'<span class="kesco-badge {tier}">{label}</span>'
    except Exception:  # noqa: BLE001
        return '<span class="kesco-badge kesco-badge-info">Unknown Confidence</span>'


def _log_mapping_change(
    action_type: str,
    description: str,
    rows_affected: Optional[int] = None,
) -> None:
    """Appends a schema-mapping action into st.session_state['audit_results']
    ['audit_entries'] using the exact flat schema produced by
    utils.audit_log.AuditLog.as_table() (Timestamp / Type / Description /
    Rows Affected), inserted at index 0 to preserve the platform-wide
    most-recent-first ordering convention. Never raises."""
    try:
        entry: Dict[str, Any] = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Type": action_type,
            "Description": description,
            "Rows Affected": rows_affected if rows_affected is not None else "—",
            "Details": "—",
        }
        audit_results: Dict[str, Any] = st.session_state.setdefault("audit_results", {})
        entries: List[Dict[str, Any]] = audit_results.setdefault("audit_entries", [])
        entries.insert(0, entry)
        audit_results["audit_entries"] = entries
        st.session_state["audit_results"] = audit_results
    except Exception:  # noqa: BLE001
        return


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS — BULK MAPPING ACTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _auto_suggest_unmapped_roles(registry: ColumnRegistry, profiles: List[ColumnProfile]) -> int:
    """
    Applies the highest-confidence candidate column to every role that is
    not yet confirmed in `registry`, using the same auto-confirmation
    threshold (MIN_FUZZY_ROLE_SCORE + not needs_manual_review) as
    ColumnRegistry.bootstrap_from_profiles, but WITHOUT touching any role
    that already carries a confirmed manual or auto mapping. Returns the
    count of newly-applied mappings. Never raises.
    """
    applied = 0
    try:
        profiles_by_col = _profiles_by_column(profiles)
        claimed = _claimed_columns(registry)
        for role in _ALL_ROLES:
            if registry.has_role(role):
                continue
            best_col, score = _best_candidate_column_for_role(role, profiles, claimed)
            if best_col is None or score < MIN_FUZZY_ROLE_SCORE:
                continue
            profile = profiles_by_col.get(best_col)
            auto_confirm = bool(profile) and not profile.needs_manual_review
            registry.mappings[role] = RoleMapping(
                role=role,
                column_name=best_col,
                confidence=round(score / 100.0, 3),
                source="auto" if auto_confirm else "auto_low_confidence",
                confirmed=auto_confirm,
            )
            registry.version += 1
            claimed.add(best_col)
            applied += 1
        return applied
    except Exception:  # noqa: BLE001
        return applied


def _reset_all_mappings(registry: ColumnRegistry) -> int:
    """Clears every role mapping currently held by `registry`, returning the
    count of mappings cleared. Never raises."""
    try:
        cleared = len(registry.mappings)
        registry.mappings.clear()
        registry.version += 1
        return cleared
    except Exception:  # noqa: BLE001
        return 0


def _apply_type_override(series: pd.Series, target_type: InferredType) -> Tuple[pd.Series, bool]:
    """
    Vectorized, exception-safe re-cast of `series` into `target_type`.
    Returns (coerced_series, success_flag) — on any internal failure or an
    unsupported target type, returns the original series unchanged with
    success=False rather than raising.

    C2 fix: CATEGORICAL and TEXT/ID branches now use `.astype("string")`
    instead of `.astype(str)`. `.astype(str)` on a nullable StringDtype /
    ArrowDtype(pa.string()) column — or on a plain object column holding
    Python `None` — converts every true null into the literal text
    "<NA>" / "None", which then becomes a real, permanent category value
    or text value with no downstream coercion step to catch it (unlike
    the numeric/boolean branches below, which route through
    `errors="coerce"` and self-heal). `.astype("string")` correctly
    preserves nulls as pd.NA instead of stringifying them.

    W1 fix: DATETIME branch now tz-strips the parsed result, matching the
    `_safe_tz_naive` convention already used in analytics.py /
    chart_factory.py / filters.py, so a Column Type Override applied from
    this UI can't reintroduce a tz-aware column into
    analytics_ready_dataframe after the fact.
    """
    try:
        if target_type == InferredType.DATETIME:
            parsed = pd.to_datetime(series, errors="coerce")
            if getattr(parsed.dt, "tz", None) is not None:
                parsed = parsed.dt.tz_localize(None)
            return parsed, True
        if target_type in (InferredType.NUMERIC, InferredType.CURRENCY):
            cleaned = series.astype(str).apply(
                lambda v: _CURRENCY_STRIP_RE.sub("", v) if isinstance(v, str) else v
            )
            return pd.to_numeric(cleaned, errors="coerce"), True
        if target_type == InferredType.PERCENTAGE:
            cleaned = series.astype(str).apply(
                lambda v: _PERCENT_STRIP_RE.sub("", v).strip() if isinstance(v, str) else v
            )
            return pd.to_numeric(cleaned, errors="coerce"), True
        if target_type == InferredType.BOOLEAN:
            normalized = series.astype(str).str.strip().str.lower()
            coerced = normalized.map(
                lambda v: True if v in _BOOLEAN_TRUE_TOKENS
                else (False if v in _BOOLEAN_FALSE_TOKENS else np.nan)
            )
            return coerced, True
        if target_type == InferredType.CATEGORICAL:
            return series.astype("string").astype("category"), True
        if target_type in (InferredType.TEXT, InferredType.ID):
            if pd.api.types.is_float_dtype(series):
                non_null = series.dropna()
                is_whole_mask = non_null.apply(lambda v: np.isfinite(v) and float(v).is_integer())
                whole_ratio = float(is_whole_mask.mean()) if len(non_null) else 0.0
                if whole_ratio >= 0.99:
                    coerced = series.apply(
                        lambda v: str(int(v)) if pd.notna(v) and np.isfinite(v) and float(v).is_integer()
                        else (str(v) if pd.notna(v) else v)
                    )
                    return coerced, True
            return series.astype("string"), True

        return series, False
    except Exception:  # noqa: BLE001
        return series, False


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS — ROLE MAPPING CONTROL WIDGET
# ══════════════════════════════════════════════════════════════════════════════
def _render_custom_role_registration(df: pd.DataFrame, registry: ColumnRegistry) -> None:
    """
    Registers a workspace-specific 'custom role' beyond the fixed
    core.roles vocabulary, fulfilling the Workspace Management mandate's
    'unlimited custom roles' requirement. Custom roles are persisted via
    the exact same ColumnRegistry.set_mapping() public API used by every
    built-in role, so they are automatically included in
    registry.summary_table(), registry.snapshot()/.restore() (workspace
    save/load), and registry.resolve() lookups with zero new registry
    surface. Never raises.
    """
    with st.expander("Custom Roles (Workspace-Specific)", expanded=False):
        st.caption(
            "Register any additional business role specific to this dataset. Custom roles are "
            "persisted in the Column Registry exactly like built-in roles and are included in "
            "workspace save/load and mapping export."
        )
        custom_role_name = st.text_input(
            "Custom Role Identifier (snake_case)",
            key="_schema_custom_role_name_input",
            placeholder="e.g. meter_reading_type",
        )
        custom_role_column = st.selectbox(
            "Bind To Column",
            options=["— Not Mapped —"] + list(df.columns),
            key="_schema_custom_role_column_select",
        )
        if st.button("Register Custom Role", key="_schema_register_custom_role_btn"):
            normalized_role = custom_role_name.strip().lower().replace(" ", "_")
            if not normalized_role:
                st.error("Provide a custom role identifier before registering.")
            elif normalized_role in _ALL_ROLES:
                st.error(f"'{normalized_role}' is already a built-in role — choose a different identifier.")
            else:
                resolved_column = None if custom_role_column == "— Not Mapped —" else custom_role_column
                registry.set_mapping(normalized_role, resolved_column, manual=True)
                _log_mapping_change(
                    "mapping",
                    f"Custom role '{normalized_role}' registered and mapped to column "
                    f"'{resolved_column}'." if resolved_column else f"Custom role '{normalized_role}' registered (unmapped).",
                )
                st.session_state["visualization_cache"] = {}
                st.session_state["analytics_results"] = {}
                st.success(f"Custom role '{normalized_role}' registered.")
                st.rerun()


def _render_role_control(
    role: str,
    df: pd.DataFrame,
    registry: ColumnRegistry,
    profiles_by_col: Dict[str, ColumnProfile],
    claimed_columns: set,          # NEW — precomputed once per render(), not per-role
    profiles_list: List[ColumnProfile],  # NEW — precomputed once per render(), not per-role
) -> None:
    """
    Milestone 16 fix: mapping selections are now staged into
    st.session_state["_schema_pending_mappings"] (role -> column_or_None)
    instead of being committed (and rerun-triggered) immediately per
    widget. Nothing is written to the live ColumnRegistry — and no
    registry.version bump, no cache invalidation, no st.rerun() — until
    the caller's "Apply Staged Mappings" button fires
    _commit_pending_mappings() once for the whole batch.

    Performance fix: claimed_columns and profiles_list are now computed
    ONCE by the caller (render()) before the 30-role loop, instead of
    being independently recomputed inside every single call to this
    function — eliminating an O(roles x mappings) redundant scan that
    fired on every Streamlit rerun (i.e. every widget interaction
    anywhere on the page). Never raises.
    """
    try:
        display_name = registry.display_name(role)
        current_col = registry.resolve(role)
        options: List[str] = ["— Not Mapped —"] + list(df.columns)

        pending: Dict[str, Optional[str]] = st.session_state.setdefault(
            "_schema_pending_mappings", {}
        )
        staged_value = pending.get(role, current_col)

        if staged_value and staged_value in options:
            default_index = options.index(staged_value)
        else:
            best_col, _score = _best_candidate_column_for_role(
                role, profiles_list, claimed_columns
            )
            default_index = options.index(best_col) if best_col in options else 0

        selected = st.selectbox(
            display_name,
            options=options,
            index=default_index,
            key=f"_schema_map_select_{role}",
            help=f"Canonical role: '{role}'. Changes are staged — click "
                 f"'Apply Staged Mappings' below to commit.",
        )
        resolved_selection: Optional[str] = None if selected == "— Not Mapped —" else selected

        if resolved_selection != current_col:
            pending[role] = resolved_selection
            st.session_state["_schema_pending_mappings"] = pending
        elif role in pending:
            pending.pop(role, None)
            st.session_state["_schema_pending_mappings"] = pending

        mapping = registry.mappings.get(role)
        is_staged = role in pending
        if is_staged:
            st.markdown(
                '<span class="kesco-badge kesco-badge-warning">Staged — Not Yet Applied</span>',
                unsafe_allow_html=True,
            )
        elif mapping and mapping.confirmed and mapping.column_name:
            source_label = {
                "auto": "Auto-Detected",
                "auto_low_confidence": "Auto (Low Confidence)",
                "manual": "Manual Override",
            }.get(mapping.source, mapping.source)
            st.markdown(
                f'{_confidence_badge_html(mapping.confidence, True)}'
                f'&nbsp;&nbsp;<span style="font-size:0.72rem;color:#94A3B8;">{source_label}</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(_confidence_badge_html(0.0, False), unsafe_allow_html=True)
    except Exception as exc:  # noqa: BLE001
        log_exception("schema_mapping._render_role_control", exc, context={"role": role})
        st.caption(f"Unable to render mapping control for role '{role}': {exc}")


def _commit_pending_mappings(registry: ColumnRegistry, df: pd.DataFrame) -> int:
    """
    Applies every staged role->column change in one batch: one
    registry.version bump total (not one per role), one cache
    invalidation, one audit-log entry per changed role, one rerun.

    HARD CACHE INVALIDATION (fix): registry.version alone is not a safe
    st.cache_data invalidation key across concurrent sessions — two
    different ColumnRegistry instances can coincidentally reach the same
    version number with the same dataframe shape/filters and collide in
    Streamlit's process-wide cache_data store, silently serving back a
    bundle computed BEFORE this mapping change. To guarantee correctness
    regardless of that collision risk:
        1. A dedicated, monotonically-increasing, per-session
           "_registry_mutation_epoch" counter is bumped here. Every
           cache-key builder that consumes the registry (see
           1_dashboard.py::_compute_cache_key) must include this epoch,
           so a collision on registry.version alone can never mask a
           mapping change.
        2. st.cache_data.clear() is called unconditionally as an
           absolute backstop — it purges every st.cache_data-backed
           function's entire cache (analytics bundle, metric cards,
           executive narrative, chart renders) process-wide, so even a
           theoretically-missed cache-key dependency cannot serve stale
           analytics after a mapping commit.
    Returns the count of mappings committed. Never raises.
    """
    committed = 0
    try:
        pending: Dict[str, Optional[str]] = st.session_state.get("_schema_pending_mappings", {})
        if not pending:
            return 0
        for role, resolved_selection in pending.items():
            registry.set_mapping(role, resolved_selection, manual=True)
            _log_mapping_change(
                "mapping",
                f"Role '{registry.display_name(role)}' manually mapped to column "
                f"'{resolved_selection}'." if resolved_selection
                else f"Role '{registry.display_name(role)}' manually unmapped.",
            )
            if resolved_selection:
                parse_warning = _validate_strict_role_parse_rate(role, resolved_selection, df)
                if parse_warning:
                    st.session_state.setdefault("_schema_mapping_pending_warnings", []).append(parse_warning)
            committed += 1

        st.session_state["_schema_pending_mappings"] = {}

        # ── HARD CACHE INVALIDATION ──────────────────────────────────
        # 1. Dedicated epoch counter — independent of registry.version,
        #    consumed by every downstream cache_key builder.
        st.session_state["_registry_mutation_epoch"] = (
            int(st.session_state.get("_registry_mutation_epoch", 0)) + 1
        )
        # 2. Legacy dict-shaped markers (kept for backward compatibility
        #    with any consumer still reading them; no longer relied upon
        #    as the actual invalidation mechanism).
        st.session_state["visualization_cache"] = {}
        st.session_state["analytics_results"] = {}
        # 3. Absolute backstop — purge every st.cache_data-backed
        #    function process-wide (dashboard analytics bundle, KPI
        #    cards, executive narrative, chart renders, self-service
        #    builder cache). Guarantees zero stale reads regardless of
        #    any future cache_key composition drift.
        try:
            st.cache_data.clear()
        except Exception:  # noqa: BLE001 — never let cache-clearing itself fail the commit
            pass

        return committed
    except Exception:  # noqa: BLE001
        return committed

# --- Fragment Start ---
@st.fragment
def _render_apply_staged_mappings_fragment(registry: ColumnRegistry, df: pd.DataFrame) -> None:
    """
    Isolated rerun boundary for the 'Apply Staged Mappings' action.

    Milestone: Interaction Bottleneck Fix. Prior to this, clicking Apply
    triggered a bare st.rerun() from inside the main render() body, which
    re-executes the ENTIRE host page script (KPI row, chart rows, filters,
    executive narrative — everything) even though only the Role Mapping
    Deck's staged state actually changed. @st.fragment scopes both the
    button's rerun AND this function's own internal st.rerun() call to
    just this fragment's render tree, leaving every other component on
    the host page (1_dashboard.py, etc.) completely untouched.

    Reads registry/df by reference (never mutates the caller's copies
    except via the registry's own public set_mapping API, exactly as
    _commit_pending_mappings already does). Writes to
    st.session_state["_schema_pending_mappings"],
    ["visualization_cache"], and ["analytics_results"] — all of which are
    global keys, so downstream pages still see the committed mappings on
    their NEXT natural rerun (navigation, widget interaction) without
    requiring this fragment to force a full-page rerun itself.

    Never raises: _commit_pending_mappings() is already internally
    exception-safe (returns 0 on failure), so this wrapper adds no new
    failure surface.
    """
    inject_fragment_atomic_style(st.session_state.get("theme", DEFAULT_THEME_KEY))
    _pending_count = len(st.session_state.get("_schema_pending_mappings", {}))
    if _pending_count:
        st.warning(f"{_pending_count} role mapping(s) staged but not yet applied.")

    if st.button(
        f"Apply Staged Mappings ({_pending_count})",
        key="_schema_apply_pending_btn",
        width="stretch",
        type="primary",
        disabled=_pending_count == 0,
    ):
        applied = _commit_pending_mappings(registry, df)
        st.success(f"Applied {applied} mapping change(s).")
        st.rerun()
# --- Fragment End ---

# Fragment-scoped rerun: re-executes ONLY this fragment (the
        # apply button + its warning banner), not the full Schema Mapping
        # Studio page and not any other page's session state.
        
# --- Fragment Start ---
@st.fragment
def _render_role_mapping_deck_fragment(
    df: pd.DataFrame,
    registry: ColumnRegistry,
    profiles_by_col: Dict[str, ColumnProfile],
    claimed_columns: set,
    profiles_list: List[ColumnProfile],
) -> None:
    """
    Isolated rerun boundary for the entire Role Mapping Deck (~30 role
    selectboxes across 8 cognitive groups).

    FRAGMENT ENCAPSULATION FIX: claimed_columns and profiles_list are now
    REQUIRED PARAMETERS, computed exactly once by render() and passed in
    explicitly — this fragment no longer computes or reaches for any
    variable defined outside its own body or arguments. This both
    eliminates the NameError risk (the fragment never assumes an
    outer-scope local exists) and preserves the original performance
    intent (one snapshot per render() call, not one per role, not one per
    fragment-scoped rerun).

    Selecting a role reruns ONLY this fragment — Streamlit's automatic
    widget-driven rerun is scoped to this function's render tree, leaving
    the KPI row, executive narrative, and every chart on the host page
    completely untouched.

    Never raises: delegates entirely to the already exception-safe
    _render_role_control per-role renderer, which itself only writes to
    st.session_state["_schema_pending_mappings"] — no rerun, no registry
    mutation.
    """
    
    inject_fragment_atomic_style(st.session_state.get("theme", DEFAULT_THEME_KEY))
    for group_label, role_list in _ROLE_GROUPS.items():
        mapped_in_group = sum(1 for r in role_list if registry.has_role(r))
        unmapped_required = mapped_in_group < len(role_list)
        with st.expander(
            f"{group_label} ({mapped_in_group}/{len(role_list)} mapped)",
            expanded=unmapped_required and group_label in ("Identifiers", "Temporal", "Status & Workflow"),
        ):
            grid_cols = st.columns(2)
            for idx, role in enumerate(role_list):
                with grid_cols[idx % 2]:
                    _render_role_control(
                        role, df, registry, profiles_by_col,
                        claimed_columns, profiles_list,
                    )
# --- Fragment End ---
# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC UNIVERSAL COMPONENT INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def validate() -> Dict[str, Any]:
    """
    Verifies the Schema Mapping Studio's structural preconditions: presence
    of an active analytics-ready dataset and Column Registry, and computes
    confirmed-vs-total addressable role counts plus the count of columns
    still flagged needs_manual_review by the Type Inference Engine. Never
    raises.
    """
    try:
        df: Optional[pd.DataFrame] = st.session_state.get("analytics_ready_dataframe")
        registry: Optional[ColumnRegistry] = st.session_state.get("column_registry")
        profiles: List[ColumnProfile] = st.session_state.get("column_profiles", [])
        has_dataset = df is not None and not df.empty
        has_registry = registry is not None
        confirmed_count = sum(1 for r in _ALL_ROLES if registry.has_role(r)) if has_registry else 0
        needs_review_count = sum(1 for p in profiles if p.needs_manual_review)
        return {
            "component": _COMPONENT_NAME,
            "has_dataset": has_dataset,
            "has_registry": has_registry,
            "total_addressable_roles": len(_ALL_ROLES),
            "confirmed_role_count": confirmed_count,
            "total_columns": int(df.shape[1]) if has_dataset else 0,
            "needs_review_column_count": needs_review_count,
            "is_ready": has_dataset and has_registry,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "component": _COMPONENT_NAME,
            "has_dataset": False,
            "has_registry": False,
            "total_addressable_roles": len(_ALL_ROLES),
            "confirmed_role_count": 0,
            "total_columns": 0,
            "needs_review_column_count": 0,
            "is_ready": False,
            "error": str(exc),
        }


def refresh() -> None:
    """
    Purges localized execution caches (visualization_cache,
    analytics_results) so that any mapping or type-override change is
    reflected on the next dashboard/audit rerun, without discarding the
    active dataset, Column Registry, or type-override ledger. Never raises.
    """
    try:
        st.session_state["visualization_cache"] = {}
        st.session_state["analytics_results"] = {}
    except Exception:  # noqa: BLE001
        return


def export(export_format: str) -> Optional[Any]:
    """
    Exports the current schema-mapping state. Supported formats:
        "mapping_csv"          -> bytes  (registry.summary_table() as CSV)
        "mapping_json"         -> str    (role -> mapping detail JSON)
        "type_overrides_json"  -> str    (column -> overridden InferredType)
    Returns None for an unrecognized format, a missing Column Registry, or
    any internal failure — never raises.
    """
    try:
        fmt = export_format.strip().lower()
        registry: Optional[ColumnRegistry] = st.session_state.get("column_registry")
        if registry is None:
            return None

        if fmt == "mapping_csv":
            summary_rows = registry.summary_table()
            if not summary_rows:
                return None
            buffer = io.StringIO()
            pd.DataFrame(summary_rows).to_csv(buffer, index=False)
            return buffer.getvalue().encode("utf-8")

        if fmt == "mapping_json":
            payload = {
                role: {
                    "column_name": mapping.column_name,
                    "confidence": mapping.confidence,
                    "source": mapping.source,
                    "confirmed": mapping.confirmed,
                }
                for role, mapping in registry.mappings.items()
            }
            return json.dumps(payload, indent=2, default=str)

        if fmt == "type_overrides_json":
            overrides = st.session_state.get("column_type_overrides", {})
            return json.dumps(overrides, indent=2, default=str)

        return None
    except Exception:  # noqa: BLE001
        return None


def metadata() -> Dict[str, Any]:
    """
    Returns the Schema Mapping Studio's capability descriptor: every
    canonical role it can bind (grouped by cognitive cluster), supported
    export formats, and supported column-type override targets. Never
    raises.
    """
    try:
        return {
            "component": _COMPONENT_NAME,
            "supported_roles": list(_ALL_ROLES),
            "role_groups": {label: list(roles) for label, roles in _ROLE_GROUPS.items()},
            "supported_export_formats": ["mapping_csv", "mapping_json", "type_overrides_json"],
            "supported_type_overrides": [t.value for t in InferredType],
        }
    except Exception as exc:  # noqa: BLE001
        return {"component": _COMPONENT_NAME, "error": str(exc)}


def render(**kwargs: Any) -> None:
    """
    Renders the complete Schema Mapping Studio: summary metrics, bulk
    mapping actions (Auto-Suggest / Reset / Refresh Cache), the grouped
    Role Mapping Deck, the Column Type Override Studio, the Unmapped
    Columns & Fuzzy Suggestions panel, and the Active Mapping Summary with
    CSV/JSON export. This is the sole entry point components/schema_mapping
    .py exposes for pages/1_dashboard.py or a dedicated Schema Mapping page
    to invoke. Never raises: every interactive action is wrapped so a
    single failure degrades to an inline st.error/st.caption rather than
    crashing the host page.

    SCOPE-NORMALIZATION FIX: _claimed_cols_snapshot and
    _profiles_list_snapshot are computed exactly ONCE here, at the top of
    render(), then persisted into st.session_state so every consumer —
    the Role Mapping Deck fragment, the Apply Staged Mappings fragment,
    and the later Unmapped Columns expander in this same render() body —
    reads the identical snapshot without recomputation and without a
    NameError, regardless of which fragment boundary last reran.
    """
    df: Optional[pd.DataFrame] = st.session_state.get("analytics_ready_dataframe")
    registry: Optional[ColumnRegistry] = st.session_state.get("column_registry")
    profiles: List[ColumnProfile] = st.session_state.get("column_profiles", [])

    st.markdown(
        '<div class="kesco-section-title">Schema Mapping Studio — Raw-to-Domain Metadata Deck</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Bind incoming column headers to canonical KESCO operational roles. Every downstream KPI, "
        "chart, and executive narrative resolves exclusively through these bindings — nothing in the "
        "analytics or visualization layer ever references a raw column name directly."
    )

    if df is None or df.empty or registry is None:
        st.info(
            "No active dataset is available for schema mapping. Upload a dataset via the sidebar's "
            "Unified Ingestion panel to begin binding columns to operational roles."
        )
        return
    # ── NEW: top-of-page staged-mapping visibility guard ─────────────────
    # FIX: previously the "N staged, not applied" warning only rendered
    # inside _render_apply_staged_mappings_fragment, BELOW the entire
    # 8-group Role Mapping Deck. A user could see the correct column
    # already pre-selected in every dropdown (staged automatically by the
    # fuzzy-suggestion default_index logic), assume the mapping was live,
    # and navigate to the Dashboard without ever pressing "Apply Staged
    # Mappings" — leaving registry.mappings[role].confirmed = False and
    # every downstream registry.resolve(role) call returning None. This
    # banner is now rendered FIRST, before any role controls, so it is
    # impossible to scroll past without seeing it.
    _pending_preview: Dict[str, Optional[str]] = st.session_state.get("_schema_pending_mappings", {})
    if _pending_preview:
        st.warning(
            f"⚠️ {len(_pending_preview)} role mapping(s) are staged but NOT YET applied to the "
            f"active registry: {', '.join(registry.display_name(r) for r in _pending_preview.keys())}. "
            f"These changes will NOT be visible on the Dashboard or Self-Service pages until you "
            f"scroll down and click **Apply Staged Mappings**."
        )

    profiles_by_col = _profiles_by_column(profiles)

    # ── SCOPE-NORMALIZATION: compute once, persist to session_state ──────
    # These two snapshots feed THREE separate consumers below:
    #   1. _render_role_mapping_deck_fragment (passed as args)
    #   2. _render_apply_staged_mappings_fragment (via registry/df only —
    #      does not need the snapshot, but is listed here for clarity)
    #   3. The "Unmapped Columns & Fuzzy Suggestions" expander further
    #      down in THIS SAME render() call, which previously referenced a
    #      variable that only existed inside the fragment's local scope —
    #      the exact cause of the reported NameError.
    # Recomputed once per main-page rerun (i.e. once per render() call),
    # NOT once per fragment-scoped rerun, since registry.version only
    # changes when a mapping is actually committed.
    _claimed_cols_snapshot: set = _claimed_columns(registry)
    _profiles_list_snapshot: List[ColumnProfile] = list(profiles_by_col.values())
    st.session_state["_schema_claimed_columns_snapshot"] = _claimed_cols_snapshot
    st.session_state["_schema_profiles_list_snapshot"] = _profiles_list_snapshot

    validation_report = validate()

    pending_warnings: List[str] = st.session_state.get("_schema_mapping_pending_warnings", [])
    if pending_warnings:
        for warning_text in pending_warnings:
            st.warning(warning_text)
        st.session_state["_schema_mapping_pending_warnings"] = []

    metric_cols = st.columns(4)
    metric_cols[0].metric("Total Columns", f"{validation_report['total_columns']:,}")
    metric_cols[1].metric(
        "Confirmed Role Mappings",
        f"{validation_report['confirmed_role_count']} / {validation_report['total_addressable_roles']}",
    )
    metric_cols[2].metric("Columns Needing Review", f"{validation_report['needs_review_column_count']}")
    metric_cols[3].metric("Analytics Readiness", f"{st.session_state.get('readiness_score', 0)}/100")

    st.divider()

    action_col_a, action_col_b, action_col_c = st.columns(3)
    with action_col_a:
        if st.button("Auto-Suggest Unmapped Roles", key="_schema_auto_suggest_btn", width="stretch"):
            with st.spinner("Resolving Fuzzy Header Dictionaries, Ranking Candidate Columns..."):
                applied = _auto_suggest_unmapped_roles(registry, profiles)
            if applied > 0:
                _log_mapping_change(
                    "mapping",
                    f"Auto-suggestion applied {applied} role mapping(s) from fuzzy header matching.",
                )
                st.session_state["visualization_cache"] = {}
                st.session_state["analytics_results"] = {}
                st.success(f"Applied {applied} suggested role mapping(s).")
                st.rerun()
            else:
                st.info("No additional confident role suggestions were found for the remaining columns.")
    with action_col_b:
        if st.button("Reset All Mappings", key="_schema_reset_all_btn", width="stretch"):
            cleared = _reset_all_mappings(registry)
            _log_mapping_change("mapping", f"Reset {cleared} role mapping(s) to an unmapped state.")
            st.session_state["visualization_cache"] = {}
            st.session_state["analytics_results"] = {}
            st.success(f"Cleared {cleared} role mapping(s).")
            st.rerun()
    with action_col_c:
        if st.button("Refresh Mapping Cache", key="_schema_refresh_cache_btn", width="stretch"):
            refresh()
            st.success("Visualization and analytics caches purged.")
            st.rerun()

    st.divider()
    st.markdown('<div class="kesco-section-title">Role Mapping Deck</div>', unsafe_allow_html=True)

    # --- Fragment Start ---
    # Snapshots passed explicitly as arguments — the fragment never
    # recomputes them and never reaches into an outer-scope local variable.
    _render_role_mapping_deck_fragment(
        df, registry, profiles_by_col,
        _claimed_cols_snapshot, _profiles_list_snapshot,
    )
    # --- Fragment End ---

    # --- Fragment Start ---
    _render_apply_staged_mappings_fragment(registry, df)
    # --- Fragment End ---

    _render_custom_role_registration(df, registry)
    st.divider()
    with st.expander("Column Type Override Studio", expanded=False):
        st.caption(
            "Override the auto-detected type for any column. Applying an override re-casts the "
            "column on a fresh copy of the analytics-ready dataset — the original upload and cleaned "
            "dataset references remain untouched."
        )
        if not profiles:
            st.caption("No column profiles are available for the active dataset.")
        else:
            overrides: Dict[str, str] = st.session_state.setdefault("column_type_overrides", {})
            type_options: List[str] = [t.value for t in InferredType]
            for profile in profiles:
                col_a, col_b, col_c, col_d = st.columns([3, 2, 2, 1])
                with col_a:
                    st.write(f"**{profile.original_name}**")
                with col_b:
                    st.caption(f"Detected: {profile.inferred_type.value.title()} ({profile.confidence:.0%})")
                with col_c:
                    current_override = overrides.get(profile.original_name, profile.inferred_type.value)
                    override_index = (
                        type_options.index(current_override) if current_override in type_options else 0
                    )
                    selected_type = st.selectbox(
                        "Override",
                        options=type_options,
                        index=override_index,
                        key=f"_schema_type_override_{profile.original_name}",
                        label_visibility="collapsed",
                        format_func=lambda v: v.title(),
                    )
                with col_d:
                    if st.button(
                        "Apply", key=f"_schema_apply_override_{profile.original_name}", width="stretch"
                    ):
                        target_type = InferredType(selected_type)
                        working_df: Optional[pd.DataFrame] = st.session_state.get("analytics_ready_dataframe")
                        if working_df is not None and profile.original_name in working_df.columns:
                            casted_df = working_df.copy(deep=True)
                            coerced_series, success = _apply_type_override(
                                casted_df[profile.original_name], target_type
                            )
                            if success:
                                casted_df[profile.original_name] = coerced_series
                                st.session_state["analytics_ready_dataframe"] = casted_df
                                st.session_state["filtered_dataframe"] = casted_df
                                overrides[profile.original_name] = selected_type
                                st.session_state["column_type_overrides"] = overrides
                                st.session_state["visualization_cache"] = {}
                                st.session_state["analytics_results"] = {}
                                _log_mapping_change(
                                    "mapping",
                                    f"Column '{profile.original_name}' type overridden to "
                                    f"'{target_type.value}'.",
                                    rows_affected=int(len(casted_df)),
                                )
                                st.success(f"'{profile.original_name}' recast as {target_type.value}.")
                                st.rerun()
                            else:
                                log_exception(
                                    "schema_mapping.apply_type_override",
                                    ValueError(f"Coercion failed for column '{profile.original_name}' to type '{target_type.value}'"),
                                    severity="warning",
                                    context={"column": profile.original_name, "target_type": target_type.value},
                                )
                                st.error(
                                    f"Unable to coerce '{profile.original_name}' to {target_type.value}."
                                )
                        else:
                            st.error(f"Column '{profile.original_name}' is no longer present in the active dataset.")

    @st.cache_data(show_spinner=False)
    def _cached_unmapped_suggestions(
        _registry_ref: ColumnRegistry,
        registry_version: int,
        unclaimed_headers_tuple: Tuple[str, ...],
    ) -> Dict[str, List[str]]:
        """Cached wrapper around registry.suggestions_for_unmapped. Hashed
        on registry_version + the unclaimed header tuple, so this only
        recomputes when the actual mapping state or column set changes —
        not on every unrelated widget interaction on the page."""
        return _registry_ref.suggestions_for_unmapped(list(unclaimed_headers_tuple))

    with st.expander("Unmapped Columns & Fuzzy Suggestions", expanded=False):
        # FIXED: previously read the bare local `_claimed_cols_snapshot`
        # which — after the fragment refactor — no longer existed in this
        # scope once the deck loop's computation moved into the fragment.
        # Now reads the render()-level snapshot straight from
        # st.session_state, which is always populated at the top of this
        # very function call, eliminating the NameError.
        claimed_cols = st.session_state.get("_schema_claimed_columns_snapshot", set())
        unclaimed_headers = [c for c in df.columns if c not in claimed_cols]
        if not unclaimed_headers:
            st.caption("Every column in the active dataset is currently claimed by a confirmed role mapping.")
        else:
            suggestion_map = _cached_unmapped_suggestions(
                registry, registry.version, tuple(unclaimed_headers)
            )
            suggestion_rows: List[Dict[str, str]] = []
            for header in unclaimed_headers:
                candidates = suggestion_map.get(header, [])
                suggestion_rows.append({
                    "Column": header,
                    "Suggested Role(s)": (
                        ", ".join(registry.display_name(r) for r in candidates)
                        if candidates else "No confident match"
                    ),
                })
            render_enterprise_grid(
                pd.DataFrame(suggestion_rows), key="_schema_unmapped_grid", search_columns=["Column"],
            )

    st.divider()
    st.markdown('<div class="kesco-section-title">Active Mapping Summary</div>', unsafe_allow_html=True)
    summary_rows = registry.summary_table()
    if summary_rows:
        render_enterprise_grid(
            pd.DataFrame(summary_rows), key="_schema_mapping_summary_grid",
            search_columns=["Role", "Resolved Column"],
        )
    else:
        st.caption("No role mappings have been established yet.")

    export_col_a, export_col_b = st.columns(2)
    with export_col_a:
        mapping_csv = export("mapping_csv")
        if mapping_csv:
            st.download_button(
                "Export Mapping Summary (CSV)",
                data=mapping_csv,
                file_name="schema_mapping_summary.csv",
                mime="text/csv",
                key="_schema_export_csv",
                width="stretch",
            )

    with export_col_b:
        mapping_json = export("mapping_json")
        if mapping_json:
            st.download_button(
                "Export Mapping Profile (JSON)",
                data=mapping_json.encode("utf-8"),
                file_name="schema_mapping_profile.json",
                mime="application/json",
                key="_schema_export_json",
                width="stretch",
            )