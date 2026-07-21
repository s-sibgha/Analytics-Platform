## analytics.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import pandas as pd


# AFTER
from core.roles import (
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
    ROLE_REOPEN_FLAG,
    ROLE_SLA_DEADLINE,
    ROLE_STATUS,
    ROLE_SUBCATEGORY,
    ROLE_SUBDIVISION,
    ROLE_SUBSTATION,
    ROLE_TRANSFORMER,
    ROLE_ZONE,
    CANONICAL_GEO_HIERARCHY,
)
from core.column_registry import ColumnRegistry

# ── Milestone 17 / Task 3 — DuckDB Count Acceleration imports ───────────────
# Additive-only import. Nothing already imported anywhere in this file (or
# any dependent module) is altered, removed, or re-pointed — this is a new,
# purely additive import used exclusively by the safety-audited, integer-
# count-only acceleration hooks inside compute_officer_productivity and
# compute_category_breakdown (see the module-level safety-audit comment
# above those two functions for the full risk analysis). Any import
# failure (e.g. the optional `duckdb` package genuinely unavailable at the
# Python-package level, as opposed to merely unconnectable at runtime)
# degrades every acceleration hook to a no-op via the try/except guards
# around each call site — this module NEVER hard-fails if
# engine.duckdb_executor cannot be imported.
try:
    from engine.duckdb_executor import (
        should_use_duckdb as _should_use_duckdb,
        _get_connection_for_thread as _duckdb_get_connection_for_thread,
        _sanitize_identifier as _duckdb_sanitize_identifier,
        _quote_identifier as _duckdb_quote_identifier,
    )
    _DUCKDB_ACCELERATOR_AVAILABLE = True
except Exception:  # noqa: BLE001
    _DUCKDB_ACCELERATOR_AVAILABLE = False

# ── Refactor Phase 2B — Parquet-native accelerator import ───────────────────
# Additive-only, mirrors the exact defensive-import pattern above. Used
# exclusively by _duckdb_accelerated_group_counts's Parquet-native branch
# (see ComplaintKPIEngine.__init__'s `parquet_path` parameter). Any import
# failure degrades the Parquet-native path to a no-op, falling through to
# the existing, unmodified pandas-replacement-scan / pandas groupby paths —
# this module NEVER hard-fails if engine.analytics_duckdb_accelerator
# cannot be imported.
try:
    from engine.analytics_duckdb_accelerator import (
        duckdb_officer_or_category_productivity as _duckdb_officer_or_category_productivity,
    )
    _PARQUET_ACCELERATOR_AVAILABLE = True
except Exception:  # noqa: BLE001
    _PARQUET_ACCELERATOR_AVAILABLE = False


# ── Status ontology (single source of truth) ─────────────────────────────────
# These three constants are the canonical business-status buckets. They were
# referenced-but-never-assigned in the prior revision; the exhaustive token
# lists themselves are preserved verbatim from the original authoritative
# source — only the missing string assignments have been added.
_STATUS_CLOSED = "CLOSED"
_STATUS_PENDING = "PENDING"
_STATUS_REOPENED = "REOPENED"

_STATUS_TOKENS: Dict[str, FrozenSet[str]] = {
    _STATUS_CLOSED: frozenset({"closed", "resolved", "completed"}),
    _STATUS_PENDING: frozenset({"pending", "open", "assigned", "in progress"}),
    _STATUS_REOPENED: frozenset({"reopened", "re-opened", "escalated"}),
}

_CLOSURE_RATE_EXCELLENT = 80.0
_CLOSURE_RATE_GOOD = 60.0
_CLOSURE_RATE_FAIR = 40.0
_REOPEN_RATE_EXCELLENT = 5.0
_REOPEN_RATE_GOOD = 10.0
_REOPEN_RATE_WARNING = 20.0
_SLA_COMPLIANCE_EXCELLENT = 90.0
_SLA_COMPLIANCE_GOOD = 75.0
_SLA_COMPLIANCE_WARNING = 50.0

# Legacy day-denominated resolution-time bands — preserved (unused by the
# now-hour-denominated _avg_resolution_time) per the non-removal mandate.
_AVG_RESOLUTION_EXCELLENT = 3.0
_AVG_RESOLUTION_GOOD = 7.0
_AVG_RESOLUTION_WARNING = 15.0

# Authoritative hour-denominated MTTR bands (72h/168h/360h == 3/7/15 days),
# used by the unified, 95th-percentile-clipped _avg_resolution_time KPI.
_AVG_RESOLUTION_EXCELLENT_HRS = 72.0
_AVG_RESOLUTION_GOOD_HRS = 168.0
_AVG_RESOLUTION_WARNING_HRS = 360.0

_AVG_PENDING_AGE_EXCELLENT = 7.0
_AVG_PENDING_AGE_GOOD = 15.0
_AVG_PENDING_AGE_WARNING = 30.0
_PENDING_AGE_HARD_CEILING_DAYS: float = 3650.0   # 10-year sanity ceiling (filters 1900-01-01 defaults)
_PENDING_AGE_ADAPTIVE_FLOOR_DAYS: float = 365.0  # never fence tighter than 1 year
_PENDING_AGE_IQR_MULTIPLIER: float = 3.0         # Tukey's Fence multiplier

# ── Executive Narrative Engine constants (migrated from the report layer) ────
_MODIFIED_Z_CONSTANT: float = 0.6745
_PARETO_TARGET_PCT: float = 80.0
_PARETO_TRUNCATE_THRESHOLD: int = 5
_PARETO_TRUNCATE_KEEP: int = 3

# Absolute geographic priority for the Pareto hierarchy drill, per explicit
# Phase 2 confirmation: ZONE -> CIRCLE -> DIVISION -> SUBDIVISION ->
# SUBSTATION (ROLE_TRANSFORMER) -> FEEDER.
_GEO_HIERARCHY_PRIORITY: Tuple[Tuple[str, str], ...] = CANONICAL_GEO_HIERARCHY

# Administrative-only subset of the canonical geo hierarchy (excludes the
# asset-level Feeder/Transformer tiers), used by compute_hierarchy_risk and
# by MetricEligibilityEngine's dynamic hierarchy_risk eligibility check
# (Issue 1 remediation) to determine whether ANY administrative grouping
# dimension is resolvable.
_ADMIN_HIERARCHY_ROLES: Tuple[str, ...] = (
    ROLE_ZONE, ROLE_CIRCLE, ROLE_DIVISION, ROLE_SUBDIVISION, ROLE_SUBSTATION,
)

# Secondary, flat, non-geographic fallback — used ONLY when no geographic
# hierarchy role can be resolved via the registry.
_FALLBACK_HIERARCHY_PRIORITY: Tuple[Tuple[str, str], ...] = (
    (ROLE_OFFICER, "Officer"),
    (ROLE_CATEGORY, "Complaint Category"),
)

_EMPTY_DATASET_MESSAGE: str = (
    "The active dataset is empty — no records available for KPI computation."
)


@dataclass
class KPIResult:
    name: str
    value: Any
    formatted_value: str
    unit: str
    definition: str
    formula: str
    interpretation: str
    recommendation: str
    benchmark_status: str
    trend_direction: str
    trend_is_positive: Optional[bool]
    previous_value: Optional[Any]
    pct_change: Optional[float]
    required_roles: List[str]
    missing_roles: List[str]
    is_eligible: bool
    ineligibility_reason: str = ""

    @property
    def status(self) -> str:
        """Backward-compatible alias for `benchmark_status`.

        Additive-only: existing consumers (e.g. visualization/kpi_cards.py)
        read `.status` directly. This property does not rename or remove
        `benchmark_status` — it simply exposes the same value under the
        name the established downstream contract expects, with zero risk
        to any other consumer that already uses `benchmark_status`.
        """
        return self.benchmark_status


@dataclass
class EligibilityReport:
    eligible: List[str]
    ineligible: Dict[str, List[str]]


@dataclass
class OfficerProductivityRow:
    officer: str
    total_cases: int
    closed_cases: int
    pending_cases: int
    reopened_cases: int
    closure_rate: float
    avg_resolution_days: Optional[float]
    pending_load: int


@dataclass
class CategoryBreakdownRow:
    category: str
    total: int
    closed: int
    pending: int
    closure_rate: float
    avg_resolution_days: Optional[float]


@dataclass
class MonthlyTrendRow:
    period_label: str
    year: int
    month: int
    total: int
    closed: int
    pending: int
    closure_rate: float


@dataclass
class HierarchyRiskRow:
    group_name: str
    group_field: str
    total_cases: int
    pending_cases: int
    avg_pending_age_days: Optional[float]
    max_pending_age_days: Optional[float]
    risk_score: float
    risk_tier: str


@dataclass
class ComplaintAnalyticsBundle:
    kpis: Dict[str, KPIResult]
    eligibility_report: EligibilityReport
    officer_productivity: Optional[pd.DataFrame]
    category_breakdown: Optional[pd.DataFrame]
    monthly_trend: Optional[pd.DataFrame]
    hierarchy_risk: Optional[pd.DataFrame]
    top_repeat_consumers: Optional[pd.DataFrame]
    sla_breach_detail: Optional[pd.DataFrame]


# ── Phase 2: Executive Narrative Engine data contracts ───────────────────────

@dataclass
class ExecutiveMeta:
    generated_at: str
    date_range: Tuple[str, str]
    lowest_hierarchy_unit: str
    hierarchy_role: Optional[str]
    lifecycle_method: str          # "dates" | "status" | "fallback" | "unresolved"
    calculation_quality: str       # "MEASURED" | "ESTIMATED"
    estimation_reason: str


@dataclass
class ExecutiveGlobalKPIs:
    sla_compliance: float
    sla_pop_delta: float
    mttr_hours: float
    mttr_pop_delta: float
    total_volume: int
    pending_count: int
    avg_pending_age_days: float
    business_risk_score: float


@dataclass
class HierarchyAnomaly:
    unit_type: str
    unit_name: str
    pending_count: int
    avg_pending_age: float
    risk_score: float
    z_score: float
    severity: str


@dataclass
class ParetoHierarchyNode:
    level: str
    name: str
    pending_volume: int
    backlog_contribution_pct: float
    cumulative_pct: float
    is_vital_few: bool
    children: List["ParetoHierarchyNode"] = field(default_factory=list)


@dataclass
class ParetoHotspotRow:
    category: str
    administrative_unit: str
    backlog_contribution_pct: float


@dataclass
class ExecutiveNarrativeBundle:
    meta: ExecutiveMeta
    global_kpis: ExecutiveGlobalKPIs
    anomalies: List[HierarchyAnomaly]
    pareto_hierarchy: List[ParetoHierarchyNode]
    pareto_hotspots_flat: List[ParetoHotspotRow]
    pareto_omitted_count: int
    kpi_snapshot: Dict[str, KPIResult]


_KPI_REQUIRED_ROLES: Dict[str, List[str]] = {
    "total_cases": [ROLE_RECORD_ID],
    "closed_cases": [ROLE_RECORD_ID, ROLE_STATUS],
    "pending_cases": [ROLE_RECORD_ID, ROLE_STATUS],
    "reopened_cases": [ROLE_RECORD_ID, ROLE_STATUS],
    "closure_rate": [ROLE_RECORD_ID, ROLE_STATUS],
    "pending_rate": [ROLE_RECORD_ID, ROLE_STATUS],
    "reopen_rate": [ROLE_RECORD_ID, ROLE_STATUS],
    "backlog": [ROLE_RECORD_ID, ROLE_STATUS],
    "avg_resolution_time": [ROLE_REGISTRATION_DATE, ROLE_CLOSING_DATE, ROLE_STATUS],
    "median_resolution_time": [ROLE_REGISTRATION_DATE, ROLE_CLOSING_DATE, ROLE_STATUS],
    "p95_resolution_time": [ROLE_REGISTRATION_DATE, ROLE_CLOSING_DATE, ROLE_STATUS],
    "avg_pending_age": [ROLE_REGISTRATION_DATE, ROLE_STATUS],
    "median_pending_age": [ROLE_REGISTRATION_DATE, ROLE_STATUS],
    "p95_pending_age": [ROLE_REGISTRATION_DATE, ROLE_STATUS],
    "sla_compliance_rate": [ROLE_STATUS, ROLE_CLOSING_DATE, ROLE_SLA_DEADLINE],
    "sla_breach_rate": [ROLE_STATUS, ROLE_CLOSING_DATE, ROLE_SLA_DEADLINE],
    "unique_consumers": [ROLE_CONSUMER_ID],
    "repeat_consumer_rate": [ROLE_CONSUMER_ID, ROLE_RECORD_ID],
    "first_time_resolution_rate": [ROLE_RECORD_ID, ROLE_STATUS],
    "mom_growth": [ROLE_RECORD_ID, ROLE_REGISTRATION_DATE],
    "qoq_growth": [ROLE_RECORD_ID, ROLE_REGISTRATION_DATE],
    "yoy_growth": [ROLE_RECORD_ID, ROLE_REGISTRATION_DATE],
    "officer_productivity": [ROLE_OFFICER, ROLE_RECORD_ID, ROLE_STATUS],
    "category_breakdown": [ROLE_CATEGORY, ROLE_RECORD_ID, ROLE_STATUS],
    "monthly_trend": [ROLE_RECORD_ID, ROLE_REGISTRATION_DATE],
    "hierarchy_risk": [ROLE_RECORD_ID, ROLE_STATUS],
    "top_repeat_consumers": [ROLE_CONSUMER_ID, ROLE_RECORD_ID],
    "sla_breach_detail": [ROLE_STATUS, ROLE_CLOSING_DATE, ROLE_SLA_DEADLINE],
}

# ══════════════════════════════════════════════════════════════════════════════
# Milestone 17 / Task 1 — Flexible-OR Eligibility Groups
#
# `_KPI_REQUIRED_ROLES` above is the platform's original, UNCHANGED static
# strict-AND role-requirement declaration (per explicit deliverable
# constraint: "Do not change the KPI_REQUIRED_ROLES structure"). The two
# frozensets below do not touch that dict at all — they instead tell
# MetricEligibilityEngine.check() which KPI names should have their
# ROLE_STATUS requirement RELAXED at eligibility-check time, because the
# platform can derive an equivalent Closed/Pending lifecycle signal
# directly from Closing Date presence/absence whenever Status is not
# mapped (see ComplaintKPIEngine._resolve_flexible_closed_pending_masks).
#
#   GROUP A — Operational breakdown tables. Requirement becomes:
#       (ROLE_STATUS is mapped) OR (ROLE_CLOSING_DATE is mapped)
#       — i.e. "at least one of the two", exactly matching the requested
#       "STATUS OR CLOSING_DATE OR (both)" rule (that phrasing is
#       logically an inclusive-OR over two booleans).
#
#   GROUP B — Resolution/SLA metrics. Requirement becomes:
#       ROLE_CLOSING_DATE is mapped (unconditionally) — ROLE_STATUS
#       becomes fully optional. This is the flattened form of the
#       requested "CLOSING_DATE (primary) OR (both STATUS and
#       CLOSING_DATE)" rule: the second disjunct is a strict subset of
#       the first (it already requires CLOSING_DATE), so the rule
#       simplifies to "CLOSING_DATE required, STATUS optional."
#       ROLE_CLOSING_DATE is added to the effective required-role set
#       even for KPIs whose static _KPI_REQUIRED_ROLES entry does not
#       declare it (closed_cases, first_time_resolution_rate) — see
#       MetricEligibilityEngine.check() for the exact mechanics.
# ══════════════════════════════════════════════════════════════════════════════
_FLEXIBLE_GROUP_A_KPIS: FrozenSet[str] = frozenset({
    "officer_productivity", "category_breakdown", "hierarchy_risk",
})
_FLEXIBLE_GROUP_B_KPIS: FrozenSet[str] = frozenset({
    "closed_cases",
    "avg_resolution_time", "median_resolution_time", "p95_resolution_time",
    "sla_compliance_rate", "sla_breach_rate", "sla_breach_detail",
    "first_time_resolution_rate",
})


class MetricEligibilityEngine:

    def __init__(self, registry: ColumnRegistry) -> None:
        self._registry = registry

    def check(self, kpi_name: str) -> Tuple[bool, List[str]]:
        """
        Returns (is_eligible, missing_roles).

        Milestone 18 / per-KPI conditional-branch remediation: this method
        no longer applies a single blanket rule across an entire group of
        KPIs — it now resolves eligibility strictly from each KPI's own
        `_KPI_REQUIRED_ROLES` declaration first, and only then asks
        "does this specific KPI need a Flexible-OR relaxation, and if so,
        which shape?" This eliminates the risk of a future KPI being
        silently mis-classified by group membership alone; the group
        membership sets (`_FLEXIBLE_GROUP_A_KPIS` / `_FLEXIBLE_GROUP_B_KPIS`)
        are still the single source of truth for WHICH relaxation shape
        applies, but the actual role-presence checks are now driven
        explicitly off `required` for every branch.

        Resolution order for every kpi_name:
          1. Look up `required = _KPI_REQUIRED_ROLES.get(kpi_name, [])` —
             the KPI's own static declaration. Nothing below ever expands
             this list implicitly; every relaxation is a SUBTRACTION or a
             substitution of an already-declared role, never an addition
             of a role the KPI didn't originally declare.
          2. GROUP A (e.g. officer_productivity, category_breakdown,
             hierarchy_risk): Status and Closing Date act as an inclusive
             OR pair — EITHER satisfies the lifecycle-resolution
             requirement. This OR gate is only ever evaluated if
             `ROLE_STATUS` is actually present in `required` for this KPI;
             if a Group-A KPI's declaration never listed Status in the
             first place, the OR gate is skipped entirely and Closing Date
             (if declared) is checked as a plain mandatory role instead.
          3. GROUP B (e.g. avg_resolution_time, median_resolution_time,
             p95_resolution_time, sla_compliance_rate, sla_breach_rate,
             sla_breach_detail, closed_cases, first_time_resolution_rate):
             Closing Date is the mandatory anchor; Status is ALWAYS
             optional for these KPIs regardless of whether the static
             `required` list happens to include it — explicitly guarded
             below via `if ROLE_STATUS in required:` so Status is stripped
             out of the enforced set rather than silently left in.
          4. DEFAULT (every other KPI, e.g. total_cases, pending_cases,
             unique_consumers, mom_growth, ...): strict AND over exactly
             the roles the KPI declared — nothing added, nothing relaxed.
             `ROLE_STATUS` is enforced here ONLY if it is actually present
             in `required`; a KPI that never declared it can never be
             blocked by it.

        Every branch still resolves eligibility exclusively via
        `registry.resolve(role) is not None` (through `has_role`/
        `missing_roles`), so no role is ever reported Missing while
        `registry.resolve(role)` actually returns a non-None value.
        Never raises.
        """
        required = _KPI_REQUIRED_ROLES.get(kpi_name, [])

        # ── STEP 1: identify which mandatory (non-relaxed) roles this KPI
        # actually declares, independent of any group membership. This is
        # recomputed inside each branch below against that branch's own
        # effective_required, but `required` itself is never mutated.
        status_is_declared = ROLE_STATUS in required
        closing_date_is_declared = ROLE_CLOSING_DATE in required

        # ══════════════════════════════════════════════════════════════
        # STEP 2 — GROUP A: Flexible-OR (Status OR Closing Date)
        # ══════════════════════════════════════════════════════════════
        if kpi_name in _FLEXIBLE_GROUP_A_KPIS:
            # Strip Status from the strictly-enforced set — it participates
            # only in the OR gate below, never as an unconditional AND term.
            effective_required = [r for r in required if r != ROLE_STATUS]
            missing = self._registry.missing_roles(effective_required)

            # The OR gate itself only makes sense if this KPI's original
            # declaration actually named Status as a requirement in the
            # first place. If it didn't, there is nothing to relax — Closing
            # Date (if declared) is already being checked as a plain
            # mandatory role via effective_required above, and Status is
            # simply irrelevant to this KPI.
            if status_is_declared:
                has_status = self._registry.has_role(ROLE_STATUS)
                has_closing_date = self._registry.has_role(ROLE_CLOSING_DATE)
                if not has_status and not has_closing_date:
                    # Neither half of the OR pair resolves — report BOTH
                    # roles as missing so the UI communicates that mapping
                    # EITHER one clears the requirement, not just Status.
                    for candidate in (ROLE_STATUS, ROLE_CLOSING_DATE):
                        if candidate not in missing:
                            missing = list(missing) + [candidate]

            if kpi_name == "hierarchy_risk" and len(missing) == 0:
                if not any(self._registry.has_role(r) for r in _ADMIN_HIERARCHY_ROLES):
                    missing = list(missing) + [
                        r for r in _ADMIN_HIERARCHY_ROLES if not self._registry.has_role(r)
                    ]

            return len(missing) == 0, missing

        # ══════════════════════════════════════════════════════════════
        # STEP 2 — GROUP B: Closing Date mandatory anchor, Status optional
        # (avg_resolution_time, median_resolution_time, p95_resolution_time,
        # sla_compliance_rate, sla_breach_rate, sla_breach_detail,
        # closed_cases, first_time_resolution_rate)
        # ══════════════════════════════════════════════════════════════
        if kpi_name in _FLEXIBLE_GROUP_B_KPIS:
            # STEP 3: if Status was declared, ignore it entirely for this
            # KPI — it is never enforced and never checked, regardless of
            # whether it resolves in the registry or not.
            effective_required = [r for r in required if r != ROLE_STATUS]

            # Closing Date is the mandatory anchor for every Group B KPI,
            # even for KPIs (e.g. closed_cases, first_time_resolution_rate)
            # whose static declaration never explicitly listed it — this is
            # the one deliberate, documented widening in this method, and
            # it exists because these KPIs are mathematically defined in
            # terms of a resolved lifecycle date, not a status label.
            if not closing_date_is_declared and ROLE_CLOSING_DATE not in effective_required:
                effective_required = effective_required + [ROLE_CLOSING_DATE]

            missing = self._registry.missing_roles(effective_required)
            return len(missing) == 0, missing

        # ══════════════════════════════════════════════════════════════
        # STEP 2 — DEFAULT: strict AND over exactly the declared roles.
        # STEP 3: ROLE_STATUS is enforced here ONLY if it is actually
        # present in `required` — a KPI that never declared it can never
        # be blocked by it, and nothing is ever added to this set.
        # ══════════════════════════════════════════════════════════════
        effective_required = list(required)
        missing = self._registry.missing_roles(effective_required)

        if kpi_name == "hierarchy_risk" and len(missing) == 0:
            # Defensive: hierarchy_risk is normally routed through GROUP A
            # above and should never reach this branch, but this mirrors
            # the original safeguard verbatim in case of future
            # reclassification.
            if not any(self._registry.has_role(r) for r in _ADMIN_HIERARCHY_ROLES):
                missing = [r for r in _ADMIN_HIERARCHY_ROLES if not self._registry.has_role(r)]

        return len(missing) == 0, missing

    def report(self, kpi_names: Optional[List[str]] = None) -> EligibilityReport:
        names = kpi_names or list(_KPI_REQUIRED_ROLES.keys())
        eligible: List[str] = []
        ineligible: Dict[str, List[str]] = {}
        for name in names:
            ok, missing = self.check(name)
            if ok:
                eligible.append(name)
            else:
                ineligible[name] = missing
        return EligibilityReport(eligible=eligible, ineligible=ineligible)

    def _ineligible_result(self, name: str, missing: List[str]) -> KPIResult:
        required = _KPI_REQUIRED_ROLES.get(name, [])
        display_missing = [self._registry.display_name(r) for r in missing]
        return KPIResult(
            name=name,
            value=None,
            formatted_value="N/A",
            unit="",
            definition="",
            formula="",
            interpretation=f"Cannot compute: missing role mapping(s): {', '.join(display_missing)}.",
            recommendation=f"Map the following column roles in the Schema Mapping Studio to enable this KPI: "
                           f"{', '.join(display_missing)}.",
            benchmark_status="na",
            trend_direction="none",
            trend_is_positive=None,
            previous_value=None,
            pct_change=None,
            required_roles=required,
            missing_roles=missing,
            is_eligible=False,
            ineligibility_reason=f"Missing: {', '.join(display_missing)}",
        )


def _safe_pct_change(current: Optional[float], previous: Optional[float]) -> Optional[float]:
    if current is None or previous is None:
        return None
    if previous == 0:
        return None
    return round((current - previous) / abs(previous) * 100, 2)


def _trend_direction(pct_change: Optional[float], min_threshold: float = 0.5) -> str:
    if pct_change is None:
        return "none"
    if pct_change > min_threshold:
        return "up"
    if pct_change < -min_threshold:
        return "down"
    return "neutral"


def _benchmark_rate(
    value: float,
    excellent_threshold: float,
    good_threshold: float,
    fair_threshold: float,
    higher_is_better: bool = True,
) -> str:
    if higher_is_better:
        if value >= excellent_threshold:
            return "excellent"
        if value >= good_threshold:
            return "good"
        if value >= fair_threshold:
            return "fair"
        return "critical"
    else:
        if value <= excellent_threshold:
            return "excellent"
        if value <= good_threshold:
            return "good"
        if value <= fair_threshold:
            return "fair"
        return "critical"




def _period_comparison_growth(
    series: pd.Series, freq: str
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    try:
        try:
            dt = pd.to_datetime(series, errors="coerce", format="mixed").dropna()
        except (ValueError, TypeError):
            dt = pd.to_datetime(series, errors="coerce").dropna()
        if len(dt) < 2:
            return None, None, None
        grouped = dt.dt.to_period(freq).value_counts().sort_index()
        if len(grouped) < 2:
            return None, None, None
        current_val = float(grouped.iloc[-1])
        previous_val = float(grouped.iloc[-2])
        pct = _safe_pct_change(current_val, previous_val)
        return current_val, previous_val, pct
    except Exception:
        return None, None, None

def _resolve_status_mask(
    df: pd.DataFrame, registry: ColumnRegistry, target_status: str
) -> pd.Series:
    status_col = registry.resolve(ROLE_STATUS)
    if not status_col or status_col not in df.columns:
        return pd.Series([False] * len(df), index=df.index)

    # Dynamically fetch the expanded token set, falling back to a single string if not found
    target_set = _STATUS_TOKENS.get(target_status.upper(), frozenset({target_status.lower()}))

    # Normalize the dataframe column and check against the multi-token set
    return df[status_col].astype(str).str.strip().str.lower().isin(target_set)


def _resolve_reopen_mask(df: pd.DataFrame, registry: ColumnRegistry) -> pd.Series:
    status_mask = _resolve_status_mask(df, registry, _STATUS_REOPENED)
    reopen_col = registry.resolve(ROLE_REOPEN_FLAG)
    if not reopen_col or reopen_col not in df.columns:
        return status_mask
    reopen_flag_series = df[reopen_col]
    flag_mask = reopen_flag_series.apply(
        lambda v: str(v).strip().lower() in ("1", "true", "yes", "y", "reopened", "re-opened")
        if pd.notna(v) else False
    )
    return status_mask | flag_mask


def _compute_duration_days(
    df: pd.DataFrame, start_role: str, end_role: str, registry: ColumnRegistry
) -> Optional[pd.Series]:
    start_col = registry.resolve(start_role)
    end_col = registry.resolve(end_role)
    if not start_col or not end_col:
        return None
    if start_col not in df.columns or end_col not in df.columns:
        return None
    try:
        # Refactor Phase 2B / Datetime & Lifecycle Fix: routed through
        # _safe_tz_naive so a tz-aware Parquet-origin column (e.g.
        # closing_date stamped UTC by one upstream system) can never raise
        # "Cannot subtract tz-naive and tz-aware datetime-like objects"
        # against a tz-naive registration_date column parsed independently.
        start = _safe_tz_naive(df[start_col])
        end = _safe_tz_naive(df[end_col])
        days = (end - start).dt.total_seconds() / 86400
        return days.where(days >= 0)
    except Exception:
        return None


# ── Phase 2: shared helpers for the migrated Executive Narrative Engine ─────


def _safe_tz_naive(series: pd.Series) -> pd.Series:
    """Coerces to datetime and strips timezone info, never raising."""
    try:
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        parsed = pd.to_datetime(series, errors="coerce")
    try:
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return parsed


def _normalize_reference_date(reference_date: Optional[pd.Timestamp]) -> Optional[pd.Timestamp]:
    """
    Refactor Phase 2B / Reference-Date Normalization: defensively strips
    timezone metadata from an externally-supplied `reference_date` (e.g. a
    UI-originated `pd.Timestamp` from `st.date_input`, or a caller-supplied
    value derived from a tz-aware Parquet timestamp column) before it is
    ever compared or subtracted against any internally-parsed datetime
    series — every such series is itself normalized via `_safe_tz_naive`
    elsewhere in this module, so leaving `reference_date` tz-aware would
    reintroduce exactly the tz-mismatch `TypeError` this pass eliminates.
    Never raises: any normalization failure returns the original value
    unchanged rather than propagating an exception, and a `None` input
    passes through as `None`.
    """
    if reference_date is None:
        return None
    try:
        ts = pd.Timestamp(reference_date)
    except Exception:  # noqa: BLE001
        return reference_date
    try:
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_localize(None)
    except (TypeError, AttributeError):  # noqa: BLE001
        pass
    return ts


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(v):
        return default
    return v


def _compute_modified_z_scores(values: pd.Series) -> pd.Series:
    """Robust textbook Modified Z-Score. Short-circuits to all-zero when
    MAD == 0.0 to prevent division-by-zero."""
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    median_val = float(clean.median()) if clean.notna().any() else 0.0
    abs_dev = (clean - median_val).abs()
    mad = float(abs_dev.median()) if abs_dev.notna().any() else 0.0
    if mad == 0.0:
        return pd.Series(0.0, index=values.index)
    z = (_MODIFIED_Z_CONSTANT * (clean - median_val)) / mad
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return z


def _severity_from_percentile(percentile: float) -> str:
    if percentile >= 90.0:
        return "CRITICAL"
    if percentile >= 75.0:
        return "HIGH"
    if percentile >= 50.0:
        return "MEDIUM"
    return "LOW"


def _resolve_lowest_hierarchy_unit(
    registry: ColumnRegistry, df: pd.DataFrame
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Returns (role, display_label, column_name) for the most granular
    resolvable, non-empty geographic hierarchy dimension, scanning
    ZONE..FEEDER from most-granular to least-granular. Falls back to the
    secondary flat (Officer/Category) tier only if no geographic role
    resolves. Returns (None, None, None) when nothing is mapped."""
    for role, label in reversed(_GEO_HIERARCHY_PRIORITY):
        try:
            col = registry.resolve(role)
        except Exception:
            col = None
        if col and col in df.columns and df[col].notna().any():
            return role, label, col
    for role, label in reversed(_FALLBACK_HIERARCHY_PRIORITY):
        try:
            col = registry.resolve(role)
        except Exception:
            col = None
        if col and col in df.columns and df[col].notna().any():
            return role, label, col
    return None, None, None


def _build_pareto_hierarchy_recursive(
    data: pd.DataFrame,
    path_cols: List[str],
    path_labels: List[str],
    depth: int,
) -> List[ParetoHierarchyNode]:
    """Recursively builds a nested Pareto drill tree across `path_cols`.

    NOTE ON VECTORIZATION: the per-node object construction loop below
    iterates over an already-aggregated (groupby'd) summary table, whose
    row count is bounded by categorical cardinality at that hierarchy
    level — never by raw dataset row count. Constructing a nested tree of
    dataclass instances has no practical vectorized equivalent, which is
    the explicit exception carved out by the project's row-loop policy.
    Only "vital few" branches are recursed into (trivial-many branches
    become leaves), which both matches standard 80/20 drill-down semantics
    and bounds total tree size on wide, high-cardinality hierarchies.
    """
    if depth >= len(path_cols) or data.empty:
        return []

    col = path_cols[depth]
    label = path_labels[depth]

    grouped = (
        data.groupby(col, dropna=True)
        .size()
        .reset_index(name="pending_volume")
        .sort_values("pending_volume", ascending=False)
        .reset_index(drop=True)
    )
    total = float(grouped["pending_volume"].sum())
    if total <= 0.0 or grouped.empty:
        return []

    grouped["cumulative_pct"] = (grouped["pending_volume"].cumsum() / total * 100.0).round(4)
    grouped["backlog_contribution_pct"] = (grouped["pending_volume"] / total * 100.0).round(4)
    reached_target = grouped["cumulative_pct"] >= _PARETO_TARGET_PCT
    cutoff_idx = int(reached_target.idxmax()) if bool(reached_target.any()) else int(len(grouped) - 1)
    grouped["is_vital_few"] = grouped.index <= cutoff_idx

    records = grouped.to_dict("records")
    nodes: List[ParetoHierarchyNode] = []
    for record in records:
        group_value = str(record[col])
        is_vital = bool(record["is_vital_few"])
        children: List[ParetoHierarchyNode] = []
        if is_vital and (depth + 1) < len(path_cols):
            child_data = data.loc[data[col].astype(str) == group_value]
            children = _build_pareto_hierarchy_recursive(child_data, path_cols, path_labels, depth + 1)
        nodes.append(
            ParetoHierarchyNode(
                level=label,
                name=group_value,
                pending_volume=int(record["pending_volume"]),
                backlog_contribution_pct=float(record["backlog_contribution_pct"]),
                cumulative_pct=float(record["cumulative_pct"]),
                is_vital_few=is_vital,
                children=children,
            )
        )
    return nodes


class ComplaintKPIEngine:

    def __init__(
        self,
        registry: ColumnRegistry,
        eligibility: MetricEligibilityEngine,
        parquet_path: Optional[str] = None,
    ) -> None:
        self._registry = registry
        self._eligibility = eligibility
        # Refactor Phase 2B / Parquet Pathing: optional, purely additive.
        # When supplied, DuckDB count-acceleration hooks
        # (_duckdb_accelerated_group_counts) route through the Parquet-
        # native accelerator (engine.analytics_duckdb_accelerator) instead
        # of the in-memory pandas replacement scan. Every downstream
        # consumer that never passes this argument (the entire pre-Phase-
        # 2B call surface) is completely unaffected — the pandas
        # replacement-scan / pure-pandas fallback path is untouched.
        self._parquet_path = parquet_path

    # ── Public: status ontology accessor (mandate item — Status Token Integrity) ──

    def get_status_tokens(self) -> Dict[str, FrozenSet[str]]:
        """Returns a defensive copy of the canonical status-token ontology.
        This is the sanctioned public accessor for the reporting layer —
        it must never re-derive or hardcode its own token sets."""
        return dict(_STATUS_TOKENS)

    # ── Milestone 17 / Task 2 — Defensive Flexible Lifecycle Resolver ──────

    def _resolve_flexible_closed_pending_masks(
        self, df: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, str]:
        """
        Flexible-OR lifecycle resolver backing every Group A/B KPI that
        became eligible via partial (Status-optional) mapping. This is the
        single, shared "defensive calculation" implementation referenced
        by Task 2 — every consumer below calls this instead of hand-rolling
        its own Status/Closing-Date fallback branch, so the fallback
        semantics can never silently diverge between KPIs.

        Resolution priority (strictly additive / backward-compatible):

          Priority 1 — ROLE_STATUS mapped:
            Delegates to the ORIGINAL, UNMODIFIED module-level
            `_resolve_status_mask()` token-matching function for both the
            Closed and Pending buckets. This produces BYTE-IDENTICAL
            output to the pre-Milestone-17 engine whenever a Status role
            is mapped, regardless of whether Closing Date is also mapped
            — satisfying the "byte-identical when full data is available"
            Integrity Constraint exactly.

          Priority 2 — ROLE_STATUS absent, ROLE_CLOSING_DATE mapped:
            A record is treated as Closed iff its Closing Date value is
            non-null (parseable as a date), and Pending iff its Closing
            Date is null/unparseable. This mirrors the identical
            closing-date-presence heuristic already used elsewhere in the
            platform for the same purpose (see
            visualization.chart_factory._resolve_lifecycle_state's
            Priority 2 branch), so the platform's two independent
            lifecycle resolvers agree by construction rather than by
            coincidence.

          Priority 3 — neither ROLE_STATUS nor ROLE_CLOSING_DATE mapped:
            Both masks are returned all-False, exactly matching the
            original engine's behavior for a fully-unmapped lifecycle
            (i.e. degrades to "cannot determine lifecycle", never guesses).

        Returns:
            (closed_mask, pending_mask, method) where method is one of
            "status_column" | "closing_date_presence" | "unresolved" —
            surfaced for audit/debugging/logging by callers that want it,
            but not required.

        Never raises: any internal coercion failure degrades to the
        Priority 3 all-False fallback.
        """
        try:
            status_col = self._registry.resolve(ROLE_STATUS)
            if status_col and status_col in df.columns:
                closed_mask = _resolve_status_mask(df, self._registry, _STATUS_CLOSED)
                pending_mask = _resolve_status_mask(df, self._registry, _STATUS_PENDING)
                return closed_mask, pending_mask, "status_column"

            closing_col = self._registry.resolve(ROLE_CLOSING_DATE)
            if closing_col and closing_col in df.columns:
                # Refactor Phase 2B — routed through _safe_tz_naive.
                closing_dt = _safe_tz_naive(df[closing_col])
                closed_mask = closing_dt.notna()
                pending_mask = closing_dt.isna()
                return closed_mask, pending_mask, "closing_date_presence"

            false_mask = pd.Series(False, index=df.index)
            return false_mask, false_mask, "unresolved"
        except Exception:  # noqa: BLE001 — never raise; degrade to unresolved
            false_mask = pd.Series(False, index=df.index)
            return false_mask, false_mask, "unresolved"

    # ── Public: unified scalar KPI computation ──────────────────────────────

    def compute_all(self, df: pd.DataFrame, reference_date: Optional[pd.Timestamp] = None) -> Dict[str, KPIResult]:
        """
        Computes every scalar (non-dataframe) KPI defined in this engine.
        Each KPI is independently eligibility-checked and independently
        fault-isolated: a failure or missing-role condition in one KPI can
        never prevent any other KPI from being computed, and can never
        raise to the caller.
        """
        # Refactor Phase 2B / Reference-Date Normalization: strips tz
        # metadata from an externally-supplied reference_date BEFORE it is
        # threaded into any date-dependent KPI dispatch below, guaranteeing
        # mathematical compatibility with every internally-parsed
        # (tz-naive, via _safe_tz_naive) datetime series in this engine.
        reference_date = _normalize_reference_date(reference_date)

        if df is None or df.empty:
            return self._generate_empty_kpis()

        results: Dict[str, KPIResult] = {}

        scalar_dispatch: Dict[str, Callable[[pd.DataFrame], KPIResult]] = {
            "total_cases": self._total_cases,
            "closed_cases": self._closed_cases,
            "pending_cases": self._pending_cases,
            "reopened_cases": self._reopened_cases,
            "closure_rate": self._closure_rate,
            "pending_rate": self._pending_rate,
            "reopen_rate": self._reopen_rate,
            "backlog": self._backlog,
            "avg_resolution_time": self._avg_resolution_time,
            "median_resolution_time": self._median_resolution_time,
            "p95_resolution_time": self._p95_resolution_time,
            "sla_compliance_rate": self._sla_compliance_rate,
            "sla_breach_rate": self._sla_breach_rate,
            "unique_consumers": self._unique_consumers,
            "repeat_consumer_rate": self._repeat_consumer_rate,
            "first_time_resolution_rate": self._first_time_resolution_rate,
            "mom_growth": self._mom_growth,
            "qoq_growth": self._qoq_growth,
            "yoy_growth": self._yoy_growth,
        }

        date_dependent_dispatch: Dict[str, Callable[[pd.DataFrame, Optional[pd.Timestamp]], KPIResult]] = {
            "avg_pending_age": self._avg_pending_age,
            "median_pending_age": self._median_pending_age,
            "p95_pending_age": self._p95_pending_age,
        }

        for name, fn in scalar_dispatch.items():
            ok, missing = self._eligibility.check(name)
            if not ok:
                results[name] = self._eligibility._ineligible_result(name, missing)
                continue
            try:
                results[name] = fn(df)
            except Exception as exc:
                results[name] = self._safe_fallback_result(name, str(exc))

        for name, fn in date_dependent_dispatch.items():
            ok, missing = self._eligibility.check(name)
            if not ok:
                results[name] = self._eligibility._ineligible_result(name, missing)
                continue
            try:
                results[name] = fn(df, reference_date)
            except Exception as exc:
                results[name] = self._safe_fallback_result(name, str(exc))

        return results

    def _generate_empty_kpis(self) -> Dict[str, KPIResult]:
        names = [
            "total_cases", "closed_cases", "pending_cases", "reopened_cases",
            "closure_rate", "pending_rate", "reopen_rate", "backlog",
            "avg_resolution_time", "median_resolution_time", "p95_resolution_time",
            "avg_pending_age", "median_pending_age", "p95_pending_age",
            "sla_compliance_rate", "sla_breach_rate",
            "unique_consumers", "repeat_consumer_rate", "first_time_resolution_rate",
            "mom_growth", "qoq_growth", "yoy_growth",
        ]
        results: Dict[str, KPIResult] = {}
        for name in names:
            required = _KPI_REQUIRED_ROLES.get(name, [])
            results[name] = KPIResult(
                name=name,
                value=None,
                formatted_value="N/A",
                unit="",
                definition="",
                formula="",
                interpretation=_EMPTY_DATASET_MESSAGE,
                recommendation="Upload or select a non-empty dataset to compute this KPI.",
                benchmark_status="na",
                trend_direction="none",
                trend_is_positive=None,
                previous_value=None,
                pct_change=None,
                required_roles=required,
                missing_roles=[],
                is_eligible=True,
                ineligibility_reason="",
            )
        return results

    @staticmethod
    def _safe_fallback_result(name: str, error_text: str) -> KPIResult:
        required = _KPI_REQUIRED_ROLES.get(name, [])
        return KPIResult(
            name=name,
            value=None,
            formatted_value="N/A",
            unit="",
            definition="",
            formula="",
            interpretation=f"This metric could not be computed due to an internal calculation issue: {error_text}",
            recommendation="Review the underlying data for this metric's required columns for malformed "
                           "or unexpected values.",
            benchmark_status="na",
            trend_direction="none",
            trend_is_positive=None,
            previous_value=None,
            pct_change=None,
            required_roles=required,
            missing_roles=[],
            is_eligible=True,
            ineligibility_reason="",
        )

    def _total_cases(self, df: pd.DataFrame) -> KPIResult:
        id_col = self._registry.resolve(ROLE_RECORD_ID)
        total = int(df[id_col].notna().sum()) if id_col else len(df)
        return KPIResult(
            name="total_cases",
            value=total,
            formatted_value=f"{total:,}",
            unit="cases",
            definition="Total number of complaint records in the active dataset.",
            formula="COUNT(record_id)",
            interpretation=f"The dataset contains {total:,} total complaint records.",
            recommendation="Use filters to narrow scope; review data quality if count appears anomalous.",
            benchmark_status="na",
            trend_direction="none",
            trend_is_positive=None,
            previous_value=None,
            pct_change=None,
            required_roles=[ROLE_RECORD_ID],
            missing_roles=[],
            is_eligible=True,
        )

    def _closed_cases(self, df: pd.DataFrame) -> KPIResult:
        # Milestone 17 / Task 2 — defensive: was `_resolve_status_mask(df,
        # self._registry, _STATUS_CLOSED)` directly. Now routes through the
        # flexible resolver so this KPI (a Group B flexible-eligibility
        # KPI) can compute correctly when only Closing Date is mapped.
        # Byte-identical to the original when Status IS mapped, since the
        # resolver's Priority 1 branch calls the exact same
        # `_resolve_status_mask` function unchanged.
        mask, _pending_mask, _method = self._resolve_flexible_closed_pending_masks(df)
        value = int(mask.sum())
        total = len(df)
        pct = round(value / total * 100, 1) if total else 0.0
        return KPIResult(
            name="closed_cases",
            value=value,
            formatted_value=f"{value:,}",
            unit="cases",
            definition="Count of complaints with status mapped to CLOSED.",
            formula="COUNT(record_id WHERE status = 'CLOSED')",
            interpretation=f"{value:,} complaints ({pct:.1f}% of total) have been resolved and closed.",
            recommendation=(
                "Closed count is healthy." if pct >= _CLOSURE_RATE_GOOD
                else "Accelerate resolution workflows to increase closure count."
            ),
            benchmark_status="na",
            trend_direction="none",
            trend_is_positive=True,
            previous_value=None,
            pct_change=None,
            required_roles=[ROLE_RECORD_ID, ROLE_STATUS],
            missing_roles=[],
            is_eligible=True,
        )

    def _pending_cases(self, df: pd.DataFrame) -> KPIResult:
        mask = _resolve_status_mask(df, self._registry, _STATUS_PENDING)
        value = int(mask.sum())
        total = len(df)
        pct = round(value / total * 100, 1) if total else 0.0
        return KPIResult(
            name="pending_cases",
            value=value,
            formatted_value=f"{value:,}",
            unit="cases",
            definition="Count of complaints with status mapped to PENDING (open, unresolved).",
            formula="COUNT(record_id WHERE status = 'PENDING')",
            interpretation=f"{value:,} complaints ({pct:.1f}% of total) remain unresolved.",
            recommendation=(
                "Pending load is manageable." if pct < 40
                else "High pending load detected. Prioritise backlog clearance by officer and category."
            ),
            benchmark_status="critical" if pct >= 60 else "warning" if pct >= 40 else "good",
            trend_direction="none",
            trend_is_positive=False,
            previous_value=None,
            pct_change=None,
            required_roles=[ROLE_RECORD_ID, ROLE_STATUS],
            missing_roles=[],
            is_eligible=True,
        )

    def _reopened_cases(self, df: pd.DataFrame) -> KPIResult:
        mask = _resolve_reopen_mask(df, self._registry)
        value = int(mask.sum())
        total = len(df)
        pct = round(value / total * 100, 1) if total else 0.0
        return KPIResult(
            name="reopened_cases",
            value=value,
            formatted_value=f"{value:,}",
            unit="cases",
            definition="Count of complaints that were reopened after an initial closure.",
            formula="COUNT(record_id WHERE status = 'REOPENED' OR reopen_flag = True)",
            interpretation=f"{value:,} complaints ({pct:.1f}% of total) were reopened, "
                           f"indicating unresolved first-closure quality.",
            recommendation=(
                "Reopen incidence is within acceptable range." if pct < _REOPEN_RATE_GOOD
                else "Elevated reopen count signals resolution quality issues. "
                     "Investigate root causes by category and officer."
            ),
            benchmark_status="good" if pct < _REOPEN_RATE_EXCELLENT
            else "fair" if pct < _REOPEN_RATE_GOOD
            else "critical",
            trend_direction="none",
            trend_is_positive=False,
            previous_value=None,
            pct_change=None,
            required_roles=[ROLE_RECORD_ID, ROLE_STATUS],
            missing_roles=[],
            is_eligible=True,
        )

    def _closure_rate(self, df: pd.DataFrame) -> KPIResult:
        total = len(df)
        if total == 0:
            value = 0.0
        else:
            closed = int(_resolve_status_mask(df, self._registry, _STATUS_CLOSED).sum())
            value = round(closed / total * 100, 2)
        benchmark = _benchmark_rate(value, _CLOSURE_RATE_EXCELLENT, _CLOSURE_RATE_GOOD, _CLOSURE_RATE_FAIR)
        interp_map = {
            "excellent": f"Closure rate of {value:.1f}% exceeds the 80% benchmark — operations are highly effective.",
            "good": f"Closure rate of {value:.1f}% meets acceptable performance standards.",
            "fair": f"Closure rate of {value:.1f}% is below target. Operational review recommended.",
            "critical": f"Closure rate of {value:.1f}% is critically low. Immediate management intervention required.",
        }
        rec_map = {
            "excellent": "Maintain current operational tempo. Share best practices across divisions.",
            "good": "Target incremental improvement to reach 80%+ excellence threshold.",
            "fair": "Analyse pending cases by officer and category. Set monthly closure targets.",
            "critical": "Escalate to senior management. Conduct field audit and staffing review.",
        }
        return KPIResult(
            name="closure_rate",
            value=value,
            formatted_value=f"{value:.1f}%",
            unit="%",
            definition="Percentage of total complaints resolved and closed.",
            formula="(Closed Cases / Total Cases) × 100",
            interpretation=interp_map[benchmark],
            recommendation=rec_map[benchmark],
            benchmark_status=benchmark,
            trend_direction="none",
            trend_is_positive=True,
            previous_value=None,
            pct_change=None,
            required_roles=[ROLE_RECORD_ID, ROLE_STATUS],
            missing_roles=[],
            is_eligible=True,
        )

    def _pending_rate(self, df: pd.DataFrame) -> KPIResult:
        total = len(df)
        pending = int(_resolve_status_mask(df, self._registry, _STATUS_PENDING).sum()) if total else 0
        value = round(pending / total * 100, 2) if total else 0.0
        benchmark = _benchmark_rate(value, 20, 40, 60, higher_is_better=False)
        return KPIResult(
            name="pending_rate",
            value=value,
            formatted_value=f"{value:.1f}%",
            unit="%",
            definition="Percentage of total complaints still awaiting resolution.",
            formula="(Pending Cases / Total Cases) × 100",
            interpretation=f"{value:.1f}% of complaints remain unresolved.",
            recommendation=(
                "Pending rate is under control." if value < 40
                else "Focus on clearing the pending backlog through officer workload optimisation."
            ),
            benchmark_status=benchmark,
            trend_direction="none",
            trend_is_positive=False,
            previous_value=None,
            pct_change=None,
            required_roles=[ROLE_RECORD_ID, ROLE_STATUS],
            missing_roles=[],
            is_eligible=True,
        )

    def _reopen_rate(self, df: pd.DataFrame) -> KPIResult:
        closed = int(_resolve_status_mask(df, self._registry, _STATUS_CLOSED).sum())
        reopened = int(_resolve_reopen_mask(df, self._registry).sum())
        if closed == 0:
            value = 0.0
        else:
            value = round(reopened / closed * 100, 2)
        benchmark = _benchmark_rate(value, _REOPEN_RATE_EXCELLENT, _REOPEN_RATE_GOOD, _REOPEN_RATE_WARNING, higher_is_better=False)
        return KPIResult(
            name="reopen_rate",
            value=value,
            formatted_value=f"{value:.1f}%",
            unit="%",
            definition="Percentage of closed complaints that were subsequently reopened.",
            formula="(Reopened Cases / Closed Cases) × 100",
            interpretation=(
                f"Reopen rate of {value:.1f}% is within acceptable limits." if value < _REOPEN_RATE_GOOD
                else f"Reopen rate of {value:.1f}% indicates {reopened} closures were inadequate."
            ),
            recommendation=(
                "First-closure quality is good." if value < _REOPEN_RATE_GOOD
                else "Investigate officers and categories with highest reopen incidence. "
                     "Implement resolution quality checks before closure."
            ),
            benchmark_status=benchmark,
            trend_direction="none",
            trend_is_positive=False,
            previous_value=None,
            pct_change=None,
            required_roles=[ROLE_RECORD_ID, ROLE_STATUS],
            missing_roles=[],
            is_eligible=True,
        )

    def _backlog(self, df: pd.DataFrame) -> KPIResult:
        pending = int(_resolve_status_mask(df, self._registry, _STATUS_PENDING).sum())
        return KPIResult(
            name="backlog",
            value=pending,
            formatted_value=f"{pending:,}",
            unit="cases",
            definition="Total count of unresolved complaints constituting the current operational backlog.",
            formula="COUNT(record_id WHERE status = 'PENDING')",
            interpretation=f"Current backlog stands at {pending:,} unresolved complaint(s).",
            recommendation=(
                "Backlog is within manageable levels." if pending < 100
                else f"Backlog of {pending:,} cases requires structured clearance planning."
            ),
            benchmark_status="good" if pending < 100 else "warning" if pending < 500 else "critical",
            trend_direction="none",
            trend_is_positive=False,
            previous_value=None,
            pct_change=None,
            required_roles=[ROLE_RECORD_ID, ROLE_STATUS],
            missing_roles=[],
            is_eligible=True,
        )

    def _resolution_days_for_closed(self, df: pd.DataFrame) -> Optional[pd.Series]:
        # Milestone 17 / Task 2 — defensive: routes through the flexible
        # resolver (Status-mapped path is byte-identical to the original
        # `_resolve_status_mask(df, self._registry, _STATUS_CLOSED)` call).
        mask, _pending_mask, _method = self._resolve_flexible_closed_pending_masks(df)
        closed_df = df[mask]
        if closed_df.empty:
            return None
        days = _compute_duration_days(closed_df, ROLE_REGISTRATION_DATE, ROLE_CLOSING_DATE, self._registry)
        if days is None:
            return None
        return days.dropna()

    def _resolution_hours_clipped_for_closed(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Single authoritative MTTR duration series: hours, closed-only,
        95th-percentile clipped. This is the ONLY resolution-latency helper
        that applies outlier clipping, and it now backs both compute_all()
        and _avg_resolution_time() identically, eliminating the prior
        divergence between the two code paths.

        Milestone 17 / Task 2: the closed-row filter now routes through
        `_resolve_flexible_closed_pending_masks` (Status-mapped path is
        byte-identical to the pre-refactor direct `_resolve_status_mask`
        call). The datetime-delta arithmetic itself is completely
        UNCHANGED pandas code — per the Task 3 Failure Policy, this
        computation is deliberately never offloaded to DuckDB.
        """
        mask, _pending_mask, _method = self._resolve_flexible_closed_pending_masks(df)
        closed_df = df[mask]
        if closed_df.empty:
            return None
        reg_col = self._registry.resolve(ROLE_REGISTRATION_DATE)
        close_col = self._registry.resolve(ROLE_CLOSING_DATE)
        if not reg_col or not close_col or reg_col not in closed_df.columns or close_col not in closed_df.columns:
            return None
        try:
            t_reg = _safe_tz_naive(closed_df[reg_col])
            t_cls = _safe_tz_naive(closed_df[close_col])
            durations_hrs = (t_cls - t_reg).dt.total_seconds() / 3600.0
            durations_hrs = durations_hrs.replace([np.inf, -np.inf], np.nan)
            valid = durations_hrs[(durations_hrs >= 0) & durations_hrs.notna()]
            if valid.empty:
                return None
            upper_bound = float(valid.quantile(0.95))
            return valid.clip(upper=upper_bound)
        except Exception:
            return None

    def _avg_resolution_time(self, df: pd.DataFrame) -> KPIResult:
        clipped_hours = self._resolution_hours_clipped_for_closed(df)
        value = round(float(clipped_hours.mean()), 2) if clipped_hours is not None and len(clipped_hours) else None
        formatted = f"{value:.1f} hrs" if value is not None else "N/A"
        benchmark = (
            _benchmark_rate(
                value, _AVG_RESOLUTION_EXCELLENT_HRS, _AVG_RESOLUTION_GOOD_HRS,
                _AVG_RESOLUTION_WARNING_HRS, higher_is_better=False,
            )
            if value is not None else "na"
        )
        return KPIResult(
            name="avg_resolution_time",
            value=value,
            formatted_value=formatted,
            unit="hrs",
            definition="Outlier-resilient Mean Time To Resolution (MTTR): mean hours between "
                       "registration and closure for closed complaints, with the top 5% of "
                       "durations clipped at the 95th percentile to prevent orphaned/historical "
                       "records from distorting the average.",
            formula="MEAN(CLIP(closing_date − registration_date, upper=P95)) in hours WHERE status = 'CLOSED'",
            interpretation=(
                f"Outlier-resilient MTTR is {value:.1f} hours." if value is not None
                else "Insufficient closed records with both dates to compute."
            ),
            recommendation=(
                "Resolution speed is within target." if value is not None and value <= _AVG_RESOLUTION_GOOD_HRS
                else "Resolution time exceeds target. Review field response workflows and officer workload."
            ),
            benchmark_status=benchmark,
            trend_direction="none",
            trend_is_positive=False,
            previous_value=None,
            pct_change=None,
            required_roles=[ROLE_REGISTRATION_DATE, ROLE_CLOSING_DATE, ROLE_STATUS],
            missing_roles=[],
            is_eligible=True,
        )

    def _median_resolution_time(self, df: pd.DataFrame) -> KPIResult:
        days_series = self._resolution_days_for_closed(df)
        value = round(float(days_series.median()), 2) if days_series is not None and len(days_series) else None
        formatted = f"{value:.1f} days" if value is not None else "N/A"
        return KPIResult(
            name="median_resolution_time",
            value=value,
            formatted_value=formatted,
            unit="days",
            definition="Median calendar days to resolve a complaint — robust to outlier resolution times.",
            formula="MEDIAN(closing_date − registration_date) WHERE status = 'CLOSED'",
            interpretation=(
                f"50% of complaints are resolved within {value:.1f} days." if value is not None
                else "Insufficient data."
            ),
            recommendation="Compare median vs average — large divergence signals outlier long-running cases.",
            benchmark_status="na",
            trend_direction="none",
            trend_is_positive=False,
            previous_value=None,
            pct_change=None,
            required_roles=[ROLE_REGISTRATION_DATE, ROLE_CLOSING_DATE, ROLE_STATUS],
            missing_roles=[],
            is_eligible=True,
        )

    def _p95_resolution_time(self, df: pd.DataFrame) -> KPIResult:
        days_series = self._resolution_days_for_closed(df)
        value = round(float(np.percentile(days_series, 95)), 2) if days_series is not None and len(days_series) else None
        formatted = f"{value:.1f} days" if value is not None else "N/A"
        return KPIResult(
            name="p95_resolution_time",
            value=value,
            formatted_value=formatted,
            unit="days",
            definition="95th-percentile resolution time — only 5% of complaints take longer than this.",
            formula="PERCENTILE_95(closing_date − registration_date) WHERE status = 'CLOSED'",
            interpretation=(
                f"95% of complaints are resolved within {value:.1f} days." if value is not None
                else "Insufficient data."
            ),
            recommendation="High P95 signals a long tail of chronically delayed cases requiring case-level review.",
            benchmark_status="na",
            trend_direction="none",
            trend_is_positive=False,
            previous_value=None,
            pct_change=None,
            required_roles=[ROLE_REGISTRATION_DATE, ROLE_CLOSING_DATE, ROLE_STATUS],
            missing_roles=[],
            is_eligible=True,
        )

    def _pending_age_series(self, df: pd.DataFrame, reference_date: Optional[pd.Timestamp]) -> Optional[pd.Series]:
        # Refactor Phase 2B / Reference-Date Normalization: defensive
        # re-normalization at this internal entry point as well, so this
        # method remains correct even if invoked outside the compute_all()
        # dispatch path (which already normalizes reference_date upstream).
        reference_date = _normalize_reference_date(reference_date)
        mask = _resolve_status_mask(df, self._registry, _STATUS_PENDING)
        pending_df = df[mask]
        if pending_df.empty:
            return None
        reg_col = self._registry.resolve(ROLE_REGISTRATION_DATE)
        if not reg_col or reg_col not in pending_df.columns:
            return None
        try:
            # Refactor Phase 2B / Datetime & Lifecycle Fix: both the
            # pending-slice registration dates and the full-column fallback
            # used to derive a default reference_date are now routed
            # through _safe_tz_naive, eliminating a tz-mismatch TypeError
            # against a tz-aware Parquet-origin registration_date column.
            reg_series = _safe_tz_naive(pending_df[reg_col])
            if reference_date is None:
                all_reg = _safe_tz_naive(df[reg_col])
                reference_date = all_reg.dropna().max()
            else:
                reference_date = _normalize_reference_date(reference_date)
            if pd.isna(reference_date):
                return None
            age = (reference_date - reg_series).dt.total_seconds() / 86400
            return age.where(age >= 0).dropna()
        except Exception:
            return None
    # AFTER
    def _avg_pending_age(self, df: pd.DataFrame, reference_date: Optional[pd.Timestamp]) -> KPIResult:
        # Refactor Phase 2B / Reference-Date Normalization: defensive
        # re-normalization at this public analytical entry point.
        reference_date = _normalize_reference_date(reference_date)
        age_series = self._sanitize_and_bound_pending_ages(df, reference_date)
        value = round(float(age_series.mean()), 2) if age_series is not None and len(age_series) else None
        formatted = f"{value:.1f} days" if value is not None else "N/A"
        benchmark = _benchmark_rate(value, _AVG_PENDING_AGE_EXCELLENT, _AVG_PENDING_AGE_GOOD, _AVG_PENDING_AGE_WARNING, higher_is_better=False) if value is not None else "na"
        return KPIResult(
            name="avg_pending_age",
            value=value,
            formatted_value=formatted,
            unit="days",
            definition="Anomalous-resilient mean age of currently pending complaints.",
            formula="MEAN(SANIDATE(reference_date − registration_date)) WHERE status = 'PENDING'",
            
            interpretation=(
                f"Pending complaints are on average {value:.1f} days old." if value is not None
                else "Insufficient data."
            ),
            recommendation=(
                "Pending age is within acceptable range." if value and value <= _AVG_PENDING_AGE_GOOD
                else "High average pending age indicates chronic backlog. "
                     "Sort by age and dispatch oldest cases immediately."
            ),
            benchmark_status=benchmark,
            trend_direction="none",
            trend_is_positive=False,
            previous_value=None,
            pct_change=None,
            required_roles=[ROLE_REGISTRATION_DATE, ROLE_STATUS],
            missing_roles=[],
            is_eligible=True,
        )

    def _median_pending_age(self, df: pd.DataFrame, reference_date: Optional[pd.Timestamp]) -> KPIResult:
        age_series = self._pending_age_series(df, reference_date)
        value = round(float(age_series.median()), 2) if age_series is not None and len(age_series) else None
        return KPIResult(
            name="median_pending_age",
            value=value,
            formatted_value=f"{value:.1f} days" if value is not None else "N/A",
            unit="days",
            definition="Median age of pending complaints — less sensitive to extreme outliers than mean.",
            formula="MEDIAN(reference_date − registration_date) WHERE status = 'PENDING'",
            interpretation=(
                f"Half of pending complaints are older than {value:.1f} days." if value is not None
                else "Insufficient data."
            ),
            recommendation="Cross-compare with average; divergence reveals a skewed age distribution.",
            benchmark_status="na",
            trend_direction="none",
            trend_is_positive=False,
            previous_value=None,
            pct_change=None,
            required_roles=[ROLE_REGISTRATION_DATE, ROLE_STATUS],
            missing_roles=[],
            is_eligible=True,
        )

    def _p95_pending_age(self, df: pd.DataFrame, reference_date: Optional[pd.Timestamp]) -> KPIResult:
        age_series = self._pending_age_series(df, reference_date)
        value = round(float(np.percentile(age_series, 95)), 2) if age_series is not None and len(age_series) else None
        return KPIResult(
            name="p95_pending_age",
            value=value,
            formatted_value=f"{value:.1f} days" if value is not None else "N/A",
            unit="days",
            definition="95th-percentile pending complaint age — represents the worst-performing 5% of the backlog.",
            formula="PERCENTILE_95(reference_date − registration_date) WHERE status = 'PENDING'",
            interpretation=(
                f"5% of pending complaints are older than {value:.1f} days and represent the critical backlog tail."
                if value is not None else "Insufficient data."
            ),
            recommendation="These are the highest-priority cases. Assign senior officers and set hard closure deadlines.",
            benchmark_status="na",
            trend_direction="none",
            trend_is_positive=False,
            previous_value=None,
            pct_change=None,
            required_roles=[ROLE_REGISTRATION_DATE, ROLE_STATUS],
            missing_roles=[],
            is_eligible=True,
        )
    

    # REFINED helper — engine/analytics.py, ComplaintKPIEngine
    def _sanitize_and_bound_pending_ages(
       self, df: pd.DataFrame, reference_date: Optional[pd.Timestamp]
    ) -> Optional[pd.Series]:
        """
         Refined replacement for 95% clip.
         1. Eliminates systemic data corruptions (e.g., epoch zero / default 1900-01-01 dates).
         2. Uses an IQR-based Tukey's fence to clamp true statistical outliers, ensuring 
         legitimate operational backlogs are preserved while safeguarding the mean 
         against catastrophic data spikes.
        """
        reference_date = _normalize_reference_date(reference_date)
        age_series = self._pending_age_series(df, reference_date)
        if age_series is None or age_series.empty:
            return None
        return self._apply_adaptive_pending_age_fence(age_series)
        
    # NEW — single shared static helper, engine/analytics.py, ComplaintKPIEngine
    @staticmethod
    def _apply_adaptive_pending_age_fence(age_series: pd.Series) -> pd.Series:
        """Single source of truth for the adaptive Tukey's Fence winsorization
        applied to pending-age series across both the scalar avg_pending_age
        KPI and the Executive Narrative's Business Risk Score. Consolidates
        what were previously two independently-maintained copies of the same
        math into one call site, eliminating the risk of the two figures ever
        silently drifting apart after a future threshold tuning change.

        Step 1 — hard ceiling clip at _PENDING_AGE_HARD_CEILING_DAYS neutralizes
        legacy system-default dates (e.g. 1900-01-01 epoch placeholders) before
        they can influence the IQR computation itself.
        Step 2 — Tukey's Fence (Q3 + iqr_multiplier × IQR) adaptively bounds
        genuine statistical outliers without flattening legitimate long-tail
        operational backlogs.
        Step 3 — a _PENDING_AGE_ADAPTIVE_FLOOR_DAYS floor guards against a
        tightly-clustered, near-zero-IQR backlog collapsing the fence below a
        sane one-year threshold.

        Never raises: degrades to the ceiling-clipped (but not IQR-fenced)
        series on any computation failure, and to the fully unclipped series if
        even the ceiling clip cannot be applied.
        """
        try:
            bounded = age_series.clip(lower=0.0, upper=_PENDING_AGE_HARD_CEILING_DAYS)
            q25 = float(bounded.quantile(0.25))
            q75 = float(bounded.quantile(0.75))
            iqr = q75 - q25
            adaptive_bound = max(q75 + _PENDING_AGE_IQR_MULTIPLIER * iqr, _PENDING_AGE_ADAPTIVE_FLOOR_DAYS)
            return bounded.clip(upper=adaptive_bound)
        except Exception:
            try:
                return age_series.clip(lower=0.0, upper=_PENDING_AGE_HARD_CEILING_DAYS)
            except Exception:
                return age_series    
  

    def _sla_compliance_rate(self, df: pd.DataFrame) -> KPIResult:
        # Milestone 17 / Task 2 — defensive: closed-row filter now routes
        # through the flexible resolver (byte-identical to the original
        # direct `_resolve_status_mask` call whenever Status IS mapped).
        closed_mask, _pending_mask, _method = self._resolve_flexible_closed_pending_masks(df)
        closed_df = df[closed_mask]
        if closed_df.empty:
            return self._sla_na_result("sla_compliance_rate", "No closed records found.")
        sla_col = self._registry.resolve(ROLE_SLA_DEADLINE)
        close_col = self._registry.resolve(ROLE_CLOSING_DATE)
        if not sla_col or not close_col:
            return self._sla_na_result("sla_compliance_rate", "SLA deadline or closing date role not mapped.")
        try:
            # Refactor Phase 2B / Datetime & Lifecycle Fix: both SLA
            # deadline and closing date are now routed through
            # _safe_tz_naive before comparison, eliminating a tz-mismatch
            # TypeError when either Parquet-origin column carries tz
            # metadata that the other does not.
            sla_dt = _safe_tz_naive(closed_df[sla_col])
            close_dt = _safe_tz_naive(closed_df[close_col])
            valid_mask = sla_dt.notna() & close_dt.notna()
            valid_count = int(valid_mask.sum())
            if valid_count == 0:
                return self._sla_na_result("sla_compliance_rate", "No records with both SLA and closing dates.")
            compliant = int((close_dt[valid_mask] <= sla_dt[valid_mask]).sum())
            value = round(compliant / valid_count * 100, 2)
            benchmark = _benchmark_rate(value, _SLA_COMPLIANCE_EXCELLENT, _SLA_COMPLIANCE_GOOD, _SLA_COMPLIANCE_WARNING)
        except Exception as exc:
            return self._sla_na_result("sla_compliance_rate", str(exc))
        return KPIResult(
            name="sla_compliance_rate",
            value=value,
            formatted_value=f"{value:.1f}%",
            unit="%",
            definition="Percentage of closed complaints resolved on or before their SLA deadline.",
            formula="(Closed cases where closing_date ≤ sla_deadline / Total closed with valid SLA) × 100",
            interpretation=f"{value:.1f}% of closed complaints met their SLA commitment ({compliant}/{valid_count}).",
            recommendation=(
                "SLA performance meets target." if value >= _SLA_COMPLIANCE_GOOD
                else "SLA compliance is below target. Review assignment delays and field response times."
            ),
            benchmark_status=benchmark,
            trend_direction="none",
            trend_is_positive=True,
            previous_value=None,
            pct_change=None,
            required_roles=[ROLE_STATUS, ROLE_CLOSING_DATE, ROLE_SLA_DEADLINE],
            missing_roles=[],
            is_eligible=True,
        )

    def _sla_breach_rate(self, df: pd.DataFrame) -> KPIResult:
        compliance = self._sla_compliance_rate(df)
        if compliance.value is None:
            return self._sla_na_result("sla_breach_rate", compliance.interpretation)
        value = round(100.0 - float(compliance.value), 2)
        benchmark = _benchmark_rate(value, 10, 25, 50, higher_is_better=False)
        return KPIResult(
            name="sla_breach_rate",
            value=value,
            formatted_value=f"{value:.1f}%",
            unit="%",
            definition="Percentage of closed complaints that exceeded their SLA deadline.",
            formula="100% − SLA Compliance Rate",
            interpretation=f"{value:.1f}% of complaints breached their SLA.",
            recommendation=(
                "SLA breach rate is acceptable." if value < 25
                else "High SLA breach rate. Implement early warning system for approaching deadlines."
            ),
            benchmark_status=benchmark,
            trend_direction="none",
            trend_is_positive=False,
            previous_value=None,
            pct_change=None,
            required_roles=[ROLE_STATUS, ROLE_CLOSING_DATE, ROLE_SLA_DEADLINE],
            missing_roles=[],
            is_eligible=True,
        )

    @staticmethod
    def _sla_na_result(name: str, reason: str) -> KPIResult:
        return KPIResult(
            name=name,
            value=None,
            formatted_value="N/A",
            unit="%",
            definition="",
            formula="",
            interpretation=reason,
            recommendation="Ensure SLA deadline and closing date columns are mapped.",
            benchmark_status="na",
            trend_direction="none",
            trend_is_positive=None,
            previous_value=None,
            pct_change=None,
            required_roles=[ROLE_STATUS, ROLE_CLOSING_DATE, ROLE_SLA_DEADLINE],
            missing_roles=[],
            is_eligible=True,
        )

    def _unique_consumers(self, df: pd.DataFrame) -> KPIResult:
        consumer_col = self._registry.resolve(ROLE_CONSUMER_ID)
        value = int(df[consumer_col].nunique(dropna=True)) if consumer_col and consumer_col in df.columns else 0
        return KPIResult(
            name="unique_consumers",
            value=value,
            formatted_value=f"{value:,}",
            unit="consumers",
            definition="Count of distinct consumer accounts that raised at least one complaint.",
            formula="COUNT(DISTINCT consumer_id)",
            interpretation=f"{value:,} unique consumers appear in the dataset.",
            recommendation="High consumer count with low complaint density suggests broad but manageable impact.",
            benchmark_status="na",
            trend_direction="none",
            trend_is_positive=None,
            previous_value=None,
            pct_change=None,
            required_roles=[ROLE_CONSUMER_ID],
            missing_roles=[],
            is_eligible=True,
        )

    def _repeat_consumer_rate(self, df: pd.DataFrame) -> KPIResult:
        consumer_col = self._registry.resolve(ROLE_CONSUMER_ID)
        id_col = self._registry.resolve(ROLE_RECORD_ID)
        if not consumer_col or consumer_col not in df.columns:
            value = 0.0
        else:
            counts = df.groupby(consumer_col)[id_col].count() if id_col else df[consumer_col].value_counts()
            total_consumers = len(counts)
            repeat_consumers = int((counts > 1).sum())
            value = round(repeat_consumers / total_consumers * 100, 2) if total_consumers else 0.0
        benchmark = _benchmark_rate(value, 95, 85, 70, higher_is_better=False)
        return KPIResult(
            name="repeat_consumer_rate",
            value=value,
            formatted_value=f"{value:.1f}%",
            unit="%",
            definition="Percentage of consumers who have filed more than one complaint.",
            formula="(Consumers with > 1 complaint / Total unique consumers) × 100",
            interpretation=(
                f"{value:.1f}% of consumers are repeat complainants, indicating recurring service failures."
                if value > 10
                else f"{value:.1f}% repeat rate suggests most issues are one-time occurrences."
            ),
            recommendation=(
                "Repeat rate is low — service quality appears consistent."
                if value < 10
                else "High repeat rate signals systemic infrastructure issues. Segment by zone and category."
            ),
            benchmark_status=benchmark,
            trend_direction="none",
            trend_is_positive=False,
            previous_value=None,
            pct_change=None,
            required_roles=[ROLE_CONSUMER_ID, ROLE_RECORD_ID],
            missing_roles=[],
            is_eligible=True,
        )

    def _first_time_resolution_rate(self, df: pd.DataFrame) -> KPIResult:
        # Milestone 17 / Task 2 — defensive: total-closed count now routes
        # through the flexible resolver (byte-identical to the original
        # direct `_resolve_status_mask` call whenever Status IS mapped).
        closed_mask, _pending_mask, _method = self._resolve_flexible_closed_pending_masks(df)
        total_closed = int(closed_mask.sum())
        if total_closed == 0:
            return KPIResult(
                name="first_time_resolution_rate",
                value=None,
                formatted_value="N/A",
                unit="%",
                definition="",
                formula="",
                interpretation="No closed records to evaluate.",
                recommendation="",
                benchmark_status="na",
                trend_direction="none",
                trend_is_positive=True,
                previous_value=None,
                pct_change=None,
                required_roles=[ROLE_RECORD_ID, ROLE_STATUS],
                missing_roles=[],
                is_eligible=True,
            )
        reopen_mask = _resolve_reopen_mask(df, self._registry)
        closed_then_reopened = int((closed_mask & reopen_mask).sum())
        true_first_time = max(0, total_closed - closed_then_reopened)
        value = round(true_first_time / total_closed * 100, 2)
        benchmark = _benchmark_rate(value, 90, 75, 60)
        return KPIResult(
            name="first_time_resolution_rate",
            value=value,
            formatted_value=f"{value:.1f}%",
            unit="%",
            definition="Percentage of closed complaints resolved permanently on the first attempt (not subsequently reopened).",
            formula="(Closed cases not reopened / Total closed cases) × 100",
            interpretation=f"{value:.1f}% of complaints were resolved correctly on first closure.",
            recommendation=(
                "First-time resolution quality is strong."
                if value >= 75
                else "Low first-time resolution rate suggests inadequate root-cause resolution. "
                     "Review field engineer training and closure checklists."
            ),
            benchmark_status=benchmark,
            trend_direction="none",
            trend_is_positive=True,
            previous_value=None,
            pct_change=None,
            required_roles=[ROLE_RECORD_ID, ROLE_STATUS],
            missing_roles=[],
            is_eligible=True,
        )

    def _mom_growth(self, df: pd.DataFrame) -> KPIResult:
        reg_col = self._registry.resolve(ROLE_REGISTRATION_DATE)
        reg_series = df[reg_col] if reg_col and reg_col in df.columns else None
        current, previous, pct = _period_comparison_growth(reg_series, "M") if reg_series is not None else (None, None, None)
        direction = _trend_direction(pct)
        return KPIResult(
            name="mom_growth",
            value=pct,
            formatted_value=f"{pct:+.1f}%" if pct is not None else "N/A",
            unit="%",
            definition="Month-over-Month change in total complaint volume.",
            formula="(Current Month Complaints − Previous Month Complaints) / Previous Month Complaints × 100",
            interpretation=(
                f"Complaint volume changed by {pct:+.1f}% vs the prior month "
                f"({int(previous):,} → {int(current):,})." if pct is not None
                else "Insufficient monthly data for MoM comparison."
            ),
            recommendation=(
                "Volume decline — positive trend; verify it reflects genuine service improvement."
                if pct is not None and pct < 0
                else "Volume increase — investigate whether driven by seasonal factors or infrastructure deterioration."
            ),
            benchmark_status="na",
            trend_direction=direction,
            trend_is_positive=False,
            previous_value=previous,
            pct_change=pct,
            required_roles=[ROLE_RECORD_ID, ROLE_REGISTRATION_DATE],
            missing_roles=[],
            is_eligible=True,
        )

    def _qoq_growth(self, df: pd.DataFrame) -> KPIResult:
        reg_col = self._registry.resolve(ROLE_REGISTRATION_DATE)
        reg_series = df[reg_col] if reg_col and reg_col in df.columns else None
        current, previous, pct = _period_comparison_growth(reg_series, "Q") if reg_series is not None else (None, None, None)
        direction = _trend_direction(pct)
        return KPIResult(
            name="qoq_growth",
            value=pct,
            formatted_value=f"{pct:+.1f}%" if pct is not None else "N/A",
            unit="%",
            definition="Quarter-over-Quarter change in total complaint volume.",
            formula="(Current Quarter Complaints − Previous Quarter Complaints) / Previous Quarter Complaints × 100",
            interpretation=(
                f"Complaint volume changed by {pct:+.1f}% quarter-over-quarter." if pct is not None
                else "Insufficient quarterly data for QoQ comparison."
            ),
            recommendation="Track QoQ alongside infrastructure upgrade cycles to assess intervention impact.",
            benchmark_status="na",
            trend_direction=direction,
            trend_is_positive=False,
            previous_value=previous,
            pct_change=pct,
            required_roles=[ROLE_RECORD_ID, ROLE_REGISTRATION_DATE],
            missing_roles=[],
            is_eligible=True,
        )

    def _yoy_growth(self, df: pd.DataFrame) -> KPIResult:
        reg_col = self._registry.resolve(ROLE_REGISTRATION_DATE)
        reg_series = df[reg_col] if reg_col and reg_col in df.columns else None
        current, previous, pct = _period_comparison_growth(reg_series, "A") if reg_series is not None else (None, None, None)
        direction = _trend_direction(pct)
        return KPIResult(
            name="yoy_growth",
            value=pct,
            formatted_value=f"{pct:+.1f}%" if pct is not None else "N/A",
            unit="%",
            definition="Year-over-Year change in total complaint volume.",
            formula="(Current Year Complaints − Previous Year Complaints) / Previous Year Complaints × 100",
            interpretation=(
                f"Annual complaint volume changed by {pct:+.1f}% year-over-year." if pct is not None
                else "Insufficient annual data (< 2 years) for YoY comparison."
            ),
            recommendation="YoY trend is the most reliable signal of structural service quality improvement.",
            benchmark_status="na",
            trend_direction=direction,
            trend_is_positive=False,
            previous_value=previous,
            pct_change=pct,
            required_roles=[ROLE_RECORD_ID, ROLE_REGISTRATION_DATE],
            missing_roles=[],
            is_eligible=True,
        )
    
    # NEW — shared helper, engine/analytics.py, ComplaintKPIEngine
    @staticmethod
    def _clip_resolution_days_at_p95(
        res_days: Optional[pd.Series], closed_mask: pd.Series
    ) -> Optional[pd.Series]:
        """Applies the same outlier-resilience philosophy as
        _resolution_hours_clipped_for_closed (95th-percentile winsorization) to
        the raw per-row resolution-duration series consumed by
        compute_officer_productivity and compute_category_breakdown, so a
        single anomalous record can no longer produce a per-officer/per-category
        average that is wildly inconsistent with the headline
        avg_resolution_time KPI shown on the Executive KPI Card. The clip
        threshold is computed once, globally, across the full closed-record
        population — not per group — so every officer/category row remains
        comparable against the same reference scale. Units (days) and the
        'Avg Resolution (days)' column name are preserved exactly; only the
        values become outlier-resilient. Never raises: returns the original,
        unclipped series on any computation failure.
        """
        if res_days is None:
            return None
        try:
            closed_vals = res_days[closed_mask.reindex(res_days.index).fillna(False)].dropna()
            if closed_vals.empty:
                return res_days
            upper_bound = float(closed_vals.quantile(0.95))
            return res_days.clip(upper=upper_bound)
        except Exception:
            return res_days

    # ══════════════════════════════════════════════════════════════════════
    # Milestone 17 / Task 3 — DuckDB Count Acceleration (Safety-Audited)
    # (Refactor Phase 2B — Parquet-native pathing addendum)
    #
    # SAFETY AUDIT SUMMARY (see turn-level Root-Cause report for full detail):
    #   - ONLY integer COUNT(*) / SUM(CASE...) aggregation is offloaded to
    #     DuckDB. These are exact-integer operations with zero
    #     floating-point precision-drift risk between pandas and DuckDB.
    #   - Resolution-time duration math (registration/closing datetime
    #     deltas), the 95th-percentile clip, and any MEAN/MEDIAN over those
    #     durations are NEVER offloaded — they remain 100% on the original,
    #     unmodified pandas implementation in
    #     _resolution_hours_clipped_for_closed / _clip_resolution_days_at_p95
    #     per the stated Failure Policy (Integrity over performance).
    #   - NULL/NaN parity: every DuckDB query below explicitly filters
    #     `WHERE <group_col> IS NOT NULL`, mirroring pandas'
    #     `groupby(..., dropna=True)` default used everywhere in this
    #     engine — this prevents the documented DuckDB-vs-pandas NULL-group
    #     divergence (see engine/duckdb_executor.py's own
    #     "NULL-key exclusion filter" comment and its accompanying test,
    #     test_null_group_key_parity, for the exact same fix pattern).
    #   - Type/casing parity: the group column is stripped of whitespace
    #     identically before BOTH the pandas and DuckDB code paths run
    #     (see the `.astype(str).str.strip()` normalization immediately
    #     before each accelerated function's groupby/query), so a group key
    #     can never fragment differently between the two engines.
    #   - Any DuckDB unavailability (package not installed, no working
    #     connection, unsafe identifier, or any query/execution exception)
    #     returns None and the caller falls through UNCONDITIONALLY to the
    #     existing, unmodified per-group pandas loop — this hook can never
    #     change behavior, only occasionally change which engine computed
    #     an already-verified-equivalent integer count.
    #   - Refactor Phase 2B: when `self._parquet_path` is set, this hook
    #     first attempts `engine.analytics_duckdb_accelerator
    #     .duckdb_officer_or_category_productivity(..., parquet_path=...)`,
    #     which executes `total` directly against the Parquet file via
    #     `read_parquet()` with zero pandas materialization of the raw
    #     rows for that computation. `closed`/`pending` on that Parquet-
    #     native path are `0` (see that function's own docstring for the
    #     documented reason: those two counts are inherently tied to the
    #     already-materialized pandas `closed_mask`/`pending_mask` Series
    #     produced by `_resolve_flexible_closed_pending_masks`, which is
    #     not currently re-expressed as portable SQL). Any failure or
    #     unavailability of the Parquet-native path falls through
    #     unconditionally to the original in-memory replacement-scan
    #     branch below, which is completely unmodified.
    # ══════════════════════════════════════════════════════════════════════

    def _duckdb_accelerated_group_counts(
        self,
        df: pd.DataFrame,
        group_col: str,
        closed_mask: pd.Series,
        pending_mask: Optional[pd.Series] = None,
    ) -> Optional[Dict[Any, Dict[str, int]]]:
        """
        Attempts a DuckDB-accelerated per-group Total/Closed[/Pending]
        integer count aggregation. Returns a dict keyed by the (already
        string-normalized) group value -> {"total": int, "closed": int,
        "pending": int} — or None on any unavailability/failure, signaling
        the caller to use its existing pandas per-group loop unchanged.
        Never raises.

        Refactor Phase 2B: no longer a bare `@staticmethod` — promoted to
        an instance method solely so it can read `self._parquet_path` and
        attempt the Parquet-native accelerator first when that path is
        configured. The original static-method call signature (positional
        df/group_col/closed_mask/pending_mask) is fully preserved for
        every existing call site.
        """
        if not _DUCKDB_ACCELERATOR_AVAILABLE:
            return None

        # ── Parquet-native branch (Refactor Phase 2B) ────────────────────
        if self._parquet_path and _PARQUET_ACCELERATOR_AVAILABLE:
            try:
                effective_pending_mask = (
                    pending_mask if pending_mask is not None
                    else pd.Series(False, index=df.index)
                )
                accel_result = _duckdb_officer_or_category_productivity(
                    df, group_col, closed_mask, effective_pending_mask,
                    parquet_path=self._parquet_path,
                )
                if accel_result is not None and not accel_result.empty and group_col in accel_result.columns:
                    return {
                        row[group_col]: {
                            "total": int(row["total"]),
                            "closed": int(row["closed"]),
                            "pending": int(row["pending"]),
                        }
                        for _, row in accel_result.iterrows()
                    }
            except Exception:  # noqa: BLE001 — Parquet-native path must never break the KPI
                pass

        # ── In-memory pandas-replacement-scan branch (original, unmodified) ──
        try:
            if not _should_use_duckdb(df):
                return None
            safe_col = _duckdb_sanitize_identifier(str(group_col))
            if safe_col is None or safe_col not in df.columns:
                return None
            connection = _duckdb_get_connection_for_thread()
            if connection is None:
                return None

            has_pending = pending_mask is not None
            accel_df = df.assign(
                __closed=closed_mask.reindex(df.index).fillna(False).values,
                __pending=(pending_mask.reindex(df.index).fillna(False).values if has_pending else False),
            )
            quoted = _duckdb_quote_identifier(safe_col)
            pending_select = (
                "SUM(CASE WHEN __pending THEN 1 ELSE 0 END) AS __pending_n"
                if has_pending else "0 AS __pending_n"
            )
            query = (
                f"SELECT {quoted} AS __key, COUNT(*) AS __total, "
                f"SUM(CASE WHEN __closed THEN 1 ELSE 0 END) AS __closed_n, "
                f"{pending_select} "
                f"FROM working_frame WHERE {quoted} IS NOT NULL GROUP BY {quoted}"
            )
            working_frame = accel_df  # noqa: F841 — DuckDB replacement-scan target
            result = connection.execute(query).fetch_df()
            if result is None or result.empty:
                return None
            return {
                row["__key"]: {
                    "total": int(row["__total"]),
                    "closed": int(row["__closed_n"]),
                    "pending": int(row["__pending_n"]),
                }
                for _, row in result.iterrows()
            }
        except Exception:  # noqa: BLE001 — acceleration must never break the KPI
            return None

    def compute_officer_productivity(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        ok, _ = self._eligibility.check("officer_productivity")
        if not ok:
            return None
        officer_col = self._registry.resolve(ROLE_OFFICER)
        if not officer_col or officer_col not in df.columns:
            return None
        try:
            # Milestone 16 / Issue 5a — string normalization fix: strip
            # whitespace on the grouping key BEFORE the groupby so
            # "J. Sharma" and "J. Sharma " (or trailing/leading spaces of
            # any kind) never fragment into separate, near-empty-looking
            # rows in the Officer Productivity deep-dive table. Applied on
            # a local working copy only — the registry-resolved column
            # name and the caller's original df are never mutated.
            df = df.copy()
            df[officer_col] = df[officer_col].astype(str).str.strip()

            # Milestone 17 / Task 2 — defensive: lifecycle masks now route
            # through the flexible resolver (byte-identical to the
            # original direct _resolve_status_mask calls whenever Status
            # IS mapped), so this Group A flexible-eligibility KPI computes
            # correctly even when only Closing Date is mapped.
            closed_mask, pending_mask, _method = self._resolve_flexible_closed_pending_masks(df)
            reopen_mask = _resolve_reopen_mask(df, self._registry)
            res_days = _compute_duration_days(df, ROLE_REGISTRATION_DATE, ROLE_CLOSING_DATE, self._registry)
            res_days = self._clip_resolution_days_at_p95(res_days, closed_mask)

            # DuckDB count acceleration hook — see the safety-audit block
            # immediately above this method for the full risk analysis.
            duckdb_counts = self._duckdb_accelerated_group_counts(
                df, officer_col, closed_mask, pending_mask
            )

            rows: List[Dict[str, Any]] = []

            for officer_val, group in df.groupby(officer_col, sort=False):
                idx = group.index
                if duckdb_counts is not None and officer_val in duckdb_counts:
                    total = duckdb_counts[officer_val]["total"]
                    closed = duckdb_counts[officer_val]["closed"]
                    pending = duckdb_counts[officer_val]["pending"]
                else:
                    total = len(group)
                    closed = int(closed_mask.reindex(idx).fillna(False).sum())
                    pending = int(pending_mask.reindex(idx).fillna(False).sum())
                reopened = int(reopen_mask.reindex(idx).fillna(False).sum())
                closure_rate = round(closed / total * 100, 1) if total else 0.0
                avg_res: Optional[float] = None
                if res_days is not None:
                    closed_days = res_days.reindex(idx)[closed_mask.reindex(idx).fillna(False)]
                    if len(closed_days.dropna()):
                        avg_res = round(float(closed_days.dropna().mean()), 1)
                rows.append({
                    "Officer": str(officer_val),
                    "Total Cases": total,
                    "Closed": closed,
                    "Pending": pending,
                    "Reopened": reopened,
                    "Closure Rate (%)": closure_rate,
                    "Avg Resolution (days)": avg_res,
                    "Pending Load": pending,
                })
            return pd.DataFrame(rows).sort_values("Closure Rate (%)", ascending=False).reset_index(drop=True)
        except Exception:
            return None

    def compute_category_breakdown(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        ok, _ = self._eligibility.check("category_breakdown")
        if not ok:
            return None
        cat_col = self._registry.resolve(ROLE_CATEGORY)
        if not cat_col or cat_col not in df.columns:
            return None
        try:
            # Milestone 16 / Issue 5a — string normalization fix (see the
            # identical comment in compute_officer_productivity above for
            # full rationale). Local working copy only.
            df = df.copy()
            df[cat_col] = df[cat_col].astype(str).str.strip()

            # Milestone 17 / Task 2 — defensive flexible lifecycle masks.
            closed_mask, _pending_mask, _method = self._resolve_flexible_closed_pending_masks(df)
            res_days = _compute_duration_days(df, ROLE_REGISTRATION_DATE, ROLE_CLOSING_DATE, self._registry)
            res_days = self._clip_resolution_days_at_p95(res_days, closed_mask)

            # DuckDB count acceleration hook (Total/Closed only — Category
            # Breakdown does not track a Pending column). See the
            # safety-audit block above compute_officer_productivity.
            duckdb_counts = self._duckdb_accelerated_group_counts(df, cat_col, closed_mask, None)

            rows: List[Dict[str, Any]] = []
            for cat_val, group in df.groupby(cat_col, sort=False):
                idx = group.index
                if duckdb_counts is not None and cat_val in duckdb_counts:
                    total = duckdb_counts[cat_val]["total"]
                    closed = duckdb_counts[cat_val]["closed"]
                else:
                    total = len(group)
                    closed = int(closed_mask.reindex(idx).fillna(False).sum())
                closure_rate = round(closed / total * 100, 1) if total else 0.0
                avg_res: Optional[float] = None
                if res_days is not None:
                    closed_days = res_days.reindex(idx)[closed_mask.reindex(idx).fillna(False)]
                    if len(closed_days.dropna()):
                        avg_res = round(float(closed_days.dropna().mean()), 1)
                rows.append({
                    "Category": str(cat_val),
                    "Total": total,
                    "Closed": closed,
                    "Pending": total - closed,
                    "Closure Rate (%)": closure_rate,
                    "Avg Resolution (days)": avg_res,
                })
            return pd.DataFrame(rows).sort_values("Total", ascending=False).reset_index(drop=True)
        except Exception:
            return None

    def compute_monthly_trend(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        ok, _ = self._eligibility.check("monthly_trend")
        if not ok:
            return None
        reg_col = self._registry.resolve(ROLE_REGISTRATION_DATE)
        if not reg_col or reg_col not in df.columns:
            return None
        try:
            dt = pd.to_datetime(df[reg_col], errors="coerce")
            valid = df[dt.notna()].copy()
            valid["_year"] = dt[dt.notna()].dt.year
            valid["_month"] = dt[dt.notna()].dt.month
            valid["_period"] = dt[dt.notna()].dt.to_period("M")
            closed_mask = _resolve_status_mask(valid, self._registry, _STATUS_CLOSED)
            rows: List[Dict[str, Any]] = []
            for period, group in valid.groupby("_period", sort=True):
                idx = group.index
                total = len(group)
                closed = int(closed_mask.reindex(idx).fillna(False).sum())
                rows.append({
                    "Period": str(period),
                    "Year": int(group["_year"].iloc[0]),
                    "Month": int(group["_month"].iloc[0]),
                    "Total": total,
                    "Closed": closed,
                    "Pending": total - closed,
                    "Closure Rate (%)": round(closed / total * 100, 1) if total else 0.0,
                })
            return pd.DataFrame(rows)
        except Exception:
            return None

    def compute_hierarchy_risk(
        self, df: pd.DataFrame, reference_date: Optional[pd.Timestamp] = None
    ) -> Optional[pd.DataFrame]:
        ok, _ = self._eligibility.check("hierarchy_risk")
        if not ok:
            return None
        # Refactor Phase 2B / Reference-Date Normalization: defensive
        # re-normalization at this public analytical entry point.
        reference_date = _normalize_reference_date(reference_date)
        # MILESTONE 1 / ISSUE 1, 7 REMEDIATION: now scans the shared
        # _ADMIN_HIERARCHY_ROLES constant (Zone/Circle/Division/
        # Subdivision/Substation) instead of a locally-hardcoded 4-role
        # list, so this function's actual runtime dependency now matches
        # both the eligibility check above and core.roles
        # .CANONICAL_GEO_HIERARCHY exactly.
        hierarchy_roles = list(_ADMIN_HIERARCHY_ROLES)
        group_col: Optional[str] = None
        group_role: str = ""
        for role in hierarchy_roles:
            col = self._registry.resolve(role)
            if col and col in df.columns:
                group_col = col
                group_role = role
                break
        if not group_col:
            return None
        try:
            # Milestone 16 / Issue 5a — string normalization fix (see the
            # identical comment in compute_officer_productivity above).
            df = df.copy()
            df[group_col] = df[group_col].astype(str).str.strip()

            reg_col = self._registry.resolve(ROLE_REGISTRATION_DATE)
            # Milestone 17 / Task 2 — defensive flexible pending mask:
            # this Group A flexible-eligibility KPI now derives Pending
            # status from Closing Date presence when Status is unmapped,
            # instead of the original direct
            # `_resolve_status_mask(df, self._registry, _STATUS_PENDING)`
            # call (which always returned all-False when Status was
            # unmapped). Byte-identical whenever Status IS mapped.
            _closed_mask, pending_mask, _method = self._resolve_flexible_closed_pending_masks(df)
            pending_df = df[pending_mask]

            if reg_col and reg_col in df.columns:
                # Refactor Phase 2B / Datetime & Lifecycle Fix: routed
                # through _safe_tz_naive so the reference-date fallback
                # (max registration date) can never itself carry tz
                # metadata that later mismatches a tz-naive
                # reference_date override.
                all_reg_dt = _safe_tz_naive(df[reg_col])
                ref = reference_date if reference_date is not None else all_reg_dt.dropna().max()
            else:
                ref = None
            ref = _normalize_reference_date(ref)

            rows: List[Dict[str, Any]] = []
            for group_val, group in df.groupby(group_col, sort=False):
                idx = group.index
                total = len(group)
                pending = int(pending_mask.reindex(idx).fillna(False).sum())
                avg_age: Optional[float] = None
                max_age: Optional[float] = None
                if ref is not None and reg_col and reg_col in pending_df.columns:
                    pend_in_group = pending_df[pending_df[group_col] == group_val]
                    # Refactor Phase 2B / Datetime & Lifecycle Fix: routed
                    # through _safe_tz_naive, eliminating a tz-mismatch
                    # TypeError against `ref` (already tz-normalized above).
                    reg_dt = _safe_tz_naive(pend_in_group[reg_col])
                    ages = ((ref - reg_dt).dt.total_seconds() / 86400).where(lambda s: s >= 0).dropna()
                    if len(ages):
                        avg_age = round(float(ages.mean()), 1)
                        max_age = round(float(ages.max()), 1)
                pending_rate = pending / total if total else 0.0
                age_factor = min((avg_age or 0) / 30.0, 5.0)
                risk_score = round(pending_rate * (1 + age_factor) * 100, 1)
                rows.append({
                    "Group": str(group_val),
                    "Level": group_role.replace("_", " ").title(),
                    "Total Cases": total,
                    "Pending": pending,
                    "Avg Pending Age (days)": avg_age,
                    "Max Pending Age (days)": max_age,
                    "Risk Score": risk_score,
                })
            result_df = pd.DataFrame(rows).sort_values("Risk Score", ascending=False).reset_index(drop=True)
            if len(result_df):
                scores = result_df["Risk Score"]
                q33, q66 = scores.quantile(0.33), scores.quantile(0.66)
                result_df["Risk Tier"] = pd.cut(
                    scores,
                    bins=[-np.inf, q33, q66, np.inf],
                    labels=["Low", "Medium", "High"],
                )
            return result_df
        except Exception:
            return None

    def compute_top_repeat_consumers(self, df: pd.DataFrame, top_n: int = 20) -> Optional[pd.DataFrame]:
        ok, _ = self._eligibility.check("top_repeat_consumers")
        if not ok:
            return None
        consumer_col = self._registry.resolve(ROLE_CONSUMER_ID)
        id_col = self._registry.resolve(ROLE_RECORD_ID)
        if not consumer_col or consumer_col not in df.columns:
            return None
        try:
            count_series = (
                df.groupby(consumer_col)[id_col].count()
                if id_col and id_col in df.columns
                else df[consumer_col].value_counts()
            )
            repeat = count_series[count_series > 1].sort_values(ascending=False).head(top_n)
            result = repeat.reset_index()
            result.columns = ["Consumer ID", "Complaint Count"]
            return result
        except Exception:
            return None

    def compute_sla_breach_detail(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        ok, _ = self._eligibility.check("sla_breach_detail")
        if not ok:
            return None
        sla_col = self._registry.resolve(ROLE_SLA_DEADLINE)
        close_col = self._registry.resolve(ROLE_CLOSING_DATE)
        if not sla_col or not close_col:
            return None
        try:
            # Milestone 17 / Task 2 — defensive: closed-row filter now
            # routes through the flexible resolver (byte-identical to the
            # original direct `_resolve_status_mask` call whenever Status
            # IS mapped).
            closed_mask, _pending_mask, _method = self._resolve_flexible_closed_pending_masks(df)
            closed_df = df[closed_mask].copy()
            # Refactor Phase 2B / Datetime & Lifecycle Fix: both SLA
            # deadline and closing date are now routed through
            # _safe_tz_naive before the breach-days delta, eliminating a
            # tz-mismatch TypeError.
            sla_dt = _safe_tz_naive(closed_df[sla_col])
            close_dt = _safe_tz_naive(closed_df[close_col])
            valid = sla_dt.notna() & close_dt.notna()
            breach_mask = valid & (close_dt > sla_dt)
            breach_df = closed_df[breach_mask].copy()
            if breach_df.empty:
                return pd.DataFrame(columns=["Breach Days"])
            breach_df["Breach Days"] = ((close_dt[breach_mask] - sla_dt[breach_mask]).dt.total_seconds() / 86400).round(1)
            return breach_df.sort_values("Breach Days", ascending=False).reset_index(drop=True)
        except Exception:
            return None

    # ── Phase 2: Executive Narrative Engine (migrated SSOT math) ────────────

    def _compute_half_metrics(
        self, half_df: pd.DataFrame, reference_date: Optional[pd.Timestamp]
    ) -> Tuple[int, float, float]:
        """Returns (volume, mttr_hours, sla_compliance_pct) for a
        Period-over-Period half-slice, delegating entirely to compute_all()
        to guarantee mathematical consistency (95th-percentile clipping,
        SLA logic) with every other consumer of this engine."""
        if half_df.empty:
            return 0, 0.0, 0.0
        try:
            kpis = self.compute_all(half_df, reference_date=reference_date)
            total_cases_kpi = kpis.get("total_cases")
            mttr_kpi = kpis.get("avg_resolution_time")
            sla_kpi = kpis.get("sla_compliance_rate")
            volume = int(total_cases_kpi.value or 0) if total_cases_kpi is not None else 0
            mttr_hours = float(mttr_kpi.value or 0.0) if mttr_kpi is not None else 0.0
            sla_compliance = float(sla_kpi.value or 0.0) if sla_kpi is not None else 0.0
            return volume, mttr_hours, sla_compliance
        except Exception:
            return len(half_df), 0.0, 0.0

    def _compute_hierarchy_anomalies(
        self,
        pending_df: pd.DataFrame,
        pending_age_days: pd.Series,
        hier_col: Optional[str],
        hier_label: Optional[str],
    ) -> List[HierarchyAnomaly]:
        if not hier_col or hier_col not in pending_df.columns or pending_df.empty:
            return []
        try:
            pending_sub = pending_df.copy()
            pending_sub["_pending_age"] = pending_age_days.reindex(pending_sub.index)
            grouped = (
                pending_sub.groupby(hier_col, dropna=True)
                .agg(
                    pending_count=(hier_col, "size"),
                    avg_pending_age=("_pending_age", "mean"),
                )
                .reset_index()
            )
            grouped["avg_pending_age"] = grouped["avg_pending_age"].fillna(0.0)
            grouped["risk_score"] = (
                grouped["pending_count"].astype(float) * grouped["avg_pending_age"]
            ).round(4)

            if grouped.empty:
                return []

            scores = grouped["risk_score"].astype(float)
            grouped["z_score"] = _compute_modified_z_scores(scores).round(4)
            percentiles = scores.rank(pct=True) * 100.0

            severity_conditions = [
                percentiles >= 90.0,
                percentiles >= 75.0,
                percentiles >= 50.0,
            ]
            severity_choices = ["CRITICAL", "HIGH", "MEDIUM"]
            grouped["severity"] = np.select(severity_conditions, severity_choices, default="LOW")

            anomalies_df = grouped.sort_values("risk_score", ascending=False).copy()
            anomalies_df = anomalies_df.rename(columns={hier_col: "unit_name"})
            anomalies_df["unit_name"] = anomalies_df["unit_name"].astype(str)

            records = anomalies_df.to_dict("records")
            return [
                HierarchyAnomaly(
                    unit_type=hier_label or "Unit",
                    unit_name=str(r["unit_name"]),
                    pending_count=int(r["pending_count"]),
                    avg_pending_age=float(r["avg_pending_age"]),
                    risk_score=float(r["risk_score"]),
                    z_score=float(r["z_score"]),
                    severity=str(r["severity"]),
                )
                for r in records
            ]
        except Exception:
            return []

    def _compute_pareto_flat_hotspots(
        self,
        pending_df: pd.DataFrame,
        category_col: Optional[str],
        hier_col: Optional[str],
    ) -> Tuple[List[ParetoHotspotRow], int]:
        """Backward-compatible flat Category × Administrative-Unit Pareto
        hotspot list. Behaviourally identical to the original executive
        report's inline implementation — migrated verbatim, not rewritten."""
        if not category_col or not hier_col:
            return [], 0
        if category_col not in pending_df.columns or hier_col not in pending_df.columns or pending_df.empty:
            return [], 0
        try:
            same_column = category_col == hier_col
            group_cols = [hier_col] if same_column else [category_col, hier_col]

            vol_by_group = (
                pending_df.groupby(group_cols, dropna=True)
                .size()
                .reset_index(name="pending_volume")
                .sort_values("pending_volume", ascending=False)
                .reset_index(drop=True)
            )
            total_pending_vol = float(vol_by_group["pending_volume"].sum())

            if total_pending_vol <= 0.0 or vol_by_group.empty:
                return [], 0

            vol_by_group["cum_pct"] = (
                vol_by_group["pending_volume"].cumsum() / total_pending_vol * 100.0
            )
            reached_target = vol_by_group["cum_pct"] >= _PARETO_TARGET_PCT
            cutoff_idx = (
                int(reached_target.idxmax()) if bool(reached_target.any())
                else int(len(vol_by_group) - 1)
            )
            hotspot_slice = vol_by_group.iloc[: cutoff_idx + 1].copy()
            hotspot_slice["backlog_contribution_pct"] = (
                hotspot_slice["pending_volume"] / total_pending_vol * 100.0
            ).round(4)

            if same_column:
                hotspot_slice["category"] = hotspot_slice[hier_col].astype(str)
                hotspot_slice = hotspot_slice.rename(columns={hier_col: "administrative_unit"})
            else:
                hotspot_slice = hotspot_slice.rename(
                    columns={category_col: "category", hier_col: "administrative_unit"}
                )

            hotspot_slice["category"] = hotspot_slice["category"].astype(str)
            hotspot_slice["administrative_unit"] = hotspot_slice["administrative_unit"].astype(str)

            full_records = hotspot_slice[
                ["category", "administrative_unit", "backlog_contribution_pct"]
            ].to_dict("records")

            rows = [
                ParetoHotspotRow(
                    category=str(r["category"]),
                    administrative_unit=str(r["administrative_unit"]),
                    backlog_contribution_pct=float(r["backlog_contribution_pct"]),
                )
                for r in full_records
            ]

            if len(rows) > _PARETO_TRUNCATE_THRESHOLD:
                omitted = len(rows) - _PARETO_TRUNCATE_KEEP
                return rows[:_PARETO_TRUNCATE_KEEP], omitted
            return rows, 0
        except Exception:
            return [], 0

    def _compute_pareto_hierarchy(self, pending_df: pd.DataFrame) -> List[ParetoHierarchyNode]:
        """Builds the nested geographic Pareto drill tree, resolved strictly
        via the ZONE->CIRCLE->DIVISION->SUBDIVISION->SUBSTATION->FEEDER
        priority. Falls back to a single-level flat drill on Officer or
        Category ONLY when no geographic hierarchy role resolves at all."""
        if pending_df.empty:
            return []
        try:
            path_cols: List[str] = []
            path_labels: List[str] = []
            for role, label in _GEO_HIERARCHY_PRIORITY:
                col = self._registry.resolve(role)
                if col and col in pending_df.columns and pending_df[col].notna().any():
                    path_cols.append(col)
                    path_labels.append(label)

            if not path_cols:
                for role, label in _FALLBACK_HIERARCHY_PRIORITY:
                    col = self._registry.resolve(role)
                    if col and col in pending_df.columns and pending_df[col].notna().any():
                        path_cols.append(col)
                        path_labels.append(label)
                        break  # secondary fallback is intentionally flat (single level)

            if not path_cols:
                return []

            return _build_pareto_hierarchy_recursive(pending_df, path_cols, path_labels, 0)
        except Exception:
            return []

    def _empty_executive_bundle(self, reason: str, generated_at: str) -> ExecutiveNarrativeBundle:
        meta = ExecutiveMeta(
            generated_at=generated_at,
            date_range=("", ""),
            lowest_hierarchy_unit="Not Available",
            hierarchy_role=None,
            lifecycle_method="unresolved",
            calculation_quality="ESTIMATED",
            estimation_reason=reason,
        )
        global_kpis = ExecutiveGlobalKPIs(
            sla_compliance=0.0,
            sla_pop_delta=0.0,
            mttr_hours=0.0,
            mttr_pop_delta=0.0,
            total_volume=0,
            pending_count=0,
            avg_pending_age_days=0.0,
            business_risk_score=0.0,
        )
        return ExecutiveNarrativeBundle(
            meta=meta,
            global_kpis=global_kpis,
            anomalies=[],
            pareto_hierarchy=[],
            pareto_hotspots_flat=[],
            pareto_omitted_count=0,
            kpi_snapshot=self._generate_empty_kpis(),
        )

    def generate_executive_bundle(
        self,
        df: pd.DataFrame,
        reference_date: Optional[pd.Timestamp] = None,
    ) -> ExecutiveNarrativeBundle:
        """
        Computes the complete, structured Executive Narrative payload —
        lifecycle resolution, Period-over-Period deltas, Modified Z-Score
        hierarchy anomalies, and the dual (flat + nested-geographic) Pareto
        80/20 hotspot analysis. This is the SOLE authoritative source of
        this math in the platform; the presentation layer must never
        recompute any of it — only format the returned dataclass tree.

        Never raises: every structural, datetime, or grouping failure
        degrades to `_empty_executive_bundle(...)` with an explanatory
        `estimation_reason`, per the platform's non-destructive,
        never-crash contract.
        """
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Refactor Phase 2B / Reference-Date Normalization: defensive
        # re-normalization at this public analytical entry point, before
        # it is threaded into compute_all()/_compute_half_metrics() and
        # compared against any internally-parsed (tz-naive) datetime
        # series below.
        reference_date = _normalize_reference_date(reference_date)

        if df is None or df.empty:
            return self._empty_executive_bundle(
                "The active dataset is empty or unavailable.", generated_at
            )

        reg_col = self._registry.resolve(ROLE_REGISTRATION_DATE)
        if not reg_col or reg_col not in df.columns:
            return self._empty_executive_bundle(
                "Registration Date role is not mapped. Map a Registration/Open Date "
                "column in the Schema Mapping Studio to enable executive reporting.",
                generated_at,
            )

        try:
            work = df.copy(deep=True)
            work["_reg_dt"] = _safe_tz_naive(work[reg_col])
            work = work.loc[work["_reg_dt"].notna()].copy()
            if work.empty:
                return self._empty_executive_bundle(
                    f"No parseable registration dates were found in column '{reg_col}'.",
                    generated_at,
                )

            close_col = self._registry.resolve(ROLE_CLOSING_DATE)
            status_col = self._registry.resolve(ROLE_STATUS)

            has_close = bool(close_col and close_col in work.columns)
            has_status = bool(status_col and status_col in work.columns)
            estimation_reason = ""
            calculation_quality = "MEASURED"
            lifecycle_method: str

            # if has_close:
            #     work["_close_dt"] = _safe_tz_naive(work[close_col])
            #     pending_mask = work["_close_dt"].isna()
            #     closed_mask = ~pending_mask
            #     lifecycle_method = "dates"
            # elif has_status:
            #     status_norm = work[status_col].astype(str).str.strip().str.lower()
            #     closed_mask = status_norm.isin(_STATUS_TOKENS[_STATUS_CLOSED])
            #     pending_mask = status_norm.isin(_STATUS_TOKENS[_STATUS_PENDING])
            #     lifecycle_method = "status"
            # else:
            #     pending_mask = pd.Series(True, index=work.index)
            #     closed_mask = pd.Series(False, index=work.index)
            #     lifecycle_method = "fallback"
            #     calculation_quality = "ESTIMATED"
            #     estimation_reason = (
            #         "Neither a Closing Date nor a Status role is mapped, so complaint "
            #         "lifecycle state could not be verified directly. Every record has "
            #         "been provisionally treated as Pending, and Pending Age has been "
            #         "estimated using the latest available Registration Date across the "
            #         "dataset as the reference point."
            #     )
            
            if has_status:
                status_norm = work[status_col].astype(str).str.strip().str.lower()
                closed_mask = status_norm.isin(_STATUS_TOKENS[_STATUS_CLOSED])
                pending_mask = status_norm.isin(_STATUS_TOKENS[_STATUS_PENDING])
                lifecycle_method = "status"
            elif has_close:
                work["_close_dt"] = _safe_tz_naive(work[close_col])
                pending_mask = work["_close_dt"].isna()
                closed_mask = ~pending_mask
                lifecycle_method = "dates"
            else:
                pending_mask = pd.Series(True, index=work.index)
                closed_mask = pd.Series(False, index=work.index)
                lifecycle_method = "fallback"
                calculation_quality = "ESTIMATED"
                estimation_reason = (
                    "Neither a Closing Date nor a Status role is mapped, so complaint "
                    "lifecycle state could not be verified directly. Every record has "
                    "been provisionally treated as Pending, and Pending Age has been "
                    "estimated using the latest available Registration Date across the "
                    "dataset as the reference point."
                )
            

            
                

            max_reg_date = work["_reg_dt"].max()
            pending_slice_dates = work.loc[pending_mask, "_reg_dt"]
            pending_age_days = ((max_reg_date - pending_slice_dates).dt.total_seconds() / 86400.0)
            pending_age_days = pending_age_days.clip(lower=0.0)

            pending_count = int(pending_mask.sum())
            # Outlier-resilient average: top 5% of ages clipped at P95 before averaging
            if len(pending_age_days):
                avg_pending_age = float(self._apply_adaptive_pending_age_fence(pending_age_days).mean())
            else:
                avg_pending_age = 0.0
            
            kpi_snapshot = self.compute_all(work, reference_date=reference_date)
            total_cases_kpi = kpi_snapshot.get("total_cases")
            mttr_kpi = kpi_snapshot.get("avg_resolution_time")
            sla_kpi = kpi_snapshot.get("sla_compliance_rate")

            total_volume = int((total_cases_kpi.value if total_cases_kpi is not None else None) or len(work))
            mttr_hours = float((mttr_kpi.value if mttr_kpi is not None else None) or 0.0)
            sla_compliance = float((sla_kpi.value if sla_kpi is not None else None) or 0.0)

            business_risk_score_global = round(float(pending_count) * avg_pending_age, 4)

            sorted_work = work.sort_values("_reg_dt").reset_index(drop=True)
            n_rows = len(sorted_work)
            half_point = n_rows // 2
            first_half = sorted_work.iloc[:half_point]
            second_half = sorted_work.iloc[half_point:]

            _vol1, mttr1, sla1 = self._compute_half_metrics(first_half, reference_date)
            _vol2, mttr2, sla2 = self._compute_half_metrics(second_half, reference_date)

            sla_pop_delta = round(sla2 - sla1, 4)
            mttr_pop_delta = round(mttr2 - mttr1, 4)

            hier_role, hier_label, hier_col = _resolve_lowest_hierarchy_unit(self._registry, work)

            pending_sub = work.loc[pending_mask]

            anomalies = self._compute_hierarchy_anomalies(
                pending_sub, pending_age_days, hier_col, hier_label
            )

            category_col = self._registry.resolve(ROLE_CATEGORY)
            pareto_hotspots_flat, pareto_omitted_count = self._compute_pareto_flat_hotspots(
                pending_sub, category_col, hier_col
            )
            pareto_hierarchy = self._compute_pareto_hierarchy(pending_sub)

            date_range = (
                str(work["_reg_dt"].min().date()),
                str(work["_reg_dt"].max().date()),
            )
            lowest_hierarchy_unit = hier_label or "Not Available"

            meta = ExecutiveMeta(
                generated_at=generated_at,
                date_range=date_range,
                lowest_hierarchy_unit=lowest_hierarchy_unit,
                hierarchy_role=hier_role,
                lifecycle_method=lifecycle_method,
                calculation_quality=calculation_quality,
                estimation_reason=estimation_reason,
            )
            global_kpis = ExecutiveGlobalKPIs(
                sla_compliance=round(_safe_float(sla_compliance), 4),
                sla_pop_delta=round(_safe_float(sla_pop_delta), 4),
                mttr_hours=round(_safe_float(mttr_hours), 4),
                mttr_pop_delta=round(_safe_float(mttr_pop_delta), 4),
                total_volume=int(total_volume),
                pending_count=pending_count,
                avg_pending_age_days=round(_safe_float(avg_pending_age), 4),
                business_risk_score=business_risk_score_global,
            )

            return ExecutiveNarrativeBundle(
                meta=meta,
                global_kpis=global_kpis,
                anomalies=anomalies,
                pareto_hierarchy=pareto_hierarchy,
                pareto_hotspots_flat=pareto_hotspots_flat,
                pareto_omitted_count=pareto_omitted_count,
                kpi_snapshot=kpi_snapshot,
            )
        except Exception as exc:
            return self._empty_executive_bundle(
                f"An unexpected error occurred while generating the executive report: {exc}",
                generated_at,
            )


class UniversalAnalyticsEngine:

    def __init__(self, registry: ColumnRegistry, parquet_path: Optional[str] = None) -> None:
        self._registry = registry
        self._eligibility = MetricEligibilityEngine(registry)
        # Refactor Phase 2B / Parquet Pathing: optional, purely additive.
        # Threaded through to every ComplaintKPIEngine this class
        # constructs so DuckDB count-acceleration hooks can route directly
        # against a Parquet file when one is available. Every existing
        # caller (e.g. `UniversalAnalyticsEngine(registry)`, unchanged in
        # visualization/kpi_cards.py) continues to work identically, since
        # this parameter defaults to None.
        self._parquet_path = parquet_path

    def run_complaint_analytics(
        self,
        df: pd.DataFrame,
        reference_date: Optional[pd.Timestamp] = None,
        top_n_consumers: int = 20,
    ) -> ComplaintAnalyticsBundle:
        engine = ComplaintKPIEngine(self._registry, self._eligibility, parquet_path=self._parquet_path)
        kpis = engine.compute_all(df, reference_date=reference_date)
        eligibility_report = self._eligibility.report()
        return ComplaintAnalyticsBundle(
            kpis=kpis,
            eligibility_report=eligibility_report,
            officer_productivity=engine.compute_officer_productivity(df),
            category_breakdown=engine.compute_category_breakdown(df),
            monthly_trend=engine.compute_monthly_trend(df),
            hierarchy_risk=engine.compute_hierarchy_risk(df, reference_date),
            top_repeat_consumers=engine.compute_top_repeat_consumers(df, top_n=top_n_consumers),
            sla_breach_detail=engine.compute_sla_breach_detail(df),
        )

    def check_eligibility(self, kpi_names: Optional[List[str]] = None) -> EligibilityReport:
        return self._eligibility.report(kpi_names)

    def get_status_tokens(self) -> Dict[str, FrozenSet[str]]:
        """Public accessor for the canonical status-token ontology. The
        reporting layer must use this instead of maintaining its own
        token lists, per the Status Token Integrity mandate."""
        return dict(_STATUS_TOKENS)

    def generate_executive_bundle(
        self,
        df: pd.DataFrame,
        reference_date: Optional[pd.Timestamp] = None,
        **kwargs: Any,
    ) -> ExecutiveNarrativeBundle:
        """
        Public entry point for the Executive Narrative Engine. Returns the
        fully structured `ExecutiveNarrativeBundle` — global KPIs,
        Period-over-Period deltas, hierarchy anomalies, and both the flat
        and nested-geographic Pareto 80/20 hotspot analyses. This is the
        only method the presentation layer (executive_summary.py) should
        call to obtain executive-report data; it performs zero rendering
        and contains zero display concerns.

        **kwargs is accepted (and currently unused) for forward
        compatibility with future self-service override parameters
        (e.g. group_by, top_n) without breaking this signature.
        """
        engine = ComplaintKPIEngine(self._registry, self._eligibility, parquet_path=self._parquet_path)
        return engine.generate_executive_bundle(df, reference_date=reference_date)