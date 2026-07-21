"""
smoke_test.py — Unit Validator for engine.analytics.MetricEligibilityEngine

Purpose:
    Pinpoints exactly where Flexible-OR eligibility logic (Milestone 17 /
    Task 1 — Group A: officer_productivity, category_breakdown,
    hierarchy_risk; Group B: closed_cases, avg_resolution_time,
    median_resolution_time, p95_resolution_time, sla_compliance_rate,
    sla_breach_rate, sla_breach_detail, first_time_resolution_rate)
    diverges from expectation.

Design notes (why scenarios are built this way):
    Group A KPIs and Group B KPIs each carry OTHER strictly-required roles
    beyond Status/Closing Date (e.g. officer_productivity ALSO requires
    ROLE_OFFICER + ROLE_RECORD_ID; hierarchy_risk ALSO requires an
    administrative hierarchy role). To isolate the ONE variable this test
    cares about (Status vs Closing Date presence), every scenario below
    maps every OTHER required role identically across all three scenarios,
    and only toggles ROLE_STATUS / ROLE_CLOSING_DATE. This is what makes a
    'Only Status mapped -> Group A pass, Group B fail' assertion valid: if
    it were literally 'only Status and nothing else', Group A would fail
    for an unrelated reason (missing Officer/Category/Record ID), and the
    printed missing_roles would mislead you into thinking Flexible-OR
    itself was broken when it wasn't.

Usage:
    python smoke_test.py
"""
from __future__ import annotations

import sys
from typing import Dict, List, Optional, Tuple

# ── Imports aligned exactly with the refactored analytics.py ────────────
from core.column_registry import ColumnRegistry
from core.roles import (
    ROLE_RECORD_ID,
    ROLE_REGISTRATION_DATE,
    ROLE_CLOSING_DATE,
    ROLE_STATUS,
    ROLE_CATEGORY,
    ROLE_OFFICER,
    ROLE_SLA_DEADLINE,
    ROLE_ZONE,
)
from engine.analytics import (
    MetricEligibilityEngine,
    _FLEXIBLE_GROUP_A_KPIS,
    _FLEXIBLE_GROUP_B_KPIS,
)

# ── ANSI-free, terminal-safe status tokens (avoid encoding issues on
# constrained CI shells) ─────────────────────────────────────────────────
_PASS_TAG = "[SUCCESS]"
_FAIL_TAG = "[FAILURE]"

# KPI names asserted in every scenario, split by group for clarity.
_GROUP_A_KPIS: Tuple[str, ...] = ("officer_productivity", "category_breakdown", "hierarchy_risk")
_GROUP_B_KPIS: Tuple[str, ...] = (
    "closed_cases", "avg_resolution_time", "median_resolution_time",
    "p95_resolution_time", "sla_compliance_rate", "sla_breach_rate",
    "sla_breach_detail", "first_time_resolution_rate",
)

# Sanity check: fail loudly (not silently) if analytics.py's own group
# membership sets ever drift from what this test assumes.
_EXPECTED_GROUP_A = frozenset(_GROUP_A_KPIS)
_EXPECTED_GROUP_B = frozenset(_GROUP_B_KPIS)


class _ScenarioFailure(AssertionError):
    """Raised internally to short-circuit a scenario after logging detail."""


def _build_registry(
    *,
    include_status: bool,
    include_closing_date: bool,
    scenario_label: str,
) -> ColumnRegistry:
    """
    Builds a ColumnRegistry with every 'other' required role for Group A/B
    KPIs mapped identically, plus Status and/or Closing Date toggled per
    the scenario. Uses set_mapping(..., manual=True) exactly as
    schema_mapping.py / the Schema Mapping Studio does, so this exercises
    the real public ColumnRegistry API — not a mocked stand-in.
    """
    registry = ColumnRegistry(workspace_name=f"SmokeTest::{scenario_label}")

    # Roles required by Group A/B KPIs OTHER than Status/Closing Date —
    # held constant across every scenario so only Status/Closing Date
    # presence varies.
    registry.set_mapping(ROLE_RECORD_ID, "Complaint ID", manual=True)
    registry.set_mapping(ROLE_OFFICER, "Officer", manual=True)
    registry.set_mapping(ROLE_CATEGORY, "Category", manual=True)
    registry.set_mapping(ROLE_REGISTRATION_DATE, "Registration Date", manual=True)
    registry.set_mapping(ROLE_SLA_DEADLINE, "SLA Deadline", manual=True)
    # hierarchy_risk's supplemental admin-hierarchy check (Zone/Circle/
    # Division/Subdivision/Substation) needs at least one of these mapped
    # independently of the Status/Closing-Date variable under test.
    registry.set_mapping(ROLE_ZONE, "Zone", manual=True)

    if include_status:
        registry.set_mapping(ROLE_STATUS, "Status", manual=True)
    if include_closing_date:
        registry.set_mapping(ROLE_CLOSING_DATE, "Closing Date", manual=True)

    return registry


def _print_engine_state(engine: MetricEligibilityEngine, kpi_name: str, missing: List[str]) -> None:
    """Verbose diagnostic dump: exactly which roles the engine currently
    considers missing for this KPI, plus the registry's current resolvable
    role set, so a failure is traceable to a specific column/role rather
    than a bare boolean."""
    resolved_roles = sorted(
        role for role, mapping in engine._registry.mappings.items()
        if mapping.confirmed and mapping.column_name
    )
    display_missing = [engine._registry.display_name(r) for r in missing]
    print(f"        -> missing_roles (raw)     : {missing}")
    print(f"        -> missing_roles (display) : {display_missing}")
    print(f"        -> currently resolved roles: {resolved_roles}")


def _assert_eligible(
    engine: MetricEligibilityEngine,
    kpi_name: str,
    expect_eligible: bool,
    scenario_label: str,
) -> bool:
    """Runs engine.check(kpi_name), compares against expectation, prints a
    SUCCESS/FAILURE line, and returns True iff the assertion held. Never
    raises — failures are captured and reported, not thrown, so a single
    bad KPI doesn't abort the rest of the scenario."""
    try:
        is_eligible, missing = engine.check(kpi_name)
    except Exception as exc:  # noqa: BLE001
        print(f"    {_FAIL_TAG} [{scenario_label}] check('{kpi_name}') RAISED an exception: {exc!r}")
        return False

    if is_eligible == expect_eligible:
        print(
            f"    {_PASS_TAG} [{scenario_label}] {kpi_name:<28} "
            f"expected eligible={expect_eligible!s:<5} got eligible={is_eligible!s:<5}"
        )
        return True

    print(
        f"    {_FAIL_TAG} [{scenario_label}] {kpi_name:<28} "
        f"expected eligible={expect_eligible!s:<5} got eligible={is_eligible!s:<5}"
    )
    _print_engine_state(engine, kpi_name, missing)
    return False


def _run_scenario(
    scenario_label: str,
    *,
    include_status: bool,
    include_closing_date: bool,
    expect_group_a: bool,
    expect_group_b: bool,
) -> Tuple[int, int]:
    """Runs one full scenario across every Group A and Group B KPI.
    Returns (passed_count, total_count). Never raises."""
    print(f"\n{'=' * 78}")
    print(f"SCENARIO: {scenario_label}")
    print(
        f"  Status mapped={include_status} | Closing Date mapped={include_closing_date} "
        f"| expect Group A eligible={expect_group_a} | expect Group B eligible={expect_group_b}"
    )
    print(f"{'=' * 78}")

    registry = _build_registry(
        include_status=include_status,
        include_closing_date=include_closing_date,
        scenario_label=scenario_label,
    )
    engine = MetricEligibilityEngine(registry)

    passed = 0
    total = 0

    print("  -- Group A (Status OR Closing Date, inclusive) --")
    for kpi_name in _GROUP_A_KPIS:
        total += 1
        if _assert_eligible(engine, kpi_name, expect_group_a, scenario_label):
            passed += 1

    print("  -- Group B (Closing Date required, Status optional) --")
    for kpi_name in _GROUP_B_KPIS:
        total += 1
        if _assert_eligible(engine, kpi_name, expect_group_b, scenario_label):
            passed += 1

    return passed, total


def _verify_group_membership() -> bool:
    """Guards against this test silently testing the wrong thing if
    analytics.py's group-membership constants ever change without this
    file being updated in lockstep. Never raises — prints and returns
    False on mismatch instead."""
    ok = True
    if set(_GROUP_A_KPIS) != set(_FLEXIBLE_GROUP_A_KPIS):
        print(
            f"{_FAIL_TAG} Group A membership drift detected.\n"
            f"    smoke_test.py expects : {sorted(_GROUP_A_KPIS)}\n"
            f"    analytics.py declares : {sorted(_FLEXIBLE_GROUP_A_KPIS)}"
        )
        ok = False
    if set(_GROUP_B_KPIS) != set(_FLEXIBLE_GROUP_B_KPIS):
        print(
            f"{_FAIL_TAG} Group B membership drift detected.\n"
            f"    smoke_test.py expects : {sorted(_GROUP_B_KPIS)}\n"
            f"    analytics.py declares : {sorted(_FLEXIBLE_GROUP_B_KPIS)}"
        )
        ok = False
    if ok:
        print(f"{_PASS_TAG} Group A/B membership matches analytics.py exactly.")
    return ok


def main() -> int:
    print("KESCO engine.analytics.MetricEligibilityEngine — Flexible-OR Unit Validator")
    print("Validating against the live analytics.py in this environment.\n")

    membership_ok = _verify_group_membership()

    total_passed = 0
    total_checks = 0

    # ── Scenario 1: Only Status mapped ──────────────────────────────
    # Expect: Group A eligible (Status satisfies the OR).
    #         Group B ineligible (Closing Date is unconditionally
    #         required in Group B's effective role set — Status alone
    #         is never sufficient there, by design).
    p, t = _run_scenario(
        "Only 'Status' mapped",
        include_status=True,
        include_closing_date=False,
        expect_group_a=True,
        expect_group_b=False,
    )
    total_passed += p
    total_checks += t

    # ── Scenario 2: Only Closing Date mapped ────────────────────────
    # Expect: Group A eligible (Closing Date satisfies the OR).
    #         Group B eligible (Closing Date is exactly what Group B
    #         requires; Status is optional).
    p, t = _run_scenario(
        "Only 'Closing Date' mapped",
        include_status=False,
        include_closing_date=True,
        expect_group_a=True,
        expect_group_b=True,
    )
    total_passed += p
    total_checks += t

    # ── Scenario 3: Both mapped ──────────────────────────────────────
    # Expect: everything eligible.
    p, t = _run_scenario(
        "Both 'Status' and 'Closing Date' mapped",
        include_status=True,
        include_closing_date=True,
        expect_group_a=True,
        expect_group_b=True,
    )
    total_passed += p
    total_checks += t

    # ── Scenario 4 (negative control): Neither mapped ────────────────
    # Expect: everything ineligible. Proves the OR isn't accidentally
    # defaulting to True when nothing is mapped.
    p, t = _run_scenario(
        "Neither 'Status' nor 'Closing Date' mapped (negative control)",
        include_status=False,
        include_closing_date=False,
        expect_group_a=False,
        expect_group_b=False,
    )
    total_passed += p
    total_checks += t

    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"{'=' * 78}")
    print(f"Group membership check : {'OK' if membership_ok else 'MISMATCH — see above'}")
    print(f"Eligibility assertions : {total_passed} / {total_checks} passed")

    overall_ok = membership_ok and (total_passed == total_checks)
    if overall_ok:
        print(f"\n{_PASS_TAG} ALL CHECKS PASSED — Flexible-OR eligibility logic behaves as specified.")
    else:
        print(
            f"\n{_FAIL_TAG} ONE OR MORE CHECKS FAILED — see the missing_roles dump(s) above "
            f"for the exact role(s) the engine considers unresolved at the point of failure."
        )

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())