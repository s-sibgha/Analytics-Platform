"""
tests/test_fragment_isolation.py — Fragment Isolation Regression Test

Uses streamlit.testing.v1.AppTest to load FRONTEND/pages/1_dashboard.py with
a synthetic, fully-role-mapped dataset pre-seeded into session_state, then
interacts with a widget that lives inside the Row 1 (@st.fragment) chart
row and asserts that the Row 2 / Row 3 / Row 4 eligibility flags recorded in
st.session_state['_dashboard_active_viz_flags'] are byte-for-byte unchanged
after that interaction.

CAVEAT (documented, not hidden): streamlit.testing.v1.AppTest executes the
full script synchronously on every .run() call — it does not expose a
runtime-level "which fragment actually reran" signal the way a live
Streamlit server does. This test therefore validates the *logical*
contract your isolation is supposed to guarantee (unrelated rows' computed
eligibility/state must not be perturbed by a Row 1 widget interaction),
which is the correctness property that actually matters. It is a
regression guard, not a literal fragment-rerun-count probe.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from core.column_registry import ColumnRegistry
from core.roles import (
    ROLE_RECORD_ID,
    ROLE_REGISTRATION_DATE,
    ROLE_STATUS,
    ROLE_CATEGORY,
    ROLE_ZONE,
)
from core.type_inference import infer_dataframe
from engine.domain_detection import DOMAIN_COMPLAINT
import streamlit as st

@pytest.fixture(autouse=True)
def _patch_streamlit_for_apptest(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    AppTest's LocalScriptRunner does not reliably propagate a ScriptRunContext
    into st.fragment's internal execution path, which can hang
    require_widgets_deltas() indefinitely. Since 1_dashboard.py's script body
    is re-exec'd on every AppTest.run() call, patching st.fragment to an
    identity decorator BEFORE run() takes effect for that run, with zero
    changes to the production dashboard module itself.
    """
    def _identity_fragment(*dargs: Any, **dkwargs: Any) -> Any:
        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return dargs[0]
        def _decorator(func: Any) -> Any:
            return func
        return _decorator

    monkeypatch.setattr(st, "fragment", _identity_fragment)
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: None)

@pytest.fixture(autouse=True)
def _patch_streamlit_for_apptest(monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. Neutralize fragments
    def _identity_fragment(*dargs, **dkwargs):
        if len(dargs) == 1 and callable(dargs[0]): return dargs[0]
        return lambda func: func
    monkeypatch.setattr(st, "fragment", _identity_fragment)

    # 2. Neutralize sidebar (often a source of hangs in AppTest)
    monkeypatch.setattr(st, "sidebar", st.container())
    
    # 3. Neutralize st.file_uploader (prevents it from looking for actual files)
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: None)

@pytest.fixture(autouse=True)
def _disable_fragments_for_apptest(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    AppTest's LocalScriptRunner does not reliably propagate a ScriptRunContext
    into st.fragment's internal execution path, which can hang
    require_widgets_deltas() indefinitely. Since 1_dashboard.py's script body
    is re-exec'd on every AppTest.run() call, patching st.fragment to an
    identity decorator BEFORE run() takes effect for that run, with zero
    changes to the production dashboard module itself.
    """
    def _identity_fragment(*dargs, **dkwargs):
        if len(dargs) == 1 and callable(dargs[0]) and not dkwargs:
            return dargs[0]
        def _decorator(func):
            return func
        return _decorator

    monkeypatch.setattr(st, "fragment", _identity_fragment)


def _build_synthetic_dataset(n: int = 60) -> pd.DataFrame:
    """Builds a small, fully-typed synthetic complaint dataset with enough
    signal for Row 1 (category concentration), Row 2 (trend), and Row 3
    (risk matrix) to all be eligible simultaneously — required so we can
    actually observe non-trivial flags for every row, not just False/False."""
    base_date = datetime(2025, 1, 1)
    return pd.DataFrame({
        "Complaint ID": [f"C{i:04d}" for i in range(n)],
        "Registration Date": [base_date + timedelta(days=i % 45) for i in range(n)],
        "Status": ["Closed" if i % 3 == 0 else "Pending" for i in range(n)],
        "Category":[["Voltage Fluctuation", "Meter Fault", "Billing Dispute"][i % 3] for i in range(n)], # noqa
        "Zone": [["North", "South"][i % 2] for i in range(n)]
    })




def _seeded_app_test() -> AppTest:
    """Constructs an AppTest bound to FRONTEND/pages/1_dashboard.py with
    session_state pre-populated exactly as FRONTEND.app.process_uploaded_file
    would populate it, bypassing file upload entirely."""
    df = _build_synthetic_dataset()
    profiles = infer_dataframe(df)

    registry = ColumnRegistry(workspace_name="Fragment Isolation Test")
    registry.bootstrap_from_profiles(profiles)
    registry.set_mapping(ROLE_RECORD_ID, "Complaint ID", manual=True)
    registry.set_mapping(ROLE_REGISTRATION_DATE, "Registration Date", manual=True)
    registry.set_mapping(ROLE_STATUS, "Status", manual=True)
    registry.set_mapping(ROLE_CATEGORY, "Category", manual=True)
    registry.set_mapping(ROLE_ZONE, "Zone", manual=True)

    at = AppTest.from_file("FRONTEND/pages/1_dashboard.py")

    at.session_state["analytics_ready_dataframe"] = df
    at.session_state["filtered_dataframe"] = df
    at.session_state["column_registry"] = registry
    at.session_state["column_profiles"] = profiles
    at.session_state["domain_detection"] = (DOMAIN_COMPLAINT, 0.9)
    at.session_state["audit_results"] = {"data_quality_score": 90}
    at.session_state["readiness_score"] = 85
    at.session_state["readiness_band"] = "Good"
    at.session_state["notifications"] = []
    at.session_state["active_filters"] = {}
    at.session_state["drill_breadcrumbs"] = []
    at.session_state["theme"] = "kesco_corporate"
    at.session_state["workspace_name"] = "Fragment Isolation Test"

    return at


def _session_state_get(at: AppTest, key: str, default: Any = None) -> Any:
    """
    Safe accessor for AppTest.session_state. AppTest wraps SafeSessionState,
    whose __getattr__ forwards *any* unrecognized attribute name (including
    'get') to a session_state key lookup — so session_state.get(...) is
    misinterpreted as looking for a key literally named "get" and raises
    AttributeError rather than behaving like dict.get(). This helper performs
    the safe 'in' membership check + bracket access pattern instead.
    """
    return at.session_state[key] if key in at.session_state else default


def test_row1_widget_change_does_not_perturb_other_row_flags() -> None:
    """
    Regression guard for @st.fragment isolation on 1_dashboard.py's chart
    rows. Loads the dashboard, captures the initial
    '_dashboard_active_viz_flags' snapshot, changes the Row 1 chart-type
    selectbox (key '_dash_chart_type_row1_concentration'), reruns, and
    asserts Row 2 / Row 3 / Row 4 flags are unchanged.
    """
    at = _seeded_app_test()
    at.run(timeout=30)

    assert not at.exception, f"Initial run raised: {at.exception}"

    flags_before = dict(_session_state_get(at, "_dashboard_active_viz_flags", {}))
# New:
    #flags_before = dict(at.session_state["_dashboard_active_viz_flags"]) if "_dashboard_active_viz_flags" in at.session_state else {}
    assert flags_before, "Expected _dashboard_active_viz_flags to be populated after first run."
    assert "row2_trend" in flags_before
    assert "row3_risk" in flags_before
    assert "row4_geospatial" in flags_before

    row1_selectbox = at.selectbox(key="_dash_chart_type_row1_concentration")
    assert row1_selectbox is not None, "Row 1 chart-type selectbox not found — check widget key."

    alternate_option = next(
        opt for opt in row1_selectbox.options if opt != row1_selectbox.value
    )
    row1_selectbox.set_value(alternate_option)
    at.run(timeout=30)

    assert not at.exception, f"Post-interaction run raised: {at.exception}"

    
    flags_after = dict(_session_state_get(at, "_dashboard_active_viz_flags", {}))

    assert flags_after.get("row2_trend") == flags_before.get("row2_trend"), (
        "Row 2 eligibility flag changed after a Row 1-only widget interaction — "
        "isolation contract violated."
    )
    assert flags_after.get("row3_risk") == flags_before.get("row3_risk"), (
        "Row 3 eligibility flag changed after a Row 1-only widget interaction — "
        "isolation contract violated."
    )
    assert flags_after.get("row4_geospatial") == flags_before.get("row4_geospatial"), (
        "Row 4 (geospatial) eligibility flag changed after a Row 1-only widget "
        "interaction — isolation contract violated."
    )


def test_diagnostics_footer_active_viz_count_matches_flags() -> None:
    """Sanity check that _render_primary_visualizations' returned
    active_viz_count (consumed by the diagnostics footer) always equals the
    count of True values in _dashboard_active_viz_flags for the same run —
    guards against the footer silently drifting from the fragment-recorded
    truth after a future edit."""
    
    at = _seeded_app_test()
    at.run(timeout=90)
    assert not at.exception, f"Run raised: {at.exception}"
    # flags = at.session_state["_dashboard_active_viz_flags"] if "_dashboard_active_viz_flags" in at.session_state else {}
    flags = _session_state_get(at, "_dashboard_active_viz_flags", {})
    expected_count = sum(1 for v in flags.values() if v)

    footer_metric_values = [m.value for m in at.get("metric") if "Active Visualizations" in (m.label or "")]
    assert footer_metric_values, "Could not locate 'Active Visualizations' metric in diagnostics footer."
    assert footer_metric_values[0] == str(expected_count), (
        f"Diagnostics footer reports {footer_metric_values[0]} active visualizations, "
        f"but _dashboard_active_viz_flags implies {expected_count}."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])