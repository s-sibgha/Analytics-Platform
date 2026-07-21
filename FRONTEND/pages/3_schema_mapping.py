"""
FRONTEND/pages/3_schema_mapping.py — Schema Mapping Studio Page (INTEGRITY FIX)

Thin routing-mesh orchestrator that wires the previously-orphaned
FRONTEND.components.schema_mapping Universal Component into the
application's navigation, per the platform's Layer 1 mandate that role
mappings must be user-confirmable/overridable via a dedicated Schema
Mapping Studio surface. Owns zero business logic — delegates entirely to
FRONTEND.components.schema_mapping.render(), matching the presentation-
layer-only pattern established by pages/1_dashboard.py and
pages/2_audit.py.
"""
from __future__ import annotations

import streamlit as st

from core.settings import APP_ICON
from FRONTEND.components import schema_mapping as schema_mapping_component

st.markdown(f"# {APP_ICON} Schema Mapping Studio")
st.markdown(
    '<p style="font-weight:300; color:#64748B; font-size:0.95rem; margin-top:-8px;">'
    "Bind raw column headers to canonical KESCO operational roles — the single source of "
    "truth every KPI, chart, and executive narrative in this workspace resolves through. "
    "No analytics function anywhere in this platform ever references a literal column name.</p>",
    unsafe_allow_html=True,
)

schema_mapping_component.render()