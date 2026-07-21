"""
core/settings.py — App-wide constants and thresholds.
"""
from __future__ import annotations

APP_TITLE = "KESCO Enterprise Analytics Platform"
APP_ICON = "⚡" 

# Execution substrate switch: above this row count, prefer DuckDB over pandas.
DUCKDB_ROW_THRESHOLD = 0

# Type inference confidence band that triggers manual review routing.
MIN_AUTO_CONFIDENCE = 0.75

# Fuzzy role-matching minimum score (0-100, rapidfuzz/difflib scale) to
# auto-suggest a role in the Schema Mapping Studio.
MIN_FUZZY_ROLE_SCORE = 70

# Null placeholder tokens normalized to NaN during safe cleaning.
NULL_PLACEHOLDERS = ["NA", "N/A", "-", "--", "NULL", "null", "None", "none",
                      "nil", "NIL", "#N/A", ""]

# Supported upload formats.
SUPPORTED_UPLOAD_TYPES = ["csv", "xlsx", "xls"]

# Analytics Readiness Score bands.
READINESS_BANDS = [
    (90, 100, "Excellent"),
    (75, 89, "Good"),
    (55, 74, "Fair"),
    (35, 54, "Poor"),
    (0, 34, "Critical"),
]

MAX_PREVIEW_ROWS = 200
from typing import Dict

# ── Milestone 5 / Issue 13 remediation — Corporate Branding Asset Filenames ──
APP_LOGO_FILENAME: str = "kesco_logo.png"
APP_ICON_LOGO_FILENAME: str = "kesco_icon.png"

# ── Milestone 5 / Issue 14 remediation — Enterprise Copy/Terminology Map ──
ENTERPRISE_COPY_MAP: Dict[str, str] = {
    "Detecting Schema Topology, Resolving Domain Dictionaries, "
    "Assembling KESCO Grid Reports...":
        "Processing dataset: inferring schema, resolving business domain, "
        "and preparing analytics tables...",
    "Detecting Schema Topology and Resolving Domain Dictionaries...":
        "Processing dataset: inferring schema and resolving business domain...",
    "Resolving Fuzzy Header Dictionaries, Ranking Candidate Columns...":
        "Matching column headers to candidate roles...",
    "Friction Vector Risk Matrix":
        "Risk & Backlog Analysis",
    "Friction Vector Risk & Root-Cause Analysis":
        "Risk & Root-Cause Analysis",
    "Core Operational Nerve Center":
        "Core Operational Overview",
    "Chronological Trends Dashboard":
        "Trend Analysis",
    "Ad-Hoc Self Service & Network Flow Layouts":
        "Self-Service Analytics & Flow Diagrams",
    "Automated Data Sanitization Layer & Audit Trace Logs":
        "Data Cleaning & Audit Trace Logs",
}

# NOTE — Milestone 11 / Presentation Mode removal: PRESENTATION_MODE_SESSION_KEY
# and PRESENTATION_MODE_HIDDEN_TOGGLE_LABEL have been intentionally removed.
# The platform now relies exclusively on Streamlit's native sidebar collapse
# control and browser zoom for presentation-style viewing. Do not reintroduce
# these constants without explicit instruction — main.js, app.py, and
# sidebar.py have all had every reference to them removed in lockstep.