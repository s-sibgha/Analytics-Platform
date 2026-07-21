"""
core/roles.py

Canonical business roles that the Column Registry resolves columns into.
This is the single vocabulary every engine and chart in the platform is
parameterized against — analytics code never references a literal column
name, only a role from this list.

Synonym lists feed the fuzzy header-matching layer in
utils/fuzzy_match.py so that headers like 'reg_date', 'comp_dt',
'date_registered' all auto-suggest the ROLE_REGISTRATION_DATE role.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

# ── Canonical role identifiers ──────────────────────────────────────────
ROLE_RECORD_ID = "record_id"
ROLE_REGISTRATION_DATE = "registration_date"
ROLE_CLOSING_DATE = "closing_date"
ROLE_STATUS = "status"
ROLE_CATEGORY = "category"
ROLE_SUBCATEGORY = "subcategory"
ROLE_OFFICER = "officer"
ROLE_CONSUMER_ID = "consumer_id"
ROLE_CONSUMER_NAME = "consumer_name"
ROLE_ZONE = "zone"
ROLE_CIRCLE = "circle"
ROLE_DIVISION = "division"
ROLE_SUBDIVISION = "subdivision"
ROLE_FEEDER = "feeder"
ROLE_TRANSFORMER = "transformer"
ROLE_AMOUNT = "amount"
ROLE_TARGET_AMOUNT = "target_amount"
ROLE_COLLECTED_AMOUNT = "collected_amount"
ROLE_UNITS_CONSUMED = "units_consumed"
ROLE_SLA_DEADLINE = "sla_deadline"
ROLE_REOPEN_FLAG = "reopen_flag"
ROLE_PRIORITY = "priority"
ROLE_LATITUDE = "latitude"
ROLE_LONGITUDE = "longitude"
ROLE_EMPLOYEE_ID = "employee_id"
ROLE_DEPARTMENT = "department"
ROLE_ATTENDANCE_STATUS = "attendance_status"
ROLE_ASSET_ID = "asset_id"
ROLE_ASSET_HEALTH = "asset_health"
# NEW — Milestone 1 / Issue 2 & 7 remediation. Previously ROLE_TRANSFORMER
# was being overloaded to represent two conceptually distinct physical
# tiers (grid Substation vs. distribution Transformer/DTR), causing
# engine/analytics.py to relabel it "Substation" while filters.py labeled
# the same role "Transformer" in a different hierarchical position. This
# first-class role separates the two tiers cleanly.
ROLE_SUBSTATION = "substation"

# Roles required for a dataset to be considered minimally analytics-ready.
CORE_REQUIRED_ROLES: List[str] = [ROLE_RECORD_ID]

# Synonym dictionary: role -> list of lowercase header fragments that
# fuzzy-match against incoming column headers.
ROLE_SYNONYMS: Dict[str, List[str]] = {
    ROLE_RECORD_ID: ["id", "complaint id", "ticket id", "case id", "ref no",
                      "reference number", "sr no", "serial"],
    ROLE_REGISTRATION_DATE: ["reg date", "reg_date", "comp dt", "comp_dt",
                              "date registered", "registration date",
                              "created date", "open date", "lodged date",
                              "complaint date", "raised on"],
    ROLE_CLOSING_DATE: ["closing date", "close date", "closed on",
                         "resolution date", "resolved date", "completion date"],
    ROLE_STATUS: ["status", "case status", "current status", "stage"],
    ROLE_CATEGORY: ["category", "type", "complaint type", "issue type",
                     "nature of complaint"],
    ROLE_SUBCATEGORY: ["subcategory", "sub category", "sub-type"],
    ROLE_OFFICER: ["officer", "assigned to", "handled by", "engineer",
                    "jen", "sdo", "executive engineer", "linesman"],
    ROLE_CONSUMER_ID: ["consumer id", "account number", "ca number",
                        "consumer no", "meter id"],
    ROLE_CONSUMER_NAME: ["consumer name", "customer name", "name"],
    ROLE_ZONE: ["zone"],
    ROLE_CIRCLE: ["circle"],
    ROLE_DIVISION: ["division"],
    ROLE_SUBDIVISION: ["subdivision", "sub division"],
    ROLE_FEEDER: ["feeder"],
    ROLE_TRANSFORMER: ["transformer", "dtr"],
    ROLE_AMOUNT: ["amount", "bill amount", "value", "revenue"],
    ROLE_TARGET_AMOUNT: ["target", "target amount", "demand"],
    ROLE_COLLECTED_AMOUNT: ["collected", "collection amount", "amount paid",
                             "recovered amount"],
    ROLE_UNITS_CONSUMED: ["units", "units consumed", "kwh", "consumption"],
    ROLE_SLA_DEADLINE: ["sla", "sla date", "due date", "deadline"],
    ROLE_REOPEN_FLAG: ["reopen", "reopened", "re-open", "repeat complaint"],
    ROLE_PRIORITY: ["priority", "severity", "urgency"],
    ROLE_LATITUDE: ["lat", "latitude"],
    ROLE_LONGITUDE: ["lon", "lng", "longitude"],
    ROLE_EMPLOYEE_ID: ["employee id", "emp id", "staff id"],
    ROLE_DEPARTMENT: ["department", "dept"],
    ROLE_ATTENDANCE_STATUS: ["attendance", "present", "attendance status"],
    ROLE_ASSET_ID: ["asset id", "equipment id", "transformer id"],
    ROLE_ASSET_HEALTH: ["health", "asset health", "condition"],
    ROLE_SUBSTATION: ["substation", "grid substation", "power substation",
                       "ss name", "ss id", "33kv substation", "11kv substation",
                       "substation name", "substation id"],
}

# Friendly display labels shown in the Schema Mapping Studio.
ROLE_DISPLAY_NAMES: Dict[str, str] = {
    ROLE_RECORD_ID: "Record / Case ID",
    ROLE_REGISTRATION_DATE: "Registration / Open Date",
    ROLE_CLOSING_DATE: "Closing / Resolution Date",
    ROLE_STATUS: "Status",
    ROLE_CATEGORY: "Category",
    ROLE_SUBCATEGORY: "Subcategory",
    ROLE_OFFICER: "Responsible Officer",
    ROLE_CONSUMER_ID: "Consumer ID",
    ROLE_CONSUMER_NAME: "Consumer Name",
    ROLE_ZONE: "Zone",
    ROLE_CIRCLE: "Circle",
    ROLE_DIVISION: "Division",
    ROLE_SUBDIVISION: "Subdivision",
    ROLE_FEEDER: "Feeder",
    ROLE_TRANSFORMER: "Transformer",
    ROLE_AMOUNT: "Amount",
    ROLE_TARGET_AMOUNT: "Target Amount",
    ROLE_COLLECTED_AMOUNT: "Collected Amount",
    ROLE_UNITS_CONSUMED: "Units Consumed",
    ROLE_SLA_DEADLINE: "SLA Deadline",
    ROLE_REOPEN_FLAG: "Reopen Flag",
    ROLE_PRIORITY: "Priority",
    ROLE_LATITUDE: "Latitude",
    ROLE_LONGITUDE: "Longitude",
    ROLE_EMPLOYEE_ID: "Employee ID",
    ROLE_DEPARTMENT: "Department",
    ROLE_ATTENDANCE_STATUS: "Attendance Status",
    ROLE_ASSET_ID: "Asset ID",
    ROLE_ASSET_HEALTH: "Asset Health",
    ROLE_SUBSTATION: "Substation",
}


# Status-value canonicalization (raw text -> canonical bucket).
STATUS_SYNONYMS: Dict[str, List[str]] = {
    "CLOSED": ["closed", "resolved", "completed", "done"],
    "PENDING": ["pending", "open", "in progress", "under process", "wip"],
    "REOPENED": ["reopened", "re-opened", "repeat", "escalated"],
}

# ── Canonical Geographic/Asset Hierarchy (Milestone 1 / Issue 15) ──────────
# SINGLE SOURCE OF TRUTH for hierarchical ordering across the entire
# platform. Prior to this change, four independent, mutually-inconsistent
# hierarchy definitions existed (filters.py, schema_mapping.py,
# engine/analytics.py, visualization/chart_factory.py). Every module that
# needs ordered geo/asset hierarchy roles must import and reference this
# constant rather than declaring a local tuple.
#
# Physical/administrative ordering: Zone (largest) -> Circle -> Division ->
# Subdivision -> Substation (grid infrastructure) -> Feeder -> Transformer
# (distribution/DTR, smallest asset tier before the consumer).
CANONICAL_GEO_HIERARCHY: Tuple[Tuple[str, str], ...] = (
    (ROLE_ZONE, "Zone"),
    (ROLE_CIRCLE, "Circle"),
    (ROLE_DIVISION, "Division"),
    (ROLE_SUBDIVISION, "Subdivision"),
    (ROLE_SUBSTATION, "Substation"),
    (ROLE_FEEDER, "Feeder"),
    (ROLE_TRANSFORMER, "Transformer"),
)

# Extends CANONICAL_GEO_HIERARCHY with the terminal business-entity tier
# (Consumer) for use by components that drill all the way to the
# individual-account level (e.g. the Cascading Drill Panel).
CANONICAL_DRILL_HIERARCHY: Tuple[Tuple[str, str], ...] = CANONICAL_GEO_HIERARCHY + (
    (ROLE_CONSUMER_ID, "Consumer"),
)