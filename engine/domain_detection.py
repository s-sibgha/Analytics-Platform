"""
engine/domain_detection.py

Classifies an ingested dataset into a business domain using a mix of
filename hints and which roles ended up resolvable in the Column Registry.
Drives dashboard naming, KPI library selection, and navigation
(context-aware nav requirement).
"""
from __future__ import annotations
from typing import Dict, List, Tuple

from core.column_registry import ColumnRegistry

DOMAIN_COMPLAINT = "Complaint Management"
DOMAIN_REVENUE = "Revenue Analytics"
DOMAIN_SUPPLY = "Supply Operations"
DOMAIN_HR = "HR Analytics"
DOMAIN_ASSET = "Asset Management"
DOMAIN_UNKNOWN = "General / Custom Dataset"

# role-sets that, if mostly present, strongly indicate a domain
# AFTER
_DOMAIN_SIGNATURES: Dict[str, List[str]] = {
    DOMAIN_COMPLAINT: ["status", "registration_date", "officer", "reopen_flag"],
    DOMAIN_REVENUE: ["amount", "target_amount", "collected_amount"],
    DOMAIN_SUPPLY: ["feeder", "transformer", "units_consumed", "substation"],
    DOMAIN_HR: ["employee_id", "department", "attendance_status"],
    DOMAIN_ASSET: ["asset_id", "asset_health"],
}

_FILENAME_HINTS: Dict[str, List[str]] = {
    DOMAIN_COMPLAINT: ["complaint", "grievance", "ticket"],
    DOMAIN_REVENUE: ["revenue", "billing", "collection", "demand"],
    DOMAIN_SUPPLY: ["feeder", "supply", "outage", "load"],
    DOMAIN_HR: ["hr", "employee", "staff", "attendance"],
    DOMAIN_ASSET: ["asset", "transformer", "equipment"],
}


def detect_domain(registry: ColumnRegistry, filename: str = "") -> Tuple[str, float]:
    """Returns (domain_label, confidence 0-1). Never raises."""
    try:
        filename_l = (filename or "").lower()
        scores: Dict[str, float] = {d: 0.0 for d in _DOMAIN_SIGNATURES}

        for domain, roles in _DOMAIN_SIGNATURES.items():
            present = sum(1 for r in roles if registry.has_role(r))
            scores[domain] += present / max(len(roles), 1)

        for domain, hints in _FILENAME_HINTS.items():
            if any(h in filename_l for h in hints):
                scores[domain] += 0.5

        best_domain = max(scores, key=lambda d: scores[d])
        best_score = scores[best_domain]
        if best_score <= 0.0:
            return DOMAIN_UNKNOWN, 0.0
        confidence = min(1.0, best_score / 1.5)
        return best_domain, round(confidence, 2)
    except Exception:
        return DOMAIN_UNKNOWN, 0.0