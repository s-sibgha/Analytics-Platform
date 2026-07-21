"""
core/schema_models.py

Typed data contracts shared between the Type Inference Engine, the Column
Registry, the Schema Mapping Studio UI, and every downstream analytics
module. Keeping these as explicit dataclasses (rather than passing raw
dicts around) is what lets the Universal Analytics Engine stay
parameterized and type-safe without hardcoding column names.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class InferredType(str, Enum):
    ID = "id"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    NUMERIC = "numeric"
    CURRENCY = "currency"
    PERCENTAGE = "percentage"
    CATEGORICAL = "categorical"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass
class ColumnProfile:
    """Result of running the Type Inference Engine on a single column."""
    original_name: str
    inferred_type: InferredType
    confidence: float                      # 0.0 - 1.0
    sample_values: List[Any] = field(default_factory=list)
    null_count: int = 0
    null_pct: float = 0.0
    distinct_count: int = 0
    needs_manual_review: bool = False
    detection_notes: List[str] = field(default_factory=list)
    suggested_roles: List[str] = field(default_factory=list)  # ranked, best first
    suggested_role_scores: Dict[str, float] = field(default_factory=dict)

    def best_suggested_role(self) -> Optional[str]:
        return self.suggested_roles[0] if self.suggested_roles else None


@dataclass
class RoleMapping:
    """A single confirmed (or pending) role -> column binding."""
    role: str
    column_name: Optional[str]             # None until user/auto resolves it
    confidence: float = 0.0
    source: str = "unmapped"               # "auto" | "manual" | "unmapped"
    confirmed: bool = False


@dataclass
class ColumnRegistrySnapshot:
    """Versioned, immutable-ish snapshot of the full registry state,
    persisted into a workspace's mapping profile."""
    workspace_name: str
    version: int
    created_at: datetime
    mappings: Dict[str, RoleMapping]
    value_canonicalization: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # e.g. {"status": {"closed": "CLOSED", "resolved": "CLOSED", ...}}

    def resolved_roles(self) -> List[str]:
        return [r for r, m in self.mappings.items() if m.column_name]


@dataclass
class AuditEntry:
    timestamp: datetime
    action_type: str          # "cleaning" | "mapping" | "filter" | "export"
    description: str
    rows_affected: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)