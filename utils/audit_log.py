"""utils/audit_log.py — Append-only audit trail."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from core.schema_models import AuditEntry


class AuditLog:
    def __init__(self) -> None:
        self._entries: List[AuditEntry] = []


    def log(self, action_type: str, description: str,
             rows_affected: Optional[int] = None,
             details: Optional[Dict[str, Any]] = None) -> None:
        self._entries.append(AuditEntry(
            timestamp=datetime.now(timezone.utc),
            action_type=action_type,
            description=description,
            rows_affected=rows_affected,
            details=details or {},
        ))

    def entries(self) -> List[AuditEntry]:
        return list(reversed(self._entries))  # most recent first
        
    
    # AFTER / FIXED CODE
    def as_table(self) -> List[Dict[str, Any]]:
        return [
            {
                "Timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "Type": e.action_type,
                "Description": e.description,
                "Rows Affected": e.rows_affected if e.rows_affected is not None else "—",
                # NEW — additive column surfacing the previously-dropped
                # AuditEntry.details payload (e.g. step name, column,
                # thresholds) that engine.cleaner already computes.
                "Details": "; ".join(f"{k}={v}" for k, v in e.details.items()) if e.details else "—",
            }
            for e in self.entries()
        ]