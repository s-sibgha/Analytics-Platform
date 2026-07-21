"""
core/column_registry.py — ★ FOUNDATION ENGINE ★

The single source of truth that every analytics function, chart, and KPI
resolves columns through. NOTHING downstream may reference a literal
DataFrame column name — it asks the registry for a role, and the registry
tells it which real column (if any) currently satisfies that role.

This makes the entire platform dataset-agnostic: swap in HR data instead
of Complaint data and the same KPI functions either find the roles they
need (and run) or don't (and are cleanly disabled by the Eligibility
Engine) — never a KeyError, never a guess.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from core.roles import ROLE_DISPLAY_NAMES, ROLE_SYNONYMS, STATUS_SYNONYMS
from core.settings import MIN_FUZZY_ROLE_SCORE
from core.schema_models import (
    ColumnProfile,
    ColumnRegistrySnapshot,
    RoleMapping,
)
from core.fuzzy_match import suggest_roles_for_header


class ColumnRegistry:
    """
    Holds the live role -> column mapping for the currently active
    dataset/workspace. Designed to be instantiated once per workspace and
    stored in st.session_state (see core/session_state.py) so it survives
    Streamlit re-runs.
    """

    def __init__(self, workspace_name: str = "default") -> None:
        self.workspace_name = workspace_name
        self.version: int = 1
        self.mappings: Dict[str, RoleMapping] = {}
        self.value_canonicalization: Dict[str, Dict[str, str]] = {
            "status": self._build_status_canonical_map()
        }
        self._column_profiles: Dict[str, ColumnProfile] = {}

    # ── Construction / bootstrap ─────────────────────────────────────────

    @staticmethod
    def _build_status_canonical_map() -> Dict[str, str]:
        canon: Dict[str, str] = {}
        for bucket, synonyms in STATUS_SYNONYMS.items():
            for syn in synonyms:
                canon[syn.lower()] = bucket
        return canon

    def bootstrap_from_profiles(self, profiles: List[ColumnProfile]) -> None:
        """
        Auto-populate role mappings from Type Inference Engine output.
        Only auto-confirms a mapping when there is a single, sufficiently
        confident, non-conflicting suggestion. Anything ambiguous is left
        unmapped/unconfirmed for the Schema Mapping Studio to resolve.
        """
        self._column_profiles = {p.original_name: p for p in profiles}
        claimed_roles: Dict[str, str] = {}  # role -> column already claiming it

        # Sort by confidence descending so the strongest matches win ties.
        sorted_profiles = sorted(
            profiles,
            key=lambda p: max(p.suggested_role_scores.values(), default=0),
            reverse=True,
        )

        for profile in sorted_profiles:
            # role = profile.best_suggested_role()
            # if not role or role in claimed_roles:
            #     continue
            # score = profile.suggested_role_scores.get(role, 0.0)
            # auto_confirm = (
            #     score >= MIN_FUZZY_ROLE_SCORE
            #     and not profile.needs_manual_review
            # )
            # self.mappings[role] = RoleMapping(
            #     role=role,
            #     column_name=profile.original_name,
            #     confidence=score / 100.0,
            #     source="auto" if auto_confirm else "auto_low_confidence",
            #     confirmed=auto_confirm,
            # )
            role: Optional[str] = None
            score: float = 0.0
            for candidate_role in profile.suggested_roles:
                if candidate_role in claimed_roles:
                    continue
                role = candidate_role
                score = profile.suggested_role_scores.get(candidate_role, 0.0)
                break

            if not role:
                continue

            auto_confirm = (
                score >= MIN_FUZZY_ROLE_SCORE
                and not profile.needs_manual_review
            )
            self.mappings[role] = RoleMapping(
                role=role,
                column_name=profile.original_name,
                confidence=score / 100.0,
                source="auto" if auto_confirm else "auto_low_confidence",
                confirmed=auto_confirm,
            )
            claimed_roles[role] = profile.original_name

    # ── Core resolution API (the only thing downstream code should call) ──

    def resolve(self, role: str) -> Optional[str]:
        """Return the real column name bound to `role`, or None if unmapped."""
        mapping = self.mappings.get(role)
        if mapping and mapping.confirmed and mapping.column_name:
            return mapping.column_name
        return None

    def resolve_series(self, role: str, df: pd.DataFrame) -> Optional[pd.Series]:
        """Resolve a role straight to its pandas Series within `df`."""
        col = self.resolve(role)
        if col is None or col not in df.columns:
            return None
        return df[col]

    def has_role(self, role: str) -> bool:
        return self.resolve(role) is not None

    def has_all_roles(self, roles: List[str]) -> bool:
        return all(self.has_role(r) for r in roles)

    def missing_roles(self, roles: List[str]) -> List[str]:
        return [r for r in roles if not self.has_role(r)]

    def canonicalize_value(self, role: str, raw_value: Optional[str]) -> Optional[str]:
        """Map a raw cell value (e.g. 'Resolved') to its canonical bucket
        (e.g. 'CLOSED') using the role's value-canonicalization table."""
        if raw_value is None:
            return None
        table = self.value_canonicalization.get(role, {})
        return table.get(str(raw_value).strip().lower(), str(raw_value).strip())

    # ── Mutation API (used by Schema Mapping Studio) ───────────────────────

    def set_mapping(self, role: str, column_name: Optional[str], *, manual: bool = True) -> None:
        self.mappings[role] = RoleMapping(
            role=role,
            column_name=column_name,
            confidence=1.0 if manual else self.mappings.get(role, RoleMapping(role, None)).confidence,
            source="manual" if manual else "auto",
            confirmed=column_name is not None,
        )
        self.version += 1

    def clear_mapping(self, role: str) -> None:
        self.mappings.pop(role, None)
        self.version += 1

    def display_name(self, role: str) -> str:
        return ROLE_DISPLAY_NAMES.get(role, role.replace("_", " ").title())

    def suggestions_for_unmapped(self, headers: List[str]) -> Dict[str, List[str]]:
        """For headers not yet claimed by any confirmed mapping, return
        candidate roles to show in the manual-review UI block."""
        claimed_cols = {m.column_name for m in self.mappings.values() if m.confirmed}
        out: Dict[str, List[str]] = {}
        for h in headers:
            if h in claimed_cols:
                continue
            ranked = suggest_roles_for_header(h, ROLE_SYNONYMS, min_score=40)
            out[h] = [r for r, _ in ranked]
        return out

    # ── Persistence ─────────────────────────────────────────────────────

  
    def snapshot(self) -> ColumnRegistrySnapshot:
        return ColumnRegistrySnapshot(
            workspace_name=self.workspace_name,
            version=self.version,
            created_at=datetime.now(timezone.utc),
            mappings=dict(self.mappings),
            value_canonicalization=dict(self.value_canonicalization),
        )


    def restore(self, snapshot: ColumnRegistrySnapshot) -> None:
        self.workspace_name = snapshot.workspace_name
        self.version = snapshot.version
        self.mappings = dict(snapshot.mappings)
        self.value_canonicalization = dict(snapshot.value_canonicalization)

    def summary_table(self) -> List[Dict[str, str]]:
        """Flat list for rendering in the Schema Mapping Studio / Metadata
        Explorer tables."""
        rows = []
        for role, mapping in sorted(self.mappings.items()):
            rows.append({
                "Role": self.display_name(role),
                "Resolved Column": mapping.column_name or "— Not Mapped —",
                "Source": mapping.source,
                "Confidence": f"{mapping.confidence * 100:.0f}%",
                "Confirmed": "Yes" if mapping.confirmed else "No",
            })
        return rows