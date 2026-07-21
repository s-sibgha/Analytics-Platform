"""
FRONTEND/pages/2_audit.py — KESCO System Diagnostics, Data Quality &
Audit Telemetry (FILE 6 / 6)

Presentation-Layer Orchestrator ONLY. This module owns zero business logic,
zero data cleaning, zero aggregation, and zero scoring — every metric,
table row, and recommendation rendered below is read verbatim from
already-populated st.session_state artifacts produced by
FRONTEND.app.process_uploaded_file's Ingestion -> Type Inference ->
Registry Bootstrap -> Domain Detection -> Safe Cleaning pipeline
(core.type_inference, engine.cleaner.SafeCleaningEngine,
core.column_registry.ColumnRegistry, engine.domain_detection).

Verified backend API surface actually exposed to this page via
session_state (Pre-Flight validation):
    - st.session_state["audit_results"]:
        cleaning_summary: Dict[str, int]                (engine.cleaner.SafeCleaningEngine._summary)
        audit_entries: List[Dict[str, Any]]              (utils.audit_log.AuditLog.as_table() schema,
                                                            also appended to by
                                                            FRONTEND.components.schema_mapping)
        flagged_business_key_duplicates: int              (count only — the underlying
                                                            FlaggedDuplicateGroup row objects are not
                                                            persisted across reruns, so only the count
                                                            is surfaced here; no detail table is
                                                            fabricated)
        flagged_outlier_summaries: int                     (count only, same constraint as above)
        rows_original / rows_cleaned / rows_removed: int
        data_quality_score: int
    - st.session_state["column_profiles"]: List[core.schema_models.ColumnProfile]
        (original_name, inferred_type, confidence, null_count, null_pct,
        distinct_count, needs_manual_review, detection_notes)
    - st.session_state["column_registry"]: core.column_registry.ColumnRegistry
        (.summary_table(), .mappings)
    - st.session_state["domain_detection"]: Tuple[str, float]
    - st.session_state["readiness_score"] / ["readiness_band"] /
      ["readiness_recommendations"]
    - st.session_state["notifications"]: List[Dict[str, str]]
    - st.session_state["uploaded_dataframe"] / ["cleaned_dataframe"] /
      ["analytics_ready_dataframe"]: pd.DataFrame

No "Processing Log" object is exposed anywhere in the backend beyond the
flattened audit_entries table, so no separate Processing Log section is
fabricated — the Audit Trace Log section below IS the processing log,
sourced from the same data. No per-row duplicate/outlier detail tables are
rendered since only summary counts are persisted in session_state; the
underlying row-level `FlaggedDuplicateGroup` / `OutlierSummary` dataclasses
from engine.cleaner are never re-derived here (that would duplicate
business logic explicitly forbidden by the Zero Logic mandate).
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from core.column_registry import ColumnRegistry
from core.schema_models import ColumnProfile
from core.settings import APP_ICON, MIN_AUTO_CONFIDENCE
from utils.grid_utils import render_enterprise_grid
from core.themes import THEMES, DEFAULT_THEME_KEY, Theme
from engine.domain_detection import DOMAIN_UNKNOWN
from utils.error_logging import log_exception

_PAGE_NAME: str = "audit"

# ── Presentation-only label map for engine.cleaner.SafeCleaningEngine's
# internal _summary counter keys. Purely cosmetic string formatting — the
# counters themselves are never recomputed here, only relabeled for
# display. ──────────────────────────────────────────────────────────────
_CLEANING_SUMMARY_LABELS: Dict[str, str] = {
    "rows_removed_exact_duplicates": "Exact Duplicate Rows Removed",
    "rows_removed_fully_empty": "Fully-Empty Rows Removed",
    "values_null_normalized": "Placeholder-Null Values Normalized",
    "values_whitespace_trimmed": "Whitespace-Trimmed Cell Values",
    "columns_dtype_converted": "Columns Converted to Numeric",
    "date_columns_standardized": "Date Columns Standardized",
    "business_key_duplicate_groups": "Business-Key Duplicate Groups Flagged",
    "outlier_columns_flagged": "Numeric Columns With Outliers Flagged",
}

_SEVERITY_BADGE_MAP: Dict[str, str] = {
    "critical": "kesco-badge-critical",
    "warning": "kesco-badge-warning",
    "info": "kesco-badge-info",
}


# ══════════════════════════════════════════════════════════════════════════════
# CSS — MIRRORS 1_dashboard.py's GLASSMORPHIC / ENTERPRISE DESIGN LANGUAGE
# ══════════════════════════════════════════════════════════════════════════════

def _inject_audit_css(theme_key: str) -> None:
    """
    Injects the audit-page-specific styling layer, replicating the exact
    glassmorphic container language, spacing scale, and badge/accent
    conventions established in FRONTEND/pages/1_dashboard.py so this page
    is visually indistinguishable from the rest of the platform. Resolved
    entirely from core.themes.THEMES. Never raises.
    """
    theme: Theme = THEMES.get(theme_key, THEMES[DEFAULT_THEME_KEY])
    css = f"""
    <style>
    /* 1. DESIGN TOKENS & ROOT VARIABLES */
    :root {{
        --kesco-audit-spacing-xs: 4px;
        --kesco-audit-spacing-sm: 8px;
        --kesco-audit-spacing-md: 16px;
        --kesco-audit-spacing-lg: 24px;
        --kesco-audit-radius: 6px;
        --kesco-audit-primary: {theme['primary']};
        --kesco-audit-secondary: {theme['secondary']};
        --kesco-audit-surface: {theme['surface']};
        --kesco-audit-text: {theme['text']};
        --kesco-audit-success: {theme['success']};
        --kesco-audit-warning: {theme['warning']};
        --kesco-audit-danger: {theme['danger']};
    }}

    /* 2. LAYOUT CONTAINERS & GLASSMORPHISM SPECIFICATION */
    .kesco-quality-banner {{
        display: flex;
        flex-wrap: wrap;
        gap: var(--kesco-audit-spacing-lg);
        background-color: {theme['surface']};
        backdrop-filter: blur(10px);
        border: 1px solid rgba(100,116,139,0.14);
        border-radius: var(--kesco-audit-radius);
        padding: var(--kesco-audit-spacing-md) var(--kesco-audit-spacing-lg);
        margin-bottom: var(--kesco-audit-spacing-lg);
        box-shadow: 0 2px 10px rgba(15,23,42,0.06);
    }}
    .kesco-quality-item {{ display: flex; flex-direction: column; min-width: 130px; }}
    .kesco-quality-value {{ font-size: 1.15rem; font-weight: 700; color: var(--kesco-audit-text); }}
    .kesco-quality-label {{ font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.03em; color: var(--kesco-audit-secondary); }}

    /* 3. EXECUTIVE ALERTS & NOTIFICATION INFRASTRUCTURE */
    .kesco-alert-card {{
        background-color: {theme['surface']};
        border-radius: var(--kesco-audit-radius);
        border: 1px solid rgba(100,116,139,0.12);
        border-left: 3px solid var(--kesco-audit-secondary);
        padding: var(--kesco-audit-spacing-sm) var(--kesco-audit-spacing-md);
        margin-bottom: var(--kesco-audit-spacing-sm);
        font-size: 0.82rem;
        color: {theme['text']};
    }}
    .kesco-alert-critical {{ border-left-color: var(--kesco-audit-danger); }}
    .kesco-alert-warning {{ border-left-color: var(--kesco-audit-warning); }}
    .kesco-alert-info {{ border-left-color: var(--kesco-audit-secondary); }}
    .kesco-alert-timestamp {{ font-size: 0.68rem; color: var(--kesco-audit-secondary); }}

    /* 4. PREMIUM KPI CARDS & INTEGRITY HEALTH GEOMETRY */
    .kesco-kpi-card {{
        background-color: {theme['surface']};
        backdrop-filter: blur(8px);
        border: 1px solid rgba(100,116,139,0.14);
        border-left-width: 4px;
        border-left-style: solid;
        border-radius: var(--kesco-audit-radius);
        padding: var(--kesco-audit-spacing-md);
        margin-bottom: var(--kesco-audit-spacing-sm);
        box-shadow: 0 1px 4px rgba(15,23,42,0.06);
        transition: box-shadow 200ms ease-out, transform 200ms ease-out;
    }}
    .kesco-kpi-card:hover {{ box-shadow: 0 6px 18px rgba(15,23,42,0.10); transform: translateY(-1px); }}
    .kesco-kpi-card.kesco-accent-success {{ border-left-color: var(--kesco-audit-success); }}
    .kesco-kpi-card.kesco-accent-warning {{ border-left-color: var(--kesco-audit-warning); }}
    .kesco-kpi-card.kesco-accent-danger {{ border-left-color: var(--kesco-audit-danger); }}
    .kesco-kpi-card.kesco-accent-secondary {{ border-left-color: var(--kesco-audit-secondary); }}
    .kesco-kpi-label {{ font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.02em; color: var(--kesco-audit-secondary); margin-bottom: 4px; }}
    .kesco-kpi-value {{ font-size: 1.55rem; font-weight: 700; color: var(--kesco-audit-text); line-height: 1.15; }}
    .kesco-kpi-badge {{ margin-top: 6px; display: inline-block; }}

    /* 5. VISUAL CONTAINERS & DATA EDITOR OVERRIDES */
    .kesco-narrative-container {{
        background-color: {theme['surface']};
        border-radius: var(--kesco-audit-radius);
        border: 1px solid rgba(100,116,139,0.12);
        padding: var(--kesco-audit-spacing-md);
        height: 100%;
        color: {theme['text']};
    }}
    div[data-testid="stDataFrameResizable"], div[data-testid="stDataEditor"] {{
        border-radius: var(--kesco-audit-radius);
        overflow: hidden;
        border: 1px solid rgba(100,116,139,0.12);
    }}

    /* 6. UTILITY MICROINTERACTIONS & RESPONSIVE RULES */
    div[data-testid="stButton"] > button {{
        transition: box-shadow 180ms ease-out, transform 180ms ease-out;
    }}
    div[data-testid="stButton"] > button:hover {{ transform: translateY(-1px); }}
    @media (max-width: 900px) {{
        .kesco-quality-banner {{ flex-direction: column; gap: var(--kesco-audit-spacing-sm); }}
    }}

    /* 7. ENTERPRISE ACCESSIBILITY & WCAG CONTRAST FRAMEWORKS */
    .kesco-kpi-value, .kesco-quality-value {{ color: {theme['text']}; }}
    :focus-visible {{ outline: 2px solid {theme['primary']}; outline-offset: 2px; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def _init_audit_page_state() -> None:
    """Idempotently seeds this page's local UI-only session_state keys
    (table search text, review-only toggle). Never disturbs the
    platform-wide contract owned by FRONTEND.app. Never raises."""
    st.session_state.setdefault("_audit_trace_search_text", "")
    st.session_state.setdefault("_audit_review_only_toggle", False)


# ══════════════════════════════════════════════════════════════════════════════
# EMPTY STATE (Data Integrity Gate failure path)
# ══════════════════════════════════════════════════════════════════════════════

def _render_empty_state() -> None:
    """Renders the Enterprise Empty State when no active dataset/audit
    trail exists, with explicit corrective pipeline actions rather than a
    broken or blank screen. Never raises."""
    st.markdown(f"# {APP_ICON} KESCO System Diagnostics & Audit Telemetry")
    st.markdown(
        '<p style="font-weight:300; color:#64748B; font-size:0.95rem; margin-top:-8px;">'
        "Enterprise Data Governance Console for Ingestion Integrity, Schema Confidence, "
        "Safe Cleaning Telemetry &amp; Analytics Readiness Verification.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="kesco-card" style="text-align:center; padding: 48px 24px;">
            <h3>No Audit Trail Available</h3>
            <p>This workspace has no ingested dataset to audit yet. Once a file is uploaded via
            the sidebar's <strong>Unified Ingestion</strong> panel, the Ingestion → Type Inference →
            Registry Bootstrap → Domain Detection → Safe Cleaning pipeline will populate this page
            with a complete integrity report, schema confidence breakdown, and chronological audit
            trace log.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 1 — AUDIT HEADER
# ══════════════════════════════════════════════════════════════════════════════

def _render_audit_header() -> None:
    """Renders the Audit Header: fixed KESCO branding, executive-context
    subtitle, and a compact workspace/domain/ingestion-timestamp context
    line sourced directly from already-populated session_state. Never
    raises."""
    st.markdown(f"# {APP_ICON} KESCO System Diagnostics & Audit Telemetry")
    st.markdown(
        '<p style="font-weight:300; color:#64748B; font-size:0.95rem; margin-top:-8px;">'
        "Enterprise Data Governance Console for Ingestion Integrity, Schema Confidence, "
        "Safe Cleaning Telemetry &amp; Analytics Readiness Verification.</p>",
        unsafe_allow_html=True,
    )
    try:
        domain_label, domain_confidence = st.session_state.get("domain_detection", (DOMAIN_UNKNOWN, 0.0))
        workspace_name = st.session_state.get("workspace_name", "Default Workspace")
        active_filename = st.session_state.get("active_filename", "—")
        last_ingestion = st.session_state.get("last_ingestion_timestamp") or "—"
        st.caption(
            f"Workspace: **{workspace_name}** · Active File: **{active_filename}** · "
            f"Business Domain: **{domain_label}** ({domain_confidence:.0%} confidence) · "
            f"Last Ingested: **{last_ingestion}**"
        )
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Audit context could not be fully resolved: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 2 — INTEGRITY HEALTH SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def _render_integrity_health(
    audit_results: Dict[str, Any],
    readiness_score: int,
    readiness_band: str,
) -> None:
    """
    Renders the high-level Integrity Health summary row: Data Quality
    Score, Analytics Readiness Score, Rows Original -> Cleaned funnel,
    Rows Removed, Business-Key Duplicate Groups Flagged, and Outlier
    Columns Flagged — every value read verbatim from
    st.session_state["audit_results"] / ["readiness_score"] /
    ["readiness_band"]. Never raises.
    """
    st.markdown('<div class="kesco-section-title">Integrity Health Summary</div>', unsafe_allow_html=True)
    try:
        data_quality_score = int(audit_results.get("data_quality_score", 0))
        rows_original = int(audit_results.get("rows_original", 0))
        rows_cleaned = int(audit_results.get("rows_cleaned", 0))
        rows_removed = int(audit_results.get("rows_removed", 0))
        dup_groups_flagged = int(audit_results.get("flagged_business_key_duplicates", 0))
        outlier_cols_flagged = int(audit_results.get("flagged_outlier_summaries", 0))

        quality_accent = "success" if data_quality_score >= 75 else "warning" if data_quality_score >= 50 else "danger"
        readiness_accent = (
            "success" if readiness_band in ("Excellent", "Good")
            else "warning" if readiness_band == "Fair"
            else "danger"
        )
        removed_accent = "success" if rows_removed == 0 else "warning"
        dup_accent = "success" if dup_groups_flagged == 0 else "warning"
        outlier_accent = "success" if outlier_cols_flagged == 0 else "warning"

        cards: List[Dict[str, Any]] = [
            {
                "label": "Data Quality Score",
                "value": f"{data_quality_score}/100",
                "accent": quality_accent,
                "tooltip": "Derived from missing-value density, duplicate-row density, and the "
                           "proportion of columns requiring manual type-inference review.",
            },
            {
                "label": "Analytics Readiness",
                "value": f"{readiness_score}/100",
                "accent": readiness_accent,
                "tooltip": f"Band: {readiness_band}. Evaluates schema completeness, role mapping "
                           "completeness, data quality, required field availability, hierarchy "
                           "detection, and date availability.",
            },
            {
                "label": "Rows Removed (Safe Cleaning)",
                "value": f"{rows_removed:,} / {rows_original:,}",
                "accent": removed_accent,
                "tooltip": f"Original: {rows_original:,} rows → Cleaned: {rows_cleaned:,} rows. "
                           "Only exact-duplicate and fully-empty rows are ever auto-removed.",
            },
            {
                "label": "Business-Key Duplicate Groups",
                "value": f"{dup_groups_flagged:,}",
                "accent": dup_accent,
                "tooltip": "Duplicate Record IDs with differing values — flagged for review, "
                           "never auto-removed per non-destructive compliance policy.",
            },
        ]

        cols = st.columns(len(cards))
        for col, card in zip(cols, cards):
            with col:
                st.markdown(
                    f"""
                    <div class="kesco-kpi-card kesco-accent-{card['accent']}" title="{card['tooltip']}">
                        <div class="kesco-kpi-label">{card['label']}</div>
                        <div class="kesco-kpi-value">{card['value']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown(
            f"""
            <div class="kesco-kpi-card kesco-accent-{outlier_accent}" style="margin-top: 4px;"
                 title="Numeric columns (IQR×3.0 fence) containing statistical outliers — retained, never removed.">
                <div class="kesco-kpi-label">Numeric Columns With Outliers Flagged</div>
                <div class="kesco-kpi-value">{outlier_cols_flagged:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as exc:  # noqa: BLE001
        incident_id = log_exception("audit._render_integrity_health", exc)
        st.warning(f"Integrity Health summary could not be rendered (Reference ID: {incident_id}).")
        


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 3 — DEEP-DIVE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def _render_cleaning_summary_table(cleaning_summary: Dict[str, int]) -> None:
    """Renders the Safe Cleaning Engine's transformation counters
    (engine.cleaner.SafeCleaningEngine._summary) as a two-column metric
    table, relabeled for executive readability. Never raises."""
    st.markdown("#### Safe Cleaning Transformation Summary")
    try:
        if not cleaning_summary:
            st.caption("No cleaning summary is available for the active dataset.")
            return
        rows = [
            {"Cleaning Action": _CLEANING_SUMMARY_LABELS.get(key, key.replace("_", " ").title()),
             "Count": int(value)}
            for key, value in cleaning_summary.items()
        ]
        render_enterprise_grid(pd.DataFrame(rows), key="_audit_cleaning_summary_grid")
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Cleaning summary table could not be rendered: {exc}")


def _render_column_profiling_table(profiles: List[ColumnProfile]) -> None:
    """Renders the Type Inference Engine's per-column ColumnProfile output
    (core.type_inference.infer_dataframe) as an interactive, filterable
    data matrix — Column, Inferred Type, Confidence, Null %, Distinct
    Count, Needs Manual Review, Detection Notes. Never raises."""
    st.markdown("#### Schema Confidence & Type Inference Profile")
    try:
        if not profiles:
            st.caption("No column profiles are available for the active dataset.")
            return

        review_only = st.checkbox(
            "Show only columns flagged for manual review",
            value=st.session_state.get("_audit_review_only_toggle", False),
            key="_audit_review_only_toggle",
        )

        rows: List[Dict[str, Any]] = []
        for profile in profiles:
            if review_only and not profile.needs_manual_review:
                continue
            rows.append({
                "Column": profile.original_name,
                "Inferred Type": profile.inferred_type.value.title(),
                "Confidence": f"{profile.confidence:.0%}",
                "Null %": f"{profile.null_pct:.1f}%",
                "Distinct Count": profile.distinct_count,
                "Needs Manual Review": "Yes" if profile.needs_manual_review else "No",
                "Detection Notes": "; ".join(profile.detection_notes) if profile.detection_notes else "—",
            })

        if not rows:
            st.caption("No columns match the current filter — every column meets the auto-confidence threshold.")
            return

        render_enterprise_grid(
            pd.DataFrame(rows), key="_audit_column_profile_grid", search_columns=["Column"],
        )

        below_threshold = sum(1 for p in profiles if p.confidence < MIN_AUTO_CONFIDENCE)
        if below_threshold:
            st.caption(
                f"{below_threshold} of {len(profiles)} column(s) fall below the "
                f"{MIN_AUTO_CONFIDENCE:.0%} auto-confidence threshold and were routed to manual "
                "review during ingestion."
            )
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Column profiling table could not be rendered: {exc}")


def _render_schema_registry_audit(registry: ColumnRegistry) -> None:
    """Renders the active Column Registry's role -> column binding audit
    trail via registry.summary_table() (Role / Resolved Column / Source /
    Confidence / Confirmed). Never raises."""
    st.markdown("#### Column Registry — Role Binding Audit")
    try:
        summary_rows = registry.summary_table()
        if not summary_rows:
            st.caption("No role mappings have been established for the active dataset yet.")
            return
        render_enterprise_grid(
            pd.DataFrame(summary_rows), key="_audit_registry_grid", search_columns=["Role", "Resolved Column"],
        )
        confirmed_count = sum(1 for m in registry.mappings.values() if m.confirmed)
        total_count = len(registry.mappings)
        st.caption(f"{confirmed_count} of {total_count} tracked role(s) are currently confirmed and resolvable.")
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Column Registry audit table could not be rendered: {exc}")


def _render_audit_trace_log(audit_entries: List[Dict[str, Any]]) -> None:
    """Renders the chronological Audit Trace Log (utils.audit_log.AuditLog
    .as_table() schema: Timestamp / Type / Description / Rows Affected),
    with a free-text search filter across Type and Description. This IS
    the platform's Processing Log — no separate object exists in the
    backend beyond this flattened, already most-recent-first-ordered list.
    Never raises."""
    st.markdown("#### Audit Trace Log")
    try:
        if not audit_entries:
            st.caption("No audit trace entries have been recorded for the active dataset yet.")
            return

        search_text = st.text_input(
            "Search Audit Trace Log (Type / Description)",
            value=st.session_state.get("_audit_trace_search_text", ""),
            key="_audit_trace_search_text",
            placeholder="e.g. 'duplicate', 'mapping', 'outlier'…",
        )

        audit_df = pd.DataFrame(audit_entries)
        if search_text.strip():
            needle = search_text.strip().lower()
            mask = (
                audit_df.get("Type", pd.Series(dtype=str)).astype(str).str.lower().str.contains(needle, na=False)
                | audit_df.get("Description", pd.Series(dtype=str)).astype(str).str.lower().str.contains(needle, na=False)
            )
            audit_df = audit_df.loc[mask]

        if audit_df.empty:
            st.caption(f"No audit trace entries match '{search_text}'.")
            return

        render_enterprise_grid(audit_df, key="_audit_trace_log_grid")
        st.caption(f"Showing {len(audit_df):,} of {len(audit_entries):,} total audit trace entries.")
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Audit Trace Log could not be rendered: {exc}")


def _render_readiness_recommendations(recommendations: List[str]) -> None:
    """Renders the Analytics Readiness Score's actionable recommendations
    (computed by FRONTEND.app._compute_analytics_readiness_score) as a
    bulleted action list inside a glassmorphic card container. Never
    raises."""
    st.markdown("#### Readiness Recommendations")
    try:
        if not recommendations:
            st.caption("No readiness recommendations are currently available.")
            return
        items_html = "".join(f"<li>{rec}</li>" for rec in recommendations)
        st.markdown(
            f'<div class="kesco-card"><ul style="margin:0; padding-left: 18px;">{items_html}</ul></div>',
            unsafe_allow_html=True,
        )
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Readiness recommendations could not be rendered: {exc}")


def _render_notification_recap(notifications: List[Dict[str, str]]) -> None:
    """Renders a compact governance-focused recap of the platform's
    Notification Center payload (already built by
    FRONTEND.app._build_notifications), reusing the exact global
    kesco-badge / kesco-card classes injected by FRONTEND.app's
    centralized design system. Never raises."""
    st.markdown("#### Notification Recap")
    try:
        if not notifications:
            st.caption("No active notifications for the current workspace.")
            return
        for note in notifications:
            severity = note.get("severity", "info")
            badge_class = _SEVERITY_BADGE_MAP.get(severity, "kesco-badge-info")
            st.markdown(
                f'<div class="kesco-card">'
                f'<span class="kesco-badge {badge_class}">{severity.upper()}</span>'
                f'&nbsp;&nbsp;<strong>{note.get("category", "")}</strong>'
                f'<br/>{note.get("message", "")}'
                f'<br/><span class="kesco-alert-timestamp">{note.get("timestamp", "")}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Notification recap could not be rendered: {exc}")


def _render_deep_dive_analysis(
    audit_results: Dict[str, Any],
    profiles: List[ColumnProfile],
    registry: ColumnRegistry,
    readiness_recommendations: List[str],
    notifications: List[Dict[str, str]],
) -> None:
    """Orchestrates the Deep-Dive Analysis section across five tabbed
    panels: Cleaning Summary, Schema Confidence, Registry Audit, Audit
    Trace Log, and Recommendations & Notifications. Never raises."""
    st.markdown('<div class="kesco-section-title">Deep-Dive Analysis</div>', unsafe_allow_html=True)
    try:
        tab_cleaning, tab_schema, tab_registry, tab_trace, tab_recs = st.tabs([
            "Safe Cleaning Summary",
            "Schema Confidence Profile",
            "Registry Binding Audit",
            "Audit Trace Log",
            "Recommendations & Notifications",
        ])
        with tab_cleaning:
            _render_cleaning_summary_table(audit_results.get("cleaning_summary", {}))
        with tab_schema:
            _render_column_profiling_table(profiles)
        with tab_registry:
            _render_schema_registry_audit(registry)
        with tab_trace:
            _render_audit_trace_log(audit_results.get("audit_entries", []))
        with tab_recs:
            _render_readiness_recommendations(readiness_recommendations)
            st.markdown("---")
            _render_notification_recap(notifications)
    except Exception as exc:  # noqa: BLE001
        incident_id = log_exception("audit._render_deep_dive_analysis", exc)
        st.warning(f"Deep-Dive Analysis encountered an issue and has been degraded gracefully (Reference ID: {incident_id}).")


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 3.5 — EXPORT CENTER (presentation-only serialization, no new logic)
# ══════════════════════════════════════════════════════════════════════════════

def _render_export_center(audit_entries: List[Dict[str, Any]], registry: ColumnRegistry) -> None:
    """Renders local export triggers for the Audit Trace Log and Column
    Registry mapping summary as CSV — pure presentation-layer
    serialization of already-computed session_state data, duplicating no
    business logic. Never raises."""
    st.markdown('<div class="kesco-section-title">Export Center</div>', unsafe_allow_html=True)
    try:
        export_col_a, export_col_b = st.columns(2)
        with export_col_a:
            if audit_entries:
                audit_csv = pd.DataFrame(audit_entries).to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Export Audit Trace Log (CSV)",
                    data=audit_csv,
                    file_name="kesco_audit_trace_log.csv",
                    mime="text/csv",
                    key="_audit_export_trace_csv",
                    use_container_width=True,
                )
            else:
                st.caption("Audit Trace Log export becomes available once entries exist.")
        with export_col_b:
            summary_rows = registry.summary_table()
            if summary_rows:
                registry_csv = pd.DataFrame(summary_rows).to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Export Registry Binding Audit (CSV)",
                    data=registry_csv,
                    file_name="kesco_registry_binding_audit.csv",
                    mime="text/csv",
                    key="_audit_export_registry_csv",
                    use_container_width=True,
                )
            else:
                st.caption("Registry export becomes available once role mappings exist.")
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Export Center could not be rendered: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 4 — GOVERNANCE FOOTER
# ══════════════════════════════════════════════════════════════════════════════

def _render_governance_footer(
    page_start_time: float,
    registry: ColumnRegistry,
    audit_entries: List[Dict[str, Any]],
) -> None:
    """Renders dynamically computed governance telemetry — page render
    duration, active domain, workspace, confirmed mapping ratio, and total
    audit trace entry count — never hardcoded or assumed. Never raises."""
    try:
        render_duration = time.perf_counter() - page_start_time
        domain_label, domain_confidence = st.session_state.get("domain_detection", (DOMAIN_UNKNOWN, 0.0))
        confirmed_count = sum(1 for m in registry.mappings.values() if m.confirmed)
        total_mappings = len(registry.mappings)

        st.markdown(
            '<div class="kesco-section-title">Governance & Runtime Telemetry</div>',
            unsafe_allow_html=True,
        )
        telemetry_cols = st.columns(4)
        telemetry_cols[0].metric("Page Render Duration", f"{render_duration:.2f}s")
        telemetry_cols[1].metric("Confirmed Role Bindings", f"{confirmed_count} / {total_mappings}")
        telemetry_cols[2].metric("Audit Trace Entries", f"{len(audit_entries):,}")
        telemetry_cols[3].metric("Business Domain Confidence", f"{domain_confidence:.0%}")

        st.caption(
            f"KESCO System Diagnostics & Audit Telemetry · Domain: {domain_label} · Workspace: "
            f"{st.session_state.get('workspace_name', 'Default Workspace')} · "
            f"Rendered at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
    except Exception as exc:  # noqa: BLE001
        log_exception("audit._render_governance_footer", exc)
        st.caption(f"Governance telemetry could not be fully rendered: {exc}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE ORCHESTRATION — IMMUTABLE LAYOUT PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

_page_start_time: float = time.perf_counter()

_init_audit_page_state()
_inject_audit_css(st.session_state.get("theme", DEFAULT_THEME_KEY))

_uploaded_df: Optional[pd.DataFrame] = st.session_state.get("uploaded_dataframe")
_registry: Optional[ColumnRegistry] = st.session_state.get("column_registry")

if _uploaded_df is None or _uploaded_df.empty or _registry is None:
    _render_empty_state()
    st.stop()

_render_audit_header()

_audit_results: Dict[str, Any] = st.session_state.get("audit_results", {})
_readiness_score: int = st.session_state.get("readiness_score", 0)
_readiness_band: str = st.session_state.get("readiness_band", "Critical")
_readiness_recommendations: List[str] = st.session_state.get("readiness_recommendations", [])
_notifications: List[Dict[str, str]] = st.session_state.get("notifications", [])
_column_profiles: List[ColumnProfile] = st.session_state.get("column_profiles", [])

_render_integrity_health(_audit_results, _readiness_score, _readiness_band)

_render_deep_dive_analysis(
    audit_results=_audit_results,
    profiles=_column_profiles,
    registry=_registry,
    readiness_recommendations=_readiness_recommendations,
    notifications=_notifications,
)

_render_export_center(_audit_results.get("audit_entries", []), _registry)

_render_governance_footer(
    page_start_time=_page_start_time,
    registry=_registry,
    audit_entries=_audit_results.get("audit_entries", []),
)