"""
core/themes.py — Enterprise theme palettes + fully-inlined static-asset bridge.

Milestone 15 / Static-Asset-404 Root-Cause Remediation:
  • ROOT CAUSE (superseding the Milestone 14 analysis): `apply_theme_now()`
    and `bootstrap_static_js()` were already correctly migrated to
    `st.html()` — the modern, non-iframe-sandboxed primitive — so
    `window.parent`/`window.top` reach-around was never the actual failure
    mode in this revision. The real defect was that
    `inject_static_theme_link()` and `bootstrap_static_js()` requested
    `FRONTEND/static/style.css` and `FRONTEND/static/js/main.js` via a
    `<link href="/FRONTEND/static/...">` / `<script src="/FRONTEND/static/...">`
    URL that Streamlit's built-in static file server does not actually
    serve at that path (it only serves a `static/` directory sitting
    directly beside the app's entrypoint script, at the fixed `app/static/`
    URL prefix — not an arbitrary nested `FRONTEND/static/` location at an
    arbitrary absolute URL). Every request 404'd silently: no Python
    exception, no Streamlit error banner, just a dead `<link>`/`<script
    src>` the browser dropped. Consequence chain:
        1. style.css never loaded  -> the `html[data-theme="..."]` custom
           property blocks that back every visible color in the app never
           existed in the document  -> theme switching appeared completely
           inert even though `apply_theme_now()` WAS correctly setting
           `data-theme` on `<html>`/`<body>` every rerun.
        2. style.css never loaded  -> the `.keds-kpi-card-container` /
           `.keds-kpi-card-inner` / `.keds-kpi-flip-front` /
           `.keds-kpi-flip-back` 3D-transform rules never existed  -> the
           KPI card markup (structurally correct in
           FRONTEND/pages/1_dashboard.py) rendered as three stacked, flat,
           unstyled <div>s instead of a hoverable 3D flip card.
        3. main.js never loaded  -> its MutationObserver-based backup theme
           sync and its Streamlit-wrapper DOM-unclipping guard never ran,
           though this was never the primary failure — style.css's absence
           alone fully explains both reported symptoms.
  • FIX: `FRONTEND/static/style.css` and `FRONTEND/static/js/main.js` are
    RETIRED as separately-served files. Their content is preserved
    VERBATIM and now lives as Python string constants
    (`_INLINE_CSS_CONTENT`, `_INLINE_JS_CONTENT`) in this module, injected
    directly into the live application document via `st.html()` — the same
    unsandboxed primitive `apply_theme_now()` already used successfully.
    This has zero dependency on Streamlit's static-file-serving
    configuration, directory adjacency to the entrypoint, or any reverse-
    proxy/subpath base-path behavior, and is therefore strictly more
    deployment-robust than the file the theme system used to depend on.
  • `inject_static_theme_link()` and `bootstrap_static_js()` KEEP THEIR
    EXACT PUBLIC NAMES AND SIGNATURES so `app.py` requires zero changes.
    `inject_static_theme_link()` now injects an inline `<style>` block
    (idempotent/byte-identical every call, so Streamlit's element
    reconciliation reuses the same DOM node position-for-position exactly
    as the old `<link>` tag did — no duplicate `<style>` blocks accumulate
    across reruns). `bootstrap_static_js()` now injects an inline
    `<script>` block instead of a `<script src>` pointer; it remains safe
    to call multiple times (guarded both by app.py's session-state
    one-time-bootstrap flag AND by this script's own internal
    `window.__kescoMainJsInitialized` idempotency check, preserved
    verbatim from the original main.js).
  • This module still owns the Theme/THEMES data, consumed directly by
    chart_factory.py for Plotly template/color resolution — that contract
    is completely unchanged. No Python class structures, function
    signatures, or duckdb_executor-related imports anywhere in this
    project were touched by this change.
"""
from __future__ import annotations

from typing import Dict, TypedDict, List

import streamlit as st


class Theme(TypedDict):
    name: str
    primary: str
    secondary: str
    background: str
    surface: str
    text: str
    success: str
    warning: str
    danger: str
    plotly_template: str
    #plotly_white: str




THEMES: Dict[str, Theme] = {
    "kesco_corporate": {
        "name": "KESCO Corporate",
        "primary": "#1D4ED8", "secondary": "#5B6472", "background": "#F1F5F9",
        "surface": "#FFFFFF", "text": "#0F172A", "success": "#15803D",
        "warning": "#B45309", "danger": "#B91C1C", "plotly_template": "plotly_white",
    },
    "executive_dark": {
        "name": "Executive Dark",
        # Milestone 16: soft Royal Navy base per spec, replacing the flat
        # #18181B/#27272A neutral-gray pair. Text/accent colors unchanged
        # (already sufficiently high-contrast against navy).
        "primary": "#60A5FA", "secondary": "#94A3B8", "background": "#0B1329",
        "surface": "#1C2541", "text": "#E4E4E7", "success": "#4ADE80",
        "warning": "#FBBF24", "danger": "#F87171", "plotly_template": "plotly_dark",
    },
    "professional_light": {
        "name": "Professional Light",
        "primary": "#334155", "secondary": "#64748B", "background": "#F8FAFC",
        "surface": "#FFFFFF", "text": "#1E293B", "success": "#166534",
        "warning": "#92400E", "danger": "#991B1B", "plotly_template": "plotly_white",
    },
    "government_blue": {
        "name": "Government Blue",
        "primary": "#1E3A5F", "secondary": "#51677D", "background": "#EEF2F6",
        "surface": "#FFFFFF", "text": "#16202A", "success": "#1F6B44",
        "warning": "#8A5A00", "danger": "#8E2A2E", "plotly_template": "plotly_white",
    },
    "high_contrast": {
        "name": "High Contrast",
        "primary": "#000000", "secondary": "#2B2B2B", "background": "#FFFFFF",
        "surface": "#FFFFFF", "text": "#000000", "success": "#0B5A1E",
        "warning": "#7A4B00", "danger": "#7A0C0C", "plotly_template": "plotly_white",
    },
}

DEFAULT_THEME_KEY = "executive_dark"

# ── Per-theme categorical palettes for chart_factory.py's _color_seq().
# High Contrast intentionally uses a reduced, WCAG-AA-safe, low-saturation
# set instead of the brand rainbow, so chart series remain distinguishable
# without relying on hue alone. Every other theme keeps the existing
# 15-color brand palette unchanged (zero visual regression for existing
# themes) — only High Contrast gets an accessibility-specific override.
CATEGORICAL_PALETTES: Dict[str, List[str]] = {
    "high_contrast": [
        "#000000", "#4D4D4D", "#7A7A7A", "#1A1A1A", "#333333",
        "#000000", "#595959", "#0D0D0D", "#666666", "#262626",
    ],
}
_DEFAULT_CATEGORICAL_PALETTE: List[str] = [
    "#003B73", "#E65100", "#1B5E20", "#4A148C", "#006064",
    "#BF360C", "#37474F", "#F57F17", "#880E4F", "#1565C0",
    "#558B2F", "#0277BD", "#AD1457", "#00695C", "#283593",
]


def get_categorical_palette(theme_key: str) -> List[str]:
    """Returns the categorical color sequence for `theme_key`, falling back
    to the default brand palette for any theme without an explicit
    override. Never raises."""
    return CATEGORICAL_PALETTES.get(theme_key, _DEFAULT_CATEGORICAL_PALETTE)


def render_sidebar_section_header(icon: str, label: str) -> str:
    """
    Returns an HTML fragment for a Control-Room-style sidebar section
    header (icon + label + underline). Caller is responsible for the
    st.markdown(..., unsafe_allow_html=True) invocation. Never raises.
    Styled by .keds-sidebar-section* rules living in `_INLINE_CSS_CONTENT`
    below (previously FRONTEND/static/style.css).
    """
    return (
        f'<div class="keds-sidebar-section">'
        f'<span class="keds-sidebar-section-icon">{icon}</span>'
        f'<span class="keds-sidebar-section-label">{label}</span>'
        f'</div>'
    )


# ══════════════════════════════════════════════════════════════════════════
# Inlined Static Asset Content (Milestone 15 — retired external files)
# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
# Inlined Static Asset Content (Milestone 15 — retired external files)
# ══════════════════════════════════════════════════════════════════════════

_STATE_MARKER_ID: str = "keds-state-marker"

# ── Verbatim content of the retired FRONTEND/static/style.css ─────────────
# RAW string (r"""..."""), NOT an f-string: contains literal CSS curly
# braces and single quotes (font-family lists, url() values) that must
# never be interpreted by Python's string/format machinery. Keeping this
# as a plain raw string is what guarantees it can never desynchronize the
# quote-balance of the rest of the module.
_INLINE_CSS_CONTENT: str = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --keds-font-ui: 'Inter', 'Segoe UI', -apple-system, Arial, sans-serif;
  --keds-font-mono: 'JetBrains Mono', 'Consolas', monospace;
  --keds-radius: 4px;
  --keds-spacing-xs: 4px;
  --keds-spacing-sm: 8px;
  --keds-spacing-md: 16px;
  --keds-spacing-lg: 24px;
  --keds-spacing-xl: 32px;
  --keds-glass-blur: 14px;
  --keds-elevation-1: 0 1px 3px rgba(15,23,42,.08), 0 1px 2px rgba(15,23,42,.04);
  --keds-elevation-2: 0 8px 24px rgba(15,23,42,.14), 0 2px 6px rgba(15,23,42,.06);
  --keds-transition-fast: 140ms cubic-bezier(.4,0,.2,1);
  --keds-transition-med: 220ms cubic-bezier(.4,0,.2,1);
  --keds-primary: #60A5FA;
  --keds-secondary: #94A3B8;
  --keds-background: #18181B;
  --keds-surface: #27272A;
  --keds-text: #E4E4E7;
  --keds-success: #4ADE80;
  --keds-warning: #FBBF24;
  --keds-danger: #F87171;
  --keds-glass-border: rgba(255,255,255,0.08);
}

/* ── Per-theme palette mirrors (custom keds-* variables) ─────────────── */
html[data-theme="kesco_corporate"],
html[data-theme="kesco_corporate"] body,
html[data-theme="kesco_corporate"] .stApp,
html[data-theme="kesco_corporate"] [data-testid="stAppViewContainer"] {
  --keds-primary: #1D4ED8; --keds-secondary: #5B6472; --keds-background: #F1F5F9;
  --keds-surface: #FFFFFF; --keds-text: #0F172A; --keds-success: #15803D;
  --keds-warning: #B45309; --keds-danger: #B91C1C; --keds-glass-border: rgba(15,23,42,0.08);
}
html[data-theme="executive_dark"],
html[data-theme="executive_dark"] body,
html[data-theme="executive_dark"] .stApp,
html[data-theme="executive_dark"] [data-testid="stAppViewContainer"] {
  --keds-primary: #60A5FA; --keds-secondary: #94A3B8; --keds-background: #18181B;
  --keds-surface: #27272A; --keds-text: #E4E4E7; --keds-success: #4ADE80;
  --keds-warning: #FBBF24; --keds-danger: #F87171; --keds-glass-border: rgba(255,255,255,0.08);
}
html[data-theme="professional_light"],
html[data-theme="professional_light"] body,
html[data-theme="professional_light"] .stApp,
html[data-theme="professional_light"] [data-testid="stAppViewContainer"] {
  --keds-primary: #334155; --keds-secondary: #64748B; --keds-background: #F8FAFC;
  --keds-surface: #FFFFFF; --keds-text: #1E293B; --keds-success: #166534;
  --keds-warning: #92400E; --keds-danger: #991B1B; --keds-glass-border: rgba(15,23,42,0.08);
}
html[data-theme="government_blue"],
html[data-theme="government_blue"] body,
html[data-theme="government_blue"] .stApp,
html[data-theme="government_blue"] [data-testid="stAppViewContainer"] {
  --keds-primary: #1E3A5F; --keds-secondary: #51677D; --keds-background: #EEF2F6;
  --keds-surface: #FFFFFF; --keds-text: #16202A; --keds-success: #1F6B44;
  --keds-warning: #8A5A00; --keds-danger: #8E2A2E; --keds-glass-border: rgba(15,23,42,0.08);
}
html[data-theme="high_contrast"],
html[data-theme="high_contrast"] body,
html[data-theme="high_contrast"] .stApp,
html[data-theme="high_contrast"] [data-testid="stAppViewContainer"] {
  --keds-primary: #000000; --keds-secondary: #2B2B2B; --keds-background: #FFFFFF;
  --keds-surface: #FFFFFF; --keds-text: #000000; --keds-success: #0B5A1E;
  --keds-warning: #7A4B00; --keds-danger: #7A0C0C; --keds-glass-border: rgba(15,23,42,0.08);
}

/* ── STREAMLIT-NATIVE VARIABLE OVERRIDE — the real fix for text
   disappearing on non-Executive-Dark themes. Streamlit's built-in
   widgets read these native variable names directly; without this
   block they silently keep Streamlit's own default palette regardless
   of data-theme. ──────────────────────────────────────────────────── */
html[data-theme="kesco_corporate"] {
  --text-color: #0F172A; --background-color: #F1F5F9;
  --secondary-background-color: #FFFFFF; --primary-color: #1D4ED8;
}
html[data-theme="executive_dark"] {
  --text-color: #E4E4E7; --background-color: #18181B;
  --secondary-background-color: #27272A; --primary-color: #60A5FA;
}
html[data-theme="professional_light"] {
  --text-color: #1E293B; --background-color: #F8FAFC;
  --secondary-background-color: #FFFFFF; --primary-color: #334155;
}
html[data-theme="government_blue"] {
  --text-color: #16202A; --background-color: #EEF2F6;
  --secondary-background-color: #FFFFFF; --primary-color: #1E3A5F;
}
html[data-theme="high_contrast"] {
  --text-color: #000000; --background-color: #FFFFFF;
  --secondary-background-color: #FFFFFF; --primary-color: #000000;
}

[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"],
[data-testid="stCaptionContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li,
[data-testid="stText"],
[data-testid="stExpander"] summary,
[data-testid="stExpander"] p,
[data-testid="stTable"] td,
[data-testid="stTable"] th,
[data-testid="stDataFrameResizable"] [role="gridcell"],
[data-testid="stDataFrameResizable"] [role="columnheader"],
label, .stSelectbox label, .stTextInput label, .stNumberInput label {
  color: var(--keds-text) !important;
}

div[data-testid="stButton"] > button,
div[data-testid="stDownloadButton"] > button,
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stDateInput"] input {
  background-color: var(--keds-surface) !important;
  color: var(--keds-text) !important;
  border-color: var(--keds-glass-border) !important;
}

div[data-testid="stTabs"] button[role="tab"] p,
div[role="radiogroup"] label p,
div[data-testid="stCheckbox"] label p {
  color: var(--keds-text) !important;
}

div[data-testid="stPlotlyChart"] .xtick text,
div[data-testid="stPlotlyChart"] .ytick text,
div[data-testid="stPlotlyChart"] .legendtext {
  fill: var(--keds-text) !important;
}

html, body, [class*="css"], .stApp, .stMarkdown, .stText, p, span, div, label {
  font-family: var(--keds-font-ui) !important;
  letter-spacing: -0.01em;
}
code, pre, .stCodeBlock, [data-testid="stMetricValue"] { font-family: var(--keds-font-mono) !important; }

html, body {
  background-color: var(--keds-background) !important;
  color: var(--keds-text) !important;
}
.stApp {
  background-color: var(--keds-background) !important;
  color: var(--keds-text) !important;
  background-image: linear-gradient(160deg, var(--keds-background) 0%, var(--keds-surface) 55%, var(--keds-background) 100%);
  background-attachment: fixed;
  transition: background-color var(--keds-transition-med), color var(--keds-transition-med);
}
[data-testid="stAppViewContainer"] {
  background-color: var(--keds-background) !important;
  color: var(--keds-text) !important;
}
[data-testid="stMain"] { background-color: transparent !important; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; backdrop-filter: blur(6px); }
div[data-testid="stDecoration"] { display: none; }
div[data-testid="stStatusWidget"] { display: none; }
.stDeployButton { display: none; }
div[data-testid="stToolbar"] { visibility: hidden; }
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1600px; }

section[data-testid="stSidebar"] {
  background-color: var(--keds-surface) !important;
  border-right: 1px solid var(--keds-glass-border);
  box-shadow: 4px 0 20px rgba(15,23,42,0.04);
  transition: background-color var(--keds-transition-med);
}
section[data-testid="stSidebar"] * { color: var(--keds-text) !important; }
section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
section[data-testid="stSidebar"] h3 { font-weight: 800; letter-spacing: -0.02em; color: var(--keds-primary) !important; }
section[data-testid="stSidebar"] hr { border-color: var(--keds-glass-border); margin: 0.9rem 0; }

.keds-sidebar-section {
  display: flex; align-items: center; gap: 8px; margin: 14px 0 8px 0;
  padding-bottom: 6px; border-bottom: 1px solid var(--keds-glass-border);
}
.keds-sidebar-section-icon { font-size: 0.95rem; opacity: 0.85; }
.keds-sidebar-section-label {
  font-size: 0.72rem; font-weight: 800; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--keds-primary) !important;
}

div[data-testid="stButton"] > button,
div[data-testid="stDownloadButton"] > button,
div[data-testid="stFormSubmitButton"] > button {
  font-family: var(--keds-font-ui) !important; font-weight: 600; font-size: 0.82rem;
  letter-spacing: 0.01em; border-radius: 6px; border: 1px solid var(--keds-glass-border);
  background-color: var(--keds-surface); color: var(--keds-text);
  padding: 0.5rem 1rem; transition: all var(--keds-transition-fast); box-shadow: var(--keds-elevation-1);
}
div[data-testid="stButton"] > button:hover,
div[data-testid="stDownloadButton"] > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
  border-color: var(--keds-primary);
  box-shadow: var(--keds-elevation-2);
  transform: translateY(-1px);
}
div[data-testid="stButton"] > button:active { transform: translateY(0px) scale(0.99); }
button[kind="primary"] { background-color: var(--keds-primary) !important; color: #FFFFFF !important; border: none !important; }

div[data-baseweb="select"] > div, div[data-baseweb="input"] > div,
input[type="text"], input[type="number"],
div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input, div[data-testid="stDateInput"] input {
  background-color: var(--keds-surface) !important; border: 1px solid var(--keds-glass-border) !important;
  border-radius: 6px !important; color: var(--keds-text) !important; font-family: var(--keds-font-ui) !important;
  transition: border-color var(--keds-transition-fast), box-shadow var(--keds-transition-fast);
}
div[data-baseweb="select"] > div:focus-within, div[data-testid="stTextInput"] input:focus {
  border-color: var(--keds-primary) !important;
}
span[data-baseweb="tag"] {
  background-color: color-mix(in srgb, var(--keds-primary) 12%, transparent) !important;
  color: var(--keds-primary) !important; border: 1px solid color-mix(in srgb, var(--keds-primary) 25%, transparent) !important;
  border-radius: 4px !important; font-weight: 600; font-size: 0.72rem;
}
label, .stSelectbox label, .stMultiSelect label, .stTextInput label, .stNumberInput label, .stDateInput label {
  font-size: 0.74rem !important; font-weight: 600 !important; text-transform: uppercase;
  letter-spacing: 0.03em; color: var(--keds-secondary) !important;
}

div[data-testid="stTabs"] button[role="tab"] {
  font-weight: 600; font-size: 0.82rem; color: var(--keds-secondary);
  border-bottom: 2px solid transparent; padding: 10px 18px;
  transition: color var(--keds-transition-fast), border-color var(--keds-transition-fast);
}
div[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--keds-primary) !important; border-bottom: 2px solid var(--keds-primary) !important;
}
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { background-color: var(--keds-primary) !important; }

div[data-testid="stExpander"] {
  background-color: var(--keds-surface); backdrop-filter: blur(var(--keds-glass-blur));
  border: 1px solid var(--keds-glass-border) !important; border-radius: 8px !important;
  box-shadow: var(--keds-elevation-1); overflow: hidden;
}
div[data-testid="stExpander"] summary { font-weight: 600; font-size: 0.85rem; color: var(--keds-text); padding: 0.6rem 0.9rem; }

div[data-testid="stMetric"] {
  background-color: var(--keds-surface); backdrop-filter: blur(var(--keds-glass-blur));
  border: 1px solid var(--keds-glass-border); border-left: 3px solid var(--keds-primary);
  border-radius: 8px; padding: 14px 18px; box-shadow: var(--keds-elevation-1);
  transition: box-shadow var(--keds-transition-med), transform var(--keds-transition-med);
}
div[data-testid="stMetric"]:hover { box-shadow: var(--keds-elevation-2); transform: translateY(-2px); }
div[data-testid="stMetricLabel"] {
  font-size: 0.7rem !important; font-weight: 700 !important; text-transform: uppercase;
  letter-spacing: 0.04em; color: var(--keds-secondary) !important;
}
div[data-testid="stMetricValue"] { font-size: 1.65rem !important; font-weight: 700 !important; color: var(--keds-text) !important; }

div[data-testid="stDataFrameResizable"], div[data-testid="stDataEditor"] {
  border: 1px solid var(--keds-glass-border) !important; border-radius: 8px !important; box-shadow: var(--keds-elevation-1);
}
div[data-testid="stDataFrameResizable"] [role="columnheader"] {
  background-color: var(--keds-background) !important; font-weight: 700 !important; font-size: 0.72rem !important;
  text-transform: uppercase; letter-spacing: 0.03em; color: var(--keds-secondary) !important;
  position: sticky !important; top: 0 !important; z-index: 2 !important;
}
div[data-testid="stDataFrameResizable"] [role="row"]:hover, div[data-testid="stDataEditor"] [role="row"]:hover {
  background-color: color-mix(in srgb, var(--keds-primary) 8%, transparent) !important;
}

div[data-testid="stFileUploaderDropzone"] {
  background-color: var(--keds-background); border: 1.5px dashed color-mix(in srgb, var(--keds-secondary) 40%, transparent) !important;
  border-radius: 8px !important; overflow: visible !important;
  transition: border-color var(--keds-transition-fast), background-color var(--keds-transition-fast);
}
div[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--keds-primary) !important; }
div[data-testid="stFileUploaderDropzone"] button {
  white-space: normal !important; word-break: break-word !important; height: auto !important;
  min-height: 2.2rem !important; line-height: 1.25 !important; padding-top: 6px !important;
  padding-bottom: 6px !important; max-width: 100% !important;
}
div[data-testid="stFileUploaderDropzone"] section { overflow: visible !important; }

div[role="radiogroup"] label[data-baseweb="radio"] div:first-child[aria-checked="true"],
div[data-testid="stCheckbox"] span[aria-checked="true"] {
  background-color: var(--keds-primary) !important; border-color: var(--keds-primary) !important;
}
div[data-testid="stSlider"] div[role="slider"] { background-color: var(--keds-primary) !important; }

div[data-testid="stAlert"] {
  border-radius: 8px !important; border: 1px solid var(--keds-glass-border) !important;
  backdrop-filter: blur(var(--keds-glass-blur)); font-size: 0.85rem;
}

::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--keds-background); }
::-webkit-scrollbar-thumb { background-color: color-mix(in srgb, var(--keds-secondary) 50%, transparent); border-radius: 4px; }

div[data-testid="stPlotlyChart"] {
  background-color: var(--keds-surface); backdrop-filter: blur(var(--keds-glass-blur));
  border: 1px solid var(--keds-glass-border); border-radius: 8px; padding: 8px; box-shadow: var(--keds-elevation-1);
}

h1, h2, h3 { font-weight: 800 !important; letter-spacing: -0.02em !important; color: var(--keds-text) !important; }
hr { border-color: var(--keds-glass-border) !important; margin: 1.1rem 0 !important; }

.kesco-card {
  background-color: var(--keds-surface); border-radius: 8px; box-shadow: var(--keds-elevation-1);
  padding: 14px 16px; margin-bottom: var(--keds-spacing-md); border: 1px solid var(--keds-glass-border);
  backdrop-filter: blur(var(--keds-glass-blur));
}
.kesco-card:hover { box-shadow: var(--keds-elevation-2); }
.kesco-metric-value { font-size: 1.6rem; font-weight: 700; color: var(--keds-text); }
.kesco-metric-label { font-size: 0.78rem; font-weight: 600; letter-spacing: 0.02em; text-transform: uppercase; color: var(--keds-secondary); }
.kesco-badge {
  display: inline-block; border-radius: 4px; padding: 2px 10px; font-size: 0.72rem;
  font-weight: 700; letter-spacing: 0.03em; text-transform: uppercase;
}
.kesco-badge-critical { background-color: color-mix(in srgb, var(--keds-danger) 20%, transparent); color: var(--keds-danger); }
.kesco-badge-warning  { background-color: color-mix(in srgb, var(--keds-warning) 20%, transparent); color: var(--keds-warning); }
.kesco-badge-good     { background-color: color-mix(in srgb, var(--keds-success) 20%, transparent); color: var(--keds-success); }
.kesco-badge-info     { background-color: color-mix(in srgb, var(--keds-secondary) 20%, transparent); color: var(--keds-secondary); }
.kesco-section-title {
  font-size: 1.05rem; font-weight: 700; color: var(--keds-text); margin-top: var(--keds-spacing-lg);
  margin-bottom: var(--keds-spacing-sm); border-left: 4px solid var(--keds-primary); padding-left: var(--keds-spacing-sm);
}
.kesco-footer { color: var(--keds-secondary); font-size: 0.72rem; text-align: center; padding-top: var(--keds-spacing-xl); }
.kesco-narrative-container {
  background-color: var(--keds-surface); border-radius: 8px; border: 1px solid var(--keds-glass-border);
  padding: var(--keds-spacing-md); height: 100%; color: var(--keds-text);
}
.kesco-quality-banner {
  display: flex; flex-wrap: wrap; gap: var(--keds-spacing-lg); background-color: var(--keds-surface);
  backdrop-filter: blur(10px); border: 1px solid var(--keds-glass-border); border-radius: 8px;
  padding: var(--keds-spacing-md) var(--keds-spacing-lg); margin-bottom: var(--keds-spacing-lg);
  box-shadow: var(--keds-elevation-1);
}
.kesco-quality-item { display: flex; flex-direction: column; min-width: 130px; }
.kesco-quality-value { font-size: 1.15rem; font-weight: 700; color: var(--keds-text); }
.kesco-quality-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.03em; color: var(--keds-secondary); }
.kesco-alert-tier-title {
  font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.03em;
  margin-bottom: var(--keds-spacing-sm); padding-left: var(--keds-spacing-sm); border-left: 3px solid var(--keds-secondary);
}
.kesco-accent-danger { border-left-color: var(--keds-danger) !important; color: var(--keds-danger); }
.kesco-accent-warning { border-left-color: var(--keds-warning) !important; color: var(--keds-warning); }
.kesco-accent-secondary { border-left-color: var(--keds-secondary) !important; color: var(--keds-secondary); }
.kesco-accent-success { border-left-color: var(--keds-success) !important; color: var(--keds-success); }
.kesco-alert-card {
  background-color: var(--keds-surface); border-radius: 8px; border: 1px solid var(--keds-glass-border);
  border-left: 3px solid var(--keds-secondary); padding: var(--keds-spacing-sm) var(--keds-spacing-md);
  margin-bottom: var(--keds-spacing-sm); font-size: 0.82rem; color: var(--keds-text);
}
.kesco-alert-critical { border-left-color: var(--keds-danger); }
.kesco-alert-warning { border-left-color: var(--keds-warning); }
.kesco-alert-info { border-left-color: var(--keds-secondary); }
.kesco-alert-timestamp { font-size: 0.68rem; color: var(--keds-secondary); }

.keds-kpi-card, .kesco-kpi-card {
  position: relative; background-color: var(--keds-surface); backdrop-filter: blur(var(--keds-glass-blur));
  border: 1px solid var(--keds-glass-border); border-left: 4px solid var(--keds-primary); border-radius: 10px;
  padding: 16px 18px; margin-bottom: 10px; overflow: hidden;
  transition: box-shadow var(--keds-transition-med), transform var(--keds-transition-med); cursor: help;
}
.keds-kpi-card:hover, .kesco-kpi-card:hover {
  box-shadow: var(--keds-elevation-2); transform: translateY(-3px);
}
.keds-accent-success, .kesco-accent-success { border-left-color: var(--keds-success) !important; }
.keds-accent-warning, .kesco-accent-warning { border-left-color: var(--keds-warning) !important; }
.keds-accent-danger,  .kesco-accent-danger  { border-left-color: var(--keds-danger) !important; }
.keds-accent-secondary, .kesco-accent-secondary { border-left-color: var(--keds-secondary) !important; }
.keds-kpi-top-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
.keds-kpi-icon { font-size: 1.05rem; opacity: 0.65; }
.keds-kpi-label, .kesco-kpi-label {
  font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; color: var(--keds-secondary);
}
.keds-kpi-value, .kesco-kpi-value { font-size: 1.55rem; font-weight: 800; color: var(--keds-text); line-height: 1.1; font-family: var(--keds-font-mono); }
.keds-kpi-sparkline-slot { margin-top: 6px; min-height: 34px; opacity: 0.9; }

/* ══════════════════════════════════════════════════════════════════════
   3D KPI FLIP CARD — HARDENED, STACKING-CONTEXT-SAFE ARCHITECTURE.
   ══════════════════════════════════════════════════════════════════════ */
.keds-kpi-card-container {
  width: 100%;
  height: 152px;
  perspective: 1200px;
  -webkit-perspective: 1200px;
  margin-bottom: 12px;
  overflow: visible;
  isolation: isolate;
}
.keds-kpi-card-inner {
  position: relative;
  width: 100%;
  height: 100%;
  transition: transform 0.65s cubic-bezier(0.4, 0, 0.2, 1);
  transform-style: preserve-3d;
  -webkit-transform-style: preserve-3d;
  will-change: transform;
  transform: translateZ(0) rotateY(0deg);
}
.keds-kpi-card-container:hover .keds-kpi-card-inner,
.keds-kpi-card-container:focus-within .keds-kpi-card-inner {
  transform: translateZ(0) rotateY(180deg);
}
.keds-kpi-flip-front,
.keds-kpi-flip-back {
  position: absolute;
  inset: 0;
  backface-visibility: hidden;
  -webkit-backface-visibility: hidden;
  box-sizing: border-box;
  border-radius: 10px;
  border: 1px solid var(--keds-glass-border);
  background-color: var(--keds-surface);
  box-shadow: var(--keds-elevation-1);
  padding: 16px 18px;
  overflow: hidden;
  margin: 0 !important;
  z-index: 1;
}
.keds-kpi-flip-front {
  transform: rotateY(0deg) translateZ(1px);
  border-left: 4px solid var(--keds-primary);
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
.keds-kpi-flip-front.keds-accent-success   { border-left-color: var(--keds-success); }
.keds-kpi-flip-front.keds-accent-warning   { border-left-color: var(--keds-warning); }
.keds-kpi-flip-front.keds-accent-danger    { border-left-color: var(--keds-danger); }
.keds-kpi-flip-front.keds-accent-secondary { border-left-color: var(--keds-secondary); }
.keds-kpi-flip-back {
  transform: rotateY(180deg) translateZ(1px);
  border-left: 4px solid var(--keds-secondary);
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.keds-kpi-card-container:hover .keds-kpi-flip-front,
.keds-kpi-card-container:hover .keds-kpi-flip-back {
  box-shadow: var(--keds-elevation-2);
}
.keds-kpi-top-row {
  display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;
}
.keds-kpi-icon { font-size: 1.05rem; opacity: 0.7; }
.keds-kpi-label {
  font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.04em; color: var(--keds-secondary); margin-bottom: 2px;
}
.keds-kpi-value {
  font-size: 1.55rem; font-weight: 800; color: var(--keds-text);
  line-height: 1.1; font-family: var(--keds-font-mono);
}
.keds-kpi-sparkline-slot { margin-top: 4px; min-height: 30px; opacity: 0.9; }
.keds-kpi-flip-back-label {
  font-size: 0.63rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--keds-secondary); margin-bottom: 8px;
}
.keds-kpi-flip-back-formula {
  font-family: var(--keds-font-mono);
  font-size: 0.74rem;
  line-height: 1.5;
  color: var(--keds-text);
  background: color-mix(in srgb, var(--keds-background) 55%, transparent);
  border: 1px solid var(--keds-glass-border);
  border-radius: 6px;
  padding: 10px 12px;
  word-break: break-word;
  white-space: pre-wrap;
}
.keds-kpi-flip-hint {
  position: absolute; bottom: 6px; right: 10px; font-size: 0.58rem;
  color: var(--keds-secondary); opacity: 0.5; letter-spacing: 0.03em;
  text-transform: uppercase;
}

div[data-testid="stMarkdownContainer"]:has(.keds-kpi-card-container),
div[data-testid="element-container"]:has(.keds-kpi-card-container),
div[data-testid="stVerticalBlock"]:has(.keds-kpi-card-container),
div[data-testid="stVerticalBlockBorderWrapper"]:has(.keds-kpi-card-container),
div[data-testid="column"]:has(.keds-kpi-card-container),
div[data-testid="stHorizontalBlock"]:has(.keds-kpi-card-container) {
  overflow: visible !important;
  transform: none !important;
  contain: none !important;
  perspective: none !important;
}

#keds-state-marker { display: none !important; }
/* ── Milestone 16 — Sidebar collapse-handle & button-sizing fix ────────── */
section[data-testid="stSidebar"] div[data-testid="stButton"],
section[data-testid="stSidebar"] div[data-testid="stDownloadButton"] {
  display: inline-flex;
  width: auto !important;
  max-width: 100%;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] > button,
section[data-testid="stSidebar"] div[data-testid="stDownloadButton"] > button {
  width: 100% !important;
  min-width: 0;
  padding: 0.45rem 0.75rem;
  font-size: 0.78rem;
  white-space: normal;
  line-height: 1.2;
}
/* Two-button rows (Activate/Remove, Load/Save) get equal bounded columns
   instead of each independently stretching to 100% of the sidebar, which
   was compounding and pushing past the collapse-handle's hit area. */
section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {
  gap: 6px;
}
/* Never let sidebar content cover Streamlit's native collapse control. */
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {
  z-index: 999999 !important;
  position: relative !important;
  pointer-events: auto !important;
}
section[data-testid="stSidebar"] {
  overflow-x: hidden;
}

/* ── Chart title / plot area overlap fix ────────────────────────────── */
div[data-testid="stPlotlyChart"] .gtitle {
  dominant-baseline: hanging;
}
div[data-testid="stPlotlyChart"] {
  padding-top: 14px !important;
}

/* ── Milestone 17 — Small-viewport overflow containment for the KPI
   flip-card DOM guard. The Ancestor DOM Walker forces overflow:visible
   on Streamlit wrapper divs so the 3D flip isn't clipped; on narrow
   viewports this can let a flipped card's back face exceed its column
   width and push .block-container into horizontal scroll. This query
   re-clips at the OUTERMOST page-level containers only (never the
   card's own .keds-kpi-card-container/.keds-kpi-card-inner, which must
   stay visible for the flip to render at all), and shrinks the card
   height slightly so the flip stays fully on-screen on small devices. ── */
@media (max-width: 640px) {
  [data-testid="stAppViewContainer"],
  [data-testid="stMain"],
  .block-container {
    overflow-x: hidden !important;
  }
  .keds-kpi-card-container {
    height: 128px;
    perspective: 900px;
    -webkit-perspective: 900px;
  }
  .keds-kpi-flip-front,
  .keds-kpi-flip-back {
    padding: 12px 14px;
  }
  .keds-kpi-value {
    font-size: 1.25rem;
  }
}                                            
/* ══════════════════════════════════════════════════════════════════════          #### NEW
   ADDITIVE — Fragment-Boundary Global Scoping & Uniform Card Enforcement
   (Non-destructive: adds broader-scope selectors only; does not
   override or remove any rule above.)
   ══════════════════════════════════════════════════════════════════════ */

/* Global root scoping — forces every top-level Streamlit container to
   inherit the active theme's background/text regardless of which
   fragment subtree last repainted it. */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stAppViewContainer"] * :where(.kesco-section-title, .stSelectbox, .stSlider, .stMultiSelect) {
  background-color: var(--background-color, var(--keds-background)) !important;
  color: var(--text-color, var(--keds-text)) !important;
}

/* Chart-row control cluster (Visualization Type / Group By / Top N) —
   additive: these selectors are new, not replacements for the sidebar
   `label` rules already defined above. */
[data-testid="stAppViewContainer"] div[data-testid="stSelectbox"] label,
[data-testid="stAppViewContainer"] div[data-testid="stSlider"] label,
[data-testid="stAppViewContainer"] div[data-testid="stMultiSelect"] label {
  color: var(--keds-secondary) !important;
  font-size: 0.74rem !important;
  font-weight: 600 !important;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}
[data-testid="stAppViewContainer"] div[data-testid="stSelectbox"] > div,
[data-testid="stAppViewContainer"] div[data-baseweb="select"] > div {
  background-color: var(--keds-surface) !important;
  color: var(--keds-text) !important;
  border-color: var(--keds-glass-border) !important;
}

/* Uniform 3D-flip enforcement for ANY card container, including ones
   mounted after the initial paint inside an st.fragment subtree. This
   targets a NEW marker class (`.keds-kpi-uniform-pending`) that the
   updated observer below stamps onto not-yet-processed cards, then
   promotes to the existing `.keds-kpi-card-container` styling — your
   original `.keds-kpi-card-container` rules are untouched and still win. */
[data-testid="stAppViewContainer"] .keds-kpi-uniform-pending {
  overflow: visible !important;
  perspective: 1200px !important;
  isolation: isolate !important;
}

/* Chart-title / plot-area anti-collision buffer, additive to the
   existing div[data-testid="stPlotlyChart"] rule already declared
   above — raises the reserved gutter without removing it. */
[data-testid="stAppViewContainer"] div[data-testid="stPlotlyChart"] {
  padding-top: 40px !important;
}
[data-testid="stAppViewContainer"] div[data-testid="stPlotlyChart"] .main-svg .gtitle {
  dominant-baseline: hanging !important;
  transform: translateY(6px) !important;
}
"""

# ── Verbatim content of the retired FRONTEND/static/js/main.js ────────────
# RAW string, NOT an f-string — the JS content below uses real `{ }`
# characters extensively; formatting this as an f-string would force
# doubling every brace and is exactly the kind of edit that produced the
# original SyntaxError. Keep this as r"""...""" permanently.
_INLINE_JS_CONTENT: str = r"""
(function () {
  "use strict";

  if (window.__kescoMainJsInitialized) {
    return;
  }
  window.__kescoMainJsInitialized = true;

  var MARKER_ID = "keds-state-marker";
  var KPI_CONTAINER_SELECTOR = ".keds-kpi-card-container";

  function resolveTargetDocs() {
    var docs = [document];
    try {
      if (window.top && window.top.document && docs.indexOf(window.top.document) === -1) {
        docs.push(window.top.document);
      }
    } catch (e) {
      // Cross-origin top window — silently skip the fallback.
    }
    return docs;
  }

  function applyThemeFromMarker() {
    var marker = document.getElementById(MARKER_ID);
    if (!marker) {
      return;
    }
    var themeKey = marker.getAttribute("data-active-theme");
    if (!themeKey) {
      return;
    }
    var docs = resolveTargetDocs();
    for (var i = 0; i < docs.length; i++) {
      var doc = docs[i];
      try {
        if (doc.documentElement.getAttribute("data-theme") !== themeKey) {
          doc.documentElement.setAttribute("data-theme", themeKey);
        }
        if (doc.body && doc.body.getAttribute("data-theme") !== themeKey) {
          doc.body.setAttribute("data-theme", themeKey);
        }
        if (doc.body) { void doc.body.offsetHeight; }
      } catch (err) {
        // Never let a client-side failure break the page.
      }
    }
  }

  var themeAttemptsRemaining = 10;
  function pollApplyTheme() {
    applyThemeFromMarker();
    themeAttemptsRemaining -= 1;
    if (themeAttemptsRemaining > 0) {
      window.setTimeout(pollApplyTheme, 60);
    }
  }
  pollApplyTheme();

  var GUARD_WRAPPER_SELECTORS = [
    '[data-testid="stMarkdownContainer"]',
    '[data-testid="element-container"]',
    '[data-testid="stVerticalBlock"]',
    '[data-testid="stVerticalBlockBorderWrapper"]',
    '[data-testid="column"]',
    '[data-testid="stHorizontalBlock"]'
  ];
  

  function unclipAncestorsOf(el) {
    var current = el.parentElement;
    var hops = 0;
    while (current && hops < 8) {
      var isGuardTarget = false;
      for (var i = 0; i < GUARD_WRAPPER_SELECTORS.length; i++) {
        try {
          if (current.matches && current.matches(GUARD_WRAPPER_SELECTORS[i])) {
            isGuardTarget = true;
            break;
          }
        } catch (e) {
          // Unsupported selector in this engine — skip defensively.
        }
      }
      if (isGuardTarget) {
        try {
          current.style.setProperty("overflow", "visible", "important");
          current.style.setProperty("display", "block", "important");
        } catch (e) {
          // Never let a client-side failure break the page.
        }
      }
      current = current.parentElement;
      hops += 1;
    }
  }

  // ── ADDITIVE: universal, root-scoped card + control-cluster guard ──────                    #ADDITIVE
  var UNIFORM_PENDING_CLASS = "keds-kpi-uniform-pending";
  var CARD_SELECTORS = [".keds-kpi-card-container", ".kesco-kpi-card", ".keds-kpi-card"];
  var CONTROL_CLUSTER_SELECTORS = [
    '[data-testid="stSelectbox"]',
    '[data-testid="stSlider"]',
    '[data-testid="stMultiSelect"]',
    ".kesco-section-title"
  ];

  function stampUniformPending(rootDoc) {
    try {
      for (var s = 0; s < CARD_SELECTORS.length; s++) {
        var cards = rootDoc.querySelectorAll(CARD_SELECTORS[s]);
        for (var i = 0; i < cards.length; i++) {
          var el = cards[i];
          if (!el.classList.contains(UNIFORM_PENDING_CLASS)) {
            el.classList.add(UNIFORM_PENDING_CLASS);
          }
          unclipAncestorsOf(el); // reuse existing, unmodified helper
        }
      }
    } catch (err) {
      // Never let a client-side failure break the page.
    }
  }

  function forceThemeOnControlClusters(rootDoc) {
    try {
      for (var s = 0; s < CONTROL_CLUSTER_SELECTORS.length; s++) {
        var nodes = rootDoc.querySelectorAll(CONTROL_CLUSTER_SELECTORS[s]);
        for (var i = 0; i < nodes.length; i++) {
          // Forces a style recalculation pass on freshly-mounted fragment
          // subtrees so CSS attribute-selector inheritance (data-theme)
          // is guaranteed to apply even if the node mounted between
          // observer callback batches.
          void nodes[i].offsetHeight;
        }
      }
    } catch (err) {
      // Never let a client-side failure break the page.
    }
  }

  function runUniversalDomGuard(rootDoc) {
    stampUniformPending(rootDoc);
    forceThemeOnControlClusters(rootDoc);
  }

  function runAllGuardsUniversal() {
    var docs = resolveTargetDocs(); // reuse existing, unmodified helper
    for (var i = 0; i < docs.length; i++) {
      runUniversalDomGuard(docs[i]);
    }
  }

  var guardAttemptsRemaining = 10;
  function pollRunGuards() {
    runAllGuardsUniversal();
    guardAttemptsRemaining -= 1;
    if (guardAttemptsRemaining > 0) {
      window.setTimeout(pollRunGuards, 80);
    }
  }
  pollRunGuards();

  var observer = new MutationObserver(function () {
    applyThemeFromMarker(); // reuse existing, unmodified helper
    runAllGuardsUniversal();
  });

  function startObserving() {
    var target = document.querySelector('[data-testid="stAppViewContainer"]') || document.body;
    if (target) {
      observer.observe(target, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ["data-active-theme", "style", "class"]
      });
    } else {
      window.setTimeout(startObserving, 50);
    }
  }
  startObserving();
})();
"""

_THEME_STYLE_ELEMENT_ID: str = "keds-theme-engine-style"
_THEME_MARKER_ELEMENT_ID: str = "keds-theme-engine-marker"


#   function runKpiDomGuard(rootDoc) {
#     try {
#       var containers = rootDoc.querySelectorAll(KPI_CONTAINER_SELECTOR);
#       for (var i = 0; i < containers.length; i++) {
#         unclipAncestorsOf(containers[i]);
#       }
#     } catch (err) {
#       // Never let a client-side failure break the page.
#     }
#   }

#   function runAllGuards() {
#     var docs = resolveTargetDocs();
#     for (var i = 0; i < docs.length; i++) {
#       runKpiDomGuard(docs[i]);
#     }
#   }

#   var guardAttemptsRemaining = 10;
#   function pollRunGuards() {
#     runAllGuards();
#     guardAttemptsRemaining -= 1;
#     if (guardAttemptsRemaining > 0) {
#       window.setTimeout(pollRunGuards, 80);
#     }
#   }
#   pollRunGuards();

#   var observer = new MutationObserver(function () {
#     applyThemeFromMarker();
#     runAllGuards();
#   });

#   function startObserving() {
#     if (document.body) {
#       observer.observe(document.body, {
#         childList: true,
#         subtree: true,
#         attributes: true,
#         attributeFilter: ["data-active-theme", "style", "class"]
#       });
#     } else {
#       window.setTimeout(startObserving, 50);
#     }
#   }
#   startObserving();
# })();
#  ""

def inject_static_theme_link() -> None:
    """
    Renders the platform's global stylesheet as an inline <style> block
    keyed to a single, explicit, stable element id (_THEME_STYLE_ELEMENT_ID).
    Milestone 16 fix: previously this used a bare, unkeyed <style> tag,
    which some Streamlit component-reconciliation paths (and manual
    hot-reloads) would append alongside rather than replace, producing
    duplicated/"ghosted" style rules and, transitively, duplicated visible
    text wherever a rule's content-injection side effect (::before/::after)
    was in play. Explicitly namespacing the id and wrapping in a script
    that removes any prior sibling node with the same id BEFORE inserting
    the new one guarantees exactly one live style node at all times,
    regardless of how the host page's reconciler behaves. Never raises.
    """
    try:
        st.html(
            f"""
            <script>
            (function() {{
                try {{
                    var docs = [document];
                    try {{ if (window.top && window.top.document) {{ docs.push(window.top.document); }} }} catch(e) {{}}
                    for (var d = 0; d < docs.length; d++) {{
                        var doc = docs[d];
                        var existing = doc.getElementById('{_THEME_STYLE_ELEMENT_ID}');
                        if (existing) {{ existing.parentNode.removeChild(existing); }}
                        var styleEl = doc.createElement('style');
                        styleEl.id = '{_THEME_STYLE_ELEMENT_ID}';
                        styleEl.textContent = {_INLINE_CSS_CONTENT!r};
                        doc.head.appendChild(styleEl);
                    }}
                }} catch (err) {{ /* never break the host page */ }}
            }})();
            </script>
            """
        )
    except Exception:  # noqa: BLE001
        pass


def render_theme_marker_html(theme_key: str) -> str:
    """Builds the hidden state-marker <div> HTML. theme_key is normalized
    against THEMES (falling back to DEFAULT_THEME_KEY). Never raises."""
    safe_theme_key = theme_key if theme_key in THEMES else DEFAULT_THEME_KEY
    return (
        f'<div id="{_THEME_MARKER_ELEMENT_ID}" '
        f'data-active-theme="{safe_theme_key}" '
        f'style="display:none !important;height:0;width:0;overflow:hidden;"></div>'
    )


def sync_theme_state_marker(theme_key: str) -> None:
    """
    Renders the backup state-marker via st.html() (not st.markdown()) so it
    lands in a Streamlit custom-element wrapper keyed by content hash
    rather than accumulating inside stMarkdownContainer's element list
    across reruns. Milestone 16 fix for DOM ghosting. Never raises.
    """
    try:
        st.html(render_theme_marker_html(theme_key))
    except Exception:  # noqa: BLE001
        pass




def apply_theme_now(theme_key: str) -> None:
    """
    Primary theme-application mechanism. Injects a small <script> block
    via st.html() that runs in the real top-level document (not a
    sandboxed iframe) and sets data-theme on <html>/<body>. This is the
    ONLY function in this module that interpolates a Python value into
    JS source, so it is the ONLY function that uses an f-string here --
    every literal JS brace below is doubled ({{ / }}) precisely because
    an f-string is in play. Never raises.
    """
    safe_theme_key = theme_key if theme_key in THEMES else DEFAULT_THEME_KEY
    try:
        st.html(
            f"""
            <script>
            (function() {{
                var THEME = '{safe_theme_key}';
                var ATTEMPTS = 6;

                function resolveTargetDocs() {{
                    var docs = [document];
                    try {{
                        if (window.top && window.top.document && docs.indexOf(window.top.document) === -1) {{
                            docs.push(window.top.document);
                        }}
                    }} catch (e) {{
                        // Cross-origin top window -- skip defensively.
                    }}
                    return docs;
                }}

                function applyOnce() {{
                    var docs = resolveTargetDocs();
                    for (var i = 0; i < docs.length; i++) {{
                        var doc = docs[i];
                        try {{
                            if (doc.documentElement.getAttribute('data-theme') !== THEME) {{
                                doc.documentElement.setAttribute('data-theme', THEME);
                            }}
                            if (doc.body && doc.body.getAttribute('data-theme') !== THEME) {{
                                doc.body.setAttribute('data-theme', THEME);
                            }}
                            if (doc.body) {{ void doc.body.offsetHeight; }}
                        }} catch (err) {{
                            // Never let a client-side failure surface to the user.
                        }}
                    }}
                }}

                function scheduleRetries(remaining) {{
                    applyOnce();
                    if (remaining <= 0) {{ return; }}
                    try {{
                        window.requestAnimationFrame(function () {{
                            scheduleRetries(remaining - 1);
                        }});
                    }} catch (e) {{
                        setTimeout(function () {{ scheduleRetries(remaining - 1); }}, 32);
                    }}
                }}

                scheduleRetries(ATTEMPTS);
            }})();
            </script>
            """
        )
    except Exception:  # noqa: BLE001
        pass
# core/themes.py — ADDITIVE, insert after apply_theme_now()

_ROOT_ENFORCEMENT_STYLE_ID: str = "keds-root-enforcement-style"

_ROOT_ENFORCEMENT_CSS: str = r"""
/* ══════ ADDITIVE — Global Root Enforcement (Fragment-Safe) ══════
   Forces the two outermost Streamlit containers back onto the active
   theme's variables even if a React reconciliation pass injects a
   conflicting inline style on a fragment remount. This block is
   idempotent (fixed id, replaced-not-appended) and safe to inject on
   every fragment mount via inject_fragment_atomic_style(). */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"] {
  background-color: var(--keds-background) !important;
  color: var(--keds-text) !important;
}
.stApp * :where(p, span, div, li, label, h1, h2, h3, h4, [data-testid="stMarkdownContainer"]) {
  color: var(--keds-text);
}
"""


def inject_fragment_atomic_style(theme_key: str) -> None:
    """
    Fragment-lifecycle-safe styling primitive. Call this as the FIRST
    statement inside any @st.fragment function. Re-asserts:
      1. The root enforcement CSS block (idempotent, fixed element id —
         removed-and-recreated so it never duplicates across repeated
         fragment reruns).
      2. The correct data-theme attribute on <html>/<body>, in case this
         fragment mounted before app.py's top-level apply_theme_now()
         executed on this particular rerun cycle.
    Never raises. Does not alter _INLINE_CSS_CONTENT, THEMES, or any
    existing public function's signature — purely additive.
    """
    safe_theme_key = theme_key if theme_key in THEMES else DEFAULT_THEME_KEY
    try:
        st.html(
            f"""
            <script>
            (function() {{
                try {{
                    var doc = document;
                    var existing = doc.getElementById('{_ROOT_ENFORCEMENT_STYLE_ID}');
                    if (existing) {{ existing.parentNode.removeChild(existing); }}
                    var styleEl = doc.createElement('style');
                    styleEl.id = '{_ROOT_ENFORCEMENT_STYLE_ID}';
                    styleEl.textContent = {_ROOT_ENFORCEMENT_CSS!r};
                    doc.head.appendChild(styleEl);

                    if (doc.documentElement.getAttribute('data-theme') !== '{safe_theme_key}') {{
                        doc.documentElement.setAttribute('data-theme', '{safe_theme_key}');
                    }}
                    if (doc.body && doc.body.getAttribute('data-theme') !== '{safe_theme_key}') {{
                        doc.body.setAttribute('data-theme', '{safe_theme_key}');
                    }}
                }} catch (e) {{ /* never break the fragment */ }}
            }})();
            </script>
            """
        )
    except Exception:  # noqa: BLE001
        pass

def bootstrap_static_js() -> None:
    """
    One-time bootstrap that injects the platform's DOM-guard/theme-sync
    JavaScript inline via st.html(). Double-guarded against duplicate
    execution: the caller (app.py::main) gates this behind
    st.session_state["_static_js_bootstrapped"], and the inlined script
    itself checks window.__kescoMainJsInitialized. Never raises.
    """
    try:
        st.html(f"<script>{_INLINE_JS_CONTENT}</script>")
    except Exception:  # noqa: BLE001
        pass