"""
utils/layout_guard.py — Fixed-Margin Layout Guard for Chart-Block rendering.

Standardizes the title -> chart vertical anchor across every page that
renders a Plotly/Folium visualization beneath a `.kesco-section-title`
header (1_dashboard.py, 4_self_service.py, 2_audit.py). Eliminates
browser margin-collapse between an st.markdown title and the immediately
following st.plotly_chart by wrapping both inside a single
st.container(border=False) — an "indivisible block" — with an explicit,
non-collapsible spacer div between them, rather than relying on default
Streamlit block spacing or the chart's own internal SVG margin alone.

This module performs NO business logic, NO aggregation, and NO chart
computation — it is purely a DOM/CSS layout primitive. Never raises: any
internal failure degrades to a plain st.markdown title + unstyled
container rather than crashing the host page.
"""
from __future__ import annotations

import contextlib
from typing import Any, Dict, Iterator, Optional

import streamlit as st

_GUARD_CSS_ID: str = "keds-chart-block-guard-style"
_DEFAULT_GAP_PX: int = 18
_MIN_CHART_TOP_MARGIN_PX: int = 64


def inject_chart_block_css() -> None:
    """
    Idempotent CSS injection establishing the Fixed-Margin Layout
    contract. st.html() replaces-by-id under the hood in this platform's
    convention (see core/themes.py), so repeated calls across reruns
    never accumulate duplicate style nodes. Safe to call once from
    app.py's bootstrap path, or defensively at the top of any page.
    """
    try:
        st.html(f"""
        <style id="{_GUARD_CSS_ID}">
        .keds-chart-block {{
            display: block;
            overflow: visible;
        }}
        .keds-chart-block-title {{
            margin-bottom: 0 !important;
        }}
        .keds-chart-block-spacer {{
            display: block;
            width: 100%;
            height: var(--keds-chart-gap, 18px);
            min-height: var(--keds-chart-gap, 18px);
            flex-shrink: 0;
            pointer-events: none;
        }}
        /* Neutralizes Streamlit's own margin-collapse between adjacent
           element-containers once they sit inside a chart-block scope. */
        div[data-testid="stVerticalBlock"]:has(> div .keds-chart-block)
            > div[data-testid="element-container"] {{
            margin-bottom: 0;
        }}
        div[data-testid="stPlotlyChart"] {{
            margin-top: 4px !important;
        }}
        </style>
        """)
    except Exception:  # noqa: BLE001
        pass


@contextlib.contextmanager
def chart_block(
    title: str,
    *,
    gap_px: int = _DEFAULT_GAP_PX,
    title_class: str = "kesco-section-title",
) -> Iterator[None]:
    """
    Fixed-Margin Layout 'Chart-Block' pattern.

    Wraps a title + its chart in ONE st.container(border=False) — Layout
    Partitioning — with an explicit, non-collapsible spacer div injected
    between them — Vertical Anchoring — so the title can never visually
    bleed into the chart below it, regardless of viewport width, resize,
    or @st.fragment rerun timing.

    Usage (drop-in replacement for a bare
    `st.markdown('<div class="kesco-section-title">...</div>', ...)`
    call that immediately precedes a chart render):

        from utils.layout_guard import chart_block

        with chart_block("Operational Concentration — Complaint Management"):
            st.plotly_chart(fig, width="stretch", key="_dash_fig_row1")
            ...

    Never raises: any internal failure degrades to a bare st.container()
    with the title rendered via plain st.markdown — worst case is lost
    styling, never a crashed page.
    """
    try:
        with st.container(border=False):
            st.markdown(
                f'<div class="keds-chart-block">'
                f'<div class="{title_class} keds-chart-block-title">{title}</div>'
                f'<div class="keds-chart-block-spacer" style="--keds-chart-gap:{gap_px}px;"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            yield
    except Exception:  # noqa: BLE001
        with st.container(border=False):
            st.markdown(f"### {title}")
            yield


def apply_fixed_margin_chart_layout(
    fig: Any,
    *,
    top_margin_px: int = _MIN_CHART_TOP_MARGIN_PX,
) -> Any:
    """
    Chart-Specific Cleanup: clamps a guaranteed-minimum top margin
    directly on the Plotly figure's own layout object — not just the
    surrounding DOM — so the plot area never renders flush against the
    container's top edge even before any CSS has painted.

    Belt-and-braces companion to chart_factory._apply_layout's existing
    `t: 64` margin: safe (idempotent, non-destructive) to call a second
    time on any go.Figure/px.* figure returned by
    visualization.chart_factory.render(). Never lowers an existing,
    larger margin — only raises it to at least `top_margin_px`.

    Returns fig unchanged (no-op) for any non-Plotly object (e.g. a
    folium.Map) or on any internal failure — never raises.
    """
    try:
        if fig is None or not hasattr(fig, "update_layout"):
            return fig
        current_margin: Dict[str, int] = {}
        try:
            if fig.layout.margin is not None:
                current_margin = {
                    "l": fig.layout.margin.l, "r": fig.layout.margin.r,
                    "t": fig.layout.margin.t, "b": fig.layout.margin.b,
                }
        except Exception:  # noqa: BLE001
            current_margin = {}
        resolved_top = max(int(current_margin.get("t") or 0), top_margin_px)
        fig.update_layout(margin=dict(
            l=current_margin.get("l") or 20,
            r=current_margin.get("r") or 20,
            t=resolved_top,
            b=current_margin.get("b") or 30,
        ))
        return fig
    except Exception:  # noqa: BLE001
        return fig