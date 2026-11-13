"""Visual system: colour tokens, Plotly template, global CSS and small HTML components."""

from __future__ import annotations

import html
import re

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ----------------------------------------------------------------------------- tokens
INK = "#0B1F33"        # headings
TEXT = "#334155"       # body
MUTED = "#64748B"      # captions
LINE = "#E2E8F0"       # borders
SURFACE = "#F8FAFC"    # page tint
PRIMARY = "#0F4C5C"    # deep petrol
ACCENT = "#C8553D"     # terracotta – used sparingly for emphasis / risk
POSITIVE = "#2F7D4F"
NEGATIVE = "#B42318"

PALETTE = ["#0F4C5C", "#5F8D96", "#C8553D", "#9AA5B1", "#D9A441", "#2F6690", "#7A8B6F", "#B9C7CC"]
SEQ = ["#EEF4F5", "#C9DCE0", "#94BAC2", "#5F8D96", "#2F6A78", "#0F4C5C"]
SEQ_WARM = ["#FBF1EE", "#F0CFC6", "#E1A291", "#D07A62", "#C8553D", "#8E3423"]

pio.templates["dss"] = go.layout.Template(
    layout=dict(
        font=dict(family="Inter, 'Segoe UI', Helvetica, Arial, sans-serif", size=12, color=TEXT),
        colorway=PALETTE,
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showgrid=False, linecolor="#CBD5E1", tickcolor="#CBD5E1", ticks="outside", ticklen=4,
                   title=dict(font=dict(size=11, color=MUTED))),
        yaxis=dict(gridcolor="#EEF2F6", zeroline=False, linecolor="rgba(0,0,0,0)",
                   title=dict(font=dict(size=11, color=MUTED))),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11), title=None),
        hoverlabel=dict(bgcolor="white", bordercolor=LINE, font=dict(size=12, color=INK)),
        margin=dict(l=8, r=8, t=36, b=8),
        coloraxis=dict(colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(size=10))),
        bargap=0.28,
    )
)
pio.templates.default = "plotly_white+dss"
px.defaults.template = "plotly_white+dss"
px.defaults.color_discrete_sequence = PALETTE
px.defaults.color_continuous_scale = SEQ

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {{
    font-family: 'Inter', 'Segoe UI', Helvetica, Arial, sans-serif !important;
}}
#MainMenu, footer, [data-testid="stDecoration"] {{ visibility: hidden; height: 0; }}
[data-testid="stAppViewContainer"] > .main {{ background: {SURFACE}; }}
.stApp {{ background: {SURFACE}; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }}
h1, h2, h3, h4 {{ color: {INK}; letter-spacing: -0.01em; }}

/* header */
.app-header {{ display:flex; justify-content:space-between; align-items:flex-end; padding: 0 0 18px 0;
               border-bottom: 1px solid {LINE}; margin-bottom: 8px; }}
.app-overline {{ font-size: 11px; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; color: {PRIMARY}; }}
.app-title {{ font-size: 26px; font-weight: 700; color: {INK}; margin: 4px 0 2px 0; letter-spacing: -0.02em; }}
.app-sub {{ font-size: 13px; color: {MUTED}; }}
.app-meta {{ text-align:right; font-size: 12px; color: {MUTED}; line-height: 1.6; }}
.app-meta b {{ color: {INK}; font-weight: 600; }}

/* tabs as underline navigation */
.stTabs [data-baseweb="tab-list"] {{ gap: 26px; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ height: 44px; padding: 0 2px; background: transparent; font-weight: 500;
                                color: {MUTED}; font-size: 14px; }}
.stTabs [aria-selected="true"] {{ color: {INK}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {PRIMARY}; height: 2px; }}
.stTabs [data-baseweb="tab-border"] {{ display: none; }}

/* section heading */
.sec {{ margin: 26px 0 10px 0; }}
.sec h3 {{ font-size: 16px; font-weight: 600; margin: 0; color: {INK}; }}
.sec p {{ font-size: 12.5px; color: {MUTED}; margin: 3px 0 0 0; }}

/* bordered containers → cards */
[data-testid="stVerticalBlockBorderWrapper"] {{ background: white; border-color: {LINE} !important; border-radius: 10px; }}
.card-title {{ font-size: 13px; font-weight: 600; color: {INK}; margin-bottom: 2px; }}
.card-note {{ font-size: 12px; color: {MUTED}; margin-bottom: 6px; }}

/* KPI tiles */
.kpi {{ background: white; border: 1px solid {LINE}; border-radius: 10px; padding: 14px 16px 12px 16px; height: 100%; }}
.kpi-label {{ font-size: 11px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: {MUTED}; }}
.kpi-value {{ font-size: 26px; font-weight: 600; color: {INK}; margin: 6px 0 4px 0; font-variant-numeric: tabular-nums;
              letter-spacing: -0.02em; }}
.kpi-foot {{ font-size: 12px; color: {MUTED}; font-variant-numeric: tabular-nums; }}
.kpi-up {{ color: {POSITIVE}; font-weight: 600; }}
.kpi-down {{ color: {NEGATIVE}; font-weight: 600; }}
.kpi-flat {{ color: {MUTED}; font-weight: 600; }}

/* insight list */
.ins {{ display:flex; gap: 14px; padding: 12px 0; border-bottom: 1px solid {LINE}; font-size: 13.5px; color: {TEXT};
        line-height: 1.5; }}
.ins:last-child {{ border-bottom: none; }}
.ins-tag {{ flex: 0 0 92px; font-size: 10.5px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
            padding-top: 2px; }}
.ins-tag.info {{ color: {PRIMARY}; }}
.ins-tag.good {{ color: {POSITIVE}; }}
.ins-tag.warn {{ color: {ACCENT}; }}
.ins b {{ color: {INK}; }}

/* sidebar */
[data-testid="stSidebar"] {{ background: white; border-right: 1px solid {LINE}; }}
[data-testid="stSidebar"] .side-title {{ font-size: 11px; font-weight: 600; letter-spacing: .12em; text-transform: uppercase;
                                         color: {MUTED}; margin: 6px 0 12px 0; }}
[data-testid="stSidebar"] label p {{ font-size: 12.5px; font-weight: 500; color: {TEXT}; }}

/* tables & buttons */
[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 8px; }}
.stDownloadButton button, .stButton button {{ border-radius: 8px; border: 1px solid {LINE}; font-weight: 500; font-size: 13px; }}
.stButton button:hover, .stDownloadButton button:hover {{ border-color: {PRIMARY}; color: {PRIMARY}; }}

/* assistant */
.chip-row {{ font-size: 11px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: {MUTED};
             margin: 8px 0 6px 0; }}
.spec {{ font-size: 11.5px; color: {MUTED}; margin-top: 4px; font-variant-numeric: tabular-nums; }}
[data-testid="stChatMessage"] {{ background: white; border: 1px solid {LINE}; border-radius: 10px; padding: 12px 14px; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(overline: str, title: str, subtitle: str, meta: list[tuple[str, str]]) -> None:
    meta_html = "<br>".join(f"{html.escape(k)} <b>{html.escape(v)}</b>" for k, v in meta)
    st.markdown(
        f"""<div class="app-header"><div><div class="app-overline">{html.escape(overline)}</div>
        <div class="app-title">{html.escape(title)}</div><div class="app-sub">{html.escape(subtitle)}</div></div>
        <div class="app-meta">{meta_html}</div></div>""",
        unsafe_allow_html=True,
    )


def section(title: str, note: str | None = None) -> None:
    st.markdown(f'<div class="sec"><h3>{html.escape(title)}</h3>'
                + (f"<p>{html.escape(note)}</p>" if note else "") + "</div>", unsafe_allow_html=True)


def card_title(title: str, note: str | None = None) -> None:
    st.markdown(f'<div class="card-title">{html.escape(title)}</div>'
                + (f'<div class="card-note">{html.escape(note)}</div>' if note else ""), unsafe_allow_html=True)


def kpi(label: str, value: str, delta: float | None = None, delta_text: str | None = None,
        higher_is_better: bool = True, foot: str | None = None) -> str:
    """Return HTML for a KPI tile. ``delta`` is a relative change (0.05 = +5%)."""
    if delta is None or delta != delta:  # None or NaN
        d_html = f'<span class="kpi-flat">{html.escape(foot or "")}</span>'
    else:
        good = (delta >= 0) == higher_is_better
        cls = "kpi-flat" if abs(delta) < 0.0005 else ("kpi-up" if good else "kpi-down")
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "–")
        txt = delta_text or f"{delta * 100:+.1f}%"
        d_html = f'<span class="{cls}">{arrow} {html.escape(txt)}</span> <span>{html.escape(foot or "")}</span>'
    return (f'<div class="kpi"><div class="kpi-label">{html.escape(label)}</div>'
            f'<div class="kpi-value">{html.escape(value)}</div><div class="kpi-foot">{d_html}</div></div>')


def kpi_row(tiles: list[str]) -> None:
    cols = st.columns(len(tiles))
    for c, t in zip(cols, tiles):
        c.markdown(t, unsafe_allow_html=True)


def md_bold_to_html(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(text))


def insight_list(items: list[tuple[str, str]], tags: dict[str, str]) -> None:
    rows = "".join(f'<div class="ins"><div class="ins-tag {sev}">{tags.get(sev, "Note")}</div>'
                   f"<div>{md_bold_to_html(txt)}</div></div>" for sev, txt in items)
    st.markdown(rows, unsafe_allow_html=True)


def style_fig(fig, height: int = 360, title: str | None = None):
    fig.update_layout(height=height, title=dict(text=title, font=dict(size=13, color=INK), x=0, xanchor="left")
                      if title else None)
    return fig
