"""Visual system: colour tokens, Plotly template, global CSS and small HTML components."""

from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ----------------------------------------------------------------------------- tokens
INK = "#14161F"         # near-black (rail, buttons, headings)
TEXT = "#3B3F4C"
MUTED = "#8A8D98"
BG = "#ECE9E6"          # warm light grey page
CARD = "#F8F7F5"        # off-white cards
LINE = "#E3DFDB"
CORAL = "#EF7B5C"
PINK = "#E2558A"
PEACH = "#F7D9CF"       # pastel tile
YELLOW = "#F2C14E"
GREY_BAR = "#D9D4CF"
POSITIVE = "#2E9D62"
NEGATIVE = "#D9534F"

PRIMARY = CORAL
ACCENT = PINK
PALETTE = [CORAL, PINK, INK, YELLOW, "#B8B2AC", "#F4B39F", "#8E7DBE", "#5FB49C"]
SEQ = ["#FBF1EC", "#F7D3C4", "#F2AA8E", "#EE8466", "#E8678A", "#C9497A"]
SEQ_WARM = SEQ
GRADIENT = f"linear-gradient(135deg, {CORAL} 0%, {PINK} 100%)"

pio.templates["dss"] = go.layout.Template(
    layout=dict(
        font=dict(family="Poppins, 'Segoe UI', Helvetica, Arial, sans-serif", size=12, color=TEXT),
        colorway=PALETTE,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, linecolor="rgba(0,0,0,0)", ticks="", tickfont=dict(color=MUTED, size=11),
                   title=dict(font=dict(size=11, color=MUTED))),
        yaxis=dict(gridcolor="#E7E2DD", griddash="dot", zeroline=False, linecolor="rgba(0,0,0,0)",
                   tickfont=dict(color=MUTED, size=11), title=dict(font=dict(size=11, color=MUTED))),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11, color=TEXT), title=None),
        hoverlabel=dict(bgcolor=INK, bordercolor=INK, font=dict(size=12, color="white")),
        margin=dict(l=8, r=8, t=30, b=8),
        coloraxis=dict(colorbar=dict(thickness=8, outlinewidth=0, tickfont=dict(size=10, color=MUTED))),
        bargap=0.35,
    )
)
pio.templates.default = "plotly_white+dss"
px.defaults.template = "plotly_white+dss"
px.defaults.color_discrete_sequence = PALETTE
px.defaults.color_continuous_scale = SEQ

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, .stApp, .stApp p, .stApp label, .stApp input, .stApp textarea, .stApp button, .stApp li,
.stMarkdown, .stMarkdown div, .stMarkdown span {{
    font-family: 'Poppins', 'Segoe UI', Helvetica, Arial, sans-serif;
}}
[data-testid="stIconMaterial"], .material-symbols-rounded {{ font-family: 'Material Symbols Rounded' !important; }}
#MainMenu, footer, [data-testid="stDecoration"] {{ visibility: hidden; height: 0; }}
[data-testid="stHeader"] {{ background: transparent; }}
.stApp, [data-testid="stAppViewContainer"] {{ background: {BG}; }}
.block-container {{ padding-top: 2.6rem; padding-bottom: 3rem; max-width: 1480px; }}
h1, h2, h3, h4 {{ color: {INK}; }}

/* ---------- header ---------- */
.hdr-title {{ font-size: 34px; font-weight: 600; color: {INK}; letter-spacing: -0.02em; line-height: 1.1; }}
.hdr-sub {{ font-size: 13px; color: {MUTED}; margin-top: 4px; }}
.pill {{ display:inline-flex; align-items:center; gap:8px; background:{CARD}; border-radius: 999px; padding: 8px 16px;
         font-size: 12.5px; color: {TEXT}; box-shadow: 0 1px 0 rgba(0,0,0,.03); }}
.pill .dot {{ width:18px; height:18px; border-radius:50%; background:{INK}; display:inline-block; }}
[data-testid="stPopover"] > div > button, [data-testid="stPopover"] button {{
    background: {INK} !important; color: white !important; border-radius: 999px !important; border: none !important;
    padding: 8px 20px !important; font-weight: 500 !important; }}
[data-testid="stPopover"] button p {{ color: white !important; }}

/* ---------- tabs as pills ---------- */
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; background: {CARD}; padding: 6px; border-radius: 999px; width: fit-content; }}
.stTabs [data-baseweb="tab"] {{ height: 38px; padding: 0 18px; border-radius: 999px; background: transparent;
                                font-weight: 500; color: {MUTED}; font-size: 13.5px; }}
.stTabs [aria-selected="true"] {{ background: {INK}; color: white !important; }}
.stTabs [aria-selected="true"] p {{ color: white !important; }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}

/* ---------- cards ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {{ background: {CARD}; border: none !important; border-radius: 24px;
                                               padding: 6px 8px; box-shadow: 0 1px 2px rgba(20,22,31,.04); }}
.sec {{ margin: 26px 0 10px 2px; }}
.sec h3 {{ font-size: 17px; font-weight: 600; margin: 0; color: {INK}; }}
.sec p {{ font-size: 12.5px; color: {MUTED}; margin: 2px 0 0 0; }}
.card-title {{ font-size: 15px; font-weight: 600; color: {INK}; margin: 4px 0 2px 2px; }}
.card-note {{ font-size: 12px; color: {MUTED}; margin: 0 0 6px 2px; }}

/* hero gradient card */
.hero {{ background: {GRADIENT}; border-radius: 26px; padding: 22px 24px 20px 24px; color: white; height: 100%;
         box-shadow: 0 10px 30px rgba(226,85,138,.25); }}
.hero-label {{ font-size: 14px; opacity: .9; display:flex; justify-content:space-between; }}
.hero-value {{ font-size: 38px; font-weight: 600; letter-spacing: -0.02em; margin: 6px 0 2px 0; }}
.hero-delta {{ font-size: 12.5px; opacity: .95; }}
.hero svg {{ display:block; margin: 8px 0 10px 0; }}
.hero-stats {{ display:flex; justify-content:space-between; border-top: 1px dashed rgba(255,255,255,.35); padding-top: 12px; }}
.hero-stats div {{ flex:1; padding-left: 12px; border-left: 1px dashed rgba(255,255,255,.35); }}
.hero-stats div:first-child {{ border-left: none; padding-left: 0; }}
.hero-stats span {{ display:block; font-size: 11.5px; opacity:.85; }}
.hero-stats b {{ font-size: 18px; font-weight: 600; }}

/* stat card (white) */
.stat {{ background:{CARD}; border-radius: 24px; padding: 20px 22px; height: 100%; }}
.stat-label {{ font-size: 14px; color: {INK}; font-weight: 500; display:flex; justify-content:space-between; }}
.stat-value {{ font-size: 30px; font-weight: 600; color: {INK}; margin: 6px 0; letter-spacing: -0.02em; }}
.legend-row {{ font-size: 11.5px; color: {MUTED}; margin-top: 8px; }}
.legend-row b {{ display:block; color:{INK}; font-size: 15px; font-weight:600; }}
.legend-dot {{ display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:6px; }}

/* mini tile (pastel badge + progress) */
.mini {{ background:{CARD}; border-radius: 24px; padding: 16px 18px; display:flex; gap: 16px; align-items:center; height:100%; }}
.mini-badge {{ background:{PEACH}; border-radius: 18px; min-width: 86px; height: 70px; display:flex; flex-direction:column;
               align-items:center; justify-content:center; font-weight:600; color:{INK}; font-size: 19px; }}
.mini-badge small {{ font-size: 10.5px; font-weight: 500; color: {TEXT}; margin-top: 2px; }}
.mini-body {{ flex:1; }}
.mini-title {{ font-size: 14px; font-weight: 500; color: {INK}; margin-bottom: 10px; }}
.bar {{ height: 9px; border-radius: 99px; background: #ECE7E2; overflow: hidden; }}
.bar > span {{ display:block; height:100%; border-radius: 99px; }}
.mini-foot {{ font-size: 11px; color: {MUTED}; margin-top: 6px; }}

/* KPI tiles (other pages) */
.kpi {{ background:{CARD}; border-radius: 22px; padding: 16px 18px 14px 18px; height: 100%; }}
.kpi-label {{ font-size: 12.5px; font-weight: 500; color: {MUTED}; }}
.kpi-value {{ font-size: 26px; font-weight: 600; color: {INK}; margin: 4px 0 6px 0; letter-spacing: -0.02em; }}
.kpi-foot {{ font-size: 11.5px; color: {MUTED}; }}
.badge {{ display:inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; margin-right: 4px; }}
.badge.up {{ background:#E1F3E8; color:{POSITIVE}; }}
.badge.down {{ background:#FBE3E1; color:{NEGATIVE}; }}
.badge.flat {{ background:#EEEAE6; color:{MUTED}; }}

/* activity list */
.act {{ display:grid; grid-template-columns: 44px 1fr 150px 90px 80px; align-items:center; gap: 12px;
        padding: 12px 14px; border-radius: 16px; font-size: 13px; color: {TEXT}; }}
.act:nth-child(odd) {{ background: white; }}
.act-thumb {{ width: 40px; height: 40px; border-radius: 12px; display:flex; align-items:center; justify-content:center;
              color: white; font-weight: 600; font-size: 13px; }}
.act-title {{ color:{INK}; font-weight: 500; white-space: nowrap; overflow:hidden; text-overflow: ellipsis; }}
.act-sub {{ font-size: 11px; color:{MUTED}; }}
.act-cat {{ font-size: 12px; font-weight: 500; color:{INK}; white-space: nowrap; overflow:hidden; text-overflow: ellipsis; }}
.act-num {{ font-weight: 600; color:{INK}; }}

/* insight list */
.ins {{ display:flex; gap: 12px; padding: 12px 4px; border-bottom: 1px dashed {LINE}; font-size: 13.5px; color:{TEXT}; line-height:1.55; }}
.ins:last-child {{ border-bottom: none; }}
.ins-tag {{ flex: 0 0 auto; height: fit-content; font-size: 10.5px; font-weight: 600; padding: 3px 10px; border-radius: 999px; }}
.ins-tag.info {{ background:#EEEAE6; color:{TEXT}; }}
.ins-tag.good {{ background:#E1F3E8; color:{POSITIVE}; }}
.ins-tag.warn {{ background:#FBE3E1; color:{NEGATIVE}; }}
.ins b {{ color:{INK}; }}

/* tables, buttons, inputs (main) */
[data-testid="stDataFrame"] {{ border-radius: 16px; overflow: hidden; }}
.stDownloadButton button {{ border-radius: 999px; background:{INK}; color:white; border:none; font-weight:500; }}
.stDownloadButton button p {{ color: white; }}

/* ---------- sidebar = assistant rail ---------- */
[data-testid="stSidebar"] {{ background: {INK}; min-width: 400px !important; max-width: 400px !important; }}
[data-testid="stSidebar"] > div {{ background: {INK}; }}
[data-testid="stSidebar"] * {{ color: #E8E9EE; }}
[data-testid="stSidebarHeader"] {{ height: 0; padding: 0; }}
.brand {{ display:flex; align-items:center; gap: 12px; margin: 4px 0 18px 0; }}
.brand-mark {{ width: 40px; height: 40px; border-radius: 12px; background: {GRADIENT}; display:flex; align-items:center;
               justify-content:center; font-weight:700; color:white; font-size: 15px; }}
.brand-name {{ font-size: 16px; font-weight: 600; color: white; }}
.brand-sub {{ font-size: 11.5px; color: #9A9DAA !important; }}
.side-label {{ font-size: 11px; letter-spacing: .1em; text-transform: uppercase; color: #8C8FA0 !important; margin: 14px 0 8px 0; }}
[data-testid="stSidebar"] .stButton button {{ background: #22253A; border: 1px solid #2C3048; border-radius: 14px;
    text-align: left; justify-content: flex-start; font-size: 12px; padding: 8px 12px; min-height: 0; }}
[data-testid="stSidebar"] .stButton button:hover {{ border-color: {CORAL}; }}
[data-testid="stSidebar"] .stButton button p {{ font-size: 12px; color: #D9DAE2 !important; }}
[data-testid="stSidebar"] [data-testid="stTextInput"] input {{ background: #22253A; border: 1px solid #2C3048;
    border-radius: 14px; color: white; }}
[data-testid="stSidebar"] [data-testid="stTextInput"] > div {{ background: transparent; border: none; }}
[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button {{ background: {GRADIENT}; border: none; border-radius: 14px; }}
[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] button p {{ color: white !important; font-weight: 600; }}
[data-testid="stSidebar"] .stDownloadButton button {{ background: #22253A; border: 1px solid #2C3048; font-size: 11.5px; }}
.msg-user {{ background: {GRADIENT}; color: white !important; border-radius: 16px 16px 4px 16px; padding: 9px 13px;
             font-size: 13px; margin: 14px 0 8px 40px; }}
.msg-bot {{ background: #1E2133; border: 1px solid #2C3048; border-radius: 16px 16px 16px 4px; padding: 11px 13px;
            font-size: 13px; line-height: 1.55; }}
.msg-bot b {{ color: white; }}
.msg-spec {{ font-size: 10.5px; color: #7E8193 !important; margin-top: 6px; }}
.dtable {{ width:100%; border-collapse: collapse; font-size: 11.5px; margin-top: 8px; }}
.dtable th {{ text-align:left; color:#9A9DAA !important; font-weight:500; padding: 5px 6px; border-bottom: 1px solid #2C3048; }}
.dtable td {{ padding: 5px 6px; border-bottom: 1px solid #23263A; color:#E8E9EE; }}
.dtable td.num, .dtable th.num {{ text-align:right; font-variant-numeric: tabular-nums; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(x) -> str:
    return html.escape(str(x))


def md_bold_to_html(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(text))


def section(title: str, note: str | None = None) -> None:
    st.markdown(f'<div class="sec"><h3>{esc(title)}</h3>' + (f"<p>{esc(note)}</p>" if note else "") + "</div>",
                unsafe_allow_html=True)


def card_title(title: str, note: str | None = None) -> None:
    st.markdown(f'<div class="card-title">{esc(title)}</div>'
                + (f'<div class="card-note">{esc(note)}</div>' if note else ""), unsafe_allow_html=True)


def badge(delta: float | None, higher_is_better: bool = True, text: str | None = None) -> str:
    if delta is None or delta != delta:
        return ""
    good = (delta >= 0) == higher_is_better
    cls = "flat" if abs(delta) < 0.0005 else ("up" if good else "down")
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "–")
    return f'<span class="badge {cls}">{arrow} {esc(text or f"{abs(delta) * 100:.1f}%")}</span>'


def kpi(label: str, value: str, delta: float | None = None, delta_text: str | None = None,
        higher_is_better: bool = True, foot: str | None = None) -> str:
    return (f'<div class="kpi"><div class="kpi-label">{esc(label)}</div><div class="kpi-value">{esc(value)}</div>'
            f'<div class="kpi-foot">{badge(delta, higher_is_better, delta_text)}{esc(foot or "")}</div></div>')


def kpi_row(tiles: list[str]) -> None:
    cols = st.columns(len(tiles))
    for c, t in zip(cols, tiles):
        c.markdown(t, unsafe_allow_html=True)


def sparkline_svg(values, width=420, height=70) -> str:
    """Two smooth white lines (actual + 3-period moving average) as inline SVG."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return ""

    def path(arr):
        lo, hi = np.nanmin(v), np.nanmax(v)
        rng = (hi - lo) or 1
        xs = np.linspace(4, width - 4, len(arr))
        ys = height - 6 - (arr - lo) / rng * (height - 12)
        d = f"M{xs[0]:.1f},{ys[0]:.1f}"
        for i in range(1, len(arr)):
            cx = (xs[i - 1] + xs[i]) / 2
            d += f" C{cx:.1f},{ys[i - 1]:.1f} {cx:.1f},{ys[i]:.1f} {xs[i]:.1f},{ys[i]:.1f}"
        return d

    ma = pd.Series(v).rolling(3, min_periods=1).mean().to_numpy()
    return (f'<svg width="100%" height="{height}" viewBox="0 0 {width} {height}" preserveAspectRatio="none">'
            f'<path d="{path(ma)}" fill="none" stroke="rgba(255,255,255,.45)" stroke-width="2"/>'
            f'<path d="{path(v)}" fill="none" stroke="white" stroke-width="2.4"/></svg>')


def hero_card(label: str, value: str, delta_html: str, spark: list[float], stats: list[tuple[str, str]]) -> str:
    st_html = "".join(f"<div><span>{esc(k)}</span><b>{esc(v)}</b></div>" for k, v in stats)
    return (f'<div class="hero"><div class="hero-label"><span>{esc(label)}</span><span>•••</span></div>'
            f'<div class="hero-value">{esc(value)}</div><div class="hero-delta">{delta_html}</div>'
            f'{sparkline_svg(spark)}<div class="hero-stats">{st_html}</div></div>')


def mini_tile(badge_value: str, badge_sub: str, title: str, pct: float, foot: str, color: str = CORAL) -> str:
    pct = float(np.clip(pct, 0, 1))
    return (f'<div class="mini"><div class="mini-badge">{esc(badge_value)}<small>{esc(badge_sub)}</small></div>'
            f'<div class="mini-body"><div class="mini-title">{esc(title)}</div>'
            f'<div class="bar"><span style="width:{pct * 100:.0f}%;background:{color}"></span></div>'
            f'<div class="mini-foot">{esc(foot)}</div></div></div>')


THUMB_COLORS = [CORAL, PINK, YELLOW, "#8E7DBE", "#5FB49C", INK]


def activity_list(rows: list[dict]) -> str:
    out = []
    for i, r in enumerate(rows):
        initials = "".join(w[0] for w in str(r["title"]).split()[:2]).upper()
        out.append(
            f'<div class="act"><div class="act-thumb" style="background:{THUMB_COLORS[i % len(THUMB_COLORS)]}">{esc(initials)}</div>'
            f'<div><div class="act-title">{esc(r["title"])}</div><div class="act-sub">{esc(r["sub"])}</div></div>'
            f'<div class="act-cat">{esc(r["cat"])}</div><div class="act-num">{esc(r["value"])}</div>'
            f'<div>{r["badge"]}</div></div>')
    return "".join(out)


def insight_list(items: list[tuple[str, str]], tags: dict[str, str]) -> None:
    rows = "".join(f'<div class="ins"><span class="ins-tag {sev}">{tags.get(sev, "Note")}</span>'
                   f"<div>{md_bold_to_html(txt)}</div></div>" for sev, txt in items)
    st.markdown(rows, unsafe_allow_html=True)


def dark_table(df: pd.DataFrame, max_rows: int = 10) -> str:
    """Compact HTML table for the dark assistant rail."""
    d = df.head(max_rows)
    head = "".join(f'<th class="{"num" if pd.api.types.is_numeric_dtype(d[c]) else ""}">{esc(c)}</th>' for c in d.columns)
    body = []
    for _, r in d.iterrows():
        cells = []
        for c in d.columns:
            v = r[c]
            if isinstance(v, pd.Timestamp):
                txt, cls = v.strftime("%b %Y"), ""
            elif isinstance(v, (int, float, np.integer, np.floating)):
                fv = float(v)
                txt = f"{fv:.1%}" if (abs(fv) <= 1 and c in ("Share", "Pay", "Ortalama indirim", "Avg. discount",
                                                            "Revenue share", "Support", "Confidence")) else (
                    f"{fv:,.0f}" if abs(fv) >= 100 else f"{fv:,.2f}")
                cls = "num"
            else:
                s = str(v)
                txt, cls = (s[:34] + "…" if len(s) > 35 else s), ""
            cells.append(f'<td class="{cls}">{esc(txt)}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    more = f'<div class="msg-spec">Showing {len(d)} of {len(df)} rows · full table in the download.</div>' \
        if len(df) > max_rows else ""
    return f'<table class="dtable"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>{more}'


def dark_fig(fig, height: int = 230):
    """Restyle a Plotly figure for the dark assistant rail."""
    fig.update_layout(height=height, margin=dict(l=4, r=4, t=24, b=4), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#C9CBD6", size=10),
                      legend=dict(font=dict(color="#C9CBD6", size=10)))
    fig.update_xaxes(tickfont=dict(color="#9A9DAA", size=9), title=None)
    fig.update_yaxes(tickfont=dict(color="#9A9DAA", size=9), gridcolor="#2C3048", title=None)
    return fig
