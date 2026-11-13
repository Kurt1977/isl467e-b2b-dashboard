"""
ISL 467E – Management Engineering Applications with Python
Homework I – Decision Support Dashboard for a B2B Book Distributor

Run locally:  streamlit run streamlit_app.py

Module layout
-------------
data_pipeline.py  data loading & preprocessing (cached)
analytics.py      KPIs, RFM + K-Means, basket rules, ABC-XYZ, inventory policy, forecasting, insights
assistant.py      natural-language Data Assistant (rule-based parser + optional LLM parser)
ui.py             visual system (theme, Plotly template, components)
streamlit_app.py  page layout and interactions (this file)
"""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Sales Analytics · B2B Book Distribution", layout="wide",
                   initial_sidebar_state="expanded")

import analytics as an  # noqa: E402  (imported after set_page_config on purpose)
import assistant as asst  # noqa: E402
import ui  # noqa: E402
from data_pipeline import load_and_clean  # noqa: E402

ui.inject_css()
ui.reset_cards()
WEEKDAYS = an.WEEKDAYS
MONEY = dict(tickprefix="₺", tickformat="~s")


def to_excel(frames: dict[str, pd.DataFrame]) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, f in frames.items():
            f.to_excel(w, sheet_name=name[:31], index=False)
    return buf.getvalue()


# =============================================================================
# Data & filters (header bar with a filter popover)
# =============================================================================
df, prep_log, quality = load_and_clean()
dmin, dmax = df["InvoiceDate"].min().date(), df["InvoiceDate"].max().date()
city_order = df.groupby("City", observed=True)["Revenue"].sum().sort_values(ascending=False).index.tolist()
cat_order = df.groupby("Category", observed=True)["Revenue"].sum().sort_values(ascending=False).index.tolist()

h1, h2, h3 = st.columns([5, 3, 1.3], vertical_alignment="center")
with h3:
    with st.popover("Filters", use_container_width=True):
        dr = st.date_input("Period", value=(dmin, dmax), min_value=dmin, max_value=dmax)
        sel_city = st.multiselect("City", city_order, placeholder="All cities")
        sel_cat = st.multiselect("Category", cat_order, placeholder="All categories")
        sel_ctype = st.multiselect("Customer type", sorted(df["CustomerType"].cat.categories), placeholder="All types")
        sel_doc = st.multiselect("Document type", sorted(df["DocType"].cat.categories), placeholder="All document types")
        show_n = st.slider("Items in rankings", 5, 30, 10)
start, end = (dr if isinstance(dr, (list, tuple)) and len(dr) == 2 else (dmin, dmax))

mask = (df["InvoiceDate"].dt.date >= start) & (df["InvoiceDate"].dt.date <= end)
for col, sel in [("City", sel_city), ("Category", sel_cat), ("CustomerType", sel_ctype), ("DocType", sel_doc)]:
    if sel:
        mask &= df[col].isin(sel)
fdf = df[mask]
active_filters = sum(bool(x) for x in [sel_city, sel_cat, sel_ctype, sel_doc]) + int((start, end) != (dmin, dmax))

with h1:
    st.markdown('<div class="hdr-title">Sales Analytics</div>'
                '<div class="hdr-sub">B2B book distribution · decision support for sales, marketing and supply chain</div>',
                unsafe_allow_html=True)
with h2:
    st.markdown(f'<div style="text-align:right"><span class="pill"><span class="dot"></span>'
                f'{start:%d %b %Y} – {end:%d %b %Y} · {len(fdf):,} lines'
                f'{f" · {active_filters} filter(s)" if active_filters else ""}</span></div>', unsafe_allow_html=True)
st.write("")

if fdf.empty:
    st.warning("No transactions match the current filters. Widen the selection in the Filters menu.")
    st.stop()

# Shared analytical artefacts (cached)
rfm = an.rfm_segmentation(fdf[["CustomerID", "InvoiceDate", "InvoiceNo", "Revenue", "Quantity", "City"]],
                          st.session_state.get("k_clusters", 4))
rules = an.category_associations(fdf[["InvoiceNo", "Category", "Revenue"]], 40, st.session_state.get("min_sup", 0.01))
prod = an.abc_xyz(fdf[["StockCode", "Description", "Category", "Revenue", "Quantity", "CustomerID", "Week"]])

tabs = st.tabs(["Overview", "Sales", "Customers & Marketing", "Supply Chain", "Forecast", "Insights", "Data Quality"])

# =============================================================================
# Overview
# =============================================================================
with tabs[0]:
    k = an.kpis(fdf)
    last_m = fdf["Month"].max()
    prev_m = last_m - pd.DateOffset(months=1)
    kc = an.kpis(fdf[fdf["Month"] == last_m])
    kp = an.kpis(fdf[fdf["Month"] == prev_m]) if (fdf["Month"] == prev_m).any() else None

    def rel(key):
        return None if kp is None or not kp[key] else kc[key] / kp[key] - 1

    mon = fdf.groupby("Month").agg(Revenue=("Revenue", "sum"), Orders=("InvoiceNo", "nunique")).reset_index()
    ctype_share = fdf.groupby("CustomerType", observed=True)["Revenue"].sum()
    ctype_share = (ctype_share / ctype_share.sum()).sort_values(ascending=False)
    hero_stats = [(f"{n}", f"{v * 100:.0f}%") for n, v in ctype_share.head(3).items()]
    st.write("")
    c1, c2, c3 = st.columns([1.25, 1, 1.05], gap="medium")
    with c1:
        d_rev = rel("revenue")
        delta_txt = (f"{'↑' if d_rev >= 0 else '↓'} {abs(d_rev) * 100:.1f}% in {last_m:%B} vs {prev_m:%B}"
                     if d_rev is not None else "")
        st.markdown(ui.hero_card("Net revenue", an.fmt_money(k["revenue"]), delta_txt, mon["Revenue"].tolist(),
                                 hero_stats), unsafe_allow_html=True)
    with c2, ui.card():
        orders_ct = fdf.groupby("CustomerType", observed=True)["InvoiceNo"].nunique().sort_values(ascending=False)
        top2 = orders_ct.head(2)
        cc1, cc2 = st.columns([1, 1.1])
        with cc1:
            legend = "".join(
                f'<div class="legend-row"><span class="legend-dot" style="background:{col_}"></span>{name}'
                f'<b>{v / orders_ct.sum() * 100:.0f}%</b></div>'
                for (name, v), col_ in zip(top2.items(), [ui.CORAL, ui.YELLOW]))
            st.markdown(f'<div class="stat-label">Orders</div><div class="stat-value">{k["orders"]:,}</div>'
                        f'{ui.badge(rel("orders"))}{legend}', unsafe_allow_html=True)
        with cc2:
            fig = go.Figure(go.Pie(labels=orders_ct.index.astype(str), values=orders_ct.values, hole=0.72,
                                   marker=dict(colors=[ui.CORAL, ui.YELLOW, ui.PINK, ui.INK]), sort=False,
                                   textinfo="none", direction="clockwise"))
            fig.update_layout(template=ui.TEMPLATE)
            fig.update_layout(height=200, showlegend=False, margin=dict(t=6, b=6, l=0, r=0),
                              annotations=[dict(text=f"<b>{an.fmt_money(k['aov'])}</b><br>avg order", showarrow=False,
                                                font=dict(size=12, color=ui.INK))])
            ui.chart(fig, use_container_width=True, config={"displayModeBar": False})
    with c3:
        last_cust = kc["customers"] / max(k["customers"], 1)
        st.markdown(ui.mini_tile(f"{k['customers']:,}", "customers", "Active customers",
                                 last_cust, f"{kc['customers']:,} ordered in {last_m:%B} ({last_cust:.0%})", ui.CORAL),
                    unsafe_allow_html=True)
        st.write("")
        st.markdown(ui.mini_tile(f"{k['discount'] * 100:.1f}%", "discount", "Average discount off list",
                                 k["discount"], f"List-price value {an.fmt_money(k['list'])}", ui.INK),
                    unsafe_allow_html=True)
        st.write("")
        mu = fdf.groupby("Month")["Quantity"].sum()
        st.markdown(ui.mini_tile(f"{k['units'] / 1e6:.2f}M" if k["units"] >= 1e6 else f"{k['units']:,.0f}", "units",
                                 "Units sold", mu.iloc[-1] / mu.max() if len(mu) else 0,
                                 f"{last_m:%B} at {mu.iloc[-1] / mu.max():.0%} of peak month · {k['skus']:,} SKUs",
                                 ui.PINK), unsafe_allow_html=True)

    c1, c2 = st.columns([2.1, 1], gap="medium")
    with c1, ui.card():
        ui.card_title("Top titles", f"Best sellers by net revenue · change = Q4 vs Q3")
        q = fdf["InvoiceDate"].dt.quarter
        tt = (fdf.groupby("Description", observed=True)
              .agg(Revenue=("Revenue", "sum"), Units=("Quantity", "sum"), Category=("Category", "first"))
              .nlargest(6, "Revenue"))
        q4 = fdf[q == 4].groupby("Description", observed=True)["Revenue"].sum()
        q3 = fdf[q == 3].groupby("Description", observed=True)["Revenue"].sum()
        rows = []
        for i, (title, r) in enumerate(tt.iterrows(), start=1):
            g_ = (q4.get(title, 0) / q3.get(title, np.nan) - 1) if q3.get(title, 0) else None
            rows.append({"title": title, "sub": f"#{i} · {r['Units']:,.0f} units", "cat": r["Category"],
                         "value": an.fmt_money(r["Revenue"]), "badge": ui.badge(g_)})
        st.markdown(ui.activity_list(rows), unsafe_allow_html=True)
    with c2, ui.card():
        ui.card_title("Order days", "Revenue by weekday")
        wd = fdf.groupby("Weekday")["Revenue"].sum().reindex(range(7), fill_value=0)
        colors = [ui.GREY_BAR] * 7
        colors[int(wd.values.argmax())] = ui.PINK
        fig = go.Figure()
        fig.update_layout(template=ui.TEMPLATE)
        fig.add_bar(x=[d[0] for d in WEEKDAYS], y=wd.values, marker=dict(color=colors, cornerradius=12),
                    hovertext=WEEKDAYS, hovertemplate="%{hovertext}: ₺%{y:,.0f}<extra></extra>")
        fig.add_scatter(x=[d[0] for d in WEEKDAYS], y=wd.values * 0.55, mode="lines", line=dict(color=ui.INK, width=1.6,
                        shape="spline"), hoverinfo="skip")
        fig.update_layout(height=270, showlegend=False, bargap=0.45, yaxis=dict(showticklabels=False, showgrid=False))
        ui.chart(fig, use_container_width=True, config={"displayModeBar": False})
        best_day = WEEKDAYS[int(wd.values.argmax())]
        st.markdown(f'<div class="legend-row" style="margin:0 0 6px 4px">Busiest day<b>{best_day} · '
                    f'{wd.max() / wd.sum():.0%} of revenue</b></div>', unsafe_allow_html=True)

    ui.section("Monthly performance", "Net revenue (bars) and number of orders (line).")
    with ui.card():
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.update_layout(template=ui.TEMPLATE)
        bar_colors = [ui.GREY_BAR] * len(mon)
        if len(mon):
            bar_colors[int(mon["Revenue"].values.argmax())] = ui.CORAL
        fig.add_bar(x=mon["Month"], y=mon["Revenue"], name="Net revenue", marker=dict(color=bar_colors, cornerradius=10))
        fig.add_scatter(x=mon["Month"], y=mon["Orders"], name="Orders", mode="lines+markers",
                        line=dict(color=ui.PINK, width=2.5, shape="spline"), marker=dict(size=6), secondary_y=True)
        fig.update_layout(height=340, hovermode="x unified")
        fig.update_xaxes(dtick="M1", tickformat="%b")
        fig.update_yaxes(secondary_y=False, **MONEY)
        fig.update_yaxes(secondary_y=True, showgrid=False, title_text="Orders")
        ui.chart(fig, use_container_width=True)

    ui.section("Revenue mix")
    c1, c2, c3 = st.columns(3, gap="medium")
    for col_, (title, dim, color) in zip([c1, c2, c3], [("By document type", "DocType", ui.CORAL),
                                                        ("By customer type", "CustomerType", ui.PINK),
                                                        ("Top cities", "City", ui.INK)]):
        with col_, ui.card():
            ui.card_title(title)
            mix = fdf.groupby(dim, observed=True)["Revenue"].sum().nlargest(6).sort_values().reset_index()
            mix[dim] = mix[dim].astype(str)
            fig = px.bar(mix, x="Revenue", y=dim, orientation="h", text_auto=".3s")
            fig.update_traces(marker=dict(color=color, cornerradius=8), textposition="outside", cliponaxis=False)
            fig.update_layout(height=250, yaxis_title=None, xaxis_title=None, xaxis=dict(showticklabels=False),
                              yaxis=dict(showgrid=False))
            ui.chart(fig, use_container_width=True)

# =============================================================================
# Sales
# =============================================================================
with tabs[1]:
    cat = (fdf.groupby("Category", observed=True)
           .agg(Revenue=("Revenue", "sum"), Units=("Quantity", "sum"), Orders=("InvoiceNo", "nunique"),
                Discount=("DiscountRate", "mean")).reset_index().sort_values("Revenue", ascending=False))
    cat["Share"] = cat["Revenue"] / cat["Revenue"].sum()

    ui.section("Category performance", "Area represents revenue; shading represents average discount.")
    c1, c2 = st.columns([3, 2])
    with c1, ui.card():
        fig = px.treemap(cat.head(30), path=[px.Constant("All categories"), "Category"], values="Revenue",
                         color="Discount", color_continuous_scale=ui.SEQ)
        fig.update_traces(marker=dict(cornerradius=3), root_color="white",
                          hovertemplate="<b>%{label}</b><br>Revenue ₺%{value:,.0f}<extra></extra>")
        fig.update_layout(height=430, margin=dict(t=10, l=0, r=0, b=0), coloraxis_colorbar=dict(title="Disc.", tickformat=".0%"))
        ui.chart(fig, use_container_width=True)
    with c2, ui.card():
        ui.card_title(f"Top {show_n} categories")
        st.dataframe(cat.head(show_n)[["Category", "Revenue", "Units", "Share"]], hide_index=True,
                     use_container_width=True, height=400,
                     column_config={"Revenue": st.column_config.NumberColumn(format="₺%.0f"),
                                    "Units": st.column_config.NumberColumn(format="%.0f"),
                                    "Share": st.column_config.ProgressColumn(format="%.3f", min_value=0,
                                                                             max_value=float(cat["Share"].max()))})

    ui.section("Seasonality", "Monthly revenue of the twelve largest categories.")
    with ui.card():
        topc = cat.head(12)["Category"]
        hm = (fdf[fdf["Category"].isin(topc)].groupby(["Category", "Month"], observed=True)["Revenue"].sum()
              .unstack(fill_value=0))
        hm = hm.loc[[c for c in topc if c in hm.index]]
        hm.columns = [f"{c:%b}" for c in hm.columns]
        fig = px.imshow(hm, aspect="auto", color_continuous_scale=ui.SEQ, labels=dict(color="Revenue", x="", y=""))
        fig.update_layout(height=420, margin=dict(t=10))
        ui.chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        ui.section("Best-selling titles", "Ranked by net revenue.")
        with ui.card():
            tp = (fdf.groupby("Description", observed=True)
                  .agg(Revenue=("Revenue", "sum"), Units=("Quantity", "sum"), Customers=("CustomerID", "nunique"))
                  .nlargest(show_n, "Revenue").reset_index())
            tp["Title"] = tp["Description"].astype(str).str.slice(0, 42)
            fig = px.bar(tp.sort_values("Revenue"), x="Revenue", y="Title", orientation="h",
                         hover_data={"Units": ":,.0f", "Customers": True, "Title": False, "Description": True})
            fig.update_traces(marker_color=ui.PRIMARY)
            fig.update_layout(height=max(340, 30 * show_n), yaxis_title=None, xaxis=MONEY)
            ui.chart(fig, use_container_width=True)
    with c2:
        ui.section("Order-day pattern", "Revenue by weekday and month.")
        with ui.card():
            wd = fdf.groupby(["Weekday", "Month"])["Revenue"].sum().unstack(fill_value=0)
            wd.index = [WEEKDAYS[i] for i in wd.index]
            wd.columns = [f"{c:%b}" for c in wd.columns]
            fig = px.imshow(wd, aspect="auto", color_continuous_scale=ui.SEQ_WARM, labels=dict(color="Revenue", x="", y=""))
            fig.update_layout(height=max(340, 30 * show_n), margin=dict(t=10))
            ui.chart(fig, use_container_width=True)

    @st.fragment
    def custom_breakdown() -> None:
        ui.section("Custom breakdown", "Choose a dimension and a measure.")
        with ui.card():
            c1, c2, c3 = st.columns([2, 2, 1])
            dims = {"Month": "Month", "Quarter": "Quarter", "City": "City", "Category": "Category",
                    "Document type": "DocType", "Customer type": "CustomerType"}
            metrics = {"Net revenue": ("Revenue", "sum"), "Units": ("Quantity", "sum"), "Orders": ("InvoiceNo", "nunique"),
                       "Customers": ("CustomerID", "nunique"), "Average discount": ("DiscountRate", "mean")}
            g = c1.selectbox("Dimension", list(dims), index=2)
            mname = c2.selectbox("Measure", list(metrics), index=0)
            ctype = c3.radio("Chart", ["Bar", "Line"], horizontal=True)
            col, agg = metrics[mname]
            dyn = fdf.groupby(dims[g], observed=True)[col].agg(agg).reset_index()
            dyn = dyn.sort_values(dims[g]) if g in ("Month", "Quarter") else dyn.nlargest(show_n, col)
            if g != "Month":
                dyn[dims[g]] = dyn[dims[g]].astype(str)
            fig = (px.bar(dyn, x=dims[g], y=col, text_auto=".3s") if ctype == "Bar"
                   else px.line(dyn, x=dims[g], y=col, markers=True))
            fig.update_traces(marker_color=ui.PRIMARY) if ctype == "Bar" else fig.update_traces(line_color=ui.PRIMARY)
            fig.update_layout(height=360, yaxis_title=mname, xaxis_title=None)
            ui.chart(fig, use_container_width=True)

    custom_breakdown()

# =============================================================================
# Customers & Marketing
# =============================================================================
with tabs[2]:
    ui.section("Customer segmentation",
               "Recency, frequency and monetary value are log-scaled, standardised and clustered with K-Means; "
               "clusters are ranked and named automatically.")
    st.slider("Number of segments", 2, 6, 4, key="k_clusters")
    seg = (rfm.groupby("Segment").agg(Customers=("CustomerID", "count"), Revenue=("Monetary", "sum"),
                                      AvgRecency=("Recency", "mean"), AvgOrders=("Frequency", "mean"),
                                      AvgRevenue=("Monetary", "mean"), ChurnRisk=("ChurnRisk", "mean"))
           .sort_values("Revenue", ascending=False).reset_index())
    seg["RevenueShare"] = seg["Revenue"] / seg["Revenue"].sum()
    seg_colors = {s: ui.PALETTE[i % len(ui.PALETTE)] for i, s in enumerate(seg["Segment"])}
    c1, c2 = st.columns([3, 2])
    with c1, ui.card():
        ui.card_title("Recency vs. frequency", "Bubble size represents revenue.")
        fig = px.scatter(rfm, x="Recency", y="Frequency", size="Monetary", color="Segment", log_y=True,
                         color_discrete_map=seg_colors, size_max=36, opacity=0.75,
                         hover_data={"CustomerID": True, "Monetary": ":,.0f", "City": True})
        fig.update_layout(height=400, xaxis_title="Days since last order", yaxis_title="Orders (log)")
        ui.chart(fig, use_container_width=True)
    with c2, ui.card():
        ui.card_title("Revenue share by segment")
        fig = px.bar(seg, x="RevenueShare", y="Segment", orientation="h", color="Segment",
                     color_discrete_map=seg_colors, text_auto=".0%")
        fig.update_layout(height=400, showlegend=False, xaxis_tickformat=".0%", xaxis_title=None, yaxis_title=None,
                          yaxis=dict(categoryorder="total ascending"))
        ui.chart(fig, use_container_width=True)
    st.dataframe(seg, hide_index=True, use_container_width=True,
                 column_config={"Revenue": st.column_config.NumberColumn(format="₺%.0f"),
                                "AvgRevenue": st.column_config.NumberColumn("Avg revenue per customer", format="₺%.0f"),
                                "AvgRecency": st.column_config.NumberColumn("Avg days since last order", format="%.0f"),
                                "AvgOrders": st.column_config.NumberColumn("Avg orders", format="%.1f"),
                                "ChurnRisk": st.column_config.NumberColumn("Share at churn risk", format="%.2f"),
                                "RevenueShare": st.column_config.ProgressColumn("Revenue share", format="%.2f",
                                                                                min_value=0, max_value=1)})

    c1, c2 = st.columns(2)
    with c1:
        ui.section("Win-back priority list", "High-value customers silent for more than twice their usual order gap.")
        wb = rfm[rfm["ChurnRisk"]].nlargest(show_n, "Monetary")[
            ["CustomerID", "City", "Segment", "Recency", "Frequency", "Monetary", "AvgGapDays"]]
        st.dataframe(wb, hide_index=True, use_container_width=True,
                     column_config={"Monetary": st.column_config.NumberColumn("Revenue", format="₺%.0f"),
                                    "Recency": st.column_config.NumberColumn("Days silent"),
                                    "Frequency": st.column_config.NumberColumn("Orders"),
                                    "AvgGapDays": st.column_config.NumberColumn("Usual gap (days)", format="%.0f")})
    with c2:
        ui.section("Customer concentration", "Cumulative share of revenue by share of customers.")
        with ui.card():
            cs = rfm["Monetary"].sort_values(ascending=False).reset_index(drop=True)
            par = pd.DataFrame({"Customers": np.arange(1, len(cs) + 1) / len(cs), "Revenue": cs.cumsum() / cs.sum()})
            fig = px.area(par, x="Customers", y="Revenue")
            fig.update_traces(line_color=ui.PRIMARY, fillcolor="rgba(239,123,92,0.15)")
            fig.add_hline(y=0.8, line_dash="dot", line_color=ui.ACCENT, annotation_text="80% of revenue",
                          annotation_font_color=ui.ACCENT)
            fig.update_layout(height=300, xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                              xaxis_title="Share of customers", yaxis_title="Cumulative revenue")
            ui.chart(fig, use_container_width=True)

    ui.section("Cross-selling opportunities",
               "Category pairs that appear on the same invoice more often than chance (association rules).")
    c1, c2 = st.columns([1, 3])
    with c1, ui.card():
        st.slider("Minimum support", 0.005, 0.05, 0.01, 0.005, format="%.3f", key="min_sup")
        min_lift = st.slider("Minimum lift", 1.0, 5.0, 1.5, 0.1)
        st.caption("Support: share of invoices with both categories. Confidence: probability of the second given "
                   "the first. Lift > 1: stronger than chance.")
    shown = rules[rules["Lift"] >= min_lift].head(show_n) if not rules.empty else rules
    with c2:
        if shown.empty:
            st.info("No rules meet the thresholds. Lower the support or lift.")
        else:
            st.dataframe(shown, hide_index=True, use_container_width=True,
                         column_config={"Support": st.column_config.NumberColumn(format="%.3f"),
                                        "Confidence": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
                                        "Lift": st.column_config.NumberColumn(format="%.2f")})

    ui.section("Discount effectiveness", "Average units per order line by discount band and customer type.")
    with ui.card():
        dd_ = fdf.assign(Band=pd.cut(fdf["DiscountRate"], an.DISCOUNT_BINS, labels=an.DISCOUNT_LABELS))
        dband = dd_.groupby(["Band", "CustomerType"], observed=True).agg(UnitsPerLine=("Quantity", "mean")).reset_index()
        fig = px.bar(dband, x="Band", y="UnitsPerLine", color="CustomerType", barmode="group",
                     labels={"Band": "Discount off list price", "UnitsPerLine": "Units per line", "CustomerType": ""})
        fig.update_layout(height=340)
        ui.chart(fig, use_container_width=True)

# =============================================================================
# Supply chain
# =============================================================================
with tabs[3]:
    ui.section("ABC–XYZ classification",
               "ABC: cumulative revenue share (A ≤ 80%, B ≤ 95%, C remainder). "
               "XYZ: coefficient of variation of weekly demand (X ≤ 0.5, Y ≤ 1.0, Z > 1.0).")
    c1, c2 = st.columns(2)
    with c1, ui.card():
        ui.card_title("Number of SKUs")
        mat = prod.pivot_table(index="ABC", columns="XYZ", values="StockCode", aggfunc="count", fill_value=0, observed=True)
        mat = mat.reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0)
        fig = px.imshow(mat, text_auto=",.0f", color_continuous_scale=ui.SEQ, labels=dict(color="SKUs", x="", y=""))
        fig.update_layout(height=320, margin=dict(t=10), coloraxis_showscale=False)
        ui.chart(fig, use_container_width=True)
    with c2, ui.card():
        ui.card_title("Share of revenue")
        rv = prod.pivot_table(index="ABC", columns="XYZ", values="Revenue", aggfunc="sum", fill_value=0, observed=True)
        rv = rv.reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0) / prod["Revenue"].sum()
        fig = px.imshow(rv, text_auto=".1%", color_continuous_scale=ui.SEQ, labels=dict(color="Share", x="", y=""))
        fig.update_layout(height=320, margin=dict(t=10), coloraxis_showscale=False)
        ui.chart(fig, use_container_width=True)

    @st.fragment
    def inventory_simulator() -> None:
        ui.section("Inventory policy simulator",
                   "Safety stock = z·σ·√(L+R); reorder point = μ·L + SS; EOQ = √(2DS/hC). Adjust the parameters to "
                   "see the impact on stock levels.")
        with ui.card():
            c = st.columns(5)
            service = c[0].slider("Service level", 0.80, 0.99, 0.95, 0.01)
            lead = c[1].slider("Lead time (weeks)", 0.5, 8.0, 2.0, 0.5)
            review = c[2].slider("Review period (weeks)", 0.0, 4.0, 1.0, 0.5)
            ocost = c[3].number_input("Order cost (₺ per order)", 10.0, 5000.0, 250.0, 10.0)
            hrate = c[4].slider("Holding cost (% per year)", 0.05, 0.50, 0.20, 0.01)
            classes = sorted(prod["Class"].unique())
            cls = st.multiselect("Classes included", classes, default=[x for x in ["AX", "AY", "AZ"] if x in classes])
        pol = an.inventory_policy(prod[prod["Class"].isin(cls)] if cls else prod, service, lead, review, ocost, hrate)
        unit_val = (pol["Revenue"] / pol["Units"]).fillna(0)
        ui.kpi_row([
            ui.kpi("SKUs in policy", f"{len(pol):,}", foot="selected classes"),
            ui.kpi("Total safety stock", f"{pol['SafetyStock'].sum():,.0f} units", foot=f"at {service:.0%} service level"),
            ui.kpi("Safety stock value", an.fmt_money((pol["SafetyStock"] * unit_val).sum()), foot="at average net price"),
            ui.kpi("Average reorder point", f"{pol['ReorderPoint'].mean():,.0f} units", foot=f"lead time {lead:g} weeks"),
        ])
        st.write("")
        st.dataframe(pol.nlargest(max(show_n, 15), "Revenue")[
                         ["Product", "Category", "Class", "WeeklyMean", "WeeklyStd", "SafetyStock", "ReorderPoint", "EOQ",
                          "OrdersPerYear"]],
                     hide_index=True, use_container_width=True,
                     column_config={"WeeklyMean": st.column_config.NumberColumn("Avg weekly demand", format="%.1f"),
                                    "WeeklyStd": st.column_config.NumberColumn("Demand σ", format="%.1f"),
                                    "SafetyStock": st.column_config.NumberColumn("Safety stock", format="%.0f"),
                                    "ReorderPoint": st.column_config.NumberColumn("Reorder point", format="%.0f"),
                                    "EOQ": st.column_config.NumberColumn("EOQ", format="%.0f"),
                                    "OrdersPerYear": st.column_config.NumberColumn("Orders / year", format="%.1f")})

    inventory_simulator()

    c1, c2 = st.columns(2)
    with c1:
        ui.section("Weekly unit demand", "Actual units with 4-week moving average.")
        with ui.card():
            wk = an.weekly_series(fdf, "Quantity").rename("Units").to_frame()
            wk["4-week average"] = wk["Units"].rolling(4).mean()
            fig = go.Figure()
            fig.update_layout(template=ui.TEMPLATE)
            fig.add_bar(x=wk.index, y=wk["Units"], name="Units", marker_color=ui.GREY_BAR)
            fig.add_scatter(x=wk.index, y=wk["4-week average"], name="4-week average", mode="lines",
                            line=dict(color=ui.PINK, width=2.5))
            fig.update_layout(height=320, bargap=0.15)
            ui.chart(fig, use_container_width=True)
    with c2:
        ui.section("Regional demand profile", "Order count vs. drop size for the 15 largest cities.")
        with ui.card():
            reg = fdf.groupby("City", observed=True).agg(Units=("Quantity", "sum"), Customers=("CustomerID", "nunique"),
                                                         Orders=("InvoiceNo", "nunique")).reset_index()
            reg = reg.nlargest(15, "Units")
            reg["UnitsPerOrder"] = reg["Units"] / reg["Orders"]
            fig = px.scatter(reg, x="Orders", y="UnitsPerOrder", size="Units", text="City", log_x=True, size_max=40,
                             labels={"UnitsPerOrder": "Units per order", "Orders": "Orders (log)"})
            fig.update_traces(textposition="top center", marker=dict(color=ui.PRIMARY, opacity=0.6),
                              textfont=dict(size=10, color=ui.MUTED))
            fig.update_layout(height=320)
            ui.chart(fig, use_container_width=True)

# =============================================================================
# Forecast
# =============================================================================
with tabs[4]:
    @st.fragment
    def forecast_panel() -> None:
        ui.section("Demand and revenue forecast",
                   "Three models are back-tested on the last 8 weeks. The model with the lowest MAPE is refitted on the "
                   "full history; the shaded band is an approximate 80% interval.")
        with ui.card():
            c = st.columns(4)
            target = c[0].selectbox("Target", ["Net revenue", "Units"])
            level = c[1].selectbox("Level", ["Total", "Category", "City"])
            choice = None
            if level == "Category":
                choice = c[2].selectbox("Category", cat_order[:40] if not sel_cat else sel_cat)
            elif level == "City":
                choice = c[2].selectbox("City", city_order[:30] if not sel_city else sel_city)
            else:
                c[2].selectbox("Scope", ["All filtered data"], disabled=True)
            horizon = c[3].slider("Horizon (weeks)", 4, 16, 8)
        col = "Revenue" if target == "Net revenue" else "Quantity"
        base = fdf if level == "Total" else fdf[fdf[level] == choice]
        ser = an.weekly_series(base, col)

        if len(ser) < 20:
            st.info("Not enough weekly history for a reliable forecast. Widen the period or the selection.")
        else:
            fc = an.run_forecast(ser, horizon)
            with ui.card():
                fig = go.Figure()
                fig.update_layout(template=ui.TEMPLATE)
                fig.add_scatter(x=np.concatenate([fc["future"], fc["future"][::-1]]),
                                y=np.concatenate([fc["upper"], fc["lower"][::-1]]), fill="toself", mode="lines",
                                fillcolor="rgba(226,85,138,0.13)", line=dict(width=0), name="80% interval", hoverinfo="skip")
                fig.add_scatter(x=ser.index, y=ser.values, name="Actual", mode="lines",
                                line=dict(color=ui.INK, width=2))
                for name, p in fc["test_preds"].items():
                    fig.add_scatter(x=fc["test_dates"], y=p, name=f"Back-test: {name}", mode="lines",
                                    line=dict(dash="dot", width=1.5),
                                    visible=True if name == fc["best"] else "legendonly")
                fig.add_scatter(x=fc["future"], y=fc["forecast"], name=f"Forecast ({fc['best']})", mode="lines+markers",
                                line=dict(color=ui.ACCENT, width=2.5), marker=dict(size=5))
                fig.add_vline(x=ser.index[-1], line_dash="dot", line_color=ui.MUTED)
                fig.update_layout(height=430, hovermode="x unified",
                                  legend=dict(orientation="h", y=-0.18, yanchor="top"),
                                  yaxis=MONEY if col == "Revenue" else {})
                ui.chart(fig, use_container_width=True)

            nxt, last4 = fc["forecast"][:4].sum(), ser.values[-4:].sum()
            best_row = fc["scores"].iloc[0]
            fmt = (lambda v: an.fmt_money(v)) if col == "Revenue" else (lambda v: f"{v:,.0f}")
            ui.kpi_row([
                ui.kpi("Next 4 weeks", fmt(nxt), nxt / last4 - 1 if last4 else None, foot="vs last 4 weeks"),
                ui.kpi(f"Next {horizon} weeks", fmt(fc["forecast"].sum()), foot="total forecast"),
                ui.kpi("Selected model", best_row["Model"].split(" (")[0], foot="lowest back-test error"),
                ui.kpi("Back-test MAPE", f"{best_row['MAPE (%)']:.1f}%", foot=f"MAE {fmt(best_row['MAE'])}"),
            ])
            c1, c2 = st.columns([2, 3])
            with c1:
                ui.section("Model comparison")
                st.dataframe(fc["scores"], hide_index=True, use_container_width=True,
                             column_config={"MAE": st.column_config.NumberColumn(format="%.0f"),
                                            "MAPE (%)": st.column_config.NumberColumn(format="%.1f")})
            with c2:
                ui.section("Forecast table")
                ft = pd.DataFrame({"Week starting": fc["future"].date, "Forecast": fc["forecast"],
                                   "Lower (80%)": fc["lower"], "Upper (80%)": fc["upper"]})
                st.dataframe(ft, hide_index=True, use_container_width=True,
                             column_config={k_: st.column_config.NumberColumn(format="%.0f")
                                            for k_ in ["Forecast", "Lower (80%)", "Upper (80%)"]})

    forecast_panel()

# =============================================================================
# Insights
# =============================================================================
with tabs[5]:
    tser = an.weekly_series(fdf, "Revenue")
    total_fc = an.run_forecast(tser, 8) if len(tser) >= 20 else None
    ins = an.generate_insights(fdf, rfm, prod, rules, total_fc)
    ui.section("Findings", "Generated from the filtered data; they update whenever a filter changes.")
    tags = {"info": "Finding", "good": "Opportunity", "warn": "Risk"}
    cols = st.columns(3)
    for col_, (dept, items) in zip(cols, ins.items()):
        with col_, ui.card():
            ui.card_title(dept)
            ui.insight_list(items, tags)

    ui.section("Recommended actions")
    actions = pd.DataFrame([
        ["Sales", "Build stock and sales capacity ahead of the September–October back-to-school peak; grow the "
                  "next-largest cities to reduce dependence on Istanbul.", "Monthly revenue, city revenue share"],
        ["Marketing", "Run a key-account programme for the top segment, automated win-back offers for customers at "
                      "churn risk, and bundles for high-lift category pairs.", "Retention rate, churn-risk count, basket size"],
        ["Supply chain", "Apply service-level based safety stock to class A SKUs, move CZ SKUs to make-to-order or "
                         "drop-shipping, and plan purchasing with the weekly forecast.", "Fill rate, inventory turns, forecast MAPE"],
    ], columns=["Function", "Action", "KPI to track"])
    st.dataframe(actions, hide_index=True, use_container_width=True)
    st.download_button(
        "Download analysis pack (Excel)",
        data=to_excel({"Segments": seg, "Customers_RFM": rfm.drop(columns=["FirstPurchase", "LastPurchase"]),
                       "Cross_sell_rules": rules.head(200) if not rules.empty else pd.DataFrame(),
                       "ABC_XYZ": prod.head(5000)}),
        file_name="b2b_decision_support_pack.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# =============================================================================
# Data Assistant (left rail)
# =============================================================================
try:
    api_key = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    api_key = None
vocab = asst.build_vocab(df)
if "chat" not in st.session_state:
    st.session_state.chat = []


def render_answer(res: dict, idx: int) -> None:
    st.markdown(f'<div class="msg-bot">{ui.md_bold_to_html(res["text"]).replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True)
    if res.get("fig") is not None:
        ui.chart(ui.dark_fig(res["fig"]), use_container_width=True, key=f"chatfig{idx}",
                        config={"displayModeBar": False})
    if res.get("df") is not None and len(res["df"]):
        st.markdown(ui.dark_table(res["df"]), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.download_button("CSV", res["df"].to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"assistant_result_{idx}.csv", mime="text/csv", key=f"csv{idx}",
                           use_container_width=True)
        c2.download_button("Excel", to_excel({"Result": res["df"]}), file_name=f"assistant_result_{idx}.xlsx",
                           key=f"xlsx{idx}", use_container_width=True,
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    sp = res.get("spec") or {}
    if sp:
        flt = asst.describe_filters(sp.get("filters", {}), "en")
        st.markdown(f'<div class="msg-spec">intent: {sp.get("intent")} · measure: {sp.get("metric")} · '
                    f'by: {sp.get("dim")}{" · " + ui.esc(flt) if flt else ""}</div>', unsafe_allow_html=True)


@st.fragment
def assistant_panel() -> None:
    mode = "Claude-assisted" if api_key else "Built-in query engine"
    st.markdown(f'<div class="brand"><div class="brand-mark">AI</div><div><div class="brand-name">Data Assistant</div>'
                f'<div class="brand-sub">Ask about the data · TR / EN · {mode}</div></div></div>',
                unsafe_allow_html=True)
    with st.form("ask", clear_on_submit=True, border=False):
        c1, c2 = st.columns([4, 1.3])
        typed = c1.text_input("Question", placeholder="Ask a question…", label_visibility="collapsed")
        asked = c2.form_submit_button("Ask", use_container_width=True)
    st.markdown('<div class="side-label">Try</div>', unsafe_allow_html=True)
    clicked = None
    for i, s_ in enumerate(asst.SUGGESTIONS[:6]):
        if st.button(s_, key=f"sug{i}", use_container_width=True):
            clicked = s_
    question = (typed if asked and typed.strip() else None) or clicked
    if question:
        spec = asst.llm_parse(question, vocab, api_key, st.secrets.get("ANTHROPIC_MODEL", "claude-haiku-5-5")) \
            if api_key else None
        spec = spec or asst.parse_question(question, vocab)
        try:
            result = asst.execute(spec, fdf, {"rfm": rfm, "rules": rules, "prod": prod})
        except Exception as exc:  # keep the app alive on unexpected questions
            result = {"text": f"I could not answer that ({type(exc).__name__}). Try naming a measure and a "
                              "dimension, e.g. 'revenue by city'.", "df": None, "fig": None, "spec": spec}
        st.session_state.chat.append({"q": question, "res": result})

    if st.session_state.chat:
        st.markdown('<div class="side-label">Conversation</div>', unsafe_allow_html=True)
        for i in range(len(st.session_state.chat) - 1, -1, -1):  # newest first
            turn = st.session_state.chat[i]
            st.markdown(f'<div class="msg-user">{ui.esc(turn["q"])}</div>', unsafe_allow_html=True)
            render_answer(turn["res"], i)
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state.chat = []
            st.rerun(scope="fragment")
    st.markdown(f'<div class="side-label" style="margin-top:22px">Scope</div>'
                f'<div class="brand-sub">Answers use the active filters · {len(fdf):,} lines · '
                f'source data {quality["raw_rows"]:,} raw / {quality["clean_rows"]:,} clean lines</div>',
                unsafe_allow_html=True)


with st.sidebar:
    assistant_panel()

# =============================================================================
# Data quality
# =============================================================================
with tabs[6]:
    ui.section("Preprocessing pipeline", "Every cleaning step with its effect on the number of rows.")
    ui.kpi_row([
        ui.kpi("Raw lines", f"{quality['raw_rows']:,}", foot="as delivered"),
        ui.kpi("Clean lines", f"{quality['clean_rows']:,}", foot=f"{quality['raw_rows'] - quality['clean_rows']:,} removed"),
        ui.kpi("Identical lines", f"{quality['exact_duplicates']:,}", foot="kept: repeat scans on the same invoice"),
        ui.kpi("Bulk-order lines", f"{quality['bulk_lines']:,}", foot="flagged, not removed"),
    ])
    st.write("")
    st.dataframe(prep_log, hide_index=True, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        ui.section("Missing values in raw data")
        with ui.card():
            miss = pd.Series(quality["missing"]).rename("Missing").reset_index().rename(columns={"index": "Column"})
            miss = miss[miss["Missing"] > 0]
            fig = px.bar(miss, x="Column", y="Missing", text_auto=",.0f")
            fig.update_traces(marker_color=ui.ACCENT)
            fig.update_layout(height=300, xaxis_title=None)
            ui.chart(fig, use_container_width=True)
    with c2:
        ui.section("Discount rate distribution", "Sample of up to 100,000 lines.")
        with ui.card():
            fig = px.histogram(fdf.sample(min(len(fdf), 100_000), random_state=1), x="DiscountRate", nbins=60)
            fig.update_traces(marker_color=ui.PRIMARY)
            fig.update_layout(height=300, xaxis_tickformat=".0%", xaxis_title="Discount off list price",
                              yaxis_title="Lines")
            ui.chart(fig, use_container_width=True)

    ui.section("Clean data preview")
    st.dataframe(fdf.head(200), use_container_width=True, hide_index=True)
    st.download_button("Download filtered clean data (CSV)", fdf.to_csv(index=False).encode("utf-8-sig"),
                       "b2b_clean_filtered.csv", "text/csv")

st.caption("ISL 467E – Management Engineering Applications with Python · Homework I · Streamlit, pandas, scikit-learn, Plotly")
