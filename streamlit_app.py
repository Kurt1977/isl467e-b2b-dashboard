"""
ISL 467E – Management Engineering Applications with Python
Homework I – Intelligent Decision-Support Dashboard for a B2B Book Distributor

Run locally:  streamlit run streamlit_app.py

Structure
---------
1. Configuration & styling
2. Data loading & preprocessing pipeline   (cached)
3. Analytical engines                      (KPIs, RFM + K-Means, basket analysis,
                                            ABC-XYZ, inventory policy, forecasting)
4. Insight generator                       (rule-based natural-language insights)
5. User interface                          (sidebar filters + department tabs)
"""

from __future__ import annotations

import glob
import os
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# =============================================================================
# 1. CONFIGURATION & STYLING
# =============================================================================
st.set_page_config(
    page_title="B2B Book Distributor | Decision Support Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PRIMARY = "#1F3A5F"
ACCENT = "#E07A5F"
TEAL = "#3D9A8B"
GOLD = "#F2B134"
PALETTE = [PRIMARY, TEAL, ACCENT, GOLD, "#81B29A", "#6D597A", "#B56576", "#4A7FB0", "#9C6644", "#8D99AE"]
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = PALETTE

DOC_TYPE_LABELS = {"BAD": "BAD", "BSR": "BSR", "DPE": "DPE", "DPF": "DPF", "MUH": "MUH"}
CITY_FIXES = {"IÇEL": "MERSIN", "K.MARAS": "KAHRAMANMARAS", "AFYON": "AFYONKARAHISAR"}
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

st.markdown(
    f"""
    <style>
    .block-container {{padding-top: 1.6rem; padding-bottom: 2rem;}}
    .hero {{background: linear-gradient(90deg, {PRIMARY} 0%, {TEAL} 100%); color: white;
            padding: 18px 24px; border-radius: 14px; margin-bottom: 14px;}}
    .hero h1 {{color: white; font-size: 1.75rem; margin: 0;}}
    .hero p {{color: #E8EEF4; margin: 4px 0 0 0; font-size: 0.95rem;}}
    div[data-testid="stMetric"] {{background: #F7F9FC; border: 1px solid #E3E8EF; border-left: 5px solid {PRIMARY};
            padding: 10px 14px; border-radius: 10px;}}
    div[data-testid="stMetricLabel"] p {{font-weight: 600; color: #4A5568;}}
    .insight {{background: #F7F9FC; border-left: 5px solid {TEAL}; padding: 10px 14px; border-radius: 8px;
               margin-bottom: 8px; font-size: 0.95rem;}}
    .insight.warn {{border-left-color: {ACCENT};}}
    .insight.good {{border-left-color: #2E9E5B;}}
    .section-note {{color: #6B7280; font-size: 0.88rem; margin-top: -6px;}}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# 2. DATA LOADING & PREPROCESSING
# =============================================================================
def _read_raw() -> pd.DataFrame:
    """Read the raw dataset. Prefers the original Excel file; falls back to the
    compressed CSV parts (the 40 MB Excel file is split/compressed for GitHub)."""
    xlsx = os.path.join(DATA_DIR, "B2B_Transaction_Data_.xlsx")
    if os.path.exists(xlsx):
        return pd.read_excel(xlsx, dtype={"StockCode": str})
    parts = sorted(glob.glob(os.path.join(DATA_DIR, "B2B_Transaction_Data_part*.csv.xz")))
    if not parts:
        raise FileNotFoundError("Dataset not found in ./data")
    return pd.concat([pd.read_csv(p, dtype={"StockCode": str}) for p in parts], ignore_index=True)


@st.cache_data(show_spinner="Loading and cleaning 664K transaction lines…")
def load_and_clean() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Full preprocessing pipeline. Returns (clean_df, step_log, quality_stats)."""
    df = _read_raw()
    log = []

    def step(name, action, before, after):
        log.append({"Step": name, "Action": action, "Rows before": before, "Rows after": after,
                    "Rows removed": before - after})

    n0 = len(df)
    quality = {
        "raw_rows": n0,
        "missing": df.isna().sum().to_dict(),
        "exact_duplicates": int(df.duplicated().sum()),
    }

    # 1) Types & text normalisation
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    for c in ["InvoiceNo", "CustomerID", "Description", "Category", "City"]:
        df[c] = df[c].astype("string").str.strip()
    df["City"] = df["City"].str.upper().replace(CITY_FIXES)
    for c in ["Quantity", "NetPrice", "UnitPrice"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    step("Type casting & text cleaning", "Parsed dates/numerics, trimmed whitespace, harmonised city names "
         "(IÇEL→MERSIN, K.MARAS→KAHRAMANMARAS, AFYON→AFYONKARAHISAR)", n0, len(df))

    # 2) Missing values
    n = len(df)
    df = df.dropna(subset=["InvoiceDate", "Quantity"])
    df["Category"] = df["Category"].fillna("Unknown")
    df["City"] = df["City"].fillna("UNKNOWN")
    step("Missing values", "Dropped rows without date/quantity; imputed missing Category='Unknown' and "
         "City='UNKNOWN' (kept for revenue completeness)", n, len(df))

    # 3) Invalid transactions (price corrections / zero-value lines)
    n = len(df)
    invalid = (df["Quantity"] <= 0) | (df["NetPrice"] <= 0) | (df["UnitPrice"] < 0.5)
    quality["invalid_lines"] = int(invalid.sum())
    df = df[~invalid]
    step("Invalid lines", "Removed zero-quantity / non-positive net price / ~0 list price lines "
         "(price corrections, free goods)", n, len(df))

    # 4) Price consistency: net price should not exceed list price
    n_over = int((df["NetPrice"] > df["UnitPrice"]).sum())
    quality["net_above_list"] = n_over
    df["NetPrice"] = np.minimum(df["NetPrice"], df["UnitPrice"])
    step("Price consistency", f"Capped NetPrice at UnitPrice for {n_over} lines", len(df), len(df))

    # 5) Outlier flagging (kept – bulk orders are genuine B2B behaviour)
    q1, q3 = df["Quantity"].quantile([0.25, 0.75])
    upper = q3 + 3 * (q3 - q1)
    df["BulkLine"] = df["Quantity"] > max(upper, 20)
    quality["bulk_lines"] = int(df["BulkLine"].sum())
    step("Outlier flagging", f"Flagged {quality['bulk_lines']:,} bulk lines (Qty > {max(upper, 20):.0f}); "
         "kept because large orders are normal for a distributor", len(df), len(df))

    # 6) Feature engineering
    df["Revenue"] = df["Quantity"] * df["NetPrice"]
    df["ListValue"] = df["Quantity"] * df["UnitPrice"]
    df["DiscountRate"] = 1 - df["NetPrice"] / df["UnitPrice"]
    df["DocType"] = df["InvoiceNo"].str[:3]
    df["CustomerType"] = "Type " + df["CustomerID"].str[0]
    df["Month"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    df["Week"] = df["InvoiceDate"].dt.to_period("W-SUN").dt.start_time
    df["MonthNo"] = df["InvoiceDate"].dt.month
    df["Weekday"] = df["InvoiceDate"].dt.dayofweek
    df["Quarter"] = "Q" + df["InvoiceDate"].dt.quarter.astype(str)
    step("Feature engineering", "Revenue, list value, discount rate, document type, customer type, "
         "month/week/weekday/quarter", len(df), len(df))

    for c in ["CustomerID", "Description", "Category", "City", "DocType", "CustomerType", "Quarter", "StockCode"]:
        df[c] = df[c].astype("category")
    df = df.drop(columns=["MonthNo"]).reset_index(drop=True)
    quality["clean_rows"] = len(df)
    return df, pd.DataFrame(log), quality


# =============================================================================
# 3. ANALYTICAL ENGINES
# =============================================================================
def fmt_money(x: float) -> str:
    if abs(x) >= 1e6:
        return f"₺{x / 1e6:,.2f}M"
    if abs(x) >= 1e3:
        return f"₺{x / 1e3:,.1f}K"
    return f"₺{x:,.0f}"


def pct_delta(cur: float, prev: float) -> str | None:
    if prev is None or prev == 0 or pd.isna(prev):
        return None
    return f"{(cur - prev) / prev * 100:+.1f}%"


def kpis(d: pd.DataFrame) -> dict:
    rev = d["Revenue"].sum()
    orders = d["InvoiceNo"].nunique()
    return {
        "revenue": rev,
        "list": d["ListValue"].sum(),
        "orders": orders,
        "customers": d["CustomerID"].nunique(),
        "units": d["Quantity"].sum(),
        "skus": d["StockCode"].nunique(),
        "aov": rev / orders if orders else 0,
        "discount": 1 - rev / d["ListValue"].sum() if d["ListValue"].sum() else 0,
    }


@st.cache_data(show_spinner=False)
def rfm_segmentation(d: pd.DataFrame, k: int) -> pd.DataFrame:
    """RFM table + K-Means clustering on log-scaled RFM + rule-based labels."""
    ref = d["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = d.groupby("CustomerID", observed=True).agg(
        Recency=("InvoiceDate", lambda s: (ref - s.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
        Units=("Quantity", "sum"),
        FirstPurchase=("InvoiceDate", "min"),
        LastPurchase=("InvoiceDate", "max"),
        City=("City", lambda s: s.mode().iat[0] if len(s.mode()) else "UNKNOWN"),
    ).reset_index()
    rfm = rfm[rfm["Monetary"] > 0]
    if len(rfm) < k:
        rfm["Cluster"] = 0
        rfm["Segment"] = "All customers"
        return rfm

    X = np.column_stack([np.log1p(rfm["Recency"]), np.log1p(rfm["Frequency"]), np.log1p(rfm["Monetary"])])
    X = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    rfm["Cluster"] = km.labels_

    # Score clusters: high F & M, low R = better. Rank and name them.
    prof = rfm.groupby("Cluster").agg(R=("Recency", "median"), F=("Frequency", "median"), M=("Monetary", "median"))
    prof["score"] = prof["M"].rank() + prof["F"].rank() - prof["R"].rank()
    order = prof.sort_values("score", ascending=False).index.tolist()
    names_by_k = {
        2: ["Core Accounts", "Low-Value / Lapsing"],
        3: ["Champions", "Developing", "At Risk / Lost"],
        4: ["Champions", "Loyal", "Needs Attention", "At Risk / Lost"],
        5: ["Champions", "Loyal", "Potential", "Needs Attention", "At Risk / Lost"],
        6: ["Champions", "Loyal", "Potential", "Needs Attention", "Hibernating", "Lost"],
    }
    names = names_by_k.get(k, [f"Segment {i + 1}" for i in range(k)])
    mapping = {c: names[i] for i, c in enumerate(order)}
    rfm["Segment"] = rfm["Cluster"].map(mapping)

    # Churn risk: silent for > 2x the customer's own average purchase gap (and > 30 days)
    span = (rfm["LastPurchase"] - rfm["FirstPurchase"]).dt.days
    rfm["AvgGapDays"] = np.where(rfm["Frequency"] > 1, span / (rfm["Frequency"] - 1).clip(lower=1), np.nan)
    rfm["ChurnRisk"] = (rfm["Recency"] > np.maximum(2 * rfm["AvgGapDays"].fillna(60), 30))
    return rfm


@st.cache_data(show_spinner=False)
def category_associations(d: pd.DataFrame, top_n: int = 40, min_support: float = 0.01) -> pd.DataFrame:
    """Pairwise association rules between categories at invoice level (support, confidence, lift)."""
    top = d.groupby("Category", observed=True)["Revenue"].sum().nlargest(top_n).index
    sub = d[d["Category"].isin(top)][["InvoiceNo", "Category"]].drop_duplicates()
    if sub.empty:
        return pd.DataFrame()
    basket = pd.crosstab(sub["InvoiceNo"], sub["Category"].astype(str)).clip(upper=1)
    n = len(basket)
    M = basket.values.astype(np.float32)
    co = M.T @ M
    sup_single = np.diag(co) / n
    cats = basket.columns.to_numpy()
    rows = []
    for i in range(len(cats)):
        for j in range(len(cats)):
            if i == j or co[i, j] == 0:
                continue
            sup = co[i, j] / n
            if sup < min_support:
                continue
            conf = co[i, j] / co[i, i]
            lift = conf / sup_single[j]
            rows.append((cats[i], cats[j], sup, conf, lift))
    rules = pd.DataFrame(rows, columns=["If customer buys", "They also buy", "Support", "Confidence", "Lift"])
    return rules.sort_values(["Lift", "Confidence"], ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def abc_xyz(d: pd.DataFrame) -> pd.DataFrame:
    """ABC (revenue Pareto 80/15/5) × XYZ (weekly demand coefficient of variation) per SKU."""
    weeks = pd.Index(sorted(d["Week"].unique()))
    prod = d.groupby("StockCode", observed=True).agg(
        Product=("Description", "first"), Category=("Category", "first"),
        Revenue=("Revenue", "sum"), Units=("Quantity", "sum"), Customers=("CustomerID", "nunique"),
    ).reset_index()
    prod = prod[prod["Revenue"] > 0].sort_values("Revenue", ascending=False)
    prod["CumShare"] = prod["Revenue"].cumsum() / prod["Revenue"].sum()
    prod["ABC"] = np.select([prod["CumShare"] <= 0.80, prod["CumShare"] <= 0.95], ["A", "B"], "C")

    wk = d.groupby(["StockCode", "Week"], observed=True)["Quantity"].sum().unstack(fill_value=0)
    wk = wk.reindex(columns=weeks, fill_value=0)
    stats = pd.DataFrame({"WeeklyMean": wk.mean(axis=1), "WeeklyStd": wk.std(axis=1, ddof=0)})
    stats["CV"] = stats["WeeklyStd"] / stats["WeeklyMean"].replace(0, np.nan)
    stats.index = stats.index.astype(str)
    prod["StockCode"] = prod["StockCode"].astype(str)
    prod = prod.merge(stats, left_on="StockCode", right_index=True, how="left")
    prod["XYZ"] = np.select([prod["CV"] <= 0.5, prod["CV"] <= 1.0], ["X", "Y"], "Z")
    prod["Class"] = prod["ABC"] + prod["XYZ"]
    return prod


def inventory_policy(prod: pd.DataFrame, service: float, lead_weeks: float, review_weeks: float,
                     order_cost: float, holding_rate: float) -> pd.DataFrame:
    """Safety stock, reorder point and EOQ per SKU under normally distributed weekly demand."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(service)
    p = prod.copy()
    unit_cost = (p["Revenue"] / p["Units"]).clip(lower=0.01)
    p["SafetyStock"] = np.ceil(z * p["WeeklyStd"] * np.sqrt(lead_weeks + review_weeks))
    p["ReorderPoint"] = np.ceil(p["WeeklyMean"] * lead_weeks + p["SafetyStock"])
    annual = p["WeeklyMean"] * 52
    p["EOQ"] = np.ceil(np.sqrt(2 * annual * order_cost / (holding_rate * unit_cost)))
    p["OrdersPerYear"] = (annual / p["EOQ"].replace(0, np.nan)).round(1)
    return p


# ---------- Forecasting ------------------------------------------------------
def _holt_fit(y: np.ndarray, alpha: float, beta: float, phi: float = 0.95):
    level, trend = y[0], (y[min(3, len(y) - 1)] - y[0]) / max(min(3, len(y) - 1), 1)
    fitted = np.empty(len(y))
    for t in range(len(y)):
        fitted[t] = level + phi * trend
        new_level = alpha * y[t] + (1 - alpha) * (level + phi * trend)
        trend = beta * (new_level - level) + (1 - beta) * phi * trend
        level = new_level
    return level, trend, fitted


def holt_forecast(y: np.ndarray, h: int, phi: float = 0.95):
    """Damped Holt linear trend; alpha/beta selected by grid search on in-sample SSE."""
    best = None
    for a in np.linspace(0.1, 0.9, 9):
        for b in np.linspace(0.05, 0.5, 6):
            _, _, f = _holt_fit(y, a, b, phi)
            sse = np.sum((y[1:] - f[1:]) ** 2)
            if best is None or sse < best[0]:
                best = (sse, a, b)
    _, a, b = best
    level, trend, fitted = _holt_fit(y, a, b, phi)
    fc = np.array([level + sum(phi ** i for i in range(1, k + 1)) * trend for k in range(1, h + 1)])
    return np.maximum(fc, 0), fitted


def _features(series: list[float], dates: list[pd.Timestamp]) -> np.ndarray:
    s = np.asarray(series)
    woy = dates[-1].isocalendar().week
    return np.array([s[-1], s[-2], s[-4], s[-4:].mean(), s[-8:].mean(),
                     np.sin(2 * np.pi * woy / 52), np.cos(2 * np.pi * woy / 52), len(s)])


def gbr_forecast(y: np.ndarray, dates: pd.DatetimeIndex, h: int):
    """Gradient boosting on lag / rolling / seasonal features with recursive multi-step forecasting."""
    X, Y = [], []
    for t in range(8, len(y)):
        X.append(_features(list(y[:t]), list(dates[:t + 1])))
        Y.append(y[t])
    model = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                                      subsample=0.9, random_state=42).fit(np.array(X), np.array(Y))
    hist, hd = list(y), list(dates)
    out = []
    for _ in range(h):
        hd.append(hd[-1] + pd.Timedelta(weeks=1))
        pred = max(float(model.predict(_features(hist, hd).reshape(1, -1))[0]), 0)
        out.append(pred)
        hist.append(pred)
    return np.array(out)


def moving_average_forecast(y: np.ndarray, h: int, w: int = 4):
    return np.repeat(y[-w:].mean(), h)


def _metrics(actual, pred):
    actual, pred = np.asarray(actual), np.asarray(pred)
    mae = np.mean(np.abs(actual - pred))
    mape = np.mean(np.abs(actual - pred) / np.where(actual == 0, np.nan, actual)) * 100
    return mae, mape


@st.cache_data(show_spinner="Training forecasting models…")
def run_forecast(series: pd.Series, h: int, test_weeks: int = 8) -> dict:
    """Backtest 3 models on the last `test_weeks`, pick the best by MAPE, refit on all data."""
    y = series.values.astype(float)
    dates = pd.DatetimeIndex(series.index)
    if len(y) < test_weeks + 16:
        test_weeks = max(4, len(y) // 4)
    train, test = y[:-test_weeks], y[-test_weeks:]
    preds = {
        "Damped Holt (exp. smoothing)": holt_forecast(train, test_weeks)[0],
        "Gradient Boosting (ML)": gbr_forecast(train, dates[:-test_weeks], test_weeks),
        "4-week Moving Average": moving_average_forecast(train, test_weeks),
    }
    scores = []
    for name, p in preds.items():
        mae, mape = _metrics(test, p)
        scores.append({"Model": name, "MAE": mae, "MAPE (%)": mape})
    scores = pd.DataFrame(scores).sort_values("MAPE (%)").reset_index(drop=True)
    best = scores.iloc[0]["Model"]

    if best.startswith("Damped"):
        fc, _ = holt_forecast(y, h)
    elif best.startswith("Gradient"):
        fc = gbr_forecast(y, dates, h)
    else:
        fc = moving_average_forecast(y, h)
    resid_sd = np.std(test - preds[best])
    future = pd.date_range(dates[-1] + pd.Timedelta(weeks=1), periods=h, freq="7D")
    band = 1.28 * resid_sd * np.sqrt(np.arange(1, h + 1))  # ~80% interval, widening with horizon
    return {
        "scores": scores, "best": best, "test_dates": dates[-test_weeks:], "test_preds": preds,
        "future": future, "forecast": fc, "lower": np.maximum(fc - band, 0), "upper": fc + band,
    }


def weekly_series(d: pd.DataFrame, metric: str) -> pd.Series:
    s = d.groupby("Week")[metric].sum().sort_index()
    if s.empty:
        return s
    full = pd.date_range(s.index.min(), s.index.max(), freq="7D")
    s = s.reindex(full, fill_value=0)
    # drop partial first/last weeks (fewer than 3 trading days in the data)
    days = d.groupby("Week")["InvoiceDate"].nunique().reindex(full, fill_value=0)
    keep = ~(((s.index == s.index[0]) | (s.index == s.index[-1])) & (days < 3))
    return s[keep]


# =============================================================================
# 4. INSIGHT GENERATOR
# =============================================================================
def generate_insights(d: pd.DataFrame, rfm: pd.DataFrame, prod: pd.DataFrame, rules: pd.DataFrame,
                      fc: dict | None) -> dict[str, list[tuple[str, str]]]:
    """Returns {department: [(severity, text), ...]} computed from the filtered data."""
    out = {"Sales": [], "Marketing": [], "Supply Chain": []}
    total = d["Revenue"].sum()
    if total <= 0:
        return out

    m = d.groupby("Month")["Revenue"].sum()
    if len(m) >= 2:
        best, worst = m.idxmax(), m.idxmin()
        out["Sales"].append(("good", f"Peak month is **{best:%B %Y}** ({fmt_money(m.max())}); weakest is "
                                     f"**{worst:%B %Y}** ({fmt_money(m.min())}), {m.min() / m.max() * 100:.0f}% of the peak."))
        q = d.groupby(d["InvoiceDate"].dt.quarter)["Revenue"].sum()
        if {2, 4}.issubset(q.index):
            out["Sales"].append(("info", f"Q4 revenue is **{(q[4] / q[2] - 1) * 100:+.0f}%** vs Q2 – the spring "
                                         "2020 lockdown dip was fully recovered by the back-to-school season (Sep–Oct)."))
    cat = d.groupby("Category", observed=True)["Revenue"].sum().sort_values(ascending=False)
    top5 = cat.head(5)
    out["Sales"].append(("info", f"Top-5 categories generate **{top5.sum() / total * 100:.0f}%** of revenue, led by "
                                 f"**{top5.index[0]}** ({top5.iloc[0] / total * 100:.1f}%)."))
    city = d.groupby("City", observed=True)["Revenue"].sum().sort_values(ascending=False)
    if len(city):
        share = city.iloc[0] / total * 100
        sev = "warn" if share > 40 else "info"
        out["Sales"].append((sev, f"**{city.index[0]}** accounts for **{share:.0f}%** of revenue – "
                                  f"{'high geographic concentration risk; ' if share > 40 else ''}"
                                  f"next are {', '.join(city.index[1:4])}."))
    wd = d.groupby("Weekday")["Revenue"].sum()
    if len(wd):
        out["Sales"].append(("info", f"**{WEEKDAYS[int(wd.idxmax())]}** is the busiest order day "
                                     f"({wd.max() / wd.sum() * 100:.0f}% of weekly revenue) – plan sales-rep capacity accordingly."))

    if not rfm.empty:
        cust_sorted = rfm["Monetary"].sort_values(ascending=False)
        n20 = max(1, int(len(cust_sorted) * 0.2))
        out["Marketing"].append(("info", f"Top 20% of customers ({n20}) generate **{cust_sorted.head(n20).sum() / cust_sorted.sum() * 100:.0f}%** "
                                         "of revenue – a key-account programme is justified."))
        churn = rfm[rfm["ChurnRisk"]]
        if len(churn):
            out["Marketing"].append(("warn", f"**{len(churn)} customers** show churn signals (silent > 2× their usual purchase gap). "
                                             f"They represented **{fmt_money(churn['Monetary'].sum())}** of revenue – target with win-back offers."))
        seg = rfm.groupby("Segment")["Monetary"].agg(["sum", "count"]).sort_values("sum", ascending=False)
        out["Marketing"].append(("good", f"Segment **{seg.index[0]}** ({int(seg.iloc[0]['count'])} customers) holds "
                                         f"**{seg.iloc[0]['sum'] / seg['sum'].sum() * 100:.0f}%** of revenue."))
    if rules is not None and not rules.empty:
        strong = rules[rules["Lift"] >= 2]
        r = (strong if not strong.empty else rules).sort_values("Confidence", ascending=False).iloc[0]
        out["Marketing"].append(("good", f"Strongest cross-sell: invoices with **{r['If customer buys']}** contain **{r['They also buy']}** "
                                         f"{r['Confidence'] * 100:.0f}% of the time (lift {r['Lift']:.1f}×) – bundle them in campaigns."))
    disc = d.assign(Band=pd.cut(d["DiscountRate"], [-0.01, 0.35, 0.40, 0.45, 1.0],
                                labels=["≤35%", "35–40%", "40–45%", ">45%"]))
    db = disc.groupby("Band", observed=True)["Quantity"].mean()
    if len(db) >= 2:
        out["Marketing"].append(("info", f"Average units per line range from **{db.min():.1f}** to **{db.max():.1f}** across discount bands "
                                         f"(highest in **{db.idxmax()}**) – discount depth should be tied to volume tiers."))

    if not prod.empty:
        a = prod[prod["ABC"] == "A"]
        out["Supply Chain"].append(("info", f"Only **{len(a):,} SKUs ({len(a) / len(prod) * 100:.1f}%)** make up 80% of revenue (class A) – "
                                            "prioritise them for safety stock and supplier SLAs."))
        az = prod[prod["Class"] == "AZ"]
        if len(az):
            out["Supply Chain"].append(("warn", f"**{len(az)} class-AZ SKUs** are high-value but highly erratic – use higher safety stock "
                                                "or make-to-order agreements with publishers."))
        cz = prod[prod["Class"] == "CZ"]
        out["Supply Chain"].append(("info", f"**{len(cz):,} CZ SKUs** (low value, erratic) are candidates for range rationalisation "
                                            "or drop-shipping to cut holding costs."))
    if fc is not None:
        nxt = fc["forecast"][:4].sum()
        out["Supply Chain"].append(("good", f"Best model (**{fc['best']}**, backtest MAPE "
                                            f"{fc['scores'].iloc[0]['MAPE (%)']:.1f}%) projects **{fmt_money(nxt)}** over the next 4 weeks."))
    return out


# =============================================================================
# 5. USER INTERFACE
# =============================================================================
def to_excel(frames: dict[str, pd.DataFrame]) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, f in frames.items():
            f.to_excel(w, sheet_name=name[:31], index=False)
    return buf.getvalue()


def insight_box(sev: str, text: str):
    import re
    html = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    st.markdown(f'<div class="insight {sev}">{html}</div>', unsafe_allow_html=True)


df, prep_log, quality = load_and_clean()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## 📚 Control Panel")
    st.caption("Filters apply to every tab.")
    dmin, dmax = df["InvoiceDate"].min().date(), df["InvoiceDate"].max().date()
    dr = st.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
    start, end = (dr if isinstance(dr, (list, tuple)) and len(dr) == 2 else (dmin, dmax))

    sel_doc = st.multiselect("Document type", sorted(df["DocType"].cat.categories), placeholder="All")
    sel_ctype = st.multiselect("Customer type", sorted(df["CustomerType"].cat.categories), placeholder="All")
    city_order = df.groupby("City", observed=True)["Revenue"].sum().sort_values(ascending=False).index.tolist()
    sel_city = st.multiselect("City", city_order, placeholder="All")
    cat_order = df.groupby("Category", observed=True)["Revenue"].sum().sort_values(ascending=False).index.tolist()
    sel_cat = st.multiselect("Category", cat_order, placeholder="All")
    st.divider()
    show_n = st.slider("Top-N items in rankings", 5, 30, 10)
    st.caption(f"Dataset: {quality['raw_rows']:,} raw lines → {quality['clean_rows']:,} clean lines")

mask = (df["InvoiceDate"].dt.date >= start) & (df["InvoiceDate"].dt.date <= end)
if sel_doc:
    mask &= df["DocType"].isin(sel_doc)
if sel_ctype:
    mask &= df["CustomerType"].isin(sel_ctype)
if sel_city:
    mask &= df["City"].isin(sel_city)
if sel_cat:
    mask &= df["Category"].isin(sel_cat)
fdf = df[mask]

st.markdown(
    f"""<div class="hero"><h1>📚 B2B Book Distributor – Intelligent Decision Support Dashboard</h1>
    <p>Sales · Marketing · Supply Chain &nbsp;|&nbsp; {start:%d %b %Y} – {end:%d %b %Y} &nbsp;|&nbsp;
    {len(fdf):,} transaction lines selected</p></div>""",
    unsafe_allow_html=True,
)

if fdf.empty:
    st.warning("No data for the selected filters. Please widen the selection in the sidebar.")
    st.stop()

tabs = st.tabs(["🏠 Executive Overview", "💰 Sales", "🎯 Marketing & Customers", "🚚 Supply Chain",
                "🔮 Forecast", "💡 Insights & Actions", "🧹 Data Quality"])

# ---------------- TAB 0: Executive overview ----------------
with tabs[0]:
    k = kpis(fdf)
    last_m = fdf["Month"].max()
    cur_m = fdf[fdf["Month"] == last_m]
    prev_m = fdf[fdf["Month"] == last_m - pd.DateOffset(months=1)]
    kc, kp = kpis(cur_m), (kpis(prev_m) if len(prev_m) else None)

    st.markdown("#### Key performance indicators")
    st.markdown(f'<p class="section-note">Totals for the selected period; deltas compare the last month '
                f'({last_m:%b %Y}) with the month before.</p>', unsafe_allow_html=True)
    c = st.columns(4)
    c[0].metric("Net revenue", fmt_money(k["revenue"]), pct_delta(kc["revenue"], kp["revenue"]) if kp else None)
    c[1].metric("Orders (invoices)", f"{k['orders']:,}", pct_delta(kc["orders"], kp["orders"]) if kp else None)
    c[2].metric("Active customers", f"{k['customers']:,}", pct_delta(kc["customers"], kp["customers"]) if kp else None)
    c[3].metric("Avg. order value", fmt_money(k["aov"]), pct_delta(kc["aov"], kp["aov"]) if kp else None)
    c = st.columns(4)
    c[0].metric("Units sold", f"{k['units']:,.0f}", pct_delta(kc["units"], kp["units"]) if kp else None)
    c[1].metric("Avg. discount off list", f"{k['discount'] * 100:.1f}%",
                (f"{(kc['discount'] - kp['discount']) * 100:+.1f} pp" if kp else None), delta_color="inverse")
    c[2].metric("List-price value", fmt_money(k["list"]))
    c[3].metric("Active SKUs", f"{k['skus']:,}")

    st.markdown("#### Monthly performance")
    mon = fdf.groupby("Month").agg(Revenue=("Revenue", "sum"), Orders=("InvoiceNo", "nunique")).reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=mon["Month"], y=mon["Revenue"], name="Net revenue", marker_color=PRIMARY)
    fig.add_scatter(x=mon["Month"], y=mon["Orders"], name="Orders", mode="lines+markers",
                    line=dict(color=ACCENT, width=3), secondary_y=True)
    fig.update_layout(height=380, legend=dict(orientation="h", y=1.1), margin=dict(t=30, b=10),
                      hovermode="x unified")
    fig.update_yaxes(title_text="Revenue (₺)", secondary_y=False)
    fig.update_yaxes(title_text="Orders", secondary_y=True, showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        dt = fdf.groupby("DocType", observed=True)["Revenue"].sum().reset_index()
        fig = px.pie(dt, names="DocType", values="Revenue", hole=0.55, title="Revenue by document type")
        fig.update_layout(height=330, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        ct = fdf.groupby("CustomerType", observed=True)["Revenue"].sum().reset_index()
        fig = px.pie(ct, names="CustomerType", values="Revenue", hole=0.55, title="Revenue by customer type")
        fig.update_layout(height=330, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        cy = fdf.groupby("City", observed=True)["Revenue"].sum().nlargest(8).sort_values().reset_index()
        fig = px.bar(cy, x="Revenue", y="City", orientation="h", title="Top cities by revenue",
                     color_discrete_sequence=[TEAL], text_auto=".2s")
        fig.update_layout(height=330, margin=dict(t=50, b=10), yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 1: Sales ----------------
with tabs[1]:
    st.markdown("#### Category performance")
    cat = (fdf.groupby("Category", observed=True)
           .agg(Revenue=("Revenue", "sum"), Units=("Quantity", "sum"), Orders=("InvoiceNo", "nunique"),
                Discount=("DiscountRate", "mean")).reset_index().sort_values("Revenue", ascending=False))
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.treemap(cat.head(30), path=[px.Constant("All categories"), "Category"], values="Revenue",
                         color="Discount", color_continuous_scale="Tealrose",
                         title="Top-30 categories (size = revenue, colour = avg. discount)")
        fig.update_layout(height=430, margin=dict(t=50, l=5, r=5, b=5))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        top = cat.head(show_n).copy()
        top["Share"] = top["Revenue"] / cat["Revenue"].sum()
        st.dataframe(top[["Category", "Revenue", "Units", "Share"]], hide_index=True, use_container_width=True,
                     height=430,
                     column_config={"Revenue": st.column_config.NumberColumn(format="₺%.0f"),
                                    "Units": st.column_config.NumberColumn(format="%.0f"),
                                    "Share": st.column_config.ProgressColumn(format="%.3f", min_value=0,
                                                                             max_value=float(top["Share"].max()))})

    st.markdown("#### Seasonality heatmap – top categories × month")
    topc = cat.head(12)["Category"]
    hm = (fdf[fdf["Category"].isin(topc)].groupby(["Category", "Month"], observed=True)["Revenue"].sum()
          .unstack(fill_value=0))
    hm = hm.loc[[c for c in topc if c in hm.index]]
    hm.columns = [f"{c:%b}" for c in hm.columns]
    fig = px.imshow(hm, aspect="auto", color_continuous_scale="Blues", labels=dict(color="Revenue"))
    fig.update_layout(height=420, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Best-selling titles")
        tp = (fdf.groupby("Description", observed=True)
              .agg(Revenue=("Revenue", "sum"), Units=("Quantity", "sum"), Customers=("CustomerID", "nunique"))
              .nlargest(show_n, "Revenue").reset_index())
        fig = px.bar(tp.sort_values("Revenue"), x="Revenue", y="Description", orientation="h",
                     hover_data=["Units", "Customers"], color_discrete_sequence=[PRIMARY])
        fig.update_layout(height=420, yaxis_title=None, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### Order-day pattern")
        wd = fdf.groupby(["Weekday", "Month"])["Revenue"].sum().unstack(fill_value=0)
        wd.index = [WEEKDAYS[i] for i in wd.index]
        wd.columns = [f"{c:%b}" for c in wd.columns]
        fig = px.imshow(wd, aspect="auto", color_continuous_scale="Oranges", labels=dict(color="Revenue"))
        fig.update_layout(height=420, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Dynamic analysis")
    c1, c2, c3 = st.columns(3)
    dims = {"Month": "Month", "Quarter": "Quarter", "City": "City", "Category": "Category",
            "Document type": "DocType", "Customer type": "CustomerType"}
    metrics = {"Net revenue": ("Revenue", "sum"), "Units": ("Quantity", "sum"), "Orders": ("InvoiceNo", "nunique"),
               "Customers": ("CustomerID", "nunique"), "Avg. discount": ("DiscountRate", "mean")}
    g = c1.selectbox("Group by", list(dims), index=2)
    mname = c2.selectbox("Metric", list(metrics), index=0)
    ctype = c3.radio("Chart", ["Bar", "Line"], horizontal=True)
    col, agg = metrics[mname]
    dyn = fdf.groupby(dims[g], observed=True)[col].agg(agg).reset_index()
    if g not in ("Month", "Quarter"):
        dyn = dyn.nlargest(show_n, col)
    else:
        dyn = dyn.sort_values(dims[g])
    dyn[dims[g]] = dyn[dims[g]].astype(str) if g != "Month" else dyn[dims[g]]
    fig = (px.bar(dyn, x=dims[g], y=col, color_discrete_sequence=[TEAL], text_auto=".3s") if ctype == "Bar"
           else px.line(dyn, x=dims[g], y=col, markers=True))
    fig.update_layout(height=380, yaxis_title=mname, xaxis_title=g, margin=dict(t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 2: Marketing ----------------
with tabs[2]:
    st.markdown("#### Customer segmentation (RFM + K-Means)")
    st.markdown('<p class="section-note">Recency, Frequency and Monetary value are log-scaled and standardised, '
                'then clustered with K-Means. Clusters are ranked and named automatically.</p>', unsafe_allow_html=True)
    k_clusters = st.slider("Number of segments (k)", 2, 6, 4)
    rfm = rfm_segmentation(fdf[["CustomerID", "InvoiceDate", "InvoiceNo", "Revenue", "Quantity", "City"]], k_clusters)

    seg = (rfm.groupby("Segment").agg(Customers=("CustomerID", "count"), Revenue=("Monetary", "sum"),
                                      AvgRecency=("Recency", "mean"), AvgOrders=("Frequency", "mean"),
                                      AvgRevenue=("Monetary", "mean"), ChurnRisk=("ChurnRisk", "mean"))
           .sort_values("Revenue", ascending=False).reset_index())
    seg["RevenueShare"] = seg["Revenue"] / seg["Revenue"].sum()
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.scatter(rfm, x="Recency", y="Frequency", size="Monetary", color="Segment", log_y=True,
                         hover_data={"CustomerID": True, "Monetary": ":,.0f", "City": True}, size_max=40,
                         title="Customers by recency vs. frequency (bubble = revenue)")
        fig.update_layout(height=430, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(seg, x="Segment", y="RevenueShare", color="Segment", text_auto=".0%",
                     title="Revenue share by segment")
        fig.update_layout(height=430, showlegend=False, yaxis_tickformat=".0%", margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(seg, hide_index=True, use_container_width=True,
                 column_config={"Revenue": st.column_config.NumberColumn(format="₺%.0f"),
                                "AvgRevenue": st.column_config.NumberColumn("Avg revenue / customer", format="₺%.0f"),
                                "AvgRecency": st.column_config.NumberColumn("Avg recency (days)", format="%.0f"),
                                "AvgOrders": st.column_config.NumberColumn("Avg orders", format="%.1f"),
                                "ChurnRisk": st.column_config.NumberColumn("Churn-risk share", format="%.2f"),
                                "RevenueShare": st.column_config.ProgressColumn("Revenue share", format="%.2f",
                                                                                min_value=0, max_value=1)})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### ⚠️ Win-back list – high-value customers at churn risk")
        wb = rfm[rfm["ChurnRisk"]].nlargest(show_n, "Monetary")[
            ["CustomerID", "City", "Segment", "Recency", "Frequency", "Monetary", "AvgGapDays"]]
        st.dataframe(wb, hide_index=True, use_container_width=True,
                     column_config={"Monetary": st.column_config.NumberColumn("Revenue", format="₺%.0f"),
                                    "Recency": st.column_config.NumberColumn("Days silent"),
                                    "AvgGapDays": st.column_config.NumberColumn("Usual gap (days)", format="%.0f")})
    with c2:
        st.markdown("#### Customer concentration (Pareto)")
        cs = rfm["Monetary"].sort_values(ascending=False).reset_index(drop=True)
        par = pd.DataFrame({"CustomerShare": (np.arange(1, len(cs) + 1)) / len(cs),
                            "RevenueShare": cs.cumsum() / cs.sum()})
        fig = px.area(par, x="CustomerShare", y="RevenueShare", color_discrete_sequence=[PRIMARY])
        fig.add_hline(y=0.8, line_dash="dash", line_color=ACCENT, annotation_text="80% of revenue")
        fig.update_layout(height=340, xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                          xaxis_title="Share of customers", yaxis_title="Cumulative revenue share", margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Cross-selling opportunities (category association rules)")
    c1, c2 = st.columns([1, 3])
    min_sup = c1.slider("Min. support", 0.005, 0.05, 0.01, 0.005, format="%.3f")
    min_lift = c1.slider("Min. lift", 1.0, 5.0, 1.5, 0.1)
    rules = category_associations(fdf[["InvoiceNo", "Category", "Revenue"]], 40, min_sup)
    shown = rules[rules["Lift"] >= min_lift].head(show_n) if not rules.empty else rules
    with c2:
        if shown.empty:
            st.info("No rules meet the thresholds – lower support or lift.")
        else:
            st.dataframe(shown, hide_index=True, use_container_width=True,
                         column_config={"Support": st.column_config.NumberColumn(format="%.3f"),
                                        "Confidence": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
                                        "Lift": st.column_config.NumberColumn(format="%.2f×")})

    st.markdown("#### Discount effectiveness")
    dd = fdf.assign(Band=pd.cut(fdf["DiscountRate"], [-0.01, 0.35, 0.40, 0.45, 1.0],
                                labels=["≤35%", "35–40%", "40–45%", ">45%"]))
    dband = dd.groupby(["Band", "CustomerType"], observed=True).agg(
        UnitsPerLine=("Quantity", "mean"), Revenue=("Revenue", "sum")).reset_index()
    fig = px.bar(dband, x="Band", y="UnitsPerLine", color="CustomerType", barmode="group",
                 labels={"Band": "Discount off list price", "UnitsPerLine": "Avg units per order line"})
    fig.update_layout(height=360, margin=dict(t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 3: Supply chain ----------------
with tabs[3]:
    st.markdown("#### ABC–XYZ inventory classification")
    st.markdown('<p class="section-note">ABC = cumulative revenue share (A ≤ 80%, B ≤ 95%, C rest). '
                'XYZ = coefficient of variation of weekly demand (X ≤ 0.5, Y ≤ 1.0, Z > 1.0).</p>',
                unsafe_allow_html=True)
    prod = abc_xyz(fdf[["StockCode", "Description", "Category", "Revenue", "Quantity", "CustomerID", "Week"]])
    c1, c2 = st.columns([2, 3])
    with c1:
        mat = prod.pivot_table(index="ABC", columns="XYZ", values="StockCode", aggfunc="count", fill_value=0, observed=True)
        mat = mat.reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0)
        fig = px.imshow(mat, text_auto=True, color_continuous_scale="Teal", labels=dict(color="# SKUs"),
                        title="Number of SKUs per class")
        fig.update_layout(height=360, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        rv = prod.pivot_table(index="ABC", columns="XYZ", values="Revenue", aggfunc="sum", fill_value=0, observed=True)
        rv = rv.reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0) / prod["Revenue"].sum()
        fig = px.imshow(rv, text_auto=".1%", color_continuous_scale="Blues", labels=dict(color="Revenue share"),
                        title="Revenue share per class")
        fig.update_layout(height=360, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Inventory policy simulator")
    st.markdown('<p class="section-note">Safety stock = z·σ·√(L+R); Reorder point = μ·L + SS; '
                'EOQ = √(2DS / hC). Change the parameters to see the stock impact.</p>', unsafe_allow_html=True)
    c = st.columns(5)
    service = c[0].slider("Service level", 0.80, 0.99, 0.95, 0.01)
    lead = c[1].slider("Lead time (weeks)", 0.5, 8.0, 2.0, 0.5)
    review = c[2].slider("Review period (weeks)", 0.0, 4.0, 1.0, 0.5)
    ocost = c[3].number_input("Order cost (₺/order)", 10.0, 5000.0, 250.0, 10.0)
    hrate = c[4].slider("Holding cost (%/yr)", 0.05, 0.50, 0.20, 0.01)
    cls = st.multiselect("Classes to show", sorted(prod["Class"].unique()), default=[x for x in ["AX", "AY", "AZ"]
                                                                                     if x in set(prod["Class"])])
    pol = inventory_policy(prod[prod["Class"].isin(cls)] if cls else prod, service, lead, review, ocost, hrate)
    m1, m2, m3 = st.columns(3)
    unit_val = (pol["Revenue"] / pol["Units"]).fillna(0)
    m1.metric("SKUs in policy", f"{len(pol):,}")
    m2.metric("Total safety stock (units)", f"{pol['SafetyStock'].sum():,.0f}")
    m3.metric("Safety-stock value", fmt_money((pol["SafetyStock"] * unit_val).sum()))
    st.dataframe(pol.nlargest(max(show_n, 15), "Revenue")[
                     ["Product", "Category", "Class", "WeeklyMean", "WeeklyStd", "SafetyStock", "ReorderPoint", "EOQ",
                      "OrdersPerYear"]],
                 hide_index=True, use_container_width=True,
                 column_config={"WeeklyMean": st.column_config.NumberColumn("Avg weekly demand", format="%.1f"),
                                "WeeklyStd": st.column_config.NumberColumn("Demand σ", format="%.1f"),
                                "SafetyStock": st.column_config.NumberColumn("Safety stock", format="%.0f"),
                                "ReorderPoint": st.column_config.NumberColumn("Reorder point", format="%.0f"),
                                "EOQ": st.column_config.NumberColumn("EOQ", format="%.0f")})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Weekly unit demand")
        wk = weekly_series(fdf, "Quantity").reset_index()
        wk.columns = ["Week", "Units"]
        wk["4-wk avg"] = wk["Units"].rolling(4).mean()
        fig = px.line(wk, x="Week", y=["Units", "4-wk avg"], color_discrete_sequence=[TEAL, ACCENT])
        fig.update_layout(height=360, legend_title=None, margin=dict(t=10, b=10), yaxis_title="Units")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("#### Regional demand (warehouse planning)")
        reg = fdf.groupby("City", observed=True).agg(Units=("Quantity", "sum"), Customers=("CustomerID", "nunique"),
                                                     Orders=("InvoiceNo", "nunique")).reset_index()
        reg = reg.nlargest(15, "Units")
        reg["UnitsPerOrder"] = reg["Units"] / reg["Orders"]
        fig = px.scatter(reg, x="Orders", y="UnitsPerOrder", size="Units", color="Customers", text="City",
                         log_x=True, color_continuous_scale="Teal",
                         labels={"UnitsPerOrder": "Units per order (drop size)"})
        fig.update_traces(textposition="top center")
        fig.update_layout(height=360, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

# ---------------- TAB 4: Forecast ----------------
with tabs[4]:
    st.markdown("#### Demand & revenue forecasting")
    st.markdown('<p class="section-note">Three models are back-tested on the last 8 weeks; the one with the lowest '
                'MAPE is refitted on all data and used for the forecast (shaded area ≈ 80% interval).</p>',
                unsafe_allow_html=True)
    c = st.columns(4)
    target = c[0].selectbox("Forecast target", ["Net revenue", "Units"])
    level = c[1].selectbox("Level", ["Total (filtered)", "Category", "City"])
    choice = None
    if level == "Category":
        choice = c[2].selectbox("Category", fdf.groupby("Category", observed=True)["Revenue"].sum()
                                .sort_values(ascending=False).index.tolist()[:40])
    elif level == "City":
        choice = c[2].selectbox("City", fdf.groupby("City", observed=True)["Revenue"].sum()
                                .sort_values(ascending=False).index.tolist()[:30])
    horizon = c[3].slider("Horizon (weeks)", 4, 16, 8)
    col = "Revenue" if target == "Net revenue" else "Quantity"
    base = fdf if level.startswith("Total") else fdf[fdf[level] == choice]
    ser = weekly_series(base, col)

    fc_result = None
    if len(ser) < 20:
        st.info("Not enough weekly history for a reliable forecast – widen the date range.")
    else:
        fc_result = run_forecast(ser, horizon)
        fig = go.Figure()
        fig.add_scatter(x=ser.index, y=ser.values, name="Actual", mode="lines+markers",
                        line=dict(color=PRIMARY, width=2), marker=dict(size=4))
        for name, p in fc_result["test_preds"].items():
            fig.add_scatter(x=fc_result["test_dates"], y=p, name=f"Backtest – {name}", mode="lines",
                            line=dict(dash="dot", width=1.5), visible="legendonly" if name != fc_result["best"] else True)
        fig.add_scatter(x=np.concatenate([fc_result["future"], fc_result["future"][::-1]]),
                        y=np.concatenate([fc_result["upper"], fc_result["lower"][::-1]]),
                        fill="toself", fillcolor="rgba(224,122,95,0.18)", line=dict(width=0), mode="lines",
                        name="80% interval", hoverinfo="skip")
        fig.add_scatter(x=fc_result["future"], y=fc_result["forecast"], name=f"Forecast ({fc_result['best']})",
                        mode="lines+markers", line=dict(color=ACCENT, width=3))
        fig.update_layout(height=450, hovermode="x unified", legend=dict(orientation="h", y=-0.15),
                          yaxis_title=target, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns([2, 3])
        with c1:
            st.markdown("**Model comparison (back-test)**")
            st.dataframe(fc_result["scores"], hide_index=True, use_container_width=True,
                         column_config={"MAE": st.column_config.NumberColumn(format="%.0f"),
                                        "MAPE (%)": st.column_config.NumberColumn(format="%.1f")})
            nxt = fc_result["forecast"][:4].sum()
            last4 = ser.values[-4:].sum()
            st.metric(f"Next 4 weeks ({target.lower()})",
                      fmt_money(nxt) if col == "Revenue" else f"{nxt:,.0f}", pct_delta(nxt, last4))
        with c2:
            st.markdown("**Forecast table**")
            ft = pd.DataFrame({"Week starting": fc_result["future"].date, "Forecast": fc_result["forecast"],
                               "Lower (80%)": fc_result["lower"], "Upper (80%)": fc_result["upper"]})
            st.dataframe(ft, hide_index=True, use_container_width=True,
                         column_config={k_: st.column_config.NumberColumn(format="%.0f")
                                        for k_ in ["Forecast", "Lower (80%)", "Upper (80%)"]})

# ---------------- TAB 5: Insights ----------------
with tabs[5]:
    st.markdown("#### Automatically generated insights")
    st.markdown('<p class="section-note">Insights are recalculated from the filtered data each time you change '
                'a filter.</p>', unsafe_allow_html=True)
    total_fc = None
    tser = weekly_series(fdf, "Revenue")
    if len(tser) >= 20:
        total_fc = run_forecast(tser, 8)
    ins = generate_insights(fdf, rfm, prod, rules, total_fc)
    cols = st.columns(3)
    icons = {"Sales": "💰", "Marketing": "🎯", "Supply Chain": "🚚"}
    for col_, (dept, items) in zip(cols, ins.items()):
        with col_:
            st.markdown(f"##### {icons[dept]} {dept}")
            for sev, text in items:
                insight_box(sev, text)

    st.markdown("#### Recommended actions")
    st.markdown(
        """
| Department | Recommendation | KPI to track |
|---|---|---|
| **Sales** | Build stock & sales-rep capacity before the Sep–Oct back-to-school peak; reduce reliance on Istanbul by targeting the next-largest cities. | Monthly revenue, city revenue share |
| **Marketing** | Run a key-account programme for the top segment, automated win-back offers for churn-risk customers, and bundle high-lift category pairs. | Retention rate, churn-risk count, basket size |
| **Supply Chain** | Apply service-level-based safety stock to A-class SKUs, use make-to-order/drop-ship for CZ SKUs, and plan purchases with the weekly forecast. | Fill rate, inventory turns, forecast MAPE |
"""
    )
    st.download_button(
        "📥 Download analysis pack (Excel)",
        data=to_excel({
            "Segments": seg, "Customers_RFM": rfm.drop(columns=["FirstPurchase", "LastPurchase"]),
            "Cross_sell_rules": rules.head(200) if not rules.empty else pd.DataFrame(),
            "ABC_XYZ": prod.head(5000),
        }),
        file_name="b2b_decision_support_pack.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# ---------------- TAB 6: Data quality ----------------
with tabs[6]:
    st.markdown("#### Preprocessing pipeline")
    c = st.columns(4)
    c[0].metric("Raw lines", f"{quality['raw_rows']:,}")
    c[1].metric("Clean lines", f"{quality['clean_rows']:,}",
                f"-{quality['raw_rows'] - quality['clean_rows']:,} removed", delta_color="off")
    c[2].metric("Identical duplicate lines", f"{quality['exact_duplicates']:,}",
                "kept (repeat scans in same invoice)", delta_color="off")
    c[3].metric("Bulk-order lines flagged", f"{quality['bulk_lines']:,}")
    st.dataframe(prep_log, hide_index=True, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        miss = pd.Series(quality["missing"]).rename("Missing").reset_index().rename(columns={"index": "Column"})
        miss = miss[miss["Missing"] > 0]
        fig = px.bar(miss, x="Column", y="Missing", title="Missing values in raw data", text_auto=True,
                     color_discrete_sequence=[ACCENT])
        fig.update_layout(height=330, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(fdf.sample(min(len(fdf), 100_000), random_state=1), x="DiscountRate", nbins=60,
                           title="Distribution of discount rate (sample)", color_discrete_sequence=[TEAL])
        fig.update_layout(height=330, margin=dict(t=50, b=10), xaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Clean data preview")
    st.dataframe(fdf.head(200), use_container_width=True, hide_index=True)
    st.download_button("📥 Download filtered clean data (CSV)",
                       fdf.to_csv(index=False).encode("utf-8-sig"), "b2b_clean_filtered.csv", "text/csv")

st.caption("ISL 467E – Mng. Eng. Applications with Python · Homework I · Built with Streamlit, pandas, scikit-learn & Plotly")
