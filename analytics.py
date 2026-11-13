"""Analytical engines: KPIs, RFM + K-Means segmentation, market-basket rules,
ABC-XYZ classification, inventory policy, forecasting and rule-based insights."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DISCOUNT_BINS = [-0.01, 0.35, 0.40, 0.45, 1.0]
DISCOUNT_LABELS = ["≤35%", "35–40%", "40–45%", ">45%"]


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


# ---------- Insight generator ------------------------------------------------
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
    disc = d.assign(Band=pd.cut(d["DiscountRate"], DISCOUNT_BINS, labels=DISCOUNT_LABELS))
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
