"""Data Assistant – a natural-language query engine for the transaction data.

How it works
------------
1. ``parse_question`` turns a free-text question (Turkish or English) into a structured
   query *spec* (intent, metric, dimension, filters, top-N, sort order, chart type).
2. Optionally, if an Anthropic API key is configured in ``st.secrets``, an LLM produces the
   same spec as JSON (``llm_parse``). The LLM never executes code – it only fills the spec,
   which keeps answers grounded in the data and safe.
3. ``execute`` runs the spec with pandas and returns a narrated answer, a result table,
   a Plotly chart and a downloadable file.
"""

from __future__ import annotations

import json
import re
import unicodedata

import numpy as np
import pandas as pd
import plotly.express as px

import analytics as an

# ----------------------------------------------------------------------------- vocabulary
METRICS = {
    # key: (English label, Turkish label, keywords)  – order matters (most specific first)
    "aov": ("Avg. order value", "Ortalama sipariş tutarı",
            ["aov", "average order", "avg order", "order value", "ortalama siparis", "sepet tutari",
             "ortalama fatura", "basket value"]),
    "customers": ("Active customers", "Aktif müşteri sayısı",
                  ["musteri sayisi", "kac musteri", "number of customers", "customer count", "how many customers",
                   "aktif musteri", "active customers", "musteri adedi"]),
    "skus": ("Active SKUs", "Aktif ürün sayısı",
             ["sku", "urun sayisi", "kac urun", "kac farkli", "number of products", "how many products", "cesit"]),
    "discount": ("Avg. discount", "Ortalama indirim",
                 ["discount", "indirim", "iskonto"]),
    "units": ("Units sold", "Satılan adet",
              ["units", "unit", "adet", "quantity", "miktar", "kac kitap", "kac tane", "volume", "hacim",
               "how many books", "copies"]),
    "orders": ("Orders", "Sipariş sayısı",
               ["orders", "order", "siparis", "fatura", "invoice"]),
    "revenue": ("Net revenue", "Net ciro",
                ["revenue", "ciro", "gelir", "hasilat", "satis tutari", "sales", "satis", "kazanc", "tutar",
                 "turnover", "income", "para"]),
}

DIMS = {
    # key: (column, English label, Turkish label, keyword regexes)
    "doctype": ("DocType", "Document type", "Belge tipi", [r"belge", r"document", r"doc ?type", r"fatura tip", r"kanal", r"channel"]),
    "ctype": ("CustomerType", "Customer type", "Müşteri tipi", [r"musteri tip", r"customer type"]),
    "weekday": ("Weekday", "Weekday", "Haftanın günü", [r"haftanin gun", r"weekday", r"day of (the )?week", r"\bgun(ler)?(e|lere)? (gore|bazinda)", r"\bgunlere\b", r"which day", r"hangi gun"]),
    "week": ("Week", "Week", "Hafta", [r"\bhafta(lik|lara|ya)?\b", r"\bweek(ly|s)?\b"]),
    "quarter": ("Quarter", "Quarter", "Çeyrek", [r"ceyrek", r"quarter"]),
    "month": ("Month", "Month", "Ay", [r"\bay(lik|lara|lar| bazinda|a gore)\b", r"\bmonth(ly|s)?\b", r"\btrend\b", r"zaman", r"over time", r"her ay"]),
    "city": ("City", "City", "Şehir", [r"sehir", r"\bil(ler|lere|e gore)?\b", r"\bcit(y|ies)\b", r"bolge", r"region", r"lokasyon", r"location"]),
    "category": ("Category", "Category", "Kategori", [r"kategori", r"categor", r"\btur(ler|e gore)?\b", r"genre"]),
    "product": ("Description", "Product", "Ürün", [r"urun", r"kitap", r"product", r"\btitle", r"\bbook", r"\beser"]),
    "customer": ("CustomerID", "Customer", "Müşteri", [r"musteri", r"customer", r"\bbayi", r"client", r"account"]),
}

MONTHS = {
    1: ["ocak", "january", "jan"], 2: ["subat", "february", "feb"], 3: ["mart", "march"],
    4: ["nisan", "april", "apr"], 5: ["mayis"], 6: ["haziran", "june", "jun"],
    7: ["temmuz", "july", "jul"], 8: ["agustos", "august", "aug"], 9: ["eylul", "september", "sep", "sept"],
    10: ["ekim", "october", "oct"], 11: ["kasim", "november", "nov"], 12: ["aralik", "december", "dec"],
}

TR_HINTS = ["hangi", "nedir", "kac", "nasil", "goster", "en cok", "en az", "gore", "bazinda", "mi ", "midir",
            "ver", "listele", "ciz", "olan", "icin", "ve ", "ile", "tahmin", "musteri", "ciro", "satis", "sehir",
            "ozet", "genel", "toplam", "durum", "yardim", "merhaba", "stok", "analiz", "aylik", "nerede", "hangisi"]

SUGGESTIONS = [
    "En çok ciro getiren 10 kategori hangisi?",
    "İstanbul ve Ankara'nın aylık cirosunu karşılaştır",
    "Show monthly units sold for Roman/Öykü",
    "Ekim ayında en çok satan 10 kitap",
    "Önümüzdeki 8 haftanın satış tahmini",
    "Churn riski olan en değerli müşteriler",
    "Hangi kategoriler birlikte satılıyor?",
    "Müşteri tipine göre ortalama indirim",
]


def normalize(text: str) -> str:
    text = text.replace("İ", "i").replace("I", "i").lower()
    text = text.replace("ı", "i")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[’'`]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


EN_HINTS = ["show", "what", "which", "how", " by ", "top", "the ", " for ", "sold", "monthly", "compare", " list",
            "give", "best", "most", " per ", " in ", " of ", "next", "who", "customers", "revenue", "sales"]


def detect_lang(q: str) -> str:
    n = " " + normalize(q) + " "
    tr_score = sum(h in n for h in TR_HINTS)
    en_score = sum(h in n for h in EN_HINTS)
    if tr_score != en_score:
        return "tr" if tr_score > en_score else "en"
    return "tr" if re.search(r"[ığşİĞŞ]", q) else "en"


# ----------------------------------------------------------------------------- parsing
def _empty_spec() -> dict:
    return {"intent": "query", "metric": None, "dim": None,
            "filters": {"City": [], "Category": [], "DocType": [], "CustomerType": [], "CustomerID": [],
                        "months": [], "quarters": []},
            "top_n": None, "ascending": False, "chart": None, "share": False, "horizon": 8, "lang": "en"}


def _cut(text: str, span: tuple[int, int]) -> str:
    return text[:span[0]] + " " * (span[1] - span[0]) + text[span[1]:]


def parse_question(q: str, vocab: dict) -> dict:
    """Rule-based NL parser. ``vocab`` holds the known entity values (cities, categories)."""
    spec = _empty_spec()
    spec["lang"] = detect_lang(q)
    t = " " + normalize(q) + " "

    # --- intents -----------------------------------------------------------------
    if re.search(r"tahmin|forecast|predict|ongor|onumuzdeki|gelecek (hafta|ay)|next \d* ?weeks?|projection", t):
        spec["intent"] = "forecast"
        m = re.search(r"(\d{1,2})\s*(hafta|week)", t)
        if m:
            spec["horizon"] = int(np.clip(int(m.group(1)), 2, 16))
    elif re.search(r"churn|kayip|win.?back|risk(li|i)? (musteri|customer)|at risk|segment|rfm|sadik|loyal", t):
        spec["intent"] = "segments"
    elif re.search(r"birlikte|together|cross.?sell|capraz|bundle|sepet analiz|market basket|association|iliski", t):
        spec["intent"] = "crosssell"
    elif re.search(r"\babc\b|xyz|stok|inventory|envanter|safety stock|guvenlik stogu|reorder|yeniden siparis|eoq", t):
        spec["intent"] = "inventory"
    elif re.search(r"icgoru|insight|oneri|recommend|tavsiye|ne yapmali|action", t):
        spec["intent"] = "insights"
    elif re.search(r"\b(yardim|help)\b|ne yapabilirsin|what can you|nasil kullan|ornek soru|example", t):
        spec["intent"] = "help"
    elif re.search(r"ozet|summary|genel durum|overview|kpi|performans nasil|how are we doing|durum nedir", t):
        spec["intent"] = "summary"

    # --- entities ------------------------------------------------------------------
    for raw in sorted(vocab["categories"], key=lambda s: -len(s)):
        n = normalize(raw)
        if len(n) < 4:
            continue
        m = re.search(r"(?<![a-z])" + re.escape(n) + r"(?![a-z])", t)
        if m:
            spec["filters"]["Category"].append(raw)
            t = _cut(t, m.span())
    for raw in vocab["cities"]:
        n = normalize(raw)
        m = re.search(r"\b" + re.escape(n) + r"(?:da|de|ta|te|dan|den|tan|ten|in|un|nin|nun|ya|ye|daki|deki|li|lu|a|e|i|u)?\b", t)
        if m and n not in ("unknown",):
            spec["filters"]["City"].append(raw)
            t = _cut(t, m.span())
    for code in vocab["doctypes"]:
        m = re.search(r"\b" + code.lower() + r"\b", t)
        if m:
            spec["filters"]["DocType"].append(code)
            t = _cut(t, m.span())
    for m in re.finditer(r"\b(?:type|tip|tipi)\s*([apsd])\b|\b([apsd])\s*tipi\b", t):
        letter = (m.group(1) or m.group(2)).upper()
        spec["filters"]["CustomerType"].append(f"Type {letter}")
    for m in re.finditer(r"\b([apsd]\d{5})\b", t):
        spec["filters"]["CustomerID"].append(m.group(1).upper())
        t = _cut(t, m.span())

    for qn, pat in [(1, r"\bq1\b|ilk ceyrek|1\. ceyrek|first quarter"), (2, r"\bq2\b|ikinci ceyrek|2\. ceyrek|second quarter"),
                    (3, r"\bq3\b|ucuncu ceyrek|3\. ceyrek|third quarter"), (4, r"\bq4\b|son ceyrek|dorduncu ceyrek|4\. ceyrek|fourth quarter|last quarter")]:
        m = re.search(pat, t)
        if m:
            spec["filters"]["quarters"].append(qn)
            t = _cut(t, m.span())
    if re.search(r"ilk yari|first half|h1\b", t):
        spec["filters"]["months"] += list(range(1, 7))
    elif re.search(r"ikinci yari|second half|h2\b", t):
        spec["filters"]["months"] += list(range(7, 13))
    found = []
    for mo, names in MONTHS.items():
        for name in names:
            if name in ("jan", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "feb", "apr", "sept") and spec["lang"] == "tr":
                continue
            m = re.search(r"\b" + name + r"(?:[a-z]{0,6})?\b" if len(name) > 3 else r"\b" + name + r"\b", t)
            if m:
                found.append((m.start(), mo))
                t = _cut(t, m.span())
                break
    found = [mo for _, mo in sorted(found)]
    if len(found) == 2 and re.search(r"(ile|ve|-|to|between|arasi|until|kadar|through)", t):
        a, b = found
        spec["filters"]["months"] += list(range(min(a, b), max(a, b) + 1))
    else:
        spec["filters"]["months"] += found

    # --- metric ----------------------------------------------------------------------
    for key, (_, _, kws) in METRICS.items():
        for kw in kws:
            m = re.search(r"\b" + re.escape(kw), t)
            if m:
                spec["metric"] = key
                if key in ("customers", "skus", "aov"):
                    t = _cut(t, m.span())
                break
        if spec["metric"]:
            break
    if spec["metric"] is None and re.search(r"en cok satan|best.?selling|top selling|cok satilan", t):
        spec["metric"] = "units"

    # --- dimension -------------------------------------------------------------------
    for key, (_, _, _, pats) in DIMS.items():
        if any(re.search(p, t) for p in pats):
            spec["dim"] = key
            break
    # "most sold books / categories" with explicit grouping words override generic month words
    if spec["dim"] is None:
        for col, key in [("City", "city"), ("Category", "category")]:
            if len(spec["filters"][col]) >= 2:
                spec["dim"] = key
    if spec["dim"] is None and len(spec["filters"]["CustomerID"]) >= 2:
        spec["dim"] = "customer"

    # --- modifiers -------------------------------------------------------------------
    m = re.search(r"\b(?:top|ilk|first|en (?:iyi|cok|fazla|buyuk|yuksek|az|dusuk|kotu)|best|worst|bottom|son)\s*(\d{1,3})\b", t) \
        or re.search(r"\b(\d{1,3})\s*(?:tane|adet|kategori|sehir|il|urun|kitap|musteri|cities|categories|products|books|customers|titles)\b", t)
    if m:
        spec["top_n"] = int(np.clip(int(m.group(1)), 1, 100))
    if re.search(r"en (az|dusuk|kotu|zayif)|lowest|least|worst|bottom|weakest", t):
        spec["ascending"] = True
    if re.search(r"pasta|\bpie\b|donut|halka", t):
        spec["chart"] = "pie"
    elif re.search(r"cizgi|\bline\b|trend|zaman serisi|time series", t):
        spec["chart"] = "line"
    elif re.search(r"\bbar\b|sutun|cubuk|column", t):
        spec["chart"] = "bar"
    elif re.search(r"tablo|\btable\b|liste|\blist\b", t):
        spec["chart"] = "table"
    if re.search(r"\bpay(i|lari)?\b|yuzde|share|percent|oran", t):
        spec["share"] = True
    if re.search(r"karsilastir|kiyasla|compare|\bvs\b|versus", t) and spec["dim"] is None:
        spec["dim"] = "month"

    if spec["intent"] == "query" and spec["metric"] is None and spec["dim"] is None \
            and not any(spec["filters"].values()):
        spec["intent"] = "help"
    if spec["intent"] == "query" and spec["metric"] is None:
        spec["metric"] = "revenue"
    return spec


# ----------------------------------------------------------------------------- optional LLM parser
LLM_SYSTEM = """You translate business questions about a B2B book distributor's 2020 sales data into a JSON query spec.
Return ONLY a JSON object with keys:
intent: one of query|forecast|segments|crosssell|inventory|insights|summary|help
metric: one of revenue|units|orders|customers|aov|discount|skus (null if not relevant)
dim: one of month|week|weekday|quarter|city|category|product|customer|doctype|ctype (null = single total)
filters: {City:[...], Category:[...], DocType:[...], CustomerType:[...], CustomerID:[...], months:[1-12], quarters:[1-4]}
top_n: integer or null; ascending: true/false; chart: bar|line|pie|table|null; share: true/false;
horizon: weeks for forecast (2-16); lang: "tr" if the question is Turkish else "en".
Use exact values from these lists for filters.
Cities: {cities}
Categories (top): {categories}
DocTypes: {doctypes}; CustomerTypes: Type A, Type D, Type P, Type S."""


def llm_parse(q: str, vocab: dict, api_key: str, model: str) -> dict | None:
    """Ask an LLM (Anthropic Messages API) for the spec. Returns None on any failure."""
    try:
        import requests
        system = LLM_SYSTEM.format(cities=", ".join(vocab["cities"][:80]),
                                   categories="; ".join(vocab["categories"][:120]),
                                   doctypes=", ".join(vocab["doctypes"]))
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": model, "max_tokens": 600, "system": system,
                  "messages": [{"role": "user", "content": q}]},
            timeout=25,
        )
        r.raise_for_status()
        text = "".join(b.get("text", "") for b in r.json().get("content", []))
        obj = json.loads(text[text.find("{"): text.rfind("}") + 1])
        spec = _empty_spec()
        for k in spec:
            if k in obj and obj[k] is not None:
                spec[k] = obj[k]
        for k in _empty_spec()["filters"]:
            spec["filters"].setdefault(k, [])
        if spec.get("metric") not in METRICS:
            spec["metric"] = "revenue" if spec["intent"] in ("query", "forecast") else None
        if spec.get("dim") not in DIMS:
            spec["dim"] = None
        return spec
    except Exception:
        return None


# ----------------------------------------------------------------------------- execution
def apply_filters(d: pd.DataFrame, f: dict) -> pd.DataFrame:
    m = pd.Series(True, index=d.index)
    for col in ["City", "Category", "DocType", "CustomerType", "CustomerID"]:
        if f.get(col):
            m &= d[col].astype(str).isin([str(v) for v in f[col]])
    if f.get("months"):
        m &= d["InvoiceDate"].dt.month.isin(f["months"])
    if f.get("quarters"):
        m &= d["InvoiceDate"].dt.quarter.isin(f["quarters"])
    return d[m]


def _metric_by(d: pd.DataFrame, metric: str, by: str | None):
    g = d.groupby(by, observed=True) if by else None
    if metric == "revenue":
        return g["Revenue"].sum() if g is not None else d["Revenue"].sum()
    if metric == "units":
        return g["Quantity"].sum() if g is not None else d["Quantity"].sum()
    if metric == "orders":
        return g["InvoiceNo"].nunique() if g is not None else d["InvoiceNo"].nunique()
    if metric == "customers":
        return g["CustomerID"].nunique() if g is not None else d["CustomerID"].nunique()
    if metric == "skus":
        return g["StockCode"].nunique() if g is not None else d["StockCode"].nunique()
    if metric == "aov":
        if g is None:
            return d["Revenue"].sum() / max(d["InvoiceNo"].nunique(), 1)
        return g["Revenue"].sum() / g["InvoiceNo"].nunique().replace(0, np.nan)
    if metric == "discount":
        if g is None:
            return 1 - d["Revenue"].sum() / d["ListValue"].sum()
        return 1 - g["Revenue"].sum() / g["ListValue"].sum()
    raise ValueError(metric)


def fmt_value(metric: str, v: float) -> str:
    if metric in ("revenue", "aov"):
        return an.fmt_money(v)
    if metric == "discount":
        return f"{v * 100:.1f}%"
    return f"{v:,.0f}"


def _label(metric: str, lang: str) -> str:
    en, tr, _ = METRICS[metric]
    return tr if lang == "tr" else en


def _dim_label(dim: str, lang: str) -> str:
    _, en, tr, _ = DIMS[dim]
    return tr if lang == "tr" else en


def describe_filters(f: dict, lang: str) -> str:
    parts = []
    names = {"City": ("City", "Şehir"), "Category": ("Category", "Kategori"), "DocType": ("Document type", "Belge tipi"),
             "CustomerType": ("Customer type", "Müşteri tipi"), "CustomerID": ("Customer", "Müşteri")}
    for k, (en, tr) in names.items():
        if f.get(k):
            parts.append(f"{tr if lang == 'tr' else en}: {', '.join(map(str, f[k]))}")
    if f.get("months"):
        ms = sorted(set(f["months"]))
        lab = [pd.Timestamp(2020, m, 1).strftime("%b") for m in ms]
        parts.append(("Ay" if lang == "tr" else "Month") + ": " + (f"{lab[0]}–{lab[-1]}" if len(ms) > 2 and ms == list(range(ms[0], ms[-1] + 1)) else ", ".join(lab)))
    if f.get("quarters"):
        parts.append(("Çeyrek" if lang == "tr" else "Quarter") + ": " + ", ".join(f"Q{q}" for q in sorted(set(f["quarters"]))))
    return " · ".join(parts)


def _style(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend_title=None)
    return fig


def execute(spec: dict, d: pd.DataFrame, ctx: dict) -> dict:
    """Run a spec. ``ctx`` gives access to precomputed artefacts (rfm, rules, prod)."""
    lang = spec.get("lang", "en")
    tr = lang == "tr"
    f = spec.get("filters", {})
    sub = apply_filters(d, f)
    fdesc = describe_filters(f, lang)
    scope = (f" ({fdesc})" if fdesc else "")
    out = {"text": "", "df": None, "fig": None, "spec": spec}

    if spec["intent"] == "help":
        out["text"] = ("Verilerle ilgili Türkçe ya da İngilizce soru sorabilirsiniz. Ciro, adet, sipariş, müşteri, "
                       "ortalama sipariş tutarı ve indirim gibi ölçüleri; ay, hafta, gün, çeyrek, şehir, kategori, ürün, "
                       "müşteri ve belge tipi kırılımlarında hesaplayabilir, tablo ve grafik üretebilir, tahmin, segment, "
                       "çapraz satış ve stok analizleri yapabilirim. Örnek sorular aşağıda." if tr else
                       "Ask anything about the data in English or Turkish. I can compute revenue, units, orders, customers, "
                       "average order value and discount by month, week, weekday, quarter, city, category, product, customer "
                       "or document type, build tables and charts, and run forecasts, segmentation, cross-sell and inventory "
                       "analyses. Try one of the examples below.")
        return out

    if sub.empty:
        out["text"] = "Bu filtrelerle eşleşen veri bulunamadı." if tr else "No data matches these filters."
        return out

    if spec["intent"] == "summary":
        k = an.kpis(sub)
        rows = [("Net revenue" if not tr else "Net ciro", an.fmt_money(k["revenue"])),
                ("Orders" if not tr else "Sipariş", f"{k['orders']:,}"),
                ("Active customers" if not tr else "Aktif müşteri", f"{k['customers']:,}"),
                ("Units sold" if not tr else "Satılan adet", f"{k['units']:,.0f}"),
                ("Avg. order value" if not tr else "Ort. sipariş tutarı", an.fmt_money(k["aov"])),
                ("Avg. discount" if not tr else "Ort. indirim", f"{k['discount'] * 100:.1f}%"),
                ("Active SKUs" if not tr else "Aktif ürün", f"{k['skus']:,}")]
        out["df"] = pd.DataFrame(rows, columns=["KPI", "Value"])
        mon = sub.groupby("Month")["Revenue"].sum().reset_index()
        out["fig"] = _style(px.bar(mon, x="Month", y="Revenue"), 300)
        best = mon.loc[mon["Revenue"].idxmax(), "Month"]
        out["text"] = (f"Seçili kapsamda{scope} net ciro **{an.fmt_money(k['revenue'])}**, {k['orders']:,} sipariş ve "
                       f"{k['customers']:,} aktif müşteri var. En güçlü ay **{best:%B}**." if tr else
                       f"For the selected scope{scope}, net revenue is **{an.fmt_money(k['revenue'])}** from "
                       f"{k['orders']:,} orders and {k['customers']:,} active customers. The strongest month is **{best:%B}**.")
        return out

    if spec["intent"] == "forecast":
        col = "Quantity" if spec.get("metric") == "units" else "Revenue"
        ser = an.weekly_series(sub, col)
        if len(ser) < 20:
            out["text"] = "Tahmin için yeterli haftalık veri yok." if tr else "Not enough weekly history to forecast."
            return out
        h = int(spec.get("horizon") or 8)
        fc = an.run_forecast(ser, h)
        hist = pd.DataFrame({"Week": ser.index, "Value": ser.values, "Series": "Actual"})
        fut = pd.DataFrame({"Week": fc["future"], "Value": fc["forecast"], "Series": "Forecast"})
        out["fig"] = _style(px.line(pd.concat([hist, fut]), x="Week", y="Value", color="Series", markers=True))
        out["df"] = pd.DataFrame({"Week starting": fc["future"].date, "Forecast": fc["forecast"].round(0),
                                  "Lower (80%)": fc["lower"].round(0), "Upper (80%)": fc["upper"].round(0)})
        tot = fc["forecast"].sum()
        val = an.fmt_money(tot) if col == "Revenue" else f"{tot:,.0f}"
        mape = fc["scores"].iloc[0]["MAPE (%)"]
        out["text"] = (f"Önümüzdeki **{h} hafta** için{scope} toplam tahmin **{val}**. En iyi model **{fc['best']}** "
                       f"(geriye dönük test MAPE %{mape:.1f})." if tr else
                       f"The forecast for the next **{h} weeks**{scope} totals **{val}**. Best model: **{fc['best']}** "
                       f"(back-test MAPE {mape:.1f}%).")
        return out

    if spec["intent"] == "segments":
        rfm = ctx["rfm"]
        if f.get("City"):
            rfm = rfm[rfm["City"].astype(str).isin(f["City"])]
        risk = rfm[rfm["ChurnRisk"]].nlargest(spec.get("top_n") or 15, "Monetary")
        out["df"] = risk[["CustomerID", "City", "Segment", "Recency", "Frequency", "Monetary"]].rename(
            columns={"Recency": "Days silent", "Frequency": "Orders", "Monetary": "Revenue"})
        seg = rfm.groupby("Segment")["Monetary"].sum().reset_index()
        out["fig"] = _style(px.bar(seg.sort_values("Monetary", ascending=False), x="Segment", y="Monetary"), 300)
        out["text"] = (f"**{int(rfm['ChurnRisk'].sum())} müşteri** churn sinyali veriyor (normal sipariş aralığının 2 katından "
                       f"uzun süredir sipariş yok). Bu müşterilerin cirosu **{an.fmt_money(rfm.loc[rfm['ChurnRisk'], 'Monetary'].sum())}**. "
                       "En değerli olanlar tabloda." if tr else
                       f"**{int(rfm['ChurnRisk'].sum())} customers** show churn signals (silent for more than twice their usual "
                       f"purchase gap), worth **{an.fmt_money(rfm.loc[rfm['ChurnRisk'], 'Monetary'].sum())}** in revenue. "
                       "The most valuable ones are listed below.")
        return out

    if spec["intent"] == "crosssell":
        rules = ctx["rules"]
        if f.get("Category"):
            rules = rules[rules["If customer buys"].isin(f["Category"]) | rules["They also buy"].isin(f["Category"])]
        rules = rules[rules["Lift"] >= 1.2].sort_values(["Lift", "Confidence"], ascending=False)
        out["df"] = rules.head(spec.get("top_n") or 15).round(3)
        if rules.empty:
            out["text"] = "Anlamlı bir birliktelik kuralı bulunamadı." if tr else "No meaningful association rules found."
        else:
            r = rules.iloc[0]
            out["text"] = (f"En güçlü ilişki: **{r['If customer buys']}** içeren faturaların %{r['Confidence'] * 100:.0f}'inde "
                           f"**{r['They also buy']}** de var (lift {r['Lift']:.1f}x)." if tr else
                           f"Strongest link: **{r['Confidence'] * 100:.0f}%** of invoices with **{r['If customer buys']}** also "
                           f"contain **{r['They also buy']}** (lift {r['Lift']:.1f}x).")
        return out

    if spec["intent"] == "inventory":
        prod = ctx["prod"]
        if f.get("Category"):
            prod = prod[prod["Category"].astype(str).isin(f["Category"])]
        summary = prod.groupby("Class").agg(SKUs=("StockCode", "count"), Revenue=("Revenue", "sum")).reset_index()
        summary["Revenue share"] = (summary["Revenue"] / summary["Revenue"].sum()).round(3)
        out["df"] = prod.nlargest(spec.get("top_n") or 15, "Revenue")[
            ["Product", "Category", "Class", "WeeklyMean", "WeeklyStd", "CV"]].round(2)
        out["fig"] = _style(px.bar(summary, x="Class", y="Revenue share"), 300)
        a = prod[prod["ABC"] == "A"]
        out["text"] = (f"{len(prod):,} ürünün **{len(a):,}** tanesi (A sınıfı) cironun %80'ini oluşturuyor. Tabloda en yüksek "
                       "cirolu ürünlerin talep oynaklığı yer alıyor." if tr else
                       f"**{len(a):,}** of {len(prod):,} SKUs (class A) generate 80% of revenue. The table shows demand "
                       "volatility for the top-revenue SKUs.")
        return out

    if spec["intent"] == "insights":
        ins = an.generate_insights(sub, ctx["rfm"], ctx["prod"], ctx["rules"], None)
        lines = [f"- **{dept}**: {txt}" for dept, items in ins.items() for _, txt in items[:2]]
        out["text"] = ("Öne çıkan içgörüler:\n\n" if tr else "Key insights:\n\n") + "\n".join(lines)
        return out

    # ------------------------------------------------------------------ generic query
    metric, dim = spec["metric"], spec.get("dim")
    mlabel = _label(metric, lang)
    if not dim:
        val = _metric_by(sub, metric, None)
        trend = _metric_by(sub, metric, "Month").reset_index()
        trend.columns = ["Month", mlabel]
        out["df"] = trend
        out["fig"] = _style(px.line(trend, x="Month", y=mlabel, markers=True), 320)
        out["text"] = (f"{mlabel}{scope}: **{fmt_value(metric, val)}**. Aylık dağılım aşağıda." if tr else
                       f"{mlabel}{scope}: **{fmt_value(metric, val)}**. Monthly breakdown below.")
        return out

    col, dlabel_en, dlabel_tr, _ = DIMS[dim]
    dlabel = dlabel_tr if tr else dlabel_en
    # if user compares multiple entities over time → split series
    split = None
    if dim in ("month", "week", "quarter", "weekday"):
        for fc_ in ("City", "Category"):
            if len(f.get(fc_, [])) >= 2:
                split = fc_
    if split:
        res = _metric_by(sub, metric, [col, split]).reset_index()
        split_key = "city" if split == "City" else "category"
        split = DIMS[split_key][2] if tr else DIMS[split_key][1]
        res.columns = [dlabel, split, mlabel]
    else:
        res = _metric_by(sub, metric, col).reset_index()
        res.columns = [dlabel, mlabel]
    if dim == "weekday":
        res[dlabel] = res[dlabel].map(lambda i: an.WEEKDAYS[int(i)])
    time_dim = dim in ("month", "week", "quarter", "weekday")
    if not time_dim:
        res = res.sort_values(mlabel, ascending=spec.get("ascending", False))
        n = spec.get("top_n") or 10
        total = res[mlabel].sum()
        res = res.head(n).reset_index(drop=True)
        if metric in ("revenue", "units", "orders") and total:
            res[("Pay" if tr else "Share")] = (res[mlabel] / total).round(4)
        res[dlabel] = res[dlabel].astype(str)

    chart = spec.get("chart") or ("line" if time_dim else "bar")
    if chart == "pie" and not time_dim:
        fig = px.pie(res, names=dlabel, values=mlabel, hole=0.5)
    elif chart == "line" or (time_dim and chart != "bar"):
        fig = px.line(res, x=dlabel, y=mlabel, color=split, markers=True)
    elif chart == "table":
        fig = None
    else:
        if time_dim:
            fig = px.bar(res, x=dlabel, y=mlabel, color=split, barmode="group")
        else:
            fig = px.bar(res.iloc[::-1], x=mlabel, y=dlabel, orientation="h")
            fig.update_yaxes(title=None)
    if fig is not None:
        bar_h = not time_dim and chart not in ("pie", "line")
        fig = _style(fig, max(320, 28 * len(res) + 80) if bar_h else 380)
    out["fig"] = fig
    out["df"] = res

    if split:
        tot = res.groupby(split)[mlabel].sum().sort_values(ascending=False)
        lead = tot.index[0]
        out["text"] = (f"{dlabel} bazında {mlabel.lower()} karşılaştırması{scope}. Toplamda önde olan **{lead}** "
                       f"({fmt_value(metric, tot.iloc[0])})." if tr else
                       f"{mlabel} by {dlabel.lower()}{scope}. Overall leader: **{lead}** ({fmt_value(metric, tot.iloc[0])}).")
    elif time_dim:
        best = res.loc[res[mlabel].idxmax()]
        worst = res.loc[res[mlabel].idxmin()]
        bl = best[dlabel] if not isinstance(best[dlabel], pd.Timestamp) else best[dlabel].strftime("%b %Y")
        wl = worst[dlabel] if not isinstance(worst[dlabel], pd.Timestamp) else worst[dlabel].strftime("%b %Y")
        out["text"] = (f"{dlabel} bazında {mlabel.lower()}{scope}. En yüksek: **{bl}** ({fmt_value(metric, best[mlabel])}), "
                       f"en düşük: **{wl}** ({fmt_value(metric, worst[mlabel])})." if tr else
                       f"{mlabel} by {dlabel.lower()}{scope}. Highest: **{bl}** ({fmt_value(metric, best[mlabel])}); "
                       f"lowest: **{wl}** ({fmt_value(metric, worst[mlabel])}).")
    else:
        top = res.iloc[0]
        share_col = "Pay" if tr else "Share"
        share_txt = f" – %{top[share_col] * 100:.1f}" if (tr and share_col in res) else (
            f" – {top[share_col] * 100:.1f}% of total" if share_col in res else "")
        order = ("en düşük" if spec.get("ascending") else "en yüksek") if tr else ("lowest" if spec.get("ascending") else "highest")
        out["text"] = (f"{dlabel} bazında {mlabel.lower()} sıralaması{scope} ({order} {len(res)}). Birinci: **{top[dlabel]}** "
                       f"({fmt_value(metric, top[mlabel])}{share_txt})." if tr else
                       f"{dlabel} ranking by {mlabel.lower()}{scope} ({order} {len(res)} shown). "
                       f"#1: **{top[dlabel]}** ({fmt_value(metric, top[mlabel])}{share_txt}).")
    return out


def build_vocab(d: pd.DataFrame) -> dict:
    cats = d.groupby("Category", observed=True)["Revenue"].sum().sort_values(ascending=False).index.astype(str).tolist()
    cities = d.groupby("City", observed=True)["Revenue"].sum().sort_values(ascending=False).index.astype(str).tolist()
    return {"categories": cats, "cities": [c for c in cities if c != "UNKNOWN"],
            "doctypes": sorted(d["DocType"].astype(str).unique().tolist())}


# ----------------------------------------------------------------------------- conversation helpers
def follow_ups(spec: dict, result: dict) -> list[str]:
    """Suggest three natural next questions that the parser understands."""
    tr = spec.get("lang", "en") == "tr"
    intent, dim, metric = spec.get("intent"), spec.get("dim"), spec.get("metric") or "revenue"
    df_ = result.get("df")
    if intent == "query" and dim in ("city", "category", "customer") and df_ is not None and len(df_):
        top = str(df_.iloc[0, 0])
        dlab = DIMS[dim][2 if tr else 1].lower()
        return ([f"{top} aylık ciro trendi", f"{dlab} bazında ciro payı pasta grafik", f"{dlab} bazında satılan adet"]
                if tr else
                [f"Monthly revenue trend for {top}", f"Revenue share by {dlab} as a pie chart", f"Units sold by {dlab}"])
    if intent == "query" and dim in ("month", "week", "quarter", "weekday"):
        return (["Önümüzdeki 8 haftanın satış tahmini", "Haftanın günlerine göre sipariş sayısı",
                 "En çok ciro getiren 10 şehir"] if tr else
                ["Forecast the next 8 weeks", "Orders by weekday", "Top 10 cities by revenue"])
    if intent == "query":
        return (["Aylık ciro trendi", "En çok satan 10 kitap", "Müşteri tipine göre ortalama indirim"] if tr else
                ["Monthly revenue trend", "Top 10 best-selling books", "Average discount by customer type"])
    if intent == "forecast":
        return (["Kategori bazında ciro", "Churn riski olan en değerli müşteriler", "Aylık sipariş sayısı"] if tr else
                ["Revenue by category", "Most valuable customers at churn risk", "Monthly orders"])
    if intent in ("segments", "crosssell", "inventory"):
        return (["Hangi kategoriler birlikte satılıyor?", "ABC stok analizi", "Genel özet"] if tr else
                ["Which categories sell together?", "ABC inventory analysis", "Give me a summary"])
    return (["Genel özet", "En çok ciro getiren 10 kategori", "Önümüzdeki 8 haftanın satış tahmini"] if tr else
            ["Give me a summary", "Top 10 categories by revenue", "Forecast the next 8 weeks"])


def reasoning_steps(spec: dict, rows_in_scope: int) -> list[str]:
    """Short, human-readable trace of how the question was interpreted."""
    tr = spec.get("lang", "en") == "tr"
    intent, metric, dim = spec.get("intent"), spec.get("metric"), spec.get("dim")
    steps = []
    if intent == "query" and metric:
        m = METRICS[metric][1 if tr else 0]
        d = DIMS[dim][2 if tr else 1] if dim else ("Toplam" if tr else "Total")
        steps.append(f"{'Ölçü' if tr else 'Measure'}: {m} · {'Kırılım' if tr else 'By'}: {d}")
    else:
        names = {"forecast": ("Tahmin modeli", "Forecast model"), "segments": ("RFM segmentasyonu", "RFM segmentation"),
                 "crosssell": ("Birliktelik kuralları", "Association rules"), "inventory": ("ABC–XYZ analizi", "ABC–XYZ analysis"),
                 "insights": ("İçgörü üretimi", "Insight generation"), "summary": ("KPI özeti", "KPI summary"),
                 "help": ("Yardım", "Help")}
        steps.append(names.get(intent, (intent, intent))[0 if tr else 1])
    flt = describe_filters(spec.get("filters", {}), spec.get("lang", "en"))
    if flt:
        steps.append(flt)
    steps.append(f"{rows_in_scope:,} {'satır tarandı' if tr else 'rows scanned'}")
    return steps
