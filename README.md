# B2B Book Distribution — Decision Support Dashboard

**ISL 467E – Management Engineering Applications with Python · Homework I**

An interactive Streamlit application that turns 664K B2B transaction lines (2020) into a decision-support
system for the **sales**, **marketing** and **supply chain** functions of a book distributor.

## Pages

| Page | Content |
|---|---|
| Overview | Eight KPIs with month-over-month change, monthly revenue and orders, revenue mix |
| Sales | Category treemap, seasonality heatmap, best-selling titles, weekday pattern, custom breakdown |
| Customers & Marketing | RFM + K-Means segmentation, win-back list, customer concentration, category association rules, discount effectiveness |
| Supply Chain | ABC–XYZ classification, inventory policy simulator (safety stock, reorder point, EOQ), weekly demand, regional demand |
| Forecast | Weekly revenue / unit forecast for total, category or city; back-test of Damped Holt, Gradient Boosting and Moving Average, best model selected by MAPE, 80% interval |
| Insights | Findings generated from the filtered data, recommended actions, Excel analysis pack |
| Data Quality | Preprocessing log, missing values, discount distribution, clean data download |

The **Filters** menu in the header (period, city, category, customer type, document type) applies to every page.
The **Data Assistant** lives in the left rail and is available on every page.

## Data Assistant (left rail)
Ask questions in Turkish or English and receive an answer, a table, a chart and a CSV / Excel file.
The assistant translates a question into a structured query (intent, measure, dimension, filters, ranking, chart type)
and executes it with pandas, so every answer is computed from the data.

* **Built-in engine (default):** rule-based parser for Turkish and English; no API key needed.
* **LLM-assisted (optional):** add `ANTHROPIC_API_KEY` (and optionally `ANTHROPIC_MODEL`) to the app's Secrets on
  Streamlit Cloud. The model only produces the query spec; computation stays in pandas.

Example questions: *"En çok ciro getiren 10 kategori hangisi?"*, *"İstanbul ve Ankara'nın aylık cirosunu karşılaştır"*,
*"Show monthly units sold for Roman/Öykü"*, *"Önümüzdeki 8 haftanın satış tahmini"*, *"Churn riski olan en değerli müşteriler"*.

## Data preprocessing
1. Type casting, whitespace trimming, city name harmonisation (IÇEL→MERSIN, K.MARAS→KAHRAMANMARAS, AFYON→AFYONKARAHISAR)
2. Missing values: rows without quantity dropped; missing Category / City imputed as "Unknown"
3. Invalid lines removed (zero quantity, non-positive net price, near-zero list price)
4. Price consistency check (net price ≤ list price)
5. Bulk-order outliers flagged and retained (normal B2B behaviour)
6. Feature engineering: revenue, list value, discount rate, document type, customer type, month / week / weekday / quarter

## Project structure
```
streamlit_app.py      page layout and interactions
data_pipeline.py      data loading and preprocessing
analytics.py          KPIs, segmentation, basket rules, ABC-XYZ, inventory policy, forecasting, insights
assistant.py          natural-language Data Assistant
ui.py                 visual system (theme, chart template, components)
requirements.txt      dependencies
data/                 dataset
.streamlit/config.toml
```
The original `B2B_Transaction_Data_.xlsx` (40 MB) exceeds GitHub's browser upload limit, so the identical data is stored
as two lossless compressed CSV parts in `data/`. If the Excel file is placed in `data/`, the app reads it directly.

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
