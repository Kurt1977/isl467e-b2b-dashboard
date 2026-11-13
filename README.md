# 📚 B2B Book Distributor – Intelligent Decision Support Dashboard

**ISL 467E – Management Engineering Applications with Python · Homework I**

An interactive Streamlit dashboard that turns 664K B2B transaction lines (2020) into a decision-support
system for the **Sales**, **Marketing** and **Supply Chain** departments.

## Dashboard structure

| Tab | Content |
|---|---|
| 🏠 Executive Overview | 8 KPIs with month-over-month deltas, monthly revenue/orders, revenue mix by document type, customer type and city |
| 💰 Sales | Category treemap, seasonality heatmap, best-selling titles, weekday × month pattern, dynamic group-by analysis |
| 🎯 Marketing & Customers | RFM + K-Means segmentation (adjustable k), churn-risk win-back list, Pareto concentration, category association rules (support / confidence / lift), discount effectiveness |
| 🚚 Supply Chain | ABC–XYZ classification, inventory policy simulator (safety stock, reorder point, EOQ by service level & lead time), weekly demand, regional demand |
| 🔮 Forecast | Weekly revenue/unit forecast for total, category or city; back-tests Damped Holt, Gradient Boosting and Moving Average models and picks the best by MAPE, with an 80% interval |
| 💡 Insights & Actions | Auto-generated, filter-aware insights per department + recommended actions + Excel analysis pack download |
| 🧹 Data Quality | Step-by-step preprocessing log, missing values, discount distribution, clean data download |

Global sidebar filters (date range, document type, customer type, city, category, Top-N) apply to all tabs.

## Data preprocessing
1. Type casting, whitespace trimming, city name harmonisation (IÇEL→MERSIN, K.MARAS→KAHRAMANMARAS, AFYON→AFYONKARAHISAR)
2. Missing values: rows without quantity dropped; missing Category/City imputed as "Unknown"
3. Invalid lines removed (zero quantity, non-positive net price, ~0 list price)
4. Price consistency check (net price ≤ list price)
5. Bulk-order outliers flagged (kept – normal for B2B)
6. Feature engineering: revenue, list value, discount rate, document type, customer type, month/week/weekday/quarter

## Project files
```
streamlit_app.py      # main application
requirements.txt      # dependencies
data/                 # dataset
.streamlit/config.toml
```
> The original `B2B_Transaction_Data_.xlsx` (40 MB) exceeds GitHub's browser upload limit, so the identical data is
> stored as two lossless compressed CSV parts (`data/B2B_Transaction_Data_part1/2.csv.xz`). If the original Excel file
> is placed in `data/`, the app reads it automatically.

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
