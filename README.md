# Intelligent Dashboard Design using Python Streamlit

**ISL 467E – Management Engineering Applications with Python · Homework I**
Istanbul Technical University · Management Engineering Program

A decision support dashboard for a B2B book distributor, built with Streamlit on the course dataset
`B2B_Transaction_Data_.xlsx` (664,154 transaction lines, January–December 2020). The dashboard serves the
**sales**, **marketing** and **supply chain** departments with KPIs, visuals, forecasts and insights.

| | |
|---|---|
| Live dashboard | https://isl467e-b2b-dashboard.streamlit.app |
| GitHub repository | https://github.com/Kurt1977/isl467e-b2b-dashboard |

---

## Project structure

```
isl467e-b2b-dashboard/
├── streamlit_app.py        Main Python source code (page layout and interactions)
├── requirements.txt        Python libraries required to run the app
├── data/                   Dataset file
│   ├── B2B_Transaction_Data_part1.csv.xz
│   └── B2B_Transaction_Data_part2.csv.xz
├── data_pipeline.py        Data loading and preprocessing
├── analytics.py            KPIs, segmentation, association rules, ABC-XYZ, inventory policy, forecasting, insights
├── assistant.py            Data Assistant (natural-language questions in Turkish or English)
├── ui.py                   Visual system: colours, chart template, page components
└── .streamlit/config.toml  Streamlit theme
```

**About the dataset file.** The original `B2B_Transaction_Data_.xlsx` is 40 MB, above GitHub's 25 MB browser upload
limit. The identical data (all rows and columns) is therefore stored as two lossless compressed CSV parts in `data/`.
If the original Excel file is placed in `data/`, the application reads it directly.

## Dashboard pages

| Page | Purpose | Content |
|---|---|---|
| Overview | All departments | Net revenue, orders, active customers, average order value, units, discount; monthly performance; top titles; order-day pattern; revenue mix |
| Sales | Sales | Category treemap, category × month seasonality heatmap, best-selling titles, weekday × month pattern, custom breakdown (any dimension × measure) |
| Customers & Marketing | Marketing | RFM + K-Means segmentation (adjustable number of segments), win-back priority list, customer concentration (Pareto), category association rules (support, confidence, lift), discount effectiveness |
| Supply Chain | Supply chain | ABC–XYZ classification, inventory policy simulator (safety stock, reorder point, EOQ by service level, lead time, review period and cost parameters), weekly demand, regional demand profile |
| Forecast | Sales / supply chain | Weekly revenue or unit forecast for the total, a category or a city; back-test of three models and automatic selection by MAPE; 80% interval |
| Insights | All departments | Findings generated from the filtered data, recommended actions with KPIs, Excel analysis pack |
| Data Quality | All departments | Preprocessing log, missing values, discount distribution, clean data download |

A **Filters** menu (period, city, category, customer type, document type) applies to every page.
The **Data Assistant** in the left column answers questions such as *"En çok ciro getiren 10 kategori hangisi?"* or
*"Show monthly units sold for Roman/Öykü"* with a short answer, a chart, a table and CSV / Excel downloads, and
suggests follow-up questions.

## How the evaluation criteria are addressed

| Criterion | Implementation |
|---|---|
| Data preprocessing | Type casting, text trimming, city name harmonisation (IÇEL→MERSIN, K.MARAS→KAHRAMANMARAS, AFYON→AFYONKARAHISAR), missing value handling, removal of invalid lines (zero quantity, non-positive net price, near-zero list price), price consistency check, bulk-order outlier flagging, feature engineering (revenue, list value, discount rate, document type, customer type, time features). Every step is logged on the Data Quality page. |
| Intelligent and analytical features | K-Means customer segmentation on RFM, churn-risk detection, market-basket association rules, ABC–XYZ classification, inventory policy optimisation (safety stock, reorder point, EOQ), forecasting with Damped Holt, Gradient Boosting and Moving Average with back-testing, natural-language Data Assistant. |
| Dashboard structure and hierarchy | KPI summary first, then department pages from descriptive (Sales) to diagnostic (Customers, Supply Chain) to predictive (Forecast) and prescriptive (Insights). |
| Visual design quality | Consistent colour system and chart template, card layout, highlighted key values, readable labels. |
| Interactivity and usability | Global filters, adjustable model parameters, custom breakdown, model comparison, downloads, conversational assistant. |
| Code quality | Modular files with a single responsibility each, cached data loading and analytics, docstrings, no hard-coded local paths. |
| Insights | Automatically generated findings per department and a table of recommended actions with the KPI to track. |

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deployment

The app is deployed on Streamlit Community Cloud from the `main` branch of this repository
(main file `streamlit_app.py`, Python 3.12).

**Optional:** to let a language model interpret assistant questions, add `ANTHROPIC_API_KEY` to the app's Secrets on
Streamlit Cloud. The model only translates the question into a structured query; all numbers are computed with pandas.
Without a key the built-in rule-based engine is used.
