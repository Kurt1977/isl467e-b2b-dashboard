"""Data loading and preprocessing pipeline for the B2B transaction dataset."""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CITY_FIXES = {"IÇEL": "MERSIN", "K.MARAS": "KAHRAMANMARAS", "AFYON": "AFYONKARAHISAR"}


def _read_raw() -> pd.DataFrame:
    """Read the raw dataset. Prefers the original Excel file; falls back to the
    lossless compressed CSV parts (the 40 MB workbook exceeds GitHub's web upload limit)."""
    xlsx = os.path.join(DATA_DIR, "B2B_Transaction_Data_.xlsx")
    if os.path.exists(xlsx):
        return pd.read_excel(xlsx, dtype={"StockCode": str})
    parts = sorted(glob.glob(os.path.join(DATA_DIR, "B2B_Transaction_Data_part*.csv.xz")))
    if not parts:
        raise FileNotFoundError("Dataset not found in ./data")
    return pd.concat([pd.read_csv(p, dtype={"StockCode": str}) for p in parts], ignore_index=True)


@st.cache_data(show_spinner="Loading and preparing transaction data…")
def load_and_clean() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Full preprocessing pipeline. Returns (clean_df, step_log, quality_stats)."""
    df = _read_raw()
    log: list[dict] = []

    def step(name: str, action: str, before: int, after: int) -> None:
        log.append({"Step": name, "Action": action, "Rows before": before, "Rows after": after,
                    "Rows removed": before - after})

    n0 = len(df)
    quality = {"raw_rows": n0, "missing": df.isna().sum().to_dict(),
               "exact_duplicates": int(df.duplicated().sum())}

    # 1) Types & text normalisation
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    for c in ["InvoiceNo", "CustomerID", "Description", "Category", "City"]:
        df[c] = df[c].astype("string").str.strip()
    df["City"] = df["City"].str.upper().replace(CITY_FIXES)
    for c in ["Quantity", "NetPrice", "UnitPrice"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    step("1. Type casting & text cleaning", "Parsed dates and numerics, trimmed whitespace, harmonised city names "
         "(IÇEL→MERSIN, K.MARAS→KAHRAMANMARAS, AFYON→AFYONKARAHISAR)", n0, len(df))

    # 2) Missing values
    n = len(df)
    df = df.dropna(subset=["InvoiceDate", "Quantity"])
    df["Category"] = df["Category"].fillna("Unknown")
    df["City"] = df["City"].fillna("UNKNOWN")
    step("2. Missing values", "Dropped rows without date or quantity; imputed missing Category and City "
         "as 'Unknown' to keep revenue complete", n, len(df))

    # 3) Invalid transactions
    n = len(df)
    invalid = (df["Quantity"] <= 0) | (df["NetPrice"] <= 0) | (df["UnitPrice"] < 0.5)
    quality["invalid_lines"] = int(invalid.sum())
    df = df[~invalid]
    step("3. Invalid lines", "Removed zero-quantity, non-positive net price and near-zero list price lines "
         "(price corrections, free goods)", n, len(df))

    # 4) Price consistency
    n_over = int((df["NetPrice"] > df["UnitPrice"]).sum())
    quality["net_above_list"] = n_over
    df["NetPrice"] = np.minimum(df["NetPrice"], df["UnitPrice"])
    step("4. Price consistency", f"Net price capped at list price ({n_over} lines affected)", len(df), len(df))

    # 5) Outlier flagging – kept, bulk orders are genuine B2B behaviour
    q1, q3 = df["Quantity"].quantile([0.25, 0.75])
    upper = max(q3 + 3 * (q3 - q1), 20)
    df["BulkLine"] = df["Quantity"] > upper
    quality["bulk_lines"] = int(df["BulkLine"].sum())
    step("5. Outlier flagging", f"{quality['bulk_lines']:,} bulk lines flagged (quantity > {upper:.0f}); "
         "retained because large orders are normal for a distributor", len(df), len(df))

    # 6) Feature engineering
    df["Revenue"] = df["Quantity"] * df["NetPrice"]
    df["ListValue"] = df["Quantity"] * df["UnitPrice"]
    df["DiscountRate"] = 1 - df["NetPrice"] / df["UnitPrice"]
    df["DocType"] = df["InvoiceNo"].str[:3]
    df["CustomerType"] = "Type " + df["CustomerID"].str[0]
    df["Month"] = df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    df["Week"] = df["InvoiceDate"].dt.to_period("W-SUN").dt.start_time
    df["Weekday"] = df["InvoiceDate"].dt.dayofweek
    df["Quarter"] = "Q" + df["InvoiceDate"].dt.quarter.astype(str)
    step("6. Feature engineering", "Revenue, list value, discount rate, document type, customer type, "
         "month, week, weekday and quarter", len(df), len(df))

    for c in ["CustomerID", "Description", "Category", "City", "DocType", "CustomerType", "Quarter", "StockCode"]:
        df[c] = df[c].astype("category")
    df = df.reset_index(drop=True)
    quality["clean_rows"] = len(df)
    return df, pd.DataFrame(log), quality
