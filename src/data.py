"""
data.py — Data loading utilities for sovereign credit rating prediction.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def safe_read_csv(path, **kwargs):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def load_ratings(raw_dir):
    raw_dir = Path(raw_dir)
    df_current = safe_read_csv(raw_dir / "credit_ratings" / "current_ratings_2024.csv")
    df_hist = safe_read_csv(
        raw_dir / "credit_ratings" / "historical_rating_changes.csv",
        parse_dates=["date"],
    )
    if not df_hist.empty:
        df_hist["date"] = (
            pd.to_datetime(df_hist["date"], errors="coerce")
            .dt.to_period("M")
            .dt.to_timestamp()
        )
    return df_current, df_hist


def load_macro(raw_dir):
    raw_dir = Path(raw_dir)
    df = safe_read_csv(raw_dir / "macro" / "macro_final.csv")
    if not df.empty:
        df = df.rename(columns={
            "debt_to_gdp":             "debt_gdp",
            "reserves_months_imports": "reserves_months",
            "current_account_pct_gdp": "current_account_gdp",
        })
    return df


def load_fx(raw_dir):
    raw_dir = Path(raw_dir)
    df = safe_read_csv(raw_dir / "fx" / "fred_fx_rates_monthly.csv", parse_dates=["date"])
    if df.empty:
        return df
    df["date"] = (
        pd.to_datetime(df["date"], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    if "fx_monthly_pct_change" in df.columns:
        df["delta_fx"] = df["fx_monthly_pct_change"]
    elif "fx_rate" in df.columns:
        df = df.sort_values(["country", "date"])
        df["delta_fx"] = df.groupby("country")["fx_rate"].pct_change()
    return df


def load_yields(raw_dir):
    raw_dir = Path(raw_dir)
    df = safe_read_csv(
        raw_dir / "yields" / "bond_yields_10y_monthly.csv",
        parse_dates=["date"],
    )
    if df.empty:
        return df
    df["date"] = (
        pd.to_datetime(df["date"], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    if "yield_monthly_change" in df.columns:
        df["delta_bond"] = df["yield_monthly_change"]
    elif "yield_10y" in df.columns:
        df = df.sort_values(["country", "date"])
        df["delta_bond"] = df.groupby("country")["yield_10y"].diff()
    return df


def load_gdelt(raw_dir):
    raw_dir = Path(raw_dir)
    df = safe_read_csv(
        raw_dir / "gdelt" / "gdelt_country_tone_monthly.csv",
        parse_dates=["date"],
    )
    if df.empty:
        return df
    df["date"] = (
        pd.to_datetime(df["date"], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    if "gdelt_avg_tone" in df.columns:
        df = df.rename(columns={"gdelt_avg_tone": "S_MKT"})
    elif "tone" in df.columns:
        df = df.rename(columns={"tone": "S_MKT"})
    keep = [c for c in ["country", "date", "S_MKT"] if c in df.columns]
    return df[keep].dropna(subset=["country", "date"])


def load_cb_sentiment(raw_dir):
    raw_dir = Path(raw_dir)
    df = safe_read_csv(raw_dir / "central_bank_texts" / "cb_sentiment_scores.csv")
    if df.empty:
        return pd.DataFrame(columns=["country", "S_CB"])
    if "polarity" in df.columns:
        df = df.rename(columns={"polarity": "S_CB"})
    return df.groupby("country", as_index=False)["S_CB"].mean()
