"""
features.py — Feature engineering for sovereign credit rating prediction.
"""

import numpy as np
import pandas as pd

AFRICA = [
    "South Africa", "Kenya", "Ghana", "Egypt", "Nigeria",
    "Ethiopia", "Botswana", "Morocco", "Zambia",
]

INVESTMENT_GRADE = [
    "AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-",
    "Aaa", "Aa1", "Aa2", "Aa3", "A1", "A2", "A3", "Baa1", "Baa2", "Baa3",
]
JUNK = [
    "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C",
    "Ba1", "Ba2", "Ba3", "B1", "B2", "B3", "Caa1", "Caa2", "Caa3", "Ca",
]
DEFAULT = ["SD", "D", "RD", "C"]

FEATURE_COLS = [
    "S_CB", "S_MKT", "S_CB_3m", "S_MKT_3m",
    "delta_bond", "delta_fx", "delta_bond_3m", "delta_fx_3m",
    "yield_10y", "inflation", "gdp_growth", "debt_gdp", "reserves_months",
    "S_CB_lag1", "S_CB_lag2", "S_CB_lag3",
    "S_MKT_lag1", "S_MKT_lag2", "S_MKT_lag3",
    "yield_lag1", "yield_lag2", "yield_lag3",
    "cls_lag1", "cls_lag2", "cls_lag3",
    "month_sin", "month_cos",
]


def rating_to_class(rating):
    if pd.isna(rating):
        return np.nan
    r = str(rating).strip()
    if r in DEFAULT:
        return 0
    if r in JUNK:
        return 1
    if r in INVESTMENT_GRADE:
        return 2
    return np.nan


def assign_region(country):
    return "Africa" if country in AFRICA else "Benchmark"


def expand_annual_to_monthly(df, year_col="year", value_cols=None):
    if value_cols is None:
        value_cols = [
            c for c in df.columns
            if c not in ["country", year_col, "region"]
        ]
    rows = []
    for _, row in df.iterrows():
        yr = int(row[year_col])
        for month in range(1, 13):
            rec = {
                "country": row["country"],
                "date": pd.Timestamp(year=yr, month=month, day=1),
            }
            for col in value_cols:
                rec[col] = row[col]
            rows.append(rec)
    return (
        pd.DataFrame(rows)
        .sort_values(["country", "date"])
        .reset_index(drop=True)
    )


def add_rolling_features(df, cols, window=3):
    df = df.copy()
    for col in cols:
        df[f"{col}_{window}m"] = (
            df.groupby("country")[col]
            .transform(lambda s: s.rolling(window, min_periods=1).mean())
        )
    return df


def add_lag_features(df, col_lag_map):
    df = df.copy()
    for col, lags in col_lag_map.items():
        for lag in lags:
            df[f"{col}_lag{lag}"] = df.groupby("country")[col].shift(lag)
    return df


def add_cyclic_month(df):
    df = df.copy()
    month = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    return df


def build_target(df):
    df = df.copy().sort_values(["country", "date"])
    df["future_class"] = df.groupby("country")["current_class"].shift(-1)
    df["rating_change"] = np.sign(
        df["future_class"] - df["current_class"]
    ).fillna(0).astype(int)
    return df


def fill_missing(df, cols, group_col="country"):
    df = df.copy()
    for col in cols:
        df[col] = (
            df.groupby(group_col)[col]
            .transform(lambda s: s.ffill().bfill())
        )
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def build_feature_matrix(df):
    df = df.copy().sort_values(["country", "date"]).reset_index(drop=True)

    df["region"] = df["country"].apply(assign_region)
    df = build_target(df)
    df = add_rolling_features(df, ["S_CB", "S_MKT", "delta_bond", "delta_fx"], window=3)
    df = add_lag_features(df, {
        "S_CB":          [1, 2, 3],
        "S_MKT":         [1, 2, 3],
        "yield_10y":     [1, 2, 3],
        "current_class": [1, 2, 3],
    })
    for lag in [1, 2, 3]:
        df = df.rename(columns={f"current_class_lag{lag}": f"cls_lag{lag}"})

    df = add_cyclic_month(df)
    df = fill_missing(df, FEATURE_COLS)
    df = df[df["future_class"].notna()].copy()
    df["future_class"] = df["future_class"].round().astype(int)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = fill_missing(df, FEATURE_COLS)

    return df[["country", "date", "region", "future_class"] + FEATURE_COLS].copy()
