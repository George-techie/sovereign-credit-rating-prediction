"""
monthly_experiment.py — Experiment B: the 17-country monthly model, re-examined.

The original monthly model predicts next month's consensus class (S&P, Moody's
and Fitch) from sentiment, market and macro features. Ratings change rarely,
so it is compared here with persistence, and two variants isolate what drives
it:

    as is          the original features and consensus labels
    no sentiment   without the central-bank and market sentiment features
    S&P labels     the same features, labels from the S&P history (as Experiment A)
"""

import numpy as np
import pandas as pd
import torch

from .features import FEATURE_COLS
from .ratings import sp_rating_class

SENTIMENT_COLS = [c for c in FEATURE_COLS if c.startswith(("S_CB", "S_MKT"))]
NO_SENTIMENT_COLS = [c for c in FEATURE_COLS if c not in SENTIMENT_COLS]
ORDERED_FEATURES = ["S_CB_3m", "S_MKT_3m", "delta_bond_3m", "delta_fx_3m",
                    "yield_10y", "gdp_growth", "debt_gdp", "reserves_months"]


def month_end_classes(history, countries, months):
    """S&P class in force at the end of each month, per country (NaN before the first rating)."""
    records = []
    for country in countries:
        events = history[history["country"] == country].sort_values("date")
        for month in months:
            month_end = month + pd.offsets.MonthEnd(0)
            in_force = events[events["date"] <= month_end]
            rating = in_force["rating"].iloc[-1] if not in_force.empty else None
            records.append({"country": country, "date": month, "sp_class": sp_rating_class(rating)})
    return pd.DataFrame(records)


def with_sp_labels(features, history):
    """Replace the consensus target and class lags with S&P's; unrated months are dropped.

    future_class is the class at the end of the next month and cls_lag k the
    class k months back, the same definitions the original pipeline uses.
    """
    months = sorted(features["date"].unique())
    first, last = months[0] - pd.DateOffset(months=3), months[-1] + pd.DateOffset(months=1)
    all_months = list(pd.date_range(first, last, freq="MS"))
    classes = month_end_classes(history, features["country"].unique(), all_months)
    classes = classes.sort_values(["country", "date"])
    by_country = classes.groupby("country")["sp_class"]
    classes["future_class"] = by_country.shift(-1)
    for lag in (1, 2, 3):
        classes[f"cls_lag{lag}"] = by_country.shift(lag)
    relabelled = features.drop(columns=["future_class", "cls_lag1", "cls_lag2", "cls_lag3"]).merge(
        classes.drop(columns="sp_class"), on=["country", "date"], how="inner")
    relabelled = relabelled.dropna(subset=["future_class", "cls_lag1", "cls_lag2", "cls_lag3"])
    relabelled["future_class"] = relabelled["future_class"].astype(int)
    return relabelled.reset_index(drop=True)


def persistence_monthly(test):
    """Next month = last month's class (cls_lag1, the most recent class among the features)."""
    return test["cls_lag1"].round().astype(int).to_numpy()


def lstm_test_predictions(model, test, feature_cols, scaler, seq_len=12):
    """LSTM predictions with the rows they belong to (each country loses its first seq_len months)."""
    rows, sequences = [], []
    for _, group in test.sort_values("date").groupby("country"):
        X = scaler.transform(group[feature_cols].values)
        for i in range(seq_len, len(group)):
            sequences.append(X[i - seq_len:i])
            rows.append(group.index[i])
    if not sequences:
        return pd.Series(dtype=int)
    with torch.no_grad():
        logits = model(torch.tensor(np.asarray(sequences, dtype=np.float32)))
    return pd.Series(logits.argmax(dim=1).numpy(), index=rows)
