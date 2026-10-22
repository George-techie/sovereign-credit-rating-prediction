"""
annual_panel.py — The 80-country yearly panel for Experiment A.

One row per country and year t: fundamentals known at the end of year t, the
S&P rating and outlook in force on 31 December of t, and the targets, which
are the rating at the end of year t+1:

    next_class       Default / Junk / Investment Grade (0 / 1 / 2)
    next_direction   S&P notch change from t to t+1: -1 down, 0 none, +1 up

Sources (all joined on ISO3 through src/countries.py):
    data/raw/macro/worldbank_macro.csv          World Bank WDI indicators
    data/raw/macro/imf_weo_fiscal.csv           IMF WEO debt and fiscal balance
    data/raw/credit_ratings/sp_ratings_annual.csv  S&P year-end ratings

Missing values are filled only from the same country's earlier years (forward
fill), never from later ones, so no feature carries information from after
year t. Remaining gaps are left for the experiment to impute with training
medians.
"""

import numpy as np
import pandas as pd

from .countries import REGION

FUNDAMENTALS = [
    "gdp_growth", "inflation_slog", "current_account_pct_gdp", "reserves_months_imports",
    "unemployment", "log_gdp_per_capita", "trade_pct_gdp", "fdi_pct_gdp",
    "debt_to_gdp", "fiscal_balance", "debt_change_1y", "gdp_growth_3y", "fiscal_balance_3y",
]
RATING_STATE = ["notch", "outlook_score"]
OUTLOOK_SCORE = {"Negative": -1, "Watch Neg": -1, "Stable": 0, "Developing": 0,
                 "Positive": 1, "Watch Pos": 1}


def signed_log(values):
    """log(1 + |x|) with x's sign: keeps hyperinflation (Venezuela, Zimbabwe) on a usable scale."""
    return np.sign(values) * np.log1p(np.abs(values))


def build_annual_panel(worldbank, imf_fiscal, sp_annual):
    """Join the three sources and add derived features and next-year targets."""
    panel = (
        worldbank.merge(
            imf_fiscal[["country", "year", "imf_debt_to_gdp", "imf_fiscal_balance",
                        "imf_is_estimate"]],
            on=["country", "year"], how="left",
        )
        .merge(sp_annual[["country", "year", "rating", "outlook", "notch", "rating_class"]],
               on=["country", "year"], how="left")
        .sort_values(["country", "year"])
        .reset_index(drop=True)
    )
    # IMF debt is far more complete than WDI's central-government debt; use WDI
    # only where the IMF has no figure.
    panel["debt_to_gdp"] = panel["imf_debt_to_gdp"].fillna(panel["debt_to_gdp"])
    panel["fiscal_balance"] = panel["imf_fiscal_balance"]
    panel["inflation_slog"] = signed_log(panel["inflation"])
    panel["log_gdp_per_capita"] = np.log(panel["gdp_per_capita"])
    panel["outlook_score"] = panel["outlook"].map(OUTLOOK_SCORE).fillna(0)
    panel["region"] = panel["country"].map(REGION)

    by_country = panel.groupby("country")
    raw = ["gdp_growth", "inflation_slog", "current_account_pct_gdp", "reserves_months_imports",
           "unemployment", "log_gdp_per_capita", "trade_pct_gdp", "fdi_pct_gdp",
           "debt_to_gdp", "fiscal_balance"]
    panel[raw] = by_country[raw].ffill()
    panel["debt_change_1y"] = by_country["debt_to_gdp"].diff()
    panel["gdp_growth_3y"] = by_country["gdp_growth"].transform(
        lambda s: s.rolling(3, min_periods=1).mean())
    panel["fiscal_balance_3y"] = by_country["fiscal_balance"].transform(
        lambda s: s.rolling(3, min_periods=1).mean())

    panel["next_class"] = by_country["rating_class"].shift(-1)
    panel["next_notch"] = by_country["notch"].shift(-1)
    panel["next_direction"] = np.sign(panel["next_notch"] - panel["notch"])
    return panel


def modelling_rows(panel):
    """Rows usable for training and testing: rated at t and at t+1."""
    usable = panel.dropna(subset=["rating_class", "next_class"]).copy()
    for column in ["rating_class", "next_class", "next_direction"]:
        usable[column] = usable[column].astype(int)
    usable["target_year"] = usable["year"] + 1
    return usable.reset_index(drop=True)
