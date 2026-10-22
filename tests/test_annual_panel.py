"""The 80-country yearly panel: joins, targets, and no information from the future."""

import numpy as np
import pandas as pd
import pytest

from src.annual_experiment import split_by_target_year
from src.annual_panel import build_annual_panel, modelling_rows, signed_log
from src.countries import COUNTRIES


def sources(debt_2021=60.0):
    years = [2019, 2020, 2021]
    worldbank = pd.DataFrame({
        "country": ["Kenya"] * 3, "year": years,
        "gdp_growth": [5.0, -0.3, 7.6], "inflation": [5.2, 5.4, np.nan],
        "debt_to_gdp": [np.nan] * 3, "current_account_pct_gdp": [-5.0, -4.7, -5.2],
        "reserves_months_imports": [5.5, 5.0, 4.8], "unemployment": [5.0, 5.6, 5.7],
        "gdp_per_capita": [1900.0, 1850.0, 1950.0], "population": [5e7] * 3,
        "trade_pct_gdp": [33.0, 29.0, 30.0], "fdi_pct_gdp": [1.0, 0.4, 0.4],
    })
    imf = pd.DataFrame({
        "country": ["Kenya"] * 3, "year": years,
        "imf_debt_to_gdp": [57.0, 66.0, debt_2021], "imf_fiscal_balance": [-7.4, -8.1, -7.2],
        "imf_is_estimate": [False] * 3,
    })
    sp = pd.DataFrame({
        "country": ["Kenya"] * 3, "year": years, "rating": ["B+", "B+", "B"],
        "outlook": ["Stable", "Negative", "Stable"], "notch": [8, 8, 7],
        "rating_class": [1, 1, 1],
    })
    return worldbank, imf, sp


def test_targets_are_next_years_rating():
    panel = build_annual_panel(*sources())

    assert panel["next_notch"].tolist()[:2] == [8, 7]
    assert panel["next_direction"].tolist()[:2] == [0, -1]


def test_last_year_has_no_target_and_is_not_modelled():
    rows = modelling_rows(build_annual_panel(*sources()))

    assert rows["year"].tolist() == [2019, 2020]
    assert rows["target_year"].tolist() == [2020, 2021]


def test_missing_values_are_filled_from_the_past_only():
    panel = build_annual_panel(*sources())

    assert panel.loc[2, "inflation_slog"] == pytest.approx(signed_log(5.4))  # 2020 carried forward


def test_changing_a_later_year_never_changes_an_earlier_row():
    early = build_annual_panel(*sources(debt_2021=60.0))
    late = build_annual_panel(*sources(debt_2021=95.0))
    features = ["debt_to_gdp", "debt_change_1y", "gdp_growth_3y", "fiscal_balance_3y"]

    pd.testing.assert_frame_equal(early.loc[:1, features], late.loc[:1, features])


def test_imf_debt_is_preferred_over_world_bank_debt():
    panel = build_annual_panel(*sources())

    assert panel.loc[0, "debt_to_gdp"] == 57.0


def test_outlook_is_scored_and_region_assigned():
    panel = build_annual_panel(*sources())

    assert panel["outlook_score"].tolist() == [0, -1, 0]
    assert set(panel["region"]) == {"Africa"}


def test_signed_log_keeps_the_sign_and_tames_hyperinflation():
    assert signed_log(-1.0) == -signed_log(1.0)
    assert signed_log(65000.0) < 12


def test_split_keeps_test_years_unseen():
    rows = pd.DataFrame({"target_year": range(2010, 2026)})

    train, val, test = split_by_target_year(rows)

    assert train["target_year"].between(2012, 2017).all()
    assert val["target_year"].tolist() == [2018, 2019]
    assert test["target_year"].min() == 2020


def test_country_table_covers_80_countries_with_unique_codes():
    codes = [code for code, _ in COUNTRIES.values()]

    assert len(COUNTRIES) == 80 and len(set(codes)) == 80
