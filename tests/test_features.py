"""
tests/test_features.py — Unit tests for src/features.py
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import (
    rating_to_class, assign_region, expand_annual_to_monthly,
    add_rolling_features, add_lag_features, add_cyclic_month,
    build_target, fill_missing, AFRICA,
)


class TestRatingToClass:
    def test_investment_grade(self):
        assert rating_to_class("AAA") == 2
        assert rating_to_class("BBB-") == 2

    def test_junk(self):
        assert rating_to_class("BB+") == 1
        assert rating_to_class("B") == 1

    def test_default(self):
        assert rating_to_class("SD") == 0
        assert rating_to_class("D") == 0

    def test_nan_input(self):
        assert np.isnan(rating_to_class(np.nan))

    def test_unrecognised(self):
        assert np.isnan(rating_to_class("XYZ"))

    def test_strips_whitespace(self):
        assert rating_to_class("  AAA  ") == 2


class TestAssignRegion:
    def test_african_countries(self):
        for c in AFRICA:
            assert assign_region(c) == "Africa"

    def test_benchmark(self):
        assert assign_region("Germany") == "Benchmark"


class TestExpandAnnualToMonthly:
    def _df(self):
        return pd.DataFrame({"country":["Kenya","Kenya"],"year":[2020,2021],"gdp_growth":[0.3,7.5]})

    def test_12_rows_per_year(self):
        assert len(expand_annual_to_monthly(self._df())) == 24

    def test_date_is_timestamp(self):
        result = expand_annual_to_monthly(self._df())
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_value_repeated(self):
        result = expand_annual_to_monthly(self._df())
        assert (result[result["date"].dt.year == 2020]["gdp_growth"] == 0.3).all()

    def test_sorted(self):
        result = expand_annual_to_monthly(self._df())
        assert result["date"].is_monotonic_increasing


class TestAddRollingFeatures:
    def _df(self):
        return pd.DataFrame({
            "country": ["Kenya"]*6,
            "date":    pd.date_range("2020-01-01", periods=6, freq="MS"),
            "S_CB":    [0.1,0.2,0.3,0.4,0.5,0.6],
        })

    def test_column_created(self):
        assert "S_CB_3m" in add_rolling_features(self._df(), ["S_CB"])

    def test_rolling_mean_correct(self):
        result = add_rolling_features(self._df(), ["S_CB"], window=3)
        assert result["S_CB_3m"].iloc[2] == pytest.approx(0.2, abs=1e-6)

    def test_does_not_modify_original(self):
        df = self._df()
        _ = add_rolling_features(df, ["S_CB"])
        assert "S_CB_3m" not in df.columns


class TestAddLagFeatures:
    def _df(self):
        return pd.DataFrame({
            "country": ["Kenya"]*5,
            "date":    pd.date_range("2020-01-01", periods=5, freq="MS"),
            "S_CB":    [0.1,0.2,0.3,0.4,0.5],
        })

    def test_lag_columns_created(self):
        result = add_lag_features(self._df(), {"S_CB":[1,2]})
        assert "S_CB_lag1" in result.columns
        assert "S_CB_lag2" in result.columns

    def test_lag1_correct(self):
        result = add_lag_features(self._df(), {"S_CB":[1]})
        assert np.isnan(result["S_CB_lag1"].iloc[0])
        assert result["S_CB_lag1"].iloc[1] == pytest.approx(0.1)

    def test_does_not_modify_original(self):
        df = self._df()
        _ = add_lag_features(df, {"S_CB":[1]})
        assert "S_CB_lag1" not in df.columns


class TestAddCyclicMonth:
    def _df(self):
        return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=12, freq="MS")})

    def test_columns_created(self):
        result = add_cyclic_month(self._df())
        assert "month_sin" in result.columns
        assert "month_cos" in result.columns

    def test_values_bounded(self):
        result = add_cyclic_month(self._df())
        assert result["month_sin"].between(-1,1).all()
        assert result["month_cos"].between(-1,1).all()

    def test_january_july_opposite(self):
        result = add_cyclic_month(self._df())
        jan = result[result["date"].dt.month==1]["month_sin"].values[0]
        jul = result[result["date"].dt.month==7]["month_sin"].values[0]
        assert abs(jan + jul) < 1e-6


class TestBuildTarget:
    def _df(self):
        return pd.DataFrame({
            "country":       ["Kenya"]*4,
            "date":          pd.date_range("2020-01-01", periods=4, freq="MS"),
            "current_class": [2,2,1,1],
        })

    def test_future_class_shifted(self):
        result = build_target(self._df())
        assert result["future_class"].iloc[0] == 2
        assert result["future_class"].iloc[1] == 1
        assert np.isnan(result["future_class"].iloc[3])

    def test_rating_change_sign(self):
        result = build_target(self._df())
        assert result["rating_change"].iloc[1] == -1

    def test_stable_is_zero(self):
        result = build_target(self._df())
        assert result["rating_change"].iloc[0] == 0


class TestFillMissing:
    def test_forward_fill(self):
        df = pd.DataFrame({"country":["Kenya"]*3,"S_CB":[0.5,np.nan,np.nan]})
        result = fill_missing(df, ["S_CB"])
        assert result["S_CB"].iloc[1] == pytest.approx(0.5)

    def test_median_fallback(self):
        df = pd.DataFrame({
            "country":["Kenya","Kenya","Ghana","Ghana"],
            "S_CB":[np.nan,np.nan,0.4,0.6],
        })
        assert not fill_missing(df, ["S_CB"])["S_CB"].isna().any()

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"country":["Kenya"],"S_CB":[np.nan]})
        _ = fill_missing(df, ["S_CB"])
        assert np.isnan(df["S_CB"].iloc[0])
