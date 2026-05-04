"""
tests/test_metrics.py — Unit tests for src/metrics.py
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import (
    ordinal_mae,
    compute_classification_metrics,
    regional_metrics,
    direction_accuracy,
    normalize_proba,
)


class TestOrdinalMAE:
    def test_perfect_predictions(self):
        y = [0, 1, 2, 1, 0]
        assert ordinal_mae(y, y) == 0.0

    def test_off_by_one(self):
        assert ordinal_mae([0, 1, 2], [1, 2, 1]) == pytest.approx(1.0)

    def test_worst_case(self):
        assert ordinal_mae([0, 2], [2, 0]) == pytest.approx(2.0)

    def test_single_sample(self):
        assert ordinal_mae([1], [1]) == 0.0
        assert ordinal_mae([0], [2]) == pytest.approx(2.0)

    def test_returns_float(self):
        assert isinstance(ordinal_mae([0, 1], [1, 2]), float)


class TestComputeClassificationMetrics:
    def test_perfect_accuracy(self):
        y = [0, 1, 2, 0, 1, 2]
        result = compute_classification_metrics(y, y, model_name="test")
        assert result["Accuracy"] == 1.0
        assert result["MAE"] == 0.0

    def test_model_name_in_result(self):
        y = [0, 1, 2]
        assert compute_classification_metrics(y, y, model_name="XGBoost")["Model"] == "XGBoost"

    def test_all_keys_present_without_proba(self):
        y = [0, 1, 2]
        result = compute_classification_metrics(y, y)
        expected = {"Model","Accuracy","MAE","F1 (Default)","F1 (Junk)","F1 (Inv.Grade)","Macro F1","Weighted F1"}
        assert expected.issubset(result.keys())
        assert "AUC-OvR" not in result

    def test_auc_computed_with_proba(self):
        y = [0, 1, 2, 0, 1, 2]
        proba = np.eye(3)[[0, 1, 2, 0, 1, 2]]
        result = compute_classification_metrics(y, y, y_proba=proba)
        assert result["AUC-OvR"] == pytest.approx(1.0)

    def test_invalid_proba_shape_raises(self):
        y = [0, 1, 2]
        with pytest.raises(ValueError):
            compute_classification_metrics(y, y, y_proba=np.ones((3, 2)))

    def test_missing_class_in_predictions(self):
        result = compute_classification_metrics([0,1,2,1,2], [1,1,2,1,2])
        assert result["F1 (Default)"] == pytest.approx(0.0)


class TestRegionalMetrics:
    def _make_df(self):
        return pd.DataFrame({
            "future_class": [2, 2, 1, 0, 2, 1],
            "pred":         [2, 2, 1, 1, 2, 1],
            "region":       ["Africa","Africa","Africa","Benchmark","Benchmark","Benchmark"],
        })

    def test_returns_three_values(self):
        assert len(regional_metrics(self._make_df(), "pred")) == 3

    def test_reg_df_has_two_rows(self):
        reg_df, _, _ = regional_metrics(self._make_df(), "pred")
        assert set(reg_df["Region"]) == {"Africa", "Benchmark"}

    def test_bias_gap_positive_when_africa_worse(self):
        df = pd.DataFrame({
            "future_class": [0,1,2,0,1,2],
            "pred":         [1,2,1,0,1,2],
            "region":       ["Africa","Africa","Africa","Benchmark","Benchmark","Benchmark"],
        })
        _, gap_mae, gap_acc = regional_metrics(df, "pred")
        assert gap_mae > 0
        assert gap_acc > 0

    def test_zero_bias_when_equal(self):
        df = pd.DataFrame({
            "future_class": [0,1,2,0,1,2],
            "pred":         [0,1,2,0,1,2],
            "region":       ["Africa","Africa","Africa","Benchmark","Benchmark","Benchmark"],
        })
        _, gap_mae, gap_acc = regional_metrics(df, "pred")
        assert gap_mae == pytest.approx(0.0)
        assert gap_acc == pytest.approx(0.0)


class TestDirectionAccuracy:
    def test_perfect_direction(self):
        acc, true_dir, pred_dir = direction_accuracy([2,1,0],[2,1,0],[1,1,1])
        assert acc == pytest.approx(1.0)
        np.testing.assert_array_equal(true_dir, [1,0,-1])

    def test_all_wrong(self):
        acc, _, _ = direction_accuracy([2,0],[0,2],[1,1])
        assert acc == pytest.approx(0.0)

    def test_returns_float(self):
        assert isinstance(direction_accuracy([1],[1],[1])[0], float)


class TestNormalizeProba:
    def test_already_normalized(self):
        proba = np.array([[0.2,0.5,0.3],[0.1,0.1,0.8]])
        np.testing.assert_allclose(normalize_proba(proba).sum(axis=1), [1.0,1.0])

    def test_nan_replaced_with_uniform(self):
        proba = np.array([[np.nan,np.nan,np.nan]])
        np.testing.assert_allclose(normalize_proba(proba), [[1/3,1/3,1/3]], atol=1e-6)

    def test_rows_sum_to_one(self):
        proba = np.array([[3.0,3.0,3.0],[1.0,0.0,0.0]])
        np.testing.assert_allclose(normalize_proba(proba).sum(axis=1), [1.0,1.0], atol=1e-6)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            normalize_proba(np.ones((4,2)))

    def test_output_shape_preserved(self):
        proba = np.random.dirichlet([1,1,1], size=10)
        assert normalize_proba(proba).shape == (10,3)
