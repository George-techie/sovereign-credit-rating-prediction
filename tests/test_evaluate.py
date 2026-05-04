"""
tests/test_evaluate.py — Unit tests for src/evaluate.py
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import load_predictions, run_full_evaluation


class TestLoadPredictions:
    def test_raises_if_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_predictions(tmp_path)

    def test_loads_correctly(self, tmp_path):
        df = pd.DataFrame({
            "date":         pd.date_range("2022-01-01", periods=3, freq="MS"),
            "country":      ["Kenya"]*3,
            "region":       ["Africa"]*3,
            "future_class": [1,2,1],
            "pred_xgb":     [1,2,2],
            "pred_lstm":    [1,1,1],
            "pred_ord":     [1,2,1],
        })
        df.to_csv(tmp_path / "test_predictions.csv", index=False)
        loaded = load_predictions(tmp_path)
        assert len(loaded) == 3
        assert "pred_xgb" in loaded.columns


class TestRunFullEvaluation:
    def _write(self, tmp_path):
        np.random.seed(42)
        n = 30
        df = pd.DataFrame({
            "date":         pd.date_range("2022-01-01", periods=n, freq="MS"),
            "country":      (["Kenya"]*15)+( ["Germany"]*15),
            "region":       (["Africa"]*15)+(["Benchmark"]*15),
            "future_class": np.random.choice([0,1,2], n),
            "pred_ord":     np.random.choice([0,1,2], n),
            "pred_xgb":     np.random.choice([0,1,2], n),
            "pred_lstm":    np.random.choice([0,1,2], n),
            "cls_lag1":     np.random.choice([0,1,2], n),
            "prob_default_xgb":  np.random.dirichlet([1,1,1],n)[:,0],
            "prob_junk_xgb":     np.random.dirichlet([1,1,1],n)[:,1],
            "prob_invgrade_xgb": np.random.dirichlet([1,1,1],n)[:,2],
            "prob_default_lstm":  np.random.dirichlet([1,1,1],n)[:,0],
            "prob_junk_lstm":     np.random.dirichlet([1,1,1],n)[:,1],
            "prob_invgrade_lstm": np.random.dirichlet([1,1,1],n)[:,2],
        })
        df.to_csv(tmp_path / "test_predictions.csv", index=False)

    def test_returns_three_keys(self, tmp_path):
        self._write(tmp_path)
        assert set(run_full_evaluation(tmp_path, save=False).keys()) == {"metrics","bias","direction"}

    def test_metrics_has_three_models(self, tmp_path):
        self._write(tmp_path)
        assert len(run_full_evaluation(tmp_path, save=False)["metrics"]) == 3

    def test_accuracy_between_0_and_1(self, tmp_path):
        self._write(tmp_path)
        for acc in run_full_evaluation(tmp_path, save=False)["metrics"]["Accuracy"]:
            assert 0.0 <= acc <= 1.0

    def test_saves_csvs(self, tmp_path):
        self._write(tmp_path)
        run_full_evaluation(tmp_path, save=True)
        assert (tmp_path / "full_metrics_table.csv").exists()
        assert (tmp_path / "bias_analysis.csv").exists()

    def test_direction_computed(self, tmp_path):
        self._write(tmp_path)
        result = run_full_evaluation(tmp_path, save=False)
        assert len(result["direction"]) == 3
