"""Experiment B: S&P monthly labels and feature subsets."""

import numpy as np
import pandas as pd

from src.annual_experiment import fit_ordered_logit, fit_xgboost_class
from src.features import FEATURE_COLS
from src.monthly_experiment import NO_SENTIMENT_COLS, SENTIMENT_COLS, with_sp_labels
from src.train import get_class_weights


def monthly_features(months):
    frame = pd.DataFrame({
        "country": "Ghana",
        "date": pd.date_range("2022-01-01", periods=months, freq="MS"),
        "region": "Africa",
        "future_class": 1,
    })
    for column in FEATURE_COLS:
        frame[column] = 0.0
    return frame


def test_sp_labels_use_the_rating_in_force_at_each_month_end():
    history = pd.DataFrame({
        "country": "Ghana",
        "date": pd.to_datetime(["2021-01-15", "2022-12-20"]),
        "rating": ["B-", "SD"],
    })

    relabelled = with_sp_labels(monthly_features(12), history)

    november = relabelled[relabelled["date"] == "2022-11-01"].iloc[0]
    assert november["future_class"] == 0  # December ends in default
    assert november["cls_lag1"] == 1


def test_months_before_the_first_rating_are_dropped():
    history = pd.DataFrame({"country": "Ghana", "date": pd.to_datetime(["2022-06-10"]),
                            "rating": ["B"]})

    relabelled = with_sp_labels(monthly_features(12), history)

    assert relabelled["date"].min() == pd.Timestamp("2022-09-01")  # needs three months of lags


def test_no_sentiment_drops_exactly_the_sentiment_features():
    assert set(SENTIMENT_COLS) == {c for c in FEATURE_COLS if c.startswith(("S_CB", "S_MKT"))}
    assert set(NO_SENTIMENT_COLS) | set(SENTIMENT_COLS) == set(FEATURE_COLS)
    assert not set(NO_SENTIMENT_COLS) & set(SENTIMENT_COLS)


def test_class_weights_cover_a_class_missing_from_training():
    weights = get_class_weights(np.array([1, 1, 2, 2, 2]))

    assert weights[0] == 1.0
    assert weights[1] > weights[2]  # the rarer class weighs more


def test_ordered_logit_maps_two_trained_classes_back_to_their_labels():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 1))
    y = np.where(X[:, 0] > 0, 2, 1)  # no default in training

    predicted, proba = fit_ordered_logit(X, y, np.array([[3.0], [-3.0]]))

    assert predicted.tolist() == [2, 1]
    assert proba.shape == (2, 3) and np.allclose(proba[:, 0], 0)


def test_xgboost_trained_without_defaults_predicts_original_labels():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 2))
    y = np.where(X[:, 0] > 0, 2, 1)
    X_val, y_val = X[:30], np.r_[y[:29], 0]  # a default in validation, unseen in training

    predicted, _ = fit_xgboost_class(X, y, X_val, y_val, np.array([[3.0, 0.0], [-3.0, 0.0]]))

    assert predicted.tolist() == [2, 1]
