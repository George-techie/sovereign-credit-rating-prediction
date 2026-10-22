"""
run_monthly_experiment.py
-------------------------
Experiment B: the 17-country monthly model (2018-2024) as is, without sentiment,
and with S&P labels, each against persistence. Uses the project's own training
functions (src/train.py) and split (train < 2020, validation 2020-2021, test
2022-2024). Writes results/monthly/. Run from the repository root:
    python scripts/run_monthly_experiment.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.annual_experiment import fit_ordered_logit, fit_xgboost_class  # noqa: E402
from src.features import FEATURE_COLS  # noqa: E402
from src.metrics import compute_classification_metrics  # noqa: E402
from src.monthly_experiment import (  # noqa: E402
    NO_SENTIMENT_COLS, ORDERED_FEATURES, lstm_test_predictions, persistence_monthly,
    with_sp_labels,
)
from src.train import (  # noqa: E402
    fit_scaler, get_class_weights, time_split, train_lstm,
)

OUT = Path("results/monthly")
SEED = 42


def evaluate_variant(name, frame, features):
    import torch
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train, val, test = time_split(frame)
    scaler = fit_scaler(train[features].values)
    X = {part: scaler.transform(df[features].values)
         for part, df in (("train", train), ("val", val), ("test", test))}
    y_train, y_val = train["future_class"].to_numpy(), val["future_class"].to_numpy()
    rows = [compute_classification_metrics(test["future_class"], persistence_monthly(test),
                                           model_name=f"Persistence [{name}]")]

    xgb_pred, _ = fit_xgboost_class(X["train"], y_train, X["val"], y_val, X["test"])
    rows.append(compute_classification_metrics(test["future_class"], xgb_pred,
                                               model_name=f"XGBoost [{name}]"))

    ordered = [c for c in ORDERED_FEATURES if c in features]
    ord_scaler = fit_scaler(train[ordered].values)
    ord_pred, _ = fit_ordered_logit(ord_scaler.transform(train[ordered].values), y_train,
                                    ord_scaler.transform(test[ordered].values))
    rows.append(compute_classification_metrics(test["future_class"], ord_pred,
                                               model_name=f"Ordered logit [{name}]"))

    lstm = train_lstm(train, val, scaler, get_class_weights(y_train), feature_cols=features)
    lstm_pred = lstm_test_predictions(lstm, test, features, scaler)
    lstm_row = compute_classification_metrics(test.loc[lstm_pred.index, "future_class"],
                                              lstm_pred.to_numpy(), model_name=f"LSTM [{name}]")
    lstm_row["Rows"] = len(lstm_pred)
    rows.append(lstm_row)
    for row in rows:
        row.setdefault("Rows", len(test))
    changed = int((test["future_class"] != persistence_monthly(test)).sum())
    return rows, {"rows": len(frame), "test_rows": len(test), "test_class_changes": changed,
                  "countries": int(frame["country"].nunique())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv("data/processed/model_ready_features.csv", parse_dates=["date"])
    history = pd.read_csv("data/raw/credit_ratings/sp_rating_history.csv", parse_dates=["date"])
    variants = {
        "as is": (features, FEATURE_COLS),
        "no sentiment": (features, NO_SENTIMENT_COLS),
        "S&P labels": (with_sp_labels(features, history), FEATURE_COLS),
    }
    all_rows, summary = [], {}
    for name, (frame, columns) in variants.items():
        rows, info = evaluate_variant(name, frame, columns)
        all_rows += rows
        summary[name] = info
    table = pd.DataFrame(all_rows)
    table.to_csv(OUT / "class_metrics.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 200)
    print(json.dumps(summary, indent=2))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
