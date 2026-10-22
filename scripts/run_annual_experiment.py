"""
run_annual_experiment.py
------------------------
Experiment A: next-year S&P rating for the 80-country yearly panel.
Writes everything to results/annual/. Run from the repository root:
    python scripts/run_annual_experiment.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.annual_experiment import (  # noqa: E402
    FEATURE_SETS, class_metrics_on_changes, country_sequences, direction_metrics,
    fit_direction_xgboost, fit_lstm, fit_ordered_logit, fit_xgboost_class, metrics_table,
    notch_residuals_by_region, persistence, prepare_matrices, regional_accuracy,
    split_by_target_year,
)
from src.annual_panel import build_annual_panel, modelling_rows  # noqa: E402

RAW = Path("data/raw")
OUT = Path("results/annual")


def load_panel():
    return build_annual_panel(
        pd.read_csv(RAW / "macro/worldbank_macro.csv"),
        pd.read_csv(RAW / "macro/imf_weo_fiscal.csv"),
        pd.read_csv(RAW / "credit_ratings/sp_ratings_annual.csv"),
    )


def run_feature_set(panel, train, val, test, name, features):
    X_train, X_val, X_test, medians, scaler = prepare_matrices(train, val, test, features)
    y_train, y_val = train["next_class"].to_numpy(), val["next_class"].to_numpy()
    predictions = {}
    predictions[f"Ordered logit [{name}]"], _ = fit_ordered_logit(X_train, y_train, X_test)
    predictions[f"XGBoost [{name}]"], _ = fit_xgboost_class(X_train, y_train, X_val, y_val, X_test)
    sequences = [country_sequences(panel, part, features, medians, scaler)
                 for part in (train, val, test)]
    predictions[f"LSTM [{name}]"] = fit_lstm(sequences[0], y_train, sequences[1], y_val,
                                             sequences[2])
    direction = fit_direction_xgboost(X_train, train["next_direction"].to_numpy(), X_val,
                                      val["next_direction"].to_numpy(), X_test)
    return predictions, direction


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    rows = modelling_rows(panel)
    train, val, test = split_by_target_year(rows)

    predictions = {"Persistence (same class)": persistence(test)}
    direction_rows = [direction_metrics(test["next_direction"], np.zeros(len(test)),
                                        "Persistence (no change)")]
    for name, features in FEATURE_SETS.items():
        class_predictions, direction = run_feature_set(panel, train, val, test, name, features)
        predictions.update(class_predictions)
        direction_rows.append(direction_metrics(test["next_direction"], direction,
                                                f"XGBoost direction [{name}]"))

    metrics = metrics_table(test, predictions)
    changes = class_metrics_on_changes(test, predictions)
    regional = regional_accuracy(test, predictions)
    residuals, scored = notch_residuals_by_region(train, test,
                                                  train[FEATURE_SETS["fundamentals"]].median())
    metrics = metrics.merge(changes, on="Model")

    metrics.to_csv(OUT / "class_metrics.csv", index=False)
    pd.DataFrame(direction_rows).to_csv(OUT / "direction_metrics.csv", index=False)
    regional.to_csv(OUT / "regional_accuracy.csv", index=False)
    residuals.to_csv(OUT / "notch_residuals_by_region.csv", index=False)
    test.assign(**{name: np.asarray(p) for name, p in predictions.items()}).to_csv(
        OUT / "test_predictions.csv", index=False)
    scored[["country", "region", "year", "next_notch", "predicted_notch", "residual"]].to_csv(
        OUT / "notch_residuals_test.csv", index=False)
    summary = {
        "countries": int(rows["country"].nunique()),
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "target_years": {"train": "2012-2017", "val": "2018-2019", "test": "2020-2025"},
        "test_class_counts": test["next_class"].value_counts().sort_index().to_dict(),
        "test_class_changes": int((test["next_class"] != test["rating_class"]).sum()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=int))

    pd.set_option("display.width", 200)
    print(json.dumps(summary, indent=2, default=int))
    print("\nNext-year class\n", metrics.to_string(index=False))
    print("\nDowngrade / no change / upgrade\n", pd.DataFrame(direction_rows).to_string(index=False))
    print("\nAccuracy by region\n", regional.to_string(index=False))
    print("\nActual minus fundamentals-implied notch, by region (test years)\n",
          residuals.to_string(index=False))


if __name__ == "__main__":
    main()
