"""
metrics.py — Model evaluation metrics for sovereign credit rating prediction.

Three rating classes:
    0 = Default
    1 = Junk
    2 = Investment Grade

Ordinal MAE matters here because predicting Default when truth is Junk
is a smaller error than predicting Investment Grade when truth is Default.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)

CLASS_NAMES = ["Default", "Junk", "Inv. Grade"]
REGIONS = ["Africa", "Benchmark"]


def ordinal_mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_pred - y_true)))


def compute_classification_metrics(y_true, y_pred, y_proba=None, model_name=""):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = accuracy_score(y_true, y_pred)
    mae = ordinal_mae(y_true, y_pred)

    report = classification_report(
        y_true, y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    row = {
        "Model":          model_name,
        "Accuracy":       round(acc, 4),
        "MAE":            round(mae, 4),
        "F1 (Default)":   round(report["Default"]["f1-score"], 4),
        "F1 (Junk)":      round(report["Junk"]["f1-score"], 4),
        "F1 (Inv.Grade)": round(report["Inv. Grade"]["f1-score"], 4),
        "Macro F1":       round(report["macro avg"]["f1-score"], 4),
        "Weighted F1":    round(report["weighted avg"]["f1-score"], 4),
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba, dtype=float)
        if y_proba.ndim != 2 or y_proba.shape[1] != 3:
            raise ValueError(f"y_proba must have shape (n, 3), got {y_proba.shape}")
        try:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            row["AUC-OvR"] = round(auc, 4)
        except ValueError:
            row["AUC-OvR"] = None

    return row


def regional_metrics(df_preds, pred_col, true_col="future_class"):
    records = []
    for region in REGIONS:
        sub = df_preds[df_preds["region"] == region].dropna(subset=[pred_col])
        if sub.empty:
            records.append({"Region": region, "N": 0,
                            "Accuracy": None, "MAE": None, "F1 Macro": None})
            continue

        yt = sub[true_col].values.astype(int)
        yp = sub[pred_col].values.astype(int)

        report = classification_report(
            yt, yp, labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        records.append({
            "Region":   region,
            "N":        len(yt),
            "Accuracy": round(accuracy_score(yt, yp), 4),
            "MAE":      round(ordinal_mae(yt, yp), 4),
            "F1 Macro": round(report["macro avg"]["f1-score"], 4),
        })

    reg_df = pd.DataFrame(records)

    africa    = reg_df[reg_df["Region"] == "Africa"]
    benchmark = reg_df[reg_df["Region"] == "Benchmark"]

    africa_mae    = africa["MAE"].values[0]    if not africa.empty    else None
    benchmark_mae = benchmark["MAE"].values[0] if not benchmark.empty else None
    africa_acc    = africa["Accuracy"].values[0]    if not africa.empty    else None
    benchmark_acc = benchmark["Accuracy"].values[0] if not benchmark.empty else None

    bias_gap_mae = (
        round(africa_mae - benchmark_mae, 4)
        if africa_mae is not None and benchmark_mae is not None
        else None
    )
    bias_gap_acc = (
        round(benchmark_acc - africa_acc, 4)
        if africa_acc is not None and benchmark_acc is not None
        else None
    )

    return reg_df, bias_gap_mae, bias_gap_acc


def direction_accuracy(y_true_cls, y_pred_cls, current_cls):
    y_true_cls  = np.asarray(y_true_cls,  dtype=int)
    y_pred_cls  = np.asarray(y_pred_cls,  dtype=int)
    current_cls = np.asarray(current_cls, dtype=int)

    true_dir = np.clip(y_true_cls  - current_cls, -1, 1)
    pred_dir = np.clip(y_pred_cls  - current_cls, -1, 1)

    dir_acc = accuracy_score(true_dir, pred_dir)
    return float(dir_acc), true_dir, pred_dir


def normalize_proba(proba):
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2 or proba.shape[1] != 3:
        raise ValueError(f"proba must have shape (n, 3), got {proba.shape}")

    proba = np.where(np.isnan(proba), 1.0 / 3.0, proba)
    row_sums = proba.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return proba / row_sums
