"""
evaluate.py — End-to-end model evaluation for sovereign credit rating prediction.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from .metrics import (
    compute_classification_metrics,
    regional_metrics,
    direction_accuracy,
    normalize_proba,
    CLASS_NAMES,
)

warnings.filterwarnings("ignore")

MODEL_COLS = {
    "Ordered Logistic": "pred_ord",
    "XGBoost":          "pred_xgb",
    "LSTM":             "pred_lstm",
}
PROBA_COLS = {
    "XGBoost": ["prob_default_xgb",  "prob_junk_xgb",  "prob_invgrade_xgb"],
    "LSTM":    ["prob_default_lstm", "prob_junk_lstm", "prob_invgrade_lstm"],
}


def load_predictions(results_dir):
    path = Path(results_dir) / "test_predictions.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    for col in MODEL_COLS.values():
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    return df


def run_full_evaluation(results_dir, save=True):
    results_dir = Path(results_dir)
    df = load_predictions(results_dir)

    available = {
        name: col
        for name, col in MODEL_COLS.items()
        if col in df.columns
    }
    df_full = df.dropna(subset=list(available.values())).copy()
    for col in available.values():
        df_full[col] = df_full[col].astype(int)

    y_true = df_full["future_class"].values.astype(int)

    metrics_rows = []
    for name, col in available.items():
        y_proba = None
        if name in PROBA_COLS:
            pcols = PROBA_COLS[name]
            if all(c in df_full.columns for c in pcols):
                y_proba = normalize_proba(df_full[pcols].values)
        metrics_rows.append(
            compute_classification_metrics(y_true, df_full[col].values, y_proba, model_name=name)
        )
    metrics_df = pd.DataFrame(metrics_rows)

    bias_summary = []
    for name, col in available.items():
        _, gap_mae, gap_acc = regional_metrics(df_full, col)
        bias_summary.append({
            "Model":        name,
            "Bias Gap MAE": gap_mae,
            "Bias Gap Acc": gap_acc,
        })
    bias_df = pd.DataFrame(bias_summary)

    current_col = next(
        (c for c in ["cls_lag1", "current_class"] if c in df_full.columns),
        None,
    )
    dir_rows = []
    if current_col is not None:
        current = df_full[current_col].values.astype(int)
        for name, col in available.items():
            acc, _, _ = direction_accuracy(y_true, df_full[col].values.astype(int), current)
            dir_rows.append({"Model": name, "Direction Accuracy": round(acc, 4)})
    direction_df = pd.DataFrame(dir_rows)

    if save:
        metrics_df.to_csv(results_dir / "full_metrics_table.csv", index=False)
        bias_df.to_csv(results_dir / "bias_analysis.csv", index=False)
        if not direction_df.empty:
            direction_df.to_csv(results_dir / "direction_accuracy.csv", index=False)

    return {"metrics": metrics_df, "bias": bias_df, "direction": direction_df}


def plot_confusion_matrices(df_full, available_models, results_dir=None):
    n = len(available_models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    y_true = df_full["future_class"].values.astype(int)

    for ax, (name, col) in zip(axes, available_models.items()):
        y_pred = df_full[col].values.astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, ax=ax, cbar=False,
        )
        acc = (y_true == y_pred).mean()
        mae = np.mean(np.abs(y_pred - y_true))
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.text(
            0.5, -0.18, f"Acc: {acc:.3f} | MAE: {mae:.3f}",
            transform=ax.transAxes, ha="center", fontsize=10, color="gray",
        )

    plt.tight_layout()
    if results_dir is not None:
        plt.savefig(Path(results_dir) / "confusion_matrices.png", dpi=120, bbox_inches="tight")
    plt.show()


def plot_bias_analysis(df_full, available_models, results_dir=None):
    model_names = list(available_models.keys())
    model_cols  = list(available_models.values())
    africa_acc, bench_acc = [], []
    africa_mae, bench_mae = [], []

    for col in model_cols:
        reg_df, _, _ = regional_metrics(df_full, col)
        africa_acc.append(reg_df[reg_df.Region == "Africa"]["Accuracy"].values[0])
        bench_acc.append(reg_df[reg_df.Region == "Benchmark"]["Accuracy"].values[0])
        africa_mae.append(reg_df[reg_df.Region == "Africa"]["MAE"].values[0])
        bench_mae.append(reg_df[reg_df.Region == "Benchmark"]["MAE"].values[0])

    x = np.arange(len(model_names))
    w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Regional Performance: Africa vs. Benchmark", fontsize=14, fontweight="bold")

    for ax, africa_vals, bench_vals, ylabel, title in [
        (axes[0], africa_acc, bench_acc, "Accuracy",      "Accuracy per Region"),
        (axes[1], africa_mae, bench_mae, "MAE (Ordinal)", "Ordinal MAE per Region"),
    ]:
        ax.bar(x - w/2, africa_vals, w, label="Africa",    color="#c0392b", alpha=0.85)
        ax.bar(x + w/2, bench_vals,  w, label="Benchmark", color="#2980b9", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    if results_dir is not None:
        plt.savefig(Path(results_dir) / "bias_analysis_plot.png", dpi=150, bbox_inches="tight")
    plt.show()
