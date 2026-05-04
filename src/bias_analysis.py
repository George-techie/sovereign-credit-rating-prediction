"""
bias_analysis.py — Regional bias analysis for sovereign credit rating prediction.

Computes and summarises performance gaps between African and Benchmark countries.
The bias gap is defined as Africa MAE minus Benchmark MAE — a positive value
means the model is less accurate on African sovereigns.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from .metrics import ordinal_mae, CLASS_NAMES

REGIONS = ["Africa", "Benchmark"]


def compute_bias_report(df, pred_cols, true_col="future_class"):
    """
    Compute a bias report across multiple models.

    Args:
        df:        DataFrame with predictions, true labels, and 'region' column.
        pred_cols: dict mapping model name -> prediction column name.
        true_col:  str. Column name for true labels.

    Returns:
        DataFrame with columns:
            Model, Africa Acc, Benchmark Acc, Acc Gap,
            Africa MAE, Benchmark MAE, MAE Gap
    """
    rows = []
    for model_name, pred_col in pred_cols.items():
        sub = df.dropna(subset=[pred_col]).copy()
        sub[pred_col] = sub[pred_col].astype(int)

        row = {"Model": model_name}
        for region in REGIONS:
            reg = sub[sub["region"] == region]
            if reg.empty:
                row[f"{region} Acc"] = None
                row[f"{region} MAE"] = None
                continue
            yt = reg[true_col].values.astype(int)
            yp = reg[pred_col].values.astype(int)
            row[f"{region} Acc"] = round(accuracy_score(yt, yp), 4)
            row[f"{region} MAE"] = round(ordinal_mae(yt, yp), 4)

        africa_acc    = row.get("Africa Acc")
        benchmark_acc = row.get("Benchmark Acc")
        africa_mae    = row.get("Africa MAE")
        benchmark_mae = row.get("Benchmark MAE")

        row["Acc Gap"] = (
            round(benchmark_acc - africa_acc, 4)
            if africa_acc is not None and benchmark_acc is not None
            else None
        )
        row["MAE Gap"] = (
            round(africa_mae - benchmark_mae, 4)
            if africa_mae is not None and benchmark_mae is not None
            else None
        )
        rows.append(row)

    return pd.DataFrame(rows)


def worst_performing_countries(df, pred_col, true_col="future_class", n=5):
    """
    Return the n countries with the highest ordinal MAE.

    Useful for identifying which sovereigns the model struggles with most.

    Args:
        df:       DataFrame with predictions, true labels, and 'country' column.
        pred_col: str. Prediction column name.
        true_col: str. True label column name.
        n:        int. Number of countries to return.

    Returns:
        DataFrame with columns ['country', 'region', 'N', 'MAE', 'Accuracy']
        sorted by MAE descending.
    """
    df = df.dropna(subset=[pred_col]).copy()
    df[pred_col] = df[pred_col].astype(int)

    rows = []
    for country, grp in df.groupby("country"):
        yt = grp[true_col].values.astype(int)
        yp = grp[pred_col].values.astype(int)
        rows.append({
            "country":  country,
            "region":   grp["region"].iloc[0],
            "N":        len(yt),
            "MAE":      round(ordinal_mae(yt, yp), 4),
            "Accuracy": round(accuracy_score(yt, yp), 4),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("MAE", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def bias_summary_table(df, pred_cols, true_col="future_class"):
    """
    One-line summary: which model has the smallest Africa bias gap.

    Args:
        df:        DataFrame with predictions and 'region' column.
        pred_cols: dict mapping model name -> pred column.
        true_col:  str.

    Returns:
        DataFrame sorted by MAE Gap ascending (least biased first).
    """
    report = compute_bias_report(df, pred_cols, true_col)
    return report.sort_values("MAE Gap").reset_index(drop=True)
