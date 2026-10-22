"""
annual_experiment.py — Experiment A: next-year S&P rating for 80 countries.

Split by the year being predicted (t+1), so every model is tested on years it
never saw: train 2012-2017, validation 2018-2019, test 2020-2025. The test
years include COVID, the 2022 rate shock and several defaults.

The panel starts with ratings at the end of 2011: before that the S&P source
lists only some countries (5 in 2000, 48 in 2010, against 68-71 from 2011), so
earlier years would over-represent whichever countries happen to have them.

Ratings rarely change from one year to the next, so "same as now" is already
very accurate. Every model is therefore compared with that persistence
baseline, and the downgrade / no change / upgrade task measures what persistence
cannot do.

Feature sets:
    fundamentals         economics only: what the macro and fiscal data imply
    fundamentals+rating  plus the current notch and outlook
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from statsmodels.miscmodels.ordinal_model import OrderedModel
from torch.utils.data import DataLoader, TensorDataset

from .annual_panel import FUNDAMENTALS, RATING_STATE
from .metrics import compute_classification_metrics
from .train import SovereignLSTM, get_class_weights, train_xgboost

FIRST_TARGET_YEAR = 2012
TRAIN_LAST_TARGET_YEAR = 2017
VAL_LAST_TARGET_YEAR = 2019
FEATURE_SETS = {
    "fundamentals": FUNDAMENTALS,
    "fundamentals+rating": FUNDAMENTALS + RATING_STATE,
}
SEQ_LEN = 5  # years of history the LSTM sees
SEED = 42
DIRECTIONS = [-1, 0, 1]


def split_by_target_year(rows):
    rows = rows[rows["target_year"] >= FIRST_TARGET_YEAR]
    train = rows[rows["target_year"] <= TRAIN_LAST_TARGET_YEAR]
    val = rows[(rows["target_year"] > TRAIN_LAST_TARGET_YEAR)
               & (rows["target_year"] <= VAL_LAST_TARGET_YEAR)]
    test = rows[rows["target_year"] > VAL_LAST_TARGET_YEAR]
    return train.copy(), val.copy(), test.copy()


def prepare_matrices(train, val, test, features):
    """Impute with training medians, then standardize with the training scaler."""
    medians = train[features].median()
    scaler = StandardScaler().fit(train[features].fillna(medians))

    def transform(frame):
        return scaler.transform(frame[features].fillna(medians))

    return transform(train), transform(val), transform(test), medians, scaler


# ----------------------------------------------------------------- class models

def persistence(test):
    return test["rating_class"].to_numpy()


def fit_ordered_logit(X_train, y_train, X_test):
    """Proportional-odds logit: the classic model for ordered rating classes."""
    model = OrderedModel(y_train, X_train, distr="logit").fit(method="bfgs", disp=False,
                                                              maxiter=500)
    proba = np.asarray(model.predict(X_test))
    return proba.argmax(axis=1), proba


def fit_xgboost_class(X_train, y_train, X_val, y_val, X_test):
    """The project's tuned XGBoost (grid on validation, refit on train+val)."""
    model = train_xgboost(X_train, y_train, X_val, y_val, get_class_weights(y_train))
    return model.predict(X_test).astype(int), model


def country_sequences(panel, rows, features, medians, scaler, seq_len=SEQ_LEN):
    """For each row, the country's last `seq_len` years of features up to year t.

    Years before the country's first observation repeat the first year, so
    every sequence has the same length without borrowing other countries' data.
    """
    filled = panel.copy()
    filled[features] = filled[features].fillna(medians)
    scaled = pd.DataFrame(scaler.transform(filled[features]), index=filled.index)
    sequences = []
    for _, row in rows.iterrows():
        history = filled[(filled["country"] == row["country"]) & (filled["year"] <= row["year"])]
        window = scaled.loc[history.index[-seq_len:]].to_numpy()
        if len(window) < seq_len:
            window = np.vstack([np.repeat(window[:1], seq_len - len(window), axis=0), window])
        sequences.append(window)
    return np.asarray(sequences, dtype=np.float32)


def fit_lstm(seq_train, y_train, seq_val, y_val, seq_test, epochs=60, lr=1e-3):
    """The project's SovereignLSTM on yearly sequences; best epoch by validation accuracy."""
    torch.manual_seed(SEED)
    model = SovereignLSTM(input_size=seq_train.shape[2])
    weights = get_class_weights(y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([weights[c] for c in range(3)],
                                                        dtype=torch.float))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(torch.tensor(seq_train), torch.tensor(y_train)),
                        batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    best_acc, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            criterion(model(X_batch), y_batch).backward()
            optimizer.step()
        val_acc = accuracy_score(y_val, _lstm_predict(model, seq_val))
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return _lstm_predict(model, seq_test)


def _lstm_predict(model, sequences):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(sequences)).argmax(dim=1).numpy()


# ------------------------------------------------------------- direction task

def fit_direction_xgboost(X_train, d_train, X_val, d_val, X_test):
    """Downgrade / no change / upgrade, classes shifted to 0 / 1 / 2 for XGBoost."""
    shift = 1
    model = train_xgboost(X_train, d_train + shift, X_val, d_val + shift,
                          get_class_weights(d_train + shift))
    return model.predict(X_test).astype(int) - shift


def direction_metrics(true_direction, predicted_direction, model_name):
    true_direction = np.asarray(true_direction)
    predicted_direction = np.asarray(predicted_direction)
    recalls = recall_score(true_direction, predicted_direction, labels=DIRECTIONS,
                           average=None, zero_division=0)
    return {
        "Model": model_name,
        "Accuracy": round(accuracy_score(true_direction, predicted_direction), 4),
        "Macro F1": round(f1_score(true_direction, predicted_direction, labels=DIRECTIONS,
                                   average="macro", zero_division=0), 4),
        "Downgrade recall": round(recalls[0], 4),
        "Upgrade recall": round(recalls[2], 4),
        "Downgrades": int((true_direction == -1).sum()),
        "Upgrades": int((true_direction == 1).sum()),
    }


# ------------------------------------------------------------ regional analysis

def class_metrics_on_changes(test, predictions):
    """Accuracy on the test rows whose class actually changed: where persistence scores 0."""
    changed = (test["next_class"] != test["rating_class"]).to_numpy()
    rows = []
    for name, predicted in predictions.items():
        predicted = np.asarray(predicted)
        rows.append({
            "Model": name,
            "Changed rows": int(changed.sum()),
            "Accuracy on changes": (round(accuracy_score(test["next_class"][changed],
                                                         predicted[changed]), 4)
                                    if changed.any() else None),
        })
    return pd.DataFrame(rows)


def notch_residuals_by_region(train, test, medians):
    """Is a region rated below what its fundamentals imply?

    A gradient-boosted regression of next year's notch on fundamentals only is
    fit on the training years; the test residual (actual minus predicted notch)
    averaged by region says whether S&P rates that region above (+) or below
    (-) its fundamentals. Standard errors are clustered by country, because a
    country's years are not independent.
    """
    model = xgb.XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=SEED)
    model.fit(train[FUNDAMENTALS].fillna(medians), train["next_notch"])
    scored = test.assign(
        predicted_notch=model.predict(test[FUNDAMENTALS].fillna(medians)),
    )
    scored["residual"] = scored["next_notch"] - scored["predicted_notch"]
    rows = []
    for region, group in scored.groupby("region"):
        per_country = group.groupby("country")["residual"].mean()
        rows.append({
            "Region": region,
            "Countries": int(per_country.size),
            "Country-years": int(len(group)),
            "Mean residual (notches)": round(group["residual"].mean(), 3),
            "Clustered s.e.": (round(per_country.std(ddof=1) / np.sqrt(per_country.size), 3)
                               if per_country.size > 1 else None),
        })
    return pd.DataFrame(rows).sort_values("Mean residual (notches)"), scored


def metrics_table(test, predictions):
    return pd.DataFrame([
        compute_classification_metrics(test["next_class"], predicted, model_name=name)
        for name, predicted in predictions.items()
    ])


def regional_accuracy(test, predictions):
    rows = []
    for region, group in test.groupby("region"):
        row = {"Region": region, "Rows": len(group)}
        for name, predicted in predictions.items():
            predicted = pd.Series(np.asarray(predicted), index=test.index).loc[group.index]
            row[name] = round(accuracy_score(group["next_class"], predicted), 4)
        rows.append(row)
    return pd.DataFrame(rows)
