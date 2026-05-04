"""
train.py — Model training pipeline for sovereign credit rating prediction.
"""

import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import joblib

from .features import FEATURE_COLS

warnings.filterwarnings("ignore")

TARGET_COL  = "future_class"
CLASS_NAMES = ["Default", "Junk", "Inv. Grade"]
TRAIN_END   = "2020-01-01"
VAL_END     = "2022-01-01"


def time_split(df):
    train = df[df["date"] <  TRAIN_END].copy()
    val   = df[(df["date"] >= TRAIN_END) & (df["date"] < VAL_END)].copy()
    test  = df[df["date"] >= VAL_END].copy()
    return train, val, test


def fit_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def get_class_weights(y_train):
    classes = np.array([0, 1, 2])
    cw = compute_class_weight("balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, cw)}


def train_xgboost(X_train, y_train, X_val, y_val, class_weights):
    param_grid = {
        "n_estimators":     [200, 400],
        "max_depth":        [3, 5],
        "learning_rate":    [0.05, 0.1],
        "subsample":        [0.8],
        "colsample_bytree": [0.8],
    }
    sample_weights = np.array([class_weights[y] for y in y_train])

    best_val_acc = 0.0
    best_params  = {}
    keys   = list(param_grid.keys())
    combos = list(product(*param_grid.values()))

    for combo in combos:
        params = dict(zip(keys, combo))
        clf = xgb.XGBClassifier(
            **params,
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42,
            verbosity=0,
        )
        clf.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        val_acc = accuracy_score(y_val, clf.predict(X_val))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params  = params

    X_tv  = np.vstack([X_train, X_val])
    y_tv  = np.concatenate([y_train, y_val])
    sw_tv = np.array([class_weights[y] for y in y_tv])

    best_clf = xgb.XGBClassifier(
        **best_params,
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    )
    best_clf.fit(X_tv, y_tv, sample_weight=sw_tv)
    return best_clf


class SovereignSequenceDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, scaler, seq_len=12):
        self.sequences = []
        self.labels    = []

        for country, grp in df.sort_values("date").groupby("country"):
            X = scaler.transform(grp[feature_cols].values)
            y = grp[target_col].values
            for i in range(seq_len, len(X)):
                self.sequences.append(X[i - seq_len:i])
                self.labels.append(y[i])

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.labels    = np.array(self.labels,    dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx]),
            torch.tensor(self.labels[idx]),
        )


class SovereignLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2,
                 dropout=0.3, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.dropout(out[:, -1, :])
        return self.fc(out)


def train_lstm(df_train, df_val, scaler, class_weights,
               seq_len=12, epochs=30, batch_size=32, lr=1e-3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SovereignSequenceDataset(df_train, FEATURE_COLS, TARGET_COL, scaler, seq_len)
    val_ds   = SovereignSequenceDataset(df_val,   FEATURE_COLS, TARGET_COL, scaler, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = SovereignLSTM(input_size=len(FEATURE_COLS)).to(device)

    weights_tensor = torch.tensor(
        [class_weights[i] for i in range(3)], dtype=torch.float
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_state   = None

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                preds = model(X_batch.to(device)).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)

        val_acc = accuracy_score(val_ds.labels, all_preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model.cpu().eval()


def save_artifacts(results_dir, scaler=None, xgb_model=None, lstm_model=None):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if scaler is not None:
        joblib.dump(scaler, results_dir / "scaler.pkl")
    if xgb_model is not None:
        xgb_model.save_model(str(results_dir / "xgboost_model.json"))
    if lstm_model is not None:
        torch.save(lstm_model.state_dict(), results_dir / "lstm_best.pt")
