import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score,
                             average_precision_score)
from config import *

FEATURES_PARQUET = PROCESSED / "features.parquet"

RAW_FEATURES = [
    "n_events", "n_logon", "n_connect", "n_filecopy", "n_risky_ext",
    "n_after_hours", "n_after_connect", "n_after_file", "n_pcs",
    "first_hour", "last_hour",
    "n_new_pc", "n_new_ext", "n_filecopy_usb_day",
    "n_weekend_events", "span_hours",
]
RAW_COLS = RAW_FEATURES + ["weekend"]
Z_COLS = ["z_" + c for c in RAW_FEATURES]


def load_split(train_frac=0.7):
    df = pd.read_parquet(FEATURES_PARQUET)
    dates = np.sort(df["date"].unique())
    cutoff = dates[int(len(dates) * train_frac)]
    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()
    return train, test, cutoff


def evaluate(y_true, y_pred, score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "roc_auc": float(roc_auc_score(y_true, score)),
        "pr_auc": float(average_precision_score(y_true, score)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }