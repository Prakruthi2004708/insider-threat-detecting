import json
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

from common import load_split, SEED, RESULTS


def metrics(y, pred, score, n_days):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "alerts_per_day": float(pred.sum() / n_days),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def show(name, m):
    print("\n" + name)
    print("-" * 40)
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fpr"]:
        print(f"{k:<15}: {round(m[k] * 100, 2)} %")
    print("alerts_per_day :", round(m["alerts_per_day"], 4))
    print(f"TP: {m['tp']} | FP: {m['fp']} | FN: {m['fn']} | TN: {m['tn']}")


def main():
    print("=" * 70)
    print("STEP 14 - MODEL COMPARISON (leak-free)")
    print("=" * 70)

    train, test, cutoff = load_split()
    exclude = {"label", "date", "user", "id", "index", "scenario"}
    cols = [c for c in train.select_dtypes(include=[np.number]).columns
            if c not in exclude]

    def prep(df, med=None):
        X = df[cols].replace([np.inf, -np.inf], np.nan)
        med = X.median() if med is None else med
        return X.fillna(med), med

    train = train.sort_values("date").reset_index(drop=True)
    X_train, med = prep(train)
    X_test, _ = prep(test, med)
    y_train = train["label"].values
    y_test = test["label"].values
    n_days = test["date"].nunique()

    print("Features:", len(cols), "| train insiders:", int(y_train.sum()),
          "| test insiders:", int(y_test.sum()))

    results = {}

    # 1. Isolation Forest (unsupervised baseline, threshold from train)
    iso = IsolationForest(n_estimators=100, random_state=SEED, n_jobs=-1)
    iso.fit(X_train)
    s_tr = -iso.score_samples(X_train)
    s_te = -iso.score_samples(X_test)
    thr = np.percentile(s_tr, 95)
    results["isolation_forest"] = metrics(
        y_test, (s_te >= thr).astype(int), s_te, n_days)

    # 2. Random Forest (supervised)
    # Last 20% of training rows (by date) pick the threshold
    split = int(len(train) * 0.8)
    Xa, Xv = X_train.iloc[:split], X_train.iloc[split:]
    ya, yv = y_train[:split], y_train[split:]

    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced_subsample",
        min_samples_leaf=2, random_state=SEED, n_jobs=-1)
    rf.fit(Xa, ya)

    pv = rf.predict_proba(Xv)[:, 1]
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.01, 0.95, 95):
        f = f1_score(yv, (pv >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_t, best_f1 = float(t), f

    rf.fit(X_train, y_train)  # refit on all training data
    p_te = rf.predict_proba(X_test)[:, 1]
    imp = pd.Series(rf.feature_importances_, index=cols).sort_values(ascending=False)
    print("\nTop 10 feature importances")
    print(imp.head(10).round(4).to_string())
    results["random_forest"] = metrics(
        y_test, (p_te >= best_t).astype(int), p_te, n_days)
    results["random_forest"]["threshold"] = best_t
    out = test[["user", "date", "label"]].copy()
    out["rf_score"] = p_te
    out["rf_pred"] = (p_te >= best_t).astype(int)
    out.to_parquet(RESULTS / "step14_rf_scores.parquet", index=False)

    print("\nRandom Forest threshold (picked on validation):", best_t)

    show("ISOLATION FOREST (unsupervised, percentile 95)", results["isolation_forest"])
    show("RANDOM FOREST (supervised)", results["random_forest"])

    pd.DataFrame(results).T.to_csv(RESULTS / "step14_model_comparison.csv")
    with open(RESULTS / "step14_model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved:", RESULTS / "step14_model_comparison.csv")
    print("STEP 14 COMPLETED")


if __name__ == "__main__":
    main()