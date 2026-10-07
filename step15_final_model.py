import json
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from common import load_split, SEED, RESULTS


def main():
    print("=" * 70)
    print("STEP 15 - FINAL MODEL (leak-free, train-based threshold)")
    print("=" * 70)

    # 1. Load data
    train, test, cutoff = load_split()

    # "scenario" is the insider-scenario label in CERT -> leakage, so excluded
    exclude = {"label", "date", "user", "id", "index", "scenario"}

    feature_cols = [
        c for c in train.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]

    X_train = train[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_test = test[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Fill missing values with TRAIN medians only
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    y_test = test["label"].values
    n_test_days = test["date"].nunique()

    print("\nTraining rows      :", len(X_train))
    print("Testing rows       :", len(X_test))
    print("Number of features :", len(feature_cols))
    print("'scenario' used?   :", "scenario" in feature_cols)

    # 2. Model configuration
    N_ESTIMATORS = 100
    MAX_SAMPLES = "auto"
    MAX_FEATURES = 1.0
    FINAL_THRESHOLD = 95  # fixed operating point, chosen before looking at results

    # 3. Train
    print("\nTraining Isolation Forest...")
    iso = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        max_features=MAX_FEATURES,
        random_state=SEED,
        n_jobs=-1,
    )
    iso.fit(X_train)

    # 4. Scores (higher = more anomalous)
    train_scores = -iso.score_samples(X_train)
    test_scores = -iso.score_samples(X_test)

    roc_auc = roc_auc_score(y_test, test_scores)
    pr_auc = average_precision_score(y_test, test_scores)

    # 5. Thresholds come from TRAINING scores only
    percentiles = [90, 92, 94, 95, 96, 97, 98, 99]
    results = []

    print("\n" + "=" * 70)
    print("THRESHOLD TESTING (threshold learned on training data)")
    print("=" * 70)

    for p in percentiles:
        thr = float(np.percentile(train_scores, p))
        pred = (test_scores >= thr).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test, pred, labels=[0, 1]
        ).ravel()

        row = {
            "threshold_percentile": p,
            "threshold_value": thr,
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            "alerts_per_day": float(pred.sum() / n_test_days),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        }
        results.append(row)

        print(f"\nThreshold percentile: {p}")
        print("Accuracy       :", round(row["accuracy"] * 100, 2), "%")
        print("Precision      :", round(row["precision"] * 100, 2), "%")
        print("Recall         :", round(row["recall"] * 100, 2), "%")
        print("F1-score       :", round(row["f1"] * 100, 2), "%")
        print("ROC-AUC        :", round(roc_auc * 100, 2), "%")
        print("PR-AUC         :", round(pr_auc * 100, 2), "%")
        print("FPR            :", round(row["fpr"] * 100, 2), "%")
        print("Alerts/day     :", round(row["alerts_per_day"], 4))
        print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")

    results_df = pd.DataFrame(results)

    # 6. Final operating point
    final = results_df[
        results_df["threshold_percentile"] == FINAL_THRESHOLD
    ].iloc[0]
    final_pred = (test_scores >= final["threshold_value"]).astype(int)

    print("\n" + "=" * 70)
    print("FINAL MODEL")
    print("=" * 70)
    print("Model       : Isolation Forest")
    print("Trees       :", N_ESTIMATORS)
    print("Threshold   : training-score percentile", FINAL_THRESHOLD)
    print("Precision   :", round(final["precision"] * 100, 2), "%")
    print("Recall      :", round(final["recall"] * 100, 2), "%")
    print("F1-score    :", round(final["f1"] * 100, 2), "%")
    print("ROC-AUC     :", round(roc_auc * 100, 2), "%")
    print("PR-AUC      :", round(pr_auc * 100, 2), "%")
    print("FPR         :", round(final["fpr"] * 100, 2), "%")
    print("Alerts/day  :", round(final["alerts_per_day"], 4))
    print("Accuracy    :", round(final["accuracy"] * 100, 2),
          "% (secondary metric: classes are highly imbalanced)")
    print(f"TP: {int(final['tp'])} | FP: {int(final['fp'])} | "
          f"FN: {int(final['fn'])} | TN: {int(final['tn'])}")

    # 7. Save outputs
    results_df.to_csv(RESULTS / "step15_final_threshold_results.csv", index=False)

    pd.DataFrame({
        "label": y_test,
        "anomaly_score": test_scores,
        "prediction": final_pred,
    }).to_parquet(RESULTS / "step15_final_scores.parquet", index=False)

    info = {
        "model": "Isolation Forest",
        "n_estimators": N_ESTIMATORS,
        "max_samples": MAX_SAMPLES,
        "max_features": MAX_FEATURES,
        "random_state": int(SEED),
        "number_of_features": len(feature_cols),
        "feature_columns": feature_cols,
        "threshold_percentile_on_train_scores": FINAL_THRESHOLD,
        "threshold_value": float(final["threshold_value"]),
        "precision": float(final["precision"]),
        "recall": float(final["recall"]),
        "f1_score": float(final["f1"]),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "fpr": float(final["fpr"]),
        "alerts_per_day": float(final["alerts_per_day"]),
        "accuracy": float(final["accuracy"]),
        "tp": int(final["tp"]), "fp": int(final["fp"]),
        "fn": int(final["fn"]), "tn": int(final["tn"]),
    }
    with open(RESULTS / "step15_final_model.json", "w") as f:
        json.dump(info, f, indent=2)

    with open(RESULTS / "step15_final_features.txt", "w") as f:
        f.write("\n".join(feature_cols) + "\n")

    print("\nFiles saved in:", RESULTS)
    print("\nSTEP 15 COMPLETED")


if __name__ == "__main__":
    main()