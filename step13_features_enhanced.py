import json
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score

from common import *


def evaluate_model(test, test_score, threshold, percentile):
    """
    Evaluate one threshold.
    """

    pred = (test_score >= threshold).astype(int)

    m = evaluate(
        test["label"].values,
        pred,
        test_score
    )

    accuracy = accuracy_score(
        test["label"].values,
        pred
    )

    tp = int(m["tp"])
    fp = int(m["fp"])
    fn = int(m["fn"])
    tn = int(m["tn"])

    alerts_per_day = float(
        pred.sum() / test["date"].nunique()
    )

    return {
        "percentile": percentile,
        "threshold": float(threshold),

        "accuracy": float(accuracy),
        "accuracy_percent": float(accuracy * 100),

        "precision": float(m["precision"]),
        "recall": float(m["recall"]),
        "f1": float(m["f1"]),

        "fpr": float(m["fpr"]),
        "roc_auc": float(m["roc_auc"]),
        "pr_auc": float(m["pr_auc"]),

        "alerts_per_day": alerts_per_day,

        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }



def main():

    print("=" * 70)
    print("STEP 13 - FEATURE-ENHANCED ISOLATION FOREST")
    print("=" * 70)

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    train, test, cutoff = load_split()

    print("\nTraining rows :", len(train))
    print("Testing rows  :", len(test))

    # ---------------------------------------------------------
    # FIND NUMERIC BEHAVIORAL FEATURES
    # ---------------------------------------------------------

    excluded = {
        "label",
        "date",
        "user"
    }

    numeric_cols = train.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    feature_cols = [
        c for c in numeric_cols
        if c not in excluded
    ]

    # Remove obvious identifiers if present
    identifier_words = [
        "id",
        "index"
    ]

    cleaned_features = []

    for col in feature_cols:

        col_lower = col.lower()

        # Keep normal behavioral columns
        # but avoid columns that look like identifiers.
        if col_lower in ["id", "index"]:
            continue

        cleaned_features.append(col)

    feature_cols = cleaned_features

    print("\nFeature-enhanced model")
    print("Number of features:", len(feature_cols))

    print("\nFeatures used:")
    for i, col in enumerate(feature_cols, 1):
        print(i, ".", col)

    # ---------------------------------------------------------
    # CHECK FOR MISSING VALUES
    # ---------------------------------------------------------

    train_features = train[feature_cols].copy()
    test_features = test[feature_cols].copy()

    train_features = train_features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    test_features = test_features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill missing values using training medians
    train_medians = train_features.median()

    train_features = train_features.fillna(
        train_medians
    )

    test_features = test_features.fillna(
        train_medians
    )

    # ---------------------------------------------------------
    # TRAIN ISOLATION FOREST
    # ---------------------------------------------------------

    print("\nTraining feature-enhanced Isolation Forest...")

    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        max_features=1.0,
        random_state=SEED,
        n_jobs=-1
    )

    model.fit(train_features)

    # ---------------------------------------------------------
    # ANOMALY SCORES
    # ---------------------------------------------------------

    train_score = -model.score_samples(
        train_features
    )

    test_score = -model.score_samples(
        test_features
    )

    # ---------------------------------------------------------
    # TEST MULTIPLE THRESHOLDS
    # ---------------------------------------------------------

    percentiles = [
        90,
        92,
        94,
        95,
        96,
        97,
        98,
        99
    ]

    results = []

    print("\n")
    print("=" * 70)
    print("THRESHOLD TESTING")
    print("=" * 70)

    for percentile in percentiles:

        threshold = float(
            np.percentile(
                train_score,
                percentile
            )
        )

        result = evaluate_model(
            test,
            test_score,
            threshold,
            percentile
        )

        results.append(result)

        print("\nThreshold:", percentile)

        print(
            "Accuracy       :",
            round(result["accuracy_percent"], 2),
            "%"
        )

        print(
            "Precision      :",
            round(result["precision"] * 100, 2),
            "%"
        )

        print(
            "Recall         :",
            round(result["recall"] * 100, 2),
            "%"
        )

        print(
            "F1-score       :",
            round(result["f1"] * 100, 2),
            "%"
        )

        print(
            "ROC-AUC        :",
            round(result["roc_auc"] * 100, 2),
            "%"
        )

        print(
            "PR-AUC         :",
            round(result["pr_auc"] * 100, 2),
            "%"
        )

        print(
            "FPR            :",
            round(result["fpr"] * 100, 2),
            "%"
        )

        print(
            "Alerts/day     :",
            round(result["alerts_per_day"], 4)
        )

        print(
            "TP:",
            result["tp"],
            "| FP:",
            result["fp"],
            "| FN:",
            result["fn"],
            "| TN:",
            result["tn"]
        )

    # ---------------------------------------------------------
    # RESULTS DATAFRAME
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    # ---------------------------------------------------------
    # FIND BEST RESULTS
    # ---------------------------------------------------------

    best_accuracy = results_df.loc[
        results_df["accuracy"].idxmax()
    ]

    best_precision = results_df.loc[
        results_df["precision"].idxmax()
    ]

    best_recall = results_df.loc[
        results_df["recall"].idxmax()
    ]

    best_f1 = results_df.loc[
        results_df["f1"].idxmax()
    ]

    best_roc = results_df.loc[
        results_df["roc_auc"].idxmax()
    ]

    best_pr = results_df.loc[
        results_df["pr_auc"].idxmax()
    ]

    # ---------------------------------------------------------
    # PRINT BEST RESULTS
    # ---------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("BEST RESULTS - STEP 13")
    print("=" * 70)

    print("\nBEST BY ACCURACY")
    print("-" * 40)

    print(
        "Threshold:",
        int(best_accuracy["percentile"])
    )

    print(
        "Accuracy:",
        round(best_accuracy["accuracy_percent"], 2),
        "%"
    )

    print(
        "Precision:",
        round(best_accuracy["precision"] * 100, 2),
        "%"
    )

    print(
        "Recall:",
        round(best_accuracy["recall"] * 100, 2),
        "%"
    )

    print(
        "F1:",
        round(best_accuracy["f1"] * 100, 2),
        "%"
    )

    print("\nBEST BY F1-SCORE")
    print("-" * 40)

    print(
        "Threshold:",
        int(best_f1["percentile"])
    )

    print(
        "Accuracy:",
        round(best_f1["accuracy_percent"], 2),
        "%"
    )

    print(
        "Precision:",
        round(best_f1["precision"] * 100, 2),
        "%"
    )

    print(
        "Recall:",
        round(best_f1["recall"] * 100, 2),
        "%"
    )

    print(
        "F1:",
        round(best_f1["f1"] * 100, 2),
        "%"
    )

    print("\nBEST BY RECALL")
    print("-" * 40)

    print(
        "Threshold:",
        int(best_recall["percentile"])
    )

    print(
        "Accuracy:",
        round(best_recall["accuracy_percent"], 2),
        "%"
    )

    print(
        "Precision:",
        round(best_recall["precision"] * 100, 2),
        "%"
    )

    print(
        "Recall:",
        round(best_recall["recall"] * 100, 2),
        "%"
    )

    print(
        "F1:",
        round(best_recall["f1"] * 100, 2),
        "%"
    )

    print("\nBEST BY ROC-AUC")
    print("-" * 40)

    print(
        "ROC-AUC:",
        round(best_roc["roc_auc"] * 100, 2),
        "%"
    )

    print(
        "Accuracy:",
        round(best_roc["accuracy_percent"], 2),
        "%"
    )

    print(
        "Precision:",
        round(best_roc["precision"] * 100, 2),
        "%"
    )

    print(
        "Recall:",
        round(best_roc["recall"] * 100, 2),
        "%"
    )

    # ---------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------

    output_file = RESULTS / "step13_feature_enhanced_results.csv"

    results_df.to_csv(
        output_file,
        index=False
    )

    print("\n")
    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(output_file)

    # ---------------------------------------------------------
    # SAVE BEST RESULTS
    # ---------------------------------------------------------

    best_results = {
        "best_accuracy": best_accuracy.to_dict(),
        "best_f1": best_f1.to_dict(),
        "best_recall": best_recall.to_dict(),
        "best_roc_auc": best_roc.to_dict(),
        "best_pr_auc": best_pr.to_dict(),
        "number_of_features": len(feature_cols),
        "features": feature_cols
    }

    json_file = RESULTS / "step13_best_results.json"

    with open(json_file, "w") as f:
        json.dump(
            best_results,
            f,
            indent=2,
            default=float
        )

    print(json_file)

    # ---------------------------------------------------------
    # SAVE BEST F1 SCORES
    # ---------------------------------------------------------

    best_threshold = float(
        best_f1["threshold"]
    )

    best_prediction = (
        test_score >= best_threshold
    ).astype(int)

    test_output = test.copy()

    test_output["step13_anomaly_score"] = test_score
    test_output["step13_prediction"] = best_prediction

    score_file = RESULTS / "step13_best_scores.parquet"

    test_output.to_parquet(
        score_file,
        index=False
    )

    print(score_file)

    print("\n")
    print("=" * 70)
    print("STEP 13 COMPLETED")
    print("=" * 70)


main()