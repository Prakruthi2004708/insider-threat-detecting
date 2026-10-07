import json
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
from common import *


def main():

    print("=" * 70)
    print("STEP 12 - IMPROVE INSIDER THREAT DETECTION")
    print("=" * 70)

    # ---------------------------------------------------------
    # LOAD TRAIN AND TEST DATA
    # ---------------------------------------------------------

    train, test, cutoff = load_split()

    cols = Z_COLS + ["weekend"]

    print("\nTraining rows :", len(train))
    print("Testing rows  :", len(test))
    print("Features used :", len(cols))

    # ---------------------------------------------------------
    # CONFIGURATION 1 FROM STEP 11
    # ---------------------------------------------------------

    model = IsolationForest(
        n_estimators=100,
        max_samples="auto",
        max_features=1.0,
        random_state=SEED,
        n_jobs=-1
    )

    print("\nTraining Isolation Forest...")

    model.fit(train[cols])

    # ---------------------------------------------------------
    # ANOMALY SCORES
    # ---------------------------------------------------------

    train_score = -model.score_samples(train[cols])
    test_score = -model.score_samples(test[cols])

    # ---------------------------------------------------------
    # DIFFERENT THRESHOLDS
    #
    # Lower threshold = more anomalies detected
    # Higher threshold = fewer alerts
    # ---------------------------------------------------------

    percentiles = [
        90,
        91,
        92,
        93,
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

    # ---------------------------------------------------------
    # TEST EACH THRESHOLD
    # ---------------------------------------------------------

    for percentile in percentiles:

        threshold = float(
            np.percentile(train_score, percentile)
        )

        pred = (
            test_score >= threshold
        ).astype(int)

        # Existing evaluation function
        m = evaluate(
            test["label"].values,
            pred,
            test_score
        )

        # Standard accuracy
        accuracy = accuracy_score(
            test["label"].values,
            pred
        )

        # Confusion matrix
        tp = int(m["tp"])
        fp = int(m["fp"])
        fn = int(m["fn"])
        tn = int(m["tn"])

        # Alerts per day
        alerts_per_day = float(
            pred.sum() / test["date"].nunique()
        )

        result = {
            "percentile": percentile,
            "threshold": threshold,

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

        results.append(result)

        print("\nThreshold percentile:", percentile)

        print(
            "Accuracy       :",
            round(accuracy * 100, 2),
            "%"
        )

        print(
            "Precision      :",
            round(m["precision"] * 100, 2),
            "%"
        )

        print(
            "Recall         :",
            round(m["recall"] * 100, 2),
            "%"
        )

        print(
            "F1-score       :",
            round(m["f1"] * 100, 2),
            "%"
        )

        print(
            "ROC-AUC        :",
            round(m["roc_auc"] * 100, 2),
            "%"
        )

        print(
            "PR-AUC         :",
            round(m["pr_auc"] * 100, 2),
            "%"
        )

        print(
            "FPR            :",
            round(m["fpr"] * 100, 2),
            "%"
        )

        print(
            "Alerts/day     :",
            round(alerts_per_day, 4)
        )

        print(
            "TP:",
            tp,
            "| FP:",
            fp,
            "| FN:",
            fn,
            "| TN:",
            tn
        )

    # ---------------------------------------------------------
    # RESULTS DATAFRAME
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    # ---------------------------------------------------------
    # BEST BY ACCURACY
    # ---------------------------------------------------------

    best_accuracy = results_df.loc[
        results_df["accuracy"].idxmax()
    ]

    # ---------------------------------------------------------
    # BEST BY PRECISION
    # ---------------------------------------------------------

    best_precision = results_df.loc[
        results_df["precision"].idxmax()
    ]

    # ---------------------------------------------------------
    # BEST BY RECALL
    # ---------------------------------------------------------

    best_recall = results_df.loc[
        results_df["recall"].idxmax()
    ]

    # ---------------------------------------------------------
    # BEST BY F1
    # ---------------------------------------------------------

    best_f1 = results_df.loc[
        results_df["f1"].idxmax()
    ]

    # ---------------------------------------------------------
    # BEST BY PR-AUC
    # ---------------------------------------------------------

    best_pr_auc = results_df.loc[
        results_df["pr_auc"].idxmax()
    ]

    # ---------------------------------------------------------
    # BEST BY ROC-AUC
    # ---------------------------------------------------------

    best_roc_auc = results_df.loc[
        results_df["roc_auc"].idxmax()
    ]

    # ---------------------------------------------------------
    # PRINT BEST RESULTS
    # ---------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("BEST RESULTS")
    print("=" * 70)

    print("\nBEST BY ACCURACY")
    print("-" * 40)
    print(
        "Threshold percentile:",
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

    print("\nBEST BY PRECISION")
    print("-" * 40)
    print(
        "Threshold percentile:",
        int(best_precision["percentile"])
    )
    print(
        "Precision:",
        round(best_precision["precision"] * 100, 2),
        "%"
    )
    print(
        "Recall:",
        round(best_precision["recall"] * 100, 2),
        "%"
    )
    print(
        "F1:",
        round(best_precision["f1"] * 100, 2),
        "%"
    )
    print(
        "Accuracy:",
        round(best_precision["accuracy_percent"], 2),
        "%"
    )

    print("\nBEST BY RECALL")
    print("-" * 40)
    print(
        "Threshold percentile:",
        int(best_recall["percentile"])
    )
    print(
        "Recall:",
        round(best_recall["recall"] * 100, 2),
        "%"
    )
    print(
        "Precision:",
        round(best_recall["precision"] * 100, 2),
        "%"
    )
    print(
        "F1:",
        round(best_recall["f1"] * 100, 2),
        "%"
    )
    print(
        "Accuracy:",
        round(best_recall["accuracy_percent"], 2),
        "%"
    )

    print("\nBEST BY F1-SCORE")
    print("-" * 40)
    print(
        "Threshold percentile:",
        int(best_f1["percentile"])
    )
    print(
        "F1:",
        round(best_f1["f1"] * 100, 2),
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
        "Accuracy:",
        round(best_f1["accuracy_percent"], 2),
        "%"
    )

    print("\nBEST BY PR-AUC")
    print("-" * 40)
    print(
        "Threshold percentile:",
        int(best_pr_auc["percentile"])
    )
    print(
        "PR-AUC:",
        round(best_pr_auc["pr_auc"] * 100, 2),
        "%"
    )
    print(
        "Accuracy:",
        round(best_pr_auc["accuracy_percent"], 2),
        "%"
    )

    print("\nBEST BY ROC-AUC")
    print("-" * 40)
    print(
        "Threshold percentile:",
        int(best_roc_auc["percentile"])
    )
    print(
        "ROC-AUC:",
        round(best_roc_auc["roc_auc"] * 100, 2),
        "%"
    )
    print(
        "PR-AUC:",
        round(best_roc_auc["pr_auc"] * 100, 2),
        "%"
    )
    print(
        "Accuracy:",
        round(best_roc_auc["accuracy_percent"], 2),
        "%"
    )

    # ---------------------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------------------

    output_file = RESULTS / "step12_threshold_results.csv"

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
        "best_precision": best_precision.to_dict(),
        "best_recall": best_recall.to_dict(),
        "best_f1": best_f1.to_dict(),
        "best_pr_auc": best_pr_auc.to_dict(),
        "best_roc_auc": best_roc_auc.to_dict()
    }

    json_file = RESULTS / "step12_best_results.json"

    with open(json_file, "w") as f:
        json.dump(
            best_results,
            f,
            indent=2,
            default=float
        )

    print(json_file)

    # ---------------------------------------------------------
    # SAVE SCORES FOR BEST F1 MODEL
    # ---------------------------------------------------------

    best_f1_threshold = float(
        best_f1["threshold"]
    )

    best_f1_pred = (
        test_score >= best_f1_threshold
    ).astype(int)

    test_output = test.copy()

    test_output["step12_score"] = test_score
    test_output["step12_prediction"] = best_f1_pred

    score_file = RESULTS / "step12_best_f1_scores.parquet"

    test_output.to_parquet(
        score_file,
        index=False
    )

    print(score_file)

    print("\n")
    print("=" * 70)
    print("STEP 12 COMPLETED")
    print("=" * 70)


main()