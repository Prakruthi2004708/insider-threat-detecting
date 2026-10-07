import json
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score
from common import *


def calculate_accuracy(y_true, y_pred):
    """
    Calculate standard classification accuracy.
    """
    return float(accuracy_score(y_true, y_pred))


def main():

    print("=" * 70)
    print("STEP 11 - ISOLATION FOREST OPTIMIZATION")
    print("=" * 70)

    train, test, cutoff = load_split()

    # Features used by the proposed model
    cols = Z_COLS + ["weekend"]

    # ---------------------------------------------------------
    # Configurations to test
    # ---------------------------------------------------------
    configs = [
        {
            "config": 1,
            "n_estimators": 100,
            "max_samples": "auto",
            "max_features": 1.0
        },
        {
            "config": 2,
            "n_estimators": 200,
            "max_samples": "auto",
            "max_features": 1.0
        },
        {
            "config": 3,
            "n_estimators": 500,
            "max_samples": "auto",
            "max_features": 1.0
        },
        {
            "config": 4,
            "n_estimators": 200,
            "max_samples": 0.7,
            "max_features": 1.0
        },
        {
            "config": 5,
            "n_estimators": 200,
            "max_samples": 0.5,
            "max_features": 1.0
        },
        {
            "config": 6,
            "n_estimators": 200,
            "max_samples": "auto",
            "max_features": 0.7
        },
        {
            "config": 7,
            "n_estimators": 200,
            "max_samples": "auto",
            "max_features": 0.5
        },
        {
            "config": 8,
            "n_estimators": 500,
            "max_samples": 0.7,
            "max_features": 0.7
        },
        {
            "config": 9,
            "n_estimators": 500,
            "max_samples": 0.5,
            "max_features": 0.7
        }
    ]

    results = []

    # ---------------------------------------------------------
    # Test each configuration
    # ---------------------------------------------------------
    for cfg in configs:

        print("\n" + "-" * 70)
        print("Configuration", cfg["config"])
        print("n_estimators:", cfg["n_estimators"])
        print("max_samples:", cfg["max_samples"])
        print("max_features:", cfg["max_features"])

        model = IsolationForest(
            n_estimators=cfg["n_estimators"],
            max_samples=cfg["max_samples"],
            max_features=cfg["max_features"],
            random_state=SEED,
            n_jobs=-1
        )

        # Train model
        model.fit(train[cols])

        # Anomaly scores
        train_score = -model.score_samples(train[cols])
        test_score = -model.score_samples(test[cols])

        # Threshold based on 99th percentile of training data
        threshold = float(np.percentile(train_score, 99))

        # Predictions
        pred = (test_score >= threshold).astype(int)

        # -----------------------------------------------------
        # Existing evaluation function
        # -----------------------------------------------------
        m = evaluate(
            test["label"].values,
            pred,
            test_score
        )

        # -----------------------------------------------------
        # STANDARD ACCURACY
        # -----------------------------------------------------
        accuracy = calculate_accuracy(
            test["label"].values,
            pred
        )

        # -----------------------------------------------------
        # Confusion matrix values
        # -----------------------------------------------------
        tp = int(m["tp"])
        fp = int(m["fp"])
        fn = int(m["fn"])
        tn = int(m["tn"])

        # Alerts per day
        alerts_per_day = float(
            pred.sum() / test["date"].nunique()
        )

        # Store results
        result = {
            "config": cfg["config"],
            "n_estimators": cfg["n_estimators"],
            "max_samples": cfg["max_samples"],
            "max_features": cfg["max_features"],

            "accuracy": accuracy,
            "accuracy_percent": accuracy * 100,

            "precision": float(m["precision"]),
            "recall": float(m["recall"]),
            "f1": float(m["f1"]),
            "fpr": float(m["fpr"]),
            "roc_auc": float(m["roc_auc"]),
            "pr_auc": float(m["pr_auc"]),
            "alerts_per_day": alerts_per_day,

            "threshold": threshold,

            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        }

        results.append(result)

        # -----------------------------------------------------
        # PRINT RESULTS
        # -----------------------------------------------------
        print("\nRESULTS")
        print("-" * 40)

        print("Accuracy       :", round(accuracy * 100, 2), "%")
        print("Precision      :", round(m["precision"] * 100, 2), "%")
        print("Recall         :", round(m["recall"] * 100, 2), "%")
        print("F1-score       :", round(m["f1"] * 100, 2), "%")
        print("ROC-AUC        :", round(m["roc_auc"] * 100, 2), "%")
        print("PR-AUC         :", round(m["pr_auc"] * 100, 2), "%")
        print("FPR            :", round(m["fpr"] * 100, 2), "%")
        print("Alerts per day :", round(alerts_per_day, 4))

        print("\nConfusion Matrix")
        print("TP:", tp)
        print("FP:", fp)
        print("FN:", fn)
        print("TN:", tn)

    # ---------------------------------------------------------
    # Convert results to DataFrame
    # ---------------------------------------------------------
    results_df = pd.DataFrame(results)

    # ---------------------------------------------------------
    # BEST CONFIGURATIONS
    # ---------------------------------------------------------

    # Best according to ROC-AUC
    best_roc = results_df.loc[
        results_df["roc_auc"].idxmax()
    ]

    # Best according to standard Accuracy
    best_accuracy = results_df.loc[
        results_df["accuracy"].idxmax()
    ]

    # Best according to F1
    best_f1 = results_df.loc[
        results_df["f1"].idxmax()
    ]

    # Best according to PR-AUC
    best_pr = results_df.loc[
        results_df["pr_auc"].idxmax()
    ]

    print("\n")
    print("=" * 70)
    print("BEST RESULTS")
    print("=" * 70)

    print("\nBEST BY STANDARD ACCURACY")
    print("-" * 40)
    print("Configuration :", int(best_accuracy["config"]))
    print(
        "Accuracy      :",
        round(best_accuracy["accuracy_percent"], 2),
        "%"
    )
    print(
        "Precision     :",
        round(best_accuracy["precision"] * 100, 2),
        "%"
    )
    print(
        "Recall        :",
        round(best_accuracy["recall"] * 100, 2),
        "%"
    )
    print(
        "F1-score      :",
        round(best_accuracy["f1"] * 100, 2),
        "%"
    )

    print("\nBEST BY ROC-AUC")
    print("-" * 40)
    print("Configuration :", int(best_roc["config"]))
    print(
        "ROC-AUC       :",
        round(best_roc["roc_auc"] * 100, 2),
        "%"
    )
    print(
        "PR-AUC        :",
        round(best_roc["pr_auc"] * 100, 2),
        "%"
    )
    print(
        "Accuracy      :",
        round(best_roc["accuracy_percent"], 2),
        "%"
    )
    print(
        "Precision     :",
        round(best_roc["precision"] * 100, 2),
        "%"
    )
    print(
        "Recall        :",
        round(best_roc["recall"] * 100, 2),
        "%"
    )
    print(
        "F1-score      :",
        round(best_roc["f1"] * 100, 2),
        "%"
    )

    print("\nBEST BY F1-SCORE")
    print("-" * 40)
    print("Configuration :", int(best_f1["config"]))
    print(
        "F1-score      :",
        round(best_f1["f1"] * 100, 2),
        "%"
    )
    print(
        "Accuracy      :",
        round(best_f1["accuracy_percent"], 2),
        "%"
    )

    print("\nBEST BY PR-AUC")
    print("-" * 40)
    print("Configuration :", int(best_pr["config"]))
    print(
        "PR-AUC        :",
        round(best_pr["pr_auc"] * 100, 2),
        "%"
    )
    print(
        "Accuracy      :",
        round(best_pr["accuracy_percent"], 2),
        "%"
    )

    # ---------------------------------------------------------
    # Show all configurations
    # ---------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("ALL CONFIGURATIONS")
    print("=" * 70)

    display_columns = [
        "config",
        "accuracy_percent",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "fpr",
        "alerts_per_day"
    ]

    display_df = results_df[display_columns].copy()

    display_df["precision"] *= 100
    display_df["recall"] *= 100
    display_df["f1"] *= 100
    display_df["roc_auc"] *= 100
    display_df["pr_auc"] *= 100
    display_df["fpr"] *= 100

    print(
        display_df.round(4).to_string(index=False)
    )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    output_file = RESULTS / "step11_optimization_results.csv"

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
    # Save best result as JSON
    # ---------------------------------------------------------

    best_result = {
        "best_by_accuracy": best_accuracy.to_dict(),
        "best_by_roc_auc": best_roc.to_dict(),
        "best_by_f1": best_f1.to_dict(),
        "best_by_pr_auc": best_pr.to_dict()
    }

    json_file = RESULTS / "step11_best_results.json"

    with open(json_file, "w") as f:
        json.dump(
            best_result,
            f,
            indent=2,
            default=float
        )

    print(json_file)


main()