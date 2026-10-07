# ============================================================
# STEP 16 - CATBOOST MODEL + MODEL COMPARISON
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\useer\Documents\insider_threat_project"

FEATURE_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "features.parquet"
)

RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = 42


# ============================================================
# HEADER
# ============================================================

print("=" * 90)
print("STEP 16 - CATBOOST INSIDER THREAT DETECTION")
print("=" * 90)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading feature dataset...")

df = pd.read_parquet(FEATURE_FILE)

print(f"Total records: {len(df):,}")


# ============================================================
# FEATURE SELECTION
# ============================================================

exclude_columns = [
    "label",
    "scenario",
    "date",
    "user",
    "id",
    "index"
]

feature_columns = [
    col for col in df.columns
    if col not in exclude_columns
    and pd.api.types.is_numeric_dtype(df[col])
]

print(f"\nNumber of features: {len(feature_columns)}")

print("\nFeatures used:")

for i, feature in enumerate(feature_columns, 1):
    print(f"{i:2d}. {feature}")


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)


# ============================================================
# 80/20 CHRONOLOGICAL TRAIN-TEST SPLIT
# ============================================================

split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

X_train = train_df[feature_columns]
y_train = train_df["label"].astype(int)

X_test = test_df[feature_columns]
y_test = test_df["label"].astype(int)

print("\n" + "-" * 90)
print("CHRONOLOGICAL TRAIN-TEST SPLIT")
print("-" * 90)

print(f"Training records : {len(train_df):,}")
print(f"Testing records  : {len(test_df):,}")

print(
    f"Training period  : "
    f"{train_df['date'].min().date()} to {train_df['date'].max().date()}"
)

print(
    f"Testing period   : "
    f"{test_df['date'].min().date()} to {test_df['date'].max().date()}"
)

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting class distribution:")
print(y_test.value_counts().sort_index())


# ============================================================
# CREATE VALIDATION SET FROM TRAINING DATA
# ============================================================

validation_split = int(len(X_train) * 0.80)

X_fit = X_train.iloc[:validation_split]
y_fit = y_train.iloc[:validation_split]

X_val = X_train.iloc[validation_split:]
y_val = y_train.iloc[validation_split:]

print("\n" + "-" * 90)
print("VALIDATION SPLIT")
print("-" * 90)

print(f"Training fit records : {len(X_fit):,}")
print(f"Validation records   : {len(X_val):,}")


# ============================================================
# CATBOOST MODEL
# ============================================================

print("\n" + "=" * 90)
print("TRAINING CATBOOST")
print("=" * 90)

catboost_model = CatBoostClassifier(
    iterations=1000,
    depth=6,
    learning_rate=0.03,
    loss_function="Logloss",
    eval_metric="PRAUC",
    random_seed=SEED,
    verbose=100,
    early_stopping_rounds=80,
    thread_count=-1
)

catboost_model.fit(
    X_fit,
    y_fit,
    eval_set=(X_val, y_val),
    use_best_model=True
)


# ============================================================
# BEST ITERATION
# ============================================================

best_iteration = catboost_model.get_best_iteration()

if best_iteration is None or best_iteration < 0:
    best_iteration = 999

print("\nBest iteration:", best_iteration + 1)


# ============================================================
# VALIDATION THRESHOLD SEARCH
# ============================================================

print("\n" + "-" * 90)
print("VALIDATION THRESHOLD SEARCH")
print("-" * 90)

val_probabilities = catboost_model.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.01, 1.00, 0.01)

best_threshold = 0.50
best_f1 = -1

for threshold in thresholds:

    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Best validation threshold: {best_threshold:.2f}")
print(f"Validation F1: {best_f1 * 100:.2f}%")


# ============================================================
# FINAL CATBOOST REFIT
# ============================================================

final_iterations = best_iteration + 1

print("\n" + "=" * 90)
print("FINAL CATBOOST REFIT")
print("=" * 90)

print(f"Iterations used: {final_iterations}")

final_model = CatBoostClassifier(
    iterations=final_iterations,
    depth=6,
    learning_rate=0.03,
    loss_function="Logloss",
    random_seed=SEED,
    verbose=100,
    thread_count=-1
)

final_model.fit(
    X_train,
    y_train
)


# ============================================================
# TEST PREDICTION
# ============================================================

test_probabilities = final_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)


# ============================================================
# METRICS FUNCTION
# ============================================================

def calculate_metrics(y_true, y_pred, probabilities, dates):

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    fpr = fp / (fp + tn)

    number_of_days = (
        pd.to_datetime(dates).dt.date.nunique()
    )

    alerts_per_day = (
        (tp + fp) / number_of_days
        if number_of_days > 0
        else 0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "alerts_per_day": alerts_per_day,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


# ============================================================
# CATBOOST TEST METRICS
# ============================================================

catboost_metrics = calculate_metrics(
    y_test,
    test_predictions,
    test_probabilities,
    test_df["date"]
)


# ============================================================
# DISPLAY CATBOOST RESULTS
# ============================================================

print("\n" + "=" * 90)
print("CATBOOST FINAL TEST RESULTS")
print("=" * 90)

print(
    f"Accuracy    : "
    f"{catboost_metrics['accuracy'] * 100:.2f}%"
)

print(
    f"Precision   : "
    f"{catboost_metrics['precision'] * 100:.2f}%"
)

print(
    f"Recall      : "
    f"{catboost_metrics['recall'] * 100:.2f}%"
)

print(
    f"F1 Score    : "
    f"{catboost_metrics['f1'] * 100:.2f}%"
)

print(
    f"ROC-AUC     : "
    f"{catboost_metrics['roc_auc'] * 100:.2f}%"
)

print(
    f"PR-AUC      : "
    f"{catboost_metrics['pr_auc'] * 100:.2f}%"
)

print(
    f"FPR         : "
    f"{catboost_metrics['fpr'] * 100:.2f}%"
)

print(
    f"Alerts/day  : "
    f"{catboost_metrics['alerts_per_day']:.4f}"
)

print("\nConfusion Matrix:")

print(
    f"TN = {catboost_metrics['tn']:,}"
)

print(
    f"FP = {catboost_metrics['fp']:,}"
)

print(
    f"FN = {catboost_metrics['fn']:,}"
)

print(
    f"TP = {catboost_metrics['tp']:,}"
)


# ============================================================
# SAVE CATBOOST SCORES
# ============================================================

score_output = test_df[
    ["date", "user", "label"]
].copy()

score_output["probability"] = test_probabilities

score_output["prediction"] = test_predictions

score_output.to_parquet(
    os.path.join(
        RESULTS_DIR,
        "step16_catboost_scores.parquet"
    ),
    index=False
)


# ============================================================
# SAVE METRICS JSON
# ============================================================

metrics_output = {
    "model": "CatBoost",
    "threshold": float(best_threshold),
    "iterations": int(final_iterations),
    "depth": 6,
    "learning_rate": 0.03,
    "features": feature_columns,
    **catboost_metrics
}

with open(
    os.path.join(
        RESULTS_DIR,
        "step16_catboost_metrics.json"
    ),
    "w"
) as f:

    json.dump(
        metrics_output,
        f,
        indent=4
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 90)
print("CATBOOST FEATURE IMPORTANCE")
print("=" * 90)

importance_values = final_model.get_feature_importance()

importance_df = pd.DataFrame({
    "feature": feature_columns,
    "importance": importance_values
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print("\nTop features:")

print(
    importance_df.head(15).to_string(
        index=False
    )
)

importance_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "step16_catboost_feature_importance.csv"
    ),
    index=False
)


# ============================================================
# FEATURE IMPORTANCE GRAPH
# ============================================================

top_features = importance_df.head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["feature"][::-1],
    top_features["importance"][::-1]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")

plt.title(
    "CatBoost - Top 15 Feature Importance"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "step16_catboost_feature_importance.png"
    ),
    dpi=300
)

plt.close()

print("\nFeature importance graph saved.")


# ============================================================
# CATBOOST METRIC GRAPH
# ============================================================

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC",
    "PR-AUC"
]

metric_values = [
    catboost_metrics["accuracy"] * 100,
    catboost_metrics["precision"] * 100,
    catboost_metrics["recall"] * 100,
    catboost_metrics["f1"] * 100,
    catboost_metrics["roc_auc"] * 100,
    catboost_metrics["pr_auc"] * 100
]

plt.figure(figsize=(10, 6))

plt.bar(
    metric_names,
    metric_values
)

plt.ylabel("Percentage (%)")

plt.title(
    "CatBoost Performance Metrics"
)

plt.ylim(
    0,
    max(metric_values) + 10
)

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "step16_catboost_metrics.png"
    ),
    dpi=300
)

plt.close()

print("Metric graph saved.")


# ============================================================
# MODEL COMPARISON
# ============================================================

print("\n")
print("=" * 90)
print("MODEL COMPARISON - ISOLATION FOREST vs RANDOM FOREST vs CATBOOST")
print("=" * 90)


# ------------------------------------------------------------
# RESULTS FROM PREVIOUS EXPERIMENTS
# ------------------------------------------------------------

comparison = pd.DataFrame({

    "Model": [
        "Isolation Forest",
        "Random Forest",
        "CatBoost"
    ],

    "Accuracy": [
        94.89,
        98.73,
        catboost_metrics["accuracy"] * 100
    ],

    "Precision": [
        1.97,
        38.93,
        catboost_metrics["precision"] * 100
    ],

    "Recall": [
        8.76,
        54.38,
        catboost_metrics["recall"] * 100
    ],

    "F1": [
        3.22,
        45.38,
        catboost_metrics["f1"] * 100
    ],

    "ROC-AUC": [
        82.29,
        96.98,
        catboost_metrics["roc_auc"] * 100
    ],

    "PR-AUC": [
        2.54,
        45.43,
        catboost_metrics["pr_auc"] * 100
    ],

    "FPR": [
        4.27,
        0.84,
        catboost_metrics["fpr"] * 100
    ],

    "Alerts/day": [
        11.44,
        3.59,
        catboost_metrics["alerts_per_day"]
    ]
})


# ============================================================
# DISPLAY COMPARISON TABLE
# ============================================================

print("\nMODEL COMPARISON TABLE\n")

print(
    comparison.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.2f}".format,
            "Precision": "{:.2f}".format,
            "Recall": "{:.2f}".format,
            "F1": "{:.2f}".format,
            "ROC-AUC": "{:.2f}".format,
            "PR-AUC": "{:.2f}".format,
            "FPR": "{:.2f}".format,
            "Alerts/day": "{:.2f}".format
        }
    )
)


# ============================================================
# SAVE COMPARISON CSV
# ============================================================

comparison.to_csv(
    os.path.join(
        RESULTS_DIR,
        "step16_model_comparison.csv"
    ),
    index=False
)

print(
    "\nComparison CSV saved:"
    "\nresults\\step16_model_comparison.csv"
)


# ============================================================
# COMPLETE MODEL COMPARISON GRAPH
# ============================================================

metrics_for_graph = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC",
    "PR-AUC"
]

x = np.arange(
    len(metrics_for_graph)
)

width = 0.25

plt.figure(figsize=(12, 7))

for i, model in enumerate(comparison["Model"]):

    values = comparison.loc[
        comparison["Model"] == model,
        metrics_for_graph
    ].values[0]

    plt.bar(
        x + (i - 1) * width,
        values,
        width,
        label=model
    )

plt.xticks(
    x,
    metrics_for_graph,
    rotation=20
)

plt.ylabel("Percentage (%)")

plt.title(
    "Model Performance Comparison"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "step16_model_comparison.png"
    ),
    dpi=300
)

plt.close()

print("Model comparison graph saved.")


# ============================================================
# ACCURACY COMPARISON
# ============================================================

plt.figure(figsize=(8, 6))

plt.bar(
    comparison["Model"],
    comparison["Accuracy"]
)

plt.ylabel("Accuracy (%)")

plt.title(
    "Accuracy Comparison"
)

plt.ylim(
    0,
    100
)

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "step16_accuracy_comparison.png"
    ),
    dpi=300
)

plt.close()

print("Accuracy comparison graph saved.")


# ============================================================
# PRECISION, RECALL AND F1 COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

x = np.arange(
    len(comparison["Model"])
)

width = 0.25

plt.bar(
    x - width,
    comparison["Precision"],
    width,
    label="Precision"
)

plt.bar(
    x,
    comparison["Recall"],
    width,
    label="Recall"
)

plt.bar(
    x + width,
    comparison["F1"],
    width,
    label="F1"
)

plt.xticks(
    x,
    comparison["Model"]
)

plt.ylabel("Percentage (%)")

plt.title(
    "Precision, Recall and F1 Comparison"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "step16_precision_recall_f1_comparison.png"
    ),
    dpi=300
)

plt.close()

print(
    "Precision/Recall/F1 comparison graph saved."
)


# ============================================================
# FPR AND ALERTS/DAY COMPARISON
# ============================================================

fig, ax1 = plt.subplots(
    figsize=(10, 6)
)

x = np.arange(
    len(comparison["Model"])
)

width = 0.35

ax1.bar(
    x - width / 2,
    comparison["FPR"],
    width,
    label="FPR (%)"
)

ax1.set_ylabel(
    "False Positive Rate (%)"
)

ax1.set_xticks(x)

ax1.set_xticklabels(
    comparison["Model"]
)

ax1.set_title(
    "False Positive Rate and Alerts per Day"
)

ax2 = ax1.twinx()

ax2.bar(
    x + width / 2,
    comparison["Alerts/day"],
    width,
    label="Alerts/day"
)

ax2.set_ylabel(
    "Alerts per Day"
)

fig.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "step16_fpr_alerts_comparison.png"
    ),
    dpi=300
)

plt.close()

print(
    "FPR/Alerts comparison graph saved."
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 90)
print("STEP 16 COMPLETED SUCCESSFULLY")
print("=" * 90)

print("\nFinal CatBoost Results:")

print(
    f"Accuracy   : {catboost_metrics['accuracy'] * 100:.2f}%"
)

print(
    f"Precision  : {catboost_metrics['precision'] * 100:.2f}%"
)

print(
    f"Recall     : {catboost_metrics['recall'] * 100:.2f}%"
)

print(
    f"F1 Score   : {catboost_metrics['f1'] * 100:.2f}%"
)

print(
    f"ROC-AUC    : {catboost_metrics['roc_auc'] * 100:.2f}%"
)

print(
    f"PR-AUC     : {catboost_metrics['pr_auc'] * 100:.2f}%"
)

print(
    f"FPR        : {catboost_metrics['fpr'] * 100:.2f}%"
)

print(
    f"Alerts/day : {catboost_metrics['alerts_per_day']:.2f}"
)

print("\nFiles generated in results folder:")

print("1. step16_catboost_scores.parquet")
print("2. step16_catboost_metrics.json")
print("3. step16_catboost_feature_importance.csv")
print("4. step16_catboost_feature_importance.png")
print("5. step16_catboost_metrics.png")
print("6. step16_model_comparison.csv")
print("7. step16_model_comparison.png")
print("8. step16_accuracy_comparison.png")
print("9. step16_precision_recall_f1_comparison.png")
print("10. step16_fpr_alerts_comparison.png")

print("\n" + "=" * 90)