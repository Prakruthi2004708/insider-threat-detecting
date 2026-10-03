import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import shap
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, precision_recall_curve)
from common import *

SEED = 42
PLOTS = RESULTS / "plots"
PLOTS.mkdir(exist_ok=True)


def main():
    train, test, cutoff = load_split()
    cols_z = Z_COLS + ["weekend"]
    cols_raw = RAW_COLS

    # Train proposed model
    model = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1)
    model.fit(train[cols_z])
    tr_s = -model.score_samples(train[cols_z])
    t1 = float(np.percentile(tr_s, 99))
    t2 = float(np.percentile(tr_s, 95))

    test = test.sort_values(["user", "date"]).reset_index(drop=True)
    test["score"] = -model.score_samples(test[cols_z])

    # Persistence check K=2 M=1
    scores = test["score"].values
    confirmed = np.zeros(len(test), dtype=int)
    for user, idx in test.groupby("user").indices.items():
        s = scores[idx]
        for j in range(len(idx)):
            if s[j] < t1:
                continue
            nxt = s[j + 1: j + 3]
            if len(nxt) > 0 and (nxt >= t2).sum() >= 1:
                confirmed[idx[j]] = 1
    test["pred"] = confirmed
    y = test["label"].values

    # 1. Confusion matrix
    cm = confusion_matrix(y, confirmed, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Malicious"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix (Proposed + K=2 M=1)")
    fig.tight_layout()
    fig.savefig(PLOTS / "confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved: confusion_matrix.png")

    # 2. ROC curve comparison
    mA = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1)
    mA.fit(train[cols_raw])
    sA = -mA.score_samples(test[cols_raw])
    sB = test["score"].values

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, s in [("A: Baseline (global IF)", sA),
                     ("B: Proposed (per-user IF)", sB)]:
        fpr, tpr, _ = roc_curve(y, s)
        auc = np.trapezoid(tpr, fpr)
        ax.plot(fpr, tpr, label=label + " AUC=" + str(round(auc, 3)))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "roc_curve.png", dpi=150)
    plt.close()
    print("Saved: roc_curve.png")

    # 3. Precision-Recall curve
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, s in [("A: Baseline (global IF)", sA),
                     ("B: Proposed (per-user IF)", sB)]:
        prec, rec, _ = precision_recall_curve(y, s)
        ap = np.trapezoid(prec, rec)
        ax.plot(rec, prec, label=label + " AP=" + str(round(abs(ap), 3)))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "pr_curve.png", dpi=150)
    plt.close()
    print("Saved: pr_curve.png")

    # 4. Risk score distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sB[y == 0], bins=60, alpha=0.6, label="Normal days", color="steelblue")
    ax.hist(sB[y == 1], bins=60, alpha=0.8, label="Malicious days", color="crimson")
    ax.axvline(t1, color="black", linestyle="--", linewidth=1,
               label="Alert threshold (99th pct)")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.set_title("Risk Score Distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "score_distribution.png", dpi=150)
    plt.close()
    print("Saved: score_distribution.png")

    # 5. Ablation bar chart
    ablation = pd.DataFrame([
        {"Model": "A: Baseline", "Precision": 0.0137, "Recall": 0.0072,
         "F1": 0.0095, "ROC-AUC": 0.7995, "PR-AUC": 0.0235},
        {"Model": "B: Proposed", "Precision": 0.0356, "Recall": 0.0237,
         "F1": 0.0284, "ROC-AUC": 0.8368, "PR-AUC": 0.0321},
        {"Model": "C: Proposed+Check", "Precision": 0.0249, "Recall": 0.0113,
         "F1": 0.0156, "ROC-AUC": 0.8368, "PR-AUC": 0.0321},
        {"Model": "D: Supervised RF", "Precision": 0.7858, "Recall": 0.2794,
         "F1": 0.4121, "ROC-AUC": 0.9669, "PR-AUC": 0.4988},
    ])
    metrics = ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    x = np.arange(len(metrics))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (_, row) in enumerate(ablation.iterrows()):
        ax.bar(x + i * width, [row[m] for m in metrics],
               width, label=row["Model"])
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("Ablation: Model Comparison (mean across 5 seeds)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PLOTS / "ablation_bar.png", dpi=150)
    plt.close()
    print("Saved: ablation_bar.png")

    # 6. Second-check trade-off chart
    tradeoff = pd.DataFrame([
        {"Setting": "Warning only", "Alerts/day": 1.7192,
         "Benign flagged": 41.6, "Insiders caught": 7.6},
        {"Setting": "K=2 M=1", "Alerts/day": 1.1682,
         "Benign flagged": 21.0, "Insiders caught": 4.0},
    ])
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    x = np.arange(len(tradeoff))
    ax1.bar(x - 0.2, tradeoff["Benign flagged"], 0.35,
            label="Benign users flagged", color="steelblue", alpha=0.8)
    ax1.bar(x + 0.2, tradeoff["Alerts/day"] * 10, 0.35,
            label="Alerts/day (x10)", color="orange", alpha=0.8)
    ax2.plot(x, tradeoff["Insiders caught"], "ro-",
             linewidth=2, label="Insiders caught (mean)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tradeoff["Setting"])
    ax1.set_ylabel("Count")
    ax2.set_ylabel("Insiders caught")
    ax1.set_title("Second-Check Trade-off (mean across 5 seeds)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "tradeoff_chart.png", dpi=150)
    plt.close()
    print("Saved: tradeoff_chart.png")

    # 7. SHAP explanations on Random Forest
    print("")
    print("Running SHAP (this takes a minute)...")
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED,
                                class_weight="balanced", n_jobs=-1)
    rf.fit(train[cols_z], train["label"])
    sample = test[cols_z].sample(n=min(500, len(test)), random_state=SEED)
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(sample)
    if isinstance(shap_vals, list):
        sv = shap_vals[1]
    else:
        sv = shap_vals

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(sv, sample, feature_names=cols_z,
                      plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Random Forest, malicious class)")
    plt.tight_layout()
    plt.savefig(PLOTS / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: shap_importance.png")

    # 8. Alert explanation for one flagged insider
    flagged = test[(test["pred"] == 1) & (test["label"] == 1)]
    if len(flagged) > 0:
        row = flagged.iloc[0]
        vals = {c: float(row[c]) for c in cols_z}
        top = sorted(vals.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
        names = [t[0] for t in top]
        values = [t[1] for t in top]
        colors = ["crimson" if v > 0 else "steelblue" for v in values]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(names, values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Z-score (deviation from user own baseline)")
        ax.set_title("Explanation: Top deviating features for flagged insider\n"
                     "User: " + str(row["user"]) +
                     "  Date: " + str(row["date"])[:10])
        fig.tight_layout()
        fig.savefig(PLOTS / "alert_explanation.png", dpi=150)
        plt.close()
        print("Saved: alert_explanation.png")
    else:
        print("No confirmed insider days in test for explanation plot.")

    print("")
    print("All plots saved in:", PLOTS)


main()