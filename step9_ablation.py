import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from common import *

SEEDS = [42, 7, 123, 2024, 999]


def score_det(y_true, y_pred, y_score):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return {
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "alerts_per_day": int(y_pred.sum()),
    }


def run_seed(seed, train, test):
    cols_raw = RAW_COLS
    cols_z = Z_COLS + ["weekend"]
    rows = []

    # A: Baseline global IF on raw features
    m = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1)
    m.fit(train[cols_raw])
    tr_s = -m.score_samples(train[cols_raw])
    t1 = np.percentile(tr_s, 99)
    te_s = -m.score_samples(test[cols_raw])
    pred = (te_s >= t1).astype(int)
    r = score_det(test["label"].values, pred, te_s)
    r["model"] = "A: Baseline (global IF, raw)"
    r["seed"] = seed
    rows.append(r)

    # B: Proposed per-user IF on z features
    m2 = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1)
    m2.fit(train[cols_z])
    tr_s2 = -m2.score_samples(train[cols_z])
    t1b = np.percentile(tr_s2, 99)
    t2b = np.percentile(tr_s2, 95)
    te_s2 = -m2.score_samples(test[cols_z])
    pred2 = (te_s2 >= t1b).astype(int)
    r2 = score_det(test["label"].values, pred2, te_s2)
    r2["model"] = "B: Proposed (per-user IF + new features)"
    r2["seed"] = seed
    rows.append(r2)

    # C: Proposed + persistence check K=2 M=1
    ts = test.sort_values(["user", "date"]).reset_index(drop=True)
    ts["score"] = -m2.score_samples(ts[cols_z])
    scores = ts["score"].values
    confirmed = np.zeros(len(ts), dtype=int)
    for user, idx in ts.groupby("user").indices.items():
        s = scores[idx]
        for j in range(len(idx)):
            if s[j] < t1b:
                continue
            nxt = s[j + 1: j + 3]
            if len(nxt) > 0 and (nxt >= t2b).sum() >= 1:
                confirmed[idx[j]] = 1
    r3 = score_det(ts["label"].values, confirmed, ts["score"].values)
    r3["model"] = "C: Proposed + persist K=2 M=1"
    r3["seed"] = seed
    rows.append(r3)

    # D: Supervised Random Forest (upper bound, uses labels)
    rf = RandomForestClassifier(n_estimators=200, random_state=seed,
                                class_weight="balanced", n_jobs=-1)
    rf.fit(train[cols_z], train["label"])
    rf_prob = rf.predict_proba(test[cols_z])[:, 1]
    rf_pred = rf.predict(test[cols_z])
    r4 = score_det(test["label"].values, rf_pred, rf_prob)
    r4["model"] = "D: Supervised RF (upper bound)"
    r4["seed"] = seed
    rows.append(r4)

    return rows


def main():
    train, test, cutoff = load_split()
    print("Test malicious users:", test.loc[test["label"] == 1, "user"].nunique())
    print("Running", len(SEEDS), "seeds...")

    all_rows = []
    for seed in SEEDS:
        all_rows.extend(run_seed(seed, train, test))
        print("Seed", seed, "done.")

    res = pd.DataFrame(all_rows)
    res.to_csv(RESULTS / "ablation_raw.csv", index=False)

    metrics = ["precision", "recall", "f1", "fpr", "roc_auc", "pr_auc"]
    print("")
    print("ABLATION TABLE (mean ± std across 5 seeds):")
    print("-" * 80)
    for model in res["model"].unique():
        sub = res[res["model"] == model]
        print(model)
        for m in metrics:
            print("  " + m + ":", round(sub[m].mean(), 4), "±", round(sub[m].std(), 4))
        print("")

    summary = res.groupby("model")[metrics].agg(["mean", "std"]).round(4)
    summary.to_csv(RESULTS / "ablation_summary.csv")
    print("Saved: ablation_raw.csv and ablation_summary.csv in", RESULTS)


main()