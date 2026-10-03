import json
from sklearn.ensemble import IsolationForest
from common import *


def main():
    train, test, cutoff = load_split()
    cols = Z_COLS + ["weekend"]

    model = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1)
    model.fit(train[cols])

    train_score = -model.score_samples(train[cols])
    test_score = -model.score_samples(test[cols])

    threshold = float(np.percentile(train_score, 99))
    pred = (test_score >= threshold).astype(int)

    m = evaluate(test["label"].values, pred, test_score)
    m["alerts_per_day"] = float(pred.sum() / test["date"].nunique())
    m["threshold"] = threshold

    with open(RESULTS / "baseline_metrics.json") as f:
        b = json.load(f)

    print("PROPOSED: Isolation Forest on per-user (z) features")
    print("")
    print("metric".ljust(16), "baseline".ljust(10), "proposed")
    for k in ["precision", "recall", "f1", "fpr", "roc_auc", "pr_auc", "alerts_per_day"]:
        print(k.ljust(16), str(round(b[k], 4)).ljust(10), round(m[k], 4))
    print("")
    print("tp:", m["tp"], "| fp:", m["fp"], "| fn:", m["fn"], "| tn:", m["tn"])

    test["proposed_score"] = test_score
    test["proposed_pred"] = pred
    test.to_parquet(RESULTS / "proposed_scores.parquet", index=False)
    with open(RESULTS / "proposed_metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    print("Saved results in:", RESULTS)


main()