import json
from sklearn.ensemble import IsolationForest
from common import *


def main():
    train, test, cutoff = load_split()
    print("Split date:", pd.Timestamp(cutoff).date())
    print("Train rows:", len(train), "| malicious:", int(train["label"].sum()))
    print("Test rows:", len(test), "| malicious:", int(test["label"].sum()),
          "| malicious users in test:", test.loc[test["label"] == 1, "user"].nunique())

    model = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1)
    model.fit(train[RAW_COLS])

    train_score = -model.score_samples(train[RAW_COLS])
    test_score = -model.score_samples(test[RAW_COLS])

    threshold = np.percentile(train_score, 99)
    pred = (test_score >= threshold).astype(int)

    m = evaluate(test["label"].values, pred, test_score)
    m["alerts_per_day"] = float(pred.sum() / test["date"].nunique())

    print("")
    print("BASELINE: global Isolation Forest on raw counts")
    for k, v in m.items():
        print(k, ":", round(v, 4) if isinstance(v, float) else v)

    test["baseline_score"] = test_score
    test["baseline_pred"] = pred
    test.to_parquet(RESULTS / "baseline_scores.parquet", index=False)
    with open(RESULTS / "baseline_metrics.json", "w") as f:
        json.dump(m, f, indent=2)
    print("Saved results in:", RESULTS)


main()