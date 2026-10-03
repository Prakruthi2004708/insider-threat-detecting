from sklearn.ensemble import IsolationForest
from common import *

K_LIST = [2, 3, 5]
M_LIST = [1, 2]


def apply_check(df, t1, t2, k, m):
    scores = df["score"].values
    dates = df["date"].values
    confirmed = np.zeros(len(df), dtype=int)
    decision = np.full(len(df), np.datetime64("NaT"), dtype="datetime64[ns]")
    for user, idx in df.groupby("user").indices.items():
        s = scores[idx]
        d = dates[idx]
        for j in range(len(idx)):
            if s[j] < t1:
                continue
            if k == 0:
                confirmed[idx[j]] = 1
                decision[idx[j]] = d[j]
                continue
            nxt = s[j + 1: j + 1 + k]
            if len(nxt) > 0 and (nxt >= t2).sum() >= m:
                confirmed[idx[j]] = 1
                decision[idx[j]] = d[j + len(nxt)]
    return confirmed, decision


def summarize(df, confirmed, decision, name):
    y = df["label"].values
    c = confirmed
    tp = int(((c == 1) & (y == 1)).sum())
    fp = int(((c == 1) & (y == 0)).sum())
    fn = int(((c == 0) & (y == 1)).sum())

    work = df[["user", "date", "label"]].copy()
    work["confirmed"] = c
    work["decision"] = pd.to_datetime(decision)

    insiders = work.loc[work["label"] == 1, "user"].unique()
    ins_set = set(insiders)
    caught = 0
    delays = []
    for u in insiders:
        sub = work[work["user"] == u]
        first_mal = sub.loc[sub["label"] == 1, "date"].min()
        hit = sub[(sub["confirmed"] == 1) & (sub["label"] == 1)]
        if len(hit) > 0:
            caught += 1
            delays.append((hit["decision"].min() - first_mal).days)

    flagged = work.groupby("user")["confirmed"].max()
    benign_users = [u for u in flagged.index if u not in ins_set]
    benign_flagged = int(flagged[benign_users].sum())

    return {
        "setting": name,
        "alerts": int(c.sum()),
        "tp": tp,
        "fp": fp,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "alerts_per_day": c.sum() / df["date"].nunique(),
        "insiders_caught": str(caught) + "/" + str(len(insiders)),
        "benign_flagged": str(benign_flagged) + "/" + str(len(benign_users)),
        "avg_delay_days": float(np.mean(delays)) if delays else float("nan"),
    }


def main():
    train, test, cutoff = load_split()
    cols = Z_COLS + ["weekend"]

    model = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1)
    model.fit(train[cols])
    train_score = -model.score_samples(train[cols])
    t1 = float(np.percentile(train_score, 99))
    t2 = float(np.percentile(train_score, 95))

    test = test.sort_values(["user", "date"]).reset_index(drop=True)
    test["score"] = -model.score_samples(test[cols])

    rows = []
    c, d = apply_check(test, t1, t2, 0, 1)
    rows.append(summarize(test, c, d, "warning only"))
    for k in K_LIST:
        for m in M_LIST:
            if m > k:
                continue
            c, d = apply_check(test, t1, t2, k, m)
            rows.append(summarize(test, c, d, "K=" + str(k) + " M=" + str(m)))

    res = pd.DataFrame(rows)
    print(res.round(4).to_string(index=False))
    res.to_csv(RESULTS / "second_check_results.csv", index=False)
    print("Saved:", RESULTS / "second_check_results.csv")


main()