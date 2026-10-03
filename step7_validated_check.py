import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from common import *

MIN_KEEP = 0.75


def build_rules():
    rules = [{"k": 0, "m": 1, "z": None, "name": "warning only"}]
    for k in [2, 3, 5]:
        for m in [1, 2]:
            if m <= k:
                rules.append({"k": k, "m": m, "z": None,
                              "name": "persist K=" + str(k) + " M=" + str(m)})
    for z in [1.0, 1.5, 2.0]:
        rules.append({"k": 0, "m": 1, "z": z, "name": "cross Z=" + str(z)})
    for k, m in [(3, 1), (5, 2)]:
        for z in [1.0, 1.5, 2.0]:
            rules.append({"k": k, "m": m, "z": z,
                          "name": "persist K=" + str(k) + " M=" + str(m) + " + cross Z=" + str(z)})
    return rules


def evidence_count(df, z):
    e_data = (df["z_n_filecopy_usb_day"] >= z) | (df["z_n_filecopy"] >= z) | (df["z_n_new_ext"] >= z)
    e_time = ((df["z_n_after_hours"] >= z) | (df["z_n_after_connect"] >= z)
              | (df["z_n_after_file"] >= z) | (df["z_n_weekend_events"] >= z))
    e_dev = (df["z_n_connect"] >= z) | (df["z_n_new_pc"] >= z) | (df["z_n_pcs"] >= z)
    return e_data.astype(int) + e_time.astype(int) + e_dev.astype(int)


def apply_rule(df, t1, t2, rule):
    scores = df["score"].values
    dates = df["date"].values
    if rule["z"] is None:
        cross = np.ones(len(df), dtype=bool)
    else:
        cross = (evidence_count(df, rule["z"]) >= 2).values
    confirmed = np.zeros(len(df), dtype=int)
    decision = np.full(len(df), np.datetime64("NaT"), dtype="datetime64[ns]")
    for user, idx in df.groupby("user").indices.items():
        s = scores[idx]
        d = dates[idx]
        for j in range(len(idx)):
            if s[j] < t1 or not cross[idx[j]]:
                continue
            if rule["k"] == 0:
                confirmed[idx[j]] = 1
                decision[idx[j]] = d[j]
                continue
            nxt = s[j + 1: j + 1 + rule["k"]]
            if len(nxt) > 0 and (nxt >= t2).sum() >= rule["m"]:
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
        "insiders_caught": caught,
        "n_insiders": len(insiders),
        "benign_flagged": benign_flagged,
        "n_benign": len(benign_users),
        "avg_delay_days": float(np.mean(delays)) if delays else float("nan"),
    }


def show(res):
    d = res.copy()
    d["insiders_caught"] = d["insiders_caught"].astype(str) + "/" + d["n_insiders"].astype(str)
    d["benign_flagged"] = d["benign_flagged"].astype(str) + "/" + d["n_benign"].astype(str)
    d = d[["setting", "alerts", "tp", "fp", "precision", "recall",
           "alerts_per_day", "insiders_caught", "benign_flagged", "avg_delay_days"]]
    print(d.round(4).to_string(index=False))


def main():
    train, test, cutoff = load_split()
    cols = Z_COLS + ["weekend"]
    rules = build_rules()

    tdates = np.sort(train["date"].unique())
    vcut = tdates[int(len(tdates) * 0.75)]
    fit = train[train["date"] < vcut].copy()
    val = train[train["date"] >= vcut].copy()
    print("Fit rows:", len(fit), "| Validation rows:", len(val),
          "| malicious users in validation:", val.loc[val["label"] == 1, "user"].nunique())

    model = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1)
    model.fit(fit[cols])
    fit_scores = -model.score_samples(fit[cols])
    t1 = float(np.percentile(fit_scores, 99))
    t2 = float(np.percentile(fit_scores, 95))

    val = val.sort_values(["user", "date"]).reset_index(drop=True)
    val["score"] = -model.score_samples(val[cols])

    rows = []
    for r in rules:
        c, d = apply_rule(val, t1, t2, r)
        rows.append(summarize(val, c, d, r["name"]))
    vres = pd.DataFrame(rows)
    vres.to_csv(RESULTS / "validation_rules.csv", index=False)

    base = vres.iloc[0]
    ok = vres.iloc[1:]
    ok = ok[ok["insiders_caught"] >= MIN_KEEP * base["insiders_caught"]]
    ok = ok[ok["benign_flagged"] < base["benign_flagged"]]

    print("")
    print("VALIDATION (warning only, then the 5 best candidates):")
    if len(ok) == 0:
        chosen = rules[0]
        show(vres.iloc[[0]])
        print("No rule beat warning-only on validation. Keeping warning only.")
    else:
        ok = ok.sort_values(["benign_flagged", "insiders_caught", "avg_delay_days"],
                            ascending=[True, False, True])
        show(pd.concat([vres.iloc[[0]], ok.head(5)]))
        chosen_name = ok.iloc[0]["setting"]
        chosen = [r for r in rules if r["name"] == chosen_name][0]
    print("")
    print("CHOSEN RULE (selected on validation only):", chosen["name"])

    final = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1)
    final.fit(train[cols])
    tr_scores = -final.score_samples(train[cols])
    T1 = float(np.percentile(tr_scores, 99))
    T2 = float(np.percentile(tr_scores, 95))

    test = test.sort_values(["user", "date"]).reset_index(drop=True)
    test["score"] = -final.score_samples(test[cols])

    to_run = [rules[0]]
    if chosen["name"] != rules[0]["name"]:
        to_run.append(chosen)
    rows = []
    for r in to_run:
        c, d = apply_rule(test, T1, T2, r)
        rows.append(summarize(test, c, d, r["name"]))
    tres = pd.DataFrame(rows)
    tres.to_csv(RESULTS / "final_test_results.csv", index=False)

    print("")
    print("FINAL TEST RESULT (run once):")
    show(tres)
    print("Saved: validation_rules.csv and final_test_results.csv in", RESULTS)


main()