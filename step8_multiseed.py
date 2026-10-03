import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from common import *

SEEDS = [42, 7, 123, 2024, 999]

with open(RESULTS / "final_test_results.csv") as f:
    tres = pd.read_csv(f)
chosen_name = tres.iloc[-1]["setting"]
print("Chosen rule from Step 7:", chosen_name)

def build_rules(chosen_name):
    base = {"k": 0, "m": 1, "z": None, "name": "warning only"}
    if chosen_name == "warning only":
        return [base]
    parts = chosen_name.split()
    k, m, z = 0, 1, None
    for i, p in enumerate(parts):
        if p.startswith("K="):
            k = int(p[2:])
        if p.startswith("M="):
            m = int(p[2:])
        if p.startswith("Z="):
            z = float(p[2:])
    return [base, {"k": k, "m": m, "z": z, "name": chosen_name}]


def evidence_count(df, z):
    e_data = ((df["z_n_filecopy_usb_day"] >= z) |
              (df["z_n_filecopy"] >= z) |
              (df["z_n_new_ext"] >= z))
    e_time = ((df["z_n_after_hours"] >= z) |
              (df["z_n_after_connect"] >= z) |
              (df["z_n_after_file"] >= z) |
              (df["z_n_weekend_events"] >= z))
    e_dev = ((df["z_n_connect"] >= z) |
             (df["z_n_new_pc"] >= z) |
             (df["z_n_pcs"] >= z))
    return e_data.astype(int) + e_time.astype(int) + e_dev.astype(int)


def apply_rule(df, t1, t2, rule):
    scores = df["score"].values
    dates = df["date"].values
    if rule["z"] is None:
        cross = np.ones(len(df), dtype=bool)
    else:
        cross = (evidence_count(df, rule["z"]) >= 2).values
    confirmed = np.zeros(len(df), dtype=int)
    for user, idx in df.groupby("user").indices.items():
        s = scores[idx]
        for j in range(len(idx)):
            if s[j] < t1 or not cross[idx[j]]:
                continue
            if rule["k"] == 0:
                confirmed[idx[j]] = 1
                continue
            nxt = s[j + 1: j + 1 + rule["k"]]
            if len(nxt) > 0 and (nxt >= t2).sum() >= rule["m"]:
                confirmed[idx[j]] = 1
    return confirmed


def score_rule(df, confirmed, rule_name):
    y = df["label"].values
    c = confirmed
    tp = int(((c == 1) & (y == 1)).sum())
    fp = int(((c == 1) & (y == 0)).sum())
    fn = int(((c == 0) & (y == 1)).sum())
    tn = int(((c == 0) & (y == 0)).sum())
    insiders = df.loc[df["label"] == 1, "user"].unique()
    ins_set = set(insiders)
    caught = sum(
        1 for u in insiders
        if ((df["user"] == u) & (confirmed == 1) & (df["label"] == 1)).any()
    )
    flagged = df.groupby("user")["confirmed_col"].max() if "confirmed_col" in df else None
    benign_users = [u for u in df["user"].unique() if u not in ins_set]
    bf = sum(
        1 for u in benign_users
        if ((df["user"] == u) & (confirmed == 1)).any()
    )
    return {
        "rule": rule_name,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "alerts_per_day": c.sum() / df["date"].nunique(),
        "insiders_caught": caught,
        "benign_flagged": bf,
    }


def main():
    train, test, cutoff = load_split()
    cols = Z_COLS + ["weekend"]
    rules = build_rules(chosen_name)
    test = test.sort_values(["user", "date"]).reset_index(drop=True)

    all_rows = []
    for seed in SEEDS:
        model = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1)
        model.fit(train[cols])
        tr_s = -model.score_samples(train[cols])
        t1 = float(np.percentile(tr_s, 99))
        t2 = float(np.percentile(tr_s, 95))
        test["score"] = -model.score_samples(test[cols])
        for r in rules:
            c = apply_rule(test, t1, t2, r)
            row = score_rule(test, c, r["name"])
            row["seed"] = seed
            all_rows.append(row)
        print("Seed", seed, "done.")

    res = pd.DataFrame(all_rows)
    res.to_csv(RESULTS / "multiseed_raw.csv", index=False)

    print("")
    print("MULTI-SEED SUMMARY (mean ± std across 5 seeds):")
    metrics = ["precision", "recall", "f1", "fpr",
               "alerts_per_day", "insiders_caught", "benign_flagged"]
    summary_rows = []
    for rule in res["rule"].unique():
        sub = res[res["rule"] == rule]
        row = {"rule": rule}
        for m in metrics:
            row[m + "_mean"] = round(sub[m].mean(), 4)
            row[m + "_std"] = round(sub[m].std(), 4)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS / "multiseed_summary.csv", index=False)

    for _, r in summary.iterrows():
        print("")
        print("Rule:", r["rule"])
        for m in metrics:
            print("  " + m + ":", r[m + "_mean"], "±", r[m + "_std"])

    print("")
    print("Saved: multiseed_raw.csv and multiseed_summary.csv in", RESULTS)


main()