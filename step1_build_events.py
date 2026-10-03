import numpy as np
import pandas as pd
from config import *

SOURCES = [
    ("logon", "logon.csv"),
    ("device", "device.csv"),
    ("file", "file.csv"),
]


def load_insiders():
    ins = pd.read_csv(INSIDERS_CSV)
    ins["dataset"] = ins["dataset"].astype(str)
    ins = ins[ins["dataset"] == DATASET_VERSION].copy()
    ins["start"] = pd.to_datetime(ins["start"], format="mixed")
    ins["end"] = pd.to_datetime(ins["end"], format="mixed")
    return ins


def all_users():
    users = set()
    for chunk in pd.read_csv(R42_DIR / "logon.csv", usecols=["user"], chunksize=CHUNK):
        users.update(chunk["user"].unique())
    return users


def read_source(source, fname, keep):
    path = R42_DIR / fname
    cols = list(pd.read_csv(path, nrows=0).columns)
    use = ["date", "user", "pc"]
    if "activity" in cols:
        use.append("activity")
    if source == "file":
        use.append("filename")
    frames = []
    for chunk in pd.read_csv(path, usecols=use, chunksize=CHUNK):
        chunk = chunk[chunk["user"].isin(keep)].copy()
        chunk["source"] = source
        if "activity" not in chunk.columns:
            chunk["activity"] = "FileCopy"
        if "filename" in chunk.columns:
            ext = chunk["filename"].str.extract(r"\.([A-Za-z0-9]+)$")[0]
            chunk["ext"] = ext.str.lower().fillna("none")
            chunk = chunk.drop(columns="filename")
        else:
            chunk["ext"] = "none"
        frames.append(chunk)
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["date"], format=TIME_FMT)
    return df.drop(columns="date")


def main():
    ins = load_insiders()
    insiders = set(ins["user"])
    print("Insider users for r" + DATASET_VERSION + ":", len(insiders))

    users = all_users()
    benign = sorted(users - insiders)
    if N_BENIGN is not None:
        rng = np.random.default_rng(SEED)
        benign = list(rng.choice(benign, size=min(N_BENIGN, len(benign)), replace=False))
    keep = insiders | set(benign)
    print("Users kept:", len(keep), "(", len(insiders), "insiders +", len(benign), "benign )")

    events = pd.concat([read_source(s, f, keep) for s, f in SOURCES], ignore_index=True)
    events = events.sort_values("timestamp").reset_index(drop=True)

    events["label"] = 0
    events["scenario"] = 0
    for _, r in ins.iterrows():
        m = (events["user"] == r["user"]) & (events["timestamp"] >= r["start"]) & (events["timestamp"] <= r["end"])
        events.loc[m, "label"] = 1
        events.loc[m, "scenario"] = r["scenario"]

    events.to_parquet(EVENTS_PARQUET, index=False)

    print("")
    print("Events per source:")
    print(events["source"].value_counts())
    print("Date range:", events["timestamp"].min(), "to", events["timestamp"].max())
    print("Malicious-window events:", int(events["label"].sum()), "(", round(events["label"].mean() * 100, 3), "% )")
    print("Insiders with labeled events:", events.loc[events["label"] == 1, "user"].nunique())
    print("Saved:", EVENTS_PARQUET)


main()