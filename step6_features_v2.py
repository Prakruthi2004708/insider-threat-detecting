import shutil
import numpy as np
import pandas as pd
from config import *

V1 = PROCESSED / "features_v1.parquet"
OUT = PROCESSED / "features.parquet"

NEW_FEATURES = ["n_new_pc", "n_new_ext", "n_filecopy_usb_day",
                "n_weekend_events", "span_hours"]
WARMUP_DAYS = 14


def main():
    if not V1.exists():
        shutil.copy(OUT, V1)
    base = pd.read_parquet(V1)

    ev = pd.read_parquet(EVENTS_PARQUET)
    ev = ev.sort_values("timestamp").reset_index(drop=True)
    ev["date"] = ev["timestamp"].dt.normalize()

    first_date = ev.groupby("user")["date"].transform("min")
    past_warmup = ev["date"] >= first_date + pd.Timedelta(days=WARMUP_DAYS)

    ev["new_pc"] = ((~ev.duplicated(["user", "pc"])) & past_warmup).astype(int)

    ev["new_ext"] = 0
    fe = ev[ev["source"] == "file"]
    fnew = (~fe.duplicated(["user", "ext"])) & past_warmup.loc[fe.index]
    ev.loc[fnew[fnew].index, "new_ext"] = 1

    ev["weekend_ev"] = (ev["timestamp"].dt.dayofweek >= 5).astype(int)

    daily = ev.groupby(["user", "date"]).agg(
        n_new_pc=("new_pc", "sum"),
        n_new_ext=("new_ext", "sum"),
        n_weekend_events=("weekend_ev", "sum"),
    ).reset_index()

    df = base.merge(daily, on=["user", "date"], how="left")
    for c in ["n_new_pc", "n_new_ext", "n_weekend_events"]:
        df[c] = df[c].fillna(0)
    df["n_filecopy_usb_day"] = np.where(df["n_connect"] > 0, df["n_filecopy"], 0)
    df["span_hours"] = df["last_hour"] - df["first_hour"]

    df = df.sort_values(["user", "date"]).reset_index(drop=True)
    for f in NEW_FEATURES:
        grp = df.groupby("user")[f]
        mean = grp.transform(lambda s: s.expanding(min_periods=14).mean().shift(1))
        std = grp.transform(lambda s: s.expanding(min_periods=14).std().shift(1))
        z = (df[f] - mean) / std.clip(lower=1.0)
        df["z_" + f] = z.fillna(0).clip(-10, 10)

    df.to_parquet(OUT, index=False)

    print("Rows:", len(df), "| columns:", df.shape[1])
    print("")
    print("Average value of each new feature (label 0 = normal day, 1 = malicious day):")
    print(df.groupby("label")[NEW_FEATURES].mean().round(3).T)
    print("")
    print("Saved:", OUT)
    print("Backup of old features:", V1)


main()