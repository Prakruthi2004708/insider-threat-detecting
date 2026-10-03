import numpy as np
import pandas as pd
from config import *

FEATURES_PARQUET = PROCESSED / "features.parquet"

RAW_FEATURES = [
    "n_events", "n_logon", "n_connect", "n_filecopy", "n_risky_ext",
    "n_after_hours", "n_after_connect", "n_after_file", "n_pcs",
    "first_hour", "last_hour",
]


def main():
    ev = pd.read_parquet(EVENTS_PARQUET)
    ts = ev["timestamp"]
    ev["date"] = ts.dt.normalize()
    ev["hour"] = ts.dt.hour + ts.dt.minute / 60.0
    ev["after_hours"] = ((ev["hour"] < 7) | (ev["hour"] >= 19)).astype(int)
    ev["weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    ev["is_logon"] = ((ev["source"] == "logon") & (ev["activity"] == "Logon")).astype(int)
    ev["is_connect"] = ((ev["source"] == "device") & (ev["activity"] == "Connect")).astype(int)
    ev["is_filecopy"] = (ev["source"] == "file").astype(int)
    ev["is_risky_ext"] = ((ev["source"] == "file") & ev["ext"].isin(["exe", "zip"])).astype(int)
    ev["ah_connect"] = ev["is_connect"] * ev["after_hours"]
    ev["ah_file"] = ev["is_filecopy"] * ev["after_hours"]

    daily = ev.groupby(["user", "date"]).agg(
        n_events=("timestamp", "size"),
        n_logon=("is_logon", "sum"),
        n_connect=("is_connect", "sum"),
        n_filecopy=("is_filecopy", "sum"),
        n_risky_ext=("is_risky_ext", "sum"),
        n_after_hours=("after_hours", "sum"),
        n_after_connect=("ah_connect", "sum"),
        n_after_file=("ah_file", "sum"),
        n_pcs=("pc", "nunique"),
        first_hour=("hour", "min"),
        last_hour=("hour", "max"),
        weekend=("weekend", "max"),
        label=("label", "max"),
        scenario=("scenario", "max"),
    ).reset_index()

    daily = daily.sort_values(["user", "date"]).reset_index(drop=True)

    # Per-user profile: each user's own past mean/std (shifted by 1 day, so no leakage).
    # The first 14 active days of a user are a warm-up and get z = 0.
    for f in RAW_FEATURES:
        grp = daily.groupby("user")[f]
        mean = grp.transform(lambda s: s.expanding(min_periods=14).mean().shift(1))
        std = grp.transform(lambda s: s.expanding(min_periods=14).std().shift(1))
        z = (daily[f] - mean) / std.clip(lower=1.0)
        daily["z_" + f] = z.fillna(0).clip(-10, 10)

    daily.to_parquet(FEATURES_PARQUET, index=False)

    print("Rows (user-days):", len(daily))
    print("Users:", daily["user"].nunique())
    print("Malicious user-days:", int(daily["label"].sum()), "(", round(daily["label"].mean() * 100, 2), "% )")
    print("Malicious users:", daily.loc[daily["label"] == 1, "user"].nunique())
    print("")
    print(daily[["user", "date", "n_events", "n_filecopy", "z_n_events", "z_n_filecopy", "label"]].tail(5))
    print("Saved:", FEATURES_PARQUET)


main()