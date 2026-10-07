import numpy as np
import pandas as pd
from config import *

FEATURES_PARQUET = PROCESSED / "features.parquet"

RAW_FEATURES = [
    "n_events", "n_logon", "n_connect", "n_filecopy", "n_risky_ext",
    "n_after_hours", "n_after_connect", "n_after_file", "n_pcs",
    "first_hour", "last_hour", "n_weekend_events", "span_hours",
    "n_new_pc", "n_new_ext",
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
        n_weekend_events=("weekend", "sum"),
        label=("label", "max"),
        scenario=("scenario", "max"),
    ).reset_index()

    # New PCs: number of PCs whose first-ever use by this user is on this day
    pc_first = ev.groupby(["user", "pc"])["date"].min().reset_index()
    new_pc = pc_first.groupby(["user", "date"]).size().reset_index(name="n_new_pc")