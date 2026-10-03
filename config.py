from pathlib import Path

ROOT = Path(".").resolve()
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

R42_DIR = RAW / "r4.2"
INSIDERS_CSV = RAW / "answers" / "insiders.csv"
EVENTS_PARQUET = PROCESSED / "events.parquet"

DATASET_VERSION = "4.2"
N_BENIGN = 400
SEED = 42
CHUNK = 500000
TIME_FMT = "%m/%d/%Y %H:%M:%S"

for p in (PROCESSED, RESULTS):
    p.mkdir(parents=True, exist_ok=True)