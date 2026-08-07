"""Download the ULB credit-card fraud dataset from Kaggle.

The take-home references the Kaggle copy of this dataset. `kagglehub` can
download public Kaggle datasets without requiring a manually created kaggle.json
in many local environments.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import kagglehub


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"
EXPECTED_COLUMNS = {"Time", "Amount", "Class", *(f"V{i}" for i in range(1, 29))}


def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading Kaggle dataset mlg-ulb/creditcardfraud...")
    dataset_dir = Path(kagglehub.dataset_download("mlg-ulb/creditcardfraud"))
    source_path = dataset_dir / "creditcard.csv"
    if not source_path.exists():
        raise RuntimeError(f"Expected creditcard.csv under {dataset_dir}, but it was not found.")

    shutil.copyfile(source_path, DATA_PATH)
    with DATA_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        columns = set(next(reader))
    missing = sorted(EXPECTED_COLUMNS - columns)
    if missing:
        raise RuntimeError(f"Downloaded dataset is missing expected columns: {missing}")
    print(f"Wrote dataset to {DATA_PATH}")


if __name__ == "__main__":
    main()
