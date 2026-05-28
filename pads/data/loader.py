"""Dataset and artifact IO."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from pads.data.schema import validate_dataset


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a dataframe from CSV, Parquet, Excel, or pickle based on file extension.

    Automatically renames PatientID to stay_id if needed for compatibility.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == '.parquet':
        df = pd.read_parquet(path)
    elif suffix in ['.csv', '.csv.gz', '.csv.bz2', '.csv.zip', '.csv.xz']:
        df = pd.read_csv(path)
    elif suffix == '.xlsx':
        df = pd.read_excel(path)  # needs openpyxl
    elif suffix in ['.pkl', '.pickle']:
        df = pd.read_pickle(path)
        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                f"Pickle did not contain a DataFrame (got {type(df).__name__})."
            )
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            "Expected .csv, .parquet, .xlsx, or .pkl/.pickle"
        )

    # Auto-rename PatientID to stay_id for compatibility
    if 'PatientID' in df.columns and 'stay_id' not in df.columns:
        df = df.rename(columns={'PatientID': 'stay_id'})

    return df


def load_raw_dataset(data_path: str | Path) -> pd.DataFrame:
    """Load a dataset (CSV or Parquet), validate schema, attach `outcome` and `los` columns."""
    df = validate_dataset(data_path)
    df["outcome"] = df["icu_expire_flag"]
    df["los"] = df["stay_id"].map(df.groupby("stay_id")["hr"].max())
    return df


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path, indent: int = 4) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)


def load_pkl(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_stays(path: str | Path) -> list[int]:
    """Read a stays.txt file (one stay_id per line)."""
    with open(path) as f:
        return [int(line.strip()) for line in f if line.strip()]


def save_stays(stays: list[int], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(f"{s}\n" for s in stays)
