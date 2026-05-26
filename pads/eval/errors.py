"""Combined mortality+discharge error categorisation (4 clinical groups)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pads.data.loader import load_raw_dataset

EXITUS_LT = "EXITUS <48h"
EXITUS_GT = "EXITUS >48h"
ALIVE_LT = "ALIVE <48h"
ALIVE_GT = "ALIVE >48h"


def color_group(mort_cat: np.ndarray, disch_cat: np.ndarray) -> np.ndarray:
    """Combine binary mortality/discharge labels into the 4 clinical categories."""
    return np.where(
        (mort_cat == 1) & (disch_cat == 1), EXITUS_LT,
        np.where((mort_cat == 1) & (disch_cat == 0), EXITUS_GT,
        np.where((mort_cat == 0) & (disch_cat == 0), ALIVE_GT, ALIVE_LT)),
    )


def _error_level(rg: pd.Series, cg: pd.Series) -> np.ndarray:
    """Map a (real_group, predicted_group) pair to a 0..3 severity score."""
    # Severity 3: EXITUS<48h vs ALIVE<48h (either direction)
    sev3 = (((rg == EXITUS_LT) & (cg == ALIVE_LT)) | ((cg == EXITUS_LT) & (rg == ALIVE_LT))).to_numpy()
    # Severity 2: EXITUS<48h vs ALIVE>48h (either dir), or EXITUS>48h vs ALIVE<48h (either dir)
    sev2 = (
        ((rg == EXITUS_LT) & (cg == ALIVE_GT)) | ((cg == EXITUS_LT) & (rg == ALIVE_GT))
        | ((cg == EXITUS_GT) & (rg == ALIVE_LT)) | ((rg == EXITUS_GT) & (cg == ALIVE_LT))
    ).to_numpy()
    # Severity 1: confusions within the same survival class
    sev1 = (
        ((rg == EXITUS_LT) & (cg == EXITUS_GT)) | ((cg == EXITUS_LT) & (rg == EXITUS_GT))
        | ((cg == ALIVE_GT) & (rg == ALIVE_LT))  | ((rg == ALIVE_GT) & (cg == ALIVE_LT))
        | ((rg == ALIVE_GT) & (cg == EXITUS_GT)) | ((cg == ALIVE_GT) & (rg == EXITUS_GT))
    ).to_numpy()

    err = np.zeros(len(rg), dtype=int)
    err[sev1] = 1
    err[sev2] = 2
    err[sev3] = 3
    return err


def compute_errors(
    data_path: str | Path,
    dataset: dict,
    mort_pred: np.ndarray,
    mort_gt: np.ndarray,
    disch_pred: np.ndarray,
    disch_gt: np.ndarray,
    params: dict[str, float],
    out_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build the per-row error DataFrame used by `viz.plots.plot_error`."""
    th_mort = params["th_mort"]
    th_disch = params["th_disch"]
    min_prob = params["min_prob"]
    max_prob = params["max_prob"]

    base = load_raw_dataset(data_path)
    base = base.drop_duplicates(["stay_id", "hr"], keep="last").reset_index(drop=True)
    exitus = base[["stay_id", "icu_expire_flag"]].drop_duplicates()

    stay_ids = [sid for sid, arr in dataset["data"].items() for _ in range(len(arr))]
    df = pd.DataFrame(
        {
            "stay_id": stay_ids,
            "mortality_prob": mort_pred,
            "mortality_gt": mort_gt,
            "disch_prob": disch_pred,
            "disch_gt": disch_gt,
        }
    )
    df["range"] = np.where(
        df["mortality_prob"] < th_mort, th_mort - min_prob, max_prob - th_mort
    )
    df["substract"] = np.where(
        df["mortality_prob"] < th_mort,
        df["mortality_prob"] - min_prob,
        df["mortality_prob"] - th_mort,
    )
    df["normalized"] = np.where(
        df["mortality_prob"] < th_mort,
        (df["substract"] / df["range"] / 2) * 100,
        ((df["substract"] / df["range"] + 1) / 2) * 100,
    )
    df = pd.merge(df, exitus, on="stay_id", how="left")
    df["disch_prob_cat"] = (df["disch_prob"] > th_disch).astype(int)
    df["mortality_prob_cat"] = (df["mortality_prob"] > th_mort).astype(int)
    df["color_group"] = color_group(df["mortality_prob_cat"], df["disch_prob_cat"])
    df["real_color_group"] = color_group(df["icu_expire_flag"], df["disch_gt"])

    df["error"] = _error_level(df["real_color_group"], df["color_group"])
    df["max_error_possible"] = np.where(df["real_color_group"].isin([EXITUS_LT, ALIVE_LT]), 3, 2)

    if out_path is not None:
        df.to_csv(out_path, index=False)
    return df
