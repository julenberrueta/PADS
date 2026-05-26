"""Optimal-threshold selection from a ROC / PR curve."""
from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

ThresholdMethod = Literal["youden", "min_distance", "precision_recall"]


def optimal_threshold(
    method: ThresholdMethod, y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """Pick a single decision threshold using the chosen criterion."""
    if method == "precision_recall":
        precision, recall, pr_ths = precision_recall_curve(y_true, y_pred)
        # precision/recall have one extra element vs pr_ths; align by dropping the last point
        precision = precision[:-1]
        recall = recall[:-1]
        fscore = 2 * precision * recall / (precision + recall + 1e-8)
        return float(pr_ths[np.argmax(fscore)])

    fp, tp, thresholds = roc_curve(y_true, y_pred)
    if method == "youden":
        idx = int(np.argmax(tp - fp))
    elif method == "min_distance":
        idx = int(np.argmin(np.sqrt(fp**2 + (1 - tp) ** 2)))
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    return float(thresholds[idx])
