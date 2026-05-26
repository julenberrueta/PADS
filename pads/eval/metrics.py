"""Per-model evaluation metrics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class BinaryMetrics:
    auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    threshold: float
    fp_curve: np.ndarray
    tp_curve: np.ndarray
    opt_fp: float
    opt_tp: float
    bin_pred: np.ndarray


def evaluate(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float
) -> BinaryMetrics:
    auc = float(roc_auc_score(y_true, y_pred))
    fp, tp, ths = roc_curve(y_true, y_pred)
    bin_pred = (y_pred >= threshold).astype(int)
    idx = int(np.argmin(np.abs(ths - threshold)))
    return BinaryMetrics(
        auc=auc,
        accuracy=float(accuracy_score(y_true, bin_pred)),
        precision=float(precision_score(y_true, bin_pred, zero_division=0)),
        recall=float(recall_score(y_true, bin_pred, zero_division=0)),
        f1=float(f1_score(y_true, bin_pred, zero_division=0)),
        threshold=float(threshold),
        fp_curve=fp,
        tp_curve=tp,
        opt_fp=float(fp[idx]),
        opt_tp=float(tp[idx]),
        bin_pred=bin_pred,
    )
