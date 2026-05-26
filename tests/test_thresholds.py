from __future__ import annotations

import numpy as np
import pytest

from pads.eval.thresholds import optimal_threshold


def _toy_problem(seed=0):
    rng = np.random.default_rng(seed)
    n = 200
    y_true = (rng.uniform(size=n) > 0.6).astype(int)
    # slightly informative score
    y_pred = y_true * 0.7 + rng.uniform(size=n) * 0.3
    return y_true, y_pred


@pytest.mark.parametrize("method", ["youden", "min_distance", "precision_recall"])
def test_threshold_in_range(method):
    y_true, y_pred = _toy_problem()
    th = optimal_threshold(method, y_true, y_pred)
    assert 0.0 <= th <= 1.0


def test_unknown_method_raises():
    y_true, y_pred = _toy_problem()
    with pytest.raises(ValueError):
        optimal_threshold("not_a_method", y_true, y_pred)  # type: ignore[arg-type]


def test_precision_recall_index_does_not_overflow():
    """Regression test for the v3 bug where thresholds[idx] could overflow precision/recall arrays."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.6])
    th = optimal_threshold("precision_recall", y_true, y_pred)
    assert np.isfinite(th)
