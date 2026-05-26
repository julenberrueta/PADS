from __future__ import annotations

import numpy as np
import pandas as pd

from pads.eval.errors import (
    ALIVE_GT,
    ALIVE_LT,
    EXITUS_GT,
    EXITUS_LT,
    _error_level,
    color_group,
)


def test_color_group_mapping():
    mort = np.array([1, 1, 0, 0])
    disch = np.array([1, 0, 0, 1])
    groups = color_group(mort, disch)
    assert list(groups) == [EXITUS_LT, EXITUS_GT, ALIVE_GT, ALIVE_LT]


def test_error_level_severity_3():
    """EXITUS<48h vs ALIVE<48h is the worst confusion."""
    rg = pd.Series([EXITUS_LT, ALIVE_LT])
    cg = pd.Series([ALIVE_LT, EXITUS_LT])
    assert list(_error_level(rg, cg)) == [3, 3]


def test_error_level_correct_predictions_are_zero():
    rg = pd.Series([EXITUS_LT, EXITUS_GT, ALIVE_LT, ALIVE_GT])
    cg = rg.copy()
    assert list(_error_level(rg, cg)) == [0, 0, 0, 0]


def test_error_level_severity_2():
    """Mortality/discharge boundary crossed: EXITUS<48h vs ALIVE>48h, EXITUS>48h vs ALIVE<48h."""
    rg = pd.Series([EXITUS_LT, ALIVE_GT, EXITUS_GT, ALIVE_LT])
    cg = pd.Series([ALIVE_GT, EXITUS_LT, ALIVE_LT, EXITUS_GT])
    assert list(_error_level(rg, cg)) == [2, 2, 2, 2]


def test_error_level_severity_1_intraclass():
    """Same survival class but wrong timing."""
    rg = pd.Series([EXITUS_LT, ALIVE_GT])
    cg = pd.Series([EXITUS_GT, ALIVE_LT])
    assert list(_error_level(rg, cg)) == [1, 1]
