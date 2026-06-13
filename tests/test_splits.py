"""Unit tests for stay-level 3-way splits."""
from __future__ import annotations

import numpy as np

from pads.training.splits import encode_xy, split_stays


def test_split_sizes_and_coverage():
    ids = list(range(100))
    y = np.array([1] * 10 + [0] * 90)
    tr, va, te = split_stays(ids, y, seed=42)
    assert (len(tr), len(va), len(te)) == (70, 10, 20)
    assert set(tr).isdisjoint(va) and set(tr).isdisjoint(te) and set(va).isdisjoint(te)
    assert sorted(tr + va + te) == ids


def test_split_is_stratified():
    ids = list(range(100))
    y = np.array([1] * 10 + [0] * 90)
    tr, va, te = split_stays(ids, y, seed=42)
    # every fold gets at least one positive (10% prevalence, stratified)
    assert all(sum(y[i] for i in fold) >= 1 for fold in (tr, va, te))


def test_split_is_deterministic():
    ids = list(range(100))
    y = np.array([1] * 10 + [0] * 90)
    assert split_stays(ids, y, seed=7) == split_stays(ids, y, seed=7)


def test_single_class_falls_back_to_random():
    ids = list(range(100))
    tr, va, te = split_stays(ids, np.ones(100), seed=42)
    assert (len(tr), len(va), len(te)) == (70, 10, 20)


def test_no_outcomes():
    ids = list(range(100))
    tr, va, te = split_stays(ids, None, seed=1)
    assert (len(tr), len(va), len(te)) == (70, 10, 20)
    assert sorted(tr + va + te) == ids


def test_grouped_split_keeps_episodes_together():
    # 60 stays in 30 episodes, 2 stays each; first 6 episodes are positive.
    stay_ids = list(range(60))
    episodes = [i // 2 for i in range(60)]
    outcomes = np.array([1 if episodes[i] < 6 else 0 for i in range(60)])

    tr, va, te = split_stays(stay_ids, outcomes, groups=episodes, seed=42)

    assert sorted(tr + va + te) == stay_ids
    fold = {s: name for name, g in (("tr", tr), ("va", va), ("te", te)) for s in g}
    # both stays of every episode land in the same fold
    assert all(fold[2 * e] == fold[2 * e + 1] for e in range(30))
    # episode sets are disjoint across folds
    eps = lambda g: {episodes[s] for s in g}
    assert eps(tr).isdisjoint(eps(va)) and eps(tr).isdisjoint(eps(te))
    assert eps(va).isdisjoint(eps(te))


def test_grouped_split_stratifies_by_episode_label():
    stay_ids = list(range(60))
    episodes = [i // 2 for i in range(60)]
    outcomes = np.array([1 if episodes[i] < 6 else 0 for i in range(60)])
    tr, va, te = split_stays(stay_ids, outcomes, groups=episodes, seed=42)
    # each fold should carry some positives (6 positive episodes, stratified)
    assert all(sum(outcomes[s] for s in fold) >= 1 for fold in (tr, va, te))


def test_encode_xy_zeros_nan_and_one_hots():
    X, y = encode_xy(np.array([[np.nan], [1.0]]), np.array([[1.0], [0.0]]))
    assert X.tolist() == [[0.0], [1.0]]
    assert y.tolist() == [[0.0, 1.0], [1.0, 0.0]]
