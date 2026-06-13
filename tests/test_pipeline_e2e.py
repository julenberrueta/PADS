"""End-to-end pipeline tests on the synthetic dataset.

Marked `slow` because they invoke TensorFlow model loading / prediction.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.e2e]


def _make_pipeline(base_path: Path, **kwargs):
    from pads.config import PADSConfig
    from pads.pipeline import PADSPipeline

    cfg = PADSConfig(base_path=base_path, **kwargs)
    return PADSPipeline(cfg)


def test_prepare_data_creates_artifacts(project_dir: Path):
    pipeline = _make_pipeline(project_dir, seed=42)
    pipeline.prepare_data("synthetic_dataset.csv")

    assert (project_dir / "data" / "processed" / "medians_48h.json").is_file()
    assert (project_dir / "data" / "processed" / "train_stays.txt").is_file()
    assert (project_dir / "data" / "processed" / "val_stays.txt").is_file()
    assert (project_dir / "data" / "processed" / "test_stays.txt").is_file()
    assert (project_dir / "normalizers" / "mortality_normalizer.pkl").is_file()
    assert (project_dir / "normalizers" / "discharge_normalizer.pkl").is_file()


def test_prepare_data_creates_pkls(project_dir: Path):
    pipeline = _make_pipeline(project_dir, seed=42)
    pipeline.prepare_data("synthetic_dataset.csv")

    for name in (
        "lstm_last_48h_train.pkl",
        "lstm_last_48h_val.pkl",
        "lstm_last_48h_test.pkl",
        "icu_expire_flag_train.pkl",
        "icu_expire_flag_val.pkl",
        "icu_expire_flag_test.pkl",
        "lstm_disch_3point_48h_train.pkl",
        "lstm_disch_3point_48h_val.pkl",
        "lstm_disch_3point_48h_test.pkl",
        "outcome_disch_3point_48h_train.pkl",
        "outcome_disch_3point_48h_val.pkl",
        "outcome_disch_3point_48h_test.pkl",
    ):
        assert (project_dir / "data" / "processed" / name).is_file(), f"missing {name}"


def test_prepare_data_records_provenance(project_dir: Path):
    pipeline = _make_pipeline(project_dir, seed=42)
    pipeline.prepare_data("synthetic_dataset.csv")

    prov_path = project_dir / "data" / "processed" / "source_dataset.json"
    assert prov_path.is_file()

    import json

    info = json.loads(prov_path.read_text())
    assert info["dataset"] == "synthetic_dataset.csv"
    assert len(info["dataset_sha256"]) == 64

    # split sizes match the generated stays.txt files
    processed = project_dir / "data" / "processed"
    n_train = sum(1 for line in (processed / "train_stays.txt").read_text().splitlines() if line.strip())
    n_val = sum(1 for line in (processed / "val_stays.txt").read_text().splitlines() if line.strip())
    n_test = sum(1 for line in (processed / "test_stays.txt").read_text().splitlines() if line.strip())
    assert info["n_train_stays"] == n_train
    assert info["n_val_stays"] == n_val
    assert info["n_test_stays"] == n_test

    # retrain/metrics runs (no data_filename) recover the dataset from provenance
    tags = pipeline._common_tags()
    assert tags["dataset"] == "synthetic_dataset.csv"
    assert tags["dataset_sha256"] == info["dataset_sha256"][:16]
    assert tags["n_train_stays"] == n_train
    assert tags["n_val_stays"] == n_val
    assert tags["n_test_stays"] == n_test


def test_prepare_data_groups_by_hospital_episode(project_dir: Path):
    """When hospital_episode_id is present, episodes are not split across folds."""
    import pandas as pd

    csv = project_dir / "data" / "synthetic_dataset.csv"
    df = pd.read_csv(csv)
    # Pair up stays into hospital episodes: 2 stays per episode.
    stays = sorted(df["stay_id"].unique())
    episode_of = {sid: i // 2 for i, sid in enumerate(stays)}
    df["hospital_episode_id"] = df["stay_id"].map(episode_of)
    df.to_csv(csv, index=False)

    pipeline = _make_pipeline(project_dir, seed=42)
    pipeline.prepare_data("synthetic_dataset.csv")

    processed = project_dir / "data" / "processed"
    folds = {}
    for name in ("train", "val", "test"):
        for line in (processed / f"{name}_stays.txt").read_text().splitlines():
            if line.strip():
                folds[int(line)] = name

    # every stay of a given episode must land in the same fold
    by_episode: dict[int, set[str]] = {}
    for sid, fold in folds.items():
        by_episode.setdefault(episode_of[sid], set()).add(fold)
    assert all(len(f) == 1 for f in by_episode.values())


def test_retrain_smoke_one_epoch(project_dir_with_models: Path):
    """1-epoch retraining smoke test using the base models."""
    pipeline = _make_pipeline(
        project_dir_with_models,
        seed=42,
        epochs=1,
        batch_size=32,
        early_stopping_patience=1,
        retrain_type="full",
    )
    pipeline.prepare_data("synthetic_dataset.csv")
    pipeline.retrain_models()

    assert (project_dir_with_models / "models" / "RETRAINED_full_lstm_mortality_model.keras").exists()
    assert (project_dir_with_models / "models" / "RETRAINED_full_lstm_disch_model.keras").exists()


def test_full_pipeline_smoke(project_dir_with_models: Path):
    """prepare_data → retrain (1 epoch) → metrics → inference, all glued together."""
    pipeline = _make_pipeline(
        project_dir_with_models,
        seed=42,
        epochs=1,
        batch_size=32,
        early_stopping_patience=1,
        retrain_type="full",
    )
    pipeline.prepare_data("synthetic_dataset.csv")
    pipeline.retrain_models()
    pipeline.calculate_metrics()
    errors = pipeline.run_inference("synthetic_dataset.csv", test_type="full")

    assert (project_dir_with_models / "data" / "processed" / "model_parameters_test.json").is_file()
    assert (project_dir_with_models / "results" / "full" / "model_parameters_inference.json").is_file()
    assert (project_dir_with_models / "results" / "full" / "results_inference.csv").is_file()
    assert (project_dir_with_models / "results" / "full" / "images" / "roc_combined.png").is_file()
    assert (project_dir_with_models / "results" / "full" / "images" / "barplot_error.png").is_file()
    assert "error" in errors.columns
