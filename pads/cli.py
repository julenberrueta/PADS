"""Command-line interface for the PADS pipeline."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pads import tracking
from pads.config import PADSConfig
from pads.pipeline import PADSPipeline

MODES = ["prepare_data", "retrain_models", "calculate_metrics", "inference", "all"]


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file into the environment.

    Already-set environment variables are NOT overridden, so anything you
    exported by hand still wins. A missing file is a no-op. Parsing mirrors
    scripts/load_env.*: '#' comments, blank lines and surrounding quotes are
    handled. This lets `uv run pads ...` pick up MLflow settings on any OS
    without remembering to dot-source the load_env script first.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pads", description="PADS ICU prediction pipeline")
    p.add_argument("--data_filename", required=True, type=str,
                   help="Dataset filename inside <base_path>/data/ (CSV or Parquet)")
    p.add_argument("--mode", required=True, choices=MODES,
                   help="Pipeline step to run. 'all' runs every step in order.")
    p.add_argument("--base_path", default=Path("./"), type=Path)
    # These default to None so they only override PADSConfig when explicitly
    # passed — PADSConfig is the single source of truth for default values.
    p.add_argument("--test_type", default=None,
                   choices=["full", "last_48h", "last_96h", "first_48h"],
                   help="default from PADSConfig (full)")
    p.add_argument("--retrain_type", default=None,
                   choices=["full", "dense", "lstm", "scratch", "original"],
                   help="freezing strategy; default from PADSConfig. 'original' is "
                        "evaluation-only (base model, no retraining).")
    p.add_argument("--seed", default=None, type=int, help="default from PADSConfig (42)")
    p.add_argument("--epochs", default=None, type=int, help="default from PADSConfig (1000)")
    p.add_argument("--batch_size", default=None, type=int, help="default from PADSConfig (100)")
    p.add_argument("--learning_rate_mort", default=None, type=float,
                   help="mortality LR; default from PADSConfig (1e-5)")
    p.add_argument("--learning_rate_disch", default=None, type=float,
                   help="discharge LR; default from PADSConfig (1e-5)")
    p.add_argument("--early_stopping_patience", default=None, type=int,
                   help="epochs without val improvement before stopping; default from PADSConfig (20)")
    p.add_argument("--monitor_metric", default=None,
                   choices=["loss", "AUC", "accuracy", "f1_score", "precision", "recall"],
                   help="validation metric EarlyStopping/ModelCheckpoint track during "
                        "retraining; default from PADSConfig (loss)")
    p.add_argument("--mort_normalizer", default=None, type=str,
                   help="mortality normalizer filename in <base_path>/normalizers/; default from PADSConfig")
    p.add_argument("--disch_normalizer", default=None, type=str,
                   help="discharge normalizer filename in <base_path>/normalizers/; default from PADSConfig")
    p.add_argument("--inference_mort_model", default=None, type=str,
                   help="mortality model filename to evaluate; default derived from retrain_type")
    p.add_argument("--inference_disch_model", default=None, type=str,
                   help="discharge model filename to evaluate; default derived from retrain_type")
    p.add_argument("--fixed_th_mort", default=None, type=float,
                   help="fixed mortality decision threshold; overrides the data-derived "
                        "optimum in calculate_metrics/inference. Must be paired with --fixed_th_disch.")
    p.add_argument("--fixed_th_disch", default=None, type=float,
                   help="fixed discharge decision threshold; overrides the data-derived "
                        "optimum in calculate_metrics/inference. Must be paired with --fixed_th_mort.")
    p.add_argument("--threshold_method", default=None,
                   choices=["youden", "min_distance", "precision_recall"],
                   help="criterion to pick the decision threshold when no fixed one is "
                        "set; default from PADSConfig (precision_recall)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Auto-load <base_path>/.env so MLflow (and any other) settings are picked
    # up without a manual dot-source. Must run before the MLflow check below.
    _load_dotenv(args.base_path / ".env")

    # Only pass flags that were explicitly provided; everything else falls back
    # to the PADSConfig defaults (the single source of truth).
    overrides = {
        k: v for k, v in {
            "test_type": args.test_type,
            "retrain_type": args.retrain_type,
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate_mort": args.learning_rate_mort,
            "learning_rate_disch": args.learning_rate_disch,
            "early_stopping_patience": args.early_stopping_patience,
            "monitor_metric": args.monitor_metric,
            "mort_normalizer": args.mort_normalizer,
            "disch_normalizer": args.disch_normalizer,
            "inference_mort_model": args.inference_mort_model,
            "inference_disch_model": args.inference_disch_model,
            "fixed_th_mort": args.fixed_th_mort,
            "fixed_th_disch": args.fixed_th_disch,
            "threshold_method": args.threshold_method,
        }.items() if v is not None
    }
    config = PADSConfig(base_path=args.base_path, **overrides)
    pipeline = PADSPipeline(config)

    # Fail-soft MLflow check: warn early if the server is unreachable, then
    # continue with tracking disabled rather than crashing mid-pipeline.
    tracking.check_connection()

    steps = {
        "prepare_data":      lambda: pipeline.prepare_data(args.data_filename),
        "retrain_models":    lambda: pipeline.retrain_models(),
        "calculate_metrics": lambda: pipeline.calculate_metrics(),
        "inference":         lambda: pipeline.run_inference(args.data_filename),
    }

    if args.mode == "all":
        for name in ["prepare_data", "retrain_models", "calculate_metrics", "inference"]:
            print(f"\n=== {name} ===")
            steps[name]()
    else:
        steps[args.mode]()

    return 0


if __name__ == "__main__":
    sys.exit(main())
