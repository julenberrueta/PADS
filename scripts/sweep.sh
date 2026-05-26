#!/usr/bin/env bash
# PADS retrain-type sweep (Linux / WSL2 version).
#
# Runs the pipeline once per retrain_type to compare the four freezing
# strategies (full / dense / lstm / scratch) on the same dataset. The data-prep
# step (prepare_data) does NOT depend on retrain_type, so it runs once up front
# and every iteration reuses its artifacts.
#
# If MLFLOW_TRACKING_URI is set, every iteration logs to MLflow as a separate
# run with `retrain_type` recorded as a parameter.
#
# Usage:
#     ./scripts/sweep.sh --dataset synthetic_dataset.csv             # full,dense,lstm,scratch
#     ./scripts/sweep.sh --dataset my.csv --types full,scratch
#     ./scripts/sweep.sh --dataset my.csv --epochs 2 --batch-size 32
#     ./scripts/sweep.sh --dataset my.csv --experiment pads_sweep_2epoch
#     ./scripts/sweep.sh --dataset my.csv --skip-data                # reuse existing data/*.pkl

set -uo pipefail

DATASET=""
TYPES="full,dense,lstm,scratch"
EPOCHS=1000
BATCH_SIZE=100
SEED=42
EXPERIMENT=""
SKIP_DATA=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)     DATASET="$2";     shift 2;;
        --types)       TYPES="$2";       shift 2;;
        --epochs)      EPOCHS="$2";      shift 2;;
        --batch-size)  BATCH_SIZE="$2";  shift 2;;
        --seed)        SEED="$2";        shift 2;;
        --experiment)  EXPERIMENT="$2";  shift 2;;
        --skip-data)   SKIP_DATA=1;      shift 1;;
        *) echo "Unknown arg: $1" >&2; exit 2;;
    esac
done

if [[ -z "$DATASET" ]]; then
    echo "Error: --dataset <filename inside data/> is required." >&2
    exit 2
fi

export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-linux}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Run scripts/setup.sh first." >&2
    exit 1
fi

# Auto-load .env if present (MLflow URI / credentials / encoding workarounds).
[[ -f .env ]] && . scripts/load_env.sh

[[ -n "$EXPERIMENT" ]] && export PADS_MLFLOW_EXPERIMENT="$EXPERIMENT"

echo ""
echo "Sweeping retrain_type over: $TYPES"
if [[ -n "${MLFLOW_TRACKING_URI:-}" ]]; then
    echo "MLflow tracking ON: $MLFLOW_TRACKING_URI"
else
    echo "MLflow tracking OFF (set MLFLOW_TRACKING_URI to enable)"
fi
echo ""

# Data-prep steps are retrain_type-independent — run them once.
if [[ "$SKIP_DATA" -eq 0 ]]; then
    echo "=== prepare_data ==="
    uv run pads --data_filename "$DATASET" --seed "$SEED" --mode prepare_data || {
        echo "prepare_data failed" >&2; exit 1; }
else
    echo "Skipping data prep (--skip-data); reusing existing data/*.pkl"
fi

IFS=',' read -ra TYPE_ARR <<< "$TYPES"
for t in "${TYPE_ARR[@]}"; do
    echo "=== retrain_type = $t ==="
    for mode in retrain_models calculate_metrics inference; do
        uv run pads --data_filename "$DATASET" --seed "$SEED" \
            --retrain_type "$t" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" \
            --mode "$mode" || {
            echo "$mode failed for retrain_type=$t" >&2
            exit 1
        }
    done
done

echo ""
echo "Sweep complete. Retrained models:"
ls -la models/RETRAINED_*_lstm_*.keras
