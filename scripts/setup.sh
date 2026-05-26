#!/usr/bin/env bash
# PADS one-shot setup script (Linux / WSL2).
#
# Usage:
#     ./scripts/setup.sh
#
# What it does:
#   1. Installs `uv` to $HOME/.local/bin if not already present.
#   2. Adds that location to PATH for the current session.
#   3. Runs `uv sync --all-extras`, which on Linux pulls
#      tensorflow[and-cuda] (TF + CUDA 12.x + cuDNN 9) plus every other dep.
#   4. Copies .env.example -> .env if .env does not exist.
#
# After this, GPU support depends on:
#   - WSL2 with NVIDIA driver installed on the Windows host (verified via
#     `nvidia-smi` inside WSL).
#   - Compute capability of your GPU being supported by the bundled CUDA.
#
# Run:
#     uv run pads --help
#     uv run pads --data_filename synthetic_dataset.csv --mode all
#     uv run pytest -m "not slow"
#     ./scripts/sweep.sh --dataset synthetic_dataset.csv

set -euo pipefail

# Use a Linux-specific venv directory so a Windows .venv/ in the same repo
# (e.g. when mounted from /mnt/c/...) does not get clobbered.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-linux}"

# 1. Make sure uv is available.
if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found - installing it to \$HOME/.local/bin..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo ""
echo "uv version: $(uv --version)"

# 2. Sync the project.
echo ""
echo "Running 'uv sync --all-extras' — this installs TensorFlow (GPU build), MLflow, dev tools..."
uv sync --all-extras

# 3. Bootstrap .env if it doesn't exist.
if [[ ! -f .env && -f .env.example ]]; then
    cp .env.example .env
    echo ""
    echo "Created .env from .env.example — edit it to fill in your MLflow credentials."
fi

# 4. Persist the Linux venv selection so future `uv run` calls in interactive
#    shells target .venv-linux too — and never clobber a Windows .venv living in
#    the same repo (e.g. on a /mnt/c mount). Windows uses the default .venv;
#    Linux/WSL uses .venv-linux. They never collide.
BASHRC="$HOME/.bashrc"
MARKER="# PADS: use a Linux-specific uv venv (added by scripts/setup.sh)"
if ! { [[ -f "$BASHRC" ]] && grep -qF "$MARKER" "$BASHRC"; }; then
    {
        echo ""
        echo "$MARKER"
        echo 'export UV_PROJECT_ENVIRONMENT=.venv-linux'
    } >> "$BASHRC"
    echo ""
    echo "Added UV_PROJECT_ENVIRONMENT=.venv-linux to $BASHRC."
    echo "Open a new shell (or run 'source ~/.bashrc') so 'uv run pads ...' targets it automatically."
fi

echo ""
echo "Done."
echo ""
echo "Quick sanity check (should list your NVIDIA GPU):"
echo "  uv run python -c 'import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))'"
echo ""
echo "Try it:"
echo "  . scripts/load_env.sh                                                   # load .env into your shell"
echo "  uv run pads --data_filename synthetic_dataset.csv --mode prepare_data"
echo "  uv run pads --data_filename synthetic_dataset.csv --mode all            # whole pipeline"
echo "  uv run pytest -m \"not slow\""
echo "  ./scripts/sweep.sh --dataset synthetic_dataset.csv                      # sweep all retrain_types"
