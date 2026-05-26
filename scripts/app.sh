#!/usr/bin/env bash
# Launch the PADS training web app + a local MLflow server (Linux / WSL2).
#
# Both run via uv inside the project venv — no Docker, no MinIO. MLflow uses a
# local SQLite backend and a filesystem artifact store, so retrained models,
# plots and CSVs land under ./mlartifacts and are downloadable from the app.
#
# Usage:
#     uv sync --extra app        # one-time: install the web deps
#     ./scripts/app.sh           # starts MLflow (:5000) + app (:8000)
#
# Then open http://127.0.0.1:8000

set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-linux}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Run scripts/setup.sh first." >&2
    exit 1
fi

[[ -f .env ]] && . scripts/load_env.sh
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5000}"

echo "Starting MLflow server on http://127.0.0.1:5000 ..."
uv run mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlartifacts \
    --host 127.0.0.1 --port 5000 &
MLFLOW_PID=$!
trap 'echo "Stopping MLflow server ..."; kill "$MLFLOW_PID" 2>/dev/null' EXIT

echo "Starting PADS app on http://127.0.0.1:8000 ..."
uv run pads-app
