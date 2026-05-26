# Launch the PADS training web app + a local MLflow server (Windows / PowerShell).
#
# Both run via uv inside the project venv — no Docker, no MinIO. MLflow uses a
# local SQLite backend and a filesystem artifact store, so retrained models,
# plots and CSVs land under ./mlartifacts and are downloadable from the app.
#
# Usage:
#     uv sync --extra app        # one-time: install the web deps
#     .\scripts\app.ps1          # starts MLflow (:5000) + app (:8000)
#
# Then open http://127.0.0.1:8000

$ErrorActionPreference = "Continue"
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv is not installed. Run scripts\setup.ps1 first."
    exit 1
}

# Load .env so MLFLOW_TRACKING_URI / credentials are in scope for the app.
if (Test-Path .env) { . .\scripts\load_env.ps1 }
if (-not $env:MLFLOW_TRACKING_URI) { $env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000" }

Write-Host "Starting MLflow server on http://127.0.0.1:5000 ..." -ForegroundColor Cyan
$mlflow = Start-Process -PassThru -NoNewWindow uv -ArgumentList @(
    "run", "mlflow", "server",
    "--backend-store-uri", "sqlite:///mlflow.db",
    "--default-artifact-root", "./mlartifacts",
    "--host", "127.0.0.1", "--port", "5000"
)

try {
    Write-Host "Starting PADS app on http://127.0.0.1:8000 ..." -ForegroundColor Green
    uv run pads-app
}
finally {
    Write-Host "Stopping MLflow server ..." -ForegroundColor Yellow
    if ($mlflow -and -not $mlflow.HasExited) { Stop-Process -Id $mlflow.Id -Force }
}
