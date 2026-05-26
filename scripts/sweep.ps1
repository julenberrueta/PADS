# PADS retrain-type sweep (Windows / PowerShell).
#
# Runs the pipeline once per retrain_type to compare the four freezing
# strategies (full / dense / lstm / scratch) on the same dataset. The data-prep
# step (prepare_data) does NOT depend on retrain_type, so it runs once up front
# and every iteration reuses its artifacts:
#
#   1 x prepare_data
#   + N x (retrain_models + calculate_metrics + inference)
#
# Each retrain_type writes to its own RETRAINED_<type>_lstm_*.keras, so the
# outputs do not collide.
#
# If MLFLOW_TRACKING_URI is set, every iteration logs to MLflow as a separate
# run with `retrain_type` recorded as a parameter — open the MLflow UI to
# compare them side by side.
#
# Usage:
#     .\scripts\sweep.ps1 -Dataset synthetic_dataset.csv          # sweeps full,dense,lstm,scratch
#     .\scripts\sweep.ps1 -Dataset my.csv -Types full,scratch     # custom subset
#     .\scripts\sweep.ps1 -Dataset my.csv -Epochs 2               # quick smoke run
#     .\scripts\sweep.ps1 -Dataset my.csv -SkipData               # reuse existing data/*.pkl

param(
    [Parameter(Mandatory = $true)]
    [string]   $Dataset,
    [string[]] $Types = @('full','dense','lstm','scratch'),
    [int]      $Epochs = 1000,
    [int]      $BatchSize = 100,
    [int]      $Seed = 42,
    [string]   $Experiment = '',   # overrides PADS_MLFLOW_EXPERIMENT for this sweep
    [switch]   $SkipData           # skip prepare_data
)

# Native commands (uv) write progress to stderr; PowerShell 5.1 wraps those
# lines in ErrorRecords, so we cannot use "Stop" or it aborts mid-sweep.
$ErrorActionPreference = "Continue"
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv is not installed. Run scripts\setup.ps1 first."
    exit 1
}

# Auto-load .env if present (MLflow URI / credentials / encoding workarounds).
if (Test-Path .env) {
    . .\scripts\load_env.ps1
}

if ($Experiment) { $env:PADS_MLFLOW_EXPERIMENT = $Experiment }

Write-Host ""
Write-Host "Sweeping retrain_type over: $($Types -join ', ')" -ForegroundColor Cyan
if ($env:MLFLOW_TRACKING_URI) {
    Write-Host "MLflow tracking ON: $env:MLFLOW_TRACKING_URI" -ForegroundColor Green
} else {
    Write-Host "MLflow tracking OFF (set MLFLOW_TRACKING_URI to enable)" -ForegroundColor Yellow
}
Write-Host ""

# Data-prep steps are retrain_type-independent — run them once.
if (-not $SkipData) {
    Write-Host "=== prepare_data ===" -ForegroundColor Magenta
    uv run pads --data_filename $Dataset --seed $Seed --mode prepare_data
    if ($LASTEXITCODE -ne 0) { Write-Error "prepare_data failed (exit $LASTEXITCODE)"; exit 1 }
} else {
    Write-Host "Skipping data prep (-SkipData); reusing existing data/*.pkl" -ForegroundColor Yellow
}

foreach ($t in $Types) {
    Write-Host "=== retrain_type = $t ===" -ForegroundColor Magenta
    foreach ($mode in 'retrain_models','calculate_metrics','inference') {
        uv run pads --data_filename $Dataset --seed $Seed `
            --retrain_type $t --epochs $Epochs --batch_size $BatchSize `
            --mode $mode
        if ($LASTEXITCODE -ne 0) {
            Write-Error "$mode failed for retrain_type=$t (exit $LASTEXITCODE)"
            exit 1
        }
    }
}

Write-Host ""
Write-Host "Sweep complete. Retrained models:" -ForegroundColor Green
Get-ChildItem .\models\RETRAINED_*_lstm_*.keras | Select-Object Name, Length, LastWriteTime
