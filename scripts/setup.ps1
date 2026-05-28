# PADS one-shot setup script (Windows / PowerShell).
#
# Usage:
#     .\scripts\setup.ps1
#
# What it does:
#   1. Installs `uv` to %USERPROFILE%\.local\bin if not already present.
#   2. Adds that location to PATH for the current session.
#   3. Runs `uv sync --all-extras`, which:
#        - creates .venv/ (Python 3.10+),
#        - installs pads in editable mode,
#        - installs every dependency from pyproject.toml,
#        - installs the optional dev extra (tests + docs tooling),
#        - locks exact versions in uv.lock.
#
# After this:
#   uv run pads --help                       # one-off command
#   uv run python -m pads.cli ...            # equivalent
#   uv run pytest -m "not slow"              # tests
#   uv run pads --data_filename synthetic_dataset.csv --mode all
#
# Or, if you prefer to "activate" the venv the classic way:
#   .\.venv\Scripts\Activate.ps1
#   pads --help

$ErrorActionPreference = "Stop"

# 1. Make sure uv is available.
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found - installing it to %USERPROFILE%\.local\bin..." -ForegroundColor Cyan
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
}
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"

Write-Host ""
Write-Host "uv version: $(uv --version)" -ForegroundColor Green

# 2. Sync the project (creates .venv/, resolves uv.lock).
Write-Host ""
Write-Host "Running 'uv sync --all-extras' — this installs TensorFlow, MLflow, dev tools..." -ForegroundColor Cyan
uv sync --all-extras

# 3. Bootstrap .env if it doesn't exist (so the user has somewhere to put creds).
if (-not (Test-Path .env)) {
    if (Test-Path .env.example) {
        Copy-Item .env.example .env
        Write-Host ""
        Write-Host "Created .env from .env.example - edit it to fill in your MLflow credentials." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host ""
Write-Host "Try it:" -ForegroundColor Yellow
Write-Host "  . .\scripts\load_env.ps1                                                    # load .env into your session"
Write-Host "  uv run pads --data_filename synthetic_dataset.csv --mode prepare_data"
Write-Host "  uv run pads --data_filename synthetic_dataset.csv --mode all                # whole pipeline"
Write-Host "  uv run pytest -m `"not slow`""
Write-Host "  .\scripts\sweep.ps1 -Dataset synthetic_dataset.csv                          # sweep all retrain_types"
