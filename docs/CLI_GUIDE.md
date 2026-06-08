# PADS — Command-line guide

The CLI runs the same pipeline as the [web app](../README.md), step by step, with every parameter exposed as a flag. Use it for sweeps over retrain strategies, headless servers, scripted experiments and any more exhaustive analysis than the app's UI offers.

> New here? The [web app](../README.md) is the simplest way to run PADS. For the architecture, see [`PIPELINE_GUIDE.md`](PIPELINE_GUIDE.md).

---

## 1. Setup

The repo uses [`uv`](https://docs.astral.sh/uv/) for environment management. A one-shot script does everything — pick the one for your OS:

```powershell
.\scripts\setup.ps1     # Windows (PowerShell)
```
```bash
./scripts/setup.sh      # Linux / WSL2
```

Each script installs `uv` if missing, creates the virtualenv, installs `pads` in editable mode, and pulls every dependency (including the `dev` extra for tests/docs). Versions are pinned in `uv.lock` for reproducible builds.

**Windows and Linux use separate virtualenvs by design**, so the same checkout (e.g. on a `/mnt/c` mount shared between Windows and WSL) isn't wiped and rebuilt each time you switch OS:

- **Windows** → `.venv/` (uv's default).
- **Linux/WSL** → `.venv-linux/`. `setup.sh` adds `export UV_PROJECT_ENVIRONMENT=.venv-linux` to your `~/.bashrc`, so `uv run pads ...` targets the right venv automatically in every new shell.

`pads` auto-loads `.env` at startup, so your MLflow settings are picked up without any manual step.

---

## 2. Run the pipeline

Place your input data in `data/<your_file>.<extension>`. Then:

```powershell
# Whole pipeline end-to-end
uv run pads --data_filename synthetic_dataset.csv --mode all

# Step by step
uv run pads --data_filename synthetic_dataset.csv --mode prepare_data
uv run pads --data_filename synthetic_dataset.csv --mode retrain_models --retrain_type full --epochs 1000
uv run pads --data_filename synthetic_dataset.csv --mode calculate_metrics
uv run pads --data_filename synthetic_dataset.csv --mode inference --test_type last_96h
```

The same commands work on a laptop or a server. To compare retrain strategies in one go, use the sweep script:

```powershell
.\scripts\sweep.ps1 -Dataset synthetic_dataset.csv         # full,dense,lstm,scratch
```
```bash
./scripts/sweep.sh --dataset synthetic_dataset.csv         # Linux / WSL2
```

### Pipeline modes

| Mode                | Description                                                                                                          |
|---------------------|----------------------------------------------------------------------------------------------------------------------|
| `prepare_data`      | Train/test split, median imputation values, fitted MinMaxScalers, windowed `.pkl` datasets, and dataset provenance.  |
| `retrain_models`    | Retrains the mortality and discharge LSTMs. `--retrain_type` controls the freezing strategy.                         |
| `calculate_metrics` | Picks optimal thresholds, writes the test ROC and threshold JSON.                                                    |
| `inference`         | End-to-end prediction + error categorisation + plots.                                                                |
| `all`               | Runs the four steps above, in order.                                                                                 |

### Key flags

| Flag                                             | Choices                                          | Notes                                                                                            |
|--------------------------------------------------|--------------------------------------------------|--------------------------------------------------------------------------------------------------|
| `--data_filename`                                | any filename inside `data/`                      | CSV or Parquet, auto-detected by extension.                                                      |
| `--mode`                                          | `prepare_data` / `retrain_models` / `calculate_metrics` / `inference` / `all` | Pipeline step to run.                               |
| `--retrain_type`                                 | `full` / `dense` / `lstm` / `scratch` / `original` | Freezing strategy; `scratch` ignores the base model and trains from random init.               |
| `--test_type`                                    | `full` / `last_48h` / `last_96h` / `first_48h`   | Which window of each stay to score during inference.                                             |
| `--epochs`                                       | int                                              | Default 1000. Reduce for quick smoke tests.                                                      |
| `--batch_size`                                   | int                                              | Default 100.                                                                                      |
| `--learning_rate_mort` / `--learning_rate_disch` | float                                            | Per-model learning rate. Default 1e-5.                                                            |
| `--early_stopping_patience`                      | int                                              | Default 20.                                                                                       |
| `--seed`                                         | int                                              | Default 42. Applied to Python, NumPy and TF.                                                      |
| `--base_path`                                    | path                                             | Project root. Default `./`.                                                                       |

Run `uv run pads --help` for the complete list.

---

## 3. Inspect results

Outputs are grouped per model under `results/<retrain_type>/` (e.g. `results/full/`, `results/scratch/`), so a sweep keeps each variant side by side instead of overwriting:

| Artifact (under `results/<retrain_type>/`) | Meaning                                                                          |
|--------------------------------------------|----------------------------------------------------------------------------------|
| `results_inference.csv`                         | Per-prediction row with both probabilities, both categories, and severity 0..3.  |
| `model_parameters_inference.json`          | Decision thresholds and probability range.                                       |
| `images/roc_combined_<test_type>.png`      | Combined mortality + discharge ROC curves.                                       |
| `images/barplot_error.png`                 | Error severity distribution.                                                     |
| `images/heatmap_error.png`                 | Confusion-matrix-style scatter.                                                  |

---

## 4. MLflow tracking (opt-in)

MLflow is a no-op unless `MLFLOW_TRACKING_URI` is set. The easiest way is to put it in `.env` (copied from `.env.example` by the setup script) — `pads` auto-loads it at startup:

```env
# .env
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
PADS_MLFLOW_EXPERIMENT=pads_retraining
PADS_RUN_TAGS=environment=local,operator=julen
```

```powershell
uv run pads --data_filename synthetic_dataset.csv --mode all
# at startup you'll see: [PADS] MLflow tracking ON: http://127.0.0.1:5000
```

At startup `pads` probes the server: if it's set but unreachable it prints a warning and **continues without tracking** (your models/artifacts are still written to disk) rather than crashing mid-run. When tracking is on, every step logs its params, metrics, and key artifacts (ROC, `results_inference.csv`, retrained `.keras` files). The first 16 hex chars of the dataset's SHA-256 are tagged on every run so you can answer "which dataset trained this model?" from the UI.

> The web app launches its own local MLflow server automatically (see [README](../README.md)), so you only need this section when running the CLI standalone.

See [`PIPELINE_GUIDE.md`](PIPELINE_GUIDE.md) section 5 for the full reference, including how to register models for Production.

---

## 5. Testing

```powershell
uv run pytest -m "not slow"      # fast feedback (skip E2E)
uv run pytest                    # everything (~15 s)
```

End-to-end tests are marked `slow` + `e2e` and require `models/lstm_mortality_model.keras` + `models/lstm_disch_model.keras` to be present (they `pytest.skip()` cleanly otherwise).

---

## 6. Project layout

```
PADS/
├── pads/                       # the installable package (pipeline + CLI)
│   ├── cli.py                  # pads --data_filename ... --mode ...
│   ├── config.py               # PADSConfig (pydantic)
│   ├── pipeline.py             # step orchestrator
│   ├── tracking.py             # MLflow wrapper (no-op when disabled)
│   ├── data/                   # schema, loader, preprocess, windowing, normalizer
│   ├── models/                 # mortality LSTM, discharge LSTM, registry
│   ├── training/               # splits, trainer
│   ├── eval/                   # metrics, thresholds, error categorisation
│   └── viz/                    # ROC, error bar+heatmap
├── pads_app/                   # FastAPI web app (frontend + JSON API)
├── tests/                      # unit + E2E tests
├── scripts/
│   ├── setup.ps1 / setup.sh    # one-shot environment setup
│   ├── app.ps1 / app.sh        # launch MLflow + web app
│   ├── sweep.ps1 / sweep.sh    # run the pipeline over all retrain_types
│   └── build_pdf.py            # rebuild docs/PIPELINE_GUIDE.pdf
├── data/                       # raw input file
│   └── processed/              # generated intermediate artifacts (medians, splits, .pkl)
├── normalizers/                # fitted MinMaxScalers
├── models/                     # base + retrained .keras
├── results/<retrain_type>/     # results_inference.csv, params JSON + ROC/error plots (per model)
├── docs/
│   ├── CLI_GUIDE.md            # this file
│   ├── PIPELINE_GUIDE.md       # full technical guide
│   └── PIPELINE_GUIDE.pdf      # same, as PDF
├── pyproject.toml              # package + tooling config
└── uv.lock                     # exact resolved versions (commit to git)
```
