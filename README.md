# PADS — Patient Outcome Assessment and Decision Support

ICU mortality and time-to-discharge prediction pipeline based on two LSTM models, with optional MLflow tracking. Steps are run by hand through a small CLI.

> **Looking for the deep dive?** See [`docs/PIPELINE_GUIDE.md`](docs/PIPELINE_GUIDE.md) (also available as [`docs/PIPELINE_GUIDE.pdf`](docs/PIPELINE_GUIDE.pdf)).

---

## Acknowledgement

If you use code or concepts available in this repository, please cite the PADS paper: [https://doi.org/10.3390/jcm14134515](https://doi.org/10.3390/jcm14134515)

```bibtex
@article{jcm14134515,
  author    = {Pardo, Àlex and Gómez, Josep and Berrueta, Julen and García, Alejandro and Manrique, Sara and Rodríguez, Alejandro and Bodí, María},
  title     = {Combining Predictive Models of Mortality and Time-to-Discharge for Improved Outcome Assessment in Intensive Care Units},
  journal   = {Journal of Clinical Medicine},
  volume    = {14},
  year      = {2025},
  number    = {13},
  articleno = {4515},
  url       = {https://www.mdpi.com/2077-0383/14/13/4515},
  pubmedid  = {40648890},
  issn      = {2077-0383},
  doi       = {10.3390/jcm14134515}
}
```

---

## What PADS does

Two binary LSTM classifiers consume the same 48-hour multivariate time series per ICU stay:

| Model      | Predicts                                                | Output                |
|------------|---------------------------------------------------------|-----------------------|
| Mortality  | Will this patient die in the ICU?                       | P(exitus) ∈ [0, 1]    |
| Discharge  | Will this patient leave the ICU in the next 48 h?       | P(disch ≤ 48 h) ∈ [0, 1] |

The two binary outputs are combined into **four clinical categories** (`EXITUS <48h`, `EXITUS >48h`, `ALIVE <48h`, `ALIVE >48h`), each driving a different clinical action. The severity of a prediction error (0..3) depends on which category is confused with which — see `pads/eval/errors.py`.

---

## Quick start

### 1. Setup

The repo uses [`uv`](https://docs.astral.sh/uv/) (a fast, modern Python package manager) for environment management. A one-shot script does everything — pick the one for your OS:

```powershell
.\scripts\setup.ps1     # Windows (PowerShell)
```
```bash
./scripts/setup.sh      # Linux / WSL2
```

Each script installs `uv` if missing, creates the virtualenv, installs `pads` in editable mode, and pulls every dependency (including the `dev` extra for tests/docs). MLflow and the parquet engine are core dependencies, so a plain `uv run` already has them. Versions are pinned in `uv.lock` for reproducible builds.

**Windows and Linux use separate virtualenvs by design**, so the same checkout (e.g. on a `/mnt/c` mount shared between Windows and WSL) doesn't get wiped and rebuilt each time you switch OS:

- **Windows** → `.venv/` (uv's default).
- **Linux/WSL** → `.venv-linux/`. `setup.sh` adds `export UV_PROJECT_ENVIRONMENT=.venv-linux` to your `~/.bashrc`, so after running it once, `uv run pads ...` targets the right venv automatically in every new shell.

After setup, run the pipeline the same way on both (see below) — and `pads` auto-loads `.env` at startup, so your MLflow settings are picked up without any manual step.

### 2. Run the pipeline

Place your input data in `data/<your_file>.csv` (or `.parquet`). The CLI accepts either:

```powershell
# Whole pipeline end-to-end
uv run pads --data_filename synthetic_dataset.csv --mode all

# Step by step
uv run pads --data_filename synthetic_dataset.csv --mode prepare_data
uv run pads --data_filename synthetic_dataset.csv --mode retrain_models --retrain_type full --epochs 1000
uv run pads --data_filename synthetic_dataset.csv --mode calculate_metrics
uv run pads --data_filename synthetic_dataset.csv --mode inference --test_type last_96h
```

Every run parameter (dataset, retrain type, epochs, batch size, seed) is a CLI flag — see the tables below. MLflow tracking is automatically enabled whenever `MLFLOW_TRACKING_URI` is set — either exported in the shell or placed in `.env` (auto-loaded at startup). The same commands work on a laptop or a server.

To compare the four retrain strategies in one go, use the sweep script:

```powershell
.\scripts\sweep.ps1 -Dataset synthetic_dataset.csv          # full,dense,lstm,scratch
```

### 3. Inspect results

Outputs are grouped per model under `results/<retrain_type>/` (e.g. `results/full/`, `results/scratch/`), so a sweep over retrain types keeps each variant side by side instead of overwriting:

| Artifact (under `results/<retrain_type>/`)     | Meaning                                                       |
|------------------------------------------------|---------------------------------------------------------------|
| `final_result.csv`                             | Per-prediction row with both probabilities, both categories, and severity 0..3. |
| `model_parameters_inference.json`              | Decision thresholds and probability range.                    |
| `images/roc_combined_<test_type>.png`          | Combined mortality + discharge ROC curves.                    |
| `images/barplot_error.png`                     | Error severity distribution.                                  |
| `images/heatmap_error.png`                     | Confusion-matrix-style scatter.                               |

---

## Pipeline modes

| Mode                    | Description                                                                                       |
|-------------------------|---------------------------------------------------------------------------------------------------|
| `prepare_data`          | Train/test split, median imputation values, fitted MinMaxScalers, windowed `.pkl` datasets, and dataset provenance. |
| `retrain_models`        | Retrains the mortality and discharge LSTMs. `--retrain_type` controls freezing strategy.          |
| `calculate_metrics`     | Picks optimal thresholds, writes test ROC and threshold JSON.                                     |
| `inference`             | End-to-end prediction + error categorisation + plots.                                             |
| `all`                   | Runs the four steps above, in order.                                                              |

### Key flags

| Flag                | Choices                                       | Notes                                                    |
|---------------------|-----------------------------------------------|----------------------------------------------------------|
| `--data_filename`   | any filename inside `data/`                   | CSV or Parquet auto-detected by extension.               |
| `--retrain_type`    | `full` / `dense` / `lstm` / `scratch`         | Freezing strategy for retraining; `scratch` ignores the base model and trains from random init. |
| `--test_type`       | `full` / `last_48h` / `last_96h` / `first_48h` | Which window of each stay to score during inference.    |
| `--epochs`          | int                                           | Default 1000. Reduce for quick smoke tests.              |
| `--batch_size`      | int                                           | Default 100.                                             |
| `--seed`            | int                                           | Default 42. Applied to Python, NumPy and TF.             |
| `--base_path`       | path                                          | Project root. Default `./`.                              |

---

## Input data

PADS was developed on MIMIC-IV ([Johnson et al. 2023](https://www.nature.com/articles/s41597-022-01899-x)), a freely available dataset from the Beth Israel Deaconess Medical Center. Access is provided via [PhysioNet](https://physionet.org/content/mimiciv/3.1/).

For training and evaluation we exclude:

- patients with stays shorter than 48 hours (by definition the model needs at least 48 hours of data);
- patients with life-support treatment limitation orders.

### Required schema

Your input file must contain every column in `pads.data.schema.MANDATORY_COLUMNS`, all numeric:

| Variable                                | Units      | Range          | Notes                              |
|-----------------------------------------|------------|----------------|-------------------------------------|
| `stay_id`                               | —          | —              | MIMIC-IV stay identifier.           |
| `hr`                                    | hours      | ≥ 0            | Hours since admission.              |
| `rate_epinephrine`                      | mcg/kg/min | 0 – 4.64       |                                     |
| `rate_norepinephrine`                   | mcg/kg/min | 0 – 21.19      |                                     |
| `rate_dopamine`                         | mcg/kg/min | 0.2 – 1069.52  |                                     |
| `rate_dobutamine`                       | mcg/kg/min | 0.1 – 40.22    |                                     |
| `meanbp_min`                            | mmHg       | 0.25 – 299     |                                     |
| `pao2fio2ratio_novent`                  | —          | 8 – 1706       |                                     |
| `pao2fio2ratio_vent`                    | —          | 1 – 2104       |                                     |
| `gcs_min`                               | —          | 3 – 15         |                                     |
| `bilirubin_max`                         | mg/dL      | 0.1 – 87.2     |                                     |
| `creatinine_max`                        | mg/dL      | 0.1 – 80       |                                     |
| `platelet_min`                          | K/uL       | 5 – 2360       |                                     |
| `admission_age`                         | years      | 18 – 102       |                                     |
| `charlson_comorbidity_index`            | —          | 0 – 20         |                                     |
| `admission_type_Medical`                | binary     | 0/1            | One-hot of admission type.          |
| `admission_type_ScheduledSurgical`      | binary     | 0/1            |                                     |
| `admission_type_UnscheduledSurgical`    | binary     | 0/1            |                                     |
| `icu_expire_flag`                       | binary     | 0/1            | Ground truth for mortality.         |

Preprocessing (outlier clipping → first-hour imputation → per-column NaN rules) is applied automatically. See `pads/data/preprocess.py`.

---

## Project layout

```
PADS/
├── pads/                       # the installable package
│   ├── cli.py                  # pads --data_filename ... --mode ...
│   ├── config.py               # PADSConfig (pydantic)
│   ├── pipeline.py             # 5-step orchestrator
│   ├── tracking.py             # MLflow wrapper (no-op when disabled)
│   ├── data/                   # schema, loader, preprocess, windowing, normalizer
│   ├── models/                 # mortality LSTM, discharge LSTM, registry
│   ├── training/               # splits, trainer
│   ├── eval/                   # metrics, thresholds, error categorisation
│   └── viz/                    # ROC, error bar+heatmap
├── tests/                      # 28 tests (24 unit + 4 E2E)
├── scripts/
│   ├── setup.ps1 / setup.sh    # one-shot environment setup
│   ├── sweep.ps1 / sweep.sh    # run the pipeline over all retrain_types
│   └── build_pdf.py            # rebuild docs/PIPELINE_GUIDE.pdf
├── data/                       # raw input file
│   └── processed/              # generated intermediate artifacts (medians, splits, .pkl)
├── normalizers/                # fitted MinMaxScalers
├── models/                     # base + retrained .keras
├── results/<retrain_type>/     # final_result.csv, params JSON + ROC/error plots (per model)
├── docs/
│   ├── PIPELINE_GUIDE.md       # full technical guide
│   └── PIPELINE_GUIDE.pdf      # same, as PDF
├── notebooks/                  # exploratory Jupyter notebooks
├── pyproject.toml              # package + tooling config
└── uv.lock                     # exact resolved versions (commit to git)
```

---

## Testing

```powershell
uv run pytest -m "not slow"      # fast feedback (skip E2E)
uv run pytest                    # everything (~15 s)
```

End-to-end tests are marked `slow` + `e2e` and require `models/lstm_mortality_model.keras` + `models/lstm_disch_model.keras` to be present (they `pytest.skip()` cleanly otherwise).

---

## MLflow tracking (opt-in)

MLflow is a no-op unless you set `MLFLOW_TRACKING_URI`. The easiest way is to put it in `.env` (copied from `.env.example` by the setup script) — `pads` auto-loads it at startup, so it just works:

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

At startup `pads` probes the server: if it's set but unreachable it prints a warning and **continues without tracking** (your models/artifacts are still written to disk) rather than crashing mid-run. When tracking is on, every step logs its params, metrics, and key artifacts (ROC, `final_result.csv`, retrained `.keras` files). The first 16 hex chars of the dataset's SHA-256 are tagged on every run so you can answer "which dataset trained this model?" from the UI.

See [`docs/PIPELINE_GUIDE.md`](docs/PIPELINE_GUIDE.md) section 5 for the full reference, including how to register models for Production.

---

## License

See [LICENSE](LICENSE).
