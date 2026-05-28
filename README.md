# PADS — Patient Outcome Assessment and Decision Support

ICU **mortality** and **time-to-discharge** prediction from two LSTM models that share the same 48-hour time series per stay. Retrain the models on your own data and get per-patient risk categories, ROC curves and error analysis.

> **Two ways to run PADS:**
>
> - 🖥️ **Web app** — drag in a dataset, click **Run**, watch metrics live, download everything in one `.zip`. **This README covers the app.**
> - ⌨️ **Command line** — for sweeps, servers and more exhaustive analysis. See **[`docs/CLI_GUIDE.md`](docs/CLI_GUIDE.md)**.
>
> Architecture deep-dive: **[`docs/PIPELINE_GUIDE.md`](docs/PIPELINE_GUIDE.md)**.

---

## What PADS does

Two binary LSTM classifiers consume the same 48-hour multivariate time series per ICU stay:

| Model      | Predicts                                          | Output                   |
|------------|---------------------------------------------------|--------------------------|
| Mortality  | Will this patient die in the ICU in the next 48 h?| P(exitus ≤ 48 h) ∈ [0, 1]|
| Discharge  | Will this patient leave the ICU in the next 48 h? | P(disch ≤ 48 h) ∈ [0, 1] |

The two outputs combine into **four clinical categories** — `EXITUS <48h`, `EXITUS >48h`, `ALIVE <48h`, `ALIVE >48h` — each with a different recommended action and an error severity from 0 to 3 (see [`docs/PIPELINE_GUIDE.md`](docs/PIPELINE_GUIDE.md)).

---

## Quick start (web app)

Three steps. No Docker, no MinIO — everything runs locally via [`uv`](https://docs.astral.sh/uv/) with a bundled MLflow server (SQLite + local files).

### 1. Setup — once

```powershell
.\scripts\setup.ps1     # Windows (PowerShell)
```
```bash
./scripts/setup.sh      # Linux / WSL2
```

Installs `uv` if missing, creates the virtualenv and pulls every dependency (the web app included).

### 2. Launch

```powershell
.\scripts\app.ps1       # Windows — starts MLflow (:5000) + app (:8000)
```
```bash
./scripts/app.sh        # Linux / WSL2
```

### 3. Open the app

Go to **<http://127.0.0.1:8000>** and:

1. **Drop your dataset** (`.csv` / `.parquet` / `.xlsx` / `.pkl`) — it's validated on the spot.
2. **Pick options** — retrain type(s), epochs, learning rate. Selecting several retrain types trains them one after another.
3. **Click Run** and watch per-epoch metrics and results stream in.
4. **Download all** — one button bundles the retrained models, metrics, ROC/error plots and `results_inference.csv` into a single `.zip`, organised per retrain type.

Stop the app (and its MLflow server) with `Ctrl+C` in the terminal.

---

## Input data

PADS was developed on MIMIC-IV ([Johnson et al. 2023](https://www.nature.com/articles/s41597-022-01899-x)), available via [PhysioNet](https://physionet.org/content/mimiciv/3.1/). Stays shorter than 48 h and patients with life-support treatment-limitation orders are excluded. Outlier clipping, first-hour imputation and per-column NaN rules are applied automatically.

Your file must contain every column below, all numeric:

<details>
<summary><strong>Required columns</strong> (click to expand)</summary>

| Variable                             | Units      | Range         | Notes                       |
|--------------------------------------|------------|---------------|-----------------------------|
| `stay_id`                            | —          | —             | MIMIC-IV stay identifier.   |
| `hr`                                 | hours      | ≥ 0           | Hours since admission.      |
| `rate_epinephrine`                   | mcg/kg/min | 0 – 4.64      |                             |
| `rate_norepinephrine`                | mcg/kg/min | 0 – 21.19     |                             |
| `rate_dopamine`                      | mcg/kg/min | 0.2 – 1069.52 |                             |
| `rate_dobutamine`                    | mcg/kg/min | 0.1 – 40.22   |                             |
| `meanbp_min`                         | mmHg       | 0.25 – 299    |                             |
| `pao2fio2ratio_novent`               | —          | 8 – 1706      |                             |
| `pao2fio2ratio_vent`                 | —          | 1 – 2104      |                             |
| `gcs_min`                            | —          | 3 – 15        |                             |
| `bilirubin_max`                      | mg/dL      | 0.1 – 87.2    |                             |
| `creatinine_max`                     | mg/dL      | 0.1 – 80      |                             |
| `platelet_min`                       | K/uL       | 5 – 2360      |                             |
| `admission_age`                      | years      | 18 – 102      |                             |
| `charlson_comorbidity_index`         | —          | 0 – 20        |                             |
| `admission_type_Medical`             | binary     | 0/1           | One-hot of admission type.  |
| `admission_type_ScheduledSurgical`   | binary     | 0/1           |                             |
| `admission_type_UnscheduledSurgical` | binary     | 0/1           |                             |
| `icu_expire_flag`                    | binary     | 0/1           | Ground truth for mortality. |

The authoritative list lives in `pads.data.schema.MANDATORY_COLUMNS`.

</details>

---

## Acknowledgement

If you use code or concepts from this repository, please cite the PADS paper: [https://doi.org/10.3390/jcm14134515](https://doi.org/10.3390/jcm14134515)

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

## License

See [LICENSE](LICENSE).
