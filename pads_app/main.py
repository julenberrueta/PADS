"""FastAPI app: serves the frontend and the JSON API that drives the pipeline."""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pads.data.schema import N_TIME_OFFSETS, DatasetValidationError, validate_dataset
from pads_app import mlflow_api
from pads_app.config import get_settings
from pads_app.jobs import (
    MONITOR_METRICS,
    NORMALIZER_SOURCES,
    THRESHOLD_METHODS,
    JobBusyError,
    TrainParams,
    manager,
)

_HERE = Path(__file__).parent
app = FastAPI(title="PADS Trainer")
app.mount("/static", StaticFiles(directory=_HERE / "static"), name="static")
templates = Jinja2Templates(directory=str(_HERE / "templates"))

# Per-epoch curves are logged under these prefixes by MLflowEpochLogger.
_CURVE_PREFIXES = ("mort/", "disch/")


@app.get("/")
def index(request: Request):
    settings = get_settings()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "mlflow_enabled": settings.mlflow_enabled,
            "tracking_uri": settings.tracking_uri or "",
            "experiment": settings.experiment,
        },
    )


@app.post("/api/validate")
async def api_validate(file: UploadFile):
    """Schema-check an uploaded dataset without saving it into the project."""
    info = _check_dataset(await file.read(), file.filename or "dataset.csv")
    return JSONResponse(info, status_code=200 if info["ok"] else 422)


@app.post("/api/train")
async def api_train(
    file: UploadFile,
    # Comma-separated, e.g. "full,scratch". May be empty for a baseline-only run;
    # the "nothing to do" case is validated explicitly below (a required Form here
    # would reject an empty value as missing before we can give a clear message).
    retrain_types: str = Form(""),
    epochs: int = Form(1000),
    batch_size: int = Form(100),
    # Per-model learning rates. The basic UI sends the same value for both.
    learning_rate_mort: float = Form(1e-5),
    learning_rate_disch: float = Form(1e-5),
    early_stopping_patience: int = Form(20),
    # Validation metric EarlyStopping/ModelCheckpoint track; see MONITOR_METRICS.
    monitor_metric: str = Form("loss"),
    normalizer_source: str = Form("fitted"),  # "fitted" | "mimic_iv"
    # Threshold criterion for retrained models; see THRESHOLD_METHODS.
    threshold_method: str = Form("precision_recall"),
    evaluate_original: bool = Form(False),
    seed: int = Form(42),
):
    settings = get_settings()
    raw = await file.read()
    filename = Path(file.filename or "dataset.csv").name

    check = _check_dataset(raw, filename)
    if not check["ok"]:
        raise HTTPException(status_code=422, detail=check["error"])

    types = [t.strip() for t in retrain_types.split(",") if t.strip()]
    # A run needs something to do: at least one retrain type, or the baseline alone.
    if not types and not evaluate_original:
        raise HTTPException(
            status_code=422,
            detail="Select at least one retrain type, or enable the baseline evaluation.",
        )

    if normalizer_source not in NORMALIZER_SOURCES:
        raise HTTPException(
            status_code=422,
            detail=f"normalizer_source must be one of {NORMALIZER_SOURCES}.",
        )

    if threshold_method not in THRESHOLD_METHODS:
        raise HTTPException(
            status_code=422,
            detail=f"threshold_method must be one of {THRESHOLD_METHODS}.",
        )

    if monitor_metric not in MONITOR_METRICS:
        raise HTTPException(
            status_code=422,
            detail=f"monitor_metric must be one of {MONITOR_METRICS}.",
        )

    # Persist the dataset where the pipeline expects it: <base_path>/data/.
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    (settings.data_dir / filename).write_bytes(raw)

    params = TrainParams(
        data_filename=filename,
        retrain_types=types,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate_mort=learning_rate_mort,
        learning_rate_disch=learning_rate_disch,
        early_stopping_patience=early_stopping_patience,
        monitor_metric=monitor_metric,
        normalizer_source=normalizer_source,
        threshold_method=threshold_method,
        evaluate_original=evaluate_original,
        seed=seed,
    )
    try:
        job = manager.start(params)
    except JobBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return job.public()


@app.get("/api/jobs")
def api_jobs():
    return manager.list()


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return job.public()


@app.delete("/api/jobs/{job_id}")
def api_job_delete(job_id: str):
    if not manager.delete(job_id):
        raise HTTPException(status_code=404, detail="Unknown job")
    return {"ok": True}


@app.get("/api/jobs/{job_id}/log")
def api_job_log(job_id: str, tail: int = 400):
    if manager.get(job_id) is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    lines = manager.log(job_id)
    return {"lines": lines[-tail:]}


@app.post("/api/jobs/{job_id}/cancel")
def api_job_cancel(job_id: str):
    if not manager.cancel(job_id):
        raise HTTPException(status_code=409, detail="Job is not running")
    return {"ok": True}


@app.get("/api/jobs/{job_id}/metrics")
def api_job_metrics(job_id: str):
    """Live metrics for the job's MLflow runs: per-epoch curves + final values."""
    runs = mlflow_api.job_runs(job_id)
    for run in runs:
        history: dict[str, list] = {}
        for key in run["metrics"]:
            if key.startswith(_CURVE_PREFIXES):
                history[key] = mlflow_api.metric_history(run["run_id"], key)
        run["history"] = history
    return {"mlflow_enabled": get_settings().mlflow_enabled, "runs": runs}


@app.get("/api/runs/{run_id}/artifacts")
def api_artifacts(run_id: str, path: str = ""):
    return {"artifacts": mlflow_api.list_artifacts(run_id, path)}


@app.get("/api/runs/{run_id}/charts")
def api_run_charts(run_id: str):
    """ROC + error chart data for one run, so the UI can draw them in JS."""
    if not get_settings().mlflow_enabled:
        return {"roc": None, "error_bars": None, "error_heatmap": None}
    try:
        return mlflow_api.run_charts(run_id)
    except Exception as exc:  # noqa: BLE001 - surface any MLflow/parse error
        raise HTTPException(status_code=404, detail=f"Could not build charts: {exc}") from exc


@app.get("/api/runs/{run_id}/download")
def api_download(run_id: str, path: str):
    try:
        local = mlflow_api.download_artifact(run_id, path)
    except Exception as exc:  # noqa: BLE001 - surface any MLflow error to the client
        raise HTTPException(status_code=404, detail=f"Artifact not found: {exc}") from exc
    # Serve inline (no forced attachment) so images can be previewed/opened in
    # the browser; the frontend's `download` attribute handles actual downloads.
    return FileResponse(local)


@app.get("/api/successful-runs")
def api_successful_runs():
    """All finished inference runs across succeeded jobs, ranked by mean error (asc)."""
    return {"runs": mlflow_api.successful_inference_runs(manager.list())}


@app.get("/api/jobs/{job_id}/comparison")
def api_job_comparison(job_id: str):
    """ROC curves + mean error per model (original vs retrained) for one job."""
    if not get_settings().mlflow_enabled:
        return {"models": []}
    return mlflow_api.job_comparison(job_id)


@app.get("/api/jobs/{job_id}/download-all")
def api_job_download_all(job_id: str):
    """Bundle artifacts from every run of a job (models, images, metrics) into one .zip."""
    try:
        zip_path = mlflow_api.download_job_zip(job_id)
    except Exception as exc:  # noqa: BLE001 - surface any MLflow error to the client
        raise HTTPException(status_code=404, detail=f"Could not build archive: {exc}") from exc
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"pads_job_{job_id}.zip",
    )


# --- helpers ----------------------------------------------------------------
def _check_dataset(raw: bytes, filename: str) -> dict:
    """Validate raw bytes via the existing schema validator.

    Returns {ok, rows, stays} on success, {ok: False, error} on failure.
    """
    suffix = Path(filename).suffix or ".csv"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)
    try:
        df = validate_dataset(tmp_path)
        # Mirror the pipeline's two patient filters so the UI can preview them:
        #   1. prepare_data drops stays with a missing mortality label.
        #   2. the window builders keep only stays with los (= max hr) >= 48 h.
        los = df.groupby("stay_id")["hr"].max()
        null_stays = set(df.loc[df["icu_expire_flag"].isna(), "stay_id"].dropna().unique())
        kept = [s for s in los.index if s not in null_stays]
        short_stays = [s for s in kept if los[s] < N_TIME_OFFSETS]
        final = [s for s in kept if los[s] >= N_TIME_OFFSETS]
        # When the dataset carries a hospital_episode_id, the split groups whole
        # episodes into one fold. Report how many of the used patients have more
        # than one ICU stay so the UI can flag the grouping.
        has_episode = "hospital_episode_id" in df.columns
        multi_stay_patients = 0
        if has_episode and final:
            ep = df.groupby("stay_id")["hospital_episode_id"].first().loc[final]
            multi_stay_patients = int((ep.value_counts() > 1).sum())
        return {
            "ok": True,
            "filename": Path(filename).name,
            "rows": int(len(df)),
            "stays": int(df["stay_id"].nunique()),
            "dropped_stays": len(null_stays),       # missing icu_expire_flag
            "short_stays": len(short_stays),        # stay shorter than 48 h
            "final_stays": len(final),
            "has_episode_id": has_episode,          # hospital_episode_id present
            "multi_stay_patients": multi_stay_patients,  # patients with >1 ICU stay
        }
    except DatasetValidationError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - bad file, unreadable parquet, etc.
        return {"ok": False, "error": f"Could not read dataset: {exc}"}
    finally:
        tmp_path.unlink(missing_ok=True)


def run() -> None:
    """Console-script entry point: `pads-app` (or `uv run pads-app`)."""
    import os

    import uvicorn

    uvicorn.run(
        "pads_app.main:app",
        host=os.getenv("PADS_APP_HOST", "127.0.0.1"),
        port=int(os.getenv("PADS_APP_PORT", "8000")),
        reload=bool(os.getenv("PADS_APP_RELOAD")),
    )
