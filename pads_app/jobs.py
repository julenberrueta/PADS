"""Background job runner: drives the PADS CLI as subprocesses, with a
disk-backed history so past runs survive an app restart.

Why subprocesses instead of calling the pipeline in-process: TensorFlow keeps
global state (graph, seeds, GPU memory) that does not survive being re-run in
the same process, and a crash in training would take the web server down with
it. One subprocess per CLI step isolates all of that and lets us stream logs.

Persistence: each job is written to ``<state_dir>/jobs/<id>.json`` (metadata)
and ``<id>.log`` (full output), so the frontend can list and reopen old runs.
The job's MLflow runs are stamped with ``pads_job_id=<id>`` (via PADS_RUN_TAGS,
which tracking.py already reads), so metrics/artifacts are recovered from MLflow.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pads_app.config import get_settings
from pads_app.mlflow_api import JOB_TAG


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Inference is always run for every window type so the user can compare them.
TEST_TYPES = ("full", "last_48h", "last_96h", "first_48h")


@dataclass
class TrainParams:
    data_filename: str
    retrain_types: list[str]
    epochs: int = 1000
    batch_size: int = 100
    learning_rate: float = 1e-5
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_filename": self.data_filename,
            "retrain_types": self.retrain_types,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
        }


@dataclass
class Job:
    id: str
    params: TrainParams
    status: str = "running"  # running | succeeded | failed | cancelled | interrupted
    steps: list[str] = field(default_factory=list)
    current_step: int = -1
    created_at: str = field(default_factory=_now)
    finished_at: str | None = None
    error: str | None = None
    _log: deque[str] = field(default_factory=lambda: deque(maxlen=8000), repr=False)
    _proc: subprocess.Popen | None = field(default=None, repr=False)

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "steps": self.steps,
            "current_step": self.current_step,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "params": self.params.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Job":
        import dataclasses
        known = {f.name for f in dataclasses.fields(TrainParams)}
        params = {k: v for k, v in d["params"].items() if k in known}  # tolerate old keys
        return cls(
            id=d["id"],
            params=TrainParams(**params),
            status=d.get("status", "interrupted"),
            steps=d.get("steps", []),
            current_step=d.get("current_step", -1),
            created_at=d.get("created_at", _now()),
            finished_at=d.get("finished_at"),
            error=d.get("error"),
        )


class JobBusyError(RuntimeError):
    """Raised when a job is asked to start while another is still running."""


class JobManager:
    """Owns all jobs for the process, backed by a directory on disk.

    v1 runs at most one job at a time (the pipeline writes to shared folders).
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._dir = get_settings().state_dir / "jobs"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._load()

    # --- public API ---------------------------------------------------------
    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[dict[str, Any]]:
        return [j.public() for j in sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)]

    def log(self, job_id: str) -> list[str]:
        job = self._jobs.get(job_id)
        if job is not None and job._log:
            return list(job._log)
        # Fall back to the persisted log file (e.g. a run from before a restart).
        p = self._log_path(job_id)
        if p.exists():
            return p.read_text(encoding="utf-8", errors="replace").splitlines()
        return []

    def start(self, params: TrainParams) -> Job:
        settings = get_settings()
        with self._lock:
            if self._running_count() >= settings.max_concurrent_jobs:
                raise JobBusyError("Another training job is already running.")
            job_id = uuid.uuid4().hex[:12]
            job = Job(id=job_id, params=params, steps=_step_labels(params))
            self._jobs[job_id] = job
            self._log_path(job_id).write_text("", encoding="utf-8")  # fresh log
            self._save(job)

        thread = threading.Thread(target=self._run, args=(job,), daemon=True)
        thread.start()
        return job

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None or job.status != "running":
            return False
        if job._proc is not None:
            job._proc.terminate()
        job.status = "cancelled"
        job.finished_at = _now()
        self._save(job)
        return True

    def delete(self, job_id: str) -> bool:
        """Remove a job from the app history and delete its MLflow runs."""
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if job.status == "running":
            self.cancel(job_id)
        # Drop the MLflow runs first (best-effort), then the local record.
        from pads_app import mlflow_api
        mlflow_api.delete_job_runs(job_id)
        self._jobs.pop(job_id, None)
        self._meta_path(job_id).unlink(missing_ok=True)
        self._log_path(job_id).unlink(missing_ok=True)
        return True

    # --- persistence --------------------------------------------------------
    def _meta_path(self, job_id: str) -> Path:
        return self._dir / f"{job_id}.json"

    def _log_path(self, job_id: str) -> Path:
        return self._dir / f"{job_id}.log"

    def _save(self, job: Job) -> None:
        self._meta_path(job.id).write_text(json.dumps(job.public(), indent=2), encoding="utf-8")

    def _load(self) -> None:
        for p in sorted(self._dir.glob("*.json")):
            try:
                job = Job.from_dict(json.loads(p.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001 - skip corrupt entries
                continue
            # A job marked 'running' on disk means the app died mid-run: its
            # subprocess is gone, so reflect that instead of a false 'running'.
            if job.status == "running":
                job.status = "interrupted"
                job.finished_at = job.finished_at or _now()
                job.error = job.error or "App restarted while this job was running."
                self._save(job)
            self._jobs[job.id] = job

    # --- run loop -----------------------------------------------------------
    def _running_count(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status == "running")

    def _emit(self, job: Job, text: str) -> None:
        for line in text.splitlines() or [""]:
            job._log.append(line)
        with open(self._log_path(job.id), "a", encoding="utf-8") as fh:
            fh.write(text if text.endswith("\n") else text + "\n")

    def _run(self, job: Job) -> None:
        env = _subprocess_env(job.id)
        try:
            for idx, argv in enumerate(_commands(job.params)):
                if job.status == "cancelled":
                    break
                job.current_step = idx
                self._save(job)
                self._emit(job, f"\n$ {' '.join(argv)}")
                rc = self._stream(job, argv, env)
                if rc != 0:
                    job.status = "failed"
                    detail = _last_error_line(job._log)
                    job.error = f"Step '{job.steps[idx]}' failed (exit {rc})."
                    if detail:
                        job.error += f" {detail}"
                    break
            if job.status == "running":
                job.status = "succeeded"
                job.current_step = len(job.steps)
        except Exception as exc:  # pragma: no cover - defensive
            job.status = "failed"
            job.error = repr(exc)
        finally:
            job.finished_at = _now()
            self._save(job)

    def _stream(self, job: Job, argv: list[str], env: dict[str, str]) -> int:
        proc = subprocess.Popen(
            argv,
            cwd=str(get_settings().base_path),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        job._proc = proc
        assert proc.stdout is not None
        with open(self._log_path(job.id), "a", encoding="utf-8") as fh:
            for line in proc.stdout:
                clean = line.rstrip("\n")
                job._log.append(clean)
                fh.write(clean + "\n")
        proc.wait()
        job._proc = None
        return proc.returncode


# --- command construction ---------------------------------------------------
def _step_labels(p: TrainParams) -> list[str]:
    labels = ["prepare_data"]
    for rt in p.retrain_types:
        labels += [f"retrain_models ({rt})", f"calculate_metrics ({rt})"]
        labels += [f"inference ({rt}, {tt})" for tt in TEST_TYPES]
    return labels


def _commands(p: TrainParams) -> list[list[str]]:
    base = [
        sys.executable, "-m", "pads.cli",
        "--data_filename", p.data_filename,
        "--base_path", str(get_settings().base_path),
        "--seed", str(p.seed),
    ]
    cmds: list[list[str]] = [base + ["--mode", "prepare_data"]]
    for rt in p.retrain_types:
        cmds.append(base + [
            "--mode", "retrain_models", "--retrain_type", rt,
            "--epochs", str(p.epochs), "--batch_size", str(p.batch_size),
            "--learning_rate_mort", str(p.learning_rate),
            "--learning_rate_disch", str(p.learning_rate),
        ])
        cmds.append(base + ["--mode", "calculate_metrics", "--retrain_type", rt])
        for tt in TEST_TYPES:
            cmds.append(base + ["--mode", "inference", "--retrain_type", rt, "--test_type", tt])
    return cmds


# Matches the final line of a Python traceback, e.g. "ValueError: Training ...".
_EXC_RE = re.compile(r"^[A-Za-z_][\w.]*(Error|Exception|Interrupt):\s")


def _last_error_line(log: "deque[str]", max_len: int = 400) -> str:
    """Best-effort one-line reason for a failed step, for the UI popup.

    Prefers the exception line at the end of a Python traceback; otherwise
    falls back to the last non-empty log line. Truncated to keep popups sane.
    """
    last_nonempty = ""
    for line in reversed(log):
        s = line.strip()
        if not s:
            continue
        if not last_nonempty:
            last_nonempty = s
        if _EXC_RE.match(s):
            return s[:max_len]
    return last_nonempty[:max_len]


def _subprocess_env(job_id: str) -> dict[str, str]:
    env = dict(os.environ)
    # Stamp every run of this job so the frontend can find them in MLflow.
    existing = env.get("PADS_RUN_TAGS", "").strip()
    job_tag = f"{JOB_TAG}={job_id}"
    env["PADS_RUN_TAGS"] = f"{existing},{job_tag}" if existing else job_tag
    # Windows + MLflow emoji output needs UTF-8 (mirrors .env.example).
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")
    return env


# Module-level singleton shared by the FastAPI app.
manager = JobManager()
