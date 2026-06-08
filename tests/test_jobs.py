"""Queue behaviour of the JobManager (no real training: threads are stubbed)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from pads_app import jobs
from pads_app.jobs import JobManager, TrainParams


class _NoThread:
    """Stand-in for threading.Thread whose .start() does nothing.

    Lets the test drive the queue synchronously: start() sets a job's initial
    status but never actually runs it, so promotion is exercised by calling
    _start_next_queued() by hand (what _run's finally block does for real).
    """

    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass


@pytest.fixture
def manager(tmp_path, monkeypatch):
    fake_settings = SimpleNamespace(
        state_dir=tmp_path, base_path=tmp_path, max_concurrent_jobs=1
    )
    monkeypatch.setattr(jobs, "get_settings", lambda: fake_settings)
    monkeypatch.setattr(jobs.threading, "Thread", _NoThread)
    return JobManager()


def _params() -> TrainParams:
    return TrainParams(data_filename="x.csv", retrain_types=["full"])


def test_second_submission_is_queued(manager):
    j1 = manager.start(_params())
    j2 = manager.start(_params())
    assert j1.status == "running"
    assert j2.status == "queued"


def test_finishing_a_job_promotes_the_queue(manager):
    j1 = manager.start(_params())
    j2 = manager.start(_params())
    # Simulate j1 finishing, then drain the queue like _run's finally does.
    j1.status = "succeeded"
    manager._start_next_queued()
    assert manager.get(j2.id).status == "running"


def test_queue_is_fifo(manager):
    j1 = manager.start(_params())     # running
    j2 = manager.start(_params())     # queued first
    j3 = manager.start(_params())     # queued second
    assert (j2.status, j3.status) == ("queued", "queued")
    # Free the slot → promotion picks the oldest queued (j2), j3 keeps waiting.
    j1.status = "succeeded"
    manager._start_next_queued()
    assert manager.get(j2.id).status == "running"
    assert manager.get(j3.id).status == "queued"


def test_cancel_queued_job_leaves_running_untouched(manager):
    j1 = manager.start(_params())
    j2 = manager.start(_params())
    assert manager.cancel(j2.id) is True
    assert manager.get(j2.id).status == "cancelled"
    assert manager.get(j1.id).status == "running"  # slot still held by j1


def test_no_promotion_while_slot_is_busy(manager):
    manager.start(_params())          # running, holds the only slot
    j2 = manager.start(_params())     # queued
    manager._start_next_queued()      # slot busy → must stay queued
    assert manager.get(j2.id).status == "queued"
