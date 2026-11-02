import asyncio
import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from foreman.core.run_log import (
    build_run_record,
    find_run,
    read_runs,
    render_text,
    summary_line,
    write_run,
)

T0 = datetime(2026, 11, 2, 10, 0, 0)


def _task(task_id, stage, worker, start, took, status="completed", error=None):
    return SimpleNamespace(
        id=task_id, stage=stage, worker_id=worker, status=status, error_message=error,
        assigned_at=T0 + timedelta(seconds=start),
        completed_at=T0 + timedelta(seconds=start + took) if status == "completed" else None,
    )


def _job(status="completed", completed_at=None):
    return SimpleNamespace(id="abcd1234-job", status=status, created_at=T0, completed_at=completed_at,
                           is_pipeline=True)


def _run():
    tasks = [
        _task("j_task_0_0", 0, "w1", 0, 1.0), _task("j_task_0_1", 0, "w2", 0, 3.0),
        _task("j_task_1_0", 1, "w2", 3, 2.0), _task("j_task_1_1", 1, "p1", 3, 4.0),
    ]
    failures = [SimpleNamespace(worker_id="w2", task_id="j_task_0_1", failed_at=T0, error_message="boom")]
    return build_run_record(
        _job(completed_at=T0 + timedelta(seconds=8)), tasks, failures=failures,
        stage_names={0: "cell_a", 1: "cell_b"},
        worker_info={"w1": {"kind": "pc", "platform": "Windows"}, "w2": {"kind": "pc", "platform": "Windows"},
                     "p1": {"kind": "mobile", "platform": "Android"}},
        dnn_scope="mobile", scheduler="FIFOScheduler", results_returned=2, results_missing=0, now=T0 + timedelta(seconds=9),
    )


def test_record_summarises_stages_workers_and_timing():
    record = _run()
    assert record["duration_seconds"] == 8.0 and record["throughput_tasks_per_s"] == 0.5
    assert record["tasks"] == {"total": 4, "completed": 4, "by_status": {"completed": 4}}
    first, second = record["stages"]
    assert (first["name"], first["tasks"], first["workers"]) == ("cell_a", 2, {"w1": 1, "w2": 1})
    assert first["task_seconds"]["max"] == 3.0 and first["task_seconds"]["mean"] == 2.0
    assert second["span_seconds"] == 4.0  # first assigned at 3 s, last completed at 7 s
    by_worker = {w["worker_id"]: w for w in record["workers"]}
    assert by_worker["p1"]["kind"] == "mobile" and by_worker["w2"]["tasks"] == 2
    assert by_worker["w2"]["busy_seconds"] == 5.0 and by_worker["w2"]["stages"] == {"0": 1, "1": 1}
    assert record["failures"]["count"] == 1 and record["failures"]["recent"][0]["error"] == "boom"
    assert record["results"] == {"returned": 2, "missing": 0}
    assert record["dnn_scope"] == "mobile" and record["scheduler"] == "FIFOScheduler"


def test_just_completed_job_uses_now_as_finish_time():
    record = build_run_record(_job(status="running"), [_task("t", 0, "w", 0, 2.0)], status="completed",
                              now=T0 + timedelta(seconds=10))
    assert record["status"] == "completed" and record["duration_seconds"] == 10.0
    assert record["completed_at"] is not None


def test_incomplete_tasks_and_task_errors_are_reported():
    pending = SimpleNamespace(id="t2", stage=0, worker_id=None, status="pending", assigned_at=None,
                              completed_at=None, error_message="timeout")
    record = build_run_record(_job(status="running"), [_task("t1", 0, "w", 0, 1.0), pending],
                              status="client_disconnected", now=T0 + timedelta(seconds=5))
    assert record["tasks"]["completed"] == 1 and record["tasks"]["by_status"] == {"completed": 1, "pending": 1}
    assert record["failures"]["task_errors"] == [{"task_id": "t2", "error": "timeout"}]
    assert "unassigned" in {w["worker_id"] for w in record["workers"]}
    assert render_text(record)  # renders without a completion time


def test_report_text_is_readable():
    text = render_text(_run())
    for expected in ("Run abcd1234-job", "cell_a", "cell_b", "mobile", "p1", "Failures: 1", "boom", "DNN worker scope mobile"):
        assert expected in text
    assert "4/4" in summary_line(_run()) and "failures=1" in summary_line(_run())


def test_write_read_and_find_round_trip(tmp_path):
    record = _run()
    report = write_run(record, tmp_path)
    assert report.exists() and "abcd1234" in report.name and report.read_text(encoding="utf-8") == render_text(record)
    other = dict(record, run_id="ffff0000-other")
    write_run(other, tmp_path)
    with open(tmp_path / "runs.jsonl", "a", encoding="utf-8") as handle:
        handle.write("{not json\n")  # a torn line must not hide the other runs
    assert [r["run_id"] for r in read_runs(tmp_path)] == ["abcd1234-job", "ffff0000-other"]
    assert [r["run_id"] for r in read_runs(tmp_path, limit=1)] == ["ffff0000-other"]
    assert find_run("abcd", tmp_path)["run_id"] == "abcd1234-job"
    assert find_run("nope", tmp_path) is None
    assert read_runs(tmp_path / "missing") == []


def test_collector_writes_a_record_from_the_database(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CROWDIO_RUN_LOG_DIR", str(tmp_path / "runs"))
    monkeypatch.delenv("CROWDIO_DNN_WORKERS", raising=False)

    async def scenario():
        from foreman.core.connection_manager import ConnectionManager
        from foreman.core.job_manager import JobManager
        from foreman.core.run_log import log_job_run
        from foreman.db.base import async_session, engine, init_db
        from foreman.db.models import JobModel, TaskModel, WorkerModel

        await init_db()
        async with async_session() as session:
            session.add(WorkerModel(id="phone-1", device_type="Android", os_type="Android"))
            session.add(JobModel(id="job-xyz", status="running", total_tasks=2, is_pipeline=False,
                                 created_at=T0))
            for index, took in enumerate((1.0, 2.0)):
                session.add(TaskModel(id=f"job-xyz_task_{index}", job_id="job-xyz", worker_id="phone-1",
                                      status="completed", stage=0, assigned_at=T0,
                                      completed_at=T0 + timedelta(seconds=took)))
            await session.commit()
        try:
            return await log_job_run(JobManager(), ConnectionManager(), None, "job-xyz",
                                     results_returned=2, results_missing=0, status="completed")
        finally:
            await engine.dispose()

    record = asyncio.run(scenario())
    assert record is not None and record["tasks"]["completed"] == 2
    worker = record["workers"][0]
    assert worker["worker_id"] == "phone-1" and worker["kind"] == "mobile"  # disconnected: kind from the DB row
    assert record["dnn_scope"] == "mobile"
    assert find_run("job-xyz", tmp_path / "runs") is not None


def test_collector_never_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    class Broken:
        async def get_job_info(self, job_id):
            raise RuntimeError("database is gone")

    from foreman.core.run_log import log_job_run

    assert asyncio.run(log_job_run(Broken(), None, None, "x")) is None
