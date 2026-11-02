"""Per-run log: one record per job, written when the job finishes.

Each record answers "what happened in this run": how long it took, how the
tasks were spread over stages and workers, which workers were mobile or PC,
how long tasks took, what failed and how many results came back.

Files (directory ``CROWDIO_RUN_LOG_DIR``, default ``logs/runs``):
  runs.jsonl                         one JSON record per run, appended
  <YYYYmmdd-HHMMSS>_<job8>.txt       the same run as a readable report

``build_run_record`` takes plain objects and does no I/O, so it is easy to test.
"""

from __future__ import annotations

import json
import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

DEFAULT_DIR = "logs/runs"
MAX_LISTED_FAILURES = 10


def log_dir(path: Optional[str | Path] = None) -> Path:
    return Path(path or os.getenv("CROWDIO_RUN_LOG_DIR") or DEFAULT_DIR)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat(timespec="seconds") if isinstance(value, datetime) else None


def _seconds(start: Optional[datetime], end: Optional[datetime]) -> Optional[float]:
    if isinstance(start, datetime) and isinstance(end, datetime):
        return max(0.0, (end - start).total_seconds())
    return None


def _stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "mean": round(statistics.fmean(ordered), 3),
        "p50": round(ordered[len(ordered) // 2], 3),
        "p95": round(ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))], 3),
        "max": round(ordered[-1], 3),
    }


def build_run_record(
    job: Any,
    tasks: Iterable[Any],
    *,
    failures: Iterable[Any] = (),
    stage_names: Optional[Mapping[int, str]] = None,
    worker_info: Optional[Mapping[str, Mapping[str, Any]]] = None,
    dnn_scope: Optional[str] = None,
    scheduler: Optional[str] = None,
    links: Optional[Mapping[str, Any]] = None,
    results_returned: Optional[int] = None,
    results_missing: Optional[int] = None,
    status: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Summarise a job and its tasks. Objects only need the listed attributes."""
    now = now or datetime.now()
    tasks = list(tasks)
    stage_names = stage_names or {}
    worker_info = worker_info or {}
    started = getattr(job, "created_at", None)
    finished = getattr(job, "completed_at", None)
    statuses = Counter(getattr(t, "status", "unknown") or "unknown" for t in tasks)
    completed = statuses.get("completed", 0)
    if status is None:
        status = getattr(job, "status", None) or ("completed" if completed == len(tasks) else "incomplete")
    if finished is None and status == "completed":
        finished = now  # finalize_job stamps completed_at only after the run log is written
    duration = _seconds(started, finished or now)

    by_stage: Dict[int, List[Any]] = defaultdict(list)
    for task in tasks:
        by_stage[int(getattr(task, "stage", 0) or 0)].append(task)
    stages: List[Dict[str, Any]] = []
    for stage_index in sorted(by_stage):
        group = by_stage[stage_index]
        times = [
            took for took in (
                _seconds(getattr(t, "assigned_at", None), getattr(t, "completed_at", None))
                for t in group if getattr(t, "status", "") == "completed"
            ) if took is not None
        ]
        assigned = [t.assigned_at for t in group if getattr(t, "assigned_at", None)]
        done = [t.completed_at for t in group if getattr(t, "completed_at", None)]
        workers = Counter(getattr(t, "worker_id", None) or "unassigned" for t in group)
        stages.append({
            "stage": stage_index,
            "name": stage_names.get(stage_index),
            "tasks": len(group),
            "completed": sum(1 for t in group if getattr(t, "status", "") == "completed"),
            "workers": dict(sorted(workers.items())),
            "task_seconds": _stats(times),
            "span_seconds": _seconds(min(assigned), max(done)) if assigned and done else None,
        })

    per_worker: Dict[str, Dict[str, Any]] = {}
    for task in tasks:
        worker_id = getattr(task, "worker_id", None) or "unassigned"
        entry = per_worker.setdefault(worker_id, {"tasks": 0, "busy_seconds": 0.0, "stages": Counter()})
        entry["tasks"] += 1
        entry["stages"][int(getattr(task, "stage", 0) or 0)] += 1
        busy = _seconds(getattr(task, "assigned_at", None), getattr(task, "completed_at", None))
        if busy is not None and getattr(task, "status", "") == "completed":
            entry["busy_seconds"] += busy
    workers = []
    for worker_id, entry in sorted(per_worker.items()):
        info = worker_info.get(worker_id, {})
        workers.append({
            "worker_id": worker_id,
            "kind": info.get("kind", "unknown"),
            "platform": info.get("platform"),
            "tasks": entry["tasks"],
            "busy_seconds": round(entry["busy_seconds"], 3),
            "stages": {str(k): v for k, v in sorted(entry["stages"].items())},
        })

    failures = list(failures)
    failure_items = [
        {"worker_id": getattr(f, "worker_id", None), "task_id": getattr(f, "task_id", None),
         "at": _iso(getattr(f, "failed_at", None)), "error": str(getattr(f, "error_message", ""))[:300]}
        for f in failures[:MAX_LISTED_FAILURES]
    ]
    task_errors = [
        {"task_id": t.id, "error": str(t.error_message)[:300]}
        for t in tasks if getattr(t, "error_message", None)
    ][:MAX_LISTED_FAILURES]

    return {
        "run_id": getattr(job, "id", None),
        "logged_at": _iso(now),
        "status": status,
        "started_at": _iso(started),
        "completed_at": _iso(finished),
        "duration_seconds": round(duration, 3) if duration is not None else None,
        "tasks": {"total": len(tasks), "completed": completed, "by_status": dict(statuses)},
        "throughput_tasks_per_s": round(completed / duration, 3) if duration else None,
        "pipeline": bool(getattr(job, "is_pipeline", False)),
        "stages": stages,
        "workers": workers,
        "failures": {"count": len(failures), "recent": failure_items, "task_errors": task_errors},
        "results": {"returned": results_returned, "missing": results_missing},
        "dnn_scope": dnn_scope,
        "scheduler": scheduler,
        "links": links,  # the scheduler's bandwidth/compute estimates when the run ended (or None)
    }


def render_text(record: Mapping[str, Any]) -> str:
    """Readable report for one run."""
    tasks = record["tasks"]
    lines = [
        f"Run {record['run_id']}",
        f"  status      {record['status']}",
        f"  started     {record['started_at']}    finished {record['completed_at'] or '-'}",
        f"  duration    {_fmt(record['duration_seconds'])}    throughput {_fmt(record['throughput_tasks_per_s'], ' tasks/s')}",
        f"  tasks       {tasks['completed']}/{tasks['total']} completed   {tasks['by_status']}",
        f"  scheduler   {record.get('scheduler') or '-'}    DNN worker scope {record.get('dnn_scope') or '-'}",
    ]
    results = record.get("results") or {}
    if results.get("returned") is not None:
        lines.append(f"  results     {results['returned']} returned, {results.get('missing') or 0} missing")
    if record["stages"]:
        lines += ["", "Stages"]
        lines.append(f"  {'stage':<7}{'name':<14}{'tasks':>6}{'done':>6}   task seconds (mean / p50 / p95 / max)      span")
        for s in record["stages"]:
            t = s["task_seconds"]
            lines.append(
                f"  {s['stage']:<7}{(s['name'] or '-')[:13]:<14}{s['tasks']:>6}{s['completed']:>6}   "
                f"{_fmt(t['mean']):>7} / {_fmt(t['p50']):>7} / {_fmt(t['p95']):>7} / {_fmt(t['max']):>7}   "
                f"{_fmt(s['span_seconds'])}"
            )
            lines.append("          workers: " + "  ".join(f"{w}={n}" for w, n in s["workers"].items()))
    if record["workers"]:
        lines += ["", "Workers"]
        lines.append(f"  {'worker':<22}{'kind':<8}{'platform':<12}{'tasks':>6}{'busy s':>9}   by stage")
        for w in record["workers"]:
            lines.append(
                f"  {w['worker_id'][:21]:<22}{w['kind']:<8}{(w['platform'] or '-')[:11]:<12}{w['tasks']:>6}"
                f"{w['busy_seconds']:>9.1f}   {w['stages']}"
            )
    links = (record.get("links") or {}).get("workers") or {}
    if links:
        lines += ["", "Link estimates when the run ended"]
        lines.append(f"  {'worker':<22}{'down Mbit/s':>12}{'up Mbit/s':>11}{'rtt ms':>8}{'samples':>9}{'compute s':>10}")
        for worker_id, link in sorted(links.items()):
            samples = f"{link['samples_down']}/{link['samples_up']}"
            lines.append(
                f"  {worker_id[:21]:<22}{link['down_mbps']:>12.2f}{link['up_mbps']:>11.2f}{link['rtt_ms']:>8.1f}"
                f"{samples:>9}{('-' if link['compute_s'] is None else format(link['compute_s'], '.2f')):>10}"
                + ("  (imputed)" if link["imputed"] else "")
            )
    failures = record["failures"]
    lines += ["", f"Failures: {failures['count']}"]
    for item in failures["recent"]:
        lines.append(f"  {item['at']}  worker={item['worker_id']}  task={item['task_id']}  {item['error']}")
    for item in failures["task_errors"]:
        lines.append(f"  task {item['task_id']}: {item['error']}")
    return "\n".join(lines) + "\n"


def _fmt(value: Optional[float], unit: str = " s") -> str:
    return "-" if value is None else f"{value:.2f}{unit}"


def write_run(record: Mapping[str, Any], directory: Optional[str | Path] = None) -> Path:
    """Append to runs.jsonl and write the readable report. Returns the report path."""
    out = log_dir(directory)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "runs.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
    stamp = (record.get("started_at") or record["logged_at"]).replace(":", "").replace("-", "").replace("T", "-")
    report = out / f"{stamp}_{str(record['run_id'])[:8]}.txt"
    report.write_text(render_text(record), encoding="utf-8")
    return report


def read_runs(directory: Optional[str | Path] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Records oldest first; ``limit`` keeps only the newest N."""
    path = log_dir(directory) / "runs.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a partially written line must not hide the other runs
    return records[-limit:] if limit else records


def find_run(prefix: str, directory: Optional[str | Path] = None) -> Optional[Dict[str, Any]]:
    matches = [r for r in read_runs(directory) if str(r.get("run_id", "")).startswith(prefix)]
    return matches[-1] if matches else None


def summary_line(record: Mapping[str, Any]) -> str:
    tasks = record["tasks"]
    workers = len([w for w in record["workers"] if w["worker_id"] != "unassigned"])
    return (f"[RUN LOG] job={record['run_id']} status={record['status']} "
            f"tasks={tasks['completed']}/{tasks['total']} workers={workers} "
            f"duration={_fmt(record['duration_seconds'])} failures={record['failures']['count']}")


_COMPLETED_JOBS: set = set()  # jobs whose "completed" record is written; a late client disconnect must not add a second one


async def log_job_run(
    job_manager,
    connection_manager,
    task_dispatcher,
    job_id: str,
    *,
    results_returned: Optional[int] = None,
    results_missing: Optional[int] = None,
    status: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Collect everything about ``job_id`` and write its run record.

    Never raises: a logging problem must not break job completion. Call it
    **before** ``finalize_job``, which clears the pipeline's stage names.
    """
    if status == "client_disconnected" and job_id in _COMPLETED_JOBS:
        return None
    try:
        from foreman.db.base import async_session
        from foreman.db.models import WorkerFailureModel, WorkerModel
        from sqlalchemy import select

        from .dnn_placement import dnn_worker_scope, worker_kind

        job = await job_manager.get_job_info(job_id)
        if job is None:
            return None
        tasks = await job_manager.get_job_tasks_info(job_id)
        meta = job_manager.dependency_manager.get_pipeline_metadata(job_id) or {}
        stage_names = {i: s.get("name") for i, s in enumerate(meta.get("stages", [])) if s.get("name")}
        worker_ids = sorted({t.worker_id for t in tasks if getattr(t, "worker_id", None)})
        async with async_session() as session:
            failures = (await session.execute(
                select(WorkerFailureModel).where(WorkerFailureModel.job_id == job_id)
                .order_by(WorkerFailureModel.failed_at)
            )).scalars().all()
            rows = {}
            if worker_ids:
                rows = {r.id: r for r in (await session.execute(
                    select(WorkerModel).where(WorkerModel.id.in_(worker_ids)))).scalars().all()}
        worker_info: Dict[str, Dict[str, Any]] = {}
        for worker_id in worker_ids:
            live = connection_manager.get_worker_info(worker_id)
            if live is not None:
                worker_info[worker_id] = {"kind": worker_kind(live.worker_type), "platform": live.platform}
            else:  # worker already disconnected: fall back to what registration stored
                device = (getattr(rows.get(worker_id), "device_type", None) or "").lower()
                kind = "pc" if device.startswith("pc") else "mobile" if "android" in device else "unknown"
                worker_info[worker_id] = {"kind": kind, "platform": getattr(rows.get(worker_id), "os_type", None)}
        record = build_run_record(
            job, tasks, failures=failures, stage_names=stage_names, worker_info=worker_info,
            dnn_scope=dnn_worker_scope(),
            scheduler=task_dispatcher.get_scheduler_name() if task_dispatcher is not None else None,
            links=task_dispatcher.scheduler.link_snapshot() if task_dispatcher is not None else None,
            results_returned=results_returned, results_missing=results_missing, status=status,
        )
        report = write_run(record)
        if status == "completed":
            _COMPLETED_JOBS.add(job_id)
        print(summary_line(record))
        print(f"[RUN LOG] report: {report}")
        return record
    except Exception as exc:  # noqa: BLE001 - logging must never break the run
        print(f"[RUN LOG] could not write run log for {job_id}: {exc}")
        return None
