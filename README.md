# CROWDio Backend (Prod-Backend)

CROWDio is a distributed compute backend for running Python workloads across heterogeneous worker nodes (desktop PCs and mobile devices). This repository contains:

- **Foreman**: orchestrator, scheduler, REST API, WebSocket server and web dashboard
- **PC Worker**: task executor that connects to the Foreman over WebSocket
- **Checkpointing and recovery** for long-running tasks (base + delta checkpoints)
- **Pipeline jobs** with stage dependencies and upstream-result injection
- **DNN partitioning support**: desktop-only routing, shard reassignment policy and a live reconfiguration protocol
- **Evaluation** modules for metrics collection, experiments and charts

The client SDK (`crowdio`) and the shared protocol/serialization package (`common`) are **not** in this repository. They ship in the `crowdio-sdk` package (source: the sibling `CROWDio-SDK` repository) and are installed as a dependency.

## Table of Contents
- [1. Architecture Overview](#1-architecture-overview)
- [2. Runtime Components](#2-runtime-components)
- [3. Repository Structure](#3-repository-structure)
- [4. Protocol and Data Flow](#4-protocol-and-data-flow)
- [5. Prerequisites](#5-prerequisites)
- [6. Installation](#6-installation)
- [7. Quick Start (End-to-End)](#7-quick-start-end-to-end)
- [8. Running with Docker](#8-running-with-docker)
- [9. Client SDK Usage](#9-client-sdk-usage)
- [10. Checkpointing and Recovery](#10-checkpointing-and-recovery)
- [11. Scheduling](#11-scheduling)
- [12. DNN Partitioning and Shard Reconfiguration](#12-dnn-partitioning-and-shard-reconfiguration)
- [13. REST API Reference](#13-rest-api-reference)
- [14. Evaluation and Experiments](#14-evaluation-and-experiments)
- [15. Tests and Utility Scripts](#15-tests-and-utility-scripts)
- [16. Configuration and Runtime Files](#16-configuration-and-runtime-files)
- [17. Troubleshooting](#17-troubleshooting)
- [18. Known Gaps](#18-known-gaps)

## 1. Architecture Overview

CROWDio uses a hub-and-spoke model. Clients and workers both connect to the Foreman's WebSocket server; the Foreman persists state in SQLite.

```text
+------------------------+        WebSocket (:9000)        +--------------------------+
|  SDK client            | <-----------------------------> |  Foreman                 |
|  (crowdio-sdk package) |                                 |  FastAPI REST (:8000)    |
+------------------------+                                 |  WebSocket server (:9000)|
                                                           |  Scheduler + dispatcher  |
                                                           +--------------------------+
                                                              |                  |
                                     assign / resume tasks    |                  |  jobs, tasks, workers,
                                     reconfiguration msgs     v                  v  checkpoints, scheduler cfg
                                          +--------------------------+   +---------------------------+
                                          |  Workers (1..N)          |   |  SQLite: ./crowdio.db     |
                                          |  pc_worker (PC_PYTHON)   |   |  foreman/db/models.py     |
                                          |  mobile workers          |   +---------------------------+
                                          +--------------------------+
```

Endpoints (defaults):

| What | URL |
|---|---|
| Foreman dashboard | `http://localhost:8000/` |
| Foreman OpenAPI docs | `http://localhost:8000/docs` |
| Foreman WebSocket (clients + workers) | `ws://localhost:9000` |
| PC worker status API | `http://localhost:8001/` (port configurable) |
| PC worker dashboard | `http://localhost:8001/dashboard` |

## 2. Runtime Components

### Foreman
Entry point: `foreman/main.py` (FastAPI app `foreman.main:app`)

On startup it:
- Initializes and seeds the database (`foreman/db/seed.py`), including default MCDM scheduler configs
- Creates the `WebSocketManager` and starts the shard reconfiguration coordinator
- Starts the WebSocket server on `0.0.0.0:9000` with `max_size=None` (no message size cap) and application-level ping/pong

Responsibilities:
- Accept job and pipeline submissions from SDK clients
- Track job/task lifecycle, including blocked pipeline tasks
- Schedule and dispatch tasks to available workers
- Handle worker disconnects and resume tasks from checkpoints
- Assemble results and return them to clients
- Store large task payloads on disk instead of in SQLite

### PC Worker
Entry point: `pc_worker/main.py` (`python -m pc_worker.main`)

Responsibilities:
- Connect to the Foreman WebSocket, register (`worker_ready`) and send heartbeats with bandwidth/CPU telemetry
- Execute serialized task functions, with checkpoint-aware wrapping for `@CROWDio.task(checkpoint=True)` tasks
- Report results, errors, progress and checkpoints (framework-aware deltas for PyTorch/TensorFlow/NumPy)
- Handle reconfiguration messages (prefetch/load/verify shards) and keep a local shard cache
- Serve a small status API and dashboard

CLI options:

| Flag | Default | Description |
|---|---|---|
| `--worker-id` | `worker-<random>` | Unique worker identifier |
| `--foreman-url` | `ws://localhost:9000` | Foreman WebSocket URL |
| `--api-host` | `0.0.0.0` | Worker API host |
| `--api-port`, `-p` | `8001` | Worker API port |
| `--max-concurrent-tasks` | `1` | Maximum concurrent tasks |
| `--heartbeat-interval` | `30` | Heartbeat interval (seconds) |
| `--no-auto-restart` | off | Disable automatic reconnect/restart |

### Client SDK (external)
Provided by `crowdio-sdk` (`crowdio` and `common` packages). Public functions:

- `crowdio_connect(host, port=9000)`
- `crowdio_map(func, iterable, **kwargs)`
- `crowdio_run(func, *args, **kwargs)`
- `crowdio_submit(func, iterable, **kwargs)` returns a `job_id`
- `crowdio_get(job_id, timeout=None)`
- `crowdio_pipeline(stages, ...)`
- `crowdio_disconnect()`
- `@CROWDio.task(...)` (also exported as `task` / `CROWDio_task`) for declarative checkpointing

Uppercase aliases (`CROWDio_connect`, `CROWDio_map`, ...) are also exported.

## 3. Repository Structure

```text
foreman/                    Orchestrator service
  main.py                   FastAPI app + WebSocket server startup
  api/
    routes.py               Core REST endpoints + dashboard
    scheduler_routes.py     /api/scheduler/*
    evaluation_routes.py    /api/evaluation/*
    checkpoint_routes.py    Checkpoint route factory
    websockets.py           /api/websocket-stats, /ws
    dashboard.html          Web dashboard (temp_dashboard.html is the fallback)
  core/
    ws_manager.py           Central runtime coordinator
    connection_manager.py   Worker/client connections, worker capabilities (PC/mobile)
    message_handlers.py     Client and worker message handling
    job_manager.py          Job/task lifecycle
    task_dispatcher.py      Scheduler-driven assignment, desktop-only DNN routing
    dependency_manager.py   Pipeline dependency counters (blocked -> pending)
    completion_handler.py   Final result assembly
    worker_reconnection_handler.py  Requeue or checkpoint-resume on disconnect
    payload_store.py        File-backed storage for large args/results
    dnn_observability.py    DNN routing feed for the dashboard
    foreman_policy.py       Shard assignment policy, cost model, reconfiguration executor
    reconfiguration.py      Live coordinator, WebSocket transport, metrics
    reconfiguration_protocol.py  Versioned reconfiguration envelope
    scheduling/             FIFO, round-robin, least-loaded, performance, priority, MCDM
    staged_results_manager/ Checkpoint storage and recovery
  db/                       SQLAlchemy models, CRUD, seeding
  schema/                   Pydantic schemas

pc_worker/                  Worker service
  main.py                   CLI entry point
  config.py                 WorkerConfig
  core/                     worker.py, app.py, task_wrapper.py, reconfiguration_protocol.py
  api/                      Status routes and dashboard
  staged_results_manager/   Checkpoint handler and delta computation

evaluation/                 Metrics collectors, trackers, experiments, charts
tests/                      Launchers, example clients, workloads, unit tests
Dockerfile.foreman, Dockerfile.worker, docker-compose.yml
pyproject.toml, uv.lock, requirements.txt
```

Module-level docs: [foreman/README.md](foreman/README.md), [pc_worker/README.md](pc_worker/README.md), [tests/README.md](tests/README.md).

## 4. Protocol and Data Flow

The message protocol is defined in `common/protocol.py` (from `crowdio-sdk`). Messages are JSON objects of the form `{"type", "data", "job_id"}`.

| Direction | Message types |
|---|---|
| Client -> Foreman | `submit_job`, `submit_pipeline_job`, `get_results`, `disconnect` |
| Foreman -> Worker | `assign_task`, `resume_task`, `ping` |
| Worker -> Foreman | `worker_ready`, `worker_heartbeat`, `task_result`, `task_error`, `task_checkpoint`, `task_progress`, `pong` |
| Foreman -> Client | `job_accepted`, `job_progress`, `job_results`, `job_error` |
| Foreman -> Worker (ack) | `checkpoint_ack` |

Shard reconfiguration messages use a separate envelope (`"protocol": "crowdio.reconfiguration"`, version 1) so older SDK `MessageType` enums stay compatible. See [section 12](#12-dnn-partitioning-and-shard-reconfiguration).

Standard flow:
1. The client calls `crowdio_map(...)`; the SDK serializes the function and arguments.
2. The Foreman creates the job and tasks and stores metadata in SQLite (large payloads go to `.payload_store/`).
3. The scheduler picks a worker and the Foreman sends `assign_task`.
4. The worker executes the function and returns `task_result` or `task_error` (plus optional progress/checkpoints).
5. For pipelines, completed tasks unblock downstream tasks and can inject upstream results.
6. The Foreman assembles ordered results and sends `job_results` to the client.

## 5. Prerequisites

- Python 3.10+ (`.python-version` pins 3.12.6; the Docker images use 3.12.6)
- Windows, Linux or macOS
- Optional: [uv](https://docs.astral.sh/uv/) (a `uv.lock` is committed)
- Optional: Docker / Docker Compose

## 6. Installation

There are two dependency sources, and they differ:

| File | Contents | Use for |
|---|---|---|
| `pyproject.toml` / `uv.lock` | Core Foreman/worker deps, `crowdio-sdk>=0.3.1`, `onnxruntime` | Running Foreman and workers |
| `requirements.txt` | Core deps plus ML workloads (`tensorflow`, `torch`, `transformers`, `onnx`, `opencv`, `scikit-learn`, `pandas`, `kagglehub`), `crowdio-sdk>=2.0.1` | Running the sample ML/DNN workloads in `tests/`; used by the Dockerfiles |

Pick one per environment.

With uv:

```bash
uv sync
```

With pip (Windows):

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

With pip (Linux/macOS): same, but activate with `source .venv/bin/activate`.

To develop against a local SDK checkout instead of the published package:

```bash
pip install -e ../CROWDio-SDK
```

Unit tests also need `pytest`, which is not listed in either dependency file:

```bash
pip install pytest
```

## 7. Quick Start (End-to-End)

Open three terminals in the project root (with the virtual environment active).

Terminal 1, start the Foreman:

```bash
python tests/run_foreman_simple.py
```

Equivalent: `python -m uvicorn foreman.main:app --host 0.0.0.0 --port 8000`.

Terminal 2, start a worker:

```bash
python -m pc_worker.main
```

Or use the test launcher (defaults to API port 8005):

```bash
python tests/run_worker_simple.py --port 8005
```

Terminal 3, run a sample client:

```bash
python tests/example_client.py localhost
```

To start several workers at once (default 8, ports from `--start-port`):

```bash
python tests/run_multiple_workers.py 8 --start-port 8001
```

On Windows, set `PYTHONIOENCODING=utf-8` if the console fails on emoji/arrow characters in log output.

## 8. Running with Docker

```bash
docker compose up --build -d
```

This builds two services:
- `foreman`: ports 8000 (REST) and 9000 (WebSocket), health check on `/api/stats`, volume `crowdio_data` mounted at `/data`
- `worker`: port 8001, connects to `ws://foreman:9000`; configured via `FOREMAN_URL`, `API_HOST`, `API_PORT`, `MAX_CONCURRENT_TASKS`, `HEARTBEAT_INTERVAL`

Scale workers with `docker compose up --scale worker=4 -d` (remove the fixed `container_name` and host port mapping on `worker` first, or replicas will conflict).

> The Dockerfiles currently `COPY common/`, which is not in this repository. See [Known Gaps](#18-known-gaps) before building.

## 9. Client SDK Usage

### Minimal map

```python
import asyncio
from crowdio import crowdio_connect, crowdio_map, crowdio_disconnect


def square(x):
    return x * x


async def main():
    await crowdio_connect("localhost", 9000)
    results = await crowdio_map(square, [1, 2, 3, 4, 5])
    print(results)
    await crowdio_disconnect()


if __name__ == "__main__":
    asyncio.run(main())
```

### Submit now, collect later

```python
import asyncio
from crowdio import crowdio_connect, crowdio_submit, crowdio_get, crowdio_disconnect


def heavy(x):
    return x ** 3


async def main():
    await crowdio_connect("localhost", 9000)
    job_id = await crowdio_submit(heavy, [1, 2, 3, 4, 5])
    results = await crowdio_get(job_id, timeout=60)
    print(job_id, results)
    await crowdio_disconnect()


if __name__ == "__main__":
    asyncio.run(main())
```

### Pipeline

```python
import asyncio
from crowdio import crowdio_connect, crowdio_pipeline, crowdio_disconnect


def stage_a(x):
    return x * 2


def stage_b(x):
    return x + 1


async def main():
    await crowdio_connect("localhost", 9000)
    stages = [
        {"func": stage_a, "args_list": [1, 2, 3]},
        {"func": stage_b, "args_list": [None, None, None], "pass_upstream_results": True},
    ]
    results = await crowdio_pipeline(stages)
    print(results)
    await crowdio_disconnect()


if __name__ == "__main__":
    asyncio.run(main())
```

Pipeline jobs can be inspected through `GET /api/pipeline/jobs` and `GET /api/pipeline/job/{job_id}`. For a full multi-stage example, see `tests/dnn_partitioning/pipeline_dnn_partitioning.py` and `tests/image_processing/pipeline_image_processing.py`.

## 10. Checkpointing and Recovery

Declare checkpointed tasks with `@CROWDio.task`:

```python
from crowdio import CROWDio


@CROWDio.task(
    checkpoint=True,
    checkpoint_interval=5.0,
    checkpoint_state=["i", "partial", "progress_percent"],
    retry_on_failure=True,
    max_retries=3,
)
def long_task(n):
    partial = 0
    progress_percent = 0.0
    for i in range(n):
        partial += i
        progress_percent = ((i + 1) / n) * 100
    return partial
```

Decorator parameters: `parallel=True`, `checkpoint=False`, `checkpoint_interval=10.0`, `checkpoint_state=None`, `retry_on_failure=True`, `max_retries=3`, `timeout=None`.

How it works:
- The worker wraps the task, captures the declared variables and sends base or delta checkpoints (`task_checkpoint`).
- The Foreman stores them under `.checkpoints/` and records metadata in SQLite.
- On worker disconnect, `worker_reconnection_handler.py` requeues the task from scratch (checkpointing off) or sends `resume_task` with the latest state (checkpointing on).

Run the demo:

```bash
python tests/example_checkpoint_client.py localhost
```

Inspect checkpoints with `python tests/view_checkpoints.py`, or via:
- `GET /api/checkpoints/job/{job_id}`
- `GET /api/checkpoints/recovery-events`

## 11. Scheduling

The live dispatcher is created by `WebSocketManager` with the **FIFO** scheduler by default (`foreman/core/ws_manager.py`).

Available schedulers (`foreman/core/scheduling/factory.py`):

| Type | Names |
|---|---|
| Simple | `fifo`, `round_robin`, `least_loaded`, `performance`, `priority` |
| MCDM (weighted criteria) | `aras`, `edas`, `mabac`, `wrr` |

MCDM configs (criteria weights, names, types) are seeded into the database at startup, with `aras` marked active. Manage them over REST:

```bash
curl http://localhost:8000/api/scheduler/algorithms
curl http://localhost:8000/api/scheduler/config
curl http://localhost:8000/api/scheduler/config/aras
curl -X POST http://localhost:8000/api/scheduler/activate/edas
curl http://localhost:8000/api/scheduler/stats
```

`POST /api/scheduler/config` creates or updates a config.

> Activating a config only updates the database. It does not swap the running dispatcher's scheduler, and the Foreman does not read the active config at startup (the endpoint's "restart foreman to apply changes" note does not hold yet). To change the live scheduler, construct `WebSocketManager(scheduler_type=...)` or call its scheduler-change method in code.

Scheduling decisions are logged to `logs/fifo_decisions.log` and `logs/mcdm_decisions.log`.

## 12. DNN Partitioning and Shard Reconfiguration

### Desktop-only DNN routing

The `partition_model` stage is restricted to `PC_PYTHON` workers. The Foreman prints `[DNN ROUTING]` lines showing available workers, desktop-eligible workers and each selected target. Worker heartbeats print `[NETWORK]` lines with measured bandwidth and CPU usage. The dashboard overview includes a **DNN Desktop Offload Monitor** backed by `GET /api/dnn/observability`.

Smoke test (Foreman and at least one PC worker running):

```bash
python tests/dnn_partitioning/pipeline_dnn_partitioning.py --model resnet --input-size 32 --num-partitions 2 --host localhost --port 9000
```

Confirm that the terminal shows `desktop-only=True` and a `PC_PYTHON` worker, and that the dashboard reports nonzero **DNN Jobs Seen** and **Desktop Tasks Assigned**. If no PC worker is connected, the job stays queued and the monitor shows **Waiting for Desktop** rather than sending the TensorFlow partition builder to a mobile worker.

Exported ONNX partitions are served to workers via `GET /api/models/{model_name}` from `tools/onnx_partitions_sentiment/` (override with `CROWDIO_MODEL_DIR`). See `tests/dnn_partitioning/sentiment_onnx_pipeline/README.md` to export them.

### Shard assignment policy

`foreman/core/foreman_policy.py` is the policy surface shared by the simulator and the live system:

- `SystemState` holds worker heartbeats and telemetry, the current assignment and epoch, stage profiles, workload parameters and the lower-bound move-cost model (`MoveCostModel`).
- `BreakEvenPolicy.decide(state)` returns a `Stay` or cost-aware `Reassign` decision. Latency is the default objective; select pipelined throughput with `Objective.PIPELINED_THROUGHPUT`.
- The policy enumerates injective assignments, applies conservative latency bounds, accounts for the critical path of concurrent downloads, and gates moves with cooldown and persistence.
- `ReconfigurationExecutor` runs prefetch, init, cutover, drain, unload and verification with epochs and load timeouts. It uses make-before-break for idle targets and break-before-make for targets already serving a shard. The transport owns atomic routing rollback.

### Live coordinator

`foreman/core/reconfiguration.py` provides `ForemanCoordinator`, a WebSocket transport and metrics. The Foreman starts the coordinator's periodic tick at startup and feeds heartbeat telemetry into it; prefetch/load/verify acknowledgements from workers are recorded by the WebSocket manager. To activate shard policy for a job, call `WebSocketManager.reconfiguration.configure(...)` with stage profiles and an initial assignment.

Workers handle reconfiguration messages before legacy task messages using a local copy of the envelope (`pc_worker/core/reconfiguration_protocol.py`), since the worker image does not include the `foreman` package.

## 13. REST API Reference

Base URL: `http://localhost:8000`. Full schemas: `http://localhost:8000/docs`.

Core:

| Method | Path | Description |
|---|---|---|
| GET | `/` | Dashboard |
| GET | `/api/stats` | Job/task/worker counts |
| GET | `/api/jobs` | List jobs |
| GET | `/api/jobs/{job_id}` | Job detail |
| GET | `/api/workers` | List workers |
| DELETE | `/api/workers/{worker_id}` | Remove a worker |
| GET | `/api/workers/{worker_id}/failures` | Worker failure history |
| DELETE | `/api/database/clear` | Clear database data |
| GET | `/api/scheduling-info` | Active scheduling info |
| GET | `/api/pipeline/jobs` | List pipeline jobs |
| GET | `/api/pipeline/job/{job_id}` | Pipeline job detail |
| GET | `/api/checkpoints/job/{job_id}` | Checkpoints for a job |
| GET | `/api/checkpoints/recovery-events` | Recovery events |
| GET | `/api/dnn/observability` | DNN routing summary and recent events |
| GET | `/api/models/{model_name}` | Download an exported `.onnx` model |
| GET | `/api/websocket-stats` | WebSocket connection stats |

Scheduler (`/api/scheduler`): `GET /algorithms`, `GET /config`, `GET /config/{algorithm_name}`, `POST /config`, `POST /activate/{algorithm_name}`, `GET /stats`.

Evaluation (`/api/evaluation`): `GET /metrics`, `GET /job/{job_id}`, `GET /load-distribution`, `GET /throughput-history`, `GET /worker-performance`.

PC worker (`http://<worker>:<api-port>`): `GET /` (status), `GET /stats`, `POST /restart`, `GET /dashboard`, WebSocket `/ws` (live status stream).

## 14. Evaluation and Experiments

Modules under `evaluation/`:

- `real_runner.py`: runs real jobs through a running Foreman and collects metrics
- `real_metrics_collector.py`: extracts metrics from the database
- `generate_charts.py`: charts from `evaluation_results/real_metrics.json`
- `runner.py`: experiment orchestration and reports
- Trackers: `energy_tracker.py`, `communication_tracker.py`, `load_balancing.py`, `failure_recovery.py`
- Experiments: `experiments/scalability.py`, `heterogeneity.py`, `energy_constraints.py`, `failure_simulation.py`

The scripts have CLIs (`real_runner`: `--host --port --output --quick`; `real_metrics_collector`: `--output --since-hours --job-ids`; `runner`: `--quick --output --iterations`) and write to `evaluation_results/` by default. They currently fail to import; see [Known Gaps](#18-known-gaps). The `/api/evaluation/*` endpoints work independently.

## 15. Tests and Utility Scripts

Unit tests (no Foreman/worker needed):

```bash
python -m pytest tests/test_foreman_policy.py tests/test_reconfiguration_protocol.py tests/test_task_dispatcher_routing.py tests/test_code_instrumenter_wrappers.py
```

Launchers and utilities in `tests/`:

| Script | Purpose |
|---|---|
| `run_foreman_simple.py` | Start the Foreman |
| `run_worker_simple.py` | Start one worker (`--port`, default 8005) |
| `run_multiple_workers.py` | Start N workers (`N --start-port 8001`) |
| `example_client.py` | Basic `crowdio_map` examples |
| `example_checkpoint_client.py` | Checkpointing demo |
| `view_database.py` | Inspect SQLite tables |
| `view_checkpoints.py` | Inspect checkpoint records |
| `quick_clear_db.py` | Clear DB data |
| `check_db_schema.py` | Verify schema columns |

Sample workloads (require a running Foreman and workers; most need the ML deps from `requirements.txt`):

- `tests/montecarlo/`: Monte Carlo and MCMC clients, with and without checkpointing
- `tests/sentiment_analysis/`: TextBlob, PyTorch and TensorFlow sentiment clients
- `tests/image_processing/`: image tiling, batch processing and mobile face search
- `tests/dnn_partitioning/`: VGG16/ResNet/TFLite partitioning and the sentiment ONNX pipeline (see its README files)

## 16. Configuration and Runtime Files

| Item | Location / variable | Notes |
|---|---|---|
| Database | `./crowdio.db` (relative to the working directory) | Hardcoded in `foreman/db/base.py`; WAL files alongside |
| Large payloads | `./.payload_store/` | Referenced in SQLite as `payload_ref://...` |
| Checkpoints | `./.checkpoints/` | Written by the Foreman's storage handler |
| Scheduler logs | `./logs/` | FIFO and MCDM decision logs |
| ONNX models | `CROWDIO_MODEL_DIR` (default `tools/onnx_partitions_sentiment/`) | Served by `/api/models/{model_name}` |
| Worker settings | CLI flags (section 2) or Docker env vars (section 8) | |

Run the Foreman from the project root so these paths resolve consistently.

## 17. Troubleshooting

Connection issues:
- Check that the Foreman is listening on `:9000` (WebSocket) and `:8000` (REST).
- Remote workers must use the Foreman's IP in `--foreman-url`, not `localhost`.

`ModuleNotFoundError: crowdio` or `common`:
- Install `crowdio-sdk` (section 6), or `pip install -e ../CROWDio-SDK` for local development.

No results returning:
- Check worker logs for deserialization or runtime errors (missing ML packages on the worker are a common cause).
- Check `/api/jobs` and `/api/workers`.
- DNN `partition_model` tasks wait until a `PC_PYTHON` worker is connected.

Database locked:
- Close DB viewers or other processes holding `crowdio.db` and retry.

Schema drift:
- If models changed, stop services and delete `crowdio.db` (and the `-wal`/`-shm` files) to recreate it.

Unicode errors on Windows consoles:
- Set `PYTHONIOENCODING=utf-8`.

## 18. Known Gaps

- `Dockerfile.foreman` and `Dockerfile.worker` copy a `common/` directory that is not in this repo; `common` comes from `crowdio-sdk`. Remove that `COPY` line (the pip install already provides it) or vendor the package before building.
- `Dockerfile.foreman` sets `CROWDIO_DATA_DIR=/data`, but the code does not read it; the database is written to `/app/crowdio.db`, so the `crowdio_data` volume does not persist it.
- `Dockerfile.worker` health checks `GET /health`, which the worker API does not expose (use `GET /` instead).
- `requirements.txt` pins `crowdio-sdk>=2.0.1` while `pyproject.toml` uses `>=0.3.1`.
- Scheduler activation via REST is persisted but not applied to the running dispatcher (section 11).
- `evaluation/__init__.py` imports `evaluation.metrics_collector`, which does not exist, so `python -m evaluation.<module>` fails with `ModuleNotFoundError`.
