# CROWDio Backend (Prod-Backend)

CROWDio is a distributed compute backend for running Python workloads across heterogeneous worker nodes (desktop PCs and mobile devices). This repository contains:

- **Foreman**: orchestrator, scheduler, REST API, WebSocket server and web dashboard
- **PC Worker**: task executor that connects to the Foreman over WebSocket
- **Checkpointing and recovery** for long-running tasks (base + delta checkpoints)
- **Pipeline jobs** with stage dependencies and upstream-result injection
- **DNN partitioning support**: DNN tasks run on mobile workers (PC workers only when validating partitions), plus ONNX model serving
- **Reconfiguration-aware planning** (behind feature flags, off by default): the Foreman chooses where each shard runs and where the model is cut from live telemetry, prices the cost of switching under single residency, and switches only when the gain beats that cost
- **Evaluation**: legacy metrics modules, plus a trace-driven simulator and experiment harness for the planner

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
- [13. Reconfiguration-Aware Planning](#13-reconfiguration-aware-planning)
- [14. REST API Reference](#14-rest-api-reference)
- [15. Evaluation and Experiments](#15-evaluation-and-experiments)
- [16. Tests and Utility Scripts](#16-tests-and-utility-scripts)
- [17. Configuration and Runtime Files](#17-configuration-and-runtime-files)
- [18. Troubleshooting](#18-troubleshooting)
- [19. Known Gaps](#19-known-gaps)

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
- Creates the `WebSocketManager`. When `adaptive_planning` is on, it also builds and starts the plan coordinator ([section 13](#13-reconfiguration-aware-planning)) and prints the feature flags (`[FLAGS] ...`)
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
- Handle protocol-v1 reconfiguration messages (prefetch/assign/unload/verify) and keep a local shard cache. It does **not** speak protocol v2 yet, and it creates a new ONNX session per task
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
    shard_routes.py         /api/shards/* (range + sha256), /api/probe, /api/plan/status
    dashboard.html          Web dashboard (temp_dashboard.html is the fallback)
  core/
    ws_manager.py           Central runtime coordinator
    connection_manager.py   Worker/client connections, worker capabilities (PC/mobile)
    message_handlers.py     Client and worker message handling
    job_manager.py          Job/task lifecycle
    task_dispatcher.py      Scheduler-driven assignment; DNN tasks filtered by worker scope
    dnn_placement.py        DNN worker scope (mobile by default) and DNN task detection
    dependency_manager.py   Pipeline dependency counters (blocked -> pending)
    completion_handler.py   Final result assembly
    worker_reconnection_handler.py  Requeue or checkpoint-resume on disconnect
    payload_store.py        File-backed storage for large args/results
    dnn_observability.py    DNN routing feed for the dashboard
    plan_coordinator.py     Live planned-inference engine: DAG from Plan, epochs, switches, routes, prefetch
    plan_runtime.py         Builds the coordinator from flags/env (None when adaptive_planning is off)
    worker_protocol.py      Protocol-v2 transport: acks, timeouts, retries
    foreman_policy.py       Earlier break-even policy and executor (dormant; kept for compatibility)
    reconfiguration.py      Earlier coordinator for foreman_policy (dormant)
    reconfiguration_protocol.py  Protocol-v1 reconfiguration envelope
    scheduling/             FIFO, round-robin, least-loaded, performance, priority, MCDM
    staged_results_manager/ Checkpoint storage and recovery
  planning/                 Pure planning core (no I/O, clocks or globals; reused by the simulator)
    contract.py             Shared SDK/Foreman/worker contract (ShardSpec, WorkerState, Plan, messages)
    flags.py, config.py     Feature flags and YAML run config
    telemetry.py            EWMA estimators with age and sample count; snapshots
    profiles.py             Per-device-class phase models (load/init/write/unload/peak memory)
    registry.py             Manifest loading/validation, sha256, holder map, LRU eviction
    cost_model.py           C_switch, C_fetch, t_comm, t_comp, pipeline time, memory feasibility
    planner.py              Exhaustive oracle, Pareto DP, scalarised DP
    policies.py             B0, B1, B2, B3, B4, B5a, B5b, S-aware
    controller.py           Triggers, hysteresis, break-even switch rule
    routes.py, prefetch.py, problems.py, eventlog.py, serialize.py
  db/                       SQLAlchemy models, CRUD, seeding
  schema/                   Pydantic schemas

pc_worker/                  Worker service
  main.py                   CLI entry point
  config.py                 WorkerConfig
  core/                     worker.py, app.py, task_wrapper.py, reconfiguration_protocol.py
  api/                      Status routes and dashboard
  staged_results_manager/   Checkpoint handler and delta computation

evaluation/                 Metrics collectors, trackers, experiments, charts
  reconfig/                 Planner simulator, scenarios, metrics, replay, CLI (run.py), example configs
docs/reconfiguration/       Step 0 inspection report and per-component build report
tests/                      Launchers, example clients, workloads, unit tests
  planning/                 Planner/controller/cost-model tests, fake-worker integration, live WebSocket smoke test
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

Reconfiguration messages use a separate envelope (`"protocol": "crowdio.reconfiguration"`), so older SDK `MessageType` enums stay compatible:
- **Version 1**: the earlier `prefetch`/`assign`/`cutover`/`drain`/`unload`/`verify` set. `pc_worker` understands it.
- **Version 2**: the planned-inference protocol. Handled only when `adaptive_planning` is on, and ignored with a log line otherwise. See [section 13](#13-reconfiguration-aware-planning).

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
| `pyproject.toml` / `uv.lock` | Core Foreman/worker deps, `crowdio-sdk>=0.3.1`, `onnxruntime`, `pyyaml`; optional `dev` extra (`pytest`, `httpx`) | Running Foreman and workers; tests |
| `requirements.txt` | Core deps (including `pyyaml`) plus ML workloads (`tensorflow`, `torch`, `transformers`, `onnx`, `opencv`, `scikit-learn`, `pandas`, `kagglehub`), `crowdio-sdk>=2.0.1` | Running the sample ML/DNN workloads in `tests/`; used by the Dockerfiles |

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

Tests need `pytest` (and `httpx` for the shard-route tests). Both are in the `dev` extra of `pyproject.toml`:

```bash
pip install pytest httpx
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

> The Dockerfiles currently `COPY common/`, which is not in this repository. See [Known Gaps](#19-known-gaps) before building.

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

### Where DNN tasks run

DNN tasks run on **mobile workers**. PC workers are only for testing that the partitioning is correct. `CROWDIO_DNN_WORKERS` selects the scope, for both the legacy dispatcher and the planner:

| Value | DNN tasks go to |
|---|---|
| `mobile` (default) | Android workers only (`android_kotlin`, `android_chaquopy`) |
| `pc` | PC workers only, to validate partitions on a desktop |
| `any` | Any connected worker |

- A task counts as DNN work when its code uses `onnxruntime` or defines `partition_model`, `run_onnx_partition` or `run_tflite_partition`. Other tasks can run on any worker, as before.
- If no worker in the scope is connected, DNN tasks wait in the queue. They never fall back to another kind of worker.
- The Foreman prints `[DNN ROUTING]` lines with the scope, the available workers and the eligible ones.
- The dashboard's **DNN Placement Monitor** (`GET /api/dnn/observability`) shows the scope, DNN jobs seen, DNN tasks assigned, tasks waiting for an eligible worker, and each worker's kind and eligibility.

To validate partitioning on PC workers:

```bash
CROWDIO_DNN_WORKERS=pc python tests/run_foreman_simple.py
```

```bash
python tests/dnn_partitioning/sentiment_onnx_pipeline/multi_input_test.py --dataset data/sentences.csv --limit 20
```

`data/sentences.csv` is not tracked in git. Create it with `python tests/dnn_partitioning/sentiment_onnx_pipeline/prepare_sst2_dataset.py`. With the default scope, run the same client with only Android workers connected.

> `pipeline_dnn_partitioning.py` builds TensorFlow partitions inside a `partition_model` task. Mobile workers can't run TensorFlow, so use that script only with `CROWDIO_DNN_WORKERS=pc`. For mobile runs, export the partitions ahead of time (ONNX) and run them with `run_onnx_partition`.

Exported ONNX partitions are served to workers via `GET /api/models/{model_name}` from `tools/onnx_partitions_sentiment/` (override with `CROWDIO_MODEL_DIR`). See `tests/dnn_partitioning/sentiment_onnx_pipeline/README.md` to export them.

### Today's fixed sentiment pipeline (policy B0)

`tests/dnn_partitioning/sentiment_onnx_pipeline/multi_input_test.py` builds a fixed 3-stage DAG with `crowdio_pipeline`:
- `cell_a` = embeddings + encoder layers 0–2
- `cell_b` = layers 3–5
- `cell_c` = classification head

That is one encoder cut plus a fixed head split. The cut is baked into the ONNX files by `export_partitions.py --split-idx 3`. Activations travel worker → Foreman → worker, zlib-compressed and base64-encoded inside JSON. This path is unchanged and is the B0 baseline.

### Earlier shard policy (dormant)

`foreman/core/foreman_policy.py` (`BreakEvenPolicy`, `ReconfigurationExecutor`) and `foreman/core/reconfiguration.py` (`ForemanCoordinator`) predate the planner below:
- The coordinator ticks, but nothing calls `configure(...)`, so it never acts.
- It has no communication cost and no pipeline-time model.
- It is kept so existing tests and imports keep working. Its executor's wrong-shard unload bug is fixed.

## 13. Reconfiguration-Aware Planning

The Foreman can choose **where each shard runs** (placement) and **where the model is cut** (partition) at runtime, from live compute and network telemetry. It prices the cost of switching under **single residency**: a worker holds at most one live inference session, so a switch means unload, fetch if absent, load, then session init. It only switches when the expected gain beats that cost. Design notes and every assumption are in [docs/reconfiguration/build_report.md](docs/reconfiguration/build_report.md); the code inspection is in [docs/reconfiguration/step0_report.md](docs/reconfiguration/step0_report.md).

### Feature flags

All flags default **off**. With all flags off, no coordinator is built and the Foreman behaves exactly as before.

| Flag | Enables |
|---|---|
| `adaptive_planning` | The plan coordinator: planned inference (`submit_inference`), the planner and the controller. The other flags need it. |
| `direct_path` | Worker-to-worker activations when a peer link is reachable, fresh and cheaper by a margin; otherwise the relay through the Foreman |
| `peer_fetch` | Shard files fetched from eligible donor workers' disks instead of the Foreman |
| `prefetch` | Standby workers download likely-next shards to disk (never loading them) |
| `problem_reports` | Non-urgent worker problem reports feed the controller. Urgent ones (`NEXT_HOP_DOWN`, `OOM`, `LOAD_FAILED`) are always handled. |

Enable flags with an environment variable (a comma list or JSON), or in a YAML run config:

```bash
CROWDIO_FLAGS=adaptive_planning,problem_reports python tests/run_foreman_simple.py
```

| Variable | Purpose |
|---|---|
| `CROWDIO_FLAGS` | Feature flags; overrides the run config |
| `CROWDIO_RUN_CONFIG` | YAML run config: policy, objective, thresholds, margins, cooldown, scenario, seed. Example: [evaluation/reconfig/configs/run_example.yaml](evaluation/reconfig/configs/run_example.yaml) |
| `CROWDIO_MANIFEST` | Shard manifest: a `manifest.json`, or a legacy export directory (default `tools/onnx_partitions_sentiment`) |
| `CROWDIO_PUBLIC_URL` | Base URL workers use for `/api/probe` (default `http://localhost:8000`) |
| `CROWDIO_DNN_WORKERS` | Which workers may run shards: `mobile` (default), `pc` (validate partitions on PCs) or `any`. The planner never places a shard outside this scope. Also settable as `dnn_workers` in the run config. |

The run config's `log_path` sets the JSONL event log.

### How it works

- **Telemetry** (`foreman/planning/telemetry.py`):
  - Per-worker, per-direction EWMA mean and variance for bandwidth and RTT. Every estimate carries its age and sample count.
  - Bandwidth is passive first, from timed transfers. The active probe runs only when an estimate is old or the worker is idle, and is rate-limited.
  - Both of today's heartbeat shapes are ingested, plus v2 `report` messages.
- **Cost model** (`cost_model.py`, pure):
  - `C_switch` is 0 when resident; unload + load + init when on disk; plus fetch + write when absent.
  - `C_fetch` uses the best of the Foreman and eligible donors.
  - `t_comm` is min(direct, relay).
  - Pipeline time is `finish_i = max(ready_i, finish_{i-1} + t_comm) + t_comp`, `P = max(t_comp, t_comm)`, and `T_total(N) = finish_K + (N-1)·P`.
  - Every value carries uncertainty bounds.
- **Planner** (`planner.py`, `policies.py`, pure):
  - Memory feasibility filters first. Search covers ordered assignments of distinct workers and compositions of atoms into exported shards.
  - The exhaustive search is the exact oracle. A Pareto DP is also exact; a scalarised DP is faster and reports its gap.
  - The baselines B0–B5b and S-aware sit behind one interface.
- **Controller** (`controller.py`, pure):
  - Triggers: telemetry drift, worker join or leave, deadline risk, queue growth, problem reports.
  - Switch rule: J(new) + margin < J(current), with persistence, cooldown and a per-window switch cap.
  - It refuses non-forced switches on stale estimates.
  - Forced replans (worker lost, load failed, OOM) bypass the margin.
- **Epochs**: every request, activation, ack and result carries the plan epoch.
  - Voluntary switches **drain**: in-flight requests finish under the old epoch first.
  - Forced switches **resume** from a retained stage-boundary activation when the new plan keeps that cut, and otherwise restart from the input.
  - A worker never receives `load` while it has in-flight work of the old epoch.

### Policies

| Name | Behaviour |
|---|---|
| `B0` | Fixed cut, 4-tier affinity (resident > cached on disk > idle > evict) |
| `B1` | B0 plus a network-aware score inside each tier |
| `B2` | Placement and cuts by predicted latency, ignoring switching cost, re-planned every epoch |
| `B3` | Load time in the objective but no current-residency state; plans once |
| `B4` | Data-parallel replicas of the full model where it fits; pipeline over the rest |
| `B5a` / `B5b` | Per-plan switching overhead; B5b also allows asynchronous switching (the reference showing what single residency costs) |
| `S-aware` / `S-aware-placement` | Ours: full switching cost from the current residency state, with or without re-cutting |

### Protocol v2 (`crowdio.reconfiguration`, version 2)

Schemas are in [foreman/planning/contract.py](foreman/planning/contract.py). Every command carries a `msg_id` and is retried until acknowledged.

| Direction | Messages |
|---|---|
| Foreman → worker | `load`, `prefetch`, `cancel_prefetch`, `run`, `probe`, `peer_fetch`, `serve_allowed`, `route` |
| Worker → Foreman | `report` (WorkerState), `load_result`, `problem`, `need`, `activation` (relay route), `ack`, `msg_ack`, `probe_result` |
| Worker → worker | `activation`, `ack` |
| Client ↔ Foreman | `submit_inference` → `inference_accepted`, `inference_result` |

Problem types: `NEXT_HOP_DOWN`, `MISSING_SHARD`, `WRONG_SHARD`, `WRONG_EPOCH`, `OOM`, `SLOW_STAGE`, `LINK_DEGRADED`, `LOAD_FAILED`.

### Shards and manifests

- `ShardRegistry` (`foreman/planning/registry.py`) adapts today's 3-cell `partition_metadata.json` (atoms `embed, L0..L5, head`), or loads a contract `manifest.json`.
- To let the planner re-cut, export every contiguous range on a cut grid plus the full model. This writes `manifest.json` and checks the default chain and the full model against PyTorch:

```bash
python tests/dnn_partitioning/sentiment_onnx_pipeline/export_manifest.py --output-dir tools/onnx_shards_sentiment --cut-grid 3 --split-idx 3
```

- Workers download shards with checksums and resumable ranges from `GET /api/shards/{shard_id}`.

## 14. REST API Reference

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
| GET | `/api/shards/manifest` | Shard manifest (shared contract) |
| GET | `/api/shards/{shard_id}` | Shard file; supports `Range` (206) and sends `X-Content-SHA256` |
| GET / POST | `/api/probe/{size}`, `/api/probe` | Bandwidth probe download / upload |
| GET | `/api/plan/status` | Plan coordinator status (`{"enabled": false}` when the flag is off) |

Scheduler (`/api/scheduler`): `GET /algorithms`, `GET /config`, `GET /config/{algorithm_name}`, `POST /config`, `POST /activate/{algorithm_name}`, `GET /stats`.

Evaluation (`/api/evaluation`): `GET /metrics`, `GET /job/{job_id}`, `GET /load-distribution`, `GET /throughput-history`, `GET /worker-performance`.

PC worker (`http://<worker>:<api-port>`): `GET /` (status), `GET /stats`, `POST /restart`, `GET /dashboard`, WebSocket `/ws` (live status stream).

## 15. Evaluation and Experiments

Modules under `evaluation/`:

- `real_runner.py`: runs real jobs through a running Foreman and collects metrics
- `real_metrics_collector.py`: extracts metrics from the database
- `generate_charts.py`: charts from `evaluation_results/real_metrics.json`
- `runner.py`: experiment orchestration and reports
- Trackers: `energy_tracker.py`, `communication_tracker.py`, `load_balancing.py`, `failure_recovery.py`
- Experiments: `experiments/scalability.py`, `heterogeneity.py`, `energy_constraints.py`, `failure_simulation.py`

CLIs: `python -m evaluation.real_runner --host --port --output --quick`, `python -m evaluation.real_metrics_collector --output --since-hours --job-ids`, and `python -m evaluation.runner --quick --output --iterations`. Output goes to `evaluation_results/` by default.

### Reconfiguration experiment harness (`evaluation/reconfig/`)

A trace-driven simulator runs the **same** pure planner and controller as the Foreman. Ground truth (links, compute, phase times) changes with a scenario, and the controller only sees noisy telemetry.

```bash
python -m evaluation.reconfig.run compare --config evaluation/reconfig/configs/run_example.yaml --spec evaluation/reconfig/configs/spec_distilbert_5phones.yaml
```

| Command | Purpose |
|---|---|
| `sim` | One simulated run (`--with-oracle` adds regret) |
| `compare` | Every policy plus a clairvoyant reference: total time, regret, switches, wasted switches, stall, bytes moved, planner time, deadline hit rate |
| `notax` | Healthy run, B0 with flags off vs every feature on: steady-state latency, throughput, control-plane bytes per request and per minute |
| `metrics LOG` | Metrics from any JSONL log, simulated or live |
| `replay LOG` | Re-run every logged planner call and check that the decisions match |
| `dp-gap` | Gap of the scalarised and Pareto DPs against exhaustive search on random instances |
| `live` | Submit planned inference to a running Foreman, fire scenario commands, and record Foreman CPU (`--foreman-pid`) |

Scenario events (`network`, `load`, `drop_worker`, `join_worker`, `command`) are scheduled in the run config. In live mode the harness only runs the configured external command (for example `tc` on a Linux access point), so network shaping stays outside the app.

The simulation spec (`configs/spec_distilbert_5phones.yaml`) uses **placeholder** Android phase times. Replace them with the mobile benchmark's measurements before drawing conclusions.

## 16. Tests and Utility Scripts

Automated tests (no Foreman or workers needed):

```bash
python -m pytest tests/
```

This runs the original unit tests (`tests/test_*.py`) plus `tests/planning/`:
- hand-computed cost-model cases for all three residency states, with and without peer fetch;
- planner property tests (feasibility, single residency, Pareto DP equals exhaustive, S-aware equals B2 at zero switching cost, B1 ranks like B0 at zero bandwidth weight);
- controller hysteresis and break-even tests;
- fake-worker integration tests (bandwidth drop with a warm vs cold replacement, worker loss with in-flight resume, small-N rejection, a peer path blocked mid-run, duplicate delivery after fallback, load failure, prefetch, replay);
- a live WebSocket round trip through the real `WebSocketManager`.

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

## 17. Configuration and Runtime Files

| Item | Location / variable | Notes |
|---|---|---|
| Database | `./crowdio.db` (relative to the working directory) | Hardcoded in `foreman/db/base.py`; WAL files alongside |
| Large payloads | `./.payload_store/` | Referenced in SQLite as `payload_ref://...` |
| Checkpoints | `./.checkpoints/` | Written by the Foreman's storage handler |
| Scheduler logs | `./logs/` | FIFO and MCDM decision logs |
| ONNX models | `CROWDIO_MODEL_DIR` (default `tools/onnx_partitions_sentiment/`) | Served by `/api/models/{model_name}` |
| Worker settings | CLI flags (section 2) or Docker env vars (section 8) | |
| Feature flags, run config, manifest, DNN worker scope | `CROWDIO_FLAGS`, `CROWDIO_RUN_CONFIG`, `CROWDIO_MANIFEST`, `CROWDIO_PUBLIC_URL`, `CROWDIO_DNN_WORKERS` | See [section 13](#13-reconfiguration-aware-planning) |
| Planner event log | `log_path` in the run config (JSONL) | Kept in memory only when unset |
| Worker ping interval | `CROWDIO_PING_INTERVAL` (seconds, default 30) | PING replies refresh each worker's metrics in the `workers` table |

Run the Foreman from the project root so these paths resolve consistently.

## 18. Troubleshooting

Connection issues:
- Check that the Foreman is listening on `:9000` (WebSocket) and `:8000` (REST).
- Remote workers must use the Foreman's IP in `--foreman-url`, not `localhost`.

`[V2] ignoring ...: adaptive_planning is off`:
- A worker or client sent a protocol-v2 message, but the Foreman was started without `CROWDIO_FLAGS=adaptive_planning`. `GET /api/plan/status` shows whether the coordinator is enabled.

`ModuleNotFoundError: crowdio` or `common`:
- Install `crowdio-sdk` (section 6), or `pip install -e ../CROWDio-SDK` for local development.

No results returning:
- Check worker logs for deserialization or runtime errors (missing ML packages on the worker are a common cause).
- Check `/api/jobs` and `/api/workers`.
- DNN tasks wait until a worker in the DNN scope is connected: Android by default, or PC with `CROWDIO_DNN_WORKERS=pc`. A worker that registers without a `worker_type` counts as a PC worker.

Database locked:
- Close DB viewers or other processes holding `crowdio.db` and retry.

Schema drift:
- If models changed, stop services and delete `crowdio.db` (and the `-wal`/`-shm` files) to recreate it.

Unicode errors on Windows consoles:
- Set `PYTHONIOENCODING=utf-8`.

## 19. Known Gaps

- `Dockerfile.foreman` and `Dockerfile.worker` copy a `common/` directory that is not in this repo; `common` comes from `crowdio-sdk`. Remove that `COPY` line (the pip install already provides it) or vendor the package before building.
- `Dockerfile.foreman` sets `CROWDIO_DATA_DIR=/data`, but the code does not read it; the database is written to `/app/crowdio.db`, so the `crowdio_data` volume does not persist it.
- `Dockerfile.worker` health checks `GET /health`, which the worker API does not expose (use `GET /` instead).
- `requirements.txt` pins `crowdio-sdk>=2.0.1` while `pyproject.toml` uses `>=0.3.1`.
- Scheduler activation via REST is persisted but not applied to the running dispatcher (section 11).
- Reconfiguration-aware planning (section 13) needs workers that speak protocol v2:
  - Android still uses uppercase top-level `PREFETCH`/`ASSIGN`/`UNLOAD`/`EVICT` commands and has no peer listener for the direct path.
  - `pc_worker` speaks only v1 and creates a new ONNX session per task.
  - Until both are updated, the planned path has been exercised only with fake workers (`tests/planning/`).
- The Android phase times in `evaluation/reconfig/configs/spec_distilbert_5phones.yaml` are placeholders. The cold ~48 s / warm ~6 s reference numbers are unverified and used only as stale priors.
- `network_speed_mbps` from PC workers is NIC traffic over 100 ms (utilisation), not available bandwidth. It is stored as-is in the `workers` table.
- The Foreman Docker image does not include `tools/`, so set `CROWDIO_MANIFEST` to a mounted manifest before enabling `adaptive_planning` in Docker.
- Starlette 0.27's `TestClient` does not work with `httpx>=0.28`. The tests use `httpx.ASGITransport` directly.
