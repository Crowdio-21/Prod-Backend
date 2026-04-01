# CROWDio Backend (Prod-Backend)

CROWDio is a distributed compute backend for running Python workloads across multiple worker nodes. It includes:
- A Foreman service (orchestrator + API + dashboard)
- PC Worker services (task executors)
- A Python SDK for clients to submit distributed jobs
- Checkpointing and recovery support for long-running tasks
- Evaluation and experiment modules for metrics and analysis

## Table of Contents
- [1. Architecture Overview](#1-architecture-overview)
- [2. Runtime Components](#2-runtime-components)
- [3. Repository Structure](#3-repository-structure)
- [4. Protocol and Data Flow](#4-protocol-and-data-flow)
- [5. Prerequisites](#5-prerequisites)
- [6. Installation and Environment Setup](#6-installation-and-environment-setup)
- [7. Quick Start (End-to-End)](#7-quick-start-end-to-end)
- [8. Client Setup and Usage](#8-client-setup-and-usage)
- [9. Checkpointing and Recovery](#9-checkpointing-and-recovery)
- [10. Scheduler Configuration](#10-scheduler-configuration)
- [11. REST API Reference (Foreman)](#11-rest-api-reference-foreman)
- [12. Evaluation and Experiments](#12-evaluation-and-experiments)
- [13. Test and Utility Scripts](#13-test-and-utility-scripts)
- [14. Troubleshooting](#14-troubleshooting)
- [15. Notes for Online Documentation](#15-notes-for-online-documentation)

## 1. Architecture Overview

CROWDio uses a hub-and-spoke model:

```text
+------------------------+          WebSocket (:9000)         +------------------------+
|   SDK Client App       |  <--------------------------------> |      Foreman           |
| (crowdio/api.py) |                                     | (FastAPI + Scheduler)  |
+------------------------+                                     +------------------------+
                                                                       |
                                                                       | Assign tasks / collect results
                                                                       v
                                                         +-------------------------------+
                                                         | Workers (1..N)                |
                                                         | pc_worker/core/worker.py      |
                                                         +-------------------------------+
                                                                       |
                                                                       | Status, metrics, persistence
                                                                       v
                                                         +-------------------------------+
                                                         | SQLite (crowdio.db)      |
                                                         | foreman/db/models.py          |
                                                         +-------------------------------+
```

Control plane:
- Foreman REST API: `http://localhost:8000`
- Foreman docs (OpenAPI): `http://localhost:8000/docs`
- Foreman dashboard: `http://localhost:8000`

Data plane:
- Foreman WebSocket server: `ws://localhost:9000`
- SDK clients and workers communicate with Foreman through this socket.

## 2. Runtime Components

### Foreman
Entry point: `foreman/main.py`

Responsibilities:
- Initialize DB and seed scheduler config at startup
- Accept client job submissions
- Track job/task lifecycle
- Schedule and dispatch tasks to available workers
- Aggregate task results and return job results to clients
- Track failures and checkpoint metadata

### Worker (PC)
Entry point: `pc_worker/main.py`
Alternative test launcher: `tests/run_worker_simple.py`

Responsibilities:
- Connect to Foreman WebSocket
- Register readiness and send heartbeats
- Execute serialized task functions
- Return results/errors/progress/checkpoints

### Developer SDK (Client)
Entry points:
- `crowdio/api.py` (public async functions)
- `crowdio/client.py` (`CrowdComputeClient`)

Key public functions:
- `connect(host, port=9000)`
- `map(func, iterable, **kwargs)`
- `run(func, *args, **kwargs)`
- `submit(func, iterable, **kwargs)`
- `get(job_id, timeout=None)`
- `pipeline(stages, dependency_map=None, **kwargs)`
- `disconnect()`

### Evaluation
Main modules under `evaluation/`:
- Real-run and metrics collectors
- Energy, communication, load-balancing, and failure trackers
- Experiment suites under `evaluation/experiments/`

## 3. Repository Structure

Top-level modules and intent:

- `foreman/`: Orchestration server, APIs, scheduler, DB layer
- `pc_worker/`: Worker runtime and worker-side API
- `crowdio/`: Client SDK and decorators
- `common/`: Shared protocol, serialization, instrumentation utilities
- `evaluation/`: Benchmarking and analysis framework
- `tests/`: Quick-start runners and sample workloads

Important Foreman internals:

- `foreman/core/ws_manager.py`: central runtime coordinator
- `foreman/core/job_manager.py`: job/task lifecycle management
- `foreman/core/task_dispatcher.py`: scheduler-driven assignment
- `foreman/core/message_handlers.py`: client/worker message handling
- `foreman/core/completion_handler.py`: final result assembly
- `foreman/core/staged_results_manager/`: checkpoint storage and recovery
- `foreman/core/scheduling/`: FIFO, round-robin, least-loaded, performance, priority, MCDM

## 4. Protocol and Data Flow

Protocol definition: `common/protocol.py`

Primary message categories:
- Client -> Foreman: `submit_job`, `submit_pipeline_job`, `get_results`, `disconnect`
- Foreman -> Worker: `assign_task`, `resume_task`, `ping`
- Worker -> Foreman: `worker_ready`, `worker_heartbeat`, `task_result`, `task_error`, `task_checkpoint`, `task_progress`, `pong`
- Foreman -> Client: `job_accepted`, `job_progress`, `job_results`, `job_error`

Standard flow:
1. Client calls `map(...)` (SDK serializes function + args).
2. Foreman creates job/tasks and stores metadata in DB.
3. Scheduler selects a worker and Foreman dispatches `assign_task`.
4. Worker executes function and returns result or error.
5. Foreman aggregates ordered task outputs.
6. Client receives `job_results` for the submitted job.

## 5. Prerequisites

- Python 3.10+
- Windows/Linux/macOS
- Recommended: virtual environment (`.venv`)

## 6. Installation and Environment Setup

From project root:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you prefer `pyproject.toml`-based install:

```bash
pip install -e .
```

Note: `requirements.txt` and `pyproject.toml` are not fully aligned. For now, use one dependency source consistently per environment.

## 7. Quick Start (End-to-End)

Open 3 terminals from project root.

Terminal 1 - start Foreman:

```bash
python tests/run_foreman_simple.py
```

Terminal 2 - start one worker:

```bash
python tests/run_worker_simple.py
```

Terminal 3 - run sample client:

```bash
python tests/example_client.py localhost
```

Scale workers:

```bash
python tests/run_multiple_workers.py 8 --start-port 8001
```

## 8. Client Setup and Usage

### Minimal SDK Example

```python
import asyncio
from crowdio import connect, map, disconnect


def square(x):
    return x * x


async def main():
    await connect("localhost", 9000)
    results = await map(square, [1, 2, 3, 4, 5])
    print(results)
    await disconnect()


if __name__ == "__main__":
    asyncio.run(main())
```

### Asynchronous Submit/Get Example

```python
import asyncio
from crowdio import connect, submit, get, disconnect


def heavy(x):
    return x ** 3


async def main():
    await connect("localhost", 9000)
    job_id = await submit(heavy, [1, 2, 3, 4, 5])
    results = await get(job_id, timeout=60)
    print(job_id, results)
    await disconnect()


if __name__ == "__main__":
    asyncio.run(main())
```

### Pipeline Example

```python
import asyncio
from crowdio import connect, pipeline, disconnect


def stage_a(x):
    return x * 2


def stage_b(x):
    return x + 1


async def main():
    await connect("localhost", 9000)

    stages = [
        {"func": stage_a, "args_list": [1, 2, 3]},
        {"func": stage_b, "args_list": [None, None, None], "pass_upstream_results": True},
    ]

    results = await pipeline(stages)
    print(results)
    await disconnect()


if __name__ == "__main__":
    asyncio.run(main())
```

## 9. Checkpointing and Recovery

Decorator API: `crowdio/decorators.py`

```python
from crowdio import crowdio


@crowdio.task(
    checkpoint=True,
    checkpoint_interval=1.0,
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

Run checkpoint demo:

```bash
python tests/example_checkpoint_client.py localhost
```

Checkpoint-related REST endpoints:
- `GET /api/checkpoints/job/{job_id}`
- `GET /api/checkpoints/recovery-events`

## 10. Scheduler Configuration

List available schedulers:

```bash
curl http://localhost:8000/api/scheduler/algorithms
```

Get active scheduler config:

```bash
curl http://localhost:8000/api/scheduler/config
```

Activate scheduler:

```bash
curl -X POST http://localhost:8000/api/scheduler/activate/aras
```

Built-in simple schedulers include:
- `fifo`
- `round_robin`
- `least_loaded`
- `performance`
- `priority`

MCDM configs are supported through scheduler config endpoints.
The `activate` API toggles persisted MCDM configurations by algorithm name (for example: `aras`, `edas`, `mabac`, `wrr`).

## 11. REST API Reference (Foreman)

Base URL: `http://localhost:8000`

Core endpoints:
- `GET /` dashboard HTML
- `GET /api/stats`
- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/workers`
- `DELETE /api/database/clear`
- `DELETE /api/workers/{worker_id}`
- `GET /api/workers/{worker_id}/failures`
- `GET /api/evaluation/metrics`
- `GET /api/scheduling-info`

Use generated docs for full schema details:

```text
http://localhost:8000/docs
```

## 12. Evaluation and Experiments

Evaluation API namespace: `/api/evaluation/*`

Examples:
- `GET /api/evaluation/metrics`
- `GET /api/evaluation/load-distribution`
- `GET /api/evaluation/throughput-history`
- `GET /api/evaluation/worker-performance`

Experiment modules:
- `evaluation/experiments/scalability.py`
- `evaluation/experiments/heterogeneity.py`
- `evaluation/experiments/energy_constraints.py`
- `evaluation/experiments/failure_simulation.py`

## 13. Test and Utility Scripts

From `tests/`:

- `run_foreman_simple.py`: start Foreman
- `run_worker_simple.py`: start one worker
- `run_multiple_workers.py`: start N workers
- `example_client.py`: basic SDK map examples
- `example_checkpoint_client.py`: checkpointing workflow demo
- `view_database.py`: inspect SQLite tables and rows
- `view_checkpoints.py`: inspect checkpoint records
- `quick_clear_db.py`: clear DB data
- `check_db_schema.py`: verify schema columns

## 14. Troubleshooting

Connection issues:
- Ensure Foreman is running on `:9000` (WebSocket) and `:8000` (REST).
- Ensure workers can reach Foreman host/IP.

No results returning:
- Check worker logs for task deserialization/runtime errors.
- Check Foreman `/api/jobs` and `/api/workers` endpoints.

DB/schema drift:
- If model schema changed, stop services and recreate `crowdio.db`.

Large payload disconnects:
- Current implementation uses `max_size=None` for WebSocket in Foreman and SDK to avoid default 1 MB caps.

## 15. Notes for Online Documentation

If publishing docs (MkDocs, Docusaurus, or GitHub Pages), split this README into pages:

1. `getting-started.md`: prerequisites, install, quick start
2. `architecture.md`: components, sequence flow, message types
3. `sdk-guide.md`: connect/map/run/submit/get/pipeline examples
4. `checkpointing.md`: decorator usage, recovery behavior, APIs
5. `operations.md`: scheduler config, monitoring, DB ops, troubleshooting
6. `api-reference.md`: REST endpoints and OpenAPI links

Recommended diagrams to include online:
- High-level component diagram (Foreman/Worker/Client/DB)
- Sequence diagram (submit -> schedule -> execute -> aggregate)
- Checkpoint recovery sequence (failure -> restore -> resume)

---

For module-level details, also see:
- `foreman/README.md`
- `pc_worker/README.md`
- `tests/README.md`
- `crowdio/README.md`
