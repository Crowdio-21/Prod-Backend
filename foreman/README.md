# Foreman

Foreman is the orchestration service for CrowdIO distributed execution.

It runs:
- a FastAPI HTTP service for dashboard + REST APIs
- a raw WebSocket server for worker/client execution traffic
- an async SQLite-backed persistence layer
- pluggable scheduling for heterogeneous workers

## Services and Ports

- HTTP API: http://localhost:8000
- FastAPI docs: http://localhost:8000/docs
- Dashboard WebSocket (UI updates): ws://localhost:8000/ws
- Worker/Client execution WebSocket: ws://localhost:9000

## What Foreman Handles

- Job lifecycle: create, track, finalize, and return results
- Task lifecycle: pending, assigned, blocked, completed, failed
- Worker coordination: registration, availability, failure recovery
- Scheduling: FIFO, round robin, least loaded, performance, priority, and MCDM variants
- Model-affinity scheduling: tiered worker selection preferring workers with models already resident or cached
- Pipeline execution: dependency counters and downstream unblocking
- Streaming pipeline mode: per-input dependencies for pipeline parallelism across multiple inputs
- DNN topology execution: graph-aware routing and aggregation
- Model residency tracking: tracks which model partitions are loaded in memory or cached on disk per worker
- Smart model loading: skips download/reload when the target worker already has the model (from_cache)
- Auto stage-worker assignment: assigns pipeline stages to workers by model affinity instead of requiring explicit device mappings
- Checkpointing: base/delta storage and resume-aware recovery support

## Folder Map

- main.py: app startup and lifespan orchestration
- api/: FastAPI routes and dashboard assets
- core/: orchestration runtime (dispatch, handlers, topology, scheduling integration)
- db/: SQLAlchemy engine/models/CRUD/seeding
- schema/: Pydantic response models

Detailed docs:
- core docs: core/README.md
- scheduling docs: core/scheduling/README.md
- MCDM docs: core/scheduling/mcdm/README.md
- staged results docs: core/staged_results_manager/README.md
- core utils docs: core/utils/README.md
- DB docs: db/README.md
- schema docs: schema/README.md
- API docs: api/README.md

## Startup

From repo root:

```bash
uv run python foreman/main.py
```

Or with uvicorn:

```bash
uv run uvicorn foreman.main:app --host 0.0.0.0 --port 8000
```

Startup sequence:
1. Initialize database and run seed routine
2. Create WebSocketManager and wire API websocket module
3. Start raw execution WebSocket server on port 9000
4. Serve FastAPI app on port 8000

## API Overview

Primary route groups:
- Base/dashboard and operational APIs in api/routes.py
- Scheduler config APIs in api/scheduler_routes.py
- Checkpoint monitoring APIs in api/checkpoint_routes.py
- Evaluation metrics APIs in api/evaluation_routes.py
- Dashboard websocket endpoint and ws-manager stats in api/websockets.py

Examples:
- GET /api/stats
- GET /api/jobs
- GET /api/workers
- GET /api/scheduling-info
- GET /api/evaluation/metrics
- GET /api/checkpoints/job/{job_id}

## Execution Message Flow (High Level)

1. SDK client connects to ws://localhost:9000 and submits SUBMIT_JOB or SUBMIT_PIPELINE_JOB.
2. Foreman creates DB records and prepares dispatchable tasks.
3. Workers connect to ws://localhost:9000 and announce WORKER_READY (including cached_model_partitions list).
4. Model load tracker registers each worker's cached partitions for affinity decisions.
5. Task dispatcher selects workers using the active scheduler (with model-affinity wrapper for DNN jobs) and sends ASSIGN_TASK or RESUME_TASK.
6. For DNN pipelines, LOAD_MODEL is sent with from_cache=true when the worker already has the model on disk.
7. Workers return TASK_RESULT / TASK_ERROR / TASK_CHECKPOINT / TASK_PROGRESS.
8. Foreman updates state, handles dependency unblocking (barrier or streaming mode), and emits final JOB_RESULTS.

## Scheduling Notes

Default scheduler is created in core/ws_manager.py through core/scheduling/factory.py.

Supported types include:
- fifo
- round_robin
- performance
- least_loaded
- priority
- aras
- edas
- mabac
- wrr

For MCDM algorithms, config may be loaded from scheduler_configs in the database.

## Persistence Notes

- SQLite URL is configured in db/base.py.
- Initialization includes integrity checks and additive SQLite migrations.
- Large payloads can be externalized via payload reference storage.
- Checkpoint data supports hybrid DB/filesystem storage paths.

## Operational Notes

- Scheduler activation endpoints may require runtime reload/restart behavior depending on deployment mode.
- Dashboard APIs are intentionally denormalized for visualization payloads.
- Worker capability detection is used by core dispatcher to support mobile runtimes.
