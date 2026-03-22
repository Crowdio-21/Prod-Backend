# Foreman Core

Core orchestration runtime for job execution, worker coordination, scheduling, pipeline dependencies, and DNN topology-aware flows.

## What this folder owns

- WebSocket lifecycle for clients and workers
- Job and task lifecycle state transitions
- Task dispatch using pluggable schedulers
- Pipeline dependency blocking/unblocking
- Checkpoint-aware failure recovery hooks
- DNN topology, feature routing, and aggregation
- Large payload and model artifact storage utilities

## Main modules

- `ws_manager.py`: top-level orchestrator; wires managers/handlers and owns connection cleanup/recovery flow.
- `connection_manager.py`: tracks worker/client websockets, availability, and worker capabilities.
- `message_handlers.py`: routes and handles protocol messages from client SDK and workers.
- `job_manager.py`: creates jobs (single, pipeline, DNN), caches metadata, and reads job results.
- `task_dispatcher.py`: assigns pending tasks via scheduler and adapts code for worker capabilities.
- `completion_handler.py`: final result collection + client delivery.
- `dependency_manager.py`: dependency counters and unblocking for pipeline tasks.
- `topology_manager.py`: graph validation/registration and assignment updates for DNN inference.
- `dynamic_topology.py`: best-effort reassignment/replan after worker failures.
- `feature_router.py`: in-memory intermediate feature routing for topology execution.
- `aggregation_handler.py`: merge/aggregation strategies (`average`, `weighted_sum`, `voting`).
- `payload_store.py`: file-backed large payload references (`payload_ref://...`).
- `model_registry.py`: stores partition artifacts and exposes model artifact URLs.

## Subfolders

- `scheduling/`: scheduler interface + built-in scheduler implementations and MCDM integration.
- `staged_results_manager/`: checkpoint storage, reconstruction, and resume decision logic.
- `utils/`: foreman DB wrapper helpers and checkpoint blob decode utilities.

## Runtime flow (high level)

1. Client submits a job or pipeline over WebSocket.
2. `message_handlers.py` validates/parses and delegates to `job_manager.py`.
3. Tasks become `pending` (or `blocked` when dependencies exist).
4. `task_dispatcher.py` selects workers via `scheduling/` and sends task messages.
5. Worker results update task state; dependency manager unblocks downstream tasks.
6. On completion, `completion_handler.py` emits final job results.

## Notes

- Worker capability detection allows code instrumentation for runtimes that do not support `sys.settrace`.
- DNN jobs can include topology metadata, model artifacts, and aggregation strategy.
- Large args/results are externalized from DB rows through payload references.
