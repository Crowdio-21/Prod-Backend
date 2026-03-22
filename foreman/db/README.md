# Foreman DB

Async SQLAlchemy persistence layer for jobs, tasks, workers, scheduler configs, and failure history.

## Files

- `base.py`
  - Engine/session setup (`sqlite+aiosqlite`)
  - `init_db()` table creation
  - SQLite integrity check + auto-quarantine of corrupt DB files
  - additive SQLite column migrations for existing installs
- `models.py`
  - ORM models:
    - `JobModel`
    - `TaskModel`
    - `WorkerModel`
    - `WorkerFailureModel`
    - `SchedulerConfigModel`
    - `WorkerPerformanceHistoryModel`
- `crud.py`
  - async CRUD and state-transition helpers used by core runtime
  - job/task/worker updates, task claiming, completion guards, failure logging
  - stats queries for dashboard/evaluation APIs
- `seed.py`
  - startup database initialization and default MCDM scheduler config seeding

## Schema highlights

- Jobs support pipeline and DNN metadata (`is_pipeline`, `is_dnn_inference`, `inference_graph_id`, etc.).
- Tasks include checkpoint, dependency, and topology fields.
- Workers include device/runtime/performance attributes used by scheduling.
- Scheduler configs are stored as JSON text fields (weights/names/types).

## Initialization flow

1. `init_db()` creates tables and applies additive migrations.
2. `seed.initialize_database()` verifies schema and seeds initial scheduler configs.

## Notes

- SQLite is configured with lock timeout and WAL mode for better concurrent behavior.
- On detected corruption, DB file is moved to a timestamped backup and recreated.
