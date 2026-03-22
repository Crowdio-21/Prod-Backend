# Foreman Schema

Pydantic response/request schemas used by Foreman REST APIs.

## File

- `schema.py`
  - Job/task/worker response models
  - worker failure models and aggregated stats models
  - system/job statistics models

## Key models

- `JobResponse`, `TaskResponse`, `WorkerResponse`
- `WorkerFailureResponse`, `WorkerFailureStats`, `WorkerFailureSummary`
- `JobStats`

## Notable behavior

- `WorkerResponse` includes computed `device_score` for dashboard/scheduling visibility.
- Models are configured with ORM compatibility (`from_attributes`).

## Usage pattern

CRUD/database results are converted to API payloads via `schema.py` models in routes under `foreman/api/`.
