# Foreman API

FastAPI routes for dashboard rendering, operational control, scheduler config, checkpoint observability, evaluation metrics, and websocket status.

## Files

- `routes.py`
  - main dashboard page (`/`)
  - core REST endpoints for jobs/workers/stats
  - checkpoint dashboard data, scheduling visualization, pipeline views
  - model artifact listing/download helpers
- `websockets.py`
  - websocket endpoint for dashboard real-time updates (`/ws`)
  - websocket manager stats endpoint (`/api/websocket-stats`)
- `checkpoint_routes.py`
  - modular checkpoint-focused endpoints (job/task checkpoint progress, cleanup, storage info)
  - exposes `create_checkpoint_routes(...)` for manager injection
- `scheduler_routes.py`
  - scheduler algorithm listing
  - CRUD/activation APIs for MCDM configs
- `evaluation_routes.py`
  - evaluation metrics for jobs/workers/performance/load balancing
  - chart-oriented endpoints for dashboard visualization
- `dashboard.html` / `temp_dashboard.html`
  - frontend dashboard assets
- `__init__.py`
  - module exports

## Main endpoint groups

- Dashboard and base data:
  - `GET /`
  - `GET /api/stats`
  - `GET /api/jobs`
  - `GET /api/jobs/{job_id}`
  - `GET /api/workers`
- Operational controls:
  - `DELETE /api/database/clear`
  - `DELETE /api/workers/{worker_id}`
- Scheduler management:
  - `/api/scheduler/*`
- Checkpoint monitoring:
  - `/api/checkpoints/*`
- Evaluation and charts:
  - `/api/evaluation/*`
- WebSocket dashboard updates:
  - `GET /api/websocket-stats`
  - `WS /ws`

## Notes

- Several endpoints are visualization-oriented and intentionally return denormalized structures for UI consumption.
- Scheduler activation endpoint currently notes restart/reload behavior to apply runtime changes.
