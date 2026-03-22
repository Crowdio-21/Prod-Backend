# Scheduling

Pluggable task scheduling subsystem used by `TaskDispatcher`.

## Entry points

- `scheduler_interface.py`
  - `TaskScheduler`: abstract scheduler contract.
  - `Task`: task metadata used by scheduling.
  - `Worker`: worker capability/performance snapshot used for ranking.
- `factory.py`
  - `create_scheduler(...)`: synchronous factory.
  - `create_scheduler_async(...)`: async factory (loads persisted MCDM config).

## Built-in scheduler types

Simple schedulers:
- `fifo`: first-in-first-out task assignment.
- `round_robin`: cyclical distribution across available workers.
- `performance`: favors workers with stronger historical success/perf.
- `least_loaded`: favors workers with lower observed load.
- `priority`: task-priority-first behavior.

MCDM schedulers (in `mcdm/`):
- `aras`
- `edas`
- `mabac`
- `wrr`

## Usage

```python
from foreman.core.scheduling import create_scheduler

scheduler = create_scheduler("fifo")
```

For MCDM with DB-backed configuration:

```python
from foreman.core.scheduling.factory import create_scheduler_async

scheduler = await create_scheduler_async("mabac", use_dynamic_weighting=True)
```

## Implementing a custom scheduler

1. Subclass `TaskScheduler` from `scheduler_interface.py`.
2. Implement:
   - `select_worker(task, available_workers, all_workers)`
   - `select_task(pending_tasks, worker_id)`
3. Optionally override `batch_select_workers(...)` for better throughput.
4. Register your scheduler in `factory.py`.

## Design notes

- `batch_select_workers` reduces repeated ranking overhead when multiple tasks are pending.
- `Worker` includes both basic and extended resource/performance fields for smarter ranking.
