# Core Utils

Internal utility wrappers for DB operations and checkpoint blob handling.

## Files

- `utils.py`: async helper functions used by core managers/handlers.
- `__init__.py`: re-exports helpers for package-level imports.

## What helpers provide

- Job/task/worker DB operations (create, update, query)
- Task claim and completion convenience wrappers
- Worker failure recording and latest checkpoint failure lookup
- Checkpoint blob decode (`base64 -> gzip -> gzip -> json/pickle`)
- JSON-serialization helpers for non-serializable checkpoint state

## Scope

These functions are internal foreman plumbing. They are designed for core runtime modules, not external SDK consumers.

## Caution

- Keep signatures stable because multiple core modules depend on these wrappers.
- Prefer extending these helpers instead of duplicating direct DB logic in handlers/dispatchers.
