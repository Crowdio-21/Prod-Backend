# Staged Results Manager

Checkpoint storage and recovery for long-running or failure-prone tasks.

## Modules

- `checkpoint_manager.py`
  - Stores base/delta checkpoints
  - Tracks checkpoint metadata (progress, count, state vars)
  - Reconstructs task state from base + deltas
  - Cleans checkpoint data after completion
- `storage_handler.py`
  - Hybrid storage strategy:
    - store compressed checkpoints in DB for small payloads
    - store on filesystem for larger payloads
- `checkpoint_recovery_manager.py`
  - Decides whether to resume from checkpoint
  - Builds resume payloads/messages from reconstructed state

## Storage strategy

- Compression: gzip
- Size threshold: 1 MB compressed (`StorageHandler.DB_SIZE_LIMIT`)
- Under threshold: DB blob path
- Over threshold: filesystem path under checkpoint directory

## Recovery conditions (current behavior)

A task is considered resumable when:
- checkpoint data exists
- task is not in terminal state (`completed` / `failed`)
- checkpoint is recent enough (staleness check)

## Operational notes

- Compaction can merge many deltas into a new base checkpoint.
- State variable filtering supports declarative checkpointing from SDK metadata.
