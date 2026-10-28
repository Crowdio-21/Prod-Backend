# Step 0 report: reconfiguration-aware placement and partitioning (Foreman)

Status: inspection only, no code changes. Inspected on 2026-10-27:
- `Prod-Backend` at `main` (`ea71f2f`)
- `CROWDio-SDK` at `main` (`99f3875`, with uncommitted changes)
- `Prod-Mobile` at `main` (`b921a0f`)

## 1. System facts checked against the code

| Claimed fact | What the code does | Verdict |
|---|---|---|
| Three layers: SDK, Foreman, Android workers (ONNX Runtime) | The three layers exist. A fourth worker type exists as well: `pc_worker` (Python, `PC_PYTHON`). DNN tasks are routed to both `PC_PYTHON` and `ANDROID_KOTLIN` workers (`foreman/core/task_dispatcher.py:74-89`). Android runs ONNX natively through `OnnxPartitionExecutor.kt`, intercepting `run_onnx_partition` code (`TaskProcessor.kt:752`). | Partly true |
| Persistent WebSockets between the Foreman and workers | Yes. `websockets.serve(... max_size=None)` on :9000 (`foreman/main.py`). | True |
| All activations go worker → Foreman → worker (star) | Yes. Each stage returns `task_result`. `DependencyManager._maybe_inject_upstream_result` writes the output into the downstream task's args in SQLite, spilling to `.payload_store/` when large (`foreman/core/dependency_manager.py:422`). The next stage gets it through `assign_task`. | True |
| zlib compresses tensors | Yes, but the **task code** does the compressing, not the transport: `serialize_tensor(..., compression="zlib")` in `multi_input_test.py`. The SDK default is also `"zlib"` (`common/serializer.py:145`). The bytes are then **base64-encoded inside JSON**, which adds about 33%. | True, with a caveat |
| Single residency: at most one live session per worker | **Not enforced anywhere.** PC: `run_onnx_partition` creates a new `ort.InferenceSession` on every task, so nothing stays resident and every request pays the load cost. Android: `OnnxPartitionExecutor.sessionCache` is an unbounded `ConcurrentHashMap<path, OrtSession>`, so several shards can be live at once. Android does reject `PREINIT` with `"single_residency"` (`TaskProcessor.kt:1780`). | **False today**; it's a target invariant |
| A switch is unload → (fetch) → load → init | No switch exists on `main`. Model files are fetched lazily inside the task (`urllib.request.urlretrieve` into `%TEMP%/crowdio-models` on PC, `ModelArtifactCache` on Android). | Not implemented |
| 4-tier affinity scheduler: resident > cached > idle > evict | **Not on `main`.** `main` has no affinity: DNN tasks go to `next(iter(eligible_set))` (single-task path) or to the active scheduler (FIFO by default). Affinity exists only on the unmerged branch `upstream/feature/stable-distributed` (`message_handlers.py:239-277` there), and it has **3 tiers**: resident → cached → "fewest resident models". There is no evict tier. | **False on `main`** |
| What "cached" means | On `stable-distributed`: the **file is on the worker's disk**. Workers report it in `worker_ready.cached_model_partitions`, and `ModelLoadTracker` keeps it after unload ("survives unload, fast reload"). It is not the OS page cache and not a live session. On `main`: the same idea, but the Foreman never tracks it. | File on disk |
| The SDK builds the DAG at submit time with a fixed cut | The **client script** builds it: `crowdio_pipeline(stages=[cell_a, cell_b, cell_c])` in `tests/dnn_partitioning/sentiment_onnx_pipeline/multi_input_test.py`. The Foreman expands it into N×3 tasks with 1:1 streaming dependencies (`DependencyManager.create_pipeline_tasks`). The cut is baked into the ONNX files by `export_partitions.py --split-idx 3`. | True |
| "3 stages" | `cell_a` = embeddings + encoder layers 0–2; `cell_b` = layers 3–5; `cell_c` = pre_classifier + classifier head. That is **one encoder cut (after layer index 2) plus a fixed head split**: 2 cut points and 3 parts. `cell_a` outputs `[B, S, 769]`: the hidden state plus the attention mask packed in as an extra channel. | Settled |
| Cold load ≈ 48 s, warm ≈ 6 s | No measurement or benchmark in any of the three repos produces these numbers. | Unverified |

## 2. Heartbeat schema and telemetry ingestion today

- **PC worker** sends two kinds of message:
  - The periodic heartbeat is sent as `PONG`, with metrics spread at the **top level** of `data` (`pc_worker/core/worker.py:1155`).
  - The reply to `PING` sends `data.performance_metrics` **nested** (`worker.py:236`).
- **Android** sends `worker_heartbeat` with `seq`, `t_mono_ns`, `residency{state, shard}`, `on_disk[]`, `net{}`, `res{}` and `queue{inflight, queued, stage_ms_ewma}` (`WorkerWebSocketClient.kt:428`). However:
  - `residency.shard` is filled with the **current task id**, not a shard id.
  - `on_disk` is always `[]`.
- **The Foreman** only reads `data["performance_metrics"]` (`message_handlers.py:262`). As a result:
  - the PC's periodic heartbeat metrics are ignored, and only PING replies count;
  - **every Android telemetry field is ignored**.
- **Bandwidth**: `network_speed_mbps` is NIC traffic over a 100 ms window (`common/device_info.py:300`). That is current **utilisation**, not available bandwidth, and it is often near 0. It is fed into `WorkerTelemetry.bandwidth_mean` as if it were capacity. `rtt_seconds` is never produced, so it always uses the default of 0.05.
- **Storage**: telemetry is written to `WorkerModel` columns (cpu, ram, battery, charging, network speed, storage). No age, sample count, EWMA or variance is kept.

## 3. Shards: storage, serving, loading

- **Exported files**: `tools/onnx_partitions_sentiment/{cell_a,cell_b,cell_c}.onnx` plus `partition_metadata.json`. The metadata has the model, opset, `split_idx` and tensor shapes. It has **no** sha256, byte counts, FLOPs, atom map or version.
- **Serving**: `GET /api/models/{name}` returns a plain `FileResponse` (`foreman/api/routes.py`). Starlette 0.27 doesn't support Range requests, so there are **no resumable transfers and no checksum**.
- **Loading**:
  - PC: the task downloads the file to a temp directory if it's missing, then builds a new session every call.
  - Android: `ModelArtifactCache` (files under `filesDir`, with an index file) plus the unbounded session map.
- **Holder map**: none on `main`.

## 4. Existing reconfiguration code (from earlier commits)

`foreman/core/foreman_policy.py` and `foreman/core/reconfiguration.py` already define `BreakEvenPolicy`, `MoveCostModel`, `ReconfigurationExecutor` and `ForemanCoordinator`. Their state:

- **Dormant.** `ForemanCoordinator.configure(...)` is never called, so `tick()` returns straight away.
- **Missing cost terms.** No communication cost, no pipeline `T_total(N)`, no `C_switch` separated from the steady-state period, no residency state per (worker, shard), no memory feasibility check, and no uncertainty on estimates.
- **Bug.** When the target worker is already serving another shard, the executor calls `unload(worker, shard)` with the **new** shard id instead of the old one (`foreman_policy.py:295-298`).
- **Protocol mismatch:**
  - The Foreman sends lowercase envelope types (`prefetch`, `assign`, `unload`, ... under `"protocol": "crowdio.reconfiguration"`).
  - Android listens for top-level **uppercase** `PREFETCH`, `PREINIT`, `ASSIGN`, `UNLOAD`, `EVICT`, `NET_EMULATION` and answers with `worker_command_result`, which the Foreman doesn't handle.
  - So today the two sides can't talk. Only `pc_worker` understands the envelope.
- **Not pure.** The policy methods are pure, but the module also imports `asyncio` and `time` for the executor.

## 5. Failure recovery and checkpointing

- **On worker disconnect**, `WorkerReconnectionHandler` either:
  - requeues the task from scratch (no checkpoint), or
  - parks it for up to 300 s and sends `resume_task` with the latest checkpoint.
- **On orphaned tasks**, `TaskDispatcher.recover_orphaned_tasks` re-dispatches them.
- **Dependency on the Foreman holding activations: yes.** A requeued stage task is rebuilt from its args in SQLite or `.payload_store`, and those args contain the upstream activation the Foreman injected. With a direct worker-to-worker path the Foreman would no longer have that activation. That's why the contract's worker-retained activations, acks and resends are needed.
- **ONNX stage checkpoints carry no useful state.** They hold only `model_path`, `execution_stage` and `progress_percent`, so in practice recovery means re-running the whole stage.

## 6. Language, framework, tests

- **Language and frameworks**: Python 3.12; FastAPI 0.104 / Starlette 0.27; `websockets` 12; SQLAlchemy 2 async with aiosqlite; Pydantic 2.5.
- **Unit tests**: pytest, which isn't declared in `pyproject.toml` or `requirements.txt`. The suite is 4 files, 11 tests, all passing:
  - `tests/test_foreman_policy.py`
  - `tests/test_reconfiguration_protocol.py`
  - `tests/test_task_dispatcher_routing.py`
  - `tests/test_code_instrumenter_wrappers.py`
- **Integration tests**: none. Everything else in `tests/` is a manual script that needs a live Foreman and workers.
- **Android**: Kotlin, ONNX Runtime Android, and Chaquopy for Python tasks.

## 7. Proposed change set (smallest that meets the brief)

New code lives in a new package, `foreman/planning/`. Live wiring goes in `foreman/core/`. Every new path is behind a flag, and all flags off means policy B0, which is today's behaviour.

| # | Component | Files | Notes |
|---|---|---|---|
| 1 | Schemas + flags | `foreman/planning/contract.py` (Pydantic v2, `schema_version`), `foreman/planning/flags.py`, `foreman/planning/config.py` (YAML run config) | ShardSpec, Manifest (cut_grid, atom map), WorkerState, Plan, messages, problem types, route modes |
| 2 | Telemetry | `foreman/planning/telemetry.py` (EWMA + variance, age, n), ingestion in `message_handlers.py` for **both** heartbeat shapes | Fix the nested/top-level mismatch; accept Android fields; passive bandwidth from timed transfers; rate-limited probe |
| 2 | Logging | `foreman/planning/eventlog.py` (JSONL) | Run id, seed, policy, flags, epoch on every event |
| 3 | Shard registry | `foreman/planning/registry.py`; range + sha256 serving at `GET /api/shards/{shard_id}` | Holder map from heartbeats; LRU that never evicts the resident file |
| 4 | Cost model (pure) | `foreman/planning/cost_model.py` | Formulas exactly as in the brief; device-class linear fits; uncertainty bounds |
| 5 | Planner (pure) | `foreman/planning/planner.py`, `foreman/planning/policies.py` (B0–B5b, S-aware) | Exhaustive oracle + Pareto DP (gap reported) |
| 6 | Controller | `foreman/planning/controller.py` (pure decision) + `foreman/core/plan_coordinator.py` (live) | Replaces the dormant `ForemanCoordinator` behind `adaptive_planning` |
| 7 | Route/source | `foreman/planning/routes.py` | Below the planner, as specified |
| 8 | DAG builder + epochs | `foreman/core/plan_dag.py`; hooks in `dependency_manager.py` and `task_dispatcher.py` | A fixed SDK DAG stays B0 |
| 9–11 | Problems, prefetch, protocol | `foreman/core/problems.py`, `foreman/core/prefetch.py`, `foreman/core/worker_protocol.py` | Versioned envelope v2 |
| 12 | Harness | `evaluation/reconfig/` (runner, scenarios, metrics, replay) | Network shaping via an external command only |

I'd retire `foreman_policy.ReconfigurationExecutor` and `BreakEvenPolicy` from the live path (they are dormant today), keep them importable so the existing tests pass, and fix the wrong-shard unload bug.

## 8. Questions to settle before building

1. **Base branch.** The residency/affinity tracker, the `LOAD_MODEL` flow and partition blob storage exist only on `upstream/feature/stable-distributed` (131 files differ from `main`, +13k/−8.5k lines). B0 is defined as "fixed cut, 4-tier affinity". Should I (a) build on `main` and implement B0 as today's behaviour on `main` (no affinity), or (b) merge `stable-distributed` first and use its 3-tier affinity as B0? Neither branch has a 4th "evict" tier.
2. **The existing policy code.** Should I replace `foreman_policy.py`/`reconfiguration.py` in the live path with the new planner and controller (my proposal), or extend them?
3. **Wire format.** The contract's message names (`load`, `prefetch`, `run`, `problem`, ...) match neither the Foreman's current envelope (`assign`, `cutover`, ...) nor Android's uppercase commands. I propose the contract names inside the `crowdio.reconfiguration` envelope, bumped to **version 2**, with Android moving to it. Does the mobile agent agree?
4. **Atoms and exports.** I propose 8 atoms for DistilBERT: `embed` (pinned to stage 0), `L0`…`L5`, and `head` (pinned last). The cut grid is every layer. Only 3 shards are exported today, so "restricted to exported shards" leaves the planner exactly one composition. Who owns exporting the extra shards (each contiguous layer range plus `full`)? The exporter is in this repo. Also, should the 769-channel mask packing stay in the tensor contract?
5. **Direct path.** Direct worker-to-worker activations need a listener on each worker. PC has FastAPI on :8001; Android has none. Is the mobile agent building one, and on which transport (WebSocket or HTTP)?
6. **Single residency on PC.** `pc_worker` currently has zero residency (a new session per task). Should PC workers keep one resident session to match the invariant, or are they out of scope for the experiments (Android only)?
7. **Cold/warm reference numbers.** Is there a source for 48 s and 6 s (device, shard, ORT version)? Until the mobile benchmark breaks them down I'll use them only as priors with wide uncertainty.
