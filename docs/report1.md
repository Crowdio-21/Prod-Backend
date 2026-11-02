# Report 1: How the existing CROWDio system works

**Scope**: this report covers the backend in this repository (`Prod-Backend`) as it runs today with default settings, together with the SDK (`CROWDio-SDK`) and the Android worker (`Prod-Mobile`) where they shape that behaviour. The optional reconfiguration-aware planner is summarised in section 10. Its design is in [reconfiguration/build_report.md](reconfiguration/build_report.md).

Inspected on 2026-10-28 (uncommitted working tree on `main`).

---

## 1. Purpose

CROWDio runs Python workloads across many devices. A developer writes ordinary Python functions and submits them through the SDK. A central server, the **Foreman**, splits the work into tasks and hands them to connected **workers**: Android phones, and PCs used for testing. The Foreman then gathers the results and returns them in order. For DNN inference, a model is cut into **partitions**, each partition runs on a different worker, and the outputs flow from one stage to the next.

## 2. The three parts

```text
 Developer machine                 Foreman (this repo)                        Workers
+--------------------+   WS :9000   +----------------------------------+  WS :9000  +----------------------+
| SDK (crowdio-sdk)  | -----------> | WebSocket server + REST API :8000 | ---------> | Android app (Kotlin, |
| crowdio_map /      | <----------- | Job manager, scheduler,           | <--------- |  ONNX Runtime,       |
| crowdio_pipeline   |   results    | dispatcher, dependency manager,   |  results   |  Chaquopy Python)    |
+--------------------+              | checkpoint + recovery             |            | PC worker (Python)   |
                                    | SQLite ./crowdio.db, .payload_store, |         +----------------------+
                                    | .checkpoints                      |
                                    +----------------------------------+
```

| Part | Where | Role |
|---|---|---|
| SDK | `crowdio-sdk` package (source `../CROWDio-SDK`; `crowdio` and `common`) | Turns the developer's functions into jobs; defines the wire protocol and serializer |
| Foreman | `foreman/` | Accepts jobs, stores them, schedules tasks, tracks workers, assembles results, recovers from failures |
| PC worker | `pc_worker/` | Runs tasks on a desktop. Used for testing. |
| Android worker | `../Prod-Mobile` | Runs tasks on phones. ONNX partitions run natively; other Python runs through Chaquopy. |

## 3. How a developer uses it

```python
from crowdio import crowdio_connect, crowdio_map, crowdio_disconnect

def square(x):
    return x * x

await crowdio_connect("localhost", 9000)
results = await crowdio_map(square, [1, 2, 3])   # -> [1, 4, 9]
await crowdio_disconnect()
```

The SDK offers:
- `crowdio_map`: one task per input, results back in order;
- `crowdio_submit` + `crowdio_get`: submit now, collect later;
- `crowdio_run`: a single call;
- `crowdio_pipeline`: multiple stages, where each stage can receive the previous stage's output.

The `@CROWDio.task(checkpoint=True, checkpoint_state=[...])` decorator marks variables that should be checkpointed during long tasks.

**How functions travel**: the SDK sends **source code**, not pickled objects.
- `common/serializer.serialize_function` runs `inspect.getsource`, strips decorators, and prepends a small runtime wrapper for pause/resume/kill and checkpoint hooks.
- Workers `exec` that source and call the user function (`pc_worker/core/worker.py: _deserialize_task_function`).
- So a task function must be self-contained: it does its own imports inside the function body.

## 4. Life of a normal job (`crowdio_map`)

1. **Connect**: the client opens a WebSocket to `:9000`. Its first message, `submit_job`, marks the connection as a client (`ws_manager._determine_client_type`).
2. **Accept**:
   - `ClientMessageHandler._handle_job_submission` stores the job and one task per input in SQLite (`job_manager.create_job`). Large arguments are written to `.payload_store/`, and only a `payload_ref://` string goes in the database.
   - It replies `job_accepted`.
3. **Schedule**:
   - `TaskDispatcher.assign_tasks_for_job` gets the available workers and asks the scheduler to pair tasks with workers in one batch (`scheduler.batch_select_workers`).
   - The live scheduler is **FIFO** (`WebSocketManager(scheduler_type="fifo")`).
4. **Dispatch**:
   - Each pair becomes an `assign_task` message containing the function source, the arguments and the checkpoint metadata.
   - For Chaquopy workers, which can't use `sys.settrace`, the dispatcher first instruments the code with explicit checkpoint calls (`_prepare_code_for_worker`).
5. **Execute**: the worker runs the function, can send `task_progress` and `task_checkpoint` along the way, and finishes with `task_result` or `task_error`.
6. **Collect**:
   - `WorkerMessageHandler._handle_task_result` marks the task complete and frees the worker.
   - It then tries to give that worker another pending task (`assign_task_to_available_worker`).
7. **Finish**: when every task of the job is complete, `JobCompletionHandler` reads the results in input order (`job_manager.get_job_results`) and sends `job_results` to the client.

## 5. Pipelines (multi-stage jobs)

`crowdio_pipeline(stages=[...])` sends `submit_pipeline_job`. `DependencyManager.create_pipeline_tasks` expands it into a task graph:

- Tasks are named `{job}_task_{stage}_{index}`.
- A stage marked `pass_upstream_results=True` (or `streaming`) with the same number of inputs as the previous stage is wired **1:1**: input *i* of stage *k* depends only on input *i* of stage *k-1*. Otherwise a stage waits for **all** of the previous stage (a barrier). A custom `dependency_map` can describe any other graph.
- Tasks whose dependencies aren't done have status `blocked`, and the scheduler doesn't see them.
- When a task completes, `on_task_completed` counts down its dependents. A dependent that reaches zero becomes `pending`, and the finished task's output is **injected into its arguments** as `{"original_args": ..., "upstream_results": {task_id: output}}`.
- The job's result is the output of the **last stage**, in input order.

**Data path**: every intermediate output goes worker → Foreman (stored in SQLite or `.payload_store`) → next worker. The Foreman sees and keeps every activation. This is a star topology.

## 6. DNN partitioning today: the sentiment pipeline

`tests/dnn_partitioning/sentiment_onnx_pipeline/` is the reference DNN workload.

**Model and cut**:
- DistilBERT SST-2 has 6 encoder layers. `export_partitions.py --split-idx 3` exports three ONNX files to `tools/onnx_partitions_sentiment/`:

| File | Contents | Size | Output |
|---|---|---|---|
| `cell_a.onnx` | embeddings + layers 0–2 | 180 MB | `[1, 64, 769]` (hidden state + attention mask packed as an extra channel) |
| `cell_b.onnx` | layers 3–5 | 85 MB | `[1, 64, 768]` |
| `cell_c.onnx` | classifier head | 2.4 MB | `[1, 2]` logits |

- "3 stages" therefore means one encoder cut plus a fixed head split. The cut is baked into the files.

**Running it**:
- `multi_input_test.py` tokenises the sentences and submits a 3-stage pipeline, where every stage runs the same task function `run_onnx_partition`.
- **Tensors**: they are serialized as NumPy `.npy` bytes, **zlib**-compressed and **base64**-encoded inside JSON (`common/serializer.serialize_tensor`).
- **Model files**:
  - Each task gets a `model_path` and a `model_url` (`GET /api/models/{name}` on the Foreman).
  - A PC worker downloads a missing file into `%TEMP%/crowdio-models` and then creates a **new ONNX session for every task**, so nothing stays loaded between tasks.
  - The Android app intercepts `run_onnx_partition` and runs it natively (`OnnxPartitionExecutor.kt`), caching files in the app's storage. It also caches ONNX sessions, without a limit.

**Verified on 2026-10-28**, on 5 PC workers (`CROWDIO_DNN_WORKERS=pc`) with 50 sentences:
- The 150 stage tasks were spread over all 5 workers.
- The outputs were identical to the same three partitions run locally, within 3.3e-6 of the unpartitioned model, with 100% label agreement.
- This was after fixing the result-ordering bug in section 11.

## 7. Where DNN tasks run

The default is **mobile only** (`foreman/core/dnn_placement.py`):

- **What counts as a DNN task**: code that uses `onnxruntime` or defines `partition_model`, `run_onnx_partition` or `run_tflite_partition`.
- **Which workers may run it**, set by `CROWDIO_DNN_WORKERS`:
  - `mobile` (default): Android workers (`android_kotlin`, `android_chaquopy`);
  - `pc`: PC workers only, to validate partitions;
  - `any`: both.
- If no eligible worker is connected, DNN tasks **wait**; they never fall back to another kind of worker. Non-DNN tasks can run on any worker.
- **Worker type**: each worker declares it in `worker_ready`. A worker that doesn't declare one is treated as a PC.
- **Monitoring**: the dashboard's **DNN Placement Monitor** and `GET /api/dnn/observability` show the scope, the counters, and each worker's kind and eligibility.

## 8. Workers

**Registration and liveness**:
- A worker connects to `:9000` and sends `worker_ready` with its ID, type, platform and device specs. `ConnectionManager` records a `WorkerInfo`, and the worker row in SQLite is created or marked online.
- Heartbeats differ by worker:
  - **PC**: a `PONG` every `--heartbeat-interval` seconds (default 30), with metrics at the top level of the message.
  - **Android**: a `worker_heartbeat` with network, resource, residency and queue fields.
- The Foreman pings every worker every 30 s (`CROWDIO_PING_INTERVAL`) and marks it online on any reply or heartbeat.
- The metrics in every shape (PING reply, periodic heartbeat, Android `res`/`net`) are written to the `workers` table (`foreman/core/worker_metrics.py`). The dashboard and the worker device score read them from there; no scheduler does.

**PC worker**: `pc_worker/core/worker.py`, started with `python -m pc_worker.main` or `tests/run_multiple_workers.py N`.
- Runs one task at a time by default (`--max-concurrent-tasks`).
- Can checkpoint using `sys.settrace`.
- Exposes a small status API and dashboard on its own port.

**Android worker**: `Prod-Mobile`.
- A Kotlin WebSocket client.
- ONNX partitions run natively with ONNX Runtime; other Python runs through Chaquopy.
- Checkpoints come from code instrumented by the Foreman.

## 9. Reliability: checkpoints and recovery

- **Checkpoints**:
  - A worker running a checkpointed task sends base or delta checkpoints (`task_checkpoint`). `pc_worker/staged_results_manager/delta_computer.py` makes deltas framework-aware for PyTorch, TensorFlow and NumPy.
  - The Foreman stores them under `.checkpoints/` with metadata in SQLite, and acks with `checkpoint_ack`.
- **Worker drops**: `WorkerReconnectionHandler` handles them.
  - **Checkpointing off**: the in-flight task goes back to `pending` for any worker.
  - **Checkpointing on**: the task is parked for up to 300 s. If the same worker reconnects, it gets `resume_task` with the latest state; otherwise the task is requeued.
- **Orphans**: `TaskDispatcher.recover_orphaned_tasks` resets tasks assigned to workers that are no longer connected. It runs whenever a worker becomes available.
- **Pipelines**: recovery works because the Foreman holds every intermediate activation, so a requeued stage is simply re-run from its stored arguments. For ONNX stages the checkpoint state is trivial (model path and progress), so recovery in practice means re-running the stage.

## 10. Optional: reconfiguration-aware planning (off by default)

Setting `CROWDIO_FLAGS=adaptive_planning` adds a second, planned inference path (`foreman/planning/`, `foreman/core/plan_coordinator.py`). Clients send `submit_inference` (protocol v2). The Foreman then:

- chooses which worker runs each shard and where the model is cut, from live telemetry;
- prices the cost of switching when each worker can hold only one shard in memory, and switches only when the gain beats that cost;
- moves activations between workers itself (relay), or worker-to-worker when `direct_path` is on.

The planner respects the same DNN worker scope as section 7. **Neither the Android app nor `pc_worker` speaks protocol v2 yet**, so this path has only been tested with fake workers and in the simulator (`evaluation/reconfig/`). With the flags off, none of it runs.

## 11. Storage, configuration and interfaces

**Runtime files**:

| What | Where |
|---|---|
| Jobs, tasks, workers, checkpoint metadata | `./crowdio.db` (SQLite, path fixed in `foreman/db/base.py`) |
| Large task arguments and results | `./.payload_store/` |
| Checkpoint blobs | `./.checkpoints/` |
| Scheduler decision logs | `./logs/` |
| Served ONNX models | `tools/onnx_partitions_sentiment/` (`CROWDIO_MODEL_DIR`) |

**Ports**:

| Port | Use |
|---|---|
| 8000 | REST API and dashboard (`/docs` for OpenAPI) |
| 9000 | WebSocket for clients and workers |
| 8001+ | Each PC worker's own status API |

**Main REST endpoints**:
- `/api/stats`, `/api/jobs`, `/api/workers`
- `/api/pipeline/jobs`, `/api/checkpoints/...`
- `/api/dnn/observability`, `/api/models/{name}`
- `/api/scheduling-info`, `/api/evaluation/...`
- With the planner: `/api/shards/...`, `/api/probe`, `/api/plan/status`

## 12. Known issues in the current system

1. **Pipeline results were returned out of order beyond 10 inputs.** Results were sorted by task ID as text, so `_10` came before `_2`. **Fixed on 2026-10-28** (`job_manager._task_input_index`, test `tests/test_pipeline_result_order.py`). Results from earlier pipeline runs with more than 10 inputs, including the 700-sentence sentiment runs, were mislabelled.
2. **Live worker metrics never reached the scheduler's data. Fixed on 2026-10-28.**
   - The Foreman now starts its ping loop at startup (every 30 s; `CROWDIO_PING_INTERVAL`).
   - All three heartbeat shapes are stored: nested PING reply, flat PC heartbeat or Android PONG, and Android `res`/`net`. The code is `foreman/core/worker_metrics.py`.
   - Missing values no longer overwrite known ones with NULL.
   - Verified with a real PC worker. Its `workers` row now gets live CPU, RAM, battery, charging, network and disk values.
   - These values feed the dashboard and the worker device score. None of the remaining schedulers uses them.

   Found while verifying: `python -m pc_worker.main` started only the worker's own API and never connected to the Foreman. **Fixed**: it now runs `run_with_worker()`. `tests/run_multiple_workers.py` was not affected.
3. **The scheduler is chosen by `CROWDIO_SCHEDULER` or in code, not over REST.** A `bandwidth_aware` scheduler was added on 2026-11-02 ([bandwidth_aware_scheduling.md](bandwidth_aware_scheduling.md)). The REST endpoints that pretended to choose one (`/api/scheduler/...`) and the whole MCDM scheduling family (ARAS, EDAS, MABAC, WRR) were removed on 2026-11-02. They only ever edited database rows and never changed the running scheduler.
4. **PC bandwidth is mislabelled.** `network_speed_mbps` is NIC traffic over 100 ms (utilisation), not available bandwidth, and RTT is never measured.
5. **Neither worker enforces "one model in memory at a time".** PC workers reload the ONNX session for every task; Android caches sessions without a limit.
6. **Docker images are broken as written.** They copy a `common/` folder that isn't in this repo, the database path ignores `CROWDIO_DATA_DIR`, and the worker health check calls a missing `/health` route.
7. **`partition_model` tasks need TensorFlow**, which phones don't have. Under the default mobile scope they wait or fail, so use them only with `CROWDIO_DNN_WORKERS=pc`.
8. **The "DNN Jobs Seen" counter is misleading.** It counts scheduling rounds, not jobs.

## 13. How to run it

See README section 7 (Quick Start) and section 12 (where DNN tasks run). Short version, in three terminals:

```powershell
$env:CROWDIO_DNN_WORKERS = "pc"; python tests\run_foreman_simple.py
```

```powershell
python tests\run_multiple_workers.py 5 --start-port 8101
```

```powershell
python tests\dnn_partitioning\sentiment_onnx_pipeline\multi_input_test.py --dataset data\sentences.csv --limit 50
```

Leave out the first environment variable to use the default mobile-only scope with Android workers.
