# Multi-Worker DNN Pipeline — Implementation Plan

## Current State Summary

The system today supports **one mobile worker** executing a 3-partition DNN pipeline sequentially:
- Developer submits `dnn_pipeline(stages=[cell_a, cell_b, cell_c])` with input texts
- Foreman loads model on the single worker, runs stage 0 → unloads → loads stage 1 → runs → ... → returns final output
- **Barrier pattern**: ALL stage-N tasks must finish before ANY stage-N+1 task can start
- Model load/unload on every stage transition makes mobile execution slow
- No concept of model affinity — scheduler doesn't know which worker already has a model loaded

## Problems to Solve

| # | Problem | Root Cause |
|---|---------|-----------|
| P1 | **Developer can't map stages to devices** | Devices are ephemeral; developer doesn't know worker IDs at submission time |
| P2 | **Foreman can't fully auto-map either** | Scheduler has no model-affinity awareness; doesn't know which worker has which model cached |
| P3 | **No pipeline parallelism across inputs** | Barrier pattern blocks ALL of stage N+1 until ALL of stage N completes |
| P4 | **Mobile model load/unload is slow** | ONNX session created per task; model unloaded before next stage loads; no session caching |
| P5 | **Idle workers sit unused during pipeline** | Scheduler doesn't prefer idle workers for new stages while reusing loaded workers for repeat inputs |

---

## Solution Architecture

### Core Idea: **Model-Sticky Pipeline Scheduling**

Instead of cycling one worker through all stages, **pin each worker to a stage** and flow inputs through workers like an assembly line.

```
              Input 1    Input 2    Input 3
                │          │          │
    Worker A ─ [cell_a] ─ [cell_a] ─ [cell_a] ─→  (keeps cell_a loaded)
                │          │          │
    Worker B ─ [cell_b] ─ [cell_b] ─ [cell_b] ─→  (keeps cell_b loaded)
                │          │          │
    Worker C ─ [cell_c] ─ [cell_c] ─ [cell_c] ─→  (keeps cell_c loaded)
                │          │          │
              Output 1   Output 2   Output 3
```

With 3 workers and 3 stages, after initial fill-up, all 3 workers run simultaneously (**pipeline parallelism**).

---

## Implementation Plan

### Phase 1: Model Residency Tracking (Foreman)

**Goal**: Foreman knows which model each worker currently has loaded and cached.

#### Task 1.1 — Extend `ModelLoadTracker` with residency map

**File**: `foreman/core/model_load_tracker.py`

Add a new data structure tracking per-worker model residency:

```python
# What each worker currently has loaded (in ONNX session / memory)
_worker_resident_model: Dict[str, str]  # worker_id → partition_id currently loaded

# What each worker has cached on disk (survives unload)
_worker_cached_models: Dict[str, Set[str]]  # worker_id → set of partition_ids on disk
```

Update existing methods:
- `mark_ready()` → also set `_worker_resident_model[worker_id] = partition_id`
- On `UNLOAD_MODEL` ack → remove from `_worker_resident_model` but keep in `_worker_cached_models`
- On `MODEL_LOADED` → add to both maps
- On worker disconnect → clear both maps for that worker

Add new query methods:
```python
def workers_with_model_resident(self, partition_id: str) -> List[str]
def workers_with_model_cached(self, partition_id: str) -> List[str]
def get_resident_model(self, worker_id: str) -> Optional[str]
```

#### Task 1.2 — Worker reports cached models on `WORKER_READY`

**Files**: 
- `common/protocol.py` — add `cached_model_partitions` field to `WORKER_READY` data
- `pc_worker/core/worker.py` — on connect, report `model_loader.loaded_partitions` keys
- Mobile `WorkerWebSocketClient.kt` — on connect, report `ModelArtifactCache.allArtifacts()` partition IDs
- `foreman/core/message_handlers.py` — `_handle_worker_ready()` populates `_worker_cached_models`

This handles reconnecting workers that still have models cached from prior sessions.

---

### Phase 2: Model-Affinity Scheduler (Foreman)

**Goal**: Scheduler prefers workers that already have the needed model loaded/cached.

#### Task 2.1 — Create `ModelAffinityScheduler`

**File**: `foreman/core/scheduling/model_affinity.py` (new)

A scheduler wrapper/strategy that augments any base scheduler with model affinity:

```
Scheduling priority for a DNN task requiring partition P:
  1. Worker with P already RESIDENT in memory (zero load time)
  2. Worker with P CACHED on disk (fast reload, skip download)
  3. Worker that has NO model loaded (idle — use it for a NEW stage)
  4. Worker with a DIFFERENT model loaded (requires unload + load — slowest)
```

**Key rule for pipeline parallelism (P5)**:
> When multiple stages need workers, prefer assigning idle/unloaded workers to NEW stages rather than taking a worker from another stage. This keeps model-loaded workers available for repeat inputs of the same stage.

```python
class ModelAffinityScheduler(TaskScheduler):
    def __init__(self, base_scheduler: TaskScheduler, model_load_tracker: ModelLoadTracker):
        ...

    def select_worker(self, task: SchedulerTask, available: List, all: List) -> Optional[str]:
        partition_id = task.device_requirements.get("model_partition_id")
        if not partition_id:
            return self.base_scheduler.select_worker(task, available, all)

        # Tier 1: resident model match
        resident_workers = self.model_load_tracker.workers_with_model_resident(partition_id)
        candidates = [w for w in available if w in resident_workers]
        if candidates:
            return self.base_scheduler.select_worker(task, candidates, all)

        # Tier 2: cached model match
        cached_workers = self.model_load_tracker.workers_with_model_cached(partition_id)
        candidates = [w for w in available if w in cached_workers]
        if candidates:
            return self.base_scheduler.select_worker(task, candidates, all)

        # Tier 3: idle workers (no model loaded)
        idle_workers = [w for w in available 
                        if self.model_load_tracker.get_resident_model(w) is None]
        if idle_workers:
            return self.base_scheduler.select_worker(task, idle_workers, all)

        # Tier 4: any available (will need unload+load)
        return self.base_scheduler.select_worker(task, available, all)
```

#### Task 2.2 — Integrate into `TaskDispatcher`

**File**: `foreman/core/task_dispatcher.py`

When dispatching DNN pipeline tasks:
- Wrap the active scheduler with `ModelAffinityScheduler`
- Pass `model_load_tracker` reference
- The affinity logic only activates for tasks with `model_partition_id` in requirements

#### Task 2.3 — Smart UNLOAD: Skip if same model needed

**File**: `foreman/core/model_load_tracker.py` → `dispatch_stage_model_load()`

Currently the code sends `UNLOAD_MODEL` for the previous stage before `LOAD_MODEL` for the next. Change:
- If the worker is being assigned the SAME partition again (repeat input), **skip both unload and load**
- If the worker has the target partition CACHED, skip download — send `LOAD_MODEL` with a `from_cache: true` flag so the worker reloads from disk without HTTP download

---

### Phase 3: Per-Input Pipeline Dependencies (Foreman)

**Goal**: Enable true pipeline parallelism — input 2 enters stage 0 while input 1 is in stage 1.

#### Task 3.1 — Per-input dependency chains instead of stage barriers

**File**: `foreman/core/dependency_manager.py`

Currently, stage-N depends on ALL stage-(N-1) tasks (barrier). For pipeline parallelism, each input's chain should be independent:

```
Current (barrier):          Proposed (per-input):
  S0_input1 ─┐               S0_input1 → S1_input1 → S2_input1
  S0_input2 ─┼→ S1_input1    S0_input2 → S1_input2 → S2_input2
  S0_input3 ─┘  S1_input2    S0_input3 → S1_input3 → S2_input3
                 S1_input3
```

Changes needed in `create_pipeline_tasks()`:
- Add a `pipeline_mode` parameter: `"barrier"` (default, backward-compatible) or `"streaming"`
- In `"streaming"` mode:
  - Each input `i` at stage `s` depends ONLY on input `i` at stage `s-1`
  - `dependency_count = 1` (not `len(previous_stage)`)
  - `depends_on = [task_id_of_same_input_previous_stage]`
- The `dependency_map` is auto-generated per input

#### Task 3.2 — SDK: expose pipeline mode in `dnn_pipeline()`

**File**: `developer_sdk/client.py` → `dnn_pipeline()` method

Add optional parameter:
```python
async def dnn_pipeline(
    stages: list,
    pipeline_mode: str = "streaming",  # "barrier" or "streaming"
    ...
)
```

For DNN inference, `"streaming"` is almost always correct because each input is independent.

#### Task 3.3 — Protocol: carry `pipeline_mode` in `SUBMIT_PIPELINE_JOB`

**File**: `common/protocol.py` → `create_submit_pipeline_job_message()`

Add `pipeline_mode` field to the message data.

---

### Phase 4: ONNX Session Caching (Mobile)

**Goal**: Eliminate the main mobile bottleneck — recreating ONNX sessions per task.

#### Task 4.1 — Session cache in `OnnxPartitionExecutor`

**File**: `Prod-Mobile/.../OnnxPartitionExecutor.kt`

```kotlin
object OnnxPartitionExecutor {
    private var cachedSession: OrtSession? = null
    private var cachedPartitionId: String? = null
    private val env = OrtEnvironment.getEnvironment()

    fun execute(modelPath: String, partitionId: String, taskArgsJson: String): JSONObject {
        val session = getOrCreateSession(modelPath, partitionId)
        // ... run inference with session ...
        // DO NOT close session — keep for reuse
    }

    private fun getOrCreateSession(modelPath: String, partitionId: String): OrtSession {
        if (cachedPartitionId == partitionId && cachedSession != null) {
            return cachedSession!!  // Reuse — zero cost
        }
        cachedSession?.close()  // Close previous if different
        cachedSession = env.createSession(modelPath, sessionOptions)
        cachedPartitionId = partitionId
        return cachedSession!!
    }

    fun closeSession() {  // Called on UNLOAD_MODEL
        cachedSession?.close()
        cachedSession = null
        cachedPartitionId = null
    }
}
```

This means:
- First task for a partition: ~50-500ms session creation
- Subsequent tasks for same partition: ~0ms (reuse session)
- Different partition: close old + create new

#### Task 4.2 — Wire UNLOAD_MODEL to session close

**File**: `Prod-Mobile/.../WorkerWebSocketClient.kt` or `TaskProcessor.kt`

On `UNLOAD_MODEL` message:
- Call `OnnxPartitionExecutor.closeSession()`
- Optionally call `ModelArtifactCache.removeArtifact()` if disk space is needed (configurable)

#### Task 4.3 — `LOAD_MODEL` from cache path

**File**: `Prod-Mobile/.../WorkerWebSocketClient.kt`

If `LOAD_MODEL` message contains `from_cache: true`:
- Skip HTTP download
- Look up local path from `ModelArtifactCache.getArtifact(partitionId)`
- Create ONNX session from cached path
- Send `MODEL_LOADED` immediately

---

### Phase 5: Multi-Input Batch Support in SDK

**Goal**: Developer can submit multiple inputs and get all results back, processed in pipeline-parallel fashion.

#### Task 5.1 — SDK auto-replicates inputs across stages

**File**: `developer_sdk/client.py` → `dnn_pipeline()`

Currently the user provides `args_list` per stage. For multi-input:

```python
# Current (single input):
dnn_pipeline(stages=[
    {"name": "cell_a", "model": "cell_a.onnx", "args_list": [input_1]},
    {"name": "cell_b", "model": "cell_b.onnx", "args_list": [None], "pass_upstream_results": True},
    ...
])

# Proposed (multi-input):
dnn_pipeline(stages=[
    {"name": "cell_a", "model": "cell_a.onnx", "args_list": [input_1, input_2, input_3]},
    {"name": "cell_b", "model": "cell_b.onnx", "args_list": [None, None, None], "pass_upstream_results": True},
    ...
])
```

The SDK should auto-expand: if stage 0 has N args and later stages have 1 or `None`, replicate to N:
```python
# Auto-expand: stage_0 has 3 inputs → replicate downstream
for stage in stages[1:]:
    if len(stage["args_list"]) == 1 or stage["args_list"] == [None]:
        stage["args_list"] = [None] * len(stages[0]["args_list"])
```

This creates N independent input chains through the pipeline.

#### Task 5.2 — Update smoke test for multi-input

**File**: `dnn_pipeline_model_paths_smoke.py`

Already tokenizes multiple texts — just needs the args_list to contain per-input tensors:
```python
# Instead of one tensor with shape (2, 64):
# Create two tensors each with shape (1, 64)
for i, text in enumerate(texts):
    input_ids_single = tokenize_texts([text], tokenizer_dir, max_length)
    stage0_inputs.append({
        "transport": "tensor_transport",
        "tensor_payload": serialize_tensor(input_ids_single, compression="zlib"),
    })

stages[0]["args_list"] = stage0_inputs
```

---

### Phase 6: Foreman-Side Automatic Stage-Worker Mapping

**Goal**: Solve P1 & P2 — foreman auto-assigns workers to stages without developer involvement.

#### Task 6.1 — Auto-assign workers to stages at job submission

**File**: `foreman/core/message_handlers.py` → `_handle_pipeline_submission()`

When a DNN pipeline job arrives with `pipeline_mode="streaming"`:
1. Count available workers and pipeline stages
2. **Strategy**:
   - If `workers >= stages`: Assign one worker per stage (round-robin among excess)
   - If `workers < stages`: Assign workers to multiple stages (worker will need model swaps — least-swap assignment)
   - If `workers == 0`: Queue job, assign when workers connect
3. Store stage→worker mapping in `topology_manager` or a new `stage_assignment` map
4. Pre-load models on assigned workers immediately (parallel LOAD_MODEL to all assigned workers)

```python
async def _auto_assign_stages(self, job_id: str, stages: list, available_workers: list):
    n_stages = len(stages)
    n_workers = len(available_workers)

    if n_workers >= n_stages:
        # Ideal: one worker per stage
        assignments = {}
        for i, stage in enumerate(stages):
            assignments[stage["name"]] = available_workers[i]
        # Extra workers are standby for scale-out
    else:
        # Fewer workers than stages — minimize model swaps
        # Assign contiguous stages to same worker
        assignments = {}
        per_worker = math.ceil(n_stages / n_workers)
        for i, stage in enumerate(stages):
            worker_idx = min(i // per_worker, n_workers - 1)
            assignments[stage["name"]] = available_workers[worker_idx]

    return assignments
```

#### Task 6.2 — Handle worker join/leave during pipeline

**File**: `foreman/core/ws_manager.py` → `_cleanup_connection()`

When a worker assigned to a pipeline stage disconnects:
1. Find which stage(s) that worker was handling
2. Reassign to another available worker via `dynamic_topology.replan_job()`
3. Send `LOAD_MODEL` to new worker for the affected stage's partition
4. Re-queue any pending tasks for that stage

When a NEW worker connects while a pipeline job is running:
1. Check if any stage is bottlenecked (tasks queued with no available worker)
2. If so, assign new worker to that stage, load model, start processing

---

### Phase 7: README Updates

Update all README files to reflect the new multi-worker pipeline architecture.

#### Task 7.1 — `Prod-Backend/README.md` (root)

Currently missing / just points to tests. Create a proper root README covering:
- Project overview (distributed computation on idle Android phones)
- System architecture diagram (Developer SDK ↔ Foreman ↔ Workers)
- Quick start guide (foreman + worker + smoke test)
- Component overview (links to sub-READMEs)
- New pipeline parallelism feature documentation

#### Task 7.2 — `foreman/README.md`

Add/update sections for:
- Model-affinity scheduler
- Pipeline modes (barrier vs streaming)
- Auto stage-worker assignment
- Model residency tracking
- Worker lifecycle with model caching

#### Task 7.3 — `developer_sdk/README.md`

Add/update:
- Multi-input DNN pipeline example
- `pipeline_mode` parameter documentation
- Updated `dnn_pipeline()` API reference
- Multi-worker deployment guide

#### Task 7.4 — `pc_worker/README.md`

Add/update:
- Model caching behavior and session reuse
- `LOAD_MODEL` from cache vs download
- Performance characteristics with model affinity

#### Task 7.5 — `Prod-Mobile/README.md`

Add/update:
- ONNX session caching behavior
- Model cache reporting on reconnect
- Performance improvements from session reuse
- Multi-worker deployment instructions

#### Task 7.6 — `common/README.md`

Add/update:
- New protocol fields (`cached_model_partitions`, `from_cache`, `pipeline_mode`)
- Updated message flow diagrams

#### Task 7.7 — `tests/README.md`

Add:
- Multi-worker test scenarios
- Pipeline parallelism test instructions

---

## Execution Order & Dependencies

```
Phase 1 (Model Residency Tracking)
  ├── Task 1.1: ModelLoadTracker residency map
  └── Task 1.2: Worker reports cached models
         │
Phase 2 (Model-Affinity Scheduler)     ← depends on Phase 1
  ├── Task 2.1: ModelAffinityScheduler
  ├── Task 2.2: Integrate into TaskDispatcher
  └── Task 2.3: Smart UNLOAD logic
         │
Phase 3 (Per-Input Dependencies)        ← independent of Phase 2
  ├── Task 3.1: Streaming dependency mode
  ├── Task 3.2: SDK pipeline_mode param
  └── Task 3.3: Protocol changes
         │
Phase 4 (Mobile Session Caching)        ← independent of Phase 2-3
  ├── Task 4.1: OnnxPartitionExecutor session cache
  ├── Task 4.2: Wire UNLOAD to session close
  └── Task 4.3: LOAD_MODEL from cache
         │
Phase 5 (Multi-Input Batch in SDK)      ← depends on Phase 3
  ├── Task 5.1: SDK auto-replicates inputs
  └── Task 5.2: Update smoke test
         │
Phase 6 (Auto Stage-Worker Mapping)     ← depends on Phase 1 + 2
  ├── Task 6.1: Auto-assign stages at submission
  └── Task 6.2: Handle worker join/leave
         │
Phase 7 (README Updates)                ← after all phases
  └── Tasks 7.1-7.7: Update all documentation
```

**Parallelizable work**: Phase 3 and Phase 4 can be implemented in parallel. Phase 1 is the foundation.

---

## Expected Outcome

### Before (Single Worker, Sequential)
```
Time:  ──────────────────────────────────────────────────────→
Worker A: [load_a][run_a][unload_a][load_b][run_b][unload_b][load_c][run_c]
                                                                        ↓
                                                                    Result 1
Total: 8 steps × N inputs = 8N steps (serial)
```

### After (3 Workers, Pipeline Parallel)
```
Time:  ────────────────────────────────────────→
Worker A: [load_a][run_a_in1][run_a_in2][run_a_in3]   (model stays loaded)
Worker B:         [load_b]   [run_b_in1][run_b_in2][run_b_in3]
Worker C:                    [load_c]   [run_c_in1][run_c_in2][run_c_in3]
                                            ↓        ↓        ↓
                                        Result 1  Result 2  Result 3

Total: 3 loads + (N + stages - 1) steps  (pipeline parallel)
For N=3: 3 loads + 5 inference steps vs 24 steps before → ~4x speedup
```

### Graceful Degradation
| Workers Available | Behavior |
|-------------------|----------|
| ≥ stages | Full pipeline parallelism (1 worker per stage) |
| < stages but > 1 | Partial parallelism (some workers handle multiple stages with model swaps) |
| 1 worker | Falls back to current sequential behavior (backward compatible) |
| 0 workers | Job queued until workers connect |
| Worker disconnects mid-pipeline | Reassign stage to available worker, reload model, retry pending tasks |
| New worker joins mid-pipeline | Assigned to bottleneck stage if any |
