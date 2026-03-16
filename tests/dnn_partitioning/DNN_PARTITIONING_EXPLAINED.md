# Distributed DNN Partitioning — How It Works

## Table of Contents

1. [The Core Problem](#1-the-core-problem)
2. [What Is a DNN as a DAG?](#2-what-is-a-dnn-as-a-dag)
3. [The Dependency Counter](#3-the-dependency-counter)
4. [Step-by-Step: ResNet Pipeline](#4-step-by-step-resnet-pipeline)
5. [Step-by-Step: Inception Pipeline](#5-step-by-step-inception-pipeline)
6. [How CROWDio Executes This](#6-how-crowdio-executes-this)
7. [The Kotlin ↔ Python Bridge (Chaquopy)](#7-the-kotlin--python-bridge-chaquopy)
8. [Data Flow: What Gets Sent Between Devices](#8-data-flow-what-gets-sent-between-devices)
9. [Why Atomic Counters Matter on Mobile](#9-why-atomic-counters-matter-on-mobile)
10. [Complete Execution Trace](#10-complete-execution-trace)

---

## 1. The Core Problem

A Deep Neural Network (DNN) like ResNet-50 has ~25 million parameters and
~50 layers. A single mobile phone may not have enough memory or CPU power
to run the whole model. The solution: **split the model across multiple
mobile devices** in a crowd and have each device run a slice.

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│ Phone A  │    │ Phone B  │    │ Phone C  │
│ Layers   │───►│ Layers   │───►│ Layers   │───► Prediction
│  1 – 10  │    │ 11 – 30  │    │ 31 – 50  │
└──────────┘    └──────────┘    └──────────┘
     stem        residuals       classifier
```

But DNNs are not always a straight chain. Models like **ResNet** have
**skip connections** and models like **Inception** have **parallel branches**.
This means the execution graph is a **DAG** (Directed Acyclic Graph), not
a simple pipeline. We need a mechanism to know when a layer is ready to
execute — that mechanism is the **dependency counter**.

---

## 2. What Is a DNN as a DAG?

Every DNN can be represented as a graph where:
- **Nodes** = layers (convolution, pooling, dense, etc.)
- **Edges** = data flow (the output tensor of one layer feeds into the next)

### Simple Chain (VGG-style)

```
Layer 1 → Layer 2 → Layer 3 → Layer 4 → Softmax
```

Each layer depends on exactly 1 predecessor. Simple.

### Skip Connection (ResNet-style)

```
Layer 1 ──────────────► Layer 2 ──► ADD ──► Layer 4
   │                                 ▲
   └──── Skip (1×1 conv) ───────────┘
```

The **ADD** node has **2 predecessors**: Layer 2 AND the skip branch.
It cannot execute until BOTH are done.

### Parallel Branches (Inception-style)

```
              ┌─── 1×1 conv ───┐
              ├─── 3×3 conv ───┤
Stem ────────►├─── 5×5 conv ───┼──► CONCAT ──► Classify
              └─── MaxPool  ───┘
```

The **CONCAT** node has **4 predecessors**. It waits for ALL four
branches before concatenating their outputs along the channel axis.

---

## 3. The Dependency Counter

### The Concept

Each node $v_j$ in the DAG is assigned a counter equal to its **in-degree**
(number of incoming edges):

$$\text{counter}(v_j) = |pred(v_j)|$$

| Node (Layer)     | Predecessors              | Counter Init |
|------------------|---------------------------|:------------:|
| partition_0      | (none — first layer)      |      0       |
| partition_1      | partition_0               |      1       |
| skip_branch      | partition_0               |      1       |
| **fuse (ADD)**   | partition_1, skip_branch  |    **2**     |
| classify         | fuse                      |      1       |

### The Rule

```
When a layer completes:
    For each successor of that layer:
        counter[successor] -= 1               ← atomic decrement
        if counter[successor] == 0:
            dispatch(successor)               ← ALL deps met → execute!
```

### Why "Atomic"?

On mobile (Kotlin/Android), multiple threads may complete different branches
at almost the same time. If two threads both try to decrement the fuse
counter simultaneously without synchronization, you get a **race condition**.

**AtomicInteger.decrementAndGet()** is a single CPU instruction (CAS —
Compare-And-Swap) that is:
- **Lock-free**: no mutex, no blocking
- **Thread-safe**: impossible to corrupt
- **Fast**: ~3 nanoseconds on ARM

```kotlin
// Kotlin — what happens when partition_1 finishes:
val remaining = dependencyCounters["fuse"]!!.decrementAndGet()  // 2 → 1
// remaining = 1 → NOT zero → do nothing yet

// Later, when skip_branch finishes:
val remaining = dependencyCounters["fuse"]!!.decrementAndGet()  // 1 → 0
// remaining = 0 → FIRE! Execute fuse layer
```

---

## 4. Step-by-Step: ResNet Pipeline

Our `pipeline_dnn_partitioning.py` implements this as a 4-stage CROWDio
pipeline. Here is exactly what happens:

### Stage 0 — `partition_model` (1 task, runs once)

**Purpose**: Analyse the model, build the DAG, create a partition plan.

```
Input:  { model: "resnet", input_size: 224, num_partitions: 3 }
Output: { layers: [...], edges: [...], plan: {...}, input_tensor: {...} }
```

The DAG it produces:

```python
layers = [
    { "layer_id": "partition_0",  "op": "conv_block",       "partition_idx": 0 },
    { "layer_id": "partition_1",  "op": "residual_block",   "partition_idx": 1 },
    { "layer_id": "skip_branch",  "op": "skip_connection",  "partition_idx": 0 },
    { "layer_id": "fuse",         "op": "add",              "partition_idx": 2 },
    { "layer_id": "classify",     "op": "classify",         "partition_idx": 2 },
]

edges = [
    ("partition_0", "partition_1"),   # stem feeds main branch
    ("partition_0", "skip_branch"),   # stem also feeds skip
    ("partition_1", "fuse"),          # main branch → fuse
    ("skip_branch", "fuse"),          # skip branch → fuse
    ("fuse", "classify"),             # fused output → classifier
]
```

The `plan` maps each layer to a device: `{ "partition_0": 0, "partition_1": 1, ... }`

### Stage 1 — `run_layer_slice` (3 tasks, run in parallel)

**Purpose**: Each task runs ONE layer's inference on a worker device.

CROWDio creates 3 tasks (one per "slice" layer: partition_0, partition_1,
skip_branch). The dependency counter system means:

```
Task 1-0 (partition_0):   blocked until stage-0 completes → runs first
Task 1-1 (partition_1):   blocked until stage-0 completes → runs in parallel
Task 1-2 (skip_branch):   blocked until stage-0 completes → runs in parallel
```

All three stage-1 tasks are unblocked together when stage-0 finishes.
They run on different workers simultaneously.

Each task:
1. Reads the partition plan from upstream
2. Determines which layer it is (via `layer_index`)
3. Simulates the compute (proportional to FLOPs)
4. Returns the output activation tensor

### Stage 2 — `fuse_branches` (1 task, blocked)

**Purpose**: Combine outputs from all Stage-1 tasks.

This task has `dependency_count = 3` (one for each Stage-1 task).
The CROWDio foreman decrements this counter each time a Stage-1 task
reports completion:

```
partition_0 completes → counter: 3 → 2  (still blocked)
skip_branch completes → counter: 2 → 1  (still blocked)
partition_1 completes → counter: 1 → 0  (UNBLOCKED! → dispatch)
```

When unblocked, it receives ALL upstream results via `pass_upstream_results`:

```python
task_input = {
    "upstream_results": {
        "job123_task_1_0": { "layer_id": "partition_0", "output_tensor": {...} },
        "job123_task_1_1": { "layer_id": "partition_1", "output_tensor": {...} },
        "job123_task_1_2": { "layer_id": "skip_branch", "output_tensor": {...} },
    }
}
```

The fuse function:
- For ResNet (2 tensors same shape) → element-wise **add**
- For Inception (4 tensors different channels) → channel-axis **concatenate**

### Stage 3 — `classify` (1 task, blocked until fuse completes)

**Purpose**: Run Global Average Pooling → Dense → Softmax.

```
Input:  fused feature map (1, 112, 112, 64)
         ↓
GAP:    average over spatial dims → (64,)
         ↓
Dense:  (64,) × W(64, 1000) → (1000,)
         ↓
Softmax: e^logit / Σe^logits → probabilities
         ↓
Output: { predicted_class: 487, confidence: 0.0031, top5: [...] }
```

---

## 5. Step-by-Step: Inception Pipeline

With `--model inception`, the DAG changes:

```
              ┌─── branch_1x1 (64 filters)  ───┐
              ├─── branch_3x3 (128 filters) ───┤
stem ────────►├─── branch_5x5 (32 filters)  ───┼──► concat ──► classify
              └─── branch_pool (32 filters) ───┘
```

Stage-1 creates **5 tasks** (stem + 4 branches). The concat layer has
`dependency_count = 5`. The pipeline looks like:

```
Stage 0:  partition_model  → 1 task

Stage 1:  run_layer_slice  → 5 tasks
          ├── task 0: stem        (partition 0)
          ├── task 1: branch_1x1  (partition 0)
          ├── task 2: branch_3x3  (partition 1)
          ├── task 3: branch_5x5  (partition 2)
          └── task 4: branch_pool (partition 3)

Stage 2:  fuse_branches    → 1 task (blocked, counter = 5)
          When all 5 stage-1 tasks finish → counter hits 0 → unblocked
          Concatenates: (1,112,112,64) + (1,112,112,128) + (1,112,112,32) + (1,112,112,32)
                      = (1,112,112,256)

Stage 3:  classify         → 1 task (blocked, counter = 1)
          GAP → Dense(256, 1000) → Softmax → predicted class
```

---

## 6. How CROWDio Executes This

The CROWDio platform has three components that work together:

### 6.1 Developer SDK (this Python script)

```python
results = await pipeline([
    { "func": partition_model,  "args_list": [config] },
    { "func": run_layer_slice,  "args_list": slice_args, "pass_upstream_results": True },
    { "func": fuse_branches,    "args_list": [None],     "pass_upstream_results": True },
    { "func": classify,         "args_list": [None],     "pass_upstream_results": True },
])
```

The SDK:
1. Serializes each function to source code
2. Sends a `SUBMIT_PIPELINE_JOB` message to the Foreman via WebSocket
3. Waits for `JOB_RESULTS`

### 6.2 Foreman (Central Server)

The Foreman's **DependencyManager** does the heavy lifting:

```
On receiving SUBMIT_PIPELINE_JOB:
  1. Create all tasks in database
  2. Set dependency_count for each task:
     - Stage 0 tasks: count = 0 (pending → immediately dispatchable)
     - Stage 1 tasks: count = 1 (blocked, depends on stage 0)
     - Stage 2 tasks: count = N (blocked, depends on ALL stage-1 tasks)
     - Stage 3 tasks: count = 1 (blocked, depends on stage 2)
  3. Dispatch stage-0 tasks to available workers

On receiving TASK_RESULT (a task completed):
  1. Store the result
  2. For each downstream task:
     a. Inject the upstream result into downstream args (if pass_upstream_results)
     b. Atomically decrement dependency_count
     c. If count reaches 0 → change status from "blocked" to "pending"
  3. Dispatch all newly-pending tasks to workers
```

The critical database operation:

```python
# foreman/db/crud.py
async def decrement_dependency_count(session, task_id):
    task = await session.get(TaskModel, task_id)
    new_count = max(0, (task.dependency_count or 0) - 1)
    task.dependency_count = new_count
    if new_count == 0 and task.status == "blocked":
        task.status = "pending"    # ← UNBLOCKED!
    await session.commit()
    return new_count, task.status
```

### 6.3 Workers (Distributed Executors)

Each worker:
1. Connects to Foreman via WebSocket
2. Receives `ASSIGN_TASK` message with serialized function code + args
3. Executes the function (e.g., `run_layer_slice(task_input)`)
4. Sends `TASK_RESULT` back to Foreman

Workers don't know about the DAG. They just execute whatever function
they receive. The Foreman handles all the dependency logic.

---

## 7. The Kotlin ↔ Python Bridge (Chaquopy)

On a real Android deployment, the architecture splits:

```
┌─────────────────────── Android App ──────────────────────┐
│                                                          │
│  ┌──────────── Kotlin Layer ────────────┐                │
│  │                                      │                │
│  │  DnnDagManager                       │                │
│  │  ├─ dependencyCounters: Map<AtomicInteger>            │
│  │  ├─ layerOutputs: Map<ByteArray>     │                │
│  │  └─ onLayerCompleted(id, tensor)     │                │
│  │      → decrementAndGet()             │                │
│  │      → if 0: trigger next layer      │                │
│  │                                      │                │
│  │  TensorTransport (Ktor)              │                │
│  │  ├─ sendTensor(host, port, data)     │  D2D Comms     │
│  │  └─ receiveTensor(port, callback)    │◄──────────────►│
│  │                                      │                │
│  └──────────┬───────────────────────────┘                │
│             │ Chaquopy Bridge                            │
│  ┌──────────▼───────────────────────────┐                │
│  │                                      │                │
│  │  Python Layer                        │                │
│  │  ├─ numpy for tensor ops             │                │
│  │  ├─ TFLite interpreter               │                │
│  │  └─ run_layer(id, input) → output    │                │
│  │                                      │                │
│  └──────────────────────────────────────┘                │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Why this split?

| Concern               | Kotlin     | Python           | Reason                                   |
|-----------------------|------------|------------------|------------------------------------------|
| DAG management        | ✅          |                  | AtomicInteger is lock-free, native JVM    |
| Network (D2D)         | ✅ (Ktor)   |                  | Coroutines, non-blocking sockets          |
| Tensor math           |            | ✅ (NumPy)        | Optimized C backend, vectorized ops       |
| Model inference       |            | ✅ (TFLite/Keras) | ML ecosystem is Python-native             |
| Thread safety         | ✅          |                  | JVM concurrency primitives (CAS)          |

### Data transfer between Kotlin and Python:

```kotlin
// Kotlin → Python (via Chaquopy)
val npModule = Python.getInstance().getModule("numpy")
val inputNp = npModule.callAttr("frombuffer", tensorBytes, "float32")
// This is near zero-copy! Chaquopy maps Java byte[] directly to NumPy buffer.

// Python → Kotlin
val outputBytes = result.callAttr("tobytes").toJava(ByteArray::class.java)
// Also near zero-copy. No JSON serialization overhead.
```

---

## 8. Data Flow: What Gets Sent Between Devices

### Between CROWDio workers (this test)

Activation tensors are serialized as NumPy `.npy` format → base64 string:

```python
# Serialize (sender)
buf = io.BytesIO()
np.save(buf, tensor)                               # binary .npy format
b64 = base64.b64encode(buf.getvalue()).decode()     # safe for JSON/WebSocket

# Deserialize (receiver)
raw = base64.b64decode(b64)
tensor = np.load(io.BytesIO(raw))                   # back to ndarray
```

### Between Kotlin devices (D2D via Ktor)

Raw `float32` bytes over TCP sockets — no base64 overhead:

```
Wire format:  [4B: layer_id_len][layer_id_bytes][4B: tensor_len][tensor_bytes]
Example:      [11]["partition_1"][802816][...float32 data...]
                                  ↑
                        1 × 112 × 112 × 64 × 4 bytes = 3,211,264 bytes
```

### Tensor sizes for ResNet example (224×224 input)

| Layer         | Output Shape           | Size (float32)  |
|---------------|------------------------|-----------------|
| partition_0   | (1, 112, 112, 64)     | 3.07 MB         |
| partition_1   | (1, 112, 112, 64)     | 3.07 MB         |
| skip_branch   | (1, 112, 112, 64)     | 3.07 MB         |
| fuse (add)    | (1, 112, 112, 64)     | 3.07 MB         |
| classify      | (1000,) probabilities  | 0.004 MB        |

On Wi-Fi Direct (~250 Mbps), sending 3 MB takes ~0.1ms. The bottleneck
is **computation**, not communication — which is the ideal scenario for
partitioned DNNs.

---

## 9. Why Atomic Counters Matter on Mobile

### The Race Condition Problem

Imagine partition_1 and skip_branch finish at nearly the same time on
two different devices. Both send their results to Device C (which runs
the fuse layer). Without atomics:

```
Thread A (receives partition_1 result):     Thread B (receives skip_branch result):
    count = fuse.counter      // reads 2        count = fuse.counter      // reads 2
    count = count - 1         // 2 → 1          count = count - 1         // 2 → 1
    fuse.counter = count      // writes 1       fuse.counter = count      // writes 1
    if count == 0: execute()  // 1 ≠ 0, skip    if count == 0: execute()  // 1 ≠ 0, skip
```

**Bug!** Counter ends at 1 instead of 0. The fuse layer **never executes**.
The pipeline deadlocks.

### The Atomic Solution

```kotlin
// AtomicInteger uses CPU-level Compare-And-Swap (CAS)
Thread A: decrementAndGet()  // atomically: 2 → 1, returns 1
Thread B: decrementAndGet()  // atomically: 1 → 0, returns 0  ← FIRES!
```

CAS is a single CPU instruction. It is impossible for two threads to see
the same value. One will always see the decremented result of the other.

### Performance Comparison on Mobile

| Mechanism               | Latency  | Blocks Thread? | Battery Impact |
|-------------------------|----------|:--------------:|:--------------:|
| `AtomicInteger` (CAS)   | ~3 ns   | No             | None           |
| `synchronized` (lock)   | ~50 ns  | Yes            | Low            |
| Polling loop             | ~1 ms   | Yes (spinning) | **High**       |
| Cloud round-trip         | 50-200ms| Yes (I/O)      | **High**       |

On battery-constrained mobile devices, `AtomicInteger` is the clear winner.

---

## 10. Complete Execution Trace

Here's the full timeline when you run:

```bash
uv run python tests/dnn_partitioning/pipeline_dnn_partitioning.py --model resnet
```

```
TIME   EVENT                                    WHERE
─────  ─────────────────────────────────────     ──────────────
0.00s  SDK sends SUBMIT_PIPELINE_JOB             Client → Foreman
       ├─ Stage 0: 1 task  (partition_model)
       ├─ Stage 1: 3 tasks (run_layer_slice × 3)
       ├─ Stage 2: 1 task  (fuse_branches)
       └─ Stage 3: 1 task  (classify)

0.01s  Foreman creates 6 tasks in database       Foreman
       ├─ task_0_0: status=pending,  dep_count=0
       ├─ task_1_0: status=blocked,  dep_count=1
       ├─ task_1_1: status=blocked,  dep_count=1
       ├─ task_1_2: status=blocked,  dep_count=1
       ├─ task_2_0: status=blocked,  dep_count=3
       └─ task_3_0: status=blocked,  dep_count=1

0.02s  Foreman dispatches task_0_0 to Worker A   Foreman → Worker A
       (only task with dep_count=0)

0.15s  Worker A completes partition_model         Worker A → Foreman
       Result: { layers, edges, plan, input_tensor }

0.16s  Foreman decrements dependents:            Foreman
       ├─ task_1_0: dep_count 1→0 → PENDING ✅
       ├─ task_1_1: dep_count 1→0 → PENDING ✅
       └─ task_1_2: dep_count 1→0 → PENDING ✅
       Upstream results injected into each task's args

0.17s  Foreman dispatches:                       Foreman → Workers
       ├─ task_1_0 → Worker A (partition_0)
       ├─ task_1_1 → Worker B (partition_1)
       └─ task_1_2 → Worker C (skip_branch)

0.95s  Worker C completes skip_branch            Worker C → Foreman
       Foreman decrements task_2_0: 3→2 (still blocked)

1.20s  Worker A completes partition_0             Worker A → Foreman
       Foreman decrements task_2_0: 2→1 (still blocked)

1.80s  Worker B completes partition_1             Worker B → Foreman
       Foreman decrements task_2_0: 1→0 → PENDING ✅
       All 3 results injected into task_2_0 args

1.81s  Foreman dispatches task_2_0 → Worker A    Foreman → Worker A
       (fuse_branches — has all 3 tensors)

1.90s  Worker A completes fuse_branches           Worker A → Foreman
       Fused tensor: (1, 112, 112, 64)
       Foreman decrements task_3_0: 1→0 → PENDING ✅

1.91s  Foreman dispatches task_3_0 → Worker A    Foreman → Worker A
       (classify — has fused feature map)

1.95s  Worker A completes classify                Worker A → Foreman
       Result: { predicted_class: 487, confidence: 0.0031 }

1.96s  Foreman sends JOB_RESULTS to client       Foreman → Client
       Pipeline complete! 🎉
```

### Key Observations

1. **Stage 0 → Stage 1 transition**: One decrement unblocks THREE tasks (fan-out)
2. **Stage 1 → Stage 2 transition**: THREE decrements needed to unblock ONE task (fan-in)
3. **Parallelism**: Workers A, B, C all execute stage-1 tasks simultaneously
4. **No polling**: Workers and foreman only act on events (WebSocket messages)
5. **The counter is the gatekeeper**: `fuse_branches` physically cannot run until the counter reaches 0

---

## Summary

| Concept                | What It Does                                                    |
|------------------------|-----------------------------------------------------------------|
| **DAG**                | Represents model layers and their data dependencies             |
| **Dependency Counter** | Integer per layer = number of unfinished predecessors           |
| **Decrement**          | When a layer finishes, all successors' counters decrease by 1   |
| **Fire Rule**          | Counter reaches 0 → layer is ready to execute                  |
| **AtomicInteger**      | Thread-safe decrement on Kotlin/Android (CAS instruction)       |
| **Partition Plan**     | Maps each layer to a device based on CapScore                   |
| **pass_upstream**      | Injects predecessor results into successor's function arguments |
| **Fuse/Concat**        | Join point — counter = branch count (2 for ResNet, 4 for Inception) |
| **Checkpointing**      | `run_layer_slice` checkpoints progress for crash recovery       |
