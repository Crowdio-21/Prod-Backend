# Build report: reconfiguration-aware placement and partitioning (Foreman)

Companion to [step0_report.md](step0_report.md). This report covers each component: the files changed, the interfaces chosen, the tests added, and any assumption the code didn't confirm.

## Decisions taken without an answer

These are the Step 0 questions, settled with the recommended defaults. Each one is reversible.

| Question | Decision |
|---|---|
| Base branch | Built on `main`. The pure **B0** policy implements the 4-tier ranking from the brief (resident > cached on disk > idle > evict). The live legacy path (the SDK's fixed DAG) is untouched. `upstream/feature/stable-distributed` is **not** merged. |
| Existing `foreman_policy.py` / `reconfiguration.py` | Kept, still dormant, still importable. The wrong-shard unload bug is fixed. The new live path is `foreman/core/plan_coordinator.py`. |
| Wire format | The contract's message names, inside the `crowdio.reconfiguration` envelope at **version 2**. Version 1 still parses. Android must move from its uppercase `PREFETCH`/`ASSIGN`/... commands to v2. |
| PC workers | `pc_worker` is unchanged. Integration tests use in-memory fake workers plus one real-WebSocket smoke test. |

## Contract changes the SDK and mobile agents need to review

Every one is marked `# contract extension` in `foreman/planning/contract.py`.

- **ShardSpec**:
  - `flops_method`: how FLOPs were estimated (`analytic_matmul`).
  - `file_name`: file path relative to the manifest.
- **Manifest**: `default_composition`, the fixed cut that B0 and old SDKs use.
- **WorkerState**:
  - `stage_stats.samples` and `stage_stats.shard_id`, so compute estimates carry a sample count;
  - `peer_endpoint`, the `host:port` for direct and peer traffic.
- **NextHop**: `address`, the peer endpoint for the direct path.
- **Plan**: `replicas`, the data-parallel replicas for B4.
- **Messages**:
  - `LoadMsg.stage`, `LoadMsg.next_hop`, `LoadMsg.source`;
  - `PrefetchMsg.source`;
  - `ProbeMsg.peer` and `ProbeMsg.url`;
  - `route`, a next-hop update sent on a reroute (this is what triggers the worker's resend);
  - `msg_ack`, a transport-level ack for every Foreman command except `load` (acked by `load_result`) and `run`;
  - `probe_result`.
- **Client ↔ Foreman**: `submit_inference`, `inference_accepted`, `inference_result`, `inference_error`.
- **Activation**: `ActivationMsg.compute_ms`, a per-stage summary.
- **Dedup key at the receiver**: `(request_id, epoch, stage)`, where `stage` is the stage about to run. A direct activation carries the producing stage, so the receiver adds 1.
- **Pinned shards**: `pinned=True` means the shard's boundaries are fixed cuts, so it is always its own stage. Example: the classification head.

## Components

### 1. Data model and flags

- **Files**: `foreman/planning/contract.py`, `flags.py`, `config.py`, `eventlog.py`.
- **Interface**:
  - Pydantic v2 models, frozen, with unknown fields ignored. The manifest validator checks atom order, the shard ranges, and that `default_composition` tiles every atom.
  - `FeatureFlags.from_env()` reads `CROWDIO_FLAGS` (a comma list or JSON). `RunConfig.load(yaml)`.
  - The flags are printed and logged at run start (`[FLAGS] ...` plus a `run_start` event).
- **Tests**: `test_components.py`, covering flags, config, envelope round trip, unknown types, the legacy manifest adapter, and the event log.

### 2. Telemetry and estimation

- **Files**: `foreman/planning/telemetry.py`, `profiles.py`, `types.py`.
- **Estimates**:
  - Each worker has per-direction EWMA mean and variance for bandwidth, plus RTT. Every `Est` carries `n` and `age_s`.
  - A small-sample standard-error floor (`|mean|·0.25/√n`) stops one sample from looking exact.
  - The spread grows with age (`age_std_growth`).
- **Passive first**: bandwidth comes from timed transfers. Sources are `load_result.timings` (bytes fetched over fetch time), the worker's own `observed_up/down_bps`, and probe results.
- **Active probe**: only when an estimate is older than `probe_after_s`, or the worker is idle with no data. Rate-limited per worker (`probe_min_interval_s`). Served by `GET /api/probe/{size}` and `POST /api/probe`.
- **Pairwise links**: `peer_links` when fresh; otherwise relay with `min(up_i, down_j)` and RTT summed over the two legs.
- **Compute**:
  - From `stage_stats` (p50 updates the mean; the p95–p50 gap sets the variance) and from `ActivationMsg.compute_ms`. These are folded into per-device-class profiles.
  - Phase models (`load`, `init`, `write`, `unload`, `peak_mem`) are linear in shard bytes and fitted online from `load_result` (`ProfileLearner`), starting from priors with `n=0`.
- **Planners only ever see an immutable `Snapshot`.**
- **Legacy heartbeats**:
  - Both PC shapes are accepted: flat, and nested under `performance_metrics`.
  - Android's `net` / `res` / `on_disk` fields are mapped.
  - `network_speed_mbps` is **not** used as bandwidth (it measures utilisation).
  - Android's `residency.shard` is **not** trusted (it holds a task id).
- **Tests**: EWMA age and sample count, the probe rules, both heartbeat shapes, peer links.

### 3. Shard registry and serving

- **Files**: `foreman/planning/registry.py`, `foreman/api/shard_routes.py`, `tests/dnn_partitioning/sentiment_onnx_pipeline/export_manifest.py`.
- **Interface**:
  - `ShardRegistry.load(path)` accepts a `manifest.json`, or adapts today's `partition_metadata.json`. In that case the atoms are embed, L0..L5 and head, `cell_c` is pinned, and the sha256 is computed.
  - `verify()` checks size and sha256.
  - The holder map comes from heartbeats (`Snapshot.holders`).
  - `plan_evictions(...)` is a pure LRU that never evicts protected files (the resident shard, in-flight fetches).
- **Serving**:
  - `GET /api/shards/{id}` supports `Range` (206, suffix ranges, 416) and sends `X-Content-SHA256` and an `ETag`.
  - `GET /api/shards/manifest`.
- **Exporter**:
  - Exports every contiguous range on the cut grid, plus `head` and `full`, plus a contract-format `manifest.json`.
  - Verified against PyTorch with grid 3: the default chain and `full` both match to a max abs diff of 9.5e-7.
  - Grid 1 is supported but writes about 30 files.
- **Tests**: `test_shard_routes_and_legacy.py` (range, resume, checksum, probe), and LRU in `test_components.py`.

### 4. Cost model (pure)

- **File**: `foreman/planning/cost_model.py`.
- **Interface**:
  - `c_switch`, `c_fetch` (with source selection over eligible donors), `t_comm` (min of direct and relay), `t_comp`, `memory_feasible`, `pipeline_cost`.
  - Every value is an interval `Q(lo, mid, hi)`. `PipelineCost` returns `T_total`, `T_steady`, `C_switch_total` and `C_switch_max` separately, plus a `stale` flag.
- **Formula**:
  - Exactly as briefed, plus the two I/O legs (Foreman → stage 1, stage K → Foreman).
  - `include_io_legs=False` reproduces the brief's formula verbatim.
  - Compute time is scaled by `(1 + cpu)` and by a temperature slowdown above `temp_ref_c`.
- **Tests**: `test_cost_model.py`. Hand-computed cases cover:
  - all three residency states, with and without peer fetch;
  - ineligible donors (battery, heat, critical path, upload cap);
  - relay vs direct;
  - the pipeline formula, with and without I/O legs;
  - bounds widening with spread and age;
  - memory feasibility under single residency.

### 5. Planner and policies (pure)

- **Files**: `foreman/planning/planner.py`, `policies.py`.
- **Search**:
  - `exhaustive_search` is the exact oracle. It applies the memory filter first, then a branch-and-bound cut.
  - `pareto_dp_search` keeps a Pareto set of (finish, period) per state (boundary, last shard, last worker, used workers). It is exact.
  - `scalar_dp_search` drops the used-worker set and is a heuristic.
  - `dp_gap` reports the scalar DP's gap against exhaustive. Over 150 random 5-worker instances: Pareto exact 100%; scalar exact 98%, mean gap 0.06%, max gap 7%.
- **Policies**:
  - `Policy.plan(snapshot, current_plan, n, objective) -> PolicyDecision` covers B0, B1, B2, B3, B4, B5a, B5b, S-aware and S-aware-placement.
  - Each policy declares a `switch_rule`: `static`, `always` or `break_even`.
  - `build_plan` sets next hops, allowed routes and backups.
- **Objectives**: `min_total`, and `min_steady_deadline` (the shortest period among deadline-feasible plans).
- **Tests**: `test_planner.py`, 130 cases including seeded property tests:
  - feasibility: distinct workers, so never two resident shards on one worker; memory; atoms tiled;
  - bound pruning is exact;
  - Pareto DP equals exhaustive;
  - with switching cost zero, S-aware equals B2;
  - with bandwidth terms zero, B1 equals B0;
  - the 4-tier order;
  - B4 replication;
  - re-cut search.

### 6. Controller (pure)

- **File**: `foreman/planning/controller.py`.
- **Interface**: `detect_triggers(...)` and `decide(snapshot, plan, n, policy, objective, state, cfg, triggers, now) -> (ControllerDecision, ControllerState)`.
- **Triggers**: bandwidth, RTT, CPU and temperature drift against the baseline captured at the last switch; worker join and leave; deadline risk; queue growth; aggregated problems.
- **Switch rule**: J(new) + margin < J(current). The margin is estimate uncertainty plus `margin_abs_s` plus `margin_frac`. The break-even form is logged with every decision.
- **Hysteresis**: `persist_k`, cooldown, and a per-window switch cap.
- **Stale estimates**: these block non-forced switches.
- **Forced replans** (worker lost, load failed, OOM, next hop down) bypass the margin and hysteresis.
- **Change found in testing**: an *initial* plan under a break-even policy must still beat the residency-preserving affinity plan by the margin. Without this, noisy first estimates paid a 15 s cold load for a gain that never materialised.
- **Tests**: `test_controller.py`:
  - a small-N switch is rejected; a large-N switch is taken;
  - a warm replacement breaks even sooner than a cold one;
  - persistence, cooldown, the cap, the stale guard;
  - forced replans; static policies; B2; trigger detection.

### 7. Route and source selection

- **Files**: `foreman/planning/routes.py`, plus source selection in `cost_model.c_fetch`.
- **RouteSelector**: a margin and a minimum dwell time before flipping; immediate fallback to the relay when a peer is unreachable or stale.
- **CapabilityTracker**: test on join, on network change, and every `capability_retest_s`.
- **Donors**: serve from disk only. A donor must have battery and heat headroom, be under the upload cap, and be off the critical path.
- **Tests**: hysteresis, dwell, fallback, capability rules, and the donor cases in the cost-model tests.

### 8. DAG builder and dispatcher, epochs

- **Files**: `foreman/core/plan_coordinator.py`, `worker_protocol.py`, `plan_runtime.py`, plus hooks in `ws_manager.py`, `message_handlers.py` and `main.py`.
- **DAG**: `submit_inference` builds the request DAG from the chosen Plan. Fixed-DAG submissions (`submit_pipeline_job`) take the unchanged legacy path, which is B0.
- **Routing**:
  - Relay: the Foreman forwards activations by **atom boundary**. The next stage is the one whose start atom is the producer's end atom + 1, so the routing logic survives re-cuts.
  - Direct: workers forward to the next hop, and the Foreman sees only the final output plus summaries.
- **Epochs**:
  - A **voluntary** switch **drains**: admission stops, in-flight requests finish through the old pipeline, then `load` is sent, then the switch commits.
  - A **forced** switch **resumes**: surviving changed workers finish their current stage, then every held or lost request resumes from its retained stage-boundary activation if the new plan keeps that cut, and otherwise restarts from its input.
  - The mode and the reason are logged on `switch_start`, and `resume` / `restart_from_input` per request.
  - The Foreman never sends `load` to a worker that still has in-flight work of the old epoch. Fake workers record any violation, and none occurred.
- **Recovery**: the existing checkpoint recovery depends on the Foreman holding activations (it stores task args in SQLite). The planned path doesn't use it. Its recovery is the retained boundary activations plus a per-request restart timeout.
- **Ack/resend/dedup**:
  - Workers retain sent activations until acked; an ack timeout raises `NEXT_HOP_DOWN`.
  - The Foreman falls back to the relay and sends `route`; the worker resends.
  - Duplicates are dropped at the receiver and at the Foreman, which completes each request exactly once.
- **Tests**:
  - `test_integration_fake_workers.py`:
    - initial plan;
    - a bandwidth drop with the replacement warm vs cold;
    - small N rejected;
    - worker loss with in-flight resume, and warm preferred over cold;
    - load failure;
    - a peer path that is refused, and one that silently loses traffic, both falling back cleanly;
    - duplicate delivery after fallback;
    - prefetch;
    - all flags off means no coordinator;
    - replay of the live log.
  - `test_live_websocket.py`: a real WebSocket round trip through the real `WebSocketManager`.

### 9. Problem handling

- **File**: `foreman/planning/problems.py`.
- **Behaviour**:
  - Per-worker token bucket, de-duplication per (type, stage, epoch), and a global storm cap.
  - Urgent problems act immediately: `NEXT_HOP_DOWN` → reroute; `OOM` / `LOAD_FAILED` → exclude (worker, shard) and replan.
  - Non-urgent problems feed the `problems` trigger.
- **Flag gating**: non-urgent problems are only counted when `problem_reports` is on. Urgent problems are always handled, because failure handling must not depend on a flag.
- **Tests**: dedupe, rate limit, urgency, storm cap.

### 10. Prefetch

- **File**: `foreman/planning/prefetch.py` (pure), plus `PlanCoordinator._prefetch`.
- **Candidates**: the planner's next-best plans and the current stages' backup workers.
- **Conditions**: respects the disk cap via LRU (never evicting the resident shard), battery, heat, link utilisation, and no live load.
- A live load cancels any prefetch on that worker.
- Bytes, `prefetch_used` and cancellations are logged.
- **Tests**: eligibility, eviction, and the integration prefetch test.

### 11. Worker protocol (Foreman side)

- **File**: `foreman/core/worker_protocol.py`.
- **Behaviour**:
  - Versioned envelope; every command carries a `msg_id`; timeout and retries keep the same id.
  - A failed `load` excludes (worker, shard) and triggers a replan. So does `LOAD_FAILED` or `OOM`.
  - Control-plane bytes and message counts are tracked for the no-fault tax.

### 12. Logging and experiment harness

- **Files**: `foreman/planning/eventlog.py`, `serialize.py`; `evaluation/reconfig/{sim,spec,scenario,metrics,replay,run}.py`; `configs/`.
- **JSONL events**: every event carries ts, run_id, seed, policy, flags and epoch. Each `planner_call` includes the inputs hash, the full snapshot, the cost options, the triggers and the controller config, so it can be replayed.
- **CLI**: `sim`, `compare` (all policies plus a clairvoyant reference, giving regret), `notax`, `metrics`, `replay`, `dp-gap`, `live`.
- **Scenario hooks**: in live mode they run an external command (for example `tc` on the access point); in the simulator they change ground truth.
- **Tests**: `test_harness.py`:
  - snapshot round trip;
  - simulation completes and replays identically;
  - the clairvoyant run is not beaten by any policy on the scenario;
  - the metrics fields;
  - a healthy run never switches;
  - scenario commands fire.

## Results on the example simulation (placeholder profiles, not measurements)

These come from `run compare` / `run notax` with `configs/spec_distilbert_5phones.yaml`: today's 3-cell export on 5 phones, 700 requests, p2's link dropping at t=120 s and p3 leaving at t=400 s.

| Policy | Total (s) | Regret (s) | Switches | Wasted | Shard MB moved |
|---|---|---|---|---|---|
| clairvoyant | 388.5 | 0 | 1 | 0 | 2.4 |
| B0 / B1 | 722.6 | 334.1 | 1 | 0 | 2.4 |
| B2 | 546.4 | 157.9 | 3 | 1 | 631.1 |
| B3 | 430.6 | 42.1 | 1 | 0 | 267.9 |
| B4 | 430.1 | 41.6 | 1 | 0 | 267.9 |
| B5a | 440.5 | 51.9 | 1 | 0 | 267.9 |
| B5b (async, unconstrained) | 397.0 | 8.5 | 1 | 0 | 267.9 |
| S-aware | 399.6 | 11.1 | 1 | 0 | 2.4 |

No-fault tax (healthy run, B0 with flags off vs everything on):
- Steady period is identical (0.551 s).
- Total time is 386.7 s vs 386.4 s.
- Control-plane bytes per request are about 1.16 kB vs 2.49 kB, from the simulator's message-size model.
- Worker CPU and battery, and Foreman CPU, need a live run (`run live --foreman-pid`).

The "clairvoyant" reference decides on ground truth with zero margin. It is a greedy reference, not a proven optimum.

## Assumptions the code did not confirm

1. The cold ~48 s and warm ~6 s reference numbers are only used as priors with `n=0`, which counts as stale. The Android phase fits in the example spec are placeholders.
2. `compression_ratio` defaults to 0.9 until measured. Today's code never measures it; the hook is `TelemetryStore.observe_compression`.
3. Compute time is scaled by `(1 + cpu)`. The reported CPU includes the stage's own work, so this over-penalises busy stages. Replace it once the mobile benchmark separates background load.
4. Direct worker-to-worker transport and peer file serving are worker features. The Foreman brokers them (`route`, `serve_allowed`, `peer_endpoint`), but the Android listener doesn't exist yet.
5. In direct mode, a drain waits until the worker reports `queue_len == 0` and is not `SERVING`, because intermediate activations bypass the Foreman.
6. Time-multiplexing (`allow_time_multiplexing`) is implemented in the cost model and exhaustive search but is off by default and hasn't been exercised end to end.
7. `pc_worker` still creates a new ONNX session per task, so PC workers don't satisfy single residency until they implement v2.

## Change: DNN work runs on mobile workers (2026-10-28)

Desktop-only DNN routing is removed. DNN tasks now run on mobile workers. PC workers are used only when testing that the partitioning is correct.

**Setting**: `CROWDIO_DNN_WORKERS` (or `dnn_workers` in the run config):
- `mobile`, the default: Android workers only;
- `pc`: PC workers only;
- `any`: both.

The same rule applies in both places:
- **Legacy dispatcher**: `foreman/core/dnn_placement.py`, `task_dispatcher._dnn_workers`.
- **Planner**: `foreman/planning/scope.py`. Workers outside the scope are marked ineligible in the snapshot, so no policy can place a shard on them.

If no worker in the scope is connected, DNN tasks wait. They never fall back to another kind of worker.

**Dashboard**: the panel is now the "DNN Placement Monitor", and the observability counters are `dnn_tasks_assigned` and `waiting_for_dnn_worker`.

**Tests**: `tests/test_task_dispatcher_routing.py` (rewritten), `tests/planning/test_dnn_scope.py`, and `test_live_websocket.py`, which now runs once with Android workers and once with PC workers under scope `pc`.

**Caveat**: `partition_model` tasks build TensorFlow partitions, which mobile workers can't run. Use them only with scope `pc`. For mobile runs, export ONNX partitions ahead of time.
