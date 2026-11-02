# Device simulator

`evaluation/devsim/` runs the **real Foreman** against a fleet of simulated devices, for example 15 phones, whose
bandwidth changes over time. It is for testing schedulers and workloads without 15 real devices.

## 1. Quick start

```powershell
# What can I simulate?
python -m evaluation.devsim scenarios

# Preview how each device's bandwidth moves in a scenario (nothing is run)
python -m evaluation.devsim trace --scenario drop --seconds 120

# Run one scheduler through a scenario with 15 devices
python -m evaluation.devsim run --scenario drop --scheduler bandwidth_aware --workload mixed --devices 15

# Same fleet, same bandwidth for each scheduler, compare them
python -m evaluation.devsim compare --scenario drop --schedulers fifo,round_robin,bandwidth_aware --repeats 3

# The real Foreman (dashboard :8000, WebSocket :9000) with 15 simulated devices attached,
# so you can run your own test scripts against it. Needs ports 8000 and 9000 free.
python -m evaluation.devsim serve --scenario congestion
```

Results of `run` and `compare` go to `devsim_results/<name>/`: `summary.json`, `devices.csv` (per-device usage),
`bandwidth_trace.csv` (each device's bandwidth every second), `events.jsonl`, and `run_logs/` (the Foreman's own
run report, see README section 7).

## 2. How it works

```text
  your workload (real SDK client) --WebSocket--> REAL Foreman (dispatcher, scheduler, DB, run log)
                                                       |  WebSocket, real protocol
                                                       v
                    15 virtual devices  <--- link engine: every message waits for its transfer time
                    (register as Android phones)        at that device's CURRENT bandwidth
```

| Real | Simulated |
|---|---|
| The Foreman: dispatcher, every scheduler, payload store, run log, REST API, dashboard | The devices' CPUs (a task takes `work x cpu_factor` seconds) |
| The WebSocket protocol: registration, heartbeats, PING replies, task assignment, results | The network: transfer time, round trip, outages, shared access point |
| The SDK client that submits the jobs | Battery, RAM, CPU load numbers |
| Task functions, when you use `--mode real` | The model files (a "download" over the device's link) |

A simulated phone registers exactly as the Android app does (`platform: android`, `runtime: jvm`,
`worker_type: android_kotlin`) and sends the same heartbeat shape, so the Foreman treats it as a phone.

**The link engine** (`link.py`) turns a bandwidth trace into delays. A transfer is a flow with bytes remaining; every
small step the flow moves `rate x elapsed` bytes, where the rate is the device's nominal rate times the scenario's
bandwidth factor *at that moment*. So a drop in the middle of a transfer slows it from that moment, and an outage
stalls it until the link recovers. A device's own concurrent transfers share its link, and if the scenario has a
shared medium (one access point), all devices split its capacity with a max-min fair share. Tests check these against
hand-computed times (a mid-transfer drop, an outage, three devices sharing 12 Mbit/s).

**What the scheduler can see.** Each device reports its measured link in its heartbeat, as the real app does: the true
current rate with 8% noise, refreshed every 10 simulated seconds. The measurement does not use any bandwidth.

## 3. The fleet

`make_fleet(n, seed, pc)`: with 15 devices, 3 flagship, 6 mid-range and 6 low-end phones. Each device's numbers are
scaled by a seeded random factor (about 15%), so no two are identical.

| Tier | CPU (1.0 = reference) | Down / up Mbit/s | Round trip |
|---|---|---|---|
| flagship | 1.0 | 40 / 15 | 15 ms |
| mid | 1.6 | 15 / 6 | 25 ms |
| low | 3.0 | 5 / 2 | 45 ms |
| pc (`--pc N`) | 0.6 | 200 / 100 | 3 ms |

## 4. Bandwidth scenarios

Bandwidth is a *factor* applied to both directions (1.0 = nominal), generated from a seed. The same
`(scenario, fleet, seed)` gives every scheduler the same bandwidth over time, which is what makes a comparison fair.

| Scenario | What happens |
|---|---|
| `stable` | Nominal bandwidth throughout |
| `congestion` | Each link flips between good and congested (15% of nominal) on its own schedule: about 8 s in 40 congested |
| `drift` | Bandwidth drifts up and down around nominal (mean-reverting random walk) |
| `drop` | The fastest third of the fleet falls to 8% bandwidth from 30 s to 100 s |
| `commute` | Drifting links; a third of the devices also pass through 2 to 6 s outages |
| `churn` | Stable links; three devices leave around 25 s and two come back around 70 s |
| `shared-wifi` | Congested links, and all devices share one 100 Mbit/s access point |

The building blocks (`bandwidth.py`) are `Stable`, `RandomWalk`, `Markov`, `Step`, `Outage`, `RandomOutages` and
`Compose`; a new scenario is a few lines in `scenarios.py`.

## 5. Workloads

All go through the real SDK. Tasks are deliberately heavy (seconds of work); see section 6.

| Workload | Job |
|---|---|
| `mixed` (default) | 150 tasks, payloads 20 KB to 5 MB, 1 to 3 s of work |
| `uniform` | 120 identical tasks: 500 KB in, 50 KB out, 2 s of work |
| `dnn3` | A 3-stage pipeline like the DistilBERT sentiment test, 40 inputs: three model files that each device downloads once over its own link, about 197 KB between stages. `--model-scale` (default 0.05) shrinks the model files. |
| `stream` | 10 jobs of 12 tasks, one every 8 s, so bandwidth changes between jobs |

`--tasks N` changes the size; `--work-scale X` multiplies every task's compute time.

## 6. Speed, and what limits fidelity

`--speed S` makes simulated time pass S times faster than real time. **The default is 1, and values above about 2 are
unreliable**, because the Foreman is not simulated and runs in real time. Measured here, with 15 devices:

- The Foreman dispatches about **6 to 7 tasks per second** (each completion triggers database queries for every worker).
- A 60-task job of 1-second tasks finished in 10.5 s at speed 1, and in **9.1 real seconds at speed 5**. At speed 5 that
  reads as 45.6 simulated seconds: the same real time, because the Foreman was the bottleneck, not the devices.

So a run prints a warning when devices were idle most of the time at speed above 1. Fast-forwarding helps only with
tasks that are long compared with the Foreman's own work, and the default workloads are heavy for that reason.

## 7. Modes

- `--mode model` (default): a task takes `work_s x cpu_factor` simulated seconds and returns a result of the requested size.
  Nothing really runs, so 15 devices never load 15 copies of a model on your machine.
- `--mode real`: the task function really runs (so results are real) and is slowed by `cpu_factor`. Needs `--speed 1`
  and light tasks; 15 heavy models in one process will not work.

Simulated runs always use a throwaway database in the output folder, never `crowdio.db`.

## 8. Findings made while building it

- **The Foreman classified real Android phones as PC workers.** The app registers with `runtime: "jvm"` but the Foreman
  only recognised `"kotlin"` and `"chaquopy"`, so every phone was filed under `PC_PYTHON`, and the "DNN tasks run on mobile
  workers" rule would never have offered a phone any work. Fixed in `foreman/core/connection_manager.py`
  (`resolve_worker_type`); the simulator registers like the app, which is how it showed up.
- **The `dnn3` workload was not treated as DNN work.** The Foreman decides a task is DNN work (mobile workers only) by looking for `onnxruntime` in the task's source. The simulator's stand-in stage did not contain it, so PCs were given pipeline stages. The pipeline now ships `sim_dnn_stage`, whose source names onnxruntime.
- **Foreman dispatch capacity** is about 6 to 7 tasks per second at 15 workers (section 6).

## 9. Results

Scenario `drop`, 15 devices, `mixed` workload (150 tasks), speed 1, 2 runs per scheduler (seeds 1 and 2) with identical bandwidth for every scheduler. All runs completed.

| Scheduler | Makespan (s) | vs fifo | Tasks/s | Task p95 (s) |
|---|---|---|---|---|
| fifo | 64.61 | | 2.325 | 7.62 |
| round_robin | 63.31 | 2.0% faster | 2.380 | 7.34 |
| bandwidth_aware | 60.86 | 5.8% faster | 2.467 | 7.58 |

How to read this: the order is the expected one, but the differences are small and there are only two runs each, so treat them as indicative, not significant (FIFO alone varies run to run). The gap is modest because the fleet is not the bottleneck for long: the default `mixed` tasks are mostly small, and the Foreman dispatches only about 6 to 7 tasks per second. A test with heavier payloads (`--work-scale`, `uniform`) or more repeats (`--repeats 5`) will separate the schedulers better. A separate test (`tests/test_devsim_run.py`) shows the direct effect: with the two fastest links collapsed from the start, the bandwidth-aware scheduler sent them under a quarter of the data volume the healthy phones received (1.3 MB against 27 MB), mainly by giving them only small tasks.

## 10. Limitations

- Everything runs on one machine, in one process, so the devices and the Foreman share a CPU. With heavy `--mode real`
  tasks this matters; in `model` mode it does not.
- The network model is a fluid approximation: no packet loss, no TCP dynamics, no slow start.
- The link measurement a device reports is the true rate with noise, not an actual probe, so probe overhead is not simulated.
- Battery, RAM and CPU numbers are plausible, not measured.
- A shared medium is one pool per direction; real Wi-Fi shares airtime between uplink and downlink.
- The bandwidth numbers are made up to resemble phones; none were measured on real devices.
