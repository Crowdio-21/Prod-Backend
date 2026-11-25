# CROWDio backend: run commands, step by step, for every scenario

All commands are for **Windows PowerShell**, run from the backend folder (`Prod-Backend`) unless a step says otherwise. Each scenario lists the terminals you need, in order, then how to check it worked and how to stop it.

Nothing in this file is run for you. Where a step has been checked, it says so; where it has not (for example anything on a real phone), it says that too.

## Contents

0. [Which scenario do I need?](#0-which-scenario-do-i-need)
1. [One-time setup (including rebuilding the broken `.venv`)](#1-one-time-setup)
2. [Check the code is healthy (tests)](#2-check-the-code-is-healthy)
3. [Scenario A: ordinary tasks on PC workers](#3-scenario-a-ordinary-tasks-on-pc-workers)
4. [Scenario B: choose a scheduler (including bandwidth-aware)](#4-scenario-b-choose-a-scheduler)
5. [Scenario C: checkpointing and recovery](#5-scenario-c-checkpointing-and-recovery)
6. [Scenario D: DistilBERT pipeline (fixed 3-stage split)](#6-scenario-d-distilbert-pipeline-fixed-3-stage-split)
7. [Scenario E: adaptive planning (switching placement and partition)](#7-scenario-e-adaptive-planning)
8. [Scenario F: all feature-flag cases](#8-scenario-f-all-feature-flag-cases)
9. [Scenario G: phones (Android) connecting to the Foreman](#9-scenario-g-phones-android)
10. [Scenario H: simulate 15 devices with changing bandwidth](#10-scenario-h-simulate-many-devices)
11. [Scenario I: planner and flag simulations (no Foreman)](#11-scenario-i-planner-and-flag-simulations)
12. [Scenario J: Docker](#12-scenario-j-docker)
13. [Watching, logging and resetting](#13-watching-logging-and-resetting)
14. [Environment variable reference](#14-environment-variable-reference)
15. [Troubleshooting](#15-troubleshooting)

---

## 0. Which scenario do I need?

| I want to... | Scenario | Needs |
|---|---|---|
| Check the code works | [2](#2-check-the-code-is-healthy) | Python only |
| Run ordinary jobs on PCs | [A](#3-scenario-a-ordinary-tasks-on-pc-workers) | Python only |
| Compare FIFO, round-robin, bandwidth-aware | [B](#4-scenario-b-choose-a-scheduler) | Python only |
| Test checkpointing / worker failure recovery | [C](#5-scenario-c-checkpointing-and-recovery) | Python only |
| Run the DistilBERT model across workers (fixed split) | [D](#6-scenario-d-distilbert-pipeline-fixed-3-stage-split) | Python; phones for the real target |
| Let the Foreman move shards when conditions change | [E](#7-scenario-e-adaptive-planning) | Phones (or the simulators) |
| Try each feature flag | [F](#8-scenario-f-all-feature-flag-cases) | Phones for real runs; tests/simulators otherwise |
| Connect Android phones | [G](#9-scenario-g-phones-android) | Phones on the same network |
| Test schedulers on 15 virtual devices | [H](#10-scenario-h-simulate-many-devices) | Python only |
| Study planner policies and flag costs | [I](#11-scenario-i-planner-and-flag-simulations) | Python only |
| Run in containers | [J](#12-scenario-j-docker) | Docker |

Two ports are used by the Foreman: **8000** (REST API and dashboard) and **9000** (WebSocket for workers and clients). Only one Foreman can run at a time, and the device simulator (`serve`/`run`/`compare`) also uses them, so stop one before starting the other.

---

## 1. One-time setup

### 1.1 Rebuild the virtual environment

If `python -c "import foreman.main"` fails with `ModuleNotFoundError` or `No module named pip`, the `.venv` is broken (it was missing its `Lib` folder at one point and has since been rebuilt). Rebuild it:

```powershell
cd C:\Users\DELL\Desktop\Path\CROWDio-Research\Prod-Backend
deactivate 2>$null
Remove-Item -Recurse -Force .venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Then install **one** of the two sets.

**Minimal** (Foreman, workers, tests, simulators; this is the set the tests and live runs in this project were checked with):

```powershell
python -m pip install websockets==12.0 cloudpickle fastapi==0.104.1 uvicorn==0.24.0 python-multipart "sqlalchemy>=2.0.30,<2.1" aiosqlite pydantic numpy pyyaml psutil requests httpx pytest onnxruntime h5py opencv-python-headless
```

**Full** (adds TensorFlow, PyTorch, transformers and the other sample workloads; needed to export model shards in Scenario D/E; several GB):

```powershell
python -m pip install -r requirements.txt
```

Either way, finish with the SDK from the sibling checkout. Do this last, because `requirements.txt` and a bare `pip install crowdio-sdk` can pull the published 2.0.1, which lacks `deserialize_tensor` and breaks the DNN pipeline:

```powershell
python -m pip install --force-reinstall --no-deps ..\CROWDio-SDK
python -m pip list | Select-String "crowdio|sqlalchemy"
```

Expected: `crowdio-sdk 0.4.0` (or newer, built from the local folder) and `SQLAlchemy 2.0.x` (2.0.30 or later). SQLAlchemy 2.0.23 does **not** import on Python 3.13.

> Do not run `pip install -r requirements.txt` after the SDK step without repeating the `--force-reinstall --no-deps ..\CROWDio-SDK` line.

### 1.2 Check the installation

```powershell
python -c "import foreman.main, pc_worker, crowdio, common; print('imports ok')"
```

### 1.3 Every new terminal

```powershell
cd C:\Users\DELL\Desktop\Path\CROWDio-Research\Prod-Backend
.\.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING = "utf-8"    # stops emoji in log output crashing the Windows console
```

Environment variables (`$env:NAME = ...`) apply only to the terminal where you set them. Set the Foreman's variables **in the Foreman's terminal, before you start it**.

---

## 2. Check the code is healthy

```powershell
python -m pytest tests -q
```

Expected with the minimal install: about 460 tests pass in 1 to 2 minutes (the number grows as tests are added). The tests use a temporary database and never touch `crowdio.db`.

Narrower runs:

```powershell
python -m pytest tests\planning -q                     # planner, controller, flags, fake-worker integration
python -m pytest tests\planning\test_flag_matrix.py -q # all 32 flag combinations
python -m pytest tests\planning\test_capabilities.py -q# worker capability gating
python -m pytest tests\planning\test_disk_space.py -q  # disk-space checks
python -m pytest tests\test_devsim_run.py -q           # device simulator against the real Foreman (about 45 s)
```

The Android side has its own tests (needs JDK 21 and the Android SDK; run from `Prod-Mobile`):

```powershell
cd ..\Prod-Mobile
.\gradlew :app:testDebugUnitTest --offline
```

---

## 3. Scenario A: ordinary tasks on PC workers

**Three terminals** (activate the venv in each, section 1.3).

**Terminal 1: the Foreman**

```powershell
python tests\run_foreman_simple.py
```

Dashboard: <http://localhost:8000>. API docs: <http://localhost:8000/docs>. Equivalent: `python -m uvicorn foreman.main:app --host 0.0.0.0 --port 8000`. The database is `crowdio.db` in this folder.

**Terminal 2: workers**

One worker (REST port 8005):

```powershell
python tests\run_worker_simple.py --port 8005
```

or five workers (REST ports 8001 to 8005):

```powershell
python tests\run_multiple_workers.py 5 --start-port 8001
```

or the production entry point, with options:

```powershell
python -m pc_worker.main --worker-id pc-1 --foreman-url ws://localhost:9000 --api-port 8001 --max-concurrent-tasks 1
```

**Terminal 3: a client**

```powershell
python tests\example_client.py localhost
```

**Check it worked**

```powershell
python tests\view_runs.py --last          # report of the latest job
Invoke-RestMethod http://localhost:8000/api/workers | Format-Table id, status
Invoke-RestMethod http://localhost:8000/api/stats
```

**Stop:** `Ctrl+C` in each terminal.

**Foreman on one machine, workers on others:** on each worker machine run `python -m pc_worker.main --foreman-url ws://<foreman-ip>:9000 --api-port 8001`. Allow TCP 8000 and 9000 through the Foreman machine's firewall. Use `--foreman-url`'s IP, not `localhost`. Set `$env:CROWDIO_FOREMAN_HTTP = "http://<foreman-ip>:8000"` for the worker's link probes.

---

## 4. Scenario B: choose a scheduler

The scheduler is chosen by `CROWDIO_SCHEDULER` in the **Foreman's** terminal. The default is `fifo`. Available: `fifo`, `round_robin`, `performance`, `least_loaded`, `priority`, `bandwidth_aware`.

```powershell
# Terminal 1
$env:CROWDIO_SCHEDULER = "bandwidth_aware"
python tests\run_foreman_simple.py
```

The Foreman prints which one is active:

```
[SCHEDULER] BandwidthAwareScheduler (CROWDIO_SCHEDULER=bandwidth_aware)
```

Other checks: the dashboard tile reads `BANDWIDTHAWARE`, and `Invoke-RestMethod http://localhost:8000/api/scheduling-info` reports it. Start the workers and client as in Scenario A.

Bandwidth-aware extras:

```powershell
$env:CROWDIO_SHARED_CAPACITY_MBPS = "50"    # all workers share one 50 Mbit/s access point
Invoke-RestMethod http://localhost:8000/api/links | ConvertTo-Json -Depth 5   # per-worker link estimates
```

PC workers measure their link to the Foreman only while idle. Until a worker reports, the scheduler assumes 10 Mbit/s for it.

To go back to FIFO: close the Foreman and start it in a terminal where `CROWDIO_SCHEDULER` is not set (`Remove-Item Env:CROWDIO_SCHEDULER`).

---

## 5. Scenario C: checkpointing and recovery

```powershell
# Terminal 1
python tests\run_foreman_simple.py
# Terminal 2
python tests\run_multiple_workers.py 3 --start-port 8001
# Terminal 3
python tests\example_checkpoint_client.py localhost
```

To test failure recovery, close one worker's window (or `Ctrl+C` it) while the job runs. Then:

```powershell
Invoke-RestMethod http://localhost:8000/api/checkpoints/recovery-events
python tests\view_checkpoints.py
```

Restart the worker with the same `--worker-id` to see it reconnect:

```powershell
python -m pc_worker.main --worker-id pc-1 --api-port 8001
```

---

## 6. Scenario D: DistilBERT pipeline (fixed 3-stage split)

This is the original pipeline: `cell_a`, `cell_b`, `cell_c`. DNN tasks go to **phones by default**; PCs are only for testing the partitioning.

### 6.1 Export the model parts (once; needs the Full install)

```powershell
python tests\dnn_partitioning\sentiment_onnx_pipeline\export_partitions.py
```

This writes `cell_a.onnx`, `cell_b.onnx`, `cell_c.onnx` and tokenizer files to `tools\onnx_partitions_sentiment\` (already present on this machine). Options: `--split-idx 3 --max-length 64 --opset 18 --skip-verify`.

The dataset is `data\sentences.csv` (already present; CSV with a `text` column). To make a fresh SST-2 sample: `python tests\dnn_partitioning\sentiment_onnx_pipeline\prepare_sst2_dataset.py --limit 700`.

### 6.2 D1: on PC workers (test the partitioning)

```powershell
# Terminal 1
$env:CROWDIO_DNN_WORKERS = "pc"
python tests\run_foreman_simple.py
# Terminal 2
python tests\run_multiple_workers.py 5 --start-port 8001
# Terminal 3: one sentence first
python tests\dnn_partitioning\sentiment_onnx_pipeline\single_input_test.py
# Terminal 3: then a batch
python tests\dnn_partitioning\sentiment_onnx_pipeline\multi_input_test.py --dataset data\sentences.csv --text-column text --limit 100
```

`--limit` is the number of sentences (default 700). Add `--host <foreman-ip> --port 9000` if the client is on another machine.

### 6.3 D2: on phones (the real target)

```powershell
# Terminal 1: no CROWDIO_DNN_WORKERS (the default is mobile)
python tests\run_foreman_simple.py
```

Connect the phones first (see [Scenario G](#9-scenario-g-phones-android)), wait until the dashboard shows them online, then:

```powershell
python tests\dnn_partitioning\sentiment_onnx_pipeline\multi_input_test.py --dataset data\sentences.csv --text-column text --limit 700
```

### 6.4 D3: phones and PCs together

```powershell
$env:CROWDIO_DNN_WORKERS = "any"
python tests\run_foreman_simple.py
```

### 6.5 Check the result

```powershell
python tests\view_runs.py --last
```

Shows which workers (phone or PC) ran each stage, timings, failures and link estimates. Per-sentence predictions are saved to `logs\runs\sentiment_<time>.csv`.

**If the job hangs with nothing running:** DNN tasks only go to the workers allowed by `CROWDIO_DNN_WORKERS`. With PC workers and the default (`mobile`) they receive nothing. Set the variable in the Foreman's terminal and restart it.

---

## 7. Scenario E: adaptive planning

Adaptive planning lets the Foreman choose where each model shard runs and where the model is cut, and move shards when bandwidth, load or battery change or a worker is lost. It is **off by default**.

Two things turn it on, and you need both for switching:

1. **A flag** (`adaptive_planning`) builds the coordinator.
2. **A policy** decides when it re-plans. With only the flag, the policy is `B0`, which re-plans **only after a failure**. For switching on bandwidth/load/heat changes you need `policy: S-aware` from a run config.

### 7.1 Export the shard set and manifest (once; Full install)

```powershell
python tests\dnn_partitioning\sentiment_onnx_pipeline\export_manifest.py
```

Writes 30 shards (about 1 GB) and `manifest.json` to `tools\onnx_shards_sentiment\`. Already present on this machine. For a coarser, smaller set: `--cut-grid 2` (two layers per shard).

### 7.1b BERT-base, cut to each phone's RAM

Use the SDK exporter for larger models. It writes every contiguous range of layer groups ("atoms"), plus an embedding shard, a head shard and the whole model, with a storage report, and it verifies that every combination reproduces the original model's output. Needs the Full install (PyTorch, transformers, onnx).

```powershell
python -m crowdio.export --model textattack/bert-base-uncased-SST-2 --model-id bert-base-sst2 --out tools --cut-grid 2 --dry-run   # report only
python -m crowdio.export --model textattack/bert-base-uncased-SST-2 --model-id bert-base-sst2 --out tools --cut-grid 2 --max-compositions 48
```

Result in `tools\bert-base-sst2\` (this machine already has the same export in `tools\bert_base_sst2\`): 24 shards, 3.5 GB (8.5x one copy): embedding 91 MiB, 21 encoder ranges of 54 to 216 MiB, head 2.3 MiB, whole model 418 MiB. `--cut-grid 1` gives finer cuts but about 10 GB; the default budget refuses it. Without `--dry-run` the export takes about 20 minutes on a CPU.

Serve it by pointing the Foreman at the folder (the Foreman reads both its own and the SDK's manifest format):

```powershell
$env:CROWDIO_MANIFEST   = "tools\bert_base_sst2"
$env:CROWDIO_RUN_CONFIG = "evaluation\reconfig\configs\run_example.yaml"   # set cost.seq_len to your token length
python tests\run_foreman_simple.py
```

**How the cut follows RAM.** Each phone reports its free RAM every few seconds. For every shard the Foreman needs about twice the shard's file size free while it loads (the load peak), less 15% headroom, because Android can take memory back:

| Free RAM on the phone | Largest shard it can take |
|---|---|
| about 100 MiB | the head only (2.3 MiB); too small for the embedding (182 MiB needed) |
| about 300 MiB | up to 2 atoms (108 MiB) |
| about 500 MiB | up to 3 atoms (162 MiB) |
| about 1 GiB or more | the whole model (418 MiB) |

The planner picks the cut and placement that is fastest for the job among those that fit. With plenty of RAM it still splits the model across phones, because a pipeline of several phones has higher throughput. If no phone can hold even the embedding, the plan is reported as infeasible rather than an invalid one being sent.

**When a phone runs low.** If a phone holding a shard reports under 128 MiB free for two readings in a row (`ram_pressure`), the Foreman re-plans at once, ignoring the usual cost margin: that shard is moved to another phone or the model is cut finer. If no plan fits, it keeps serving the current one. Tune with `cost.ram_headroom_frac` and `cost.ram_critical_bytes` in the run config.

**Fleet size.** The planner's `planner_mode` is `auto` by default: exact search for small fleets, an exact DP up to 7 phones, and a DP plus local search above that, so planning 12 phones takes seconds (the old exhaustive search took 24 s at 7 phones and minutes at 10). Set `planner_mode: exhaustive` in the run config to force the old behaviour.

Client inputs for BERT: each input is `{"tensors": {"input_ids": <int64 [1, seq]>, "attention_mask": <int64 [1, seq]>}}`, tokenised with the model's tokenizer (`tools\bert_base_sst2\tokenizer`).

### 7.2 Start the Foreman with adaptive planning

```powershell
# Terminal 1
$env:CROWDIO_RUN_CONFIG = "evaluation\reconfig\configs\run_example.yaml"   # policy: S-aware, flags as listed in the file
$env:CROWDIO_MANIFEST   = "tools\onnx_shards_sentiment\manifest.json"
python tests\run_foreman_simple.py
```

You should see:

```
[FLAGS] {'adaptive_planning': True, ...}
[PLAN LOG] events -> logs\runs\plan_<time>.jsonl
```

and `Invoke-RestMethod http://localhost:8000/api/plan/status` shows `"enabled": true` and `"policy": "S-aware"`.

To set the flags from the environment instead of the YAML (this **replaces** the YAML's flags, it does not merge):

```powershell
$env:CROWDIO_FLAGS = "adaptive_planning,problem_reports"
```

### 7.3 Connect phones, then submit planned inference

Connect phones as in [Scenario G](#9-scenario-g-phones-android). Each phone is sent the flags once after it connects. Then, in another terminal:

```powershell
python -m evaluation.reconfig.run live --config evaluation\reconfig\configs\run_example.yaml --host localhost --port 9000 --inputs 100
```

Options: `--inputs-file <json list of serialized tensors>`, `--model-id <id>`, `--foreman-pid <pid>` (records the Foreman's CPU), `--out <log path>`.

### 7.4 Watch what the planner does

```powershell
Invoke-RestMethod http://localhost:8000/api/plan/status | ConvertTo-Json -Depth 6
```

Shows the epoch, policy, flags, current plan (shard and worker per stage), counters and the last decision. The event log is a JSONL file under `logs\runs\plan_<time>.jsonl`:

```powershell
python -m evaluation.reconfig.run metrics logs\runs\plan_<time>.jsonl      # switches, stall, bytes moved
python -m evaluation.reconfig.run replay  logs\runs\plan_<time>.jsonl      # re-runs every logged decision and checks they match
```

### 7.5 What triggers a switch

| Event | What the Foreman does |
|---|---|
| A phone in the plan disconnects | Forced re-plan at once, resuming in-flight work from saved activations |
| A shard fails to load | That worker is excluded for that shard; forced re-plan |
| Bandwidth, load or heat drifts (policy `S-aware`) | Switches only if the expected gain beats the switching cost plus an uncertainty margin, after the change persists and the cooldown has passed |
| Estimates are stale | Non-forced switches are refused |

### 7.6 Policies

Set `policy:` in the run config: `B0` (fixed, reacts to failures only), `B1`, `B2` (greedy), `B3`, `B4`, `B5a`, `B5b`, `S-aware-placement`, `S-aware`. `S-aware` is the one that prices the switch. Compare them without hardware in [Scenario I](#11-scenario-i-planner-and-flag-simulations).

### 7.7 Mixed PCs and phones

A worker that does not declare protocol v2 (every PC worker) is **not planned onto and is sent no v2 message**; it keeps running ordinary tasks. So PC workers may stay connected while adaptive planning is on. This is covered by tests (`tests\planning\test_capabilities.py`); it has not been run against a real PC worker process.

To go back to the legacy path: close the Foreman and start it in a terminal where `CROWDIO_RUN_CONFIG`, `CROWDIO_FLAGS` and `CROWDIO_MANIFEST` are not set.

---

## 8. Scenario F: all feature-flag cases

There are five flags. They are listed under `flags:` in the run config, or in `CROWDIO_FLAGS`. Only `adaptive_planning` is the master switch: every other flag is ignored (with a warning) without it.

Start every case the same way, changing only the flags line:

```powershell
$env:CROWDIO_RUN_CONFIG = "evaluation\reconfig\configs\run_example.yaml"   # policy S-aware
$env:CROWDIO_MANIFEST   = "tools\onnx_shards_sentiment\manifest.json"
$env:CROWDIO_FLAGS      = "<the flags for the case>"
python tests\run_foreman_simple.py
```

| # | `CROWDIO_FLAGS` | Behaviour | Phone support today |
|---|---|---|---|
| 1 | *(unset, or any flags without `adaptive_planning`)* | Legacy path, no coordinator | n/a |
| 2 | `adaptive_planning` | Planned inference, relay through the Foreman | Yes |
| 3 | `adaptive_planning,problem_reports` | Case 2, plus slow-stage reports can trigger a re-plan | Yes |
| 4 | `adaptive_planning,prefetch` | Case 2, plus standby phones pre-download likely shards | Yes (new build) |
| 5 | `adaptive_planning,peer_fetch` | Case 2, plus shards can come from donor phones | Yes (new build) |
| 6 | `adaptive_planning,prefetch,peer_fetch` | Prefetch also sources from donors | Yes (new build) |
| 7 | `adaptive_planning,direct_path` | Phone-to-phone activations when the peer link is better | Yes (new build) |
| 8 | `adaptive_planning,direct_path,peer_fetch` | Direct activations and donor fetches | Yes (new build) |
| 9 | `adaptive_planning,direct_path,peer_fetch,prefetch,problem_reports` | Everything on | Yes (new build) |

"New build" means the Android app with the peer service and capability declaration added in this change. A phone declares which flags it supports when it connects; the Foreman never uses a flag with a phone that did not declare it (an old app build keeps working with cases 1 to 3).

JSON form of the same thing: `$env:CROWDIO_FLAGS = '{"adaptive_planning": true, "prefetch": true}'`. A misspelt flag name stops the Foreman at startup.

**Verify a case is active**

```powershell
Invoke-RestMethod http://localhost:8000/api/plan/status | Select-Object -ExpandProperty flags
Select-String -Path logs\runs\plan_*.jsonl -Pattern '"event": "(prefetch|route_flip|peer_fetch|flags_sent)"' | Select-Object -Last 10
```

You look for: `flags_sent` (every phone is told the flags), `prefetch` events (case 4/6/9), `peer_fetch` events with a donor (case 5/6/8/9), and `route_flip` to `direct` (case 7/8/9).

**Why a flag may seem to do nothing**
- `direct_path`: the Foreman chooses it only when the measured phone-to-phone link beats relaying through the Foreman by the configured margin, and only after the phones have tested their links (a few seconds after connecting). The first job often relays; later jobs go direct.
- `peer_fetch`: a donor is used only if some other phone already holds the shard and is faster than the Foreman for that download.
- `prefetch`: only the standby phones chosen when a plan is committed are prefetched; a phone that joins later is not a standby until the next plan.

**Run all 32 flag combinations against fake workers, no phones needed**

```powershell
python -m pytest tests\planning\test_flag_matrix.py -q
```

---

## 9. Scenario G: phones (Android)

1. Put the PC running the Foreman and the phones on the **same Wi-Fi network**.
2. Find the PC's address: `ipconfig` (use the IPv4 address of the Wi-Fi adapter).
3. Allow the ports through the Windows firewall (administrator PowerShell, once):

```powershell
New-NetFirewallRule -DisplayName "CROWDio Foreman" -Direction Inbound -Protocol TCP -LocalPort 8000,9000 -Action Allow
```

4. Start the Foreman ([Scenario D2](#63-d2-on-phones-the-real-target) or [E](#72-start-the-foreman-with-adaptive-planning)).
5. In the CROWDio app on each phone, enter the Foreman's address: WebSocket `ws://<PC address>:9000`, REST `http://<PC address>:8000`. *(Where the app takes this address is in the Android project, not this one; not checked here.)*
6. Wait until the dashboard (<http://localhost:8000>) lists each phone as online and as an Android worker.
7. Phones also need the shard route and probe endpoint, which the Foreman serves: `/api/shards/manifest`, `/api/shards/{id}`, `/api/probe`. Set the Foreman's public address so phones can reach the probe endpoint:

```powershell
$env:CROWDIO_PUBLIC_URL = "http://<PC address>:8000"    # in the Foreman's terminal, before starting it
```

8. Phone-to-phone features (`direct_path`, `peer_fetch`) open a small server on each phone, on a random port, on the Wi-Fi address. Phones must be on the same network with no client isolation ("AP isolation") on the router.

What has and has not been checked: the Android runtime was run against a real Foreman process over real WebSockets and HTTP in a JVM test (flags delivered, peer addresses reported, peer links measured, prefetch, and an entire job passed directly phone-to-phone with correct results). It was **not** run on a physical phone.

---

## 10. Scenario H: simulate many devices

The device simulator runs the **real Foreman** against virtual phones whose bandwidth changes. It uses its own throwaway database and needs ports 8000 and 9000 free (stop any other Foreman first).

```powershell
python -m evaluation.devsim scenarios                     # list scenarios and workloads
python -m evaluation.devsim trace --scenario drop --seconds 120   # preview the bandwidth, nothing runs
python -m evaluation.devsim run --scenario drop --scheduler bandwidth_aware --devices 15
python -m evaluation.devsim compare --scenario drop --schedulers fifo,round_robin,bandwidth_aware --repeats 3
python -m evaluation.devsim serve --scenario congestion   # real Foreman plus 15 virtual devices; point your own client at it
```

Scenarios: `stable`, `congestion`, `drift`, `drop`, `commute`, `churn`, `shared-wifi`. Workloads (`--workload`): `mixed` (default), `uniform`, `dnn3`, `stream`. Useful options:

| Option | Meaning |
|---|---|
| `--devices 15` | Fleet size |
| `--pc 2` | How many of them are PCs |
| `--seed 1` | Same seed gives every scheduler the same bandwidth |
| `--speed 1` | Keep at 1 to 2; the Foreman is not sped up |
| `--tasks N`, `--work-scale X` | Job size and compute time |
| `--shared-capacity-mbps 100` | One shared access point |
| `--mode real` | Really run the task function (speed 1, light tasks only) |
| `--out devsim_results\name` | Output folder (`summary.json`, `devices.csv`, `bandwidth_trace.csv`, `events.jsonl`, `run_logs\`) |

A full comparison across all seven scenarios takes about 30 minutes (one run per scheduler each). The simulated devices speak only the legacy protocol, so this tool compares **schedulers**; it cannot exercise adaptive planning or the flags.

Fast scheduler model without the Foreman:

```powershell
python -m evaluation.scheduler_sim
```

---

## 11. Scenario I: planner and flag simulations

These run the same planner and controller as the Foreman against simulated phones and a scripted scenario. No Foreman is needed.

```powershell
$C = "evaluation\reconfig\configs\run_example.yaml"
$S = "evaluation\reconfig\configs\spec_distilbert_5phones.yaml"

python -m evaluation.reconfig.run compare --config $C --spec $S          # every policy plus a clairvoyant reference
python -m evaluation.reconfig.run compare --config $C --spec $S --policies B0,B2,S-aware --outdir reports\cmp
python -m evaluation.reconfig.run sim     --config $C --spec $S --out logs\sim.jsonl --with-oracle
python -m evaluation.reconfig.run notax   --config $C --spec $S          # cost of the flags when nothing goes wrong
python -m evaluation.reconfig.run dp-gap  --instances 200 --workers 5 --seed 0   # planner accuracy vs exhaustive search
python -m evaluation.reconfig.run metrics logs\sim.jsonl
python -m evaluation.reconfig.run replay  logs\sim.jsonl
```

To simulate one flag case, edit `flags:` in a copy of `run_example.yaml`, or set `$env:CROWDIO_FLAGS`, and run `sim` or `compare`. Scenario events (`network`, `load`, `drop_worker`, `join_worker`, `command`) are set under the scenario in the run config. The phone timings in `spec_distilbert_5phones.yaml` are **placeholders**: replace them with measured values before drawing conclusions.

---

## 12. Scenario J: Docker

```powershell
docker compose up --build -d              # one Foreman + one worker
docker compose up --scale worker=4 -d     # four workers
docker compose logs -f foreman
docker compose down
```

Known issues before this works (see the README, "Known Gaps"): `Dockerfile.foreman` and `Dockerfile.worker` copy a `common/` folder that is not in this repo, `CROWDIO_DATA_DIR=/data` is not read by the code (the database is written to `/app/crowdio.db`, so the volume does not persist it), and the worker health check calls `/health`, which the worker does not expose. The Foreman image does not include `tools\`, so adaptive planning in Docker needs `CROWDIO_MANIFEST` pointing at a mounted manifest.

---

## 13. Watching, logging and resetting

### 13.1 REST endpoints (Foreman, port 8000)

| Endpoint | What it shows |
|---|---|
| `GET /api/stats` | Job and worker counts |
| `GET /api/workers` | Workers, status, device specs, latest metrics |
| `GET /api/jobs`, `/api/jobs/{id}` | Jobs and their tasks |
| `GET /api/runs`, `/api/runs/{job_id}?format=text` | Run reports |
| `GET /api/links` | Bandwidth-aware scheduler's link estimates |
| `GET /api/scheduling-info` | Active scheduler and per-worker scheduling data |
| `GET /api/plan/status` | Adaptive planning: epoch, policy, flags, plan, counters |
| `GET /api/dnn/observability` | DNN task placement |
| `GET /api/pipeline/jobs`, `/api/pipeline/job/{id}` | Pipeline jobs |
| `GET /api/checkpoints/job/{id}`, `/api/checkpoints/recovery-events` | Checkpoints and recoveries |
| `GET /api/shards/manifest`, `/api/shards/{id}` | What phones download |
| `GET /api/probe/{size}`, `POST /api/probe` | Bandwidth probe download and upload |

Example: `Invoke-RestMethod http://localhost:8000/api/workers | Format-Table`.

### 13.2 Run logs

Written to `logs\runs\` (change with `$env:CROWDIO_RUN_LOG_DIR`):

```powershell
python tests\view_runs.py                 # table of the latest runs
python tests\view_runs.py --last          # full report of the latest run
python tests\view_runs.py --job 3f2a      # one run, by the start of its job id
```

### 13.3 Database tools

```powershell
python tests\view_database.py
python tests\check_db_schema.py
python tests\quick_clear_db.py            # clears data from crowdio.db
```

### 13.4 Reset between scenarios

```powershell
# stop the Foreman and workers (Ctrl+C), then:
Remove-Item crowdio.db, crowdio.db-wal, crowdio.db-shm -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .payload_store, .checkpoints -ErrorAction SilentlyContinue
Get-ChildItem Env: | Where-Object Name -like "CROWDIO_*"        # see what is set
Get-ChildItem Env: | Where-Object Name -like "CROWDIO_*" | ForEach-Object { Remove-Item "Env:$($_.Name)" }
```

Always restart the Foreman between scenarios: it keeps the plan, epoch and worker state in memory.

---

## 14. Environment variable reference

Set in the **Foreman's** terminal before it starts, unless noted.

| Variable | Default | Purpose |
|---|---|---|
| `CROWDIO_SCHEDULER` | `fifo` | `fifo`, `round_robin`, `performance`, `least_loaded`, `priority`, `bandwidth_aware` |
| `CROWDIO_SHARED_CAPACITY_MBPS` | unset | One shared medium for the bandwidth-aware scheduler |
| `CROWDIO_DNN_WORKERS` | `mobile` | Which workers may run DNN tasks: `mobile`, `pc`, `any` |
| `CROWDIO_FLAGS` | unset | Feature flags (comma list or JSON); replaces the run config's flags |
| `CROWDIO_RUN_CONFIG` | unset | YAML run config (policy, flags, thresholds). Needed for `policy: S-aware` |
| `CROWDIO_MANIFEST` | `tools\onnx_partitions_sentiment` | Shard manifest for adaptive planning |
| `CROWDIO_PUBLIC_URL` | `http://localhost:8000` | Address phones use for the probe endpoint |
| `CROWDIO_WS_PORT` | `9000` | WebSocket port for workers and clients. The REST port is uvicorn's `--port` (`python -m uvicorn foreman.main:app --port 8001`). Use both to run a second Foreman beside one on 8000/9000 |
| `CROWDIO_RUN_LOG_DIR` | `logs\runs` | Run logs and planner event logs |
| `CROWDIO_DATABASE_URL` | `sqlite+aiosqlite:///./crowdio.db` | Database location |
| `CROWDIO_MODEL_DIR` | `tools\onnx_partitions_sentiment` | Served by `/api/models/{name}` |
| `CROWDIO_PING_INTERVAL` | `30` | Seconds between Foreman pings to workers |
| `CROWDIO_FOREMAN_HTTP` | unset | **Worker's terminal**: Foreman REST address for link probes |
| `PYTHONIOENCODING` | unset | Set to `utf-8` in every terminal on Windows |

---

## 15. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` for everything, or `No module named pip` | The `.venv` is broken (no `Lib` folder) | Section 1.1 |
| `AssertionError: ... SQLCoreOperations ... TypingOnly` | SQLAlchemy 2.0.23 on Python 3.13 | `python -m pip install "sqlalchemy>=2.0.30,<2.1"` |
| `cannot import name 'deserialize_tensor' from 'common.serializer'` | Published SDK 2.0.1 installed instead of the local one | `python -m pip install --force-reinstall --no-deps ..\CROWDio-SDK` |
| `Errno 10048` / port already in use (8000 or 9000) | An earlier Foreman or simulator is still running | Find it: `Get-NetTCPConnection -LocalPort 8000,9000 \| Select-Object LocalPort, OwningProcess`, then close that terminal or `Stop-Process -Id <pid>` |
| Output is `FIFOScheduler` though you set `CROWDIO_SCHEDULER` | Variable set in a different terminal than the Foreman's | Set it in the Foreman's terminal, restart |
| Planned inference never switches on bandwidth changes | Policy is `B0` (flags only) | Add `CROWDIO_RUN_CONFIG` with `policy: S-aware` |
| DNN job hangs with no worker running it | Workers are outside `CROWDIO_DNN_WORKERS` | Match the worker type (`mobile` for phones, `pc` for PC workers, or `any`) |
| Phone connects but is classified as a PC | Old Foreman build | Use this version; the Foreman honours the declared `worker_type` |
| Phone does not reach the Foreman | Firewall, or different network | Section 9, steps 1 to 3; `Test-NetConnection <PC address> -Port 9000` from another machine |
| `direct_path` or `peer_fetch` never used | Links not yet tested, or the Foreman route is cheaper | See the notes under Scenario F; wait a few seconds after connecting and submit a second job |
| `flags` not delivered / problem reports missing | Old Foreman that never sent the `flags` message | Use this version; check `flags_sent` events in `plan_*.jsonl` |
| Emoji / arrow characters crash the console | Windows code page | `$env:PYTHONIOENCODING = "utf-8"` |
| `UnicodeEncodeError` log spam from the FIFO scheduler during `scheduler_sim` | Box-drawing characters on a Windows console | Harmless; results are unaffected |
| Tests write to `crowdio.db` | They should not | Report it: `python -m pytest tests\test_db_isolation.py -q` guards this |
