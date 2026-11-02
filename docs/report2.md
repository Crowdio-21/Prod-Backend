# Report 2: how to run CROWDio end to end (as of 2026-11-02)

This is a run book for the code in this working tree (branch `feature/reconfiguration-aware-planning`, with uncommitted changes on top of commit `06e75cf`). Report 1 explains how the system works; this one explains how to run it. Every command was checked against the code, and the parts I could not test are marked **unverified**.

## 0. Pick a run

| I want to... | Go to | Needs |
|---|---|---|
| Check that the code is healthy | section 2 | Python only |
| Run ordinary jobs on PC workers | section 3 | Python only |
| Run the DistilBERT pipeline on phones (the real target) | section 4 | Phones with the Android app, same network |
| Check the partitioning logic using PCs instead of phones | section 5 | Python only |
| Compare schedulers on 15 simulated devices with changing bandwidth | section 6 | Python only |
| Study the reconfiguration planner and flags in simulation | section 7 | Python only |
| Turn on adaptive planning for real | section 8 | Phones; read the warnings |

## 1. One-time setup

Run everything from the project root, with the virtual environment active. A `.venv` already exists here.

```powershell
.\.venv\Scripts\Activate.ps1
```

**The environment is already set up** (SQLAlchemy 2.0.52, SDK 0.4.0 from `..\CROWDio-SDK`). `requirements.txt` is now safe to re-run: it no longer pins the versions that broke it. A fresh environment needs:

```powershell
python -m pip install -r requirements.txt
python -m pip install ..\CROWDio-SDK
python -m pip install pytest httpx
```

The working set here is SQLAlchemy 2.0.52 and the SDK installed from `..\CROWDio-SDK` (version 0.4.0).

For the sentiment pipeline you also need the exported model shards. They are already in `tools/onnx_partitions_sentiment/` (`cell_a.onnx`, `cell_b.onnx`, `cell_c.onnx`). To regenerate them, and to create the shard set and `manifest.json` that adaptive planning needs (the second command writes to `tools\onnx_shards_sentiment\`, 30 shards, about 1 GB):

```powershell
python tests\dnn_partitioning\sentiment_onnx_pipeline\export_partitions.py
python tests\dnn_partitioning\sentiment_onnx_pipeline\export_manifest.py
```

If a port is already in use (error 10048 on port 8000 or 9000), a previous Foreman is still running. Stop it before starting another.

## 2. Check the code (about 70 s)

```powershell
python -m pytest tests -q
```

Expected today: 356 passed. The tests use a temporary database, never `crowdio.db`.

## 3. Ordinary jobs on PC workers

Use three terminals.

```powershell
# Terminal 1: the Foreman (REST/dashboard :8000, WebSocket :9000)
python tests\run_foreman_simple.py

# Terminal 2: workers (5 workers, API ports 8001-8005)
python tests\run_multiple_workers.py 5 --start-port 8001

# Terminal 3: a sample client
python tests\example_client.py localhost
```

Open `http://localhost:8000` for the dashboard. After the job, read what happened:

```powershell
python tests\view_runs.py --last
```

This prints the duration, tasks per stage, which workers ran what, failures and the link estimates. The files are in `logs\runs\`.

Choose a scheduler with an environment variable set before starting the Foreman (default `fifo`):

```powershell
$env:CROWDIO_SCHEDULER = "bandwidth_aware"     # or fifo, round_robin
```

PC workers measure their link to the Foreman automatically while idle, which the bandwidth-aware scheduler uses.

## 4. DistilBERT pipeline on phones (the intended setup)

DNN tasks run on phones by default (`CROWDIO_DNN_WORKERS=mobile`). With only PC workers connected, a DNN job would find no eligible worker.

1. Put the PC running the Foreman and the phones on the same network. Allow inbound TCP 8000 and 9000 in the Windows firewall.
2. Find the PC's address with `ipconfig`.
3. Start the Foreman: `python tests\run_foreman_simple.py`.
4. On each phone, open the CROWDio app and point it at `ws://<PC address>:9000`. **Unverified:** where the app takes this address; it is in the Android repository, not this one.
5. Wait until the dashboard (`http://localhost:8000`) lists the phones as online. They must show as Android workers; this classification was fixed in this tree (`resolve_worker_type`).
6. Prepare a dataset (CSV with a `text` column, or TXT with one sentence per line), then run:

```powershell
python tests\dnn_partitioning\sentiment_onnx_pipeline\multi_input_test.py --dataset data\sentences.csv --text-column text --limit 700
```

Add `--host <address> --port 9000` if the client runs on another machine. A single-sentence smoke test is `single_input_test.py`.

Results: predictions are saved to `logs\runs\sentiment_<time>.csv`, and the run report is under `logs\runs\`. In the pipeline, each input passes through `cell_a`, `cell_b`, `cell_c`, and results come back in input order.

## 5. Same pipeline on PC workers (partition testing only)

```powershell
$env:CROWDIO_DNN_WORKERS = "pc"
python tests\run_foreman_simple.py
```

In a second terminal: `python tests\run_multiple_workers.py 5 --start-port 8001`. In a third, run the `multi_input_test.py` command from section 4. Use `any` instead of `pc` to allow both kinds. Unset the variable afterwards to return to the phone-only default.

## 6. Simulate 15 devices with changing bandwidth

This runs the real Foreman against simulated phones. It uses its own database and does not need the Foreman running. Ports 8000 and 9000 must be free.

```powershell
python -m evaluation.devsim scenarios                 # what exists
python -m evaluation.devsim trace --scenario drop     # preview the bandwidth, nothing runs
python -m evaluation.devsim run --scenario drop --scheduler bandwidth_aware --devices 15
python -m evaluation.devsim compare --scenario drop --schedulers fifo,round_robin,bandwidth_aware --repeats 3
```

A 15-device run takes roughly a minute. Results go to `devsim_results\<name>\`. Keep `--speed` at 1 to 2. To run your own client against the simulated fleet:

```powershell
python -m evaluation.devsim serve --scenario congestion
# in another terminal: any client, e.g. python tests\example_client.py localhost
```

Scenarios: `stable`, `congestion`, `drift`, `drop`, `commute`, `churn`, `shared-wifi`. Workloads: `mixed`, `uniform`, `dnn3`, `stream`. Last measured (drop, 2 runs each): FIFO 64.6 s, round-robin 63.3 s, bandwidth-aware 60.9 s; indicative only.

Limit: simulated devices speak the old protocol, so this cannot exercise adaptive planning.

## 7. Study the planner and flags in simulation

```powershell
python -m evaluation.reconfig.run compare --config evaluation\reconfig\configs\run_example.yaml --spec evaluation\reconfig\configs\spec_distilbert_5phones.yaml
python -m evaluation.reconfig.run notax --config evaluation\reconfig\configs\run_example.yaml --spec evaluation\reconfig\configs\spec_distilbert_5phones.yaml
python -m evaluation.scheduler_sim
```

`compare` runs every policy against a clairvoyant reference. `notax` shows what the flags cost when nothing goes wrong. The phone timings in the spec are placeholders: replace them with measured values before drawing conclusions. To try a flag case, set `flags:` in the run config or `$env:CROWDIO_FLAGS`.

## 8. Adaptive planning on real devices

Off by default. Read these before turning it on:

- **Phones only.** Do not connect PC workers: the Foreman sends version-2 messages to every connected worker, and PC workers reject them, which can stall planned jobs.
- **Only part of the flag set works with the current app.** `adaptive_planning` and `problem_reports` are supported. `prefetch` and `peer_fetch` are not implemented by the app, and `direct_path` has no peer listener yet. Use only the first two.
- **Never tested end to end.** The Foreman and the app's version-2 layer have only been tested against fake workers and against each other's specifications, not together.

```powershell
$env:CROWDIO_FLAGS = "adaptive_planning,problem_reports"
$env:CROWDIO_MANIFEST = "tools\onnx_shards_sentiment\manifest.json"
python tests\run_foreman_simple.py
```

Check: the startup shows `[FLAGS] {...}` with `adaptive_planning: True`, `[PLAN LOG] events -> logs\runs\plan_<time>.jsonl` appears, and `http://localhost:8000/api/plan/status` returns the plan. Then connect phones and submit planned inference through the SDK (`submit_inference`). If something misbehaves, unset `CROWDIO_FLAGS` and restart: everything returns to the legacy path.

## 9. Where to look after any run

| What | Where |
|---|---|
| Per-run report and summary table | `python tests\view_runs.py` (`--last`, `--job <id prefix>`), files in `logs\runs\` |
| Planner decisions | `logs\runs\plan_<time>.jsonl` |
| Live state | `http://localhost:8000` and `GET /api/runs`, `/api/links`, `/api/plan/status` |
| Raw tables | `python tests\view_database.py` |
| Simulator output | `devsim_results\<name>\` |

## 10. What does not work or is unproven today

- Phones and the Foreman with adaptive planning: untested together (section 8).
- `prefetch`, `peer_fetch`: unsupported by the current app. `direct_path`: unverified.
- Not checked before offloading a model: the phone's free RAM or disk.
- The Foreman handles about 6 to 7 tasks per second with 15 workers, so very short tasks leave a large fleet idle.
- Bandwidth-aware scheduling has only been measured in simulation, with a small gain (about 6% in the test above); it has not been run on real phones.
- Docker: `Dockerfile.foreman` and `Dockerfile.worker` need fixes first (see README Known Gaps).
- Nothing in this tree is committed since `06e75cf`.
