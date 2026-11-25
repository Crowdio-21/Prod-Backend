# Experiment plan: network conditions for the phone runs

Factors come from the supervisor's list: distance (RSSI), network load, number of devices, type of traffic, mobility. Each row says how the factor is varied and measured on the three real phones and in the simulator, and what is not covered.

## How to run one condition

Start the Foreman with the heartbeat recorder on (this records what the phones measure while a run is going):

```powershell
$env:CROWDIO_SCHEDULER = "bandwidth_aware"
$env:CROWDIO_PUBLIC_URL = "http://<PC address>:8000"
$env:CROWDIO_EXPERIMENT_LOG = "logs\experiments\heartbeats.jsonl"
python tests\run_foreman_simple.py
```

Then run each condition (the labels are yours; the RSSI, link speed and free RAM columns are what the phones measured during the run):

```powershell
python -m evaluation.experiments.run_condition --sentences 30 --distance 5m --load none --traffic dnn-distilbert --mobility static
```

One row per run is appended to `logs\experiments\results.csv`: seconds, sentences/s, results returned, accuracy, failures, tasks per phone, mean/min/max RSSI, link speed, minimum free RAM.

## Factors

| Factor | Real phones | Simulator (`python -m evaluation.devsim`) | Measured by | Not covered |
|---|---|---|---|---|
| **Distance / RSSI** | Move phones to fixed distances from the access point, or behind a wall. Label it with `--distance`. | `drift` and `drop` scenarios change bandwidth over time. There is no RSSI-to-bandwidth model. | `rssi_dbm`, `link_mbps` in the heartbeat log | A calibrated RSSI-to-throughput curve for these phones and this access point |
| **Network load** | Needs other devices generating traffic on the same Wi-Fi (for example iperf3 or large downloads). Label it with `--load`. | `congestion` and `shared-wifi` (`--shared-capacity-mbps`) | Task time, failures, `bw_down` once measured | Real interference sources (microwave, neighbouring networks) |
| **Number of devices** | 1, 2 or 3 phones (close the app on the others). Counted automatically as `devices_online`. | `--devices N`, up to 15 or more | `tasks_per_phone` | More than 3 physical phones |
| **Type of traffic** | DistilBERT (about 490 KB of activations per task) against a small-payload job such as `tests\example_client.py` | `--workload mixed`, `uniform`, `dnn3` or `stream` | Task time, bytes per task | Video or streaming traffic |
| **Mobility** | Walk with the phones along a fixed route, and record the route in `--mobility`. | `commute` (drifting links plus short outages) and `churn` (devices leaving and returning) | RSSI over time, failures, re-runs | Vehicular speeds (not testable indoors); the simulator has no speed parameter |

## Suggested matrix (30 sentences each, 3 repeats per row)

1. Baseline: 3 phones, static, no load.
2. Distance: near, mid, far, and behind a wall (4 conditions).
3. Load: none, light, heavy.
4. Devices: 1, 2, 3 phones.
5. Traffic: DistilBERT against the small-payload job.
6. Mobility: static against walking.

That is about 14 conditions, 3 repeats each. At about 50 s per run, that is roughly an hour of runs plus setup.

## Limits to state in any write-up

- The installed app reports RSSI, link speed, free RAM, battery and thermal state, but not a measured bandwidth until a model download has been timed. In the first runs `bw_down` was empty, so the scheduler's bandwidth estimates were defaults. Read the scheduling results with that in mind.
- The fixed DistilBERT pipeline does not cut the model by available RAM. Only the planning mode (adaptive planning with a shard manifest) checks that a shard fits in a phone's free RAM.
- Three identical phones do not exercise device heterogeneity.
