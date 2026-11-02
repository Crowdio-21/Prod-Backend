# Report 2.1: results of running every simulation (2026-11-02)

Everything below was run on this working tree on one Windows machine. Nothing here was measured on real phones. Raw outputs are in the scratchpad folder `rep21` (`%TEMP%\rep21`): one `.txt` per run plus per-run folders with `summary.json`, `devices.csv` and `bandwidth_trace.csv`.

## 1. What was run

| Simulator | Runs |
|---|---|
| `evaluation.scheduler_sim` | 3 layouts (30 random layouts each) |
| `evaluation.reconfig.run compare` | 10 policies, DistilBERT 5-phone spec |
| `evaluation.reconfig.run notax` | flags off vs all five on |
| `evaluation.reconfig.run dp-gap` | 200 random instances |
| `evaluation.devsim compare` | all 7 scenarios with the `mixed` workload, plus the `dnn3`, `stream` and `uniform` workloads on `congestion`; 15 devices, FIFO vs round-robin vs bandwidth-aware, **one run each** |

Not run: `reconfig sim` with the oracle separately (`compare` already includes the oracle), `replay` and `metrics` (they need a log from a live run), `reconfig live` and `devsim serve` (they need a real client or Foreman).

## 2. Device simulator: scheduler comparison (15 devices, speed 1)

Makespan in seconds, lower is better. The percentage is relative to FIFO. The bandwidth is identical for every scheduler within a row.

| Scenario / workload | FIFO | Round-robin | Bandwidth-aware | Best |
|---|---|---|---|---|
| stable | 62.7 | 63.9 (-1.8%) | 64.0 (-2.0%) | FIFO |
| congestion | 87.2 | 66.6 (+23.7%) | **57.3 (+34.4%)** | bandwidth-aware |
| drift | 66.2 | 71.5 (-8.1%) | **53.5 (+19.1%)** | bandwidth-aware |
| drop | 59.0 | 59.9 (-1.5%) | **56.4 (+4.4%)** | bandwidth-aware |
| commute | 67.8 | **60.1 (+11.3%)** | 76.7 (-13.2%) | round-robin |
| churn | **59.2** | 72.3 (-22.1%) | 60.9 (-2.8%) | FIFO |
| shared-wifi | 61.7 | 62.7 (-1.6%) | **55.3 (+10.4%)** | bandwidth-aware |
| congestion, `dnn3` | 57.5 | 57.7 (-0.2%) | **44.2 (+23.2%)** | bandwidth-aware |
| congestion, `stream` | 81.2 | 81.7 (-0.7%) | **78.8 (+2.9%)** | bandwidth-aware |
| congestion, `uniform` | **46.6** | 58.4 (-25.2%) | 50.3 (-7.8%) | FIFO |

All 30 runs completed with every task finished and all 15 devices used (14 for bandwidth-aware on `dnn3`).

**What this shows**
- Bandwidth-aware was fastest in 6 of 10 cases and clearly ahead where link speeds vary a lot and persistently: `congestion` (+34%), `drift` (+19%), `dnn3` (+23%) and `shared-wifi` (+10%). These are the cases it was designed for.
- With stable links (`stable`) there is nothing to exploit, and it is level with the others (within about 2%).
- It was **worse than FIFO in 4 cases**: `commute` (-13%, the worst result), `uniform` (-8%), `churn` (-3%) and `stable` (-2%). The `stable` and `churn` gaps are within the noise, but `commute` and `uniform` probably are not.
- `commute` is links that drift plus short outages. A plausible cause is that the scheduler keeps trusting a link estimate that is a few seconds stale when the link drops out. I have not confirmed this.
- `uniform` is 120 identical tasks. There, choosing by link speed gives no information about task size, and the extra decisions don't pay for themselves. Again, I have not confirmed the cause.

**How much to trust it.** Each cell is a single run. In earlier repeated runs, FIFO alone varied by several percent between runs, and the drop result moved from 5.8% (two runs, earlier) to 4.4% (one run, now). Treat differences under about 5% as noise. The large wins (congestion, drift, dnn3) are well outside that. The large losses (commute, and round-robin on churn and uniform) deserve a repeat with `--repeats 5` before anyone acts on them.

## 3. `scheduler_sim`: real scheduler classes on simulated workers (no Foreman)

Mean makespan over 30 random layouts, seconds.

| Layout | FIFO | Round-robin | Least-loaded | Bandwidth-aware |
|---|---|---|---|---|
| Uneven links (0.5 to 20 MB/s), 24 tasks | 9.29 | 9.89 | 9.29 | **4.23** (worst case 5.2 vs 16.3) |
| Uneven links, only 3 tasks for 6 workers | 7.06 | 6.29 | 7.06 | **1.46** |
| Identical links (5 MB/s), 24 tasks | 4.50 | 4.50 | 4.50 | **4.27** |

This simulation uses the scheduler's own cost model, so it shows the policy working with accurate estimates, not that real estimates are accurate. The device simulator in section 2 is the harder test, because the scheduler there has to learn the links from noisy reports. The gap is much smaller there (up to 34% rather than 54 to 79%), which is the more realistic number.

The run printed many `UnicodeEncodeError` logging messages from the FIFO scheduler's log lines (box-drawing characters on a Windows console). They do not affect the results.

## 4. Planner simulator (`evaluation.reconfig.run compare`)

DistilBERT split across 5 phones, placeholder phase times (see the caveat below). Total completion time for the scenario, lower is better; regret is the gap to the clairvoyant reference.

| Policy | Total (s) | Regret (s) | Switches | Wasted | Stall (s) | Data moved (MB) |
|---|---|---|---|---|---|---|
| oracle (reference) | 388.5 | none | 1 | 0 | 1.9 | 2.4 |
| B0 (fixed pipeline) | 722.6 | 334.1 | 1 | 0 | 1.0 | 2.4 |
| B1 | 722.6 | 334.1 | 1 | 0 | 1.0 | 2.4 |
| B2 (greedy) | 546.4 | 157.9 | 3 | 1 | 168.6 | 631.1 |
| B3 | 430.6 | 42.1 | 1 | 0 | 1.9 | 267.9 |
| B4 | 430.1 | 41.6 | 1 | 0 | 1.1 | 267.9 |
| B5a | 440.5 | 51.9 | 1 | 0 | 42.5 | 267.9 |
| B5b | 397.0 | **8.5** | 1 | 0 | 0.0 | 267.9 |
| S-aware-placement | 399.6 | 11.1 | 1 | 0 | 1.8 | 2.4 |
| S-aware | 399.6 | 11.1 | 1 | 0 | 1.8 | 2.4 |

**What this shows.** The fixed pipeline (B0) takes nearly twice as long as the reference once the network degrades (722.6 s against 388.5 s). The switching-cost-aware policies (S-aware and B5b) stay within 8.5 to 11 s of the reference. The greedy policy B2 switches three times, one of them wasted, and stalls for 169 s while it moves 631 MB, which is why it loses even though it adapts. S-aware reaches almost the same result as B5b while moving only 2.4 MB instead of 268 MB, (the same as the reference). I did not investigate why; the likely reason is that it changes placement among shards workers already hold, instead of re-cutting the model.

**Caveat:** the spec uses placeholder phone timings, so the absolute seconds are not predictions for real phones. The ranking of policies is the useful part.

### Flag cost when nothing goes wrong (`notax`)

| | Flags off (B0) | All five flags on |
|---|---|---|
| Throughput (requests/s) | 1.814 | 1.814 |
| Total completion (s) | 386.7 | 386.4 |
| Control traffic per request | 1,162 bytes | 2,491 bytes |
| Control traffic per minute | 126 KB | 270 KB |
| Planner time | 0.5 ms | 8 ms |
| Data moved | 0 | 540.5 MB |

With a healthy network the flags cost no throughput or latency in this model. They roughly double the control traffic (about 144 KB more per minute), and with prefetch on, workers pull 540 MB of shard files they may never use. That prefetch volume is the real price, and it matters on mobile data or a weak link.

### Planner accuracy (`dp-gap`, 200 random instances)

| Method | Mean gap to exhaustive search | Max gap | Exactly optimal |
|---|---|---|---|
| Scalarised DP | 0.12% | 14.4% | 98% |
| Pareto DP | 0% | 0% | 100% |

The Pareto DP, which the planner uses, matched exhaustive search on every instance.

## 5. What could not be tested by simulation

- **Adaptive planning and the flags against the real Foreman.** The simulated devices speak only the old protocol. The flag matrix is covered by the fake-worker tests (`tests/planning/test_flag_matrix.py`, all 32 combinations).
- **Real phones.** None of the numbers above come from hardware. The link speeds, CPU factors and task times are plausible values, not measurements.
- **Foreman limit.** The Foreman handles about 6 to 7 tasks per second with 15 workers, so makespans of 45 to 90 s here partly reflect that limit, not only the devices. This is one reason differences between schedulers are compressed.

## 6. Suggested next steps

1. Repeat `commute`, `uniform` and `churn` with `--repeats 5` to see whether the bandwidth-aware losses are real.
2. If `commute` holds up, make the scheduler distrust a link estimate faster when a report is stale or shows an outage.
3. Replace the placeholder phase times in `spec_distilbert_5phones.yaml` with measurements from a real phone, then rerun the planner comparison.
