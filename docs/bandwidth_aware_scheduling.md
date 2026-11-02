# Bandwidth-aware scheduling

Written 2026-11-02. This is the design note for `BandwidthAwareScheduler`
(`foreman/core/scheduling/bandwidth_scheduler.py`, `link_estimator.py`). How to turn it on is in the
README, section 11.

## 1. What it does

For every task and worker it estimates the time from dispatch to result, counting the time to move the data:

```
T(task, worker) = rtt + (arguments + code + model file not yet on the worker) / downlink
                  + compute + expected result size / uplink
```

It then assigns waiting tasks to idle workers so that tasks finish as early as possible:

- big payloads get the fast links;
- when there are fewer tasks than workers, the slow links stay idle;
- a task waits briefly for a fast worker that is about to free up, instead of going to a slow idle one;
- if a medium is shared, concurrent transfers split its capacity.

## 2. Literature

### How the sources were gathered

Google Scholar blocks automated access with a CAPTCHA, so it could not be queried. I searched the open web and arXiv instead and read the papers listed as "read" below from their PDF or HTML text. For the rest I only saw search-result summaries, which are marked. **Before citing any of these in your own paper, check them against the originals.**

To repeat the search in Scholar yourself, these queries cover the same ground:

- `bandwidth-aware task scheduling mobile edge computing`
- `network-aware scheduling heterogeneous workers data transfer time`
- `DNN partitioning edge devices bandwidth pipeline scheduling`
- `throughput prediction harmonic mean exponentially weighted moving average offloading`
- `Sufferage heuristic heterogeneous computing independent tasks`

### Sources and what was taken from each

| Source | What it contributes | What we took | How it was checked |
|---|---|---|---|
| Cotter, Castineiras, Cionca, *Accuracy vs Performance: An abstraction model for deadline constrained offloading at the mobile-edge*, arXiv:2510.01885 (2025) | A mobile-edge scheduler for DNN tasks with a dynamic bandwidth estimate: devices ping each other, the controller smooths the result with an EWMA (alpha 0.3) and rebuilds its model of the link. The link is treated as a shared resource with reserved time slots. Their finding: estimation helps avoid bad placements but does not offset congestion, and probing too often hurts the transfers it is meant to schedule. | EWMA with alpha 0.3; probe rarely and only when idle; model a shared link; staleness matters | Read (EWMA, probing and discretisation sections; evaluation only from the abstract) |
| Parthasarathy and Krishnamachari, *Partitioning and Placement of Deep Neural Networks on Distributed Edge Devices to Maximize Inference Throughput*, arXiv:2210.12219 | Minimises the bottleneck latency of a DNN pipeline over edge devices on one Wi-Fi network, using pairwise bandwidth: partition where the least data crosses, place on the highest-bandwidth arrangement. Reported 10x over random, 35% over greedy, within 9.2% of optimal. | Pairwise link model; "big tensors on fast links"; the bottleneck objective matches the planner's period `P = max(compute, transfer)` | Read (abstract, introduction, problem formulation) |
| Yousaf and Welzl, *Network-Aware HEFT Scheduling for Grid*, The Scientific World Journal (2014), PMC3919093 | HEFT assumes transfers do not contend. They model parallel flows sharing a bottleneck, stretch the transfers to realistic completion times, re-run HEFT with those costs and iterate (best found around the 7th iteration). Stretching raised HEFT's own makespan estimate by 29.65% on average. | Shared-medium contention with max-min fair shares, then re-mapping with the stretched costs (up to 3 iterations here) | Read (method and results). Note: their reported 18.31% improvement is relative to the stretched makespan, not to HEFT's original estimate (about 6% above it) |
| Kang et al., *Neurosurgeon: Collaborative Intelligence Between the Cloud and Mobile Edge*, ASPLOS 2017 | Predicts per-layer latency on each side plus the transfer time at the current bandwidth, and picks the split with the lowest total. Reported 3.1x average latency improvement. | The cost shape: compute plus data / bandwidth | Search summaries and a blog summary only; not read in full |
| Maheswaran, Ali, Siegel, Hensgen, Freund, *Dynamic Mapping of a Class of Independent Tasks onto Heterogeneous Computing Systems*, JPDC 59(2), 1999 | Compares Min-min, Max-min and Sufferage for mapping independent tasks onto heterogeneous machines. Sufferage slightly beats the others on makespan; which is best depends on how heterogeneous tasks and machines are. | The Sufferage heuristic (a task that would lose most by missing its best worker chooses first), computed on completion times that include when each worker becomes free | Search summary only. The exact definition of Sufferage used here is from general knowledge of the heuristic |
| Xiao et al., INFOCOM 2015 (mobile social networks) | Task makespan includes the time to send the task to the worker. | Transfer time belongs in the objective | Search summary only |
| Zhang et al., *EdgeShard: Efficient LLM Inference via Collaborative Edge Computing*, arXiv:2405.14371 (IEEE IoT-J, 2025) | Joint device selection and model partition with a dynamic program, using heterogeneous compute, network and memory. | Context for the planner (`foreman/planning/`), not used by this scheduler | Search summary only |
| Yin et al., *A Control-Theoretic Approach for Dynamic Adaptive Video Streaming over HTTP*, SIGCOMM 2015; Phone2Cloud (arXiv:2008.05851) | Harmonic-mean throughput prediction in adaptive video streaming; an exponential moving average of bandwidth in a smartphone offloading system. | Averaging transfer *time* per byte (equivalent to a harmonic mean) so one fast sample cannot hide a slow link | Search summaries only. That the MPC paper uses a harmonic mean came from a secondary blog and **was not confirmed against the paper** |
| Manamperi et al., *CROWDio: A Practical Mobile Crowd Computing Framework...*, arXiv:2604.19363 | This project's own paper: capability-aware (multi-criteria) dispatch gave 56.9% scheduling improvement; it has no bandwidth term, and names pipeline scheduling for multi-stage DNN inference as future work. | The gap this scheduler fills | Read (discussion, limitations, conclusion). The multi-criteria schedulers it evaluates were removed from this repository on 2026-11-02 and remain in git history |

### What the literature does not settle

- None of these works addresses a hub-and-spoke system where every tensor travels worker to Foreman to worker. The shared-medium model here is a simplification of that.
- None gives a tested smoothing factor for phones on cellular links. 0.3 is borrowed from Cotter et al., who used Raspberry Pis.
- No source found specifically on bandwidth-aware scheduling in volunteer or mobile *crowd* computing.

## 3. Design

### 3.1 Estimating links (`link_estimator.py`)

- **Seconds per byte, not bytes per second.** Transfer time is linear in bytes, so an EWMA (alpha 0.3) of the reciprocal is a harmonic-style mean. Example: five 1 MB/s samples then one 100 MB/s sample give an estimate of about 1.4 MB/s, where averaging the rates would give 31 MB/s.
- **Samples ignored:** transfers under 10 KB (they measure latency, not bandwidth) and rates outside 1 KB/s to 10 GB/s.
- **Unmeasured workers** are given the fleet median, so they look average, get tried, and thereby get measured. With no data at all, a prior of 10 Mbit/s and 50 ms is used.
- **Stale estimates** drift back towards the fleet median over one more `max_age_s` (300 s).
- **A direction nobody measured** copies the other direction of the same worker, then the fleet median.
- **Compute time** is learned per worker and per task key (the model file, else a hash of the function code) as `turnaround - predicted transfer`, so transfer is not counted twice. The first task that needs a model also pays for downloading it, and that download is subtracted too.
- **Result size** is learned per task key; until then it is assumed equal to the argument size (`result_ratio`).

### 3.2 Where measurements come from

| Source | Used for | Note |
|---|---|---|
| PC worker probe (`pc_worker/core/link_probe.py`) | downlink, uplink, round trip | Times a small request (fastest of three), a 256 KB download and a 128 KB upload against the Foreman's `/api/probe`. Runs only while idle and at most once a minute. The round trip is taken out of the transfer times but never removes more than 80% of a measurement, so a link as fast as the round trip cannot produce an absurd rate. |
| Android heartbeat (`net.bw_down`, `net.bw_up`) | downlink, uplink | Bits per second, converted. Already sent by the app. |
| Task completions | compute time, result size | Measured on the Foreman with its own clock. |

PC `network_speed_mbps` is deliberately **not** used: it is NIC traffic over 100 ms (utilisation), not available bandwidth.

### 3.3 Mapping (`batch_select_workers`)

The dispatcher only uses `batch_select_workers`, so that is the method that matters. `select_worker` and `select_task` follow the same cost.

1. Look at the first 64 waiting tasks. Compute `T` for each task on each idle worker, and also on each busy worker, which becomes available when its running task is expected to finish.
2. Map with **Sufferage on completion times**: repeatedly pick the task that would lose most by missing its best worker (second-best minus best completion time), give it its best worker, and advance that worker's free time.
3. Start now only the first task mapped to each idle worker. Tasks mapped to busy workers wait for them.
4. If a medium is shared (`CROWDIO_SHARED_CAPACITY_MBPS`), compute the max-min fair share of the chosen flows, re-cost with the stretched rates and map again, up to 3 times.

Bounds on waiting, so a task cannot get stuck:

- it only waits for a worker expected free within 30 s;
- never when nothing is running (no completion would wake it up);
- never for an offline worker;
- after 60 s it is forced onto an idle worker (aging, which Maheswaran et al. found helps).

Ties go to the more reliable worker, then to the lower worker id, so behaviour is deterministic.

### 3.4 What the first version got wrong

The first version ranked workers by cost only and assigned every idle worker a task at once. In simulation it was **worse than round-robin** (10.3 s vs 4.9 s), because it handed big tasks to slow idle workers rather than waiting for a fast one. Using completion times with worker free times fixed that, and the simulation, averaged over many layouts, is what showed the remaining gap. A live check also caught a probe bug (absurd 21,000 Mbit/s readings on localhost) that unit tests with exact fake clocks could not.

## 4. Results (simulation)

`python -m evaluation.scheduler_sim`: the real scheduler classes on simulated workers, 30 random layouts per scenario (task sizes and which worker has which link are shuffled, because the baselines pick by worker id order). Makespan in seconds, lower is better. The bandwidth-aware and round-robin figures are identical on every run. FIFO and least-loaded take workers in Python set order, which changes with each process, so they are shown as the range seen over repeated runs.

| Scenario | fifo | round_robin | least_loaded | bandwidth_aware |
|---|---|---|---|---|
| 6 workers, links 0.5 to 20 MB/s, 24 mixed-size tasks: mean | 8.4 to 9.6 | 9.89 | 8.4 to 9.6 | **4.23** |
| same: worst case | 14.0 to 16.0 | 16.96 | 14.0 to 16.0 | **5.20** |
| same links, only 3 tasks: mean | 5.2 to 6.9 | 6.29 | 5.2 to 6.9 | **1.46** |
| 6 identical 5 MB/s workers, 24 mixed-size tasks: mean | 4.50 | 4.50 | 4.50 | **4.27** |

**Read this with care.** The simulation uses the scheduler's own cost structure and gives it the true link rates up front, so it shows that the policy works *when the estimates are right*. It does not show that the probes and heartbeats produce right estimates. Its sizes and speeds are made up. A measurement on real phones is still needed. The small gain on identical links comes from placing large payloads first.

## 5. Live check (2026-11-02)

A Foreman manager with this scheduler, an HTTP probe endpoint, three real PC workers and a real 18-task job on one machine:

- all three workers reported their measured links (hundreds to thousands of Mbit/s, round trips of 1.5 to 26 ms, as expected on loopback);
- the scheduler learned a compute time per worker;
- the job returned 18 correct results and its run report included the link table.

All workers shared one machine, so the check proves the plumbing, not scheduling quality.

## 6. Limitations and not done

- No evaluation on real phones or on a real shared Wi-Fi network.
- Compute time is unknown until tasks complete (1 s is assumed), so the first tasks are placed on links alone.
- Learned estimates are in memory and lost when the Foreman restarts.
- A model is assumed to stay on a worker once sent. Nothing checks RAM or disk for it.
- The shared-medium capacity is a number you configure, not something measured.
- The `select_worker` path and the dispatcher's single-worker path are not used by default, so they are covered by unit tests only.
- Android workers already report bandwidth; no app change was made or needed. An Android round-trip time is not reported, so a fleet median is used for it.
- PC probes use the same Foreman REST port as the dashboard. If workers cannot reach it (a firewall, a different host), set `CROWDIO_FOREMAN_HTTP`; without a measurement the worker is treated as average.
