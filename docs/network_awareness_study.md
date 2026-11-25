# When does network-aware scheduling help? A regime study (simulation)

Date: 2026-11-19. Code: `evaluation/netsim/` (model, simulator, sweep). Data: `evaluation_results/netsim/` (`runs.csv` has every run, `tables.md` every table). Figures: `reports/network_study/`. Reproduce: `python -m evaluation.netsim.sweep run` then `summarize`.

## 1. Question and short answer

**Question.** Does scheduling on measured bandwidth (and on RSSI) beat plain FIFO, and in which regimes?

**Short answer.** Network awareness helps modestly in most regimes and a lot in one. An oracle that knows the true current bandwidth gains 4–8% over FIFO in the base case, 11% under 50x link heterogeneity, and 29% when each phone gets one task. The scheduler's own estimates capture only part of that, and none of it where links change faster than reports arrive (heavy contention, outages, radio-limited links, very large fleets). RSSI helps in only one regime, as a calibrated prior before any bandwidth has been measured; as a correction to measured bandwidth it makes things slightly worse.

## 2. Method

The earlier simulations had four weaknesses; each is addressed here.

| Weakness | Fix |
| --- | --- |
| RSSI derived from the true link rate, so a perfect predictor by construction | RSSI and throughput come from separate mechanisms (below) |
| Scheduler handed the true link rates | Schedulers see only what phones report: heartbeats every 1, 10 or 60 s (8% noise), or a rate measured on a large transfer after a task (the Android behaviour), plus RSSI every 5 s (whole dB, 1.5 dB noise) |
| Foreman dispatches only about 6-7 tasks/s | This study drives the scheduler classes directly, so there is no dispatch cap; differences come from placement alone |
| 3 runs per scheduler against about 20 s of run-to-run variance | 30 paired seeds per condition (7-10 for the 100-device fleet); every scheduler sees the identical world, tasks and report noise |

**Network model** (`evaluation/netsim/model.py`).
- **Position:** random-waypoint mobility at a chosen speed (static, 1.4 m/s walking, 20 m/s vehicular) in a disc around the access point.
- **RSSI:** log-distance path loss (exponent 3.2), shadow fading (4 dB, AR(1) in distance travelled), 2 dB fast fading, and a floor during handoff outages.
- **Link rate:** capped Shannon-style curve of the SNR (40 MHz, 0.6 efficiency, 10 dB gap, 300 Mbit/s cap).
- **Achievable throughput:** the minimum of the link rate times (1 - airtime used by others) and a per-phone backhaul cap (median 20 Mbit/s; your real phones reached 15-17 Mbit/s on a 670 Mbit/s link).
- **Invisible to RSSI:** the backhaul cap, airtime taken by other devices, interference bursts (they lower SNR, not RSSI) and throttling to 150 kB/s. Handoff outages are visible.

**Schedulers** (the real classes): FIFO, round-robin, least-loaded, bandwidth-aware, adaptive, RSSI-aware with its default (placeholder) curve, RSSI-aware with a curve fitted on 5 held-out worlds from the base environment (as one would fit from a real run's heartbeat log), and an **oracle** (bandwidth-aware with the true current rates). Least-loaded equals FIFO and adaptive equals bandwidth-aware in every condition, because this simulator gives them no load, failure or battery signal; they stay in the tables for completeness and are left out of the figures.

**Base case.** 12 phones, walking, moderate heterogeneity (backhaul spread 5x, devices within 40 m), 20 Mbit/s cap, no contention or events, bandwidth reports every 10 s, 8 tasks per phone, transfer time about half of compute time. One axis is changed at a time; a transfer-share x heterogeneity grid gives the regime map.

**Statistics.** Gain = 1 - makespan / FIFO makespan, per seed. We report the median over seeds with a 95% percentile-bootstrap interval of the median (4,000 resamples). `*` marks an interval that excludes zero.

**Limits of the method.** Contention is an exogenous airtime process, not the fleet's own traffic. Tasks are pulled by idle phones one at a time. The scheduler is told each task's true result size. The mapping window is 24 tasks in every condition (the real default is 64; the mapping cost grows with its square); the fleet axis tests it separately. No real phones were used.

## 3. Results

Median gain over FIFO (%), bandwidth-aware / calibrated RSSI-aware / oracle. Full tables with intervals are in `tables.md`.

### 3.1 The axes

| Axis | Setting | Bandwidth-aware | RSSI (calibrated) | Oracle |
| --- | --- | --- | --- | --- |
| **Tasks per device** | 1 | **+30.4*** | +20.1* | +28.9* |
| | 2 | +10.8* | +5.6* | +11.7* |
| | 4 | +4.3* | +1.1 | +6.3* |
| | 8 (base) | +4.1* | +0.7 | +5.8* |
| | 16 | +3.1* | +2.1* | +5.3* |
| **Transfer / compute ratio** | 0.05 | +3.4* | +1.1 | +4.0* |
| | 0.5 | +4.1* | +0.7 | +5.8* |
| | 2 | +3.2 | +0.3 | +6.7* |
| **Heterogeneity** | identical | +1.8* | +1.8* | +3.1* |
| | moderate | +4.1* | +0.7 | +5.8* |
| | extreme (50x) | +1.9 | -0.9 | **+11.5*** |
| **Mobility** | static | **+8.1*** | +6.1* | +8.7* |
| | walking 1.4 m/s | +4.1* | +0.7 | +5.8* |
| | vehicular 20 m/s | +4.2* | -0.5 | +5.1* |
| **Bandwidth reports** | every 1 s | +5.3* | +3.2* | +6.2* |
| | every 10 s | +4.1* | +0.7 | +5.8* |
| | every 60 s | +2.9* | +1.1 | +5.8* |
| | only after transfers | +5.2* | +2.0* | +5.8* |
| **Contention** | none | +4.1* | +0.7 | +5.8* |
| | 30% airtime | +4.7* | +0.2 | +6.6* |
| | 60% airtime | +1.1 | -2.8 | +6.8* |
| **Backhaul cap** | 5 Mbit/s | +6.8* | +4.7* | +7.8* |
| | 20 Mbit/s | +4.1* | +0.7 | +5.8* |
| | 100 Mbit/s | -0.8 | -4.0* | +7.3* |
| | none (radio only) | -1.1 | -4.0 | +8.4* |
| **Network events** | none | +4.1* | +0.7 | +5.8* |
| | handoff outages 2/min | +0.4 | -2.6 | +1.8* |
| | interference bursts 2/min | +3.1 | +0.6 | +5.8* |
| | 25% of phones throttled | +3.5* | +1.3* | +6.1* |
| **Payload (ratio fixed)** | 20 KB | +4.6* | +2.5* | +4.7* |
| | 2 MB | +2.0* | +0.5 | +4.6* |
| | 20 MB | -0.1 | -0.0 | +0.5 |

### 3.2 When it helps, and when it does not

1. **Few tasks per phone (+30% with one, +11% with two).** With one task per phone the whole job is a single assignment, and putting the large payloads on the fast links is the only lever. Once every phone has 8 or more tasks queued, FIFO's pull-based dispatch already balances load (phones that finish early take more work), and the ceiling falls to about 5%.
2. **Stable links (+8% static against +4% walking).** Reports stay valid longer.
3. **Spread-out links, if the estimates can be trusted.** Under 50x heterogeneity the oracle gains 11.5%, but bandwidth-aware gains an insignificant 1.9%: the estimate error is 0.59 log units (about 80%) against 0.34 at moderate heterogeneity.
4. **Almost no gain when transfer is a small part of the work** (RTT-bound 20 KB tasks gain about 4.6% for both oracle and estimator; huge 20 MB tasks gain nothing because the tail is one long task).
5. **No gain, and sometimes a loss, when links change faster than reports arrive.** Radio-limited links, 100 Mbit/s backhaul, heavy contention and outages all show an oracle gain of 2-8% and an estimator gain of about zero.
6. **Fresh reports matter, and transfer-based measurements are good.** Estimate error falls from 0.34 (reports every 10 s) to 0.16 (every second), and bandwidth-aware gains +5.3% instead of +4.1%. Reports only after transfers do as well as every-second heartbeats (+5.2%, error 0.27) when each phone finishes tasks often.
7. **Fleet size.** With the scheduler's window fixed at 24 tasks, bandwidth-aware falls behind FIFO as the fleet grows (+3.8% at 10 phones, +1.5% at 30, -19% at 100, and the oracle is also negative at 100). At 30 phones, widening the window to the fleet size restores the gain (+3.8%, oracle +8.5%). At 100 phones the same change turns -19% into **+5.3%** (interval +3.3 to +11.0, 7 seeds; the oracle gains 9.3%). So the loss at 100 phones came from the window and not from the policy. The real scheduler's default window (64) is below a 100-phone fleet, so the window should scale with the fleet.

### 3.3 RSSI

- **As a correction to a measured bandwidth, RSSI does not help.** The calibrated RSSI scheduler is 1-8% worse than bandwidth-only in almost every condition, with a 95% interval below zero in most. It is not better even in the radio-only world, where RSSI correlates strongly with throughput. Its downlink estimates are slightly closer to the truth (error 0.32 against 0.34 at base) but its schedules are slower, so a lower estimate error did not translate into a better schedule here.
- **As a calibrated prior before any measurement, it helps.** With reports only after transfers and one task per phone, every assignment of the bandwidth-aware scheduler is made on an unmeasured link: it gains -2.3% (not significant) against FIFO. The calibrated RSSI prior gains +6.4% and is 9.8% faster than bandwidth-aware (interval +3.5 to +25.1). The oracle gains 28.9%, so most of the potential remains.
- **With the placeholder curve RSSI hurts.** The default curve is 3-8% worse than FIFO in the cold-start conditions and 14% worse than bandwidth-aware at 8 tasks per phone. A curve must be calibrated before RSSI is used at all.
- **Why.** In this model most of what RSSI could add is already in a fresh measurement, and RSSI cannot see the backhaul cap, contention, bursts or throttling. Its drop-detection rule reacts to RSSI noise (2 dB fast fading plus 1.5 dB measurement noise, against a 4 dB threshold).

### 3.4 Regime map

`reports/network_study/regime_map.png` shows the median gain over FIFO for bandwidth-aware, calibrated RSSI-aware and the oracle across transfer/compute ratio (0.05-2) and heterogeneity (identical, moderate, 50x). The oracle's gain rises with both axes (3% for identical links to 14% at 50x heterogeneity and ratio 2). Bandwidth-aware follows it for identical and moderate links but falls to zero or below at 50x heterogeneity and ratios of 1 and 2, where the oracle's gain is largest. The gap between the two is the part an estimator would have to close.

## 4. Threats to validity

- **Simulation only.** Not yet checked on phones. The plan: replay a few simulated traces on the real phones with `tc netem` or router shaping, and check that the simulator and the phones rank the schedulers the same way. One such comparison would anchor everything above.
- **Model assumptions.** The log-distance and Shannon parameters are textbook values, not fitted to these phones. Contention is exogenous. Backhaul caps are drawn log-uniformly. Results in the radio-only and 100 Mbit/s conditions depend most on these choices.
- **RSSI calibration was fitted on the base environment** and reused in every condition, as a deployed system would; a curve fitted per environment would do better, and the cap-limited base environment gives a flat slope (68 dB per decade).
- **RSSI rule not tuned.** Thresholds (4 dB drop, -82 dBm, 1.5x penalty) were set before this study and left unchanged. A rule with hysteresis or noise filtering might not lose to bandwidth-only; it was not tried.
- **Public traces not used.** The replay loader exists (`ReplayWorld`), but the heartbeat log from the phone experiments has RSSI without bandwidth, and no public 4G/5G or Wi-Fi dataset has been converted yet.
- **Window.** The 100-device result depends on the scheduler's mapping window (see 3.2, point 7); the mapping cost grows with its square, so a very large window needs a faster mapping step.

## 5. Reproduce

```bash
python -m evaluation.netsim.sweep run            # 48 conditions, about 20 min on 8 cores (resumable)
python -m evaluation.netsim.sweep summarize      # tables.md, summary.json, figures
python -m pytest tests/test_netsim.py            # 20 tests
```
