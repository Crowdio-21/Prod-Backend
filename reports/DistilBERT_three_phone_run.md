# DistilBERT on three real phones: run report

Run date: 2026-11-16. Source data: `logs/runs/` (runs 8d4e6ee8 and 32979992), the Foreman log `foreman_live.log`, and the Foreman REST API.

## 1. Setup

| Item | Value |
|---|---|
| Foreman | `tests/run_foreman_simple.py`, fresh database, scheduler `BandwidthAwareScheduler` |
| Model | DistilBERT SST-2, fixed three-stage split: `cell_a` (172 MB), `cell_b` (81 MB), `cell_c` (2.3 MB) |
| Workers | 3 Android phones, all Samsung SM-A366E, 8 cores at 1.8 GHz, 7.3 GB RAM, same Wi-Fi as the Foreman |
| DNN worker scope | `mobile` (PC workers receive no DNN tasks) |
| Client | `multi_input_test.py`, 3 stages per sentence, models downloaded by each phone from the Foreman |
| Dataset | first 10, 100 and 700 rows of `data/sentences.csv` (SST-2 sentences with labels) |

## 2. Results

| Run | Sentences | Tasks | Wall time | Sentences/s | Returned | Accuracy vs labels |
|---|---|---|---|---|---|---|
| Warm-up | 10 | 30 | 94.9 s | 0.11 | 10 of 10 | not computed |
| Medium | 100 | 300 | 88.7 s | 1.13 | 100 of 100 | 94 / 100 (94.0%) |
| Full | 700 | 2100 | 680.9 s | 1.03 | 700 of 700 | 638 / 700 (91.1%) |

The warm-up time is dominated by the first model downloads and session start-up on each phone. The two larger runs came after the models were already on the phones.

**Accuracy:** 638 of 700 (91.1%) is in line with the roughly 91% reported for DistilBERT on SST-2, so splitting the model over three phones did not visibly change its output. The 100-sentence run is a subset of the same data, so the two accuracy figures overlap and are not independent samples.

### Per stage, 700-sentence run

| Stage | Tasks | Mean task time | p50 | p95 | Max | Stage span | Tasks per phone |
|---|---|---|---|---|---|---|---|
| `cell_a` | 700 | 7.03 s | 4.80 s | 19.19 s | 53.35 s | 265 s | 248 / 235 / 217 |
| `cell_b` | 700 | 14.79 s | 11.62 s | 38.63 s | 80.24 s | 481 s | 214 / 246 / 240 |
| `cell_c` | 700 | 8.43 s | 6.29 s | 19.47 s | 21.82 s | 230 s | 334 / 259 / 107 |

Stage spans overlap because the pipeline streams. Task times here are far higher than in the 100-sentence run (mean 1.24 s, 4.75 s and 1.66 s per stage). The log does not separate execution from waiting on the phone, and a larger batch queues more tasks per phone, so part of this is probably queueing. I did not verify that.

### Per phone, 700-sentence run

| Phone | Tasks | Busy time | Stage 0 / 1 / 2 |
|---|---|---|---|
| `…2b3d77` | 796 | 7,827 s | 248 / 214 / 334 |
| `…60eafb` | 740 | 8,496 s | 235 / 246 / 259 |
| `…71a3f3` | 564 | 4,855 s | 217 / 240 / 107 |

Busy time is the sum of task durations, so it can exceed the 681 s wall time when a phone runs several tasks at once.

## 3. Observations

1. **Load spread depends on batch size.** With 100 sentences, one phone ran 94% of `cell_b` and 99% of `cell_c`. With 700 sentences all three phones shared every stage, with 107 to 334 tasks each in the last stage. The scheduler charges for downloading a model a phone does not yet hold, which can favour a phone that already has it. I think that explains the concentration in the short run, but I did not isolate the cause.
2. **Failures were few and recovered.** The 700-sentence run had 1 failed task (logged twice), "Downloaded ONNX model is empty" for `cell_b`, on `…2b3d77`, and it was re-run. The 100-sentence run had 3 failed tasks (logged six times) of the same kind on a different phone, `…60eafb`. So the error is not tied to one phone. The Foreman served all these downloads with HTTP 200 and **the cause is not established**. One possibility is two simultaneous downloads of the same file by one phone. I did not test it.
3. **The slowest tasks are model loads.** The 62.1 s (100-sentence run) and 80.2 s (700-sentence run) `cell_b` maxima are consistent with first-time downloads of the 81 MB model plus session start-up.
4. **No real link measurements were used.** `/api/links` listed no workers in either run, and the scheduling decisions were marked `imputed_link: true`. The scheduler used its 10 Mbit/s default for all three phones. This build of the app does not appear to report link measurements, so the bandwidth-aware part of the scheduler was not exercised.

## 4. What these runs do and do not show

**Show:** the backend, the DNN pipeline and the dashboard work end to end on three real phones for up to 700 sentences, with complete results and accuracy matching the unsplit model's published figure.

**Do not show:**
- Any benefit of bandwidth-aware or adaptive scheduling (no link data, identical phones).
- Behaviour under changing bandwidth, load, heat or phone loss.
- Repeatability: there is one run per size, with no repeats.
- Any comparison with FIFO or round-robin on the same phones.
- Power use or temperature: the battery readings were not tracked across the runs.

## 5. Next

- BERT-base (12 layers, 30 shards exported and verified in `tools/onnx_shards_bert_base/`) has **not** been run on the phones. It needs the Foreman in planning mode and an app build that supports planned inference.
- To exercise the scheduler properly: repeat the 700-sentence run under FIFO and bandwidth-aware on the same phones, and find out why link reports are missing.
