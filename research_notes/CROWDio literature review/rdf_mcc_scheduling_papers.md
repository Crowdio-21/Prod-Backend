# Mobile crowd computing, network-aware scheduling and edge AI: close reading of 6 papers and a citation audit of the CROWDio partitioned-inference paper

Method note: all six papers were read in full from the Zotero text extracts (every section, including appendices for Jalaparti et al.). Local copies used: `scratchpad/body_463.txt`, `plain/481.txt`, `body_483.txt`, `body_485.txt`, `body_487.txt`, `body_502.txt` (`C:/Users/DELL/AppData/Local/Temp/claude/C--Users-DELL-Desktop-Path-CROWDio-Research-Prod-Backend/639c5b0c-4052-4e4f-8ecd-2f88b2cda773/scratchpad/`). Each paper's findings are cited to its DOI or landing page. The text came from the Zotero PDFs; the links identify the work. The citation audit used web lookups (arXiv, MDPI, publisher and index pages, DBLP mirrors and OpenAlex).

---

## Q1. Honeybee-DR (Nagesh, Fernando, Loke, Ghari Neiat, Pathirana, 2026): what it does and how it relates to CROWDio

### Takeaway
Honeybee-DR is a delegator-centric, infrastructure-free (Wi-Fi Direct) work-stealing MCdC framework. It adds (a) runtime DAG growth through three patterns, with stateless workers, and (b) adaptive trap-job validation. On five Android devices it gets about 17% lower makespan, 21% lower delegator energy and 30.5% lower peak memory than its static-DAG mode, and 99.5% accuracy against rogue workers with up to 80.5% less validation overhead than 3-way majority voting. Of the six papers it is the closest peer to CROWDio, and the right "dynamic DAG plus reliability" comparator.

### Cited Findings
- **Citation:** Nagesh S.S., Fernando N., Loke S.W., Ghari Neiat A., Pathirana P.N., "Honeybee-DR: dynamic dependency management and lightweight reliability for mobile crowd computing," *Scientific Reports* (2026), Article in Press, received 28 May 2026, accepted 7 July 2026, DOI 10.1038/s41598-026-61849-0. Licence CC BY-NC-ND 4.0. Affiliations: Deakin University and the University of Queensland — [Sci. Rep. DOI](https://doi.org/10.1038/s41598-026-61849-0)
- **Problem:** static-DAG MCdC schedulers must provision the worst-case path for every input (e.g., run detection and motion analysis on every frame), which wastes CPU, memory and energy. Replication, majority voting and cryptographic verification cost too much on battery-limited phones. Target settings: disaster response, field science, classrooms, ad-hoc multi-camera setups, with no infrastructure — [Sci. Rep.](https://doi.org/10.1038/s41598-026-61849-0)
- **System model:** star topology. One delegator owns the JobPool and the DAG; workers are stateless, never see the DAG and never talk to each other. Idle workers send "steal requests". Job states are WAITING, READY, STOLEN and COMPLETED. Delegator failure ends the session (no replicated DAG), and the delegator reissues STOLEN jobs from dropped workers — [Sci. Rep.](https://doi.org/10.1038/s41598-026-61849-0)
- **Method, dynamic dependencies (Algorithm 1):** the delegator runs an application callback `AnalyzeResultForDynamics` on every result. The callback selects one of three patterns: data-dependent branching (simple or complex branch by a threshold), iterative refinement (spawn a refinement job while quality is below a threshold), or conditional cascade (a flag adds a fixed sub-graph). The new jobs and edges are added atomically; jobs whose predecessors are all complete go to READY, the rest to WAITING. Mechanism and policy are separated: the middleware owns atomic DAG updates, state transitions and chunking, and the application only decides what to spawn. The DAG grows monotonically, and cancelling running jobs or demoting READY to WAITING is not allowed. Cost is O(|J_new|+|E_new|) per completed job, using sparse adjacency lists — [Sci. Rep.](https://doi.org/10.1038/s41598-026-61849-0)
- **Method, unified DAG-aware work stealing (Algorithm 2):** three phases. (1) Dependency resolution promotes WAITING jobs to READY. (2) Capability filtering against the requesting worker's profile keeps weak phones off heavy jobs such as DL inference. (3) Adaptive chunking: chunk size comes from the worker's historical throughput and current network latency, to spread Wi-Fi Direct overhead and avoid "ping-pong" — [Sci. Rep.](https://doi.org/10.1038/s41598-026-61849-0)
- **Battery-aware policy:** above 30% battery the delegator runs normally. Below 30%, or in power-save mode, only refinement and cascade are allowed and heavy branches are deferred. Below 20%, discovery is disabled and the session finishes the existing DAG. The tiers match Android Battery Saver — [Sci. Rep.](https://doi.org/10.1038/s41598-026-61849-0)
- **Method, trap jobs:** per-worker reliability score s_w ∈ [0,1], starting at 1.0, learning rate η=0.1, with asymmetric updates α_w = η·s_w (reward) and β_w = η·(1−s_w) (penalty). Traps per chunk are T_w = round(T_max(1−s_w)) with T_max=5, and the intervention threshold is 0.5 (more traps, smaller chunks or suspension). Traps go through the same JobPool and the same capability filter as ordinary jobs — [Sci. Rep.](https://doi.org/10.1038/s41598-026-61849-0)
- **Evaluation setup:** five devices over Wi-Fi Direct, held stationary on one table: a OnePlus 5T delegator (SD835, 8 GB, Android 10), a Pixel 7 and three Pixel 6 Pros (Android 13–14). Workload: multi-camera video (EPFL and Stanford datasets, 10 s segments) through frame extraction, YOLOv4-tiny detection and feature matching; in the dynamic mode, detection and matching run only when motion is above a threshold. Baselines: monolithic, static partition, non-dependency-aware work stealing, Honeybee-DR static. 5 runs each. Delegator energy measured with Battery Historian, memory sampled at 1 s — [Sci. Rep.](https://doi.org/10.1038/s41598-026-61849-0)
- **Numbers, static mode vs. baselines (30 videos):** makespan 468 s → 227 s (−51%); average speedup 2.06× (peak 2.11×); delegator energy 1912 → 928 mAh (−51.4%); peak delegator memory 2234 → 635 MB (−71.6%); network 199 MB vs 344 MB for non-dependency-aware stealing (−42%) — [Sci. Rep.](https://doi.org/10.1038/s41598-026-61849-0)
- **Numbers, dynamic vs static:** processing time 14–19% lower (about 17% on average), 226.7 → 195.4 s at 30 videos. Energy 21% lower on average (928.4 → 779.8 mAh, 16% at 30 videos). Peak memory 30.5% lower on average (634.8 → 456.2 MB at 30 videos). Network traffic about 5% higher (199.3 → 209.8 MB). Discovery overhead 87.4–104.3 ms per segment, 96.5 ms on average, included in makespan (Table 6) — [Sci. Rep.](https://doi.org/10.1038/s41598-026-61849-0)
- **Numbers, reliability:** 10,000 WIDER FACE images; workers C and E are rogue with p_error ∈ {0.3, 0.7}. Majority vote (3 replicas): 99–100% accuracy, about 278 s overhead. 20% duplication: 69–71%. Fixed traps: 85–86%. Adaptive traps: 99.5% accuracy, 12–81 s overhead, i.e. up to 80.5% less than majority vote at p_error=0.7 and 70.9% less at p_error=0.3 — [Sci. Rep.](https://doi.org/10.1038/s41598-026-61849-0)
- **Stated limitations:** the single delegator is the scalability ceiling and a single point of failure. The two contributions are evaluated on separate workloads (no joint experiment). Mobility, weak signal and Doze are out of scope. Dropped workers are not reconnected automatically. Colluding workers and a compromised delegator are not handled. The DAG is kept in memory only. There are no live baselines against other dynamic-DAG schedulers (they need servers or SDN) — [Sci. Rep.](https://doi.org/10.1038/s41598-026-61849-0)
- **Lineage:** Honeybee (independent jobs) → Honeybee-T (sequential chains) → Honeybee-RS (trap jobs) → Honeybee-Tx (static DAG, capability-aware stealing; Future Internet 2025) → Honeybee-DR — [Sci. Rep.](https://doi.org/10.1038/s41598-026-61849-0); [Future Internet 17(10):446](https://www.mdpi.com/1999-5903/17/10/446)

### Inferences
- **Relation to CROWDio, differences:** CROWDio is push-scheduled. A server-side Foreman (FastAPI, WebSocket, SQLite) assigns work with pluggable rankers (MCDM, Sufferage bandwidth-aware). Honeybee-DR is pull-based work stealing over Wi-Fi Direct with no server. CROWDio's DAGs are static and declared through the SDK (pipelines with 1:1 streaming dependencies). Honeybee-DR grows the DAG at runtime.
- **Relation to CROWDio, what CROWDio has that Honeybee-DR lacks:** checkpoint (base+delta) and resume on disconnect (Honeybee-DR just reissues STOLEN jobs from scratch); a persisted job store (Honeybee-DR keeps the DAG in memory and loses the session if the delegator dies); explicit network-cost modelling (rtt + bytes/downlink + compute + result/uplink, max-min fair Wi-Fi share), where Honeybee-DR only folds latency into chunk size; and DNN partition placement with switching cost.
- **Relation to CROWDio, what it could adopt:** (1) Trap-job reliability scoring. CROWDio has no result-validation mechanism, and the asymmetric score could feed straight into the MCDM criteria as a "reliability" column. (2) The mechanism/policy split for runtime DAG growth, which would let `crowdio_pipeline` support conditional or branching stages. (3) Battery tiers aligned with Android Battery Saver. (4) Adaptive chunk size from historical throughput, which parallels CROWDio's batch mapping.
- Honeybee-DR's numbers are against its own static mode, not against other systems. CROWDio's write-up can make the same "same-platform ablation" argument where no infrastructure-free baseline exists.

### Gaps
- The paper is an unedited "Article in Press"; volume and article number were not yet assigned in the extracted text.
- The source code is only available on request, so a direct reproduction or comparison with CROWDio is not possible.

---

## Q2. The user's own paper (Manamperi et al., 2026): what it claims and how it maps to the current CROWDio code

### Takeaway
A 6-page workshop paper (arXiv 2605.20723, accepted at the ICML 2026 Workshop on Machine Learning for the Global South). It describes CROWDio's DNN pipeline subsystem: JIT shard loading, single-partition residency, a 4-tier affinity scheduler, zlib tensor transport and 1:1 streaming dependencies. On 5 Android phones running DistilBERT/SST-2 it reports 43±2 MB peak RSS per device, 50±3 mAh per run, and 34% lower batch latency for streaming than for barrier mode. Several numbers and related-work claims need correcting (see Q7).

### Cited Findings
- **Citation:** Manamperi L., Pathirana D., Pathirana T., Premarathna N., Gunasekera K., "Memory-Efficient Partitioned DNN Inference on Resource-Constrained Android Crowds," arXiv:2605.20723, v1 20 May 2026, "6 pages, 3 figures, 4 tables. Accepted at the ICML 2026 Workshop on Machine Learning for the Global South" — [arXiv](https://arxiv.org/abs/2605.20723)
  - The Zotero PDF lists four equal-contribution authors (Manamperi, T. Pathirana, D. Pathirana, Premarathna) and has the main-conference footer "Proceedings of the 43rd ICML, Seoul, PMLR 306, 2026". The arXiv record has five authors (Gunasekera added) in a different order, and describes the venue as a workshop.
- **Problem:** commodity handsets have 3.3–7.4 GB RAM. Compression, pruning and distillation still assume one host device. Split computing (SPINN, Neurosurgeon) assumes one cut to a reliable cloud — [arXiv](https://arxiv.org/abs/2605.20723)
- **System model:** SDK → Foreman → Android workers over WebSocket, with telemetry every 30 s. DistilBERT (67 M parameters, 6 layers) is split into 3 cells: (a) embedding + layers 0–2, eagerly broadcast; (b) layers 3–5, JIT-loaded on the first stage-0 output; (c) pre-classifier + classifier, JIT-loaded on the first stage-1 output. Each worker holds one ONNX session. N×S tasks are created, each stage-k task depending 1:1 on stage k−1 for the same input; barrier mode uses 1:N — [arXiv](https://arxiv.org/abs/2605.20723)
- **Transport:** self-describing JSON (dtype, shape, compression, base64-zlib). Payloads above a threshold (1 MB by default) go to a shared filesystem store and only a reference is forwarded. The paper reports 62±4% compression on [1,768] FP32 tensors (3072 → 1168 bytes) — [arXiv](https://arxiv.org/abs/2605.20723)
- **Affinity scheduler:** tiers are resident → cached on disk → idle → evict (an explicit UNLOAD then LOAD). A model-gating pre-dispatch check is applied. Within a tier, workers are ranked by entropy-weighted MCDM (EDAS, ARAS or MABAC) over CPU, RAM, battery, RTT and temperature. Assignment has two phases: rarest-coverage-first claiming among tier-1 workers, then an MCDM fill — [arXiv](https://arxiv.org/abs/2605.20723)
- **Evaluation:** 5 Android handsets (3.3–6 GB, Android 13–14, ONNX Runtime 1.17), local Wi-Fi, N=5, S=3, 10 runs. Streaming vs barrier: RSS 43±2 MB in both; pipeline latency 18.4±1.1 s vs 27.9±2.3 s (−34%); cold load 48±4 s; warm (tier-2) load 6±1 s; battery 50±3 vs 53±4 mAh — [arXiv](https://arxiv.org/abs/2605.20723)
- **Stated limitations:** single residency is conservative on devices above 6 GB; the 48 s cold start; zlib CPU cost on the critical path; topologies limited to linear chains; future work on energy-aware scheduling and quantization-aware shards — [arXiv](https://arxiv.org/abs/2605.20723)

### Inferences
- **Relation to the current CROWDio branch:** the paper describes the earlier affinity-tier design. The current `feature/reconfiguration-aware-planning` work goes further in four places. (a) It replaces the static 3-cell split with cut points and placement chosen from EWMA telemetry. (b) It models switching cost under single residency and switches only on a break-even rule with hysteresis and epochs. (c) It adds prefetch, peer fetch and a direct path, which address the paper's own "48 s cold start" and "prefetching" future work. (d) It adds the bandwidth-aware Sufferage scheduler, which addresses the "Wi-Fi contention" concern the paper raises only qualitatively.
- **Internal consistency issues a reviewer would flag:**
  - The "2–4 GB for a single DistilBERT ONNX session" claim is attributed to Sanh et al. (2019), which reports parameter counts and speed, not ONNX session RSS. An FP32 DistilBERT checkpoint is roughly 250 MB.
  - A peak RSS of 43 MB is hard to square with cell (a), which contains the 30,522×768 embedding table: about 23.4 M parameters, or about 94 MB in FP32. The RSS figure probably measures something narrower (e.g., native heap after mmap) or the shards are quantized. The paper should say which.
  - Table 4 gives identical cold- and warm-load times for streaming and barrier, which suggests they are shared measurements; the paper should say so.

### Gaps
- No ablation isolating each of the five mechanisms, no baseline other than CROWDio barrier mode (e.g., a monolithic single-device run on a higher-RAM phone, or EdgePipe-style partitioning), and no network-bandwidth sweep.

---

## Q3. Choudhary et al. (2025), "Edge AI: Deploying AI models on edge devices for real-time analytics": usable content and reliability

### Takeaway
A broad position or framework paper. It lists compression, edge-cloud offloading, federated learning, blockchain, differential privacy, XAI, RL energy policies and spiking neural networks, and then reports headline numbers (45% lower latency, 40 ms vs 120 ms, 94% accuracy, 1.5 W, a 67% "security resilience" gain) with no stated workloads, datasets, baselines or procedure. Use it only as a general "edge AI motivation" citation, not as evidence for any number.

### Cited Findings
- **Citation:** Choudhary S., Vijitha S., Bhavani D.D., Bhuvaneswari N., Tiwari M., Subburam S., "Edge AI: Deploying Artificial Intelligence Models on Edge Devices for Real-Time Analytics," *ITM Web of Conferences* 76, 01009 (2025), ICSICE-2025, EDP Sciences, CC BY 4.0, DOI 10.1051/itmconf/20257601009 — [ITM Web Conf. DOI](https://doi.org/10.1051/itmconf/20257601009)
- **Problem stated:** cloud AI has high latency, bandwidth cost and privacy risk, while existing edge-AI work lacks scalability, energy efficiency, security and explainability — [ITM](https://doi.org/10.1051/itmconf/20257601009)
- **Method (descriptive only):** quantization, pruning and distillation; "adaptive edge-cloud coordination" that offloads complex DL to the cloud based on bandwidth, load and latency; federated learning with blockchain-recorded updates; differential privacy; Zero Trust; an RL-based energy policy and SNNs; saliency and attention XAI; online and transfer learning. No algorithm, equation or pseudocode is given — [ITM](https://doi.org/10.1051/itmconf/20257601009)
- **Setup stated:** NVIDIA Jetson Nano and Raspberry Pi 4, with a quad-core A57, 128-core Maxwell GPU and 4 GB LPDDR4 listed in one column; TFLite and PyTorch; 128 GB SSD — [ITM](https://doi.org/10.1051/itmconf/20257601009)
- **Numbers reported:**
  - Latency: cloud AI 120 ms, traditional edge 85 ms, proposed 40 ms.
  - Accuracy: 85%, 88%, 94%.
  - Power: 5.2 W, 3.8 W, 1.5 W.
  - "Reduce latency by 45%", "12–18%" gain from multimodal fusion, "35% memory usage decrease and 50% energy saving" from compression, "67% improvement" in security resilience.
  - None of these comes with datasets, models or run counts — [ITM](https://doi.org/10.1051/itmconf/20257601009)
- **Reliability signals in the text:**
  - Table numbering is swapped (Table 12 and Table 21).
  - It cites a reference "[42]" when the list has only 15 entries.
  - One reference is "Anonymous (2024)".
  - The conclusion says "as of October 2023" and refers to "the chapter".
  - The literature review cites authors without years in places.
  - [ITM](https://doi.org/10.1051/itmconf/20257601009)

### Inferences
- **Relation to CROWDio:** different scope. It assumes a single edge device plus cloud offload, not a crowd of phones. Its "adaptive edge-cloud coordination using bandwidth, load and latency" is the same intuition as CROWDio's bandwidth-aware cost model, but with no model to compare against. CROWDio could cite it only as general motivation (latency, privacy and bandwidth arguments for on-device inference). The latency and energy numbers should not be quoted as comparators.
- Some reference DOIs in its list (e.g., Li et al. 2023 TII "10.1109/TII.2023.3245678"; Zhao, Hu & Xu 2024 IoT-J "10.1109/JIOT.2024.3276598") have placeholder-looking digit patterns. They were not checked here, but they are a further reason not to rely on this paper.

### Gaps
- The figures (energy bar chart, cloud vs edge) were not legible in the text extract, so whatever they show could not be checked.

---

## Q4. Jalaparti et al. (SIGCOMM 2015), "Network-aware scheduling for data-parallel jobs: plan when you can" (Corral)

### Takeaway
Corral uses the predictability of recurring jobs (input size predicted within 6.5%) to plan offline where each job's input data and tasks go: which set of racks, and with what priority. It treats this as malleable job scheduling, using a provisioning heuristic plus a widest-job-first extension of LPT that comes within 3% (batch) and 15% (online) of an LP lower bound. On a 210-machine YARN cluster it cuts makespan by 10–33%, median job completion time by 30–56% and cross-rack data by 20–90%. Its core lesson for CROWDio: co-planning data placement with compute placement beats network-aware task placement over fixed data, and the gains are orthogonal to flow-level scheduling.

### Cited Findings
- **Citation:** Jalaparti V., Bodik P., Menache I., Rao S., Makarychev K., Caesar M., "Network-Aware Scheduling for Data-Parallel Jobs: Plan When You Can," *Proc. ACM SIGCOMM '15*, London, 17–21 Aug 2015, pp. 407–420, DOI 10.1145/2785956.2787488 — [ACM DOI](https://doi.org/10.1145/2785956.2787488)
- **Problem:** shuffle- and join-heavy jobs are bottlenecked by oversubscribed rack-to-core links (3:1 to 10:1), and background traffic can use up to 50% of cross-rack bandwidth. Delay scheduling and Quincy get locality only for the first stage, because HDFS spreads input randomly. Flow schedulers (Varys, Baraat, Orchestra) only act once endpoints are fixed — [ACM](https://doi.org/10.1145/2785956.2787488)
- **Motivating data:**
  - Up to 40% of production jobs are recurring.
  - Input size is predicted with 6.5% average error, by averaging the same job at the same time of day on previous weekdays or weekends.
  - 75%, 87% and 95% of jobs in three clusters of more than 10k machines need less than one rack (240 slots).
  - [ACM](https://doi.org/10.1145/2785956.2787488)
- **System model:**
  - An offline planner outputs {R_j, p_j} per job: the set of racks and a priority.
  - At upload time, one replica of each chunk goes to a random rack in R_j and the other replicas elsewhere, as in HDFS.
  - Tasks are constrained to R_j, and a free slot goes to the highest-priority assigned job.
  - Planning works at job and rack granularity rather than tasks or stages: task sizes are unpredictable, and stage-level planning helped few DAGs.
  - Racks are not exclusively reserved, the scheduler stays work-conserving, and it falls back to unconstrained placement if many machines in a rack fail.
  - [ACM](https://doi.org/10.1145/2785956.2787488)
- **Method:**
  - Latency response function L_j(r) for r racks. For MapReduce it is map + shuffle + reduce. Map latency = waves × D_I/(N_M·B_M). Shuffle = waves × max(core time = D_core/(B/V), local time = D_local/(B−B/V)·(k−1)/k), where D_core = D_S/(r·k)·(r−1)/r. For DAGs, latency is the sum along the critical path.
  - A data-imbalance penalty α·D_I/r is added, with α = 1/(rack-core bandwidth).
  - Provisioning heuristic: start every job at r_j=1 and repeatedly add a rack to the current longest job until all reach R, evaluating all J×R candidate allocations.
  - Prioritization: widest-job-first, ties broken by LPT; online jobs sorted by arrival.
  - Complexity O(J²R²); 55 s for 500 jobs on 4000 machines (100 racks).
  - The objective is NP-hard. An LP relaxation (Appendix A) gives the lower bound.
  - [ACM](https://doi.org/10.1145/2785956.2787488)
- **Implementation:** YARN (Hadoop 2.4) + HDFS, about 300 lines of Java. HDFS `create()` takes a set of rack and replica-count tuples, and the MapReduce AM states locality preferences for all tasks, reducers included — [ACM](https://doi.org/10.1145/2785956.2787488)
- **Evaluation setup:** 210 machines in 7 racks of 30 (32 cores and 10 Gbps NICs each), folded CLOS at 5:1 (60 Gbps per rack to the core), emulated background traffic of up to 50% of the core. Workloads: W1 (from Quantcast), W2 (SWIM Yahoo, 400 jobs), W3 (200 Microsoft Cosmos jobs; median 180 tasks and 7.1 GB input, 95th percentile 2,060 tasks and 162.3 GB) and 15 TPC-H Hive queries on 200 GB. Baselines: YARN capacity scheduler (Yarn-CS), ShuffleWatcher, LocalShuffle — [ACM](https://doi.org/10.1145/2785956.2787488)
- **Numbers:**
  - Batch makespan 10–33% lower; cross-rack data 20–90% lower; compute hours up to 20% lower; reduce time about 40% better at the median.
  - Data-balance CoV at most 0.004, vs 0.014 for HDFS.
  - Online: median job time 30–56% lower, average 26–36% lower.
  - TPC-H: median 18.5% and average 21% lower.
  - Mix of 50 ad hoc and 100 recurring jobs: recurring jobs 33% (average) and 27% (median) faster; ad hoc jobs 37% faster at p90 and 28% lower makespan.
  - Background traffic 30 → 40 Gbps gives more than 2× larger gains.
  - With 50% input-size error, gains stay at 25–35%. Delaying up to 50% of jobs by ±4 min reduces gains from 40% to at least 25%.
  - Simulation (2000 machines, 50 racks, 1 Gbps): Yarn-CS+Varys is 46% better than Yarn-CS+TCP at the median; Corral+TCP beats Yarn-CS+Varys by about 45% at the median; Corral+Varys is best.
  - [ACM](https://doi.org/10.1145/2785956.2787488)
- **Limitations:** needs predictable recurring jobs, and ad hoc jobs get only the default scheduler. Each job is assumed to read its own dataset. Allocations are fixed, with no preemption, in the planner. Latency models assume uniform per-stage work and no skew. Remote-storage and cross-layer flow co-design are left to future work — [ACM](https://doi.org/10.1145/2785956.2787488)

### Inferences
- **Relation to CROWDio, what it adopts or parallels:**
  - CROWDio's reconfiguration-aware planner applies "plan when you can" to DNN pipelines: it predicts from EWMA telemetry and places model shards (data) together with the stages (compute).
  - Corral's "place input data where compute will run" corresponds to CROWDio's prefetch and peer fetch of model shards to the planned workers, and to the "args+code+model not yet on worker" term in the bandwidth-aware cost.
  - The latency response function L_j(r) is the analogue of CROWDio's per-batch cost estimate (rtt + bytes/downlink + compute + result/uplink).
  - Corral's TCP emulation in simulation uses max-min fair bandwidth sharing, the same model CROWDio uses for a shared Wi-Fi medium.
- **Relation to CROWDio, differences:**
  - Corral works on datacenter racks with a stable topology, an oversubscribed core and recurring jobs. CROWDio works with volatile phones on one shared Wi-Fi cell, where the bottleneck is the access medium, not a core.
  - Corral plans offline and stays fixed (no preemption). CROWDio replans by epochs and switches only when a break-even rule with hysteresis says the gain outweighs the switching cost.
  - Corral's priority-plus-rack plan is coarse-grained. CROWDio maps individual batches with Sufferage.
- **Relation to CROWDio, extends:** CROWDio's switching-cost model under single residency has no counterpart in Corral, which never pays to move or reload a model. That is a clean point of novelty to state when citing Corral.
- Corral's robustness tests (prediction error up to 50%, arrival jitter) are a template for an ablation of CROWDio's planner: inject EWMA bandwidth or compute mis-estimates and measure how much of the gain survives.

### Gaps
- None for the paper itself. All equations were recovered from the text, though the PDF-to-text conversion garbled some subscripts, which were reconstructed from context.

---

## Q5. Caminero & Muñoz-Mansilla (Sensors 2021), "QoS provision in fog computing: network-aware scheduling of containers" (IPA)

### Takeaway
IPA (IPerf Algorithm) is a Python extension to the Kubernetes scheduler. It measures each node's bandwidth and RTT with iperf every 30 s and works in two steps. Filter: keep nodes whose mean historical completion time for this pod type is within the job's `activeDeadlineSeconds`. Rank: a weighted sum of normalized bandwidth, RTT, CPU, memory and same-type pod count. Jobs no node can meet are rejected (admission control). On a 1-master, 4-worker Kind cluster with 200 bandwidth-heavy speedtest jobs, the IPA-E configuration was the only one to meet 100% of accepted jobs' deadlines at 120 s, at least 30% better than the baselines. The paper's main finding is that network status alone is not enough; it must be combined with load and completion-time prediction.

### Cited Findings
- **Citation:** Caminero A.C., Muñoz-Mansilla R., "Quality of Service Provision in Fog Computing: Network-Aware Scheduling of Containers," *Sensors* 2021, 21(12), 3978; published 9 June 2021; DOI 10.3390/s21123978 — [Sensors DOI](https://doi.org/10.3390/s21123978)
- **Problem:** IoT and smart-city workloads have deadlines that the cloud cannot meet. Prior fog schedulers either ignore network status, never reject jobs that cannot meet their deadline, or were evaluated only in simulation (iFogSim, CloudSim, MATLAB) — [Sensors](https://doi.org/10.3390/s21123978)
- **System model and implementation:**
  - Threads: Scheduler, Worker, Algorithm, Statistics, connected by queues (ActiveQueue, ScheduledQueue, UnscheduledQueue with retries).
  - An iperf server container on each worker node, and a central prober every 30 s (configurable) that writes bandwidth and RTT into node metadata through the K8s API.
  - Each pod picks its algorithm with the annotation `scheduler/algorithm`.
  - New algorithms subclass `Algorithm.evaluate()`.
  - [Sensors](https://doi.org/10.3390/s21123978)
- **Method:**
  - Filter (Algorithm 2): drop nodes whose CompletionTimeEstimation, the mean of past completion times of this pod on this node (Algorithm 4), is above the deadline.
  - Rank (Algorithm 3): quality = Σ weight × normalized metric. Bandwidth and RTT are min-max normalized against the running min and max; CPU and memory as 1 − utilization; pods as 1 − n/MAX_PODS.
  - At least 10 completed jobs are needed before predictions are used; until then, jobs are placed on random nodes.
  - Complexity: filter O(n) in nodes, estimate O(n) in samples (could be O(1) with a running sum). NAS is described as about n·O²(n) because of recursion.
  - [Sensors](https://doi.org/10.3390/s21123978)
- **Evaluation setup:**
  - Kind cluster with 1 master and 4 workers on one physical server.
  - 200 speedtest jobs (TEST_NUMBERS=2), one every 3 s, deadlines of 60, 100 and 120 s.
  - Baselines: NAS (Network Aware Scheduler: filter by minimum bandwidth, choose lowest RTT) and CA-Net (a contention-aware CPU/disk labelling scheduler, modified with a network tag).
  - Weight configurations (bandwidth / pods / RTT / CPU / memory): IPA-O 50/30/10/5/5; IPA-N 100/0/0/0/0; IPA-B 70/30/0/0/0; IPA-E 50/50/0/0/0.
  - [Sensors](https://doi.org/10.3390/s21123978)
- **Numbers:**
  - Acceptance: CA-Net and NAS accept everything (no admission control), and were 52–64% "better" in acceptance than IPA at 60 s, 18–42% at 100 s, and 7% better than IPA-O at 120 s.
  - Successful out of submitted: CA-Net and NAS at most 65% (at 120 s); NAS 27–48% worse than IPA at 100 s and 50–65% worse at 120 s.
  - Successful out of accepted: at 60 s everyone does poorly (best is IPA-E at under 10%). At 100 s, IPA-O and IPA-E are 26–39% better than the other IPA configurations, more than 35% better than CA-Net and more than 70% better than NAS. At 120 s, IPA-O, IPA-N and IPA-B are about 15% better than CA-Net and almost 50% better than NAS, and IPA-E reaches 100%.
  - IPA-N (network only) missed the deadline for about 15% of jobs.
  - IPA-E beats CA-Net by more than 30% with a similar per-node job distribution.
  - [Sensors](https://doi.org/10.3390/s21123978)
- **Limitations (stated or evident):**
  - Stale network view: at most 30 s old, and the probe itself consumes bandwidth (a trade-off between monitoring interval and accuracy).
  - Cold start: the first 10 jobs are placed randomly.
  - The estimator is a plain mean per (pod, node).
  - Emulated cluster on one host, a single application type, no real network heterogeneity.
  - Future work: autonomic prediction, scheduling in advance, loss/latency/jitter metrics, CPU and memory profiles.
  - [Sensors](https://doi.org/10.3390/s21123978)

### Inferences
- **Relation to CROWDio, parallels:**
  - IPA's rank step (a weighted sum over normalized bandwidth, RTT, CPU, memory and load) is structurally CROWDio's MCDM ranking, except that IPA uses hand-set weights where CROWDio uses entropy-derived weights with EDAS, ARAS or MABAC.
  - IPA's 30 s iperf probing matches CROWDio's 30 s worker telemetry.
- **Relation to CROWDio, CROWDio is stronger on:**
  - Bandwidth handling: CROWDio models bandwidth in transfer-time units (bytes/downlink, plus a max-min fair share of the shared medium). IPA uses a dimensionless normalized score, so it cannot tell a 10 MB job from a 1 KB job.
  - Freshness: CROWDio's EWMA telemetry is more responsive than IPA's plain historical mean.
- **Relation to CROWDio, what it could adopt:**
  - Deadline-based admission control, i.e. rejecting or deferring jobs whose predicted completion exceeds a deadline. CROWDio has priority scheduling but no admission control.
  - IPA's evidence that network-only scheduling (IPA-N, NAS) fails is a useful citation to justify CROWDio combining network cost with compute and residency, rather than a pure bandwidth scheduler.
- Caveat for citing: the "optimal" claim (IPA-E reaching 100%) applies to one deadline (120 s) on one emulated cluster with one job type.

### Gaps
- The figure values (Figs. 4–7) are given only as relative percentages in the text, so absolute acceptance and success rates per configuration are not recoverable from the extract.

---

## Q6. Fang, Yang, Wang, "Energy-efficient context-aware multimodal AI inference at the edge"

### Takeaway
A single-device runtime controller. It decides per window which modality encoders (visual, audio, sensor) to run, at what precision (INT4, INT8 or FP16) and with which fusion mode, and audits skipped modalities speculatively, using a small MLP (fewer than 0.18 M parameters) with hysteresis. On a Jetson Nano and a Raspberry Pi 5 (VQAv2, MMBench, a streaming perception task) it reports 42.3% lower energy (1.04 → 0.60 J), 112 → 52 ms latency, 2.3× throughput and at most 1.2% accuracy loss, with less than 9 ms and 0.32 W controller overhead. The venue appears to be a Research Square preprint (not peer-reviewed); treat the numbers accordingly.

### Cited Findings
- **Citation:** Fang X. (Cornell), Yang Y.-H. (University of Minnesota), Wang S. (Brandeis), "Energy-Efficient Context-Aware Multimodal AI Inference at the Edge." No venue, year or DOI appears in the extracted PDF. A Research Square preprint with the identical title and abstract (Jetson Nano and Raspberry Pi 5, 42.3% energy reduction) is at rs-10207311, but its author list and DOI could not be retrieved — [Research Square rs-10207311](https://www.researchsquare.com/article/rs-10207311)
- **Problem:** full-modality multimodal pipelines keep every encoder active whatever the context, which wastes energy on sensing, memory movement and runtime calls. Compression methods (AWQ, DynamicViT, ToMe, EdgeViTs) optimize a single path but never decide whether to run a modality at all — local text of the paper (Zotero item 502)
- **Model:**
  - Utility u_m(t) = αI_m(t) + βC_t − γE_m − δL_m.
  - Activation a_m(t) = 1[u_m ≥ θ_m(p_t)], with the threshold rising as battery falls.
  - Objective: min over activation and precision of λ₁E_tot + λ₂L_tot + λ₃ΔA_task.
  - Solved greedily: rank modalities by information per mJ and take the smallest subset that meets the confidence bound.
  - E_tot = E_ctrl + Σ a_m(E_sense,m + E_enc,m) + E_fuse.
  - Local text (item 502)
- **Method:**
  - An MLP of fewer than 0.18 M parameters trained on leave-one-modality-out pseudo-labels from replayed streams. Features include luminance, Laplacian sharpness, motion, audio RMS and spectral entropy, IMU variance, queue length, battery, CPU temperature and deadline.
  - Hysteresis over consecutive windows.
  - Per-layer precision q_l chosen to minimize |A_l(q)−A_ref| subject to L_l(q) ≤ B_t.
  - Three precompiled paths (full, balanced, emergency); the scheduler does table lookup only.
  - Cross-modal speculative skip (e.g., skip audio when vision confidently shows an empty corridor and audio entropy is low), with periodic audit windows.
  - Two-rate control: context every window, precision and fusion only after stable windows.
  - A missing-modality token for graceful degradation.
  - Local text (item 502)
- **Setup:**
  - Jetson Nano (4 GB, 5/10 W, TensorRT/ONNX) and Raspberry Pi 5 (8 GB, ONNX Runtime), batch size 1, energy measured with an inline USB-C meter or the Jetson rail monitor.
  - Calibration uses 8% of the validation stream (70/15/15 split), then the controller is frozen.
  - Baselines: full modality, static INT8, early exit.
  - Out-of-distribution stress sets: acoustic, low-light, industrial texture, combined.
  - Local text (item 502)
- **Numbers (Table II, energy / latency / throughput / accuracy drop):**

  | Method | Energy | Latency | Throughput | Accuracy drop |
  |---|---|---|---|---|
  | Full | 1.04 J | 112 ms | 1.00× | 0% |
  | INT8 | 0.82 J | 88 ms | 1.32× | 0.6% |
  | Early exit | 0.76 J | 73 ms | 1.47× | 1.0% |
  | Ours-Light | 0.63 J | 58 ms | 1.95× | 1.2% |
  | Ours-Balanced | 0.60 J | 52 ms | 2.30× | 0.8% |

  - Ablation (energy reduction / latency reduction / accuracy drop): without modality sparsity 19.6/23.1/0.4; without dynamic precision 31.7/34.8/0.7; without speculative skip 28.9/39.5/0.6; full framework 42.3/53.6/0.8.
  - Out-of-distribution (energy reduction / accuracy drop): acoustic 36.8%/1.4%; low-light 31.5%/1.7%; texture 34.2%/1.5%; combined 27.9%/2.3%.
  - Battery life more than 72% longer, from a 10,000 mAh power-bank *model* rather than a measurement.
  - Local text (item 502)
- **Limitations stated:** per-workload calibration is needed, and distribution shift erodes the savings — local text (item 502)

### Inferences
- **Relation to CROWDio, differences:** the paper adapts a single device, not a distributed crowd, and has no partitioning, network or scheduling component.
- **Relation to CROWDio, transferable ideas:**
  - Its hysteresis with consecutive-window confirmation and its two-rate control (fast telemetry, slow reconfiguration) parallel CROWDio's break-even switch rule with hysteresis and epochs, and can be cited as independent support for damping reconfiguration on edge hardware.
  - Battery-dependent thresholds θ(p_t) parallel Honeybee-DR's battery tiers, and could make CROWDio's planner more conservative (e.g., avoid assigning heavy shards) on low-battery workers.
  - Precompiled paths with table lookup parallel precomputing candidate cut and placement plans, keeping the decision cost low.
  - Measuring controller overhead separately, both as latency and as incremental power, is a good evaluation practice to copy for CROWDio's planner.
- Weaknesses for citation: preprint status; no confidence intervals or run counts; the battery-extension figure is modelled; Table II reports a single aggregate over three heterogeneous workloads.

### Gaps
- Venue, publication date and DOI could not be confirmed. The preprint page did not render its metadata, and no peer-reviewed version was found.

---

## Q7. Citation audit of the user's paper (arXiv 2605.20723 / Zotero item 481)

### Takeaway
Five problems need fixing before any camera-ready or extended version:
- **Melon:** wrong first author, and its content is misdescribed (it is about on-device *training* memory, not layer offloading).
- **"EdgePipe (Zhao et al., INFOCOM 2022)":** actually Hu et al., arXiv 2021 (Zhao is 3rd author); no INFOCOM venue found; the "43% inter-stage transfer reduction" and "affinity" claims are not in the paper.
- **PipeSwitch:** first author is Bai, not Li, and it does not support the claim it is attached to.
- **"Shah et al., ICDCS 2023":** could not be found and should be treated as unverifiable or nonexistent.
- **CROWDio arXiv self-citation:** the title and author spelling do not match arXiv.

Smaller issues: EDAS year (2015, not 2017), a misspelled Symmetry co-author, an over-stretched Wei et al. citation, the Hinton venue wording, and the PDF footer claiming main-conference ICML/PMLR proceedings for a workshop paper.

### Cited Findings

| Reference | As cited | Verified metadata | Claim accurate? | Fix |
|---|---|---|---|---|
| Melon | Chen, L. et al., MobiSys 2022; "closest prior art — offloads layers within a single device" | Qipeng Wang, Mengwei Xu et al., "Melon: Breaking the Memory Wall for Resource-Efficient On-Device Machine Learning," MobiSys 2022. Combines recomputation and micro-batching for on-device **training**, with a tensor-lifetime-aware memory pool: up to 4.33× larger batch, 1.89× average throughput, up to 49.43% energy saving — [PDF](https://xumengwei.github.io/files/MobiSys22-Melo.pdf); [AMiner](https://aminer.org/pub/634d808890e50fcafd4e4e7f/melon-breaking-the-memory-wall-for-resource-efficient-on-device-machine-learning) | **No.** Wrong first author; it is not layer offloading and not inference. | "Wang et al., 2022". Recast it as "memory-wall techniques for single-device on-device training (recomputation, micro-batching)", or drop "closest prior art". Better closest prior art: EdgePipe (Hu et al.) for multi-device inference partitioning. |
| EdgePipe | Zhao, Z. et al., "EdgePipe: Pipelined deep learning inference on heterogeneous edge clusters," INFOCOM 2022; "affinity-aware placement cuts inter-stage transfer by 43%" (cited twice) | Hu Y., Imes C., **Zhao X.**, Kundu S., Beerel P.A., Crago S.P., Walters J.P.N., "Pipeline Parallelism for Inference on Heterogeneous Edge Computing," arXiv:2110.14895 (Oct 2021). ISI lists it only as an arXiv preprint. Reported results: 10.59×/11.88× speedup with 16 devices (ViT-L/H), 3.93× over 4 nodes, up to 4.16× over PipeDream. Uses a DP partition over heterogeneous compute, memory and bandwidth — [arXiv](https://arxiv.org/abs/2110.14895v1); [ISI](https://www.isi.edu/results/publications/9586/pipeline-parallelism-for-inference-on-heterogeneous-edge-computing/); [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2110.14895) | **No.** Full text has no 43% figure and no affinity, caching or residency discussion; no INFOCOM 2022 venue found; title, first author and venue are all wrong. | Cite as Hu et al., 2021, arXiv:2110.14895. Remove "43%" and "validating this design" in Secs. 2 and 3.5. Reframe as "heterogeneity-aware partitioning across edge devices (Hu et al., 2021) motivates…". Justify the affinity tiers with CROWDio's own warm vs cold load (6 s vs 48 s) instead. |
| PipeSwitch | Li, S. et al., OSDI 2020, pp. 499–514; cited with GPipe for "distribute its layers across multiple memory-constrained devices" | Zhihao **Bai**, Zhen Zhang, Yibo Zhu, Xin Jin, "PipeSwitch: Fast Pipelined Context Switching for Deep Learning Applications," OSDI '20, pp. 499–514. Time-shares **one GPU**, pipelining model transfer over PCIe layer by layer — [JHU PDF](https://cs.jhu.edu/~bzh1994/files/OSDI20_PipeSwitch.pdf); [USENIX speaker page](https://www.usenix.org/conference/osdi20/speaker-or-organizer/xin-jin-johns-hopkins-university) | **No** for the multi-device claim; venue and pages are correct. | "Bai et al., 2020". Move it to where it fits: JIT loading or Tier-4 evict-and-reload (pipelined model loading to cut switching latency). For the multi-device sentence, cite GPipe and Hu et al. 2021. |
| Shah et al. 2023 | "Edge-based compressed feature transmission for split inference," ICDCS 2023 | **Not found** in searches of ICDCS 2023, split inference and compressed-feature topics. Related real work: FrankenSplit (arXiv 2302.10681), SC2 Benchmark (arXiv 2203.08875), an NSF-hosted 2023 scalable feature compression workshop paper — [search results: FrankenSplit](https://arxiv.org/pdf/2302.10681); [SC2](https://arxiv.org/pdf/2203.08875) | **Unverifiable**, likely a non-existent reference. | Remove it, or replace with a verified feature-compression-for-split-computing paper, e.g., Matsubara et al.'s SC2 benchmark (check the venue first) or FrankenSplit. |
| Nagesh et al. 2025 | Nagesh, Fernando, Loke, Neiat, Pathirana, "A dependency-aware task stealing framework for mobile crowd computing," Future Internet 17(10):446, 2025; "DAG work-stealing that informs our dependency manager" | Confirmed: Future Internet 17(10), article 446, Oct 2025, DOI 10.3390/fi17100446 (Honeybee-Tx: capability-aware selection, static dependency-aware stealing, staged transfers) — [MDPI](https://www.mdpi.com/1999-5903/17/10/446); [RePEc](https://ideas.repec.org/a/gam/jftint/v17y2025i10p446-d1761063.html) | **Yes.** | Add the DOI. Consider also citing Honeybee-DR (Sci. Rep. 2026, DOI 10.1038/s41598-026-61849-0) for dynamic DAGs and trap-job validation. |
| Pramanik, Pal, Choudhury 2024 | "Mobile crowd computing: potential, architecture, requirements, challenges, and applications," J. Supercomputing 80(2):96, 2024 | J. Supercomput. 80(2), **pp. 2223–2318** (2024), DOI 10.1007/s11227-023-05545-0; online 29 Jul 2023 — [OUCI](https://ouci.dntb.gov.ua/works/4EVkYrql); [Bohrium](https://www.bohrium.com/paper-details/mobile-crowd-computing-potential-architecture-requirements-challenges-and-applications/896339375531491942-2679) | Authors, title, volume and issue correct; "96" is not the page range. | Use 80(2):2223–2318 and add the DOI. |
| Pramanik et al. 2021 (Symmetry) | Pramanik, Biswas S., Pal, **Marković** D., Choudhury, Symmetry 13(9):1713, 2021; used for entropy weighting | Pramanik, Biswas, Pal, **Dragan Marinković**, Choudhury; Symmetry 13(9):1713, 16 Sep 2021, DOI 10.3390/sym13091713. Compares EDAS, ARAS, MABAC, MARCOS and COPRAS with entropy weights for MCdC resource selection — [OpenAlex](https://api.openalex.org/works/doi:10.3390/SYM13091713); [DOAJ](https://doaj.org/article/6902872a568b4fb1aa1466fa79bea763) | **Yes** for the claim; co-author name is misspelled. | Change "Marković" to "Marinković"; add the DOI. |
| Pramanik & Biswas 2024 | "Energy-efficiency analysis of different scheduling algorithms for mobile crowd computing," ICCCNT 2024 (15th), IEEE | Cited this way by later Springer work: 15th ICCCNT 2024, Kamand, India. Benchmarks 7 algorithms (PSO, GA, MinMin, MaxMin, …). The IEEE Xplore page was not retrieved — [search summary citing Springer/ResearchGate](https://link.springer.com/article/10.1007/s11277-026-11941-5) | Probably correct; DOI and pages not confirmed. | Add the IEEE DOI from Xplore. |
| CROWDio (self) | Manamperi, Pathirana D., Pathirana T., Premarathna, **Gunasekara**, "Crowdio: A developer-friendly mobile crowd computing framework with adaptive scheduling, fault tolerance, and checkpointing," arXiv:2604.19363, 2026 | arXiv title: "**CROWDio: A Practical Mobile Crowd Computing Framework with Developer-Oriented Design, Adaptive Scheduling, and Fault Resilience**"; authors Manamperi, D. Pathirana, T. Pathirana, Premarathna, **Kutila Gunasekera**; v1 21 Apr 2026 — [arXiv](https://arxiv.org/abs/2604.19363) | ID correct; **title and surname wrong**. | Use the exact arXiv title; spell "Gunasekera"; capitalize "CROWDio". |
| Wei et al. 2013 | J. Appl. Math. 2013:409539; cited for "barrier mode … amplifying straggler effects on heterogeneous hardware" | Xianglin Wei, Jianhua Fan, Ziyi Lu, Ke Ding, "Application Scheduling in Mobile Cloud Computing with Load Balancing," J. Appl. Math. 2013, DOI 10.1155/2013/409539. A Hybrid Local Mobile Cloud Model and a Hybrid Ant Colony scheduler, validated in simulation — [RePEc](https://ideas.repec.org/a/hin/jnljam/409539.html); [Project Euclid](https://projecteuclid.org/euclid.jam/1394807550) | Metadata correct; **weak support**. The paper is about ant-colony load balancing, not barrier stragglers. | Replace with a straggler or barrier reference (e.g., PipeDream's micro-batching rationale, already cited), or soften the wording. |
| Misco | Dou, A. et al., PETRA 2010 | Adam Ji Dou, Vana Kalogeraki, Dimitrios Gunopulos, Taneli Mielikäinen, Ville H. Tuulos, PETRA 2010, Samos, DOI 10.1145/1839294.1839332 — [researchr](https://researchr.org/publication/DouKGMT10/bibtex); [Unpaywall](https://unpaywall.org/10.1145%2F1839294.1839332) | **Yes.** | Add the DOI. |
| Paper header | "Proceedings of the 43rd ICML, Seoul, PMLR 306, 2026" | arXiv comments: "Accepted at the ICML 2026 Workshop on Machine Learning for the Global South"; arXiv lists 5 authors in a different order from the PDF's 4 — [arXiv 2605.20723](https://arxiv.org/abs/2605.20723) | **Misleading** venue footer. | Use the workshop footer or style option; align the author list with arXiv. |

### Inferences
- **EDAS (Ghorabaee et al.):** "Multi-criteria inventory classification using a new method of evaluation based on distance from average solution (EDAS)," *Informatica* 26(3):435–451 is, to my knowledge, from **2015**, not 2017. Not re-verified online in this session; check it on the publisher site.
- **Hinton et al.:** the "Distilling the knowledge…" paper came from the NIPS 2014 Deep Learning Workshop (arXiv 1503.02531, 2015). "NeurIPS Deep Learning Workshop, 2015" is slightly off. Not re-verified this session.
- **Sanh et al. (2019):** used for "2–4 GB ONNX session RAM". DistilBERT does not report runtime memory of that kind (see Q2). Either measure it or remove the attribution.
- **References that look correct (bibliographic knowledge, not individually re-checked online):** GPipe (NeurIPS 2019), PipeDream (SOSP 2019), Neurosurgeon (ASPLOS 2017, pp. 615–629), SPINN (MobiCom 2020), Jacob et al. (CVPR 2018, pp. 2704–2713), Deep Compression (ICLR 2016), Ray (OSDI 2018, pp. 561–577), RFC 6455, MABAC (ESWA 42(6), 2015), ARAS (TEDE 16(2), 2010), Shannon (BSTJ 27(3), 1948), Hyrax (CMU-CS-09-164).
- **Pattern:** the three misattributed systems references (Melon, EdgePipe, PipeSwitch) and the unfindable Shah paper all sit in the related-work and motivation sentences of Sections 2 and 3.5. That pattern is typical of LLM-assisted drafting. Re-check every system-paper claim against its PDF before resubmitting.

### Gaps
- A DBLP lookup was blocked by an anti-bot page; author lists were confirmed through OpenAlex, arXiv and publisher mirrors instead.
- No IEEE Xplore DOI was retrieved for Pramanik & Biswas (ICCCNT 2024).
- Shah et al. was searched under several phrasings with no match, but the ICDCS 2023 proceedings table of contents was not checked exhaustively. "Not found" is high-confidence, not certain.
