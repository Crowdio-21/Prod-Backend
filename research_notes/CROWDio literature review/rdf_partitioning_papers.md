# DNN Partitioning / Device-Edge Co-Inference: Full-Text Close Reading of 7 Papers for the CROWDio Literature Review

Scope note: every paper was read in full from the extracted body text (reference lists stripped). For the survey (Hao et al.), the cited-work numbers were resolved against the full text that still includes references (`plain/544.txt`). Numbers are copied exactly from the text. Anything I inferred is labelled **[Inference]**. Where the extracted PDF text was garbled (mostly tables and figures), I say so. The only web lookups were two citation-metadata checks (DDS and AMP4EC), noted where used.

Source keys used below:
- [Neurosurgeon] — https://doi.org/10.1145/3037697.3037698
- [Edgent] — https://doi.org/10.1145/3229556.3229562
- [DDS] — https://arxiv.org/abs/2306.12185
- [DNN-Scissor] — https://doi.org/10.3390/make7040117
- [CrowdNet] — https://doi.org/10.1109/TMC.2025.3588100
- [Hao-Survey] — https://doi.org/10.1145/3786145
- [AMP4EC] — https://arxiv.org/abs/2504.00407

CROWDio facts used for comparison (cut-point and shard planner, single-residency switching cost, 4-tier affinity, 5-phone DistilBERT measurements of ~48 s cold load, ~6 s warm and 34% streaming gain, the offline exporter, and so on) come from the project brief I was given. I did not re-verify them against the code.

---

## Q1. Kang et al. 2017 — Neurosurgeon (ASPLOS '17)

### Takeaway
Neurosurgeon is the founding work on layer-granularity, two-party (mobile ↔ cloud) DNN partitioning. It predicts per-layer latency and power with per-layer-type regression models and picks one split point per query, based on current uplink bandwidth and server load. It reports 3.1× average (up to 40.7×) latency speedup and 59.5% average (up to 94.7%) mobile energy reduction against cloud-only. Both sides hold the complete model, so moving the split point costs nothing in its model.

### Cited Findings
**Citation.** Yiping Kang, Johann Hauswald, Cao Gao, Austin Rovinski, Trevor Mudge, Jason Mars, Lingjia Tang. "Neurosurgeon: Collaborative Intelligence Between the Cloud and Mobile Edge." ASPLOS '17, April 8–12, 2017, Xi'an, China, pp. 615–629. DOI 10.1145/3037697.3037698 — [Neurosurgeon](https://doi.org/10.1145/3037697.3037698)

**Problem**
- The status quo for intelligent personal assistants is cloud-only. The paper asks how feasible mobile execution is, when the wireless upload cost makes cloud processing unjustified, and what role the mobile edge should play — [Neurosurgeon](https://doi.org/10.1145/3037697.3037698) §1.

**System model and assumptions**
- Two parties: one mobile device and one datacenter server. Exactly one partition point: layers before it run on the mobile and layers after it run in the cloud. "Note that there is exactly one partition point within the DNN for which information is sent from the mobile device to the cloud" — [Neurosurgeon](https://doi.org/10.1145/3037697.3037698) §5.3.
- **Full model replication.** "both NSmobile and NSserver host complete DNN models, and partition points are enforced by NSmobile and NSserver runtime" — [Neurosurgeon](https://doi.org/10.1145/3037697.3037698) §5.3.
- Objective: minimise either end-to-end latency or mobile energy (Algorithm 1). Inputs are N layers, per-layer data sizes D_i, regression models f and g, current datacenter load K, uplink bandwidth B, and uplink power P_U. Latency for split j = Σ_{i≤j} TM_i + Σ_{k>j} TC_k + TU_j, where TU_j = D_j/B. Energy for split j = Σ TM_i·PM_i + TU_j·P_U — [Neurosurgeon](https://doi.org/10.1145/3037697.3037698) §5.2.
- The model includes no download/return term for the result, no model-load term and no queueing term — [Neurosurgeon](https://doi.org/10.1145/3037697.3037698) Algorithm 1.

**Method**
- Deployment phase: a one-time profiling per mobile/server hardware pair builds regression models per layer type (conv, local, pool, fc, activation, norm, softmax, argmax). Each is logarithmic or linear in GFLOPS, so "per-application profiling is not needed" — [Neurosurgeon](https://doi.org/10.1145/3037697.3037698) §5, §5.1.
- Regression variables: for conv, the number of input features and (filter size/stride)²×(#filters); for local and pool, input and output feature map sizes; for fc, input and output neurons; for activation and norm, the number of neurons — [Neurosurgeon](https://doi.org/10.1145/3037697.3037698) §5.1.
- Runtime: extract layer configs → predict per-layer mobile and cloud latency and mobile power → choose the argmin split using current bandwidth and server load. Server load is "obtained by periodically pinging the server during idle period" — [Neurosurgeon](https://doi.org/10.1145/3037697.3037698) §5, §6.5.
- Prototype: modified Caffe instances (NSmobile and NSserver) with a Thrift RPC interface — [Neurosurgeon](https://doi.org/10.1145/3037697.3037698) §5.3.

**Evaluation setup**
- Mobile: NVIDIA Jetson TK1 (Tegra K1, 4-Plus-1 quad-core ARM Cortex A15, 2 GB DDR3L, Kepler GPU with 192 CUDA cores). Server: 2× Intel Xeon E5-2620 V2 plus NVIDIA Tesla K40. Software: Caffe with OpenBLAS on the mobile CPU and cuDNN on the GPUs — [Neurosurgeon](https://doi.org/10.1145/3037697.3037698) §3.1, Tables 1–2.
- Eight DNNs (Table 3): AlexNet/IMC (24 layers), VGG (46), DeepFace (10), MNIST/DIG (9), Kaldi/ASR (13), and SENNA POS/NER/CHK (3 each). Networks: 3G, LTE and Wi-Fi bandwidths measured with TestMyNet. Energy measured with a Watts Up? meter — [Neurosurgeon](https://doi.org/10.1145/3037697.3037698) §3.1–3.2, §4.4.
- 48 configurations (8 benchmarks × 3 networks × mobile CPU/GPU). Baselines: status quo (cloud-only) and MAUI. Datacenter throughput was simulated with BigHouse — [Neurosurgeon](https://doi.org/10.1145/3037697.3037698) §6, §6.6.

**Key quantitative results**
- AlexNet, 152 KB image. Upload: 3G "over 870ms", LTE 180 ms, Wi-Fi 95 ms. Compute: mobile CPU 382 ms, mobile GPU 81 ms, cloud GPU 6 ms. The status quo spends "less than 6%" of its time computing and "over 94%" transferring — [Neurosurgeon](https://doi.org/10.1145/3037697.3037698) §3.2.
- In AlexNet, fc6 takes 45% of execution time and pooling cuts data size "by up to 4.7×". With mobile GPU and Wi-Fi, the best split is pool5/fc6, a 2.0× improvement over cloud-only. The best-energy split is 18% more efficient than cloud-only — [Neurosurgeon](https://doi.org/10.1145/3037697.3037698) §4.2–4.3.
- Optimal split chosen in 44/48 configurations for latency, reaching 98.5% of optimal speedup. Also 44/48 for energy, reaching 98.8% of optimal reduction. The suboptimal energy picks still use 24.2% less energy than the status quo — [Neurosurgeon](https://doi.org/10.1145/3037697.3037698) §6.1–6.2.
- Headline: latency 3.1× average (geometric mean), up to 40.7×. Mobile energy −59.5% average, up to −94.7%. Datacenter throughput 1.5× average, up to 6.7× — [Neurosurgeon](https://doi.org/10.1145/3037697.3037698) Abstract, §8.
- Throughput by network: Wi-Fi 1.04×, LTE 1.43×, 3G 2.36× — [Neurosurgeon](https://doi.org/10.1145/3037697.3037698) §6.6.
- Against MAUI: 1.9× average, up to 32×. With MAUI on VGG, mobile CPU and LTE, a misprediction offloads before conv1.2 (12.25 MB input), giving a 20.5× slowdown — [Neurosurgeon](https://doi.org/10.1145/3037697.3037698) §6.3.
- Server load sweep: status-quo latency rises from 105 ms to 753 ms. Neurosurgeon keeps latency below 380 ms by moving from remote to partitioned to local execution — [Neurosurgeon](https://doi.org/10.1145/3037697.3037698) §6.5.
- Under real T-Mobile LTE bandwidth variation, Neurosurgeon switches between local, remote and partitioned execution to keep latency consistent — [Neurosurgeon](https://doi.org/10.1145/3037697.3037698) §6.4.

**Stated limitations**
- There is no dedicated limitations section. The only acknowledged weakness is that mispredictions happen when "partition points and its associated performance are very close" — [Neurosurgeon](https://doi.org/10.1145/3037697.3037698) §6.1.

### Inferences
- **[Inference] Relation to CROWDio.** CROWDio inherits Neurosurgeon's core insight: choose cut points from per-layer compute and activation size together with live bandwidth. CROWDio's bandwidth-aware cost model (rtt + bytes/downlink + compute + result/uplink) generalises Algorithm 1's TM + TU + TC by adding RTT and a result-return term, which Neurosurgeon omits. CROWDio departs from Neurosurgeon in four ways. (a) Neurosurgeon assumes full model replication on both sides, so repartitioning per query is free. CROWDio's single residency means a cut change forces unload + fetch + load + init (~48 s cold vs ~6 s warm on CROWDio's phones), which is why CROWDio needs the J(new)+margin < J(current) test, hysteresis and cooldown that Neurosurgeon has no reason to include. (b) Neurosurgeon has 2 parties and 1 cut; CROWDio places 3+ stages across several phones. (c) Neurosurgeon's server is reliable and always present; CROWDio's volunteer workers churn. (d) Neurosurgeon profiles offline with regression models per layer type; CROWDio uses EWMA telemetry measured live. Neurosurgeon's "periodic ping for server load" is a simple precursor of CROWDio's telemetry loop.

### Gaps
- Neurosurgeon reports no energy or latency cost of loading the model, and no per-switch cost. It does not say how often the split changed in the LTE trace (Fig. 14 is qualitative).

---

## Q2. Li, Zhou, Chen 2018 — Edgent (SIGCOMM MECOMM '18)

### Takeaway
Edgent jointly chooses a device–edge split point and an early-exit point (BranchyNet) to maximise accuracy under a latency deadline, using per-layer regression models and linear search. It is the only one of the seven papers whose latency model includes a **model-loading-time** term (a regression on model size). It is a 6-page workshop prototype tested on a Raspberry Pi 3 and a desktop PC with AlexNet/Cifar-10.

### Cited Findings
**Citation.** En Li, Zhi Zhou, Xu Chen. "Edge Intelligence: On-Demand Deep Learning Model Co-Inference with Device-Edge Synergy." MECOMM'18: ACM SIGCOMM 2018 Workshop on Mobile Edge Communications, August 20, 2018, Budapest, Hungary, 6 pages. DOI 10.1145/3229556.3229562 — [Edgent](https://doi.org/10.1145/3229556.3229562). The extended journal version is Li, Zeng, Zhou, Chen, "Edge AI: On-demand accelerating deep neural network inference via edge computing," IEEE TWC 19(1), 2019, cited as [60] in [Hao-Survey](https://doi.org/10.1145/3786145).

**Problem**
- Device-only AlexNet on a Raspberry Pi takes "more than 2s". Edge-only latency is dominated by transmission: "as the bandwidth drops from 1Mbps to 50Kbps, the latency of edge-based DNN inference climbs from 0.123s to 2.317s" (edge compute stays at 10 ms). Partitioning alone may still miss mission-critical deadlines — [Edgent](https://doi.org/10.1145/3229556.3229562) §1, §2.2.
- Problem statement: "given a predefined latency requirement, how to jointly optimize the decisions of DNN partitioning and right-sizing, in order to maximize DNN inference accuracy" — [Edgent](https://doi.org/10.1145/3229556.3229562) §2.3.

**System model and assumptions**
- One mobile device and one edge server, with one split point and one exit point. Latency only; energy is deferred: "in this paper we focus on the execution latency issue. We will also consider the energy efficiency issue in a future study" — [Edgent](https://doi.org/10.1145/3229556.3229562) footnote 1.
- Three stages: offline training (profiling regressions plus BranchyNet training), online optimisation, and co-inference — [Edgent](https://doi.org/10.1145/3229556.3229562) §3.1.
- **Model loading time is modelled.** "we also observe that the DNN model loading time also has an obvious impact on the overall runtime. Therefore, we further take the DNN model size as a input parameter to predict the model loading time" — [Edgent](https://doi.org/10.1145/3229556.3229562) §3.2. Table 2 regressions (size in bytes, latency in ms): Model Loading on the mobile device is y = 1.33e-6·x + 2.182; on the edge server it is y = 4.49e-6·x + 842.136 — [Edgent](https://doi.org/10.1145/3229556.3229562) Table 2.
- Per-layer regression variables (Table 1): conv uses the amount of input feature maps and (filter size/stride)²·(num filters); relu uses input size; pooling and fc use input and output sizes; LRN and dropout use input size; model loading uses model size — [Edgent](https://doi.org/10.1145/3229556.3229562) Table 1.
- Text inconsistency: §2.3 and §3.1 say the edge server executes the layers *before* the partition point and the device runs the rest ("offload the layers before the partition point to the edge server while running the rest layers on the device"). This is the reverse of Neurosurgeon's convention. I report it verbatim; it may be a writing error — [Edgent](https://doi.org/10.1145/3229556.3229562) §2.3, §3.1.

**Method**
- Algorithm 1: for each exit point i from most accurate (M) down to 1, linearly search partition point p to minimise A_{i,p}. Return the first (largest, most accurate) exit whose best latency is within the target. The search takes "no more than 1ms in our experiments" — [Edgent](https://doi.org/10.1145/3229556.3229562) §3.3.

**Evaluation setup**
- Edge: desktop PC, quad-core Intel 3.4 GHz, 8 GB RAM, Ubuntu. Device: Raspberry Pi 3, quad-core ARM 1.2 GHz, 1 GB RAM. Bandwidth controlled by WonderShaper. Framework: Chainer — [Edgent](https://doi.org/10.1145/3229556.3229562) §4.1.
- Model: BranchyNet AlexNet on Cifar-10 with 5 exits. Sub-model layer counts are 12, 16, 19, 20 and 22 — [Edgent](https://doi.org/10.1145/3229556.3229562) §4.1.
- "We deploy the branchynet model on the edge server and the mobile device", meaning full replication — [Edgent](https://doi.org/10.1145/3229556.3229562) §4.2.
- Experiments: latency requirement fixed at 1000 ms with bandwidth swept from 50 kbps to 1.5 Mbps; bandwidth fixed at 500 kbps with requirement swept from 100 to 1000 ms; accuracy comparison at 400 kbps. Baselines: device-only, edge-only, partition-only — [Edgent](https://doi.org/10.1145/3229556.3229562) §4.2.

**Key quantitative results**
- Higher bandwidth → higher chosen exit point (higher accuracy). The regression-predicted latency tracks the measured latency closely (Fig. 6b). At a 100 ms requirement no method meets the deadline. At 200–300 ms, Edgent meets it with a smaller exit while the other methods fail — [Edgent](https://doi.org/10.1145/3229556.3229562) §4.2, Fig. 7. The text gives no exact accuracy values for Fig. 7; the y-axis spans roughly 0.70–0.80.

**Stated limitations**
- Latency only, energy left to future work. Described as an "initial design" and a "preliminary implementation" — [Edgent](https://doi.org/10.1145/3229556.3229562) §3, §4, §5.

### Inferences
- **[Inference] Relation to CROWDio.** Edgent is the closest early precedent for treating model load time as a latency cost, and it supports CROWDio's choice to make loading first-order. Edgent fits a linear model of load time against model size with an ~842 ms intercept on the edge side. CROWDio measures ~48 s cold and ~6 s warm loads on phones, an order of magnitude larger relative to inference time. Edgent still does not price a *switch* (unload → fetch → load → init) or use residency or caching tiers, and it replicates the full model on both sides. Edgent's deadline-constrained objective (meet the deadline, then maximise accuracy) parallels CROWDio's declarative `submit_job(model_id, objective, deadline)`. CROWDio partitions losslessly, with no early exit, so its accuracy is fixed. CROWDio's 3-stage, multi-phone placement and churn handling have no counterpart in Edgent.

### Gaps
- Edgent does not report whether the model-load term is charged per inference or once per deployment, nor how it enters Algorithm 1's formula (the extracted equation text is garbled).

---

## Q3. Yang et al. 2023 — Adaptive DNN Surgery for Selfish Inference Acceleration with On-demand Edge Resource (DDS)

### Takeaway
DDS treats many selfish mobile devices sharing one finite MEC server. Each device picks a budget (resource demand) in a non-cooperative game with proportional price inflation, and solves its DAG partition as a min-cut on a "latency graph" given the allocated resources. The authors prove a unique Nash equilibrium (and PPAD-hardness of exact solution) and give a decentralised gradient-with-momentum algorithm. In simulation only (up to 100 devices) it reports ~1.25× over DADS.

### Cited Findings
**Citation.** Xiang Yang, Dezhi Chen, Qi Qi, Jingyu Wang, Haifeng Sun, Jianxin Liao, Song Guo. "Adaptive DNN Surgery for Selfish Inference Acceleration with On-demand Edge Resource." arXiv:2306.12185v1 [cs.GT], 21 Jun 2023 — [DDS](https://arxiv.org/abs/2306.12185). The extracted text shows no peer-reviewed venue (the index terms are template boilerplate: "Article submission, IEEE, IEEEtran..."). A metadata search found only the arXiv/CoRR listing — [arXiv listing](https://arxiv.org/html/2306.12185v1). Cite it as an arXiv preprint.

**Problem**
- Prior DNN surgery (Neurosurgeon, DADS, QDMP) "uniformly assumed a fixed amount of server-provided computing resources for each mobile device, which is challenging to attain in the MEC scenario." Devices are self-interested and "can enter or exit the MEC server's service area unpredictably". DNNs are DAGs, not chains — [DDS](https://arxiv.org/abs/2306.12185) §I.

**System model and assumptions**
- N independent devices D share one MEC server with capacity S. The DNN is a DAG G with partition = edge cut C. The source side runs locally (G_l) and the sink side on the server (G_s) — [DDS](https://arxiv.org/abs/2306.12185) §III-A.
- Latency terms: T_l = α_i·Σ f(v)/E(d_i) (FLOPs over device FLOPS, with a regression coefficient); T_s = α_s·Σ f(v)/g(d_i); T_t = Σ_{e∈C} o(e)/b(d_i). Total T = T_l + T_t + T_s. Cost L = T + λ·χ(g) — [DDS](https://arxiv.org/abs/2306.12185) Eqs. (1)–(4).
- Pricing: base unit price 1. If Σa_j > S, the price rises to Σa/S and allocation g = S·a_i/Σa_j; otherwise g = a_i. Cost becomes L = T(d_i, g_i) + λ·a_i — [DDS](https://arxiv.org/abs/2306.12185) Eqs. (6)–(7).
- No model-loading, memory, energy or churn cost terms appear in the formulation — [DDS](https://arxiv.org/abs/2306.12185) §III. **[Inference]** The server must hold every device's full model, since devices upload only "the intermediate features and partition strategies".

**Method**
- Latency-graph construction: edge capacities are o(e)/b. A multi-successor vertex v is split into (v, v′) to avoid double-counting transmission. Virtual vertices l (local), s (server), i (raw input), o (output) are added, with infinite-capacity edges to force raw data to start on-device and results to return. Edges (l, v_i) get server latency and (v_i, s) get local latency — [DDS](https://arxiv.org/abs/2306.12185) §III-B.
- Theorem 1: the min-cut of the latency graph is a valid cut (no server→local back-edge) — [DDS](https://arxiv.org/abs/2306.12185) §III-B.
- The cost L is non-convex and non-smooth at points where the optimal partition strategy switches (Fig. 4). Lemma 1 gives the optimal a_i in closed form for a fixed price — [DDS](https://arxiv.org/abs/2306.12185) §IV-B.
- Theorem 2: a unique NE exists (Banach fixed point; Lemma 2 needs λ > c/4S²). Theorem 3: the game is PPAD-hard — [DDS](https://arxiv.org/abs/2306.12185) §IV-C, §V-A.
- Algorithm 1: each device does gradient updates on a_i with momentum (Eqs. 11–12), receiving the scalar price A^{t+1} piggy-backed on inference results. "Resource Sniff" periodically grid-searches a_i when a device has fallen to local-only (c_i = 0) — [DDS](https://arxiv.org/abs/2306.12185) §V-B.
- Complexity is O(1) in the number of devices N. Min-cut via Boykov is O((V+E)V²), reduced using cut vertices. They cite ≈0.1 s decision time on a Raspberry Pi (from [7], QDMP) — [DDS](https://arxiv.org/abs/2306.12185) §V-C.

**Evaluation setup**
- Simulation only: "simulated on a machine equipped with an Intel Xeon E5-2660 CPU and 256 GB memory", up to 100 devices. Device bandwidth 5–10 Mbps, device compute 10–20 GFLOPS, MEC server 1.2 TFLOPS. Implementation in PyTorch (DAG conversion modified from [37]) with gRPC — [DDS](https://arxiv.org/abs/2306.12185) §VI-A.
- Models: VGG11, ResNet34, ResNet50, ViT on ImageNet inputs. Table I gives VGG11 15 vertices / 14 edges / 7.63 GFLOPs and ResNet34 55 / 57 / 3.68. The extracted table is jumbled for the others; reading it as ResNet50 73 / 75 / 4.12 and ViT 26 / 32 / 3.47 is **[Inference]** — [DDS](https://arxiv.org/abs/2306.12185) Table I.
- Baselines: Edge Only (EO, which in the paper means local-only on the device), Server Only (SO), and DADS (min-cut with fixed resources). Each baseline device is given S/N resources — [DDS](https://arxiv.org/abs/2306.12185) §VI-A.

**Key quantitative results**
- Convergence: with 100 devices, all initial values reach the same NE price A = 1.09. It takes "fewer than 25 iterations" for a0 = 0 or 0.01S and "within 50 iterations" for 0.05S and 0.1S — [DDS](https://arxiv.org/abs/2306.12185) §VI-B.
- VGG11 local inference takes 0.5–0.8 s and "reduces up to 0.5s" with DDS. ResNet34, ResNet50 and ViT take ~0.3 s locally and request less server resource — [DDS](https://arxiv.org/abs/2306.12185) §VI-C.
- SO latency grows linearly and exceeds EO beyond 65 devices. DADS "degenerates to EO" at 100 devices. DDS is "about 1.25×" better than DADS — [DDS](https://arxiv.org/abs/2306.12185) §VI-D. The abstract says "up to 1.25× acceleration" and the introduction says "improves throughput by 1.25×". The metric wording is inconsistent between sections.

**Stated limitations**
- No limitations section. The authors concede that gradient methods "do not guarantee convergence to the unique NE" — [DDS](https://arxiv.org/abs/2306.12185) §VI-B.

### Inferences
- **[Inference] Relation to CROWDio.** DDS is relevant on two points. (1) Contention on a shared resource changes each tenant's optimal cut. CROWDio meets the same coupling when several jobs compete for the same phones, and its least-loaded, MCDM and Sufferage schedulers address it centrally rather than through a market. (2) Fig. 4 shows that the per-device cost is non-convex and non-smooth exactly where the optimal partition switches. This is the analytic signature of the oscillation risk CROWDio damps with margin, hysteresis and cooldown. DDS itself prices no switching cost. Its "Resource Sniff" periodic re-probing is a crude analogue of CROWDio's re-evaluation epochs. DDS differs from CROWDio in four ways: it uses 2 tiers with a powerful MEC server rather than phone-to-phone multi-stage pipelines; the server is assumed to hold any model; entry and exit of devices are mentioned but not evaluated; and it is simulation-only. Its latency-graph min-cut is the canonical tool for DAG models. CROWDio's exporter enumerates contiguous layer ranges, which implicitly assumes a chain or linearised model. The DDS/DADS min-cut line is the reference if CROWDio extends to DAG cuts.

### Gaps
- The paper gives no wall-clock decision overhead measured on real devices; the 0.1 s figure is borrowed from QDMP. It reports no energy results.

---

## Q4. Lyu et al. 2025 — Learning to Partition (DNN-Scissor)

### Takeaway
DNN-Scissor uses an A2C agent on the local device to choose, per video frame, one of a predefined set of split points. The state is queue lengths, estimated bandwidth, estimated delay and pending data. The aim is to optimise a long-horizon reward (−w·data − delay, or −F on a drop) instead of greedy per-frame cost. On a real testbed (laptop + Jetson TX2, congested campus network), YOLOv3 total delay falls 83% versus greedy and system cost 67% versus greedy. The model is replicated on both nodes, so per-frame switching is free.

### Cited Findings
**Citation.** Yan Lyu, Likai Liu, Xuezhi Wang, Zhiyu Fan, Jinchen Wang, Guanyu Gao. "Learning to Partition: Dynamic Deep Neural Network Model Partitioning for Edge-Assisted Low-Latency Video Analytics." *Machine Learning and Knowledge Extraction* 2025, 7(4), 117. Published 13 October 2025. DOI 10.3390/make7040117 — [DNN-Scissor](https://doi.org/10.3390/make7040117)

**Problem**
- Existing dynamic partitioners are "myopic or greedy", optimising "only for the current frame" and ignoring consequences such as "a long waiting queue in the local device" for later frames — [DNN-Scissor](https://doi.org/10.3390/make7040117) §1, §2.3.

**System model and assumptions**
- Camera → local device → edge (GPU). "The DNN models for video analytics are deployed on both local and edge devices." There are two queues: a local inference queue and an offloading queue — [DNN-Scissor](https://doi.org/10.3390/make7040117) §3.1.
- Candidate split points are a fixed, human-chosen set of valid DAG cuts. YOLOv3 has 8, and some of them cut three parallel branches, producing 3 feature maps — [DNN-Scissor](https://doi.org/10.3390/make7040117) §4.1, Fig. 3, §5.1.5.
- The authors frame the problem as a "Dynamic Graph Cut" with state-dependent costs and a long-term sequential objective, and argue classic min-cut does not fit — [DNN-Scissor](https://doi.org/10.3390/make7040117) §4.2.
- MDP: state s_t = [l_t, c_t, b̂_t, τ̂_t, ô_t] (local queue length, offload queue length, estimated bandwidth, estimated delay, pending data). Action = split location. Reward = −w·o_t − τ_t when the wait m_t ≤ T, otherwise −F. T = 2 s (YOLOv3) or 0.5 s (ResNet18, AlexNet), F = 10, w = 1 by default — [DNN-Scissor](https://doi.org/10.3390/make7040117) §4.3, Eqs. (1)–(4).
- Partitioning is treated as accuracy-neutral and verified: AlexNet's top-1 logit is 9.3985 at every split — [DNN-Scissor](https://doi.org/10.3390/make7040117) §5.2.2, Table 3.

**Method**
- A2C actor and critic, each two FC layers of 16 neurons. Batch 5, discount 0.95, learning rate 0.001, updates every 5 frames from a replay buffer. Decision is O(1) in the size of the target DNN and takes "1–2 milliseconds" — [DNN-Scissor](https://doi.org/10.3390/make7040117) §4.4–4.5, §5.1.2, §5.3.2.
- Training: an episode is 1000 consecutive frames, up to 500 episodes, with results averaged over 5 seeds and the median run deployed — [DNN-Scissor](https://doi.org/10.3390/make7040117) §5.1.3.

**Evaluation setup**
- DAHUA HFW2433M-A-IL camera over RTSP. Local device: Lenovo Ideapad laptop (Core i5, 8 GB). Edge: NVIDIA Jetson TX2 (Pascal GPU, 8 GB). Connected via a router and "a congested campus network". Ubuntu 18.04, Python 3.13, PyTorch 2.3 — [DNN-Scissor](https://doi.org/10.3390/make7040117) §5.1.1.
- Models (Table 1):

  | Model | Size | Params | Local inference | Edge inference |
  |---|---|---|---|---|
  | YOLOv3 | 236 MB | 61.53 M | 4.347 s | 0.759 s |
  | ResNet18 | 44.6 MB | 11.7 M | 0.187 s | 0.5 s |
  | AlexNet | 233 MB | 62.37 M | 0.136 s | 0.616 s |

  Note: the edge is *slower* than the laptop for ResNet18 and AlexNet — [DNN-Scissor](https://doi.org/10.3390/make7040117) Table 1.
- Data: open traffic video, 5146 frames at 1080P. Read intervals 1000/150/150 ms. Baselines: Device Only, Edge Only, Semi-Fixed (split at layer 5, threshold 10), Greedy (Neurosurgeon-style) — [DNN-Scissor](https://doi.org/10.3390/make7040117) §5.1.4–5.1.6.

**Key quantitative results**
- YOLOv3 (Table 2), DNN-Scissor improvement over each baseline:

  | Metric | vs LocalOnly | vs EdgeOnly | vs SemiFixed | vs Greedy |
  |---|---|---|---|---|
  | Total delay | 86% | 25% | 77% | 83% |
  | Drop rate | 56% | −68% | 6% | 30% |
  | System cost | 63% | 42% | 50% | 67% |
  | Transmitted data | NA | 49% | −66% | 26% |

  — [DNN-Scissor](https://doi.org/10.3390/make7040117) Table 2. The −68% drop rate versus EdgeOnly means DNN-Scissor drops more frames than EdgeOnly. The abstract says it is "nearly eliminating frame drops".
- Latency breakdown for YOLOv3 (Table 4). Shallow split (low-w policy): local 260 ms (33%), transmission 145 ms (18%), edge 390 ms (49%), total 795 ms. Deep split (high-w policy): 1150 ms (84%), 48 ms (4%), 165 ms (12%), total 1363 ms — [DNN-Scissor](https://doi.org/10.3390/make7040117) Table 4.
- Fixed-bandwidth tests at 1, 5 and 10 MB/s: DNN-Scissor has the shortest delay and "zero drop rate" — [DNN-Scissor](https://doi.org/10.3390/make7040117) §5.2.3.
- Overheads: pre-processing 5–15 ms, post-processing 2–5 ms. Development effort estimated at 150–190 person-hours — [DNN-Scissor](https://doi.org/10.3390/make7040117) §5.3, §6.5.

**Stated limitations**
- Predefined split points. Static pretrained models with no online model updates. Hard-coded penalties and thresholds. Offline DRL training overhead. The accuracy-neutral assumption may fail with lossy transmission or compression. Single-edge topology only; future work targets "multi-device and multi-edge scenarios" with possible multi-agent RL — [DNN-Scissor](https://doi.org/10.3390/make7040117) Abstract, §8.

### Inferences
- **[Inference] Relation to CROWDio.** DNN-Scissor makes the same argument as CROWDio: per-decision greedy cut choice is myopic and queue state matters. It answers with a learned long-horizon policy. CROWDio answers with an explicit cost-comparison rule (J(new)+margin < J(current), with hysteresis and cooldown) because switching on phones is expensive. DNN-Scissor can change the cut every frame only because the full model sits on both nodes. Its action space has no switching penalty at all. CROWDio's single-residency regime would make per-frame action changes ruinous (~6–48 s per reload versus a 1000 ms frame interval). CROWDio adopts the same principle of a predefined candidate cut set (its exporter pre-generates every contiguous range) and the same accuracy-neutral premise. DNN-Scissor's two-queue model (local and offload) resembles CROWDio's streaming dependencies, where queue build-up decides drain versus resume. It covers one device and one edge, with no churn or memory limits. Its own future-work list names CROWDio's setting (multi-device) as unsolved.

### Gaps
- Fig. 9 and Fig. 10 values are given only as charts. The extracted text has no exact absolute delays per baseline beyond Table 2 percentages and Table 4.

---

## Q5. Chen et al. 2025 — CrowdNet (IEEE TMC)

### Takeaway
CrowdNet is the only paper in the set that targets infrastructure-less **mobile crowds** with device churn, and the only one evaluated on real Android phones (6 devices, volunteers walking, ONNX). It does this by *re-architecting* the model: it trains small "CellNets" that can run standalone or be assembled in arbitrary topologies via transition and aggregation layers, with an A2C policy that picks the topology. It does not losslessly partition a pretrained model. Accuracy is within ~2% of the vanilla models on most NLP tasks, and it is robust to failure probabilities up to 0.8. Reconfiguration takes 0.12 s.

### Cited Findings
**Citation.** Gong Chen, Wenzhong Li, Yuchu Fang, Yi Zhang, Sanglu Lu. "CrowdNet: Adaptive Collaborative Inference for Dynamic Mobile Intelligent Service." *IEEE Transactions on Mobile Computing*, accepted 2025 (author's version; volume and issue not yet assigned in the copy read). DOI 10.1109/TMC.2025.3588100 — [CrowdNet](https://doi.org/10.1109/TMC.2025.3588100)

**Problem**
- Existing D2D approaches "adhere to a fixed computation chain, which can lead to poor service availability in dynamic network topologies with intermittent communication links." Cloud and edge synergy depends on "costly, centralized infrastructure" — [CrowdNet](https://doi.org/10.1109/TMC.2025.3588100) §I.
- Motivation experiments. Early-exit accuracy drops "15%–31%" at depth = 1. Under failure probability 0.8, BranchyNet drops 15.4% and DeeBERT 39%. Deep ensembles are robust but show a "5%–9% accuracy gap" against the vanilla models — [CrowdNet](https://doi.org/10.1109/TMC.2025.3588100) §III-A, §III-B.

**System model and assumptions**
- Goals: standalone operation per device, versatile participation in arbitrary topologies, and scalable integration — [CrowdNet](https://doi.org/10.1109/TMC.2025.3588100) §III-C.
- Graph G = {V, E, P, M}. P is the physical-link adjacency, M is a learned mask, and A = M ⊙ P. An isolated node, e.g. one on a device with "energy depletion or network disruption", is dropped — [CrowdNet](https://doi.org/10.1109/TMC.2025.3588100) §IV-C.
- Convolutional CellNets follow a pyramid design. Transformer CellNets have "only one or two Transformer blocks". Transition layers project features (transposed conv or identity). The aggregation layer averages predecessors — [CrowdNet](https://doi.org/10.1109/TMC.2025.3588100) §IV-B, §IV-D, Eqs. (1)–(4).
- Training happens offline in the cloud. The multi-objective loss is L_total = L_global + λ·L_local + μ·L_assemble with λ = μ = 0.5. Topologies are sampled per epoch. An A2C policy samples the mask M (Bernoulli per edge), with reward equal to a task metric such as accuracy or F1 — [CrowdNet](https://doi.org/10.1109/TMC.2025.3588100) §IV-E, §IV-F, §V-A.
- Online: each CellNet processes independently. On disruption "detected on end devices (e.g., via heartbeat signals), RL-policy can swiftly reconfigure the topology" — [CrowdNet](https://doi.org/10.1109/TMC.2025.3588100) §IV-G.
- Heterogeneity: devices are profiled for memory and compute, and "a simple heuristic" allocates the number of CellNet instances per device. Resource-rich devices can run multiple CellNets sequentially — [CrowdNet](https://doi.org/10.1109/TMC.2025.3588100) §V-G-4.

**Evaluation setup**
- Training on an RTX 3090. 500 rounds per task with random topologies and default failure probability 0.1 — [CrowdNet](https://doi.org/10.1109/TMC.2025.3588100) §V-A.
- Models and data: ResNet-110 and MSDNet on CIFAR-10/100. BERT, DeeBERT and DistilBERT on GLUE (SST-2, RTE, WNLI, MRPC, QNLI). GPT-2 on AG News and IMDB. Baselines: vanilla (centralised upper bound); progressive inference (BranchyNet, MSDNet, DeeBERT-style EE); deep ensembles (SnapShot, FastGeometric, fusion-based) — [CrowdNet](https://doi.org/10.1109/TMC.2025.3588100) §V-A.
- Testbed: six Android devices — Xiaomi 12 (Snapdragon 8 Gen 1, 12 GB), HUAWEI MatePad (Kirin 990, 6 GB), HUAWEI P40 (Kirin 990, 8 GB), Samsung Galaxy S20 (Snapdragon 865, 12 GB), Redmi Pad SE (Snapdragon 680, 8 GB), IQOO Neo9 (Snapdragon 8 Gen 2, 12 GB). Models run in ONNX with two threads per device (receive and infer) over campus Wi-Fi. "six student volunteers ... walking freely within an office building" — [CrowdNet](https://doi.org/10.1109/TMC.2025.3588100) §V-G, Table VII.

**Key quantitative results**
- Standalone CellNets reach 63–66% and CrowdNet 69.92–72.05% (CIFAR-100, Fig. 8a). On NLP, CrowdNet gains up to 9.05% over individual CellNets — [CrowdNet](https://doi.org/10.1109/TMC.2025.3588100) §V-B.
- Table III, top-1 accuracy (CIFAR-10 / CIFAR-100):

  | Model | MFLOPs | CIFAR-10 | CIFAR-100 |
  |---|---|---|---|
  | CrowdNet(5) | 246.9 | 93.45 | 71.4 |
  | ResNet-110* | — | 93.57 | 71.85 |
  | BranchyNet | — | 82.08 | 60.23 |
  | MSDNet | — | 84.55 | 66.47 |
  | FastGeometric | — | 91.43 | 64.97 |
  | Snapshot | — | 92.09 | 69.67 |

  Claimed improvements: +11.37% and +11.17% over BranchyNet, and +6.43% over FastGeometric on CIFAR-100 — [CrowdNet](https://doi.org/10.1109/TMC.2025.3588100) Table III, §V-C.
- Table IV (6L), SST-2: DistilBERT* 91.3, DistilBERT-EE 88.6, Fusion 82.6, CrowdNet 90.1. QNLI: 85.2 / 79.4 / 66.4 / 81.8. Accuracy drop versus vanilla is "less than 2%" on most tasks. CrowdNet is up to 15.54% above Fusion and up to 6.8% above progressive — [CrowdNet](https://doi.org/10.1109/TMC.2025.3588100) Table IV, §V-C.
- Scaling over 2–5 CellNets: up to +7.04% on CIFAR — [CrowdNet](https://doi.org/10.1109/TMC.2025.3588100) §V-D.
- Robustness: up to a 7.23% gap over ensembles on CIFAR-100. On SST-2 at failure probability 0.8, CrowdNet(12L) reaches 84.3%, which is +5.54% over ensemble and +10.9% over progressive — [CrowdNet](https://doi.org/10.1109/TMC.2025.3588100) §V-E.
- RL network: the chosen configuration is 2 hidden layers × 256 neurons, 0.135 M params, 608 KB — [CrowdNet](https://doi.org/10.1109/TMC.2025.3588100) Table VI.
- Testbed latency, batch 1 (Table IX), in seconds as progressive / ensemble / CrowdNet:

  | Network | CIFAR-100 | SST-2 |
  |---|---|---|
  | Campus Wi-Fi | 1.010 / 0.047 / 0.182 | 5.155 / 1.086 / 3.181 |
  | 4G | 0.738 / 0.059 / 0.150 | 4.578 / 0.991 / 2.853 |
  | 5G | 0.534 / 0.057 / 0.143 | 2.642 / 0.600 / 1.660 |

  Progressive inference "requires 1.6–5.5× more time" — [CrowdNet](https://doi.org/10.1109/TMC.2025.3588100) Table IX, §V-G-3.
- Topology reconfiguration after a device failure takes 0.12 s. The authors call this "considerably faster than traditional fault-tolerance via model replication on a dedicated backup device, effectively mitigating the cold inference problem" — [CrowdNet](https://doi.org/10.1109/TMC.2025.3588100) §V-G-3.
- About 20% CPU on the weakest phone. In the 60 s real-time run, Fusion averages up to 14% lower accuracy and DeeBERT is "highly unstable" — [CrowdNet](https://doi.org/10.1109/TMC.2025.3588100) §V-G-1, §V-G-2. Table VIII lists per-device per-inference compute of 0.006–0.056 s and transmission of ~0.04–1.03 s. The column-to-dataset alignment in the extracted table is ambiguous, so treat the per-column assignment with care.

**Stated limitations**
- Higher communication cost than single-model inference. Evaluated only on image classification and NLP. The RL-based topology search needs efficiency improvements — [CrowdNet](https://doi.org/10.1109/TMC.2025.3588100) §VI.

### Inferences
- **[Inference] Relation to CROWDio.** CrowdNet is CROWDio's closest neighbour in setting: phones, a crowd, D2D, churn, ONNX on Android, and DistilBERT/BERT workloads. It takes the opposite design route. CrowdNet trains new modular sub-models that degrade gracefully, trading accuracy (≤2% on most tasks, more on some) for robustness and requiring task-specific offline retraining in the cloud. CROWDio keeps the pretrained model intact (lossless contiguous-range shards with sha256 manifest) and gets robustness through systems mechanisms: checkpoint/resume, drain vs resume, prefetch, peer fetch, and replanning by its EWMA-driven planner. CrowdNet's 0.12 s "reconfiguration" is a policy inference over CellNets that are already resident. It does not model or measure loading a new shard. That load cost is exactly what CROWDio prices (~48 s cold / ~6 s warm), and it is why CROWDio needs a switching-cost-aware planner. CrowdNet's "cold inference problem" remark acknowledges the issue without quantifying it. CrowdNet also has no central coordinator or job lifecycle; CROWDio's Foreman provides both. CrowdNet's per-device capacity heuristic and heartbeat-based failure detection are analogous to CROWDio's telemetry and affinity tiers. CrowdNet's SST-2 latency of 1.66–5.16 s per sample on 6 phones is a useful external reference point for CROWDio's 5-phone DistilBERT pipeline, but the architectures and measurement methods differ.

### Gaps
- No energy or battery measurements despite "energy depletion" being a motivating failure mode. No measured model load or initialisation times. Results under failure on the testbed are qualitative (Fig. 16c).

---

## Q6. Hao et al. 2026 — DNN Partitioning for Cooperative Inference in Edge Intelligence (ACM Computing Surveys)

### Takeaway
The survey provides (a) an architecture taxonomy: one-to-one, multiple-to-one, one-to-multiple, multiple-to-multiple, and peer-to-peer; (b) a unified notation and cost model covering latency (compute + transmission + queue), energy, monetary cost, accuracy and reliability, with constraints C1–C6; (c) a six-level "solution-space" taxonomy of methods; and (d) toolchain tables. It names dynamic re-partitioning under node failure as an open robustness challenge. Its unified model has **no switching or reload cost term**, and its storage constraint C6 assumes nodes "typically need to store the complete DNN model".

### Cited Findings
**Citation.** Yuntao Hao, Nan Ding, Weiguo Xia, Hongwei Ge, Li Xu. "DNN Partitioning for Cooperative Inference in Edge Intelligence: Modeling, Solutions, Toolchains." *ACM Computing Surveys* 58(8), Article 204, February 2026, 34 pages. DOI 10.1145/3786145 — [Hao-Survey](https://doi.org/10.1145/3786145)

**Positioning**
- The survey describes itself as a "problem-driven taxonomy", classifying by formulation (granularity, integration with model optimisation, dynamic adaptation) rather than by objective. It claims coverage of Transformer partitioning and toolchains that earlier surveys lack (Table 1) — [Hao-Survey](https://doi.org/10.1145/3786145) §1.1.

**Taxonomy 1: DNN structure (§2.1)**
- Chain-based (AlexNet, VGG; Transformers are sometimes treated as chains). DAG-based (ResNet, GoogleNet, MobileNet). Topology-agnostic, i.e. intra-layer matrix parallelism — [Hao-Survey](https://doi.org/10.1145/3786145) §2.1.

**Taxonomy 2: collaborative architectures (§2.2, Tables 2–6)**
- One-to-one end-edge: Neurosurgeon [59], DADS [63], LLM splitting [53]; also early exit [60, 61] and compression [67, 68].
- Multiple-to-one: key issues are resource allocation (joint [33, 69–72, 76] vs decoupled [73, 75, 81, 83]) and task offloading [77–80]. Overload is cited as "edge node utilization exceeding 90% [66]".
- One-to-multiple: task offloading [85], mobility [85–88], and reliability via "overlapping DNN partitions ... replicating critical model segments across nodes" [87, 89].
- Multiple-to-multiple: [47, 74, 90–92].
- Peer-to-peer: issues are heterogeneity ([57] CoEdge, [95], [93], LLMs [50–52]), reliability ([96] multi-replica across UAVs), task offloading ([97] chain, [98] DAG), and mobility ([94]).
- All from [Hao-Survey](https://doi.org/10.1145/3786145) §2.2.

**Unified model (§2.3)**
- Parameters are obtained by empirical analysis, predictive modelling, or formula-based estimation. [105] notes that per-layer execution times are stable across runs on the same hardware — [Hao-Survey](https://doi.org/10.1145/3786145) §2.3.
- Latency T = T_com + T_tra + T_que. Transmission may use Shannon capacity or min(uplink, downlink) [80, 97]. The result-return term is "often ignore[d]". Queueing uses M/M/1 or M/D/1. Energy = latency × power. Cost = latency × unit price. Accuracy = f(input size, model, exit). Reliability = product of (1 − node failure probability) and (1 − link failure probability) — [Hao-Survey](https://doi.org/10.1145/3786145) §2.3.
- Constraints: C1 partition feasibility; C2 compute; C3 latency deadline; C4 bandwidth; C5 accuracy; C6 storage. For C6: "Computing nodes typically need to store the complete DNN model [65, 80] along with the intermediate computational data" — [Hao-Survey](https://doi.org/10.1145/3786145) §2.3.
- Problem class: standalone partitioning is an ILP. Adding resource allocation (continuous variables) makes it an MINLP [72] — [Hao-Survey](https://doi.org/10.1145/3786145) §2.3.
- Worked example: MobileNetV2 (~300 MFLOPs) split between a phone (50 GFLOPS) and a server (500 GFLOPS) at 20 Mbps with a 2.5 MB feature map. Compute is 2.4 ms + 0.36 ms, transmission 1 s, M/M/1 queue 0.2 s, total ≈1.203 s. Energy ≈1.5048 J. Cost ≈ $4.18×10⁻⁸ — [Hao-Survey](https://doi.org/10.1145/3786145) §2.3.

**Taxonomy 3: solution space (§3)**
- **3.1 One-dimensional, single-layer two-partitioning** (Table 7). Linear search [59, 66, 76, 79]. Graph min-cut: DADS splits into a light-load DSL case and a heavy-load NP-hard DSH case. [67] reports DADS takes more than 1 minute on InceptionV3 (313 layers) and accelerates it with Dinic. [65] fixes DADS graph flaws with virtual upload vertices. DRL: [53] uses PPO for LLM split points.
- **3.2 Partitioning + model optimisation** (Table 8). Offline bandwidth→(split, exit) table [60]; confidence-based exits [61]; ILP over exit, layer set and compression ratio [62]; partition-then-prune [68].
- **3.3 Partitioning + resource allocation** (Table 9). Decoupled: auction [73], min-cut then game [75], RL + heuristic [81], DP-tree [83]. Joint: IAO [69], DRL [33, 113], game theory [72, 76].
- **3.4 Multi-layer/sub-layer + offloading** (Table 10). Up to 3 split points [109]; OCF game + GA [93]; matching game [84]. "To reduce repartitioning overhead, [88] constructs a graph where vertices represent partition-offloading configurations, and edges connect neighboring decisions" (AdaMEC). Replication [87, 89]. DFS over a layer-node matrix [97]. "[85] extends this by incorporating server-switching delays". Topological sort + greedy grouping [98]. Layer game UCL [114]. PSO-GA, GA, HCEA [115, 99, 107]. DRL [108, 94, 78, 80, 96].
- **3.5 Multi-dimensional** (Table 11). MAMO [70], DDPG [106], model-version + split + resources DRL [71], digital-twin DRL [82], decoupled [90–92], Lyapunov + diffusion multi-agent DRL [47].
- **3.6 Tensor/sub-layer parallelism** (Table 12). MoDNN/MeDNN [117, 118], DeepThings FTP [119], AOFL [120], D3 VSM [116], CoEdge "only neighbor" padding rule [57], neuron rearrangement [100, 121], Transformer block parallelism [123], Hepti WS/1D/2D [58].
- All from [Hao-Survey](https://doi.org/10.1145/3786145) §3.

**Toolchains (§4, Tables 13–16)**
- Frameworks: PyTorch, TensorFlow, Caffe, BranchyNet, Chainer. Datasets: CIFAR, Caltech-256, ImageNet/ILSVRC2012, SeaShip, BDD100K, UCF-101, AG News, GLUE, WikiText-2. Nodes: Xeon/i7/i3 CPUs; V100/2080Ti/K620 GPUs; Raspberry Pi 3B/3B+/4B; Jetson Nano/Xavier NX. Communication: Wi-Fi, Ethernet, ZeroMQ, gRPC. Bandwidth control: WonderShaper, COMCAST, tc, sleep, Belgium 4G/LTE logs. Resources: Docker, stress-ng. Analysis: TF Benchmark, PALEO, LINPACK, thop, torchstat, NetScope — [Hao-Survey](https://doi.org/10.1145/3786145) §4. **[Inference]** No smartphone or Android/ONNX-Runtime-Mobile platform appears in the node or toolchain tables.

**Challenges (§5)**
- Security and privacy: model inversion on activations, MITM, TEEs, DP, HE/MPC overhead.
- **Robustness**: "Failures of computation nodes or changes in system architecture ... often require dynamic re-partitioning without interrupting inference [134]. ... most existing works rely on multi-replica strategies, which ... may introduce significant overhead." "Automating rapid re-partitioning in response to node overload or failure will be vital."
- Large models: pipeline and layer-wise partitioning [52, 58]; memory and attention-map traffic.
- All from [Hao-Survey](https://doi.org/10.1145/3786145) §5.

**Works cited by the survey that bear on CROWDio** (metadata from the survey's reference list in `plain/544.txt`)
- Dynamic repartitioning and downtime:
  - [134] Majeed, Kilpatrick, Spence, Varghese, "NEUKONFIG: Reducing edge service downtime when repartitioning DNNs," IEEE IC2E 2021, 118–125.
  - [146] Majeed et al., "CONTINUER: Maintaining distributed DNN services during edge failures," IEEE EDGE 2022, 143–152.
  - [147] McNamee et al., "The case for adaptive deep neural networks in edge computing," IEEE CLOUD 2021, 43–52.
  - [88] Pang et al., "AdaMEC: Towards a context-adaptive and dynamically combinable DNN deployment framework for mobile edge computing," ACM TOSN 20(1), 2023.
  - [85] Tian et al., "Mobility-included DNN partition offloading from mobile devices to edge clouds," Sensors 21(1):229, 2021 (server-switching delays).
  - [145] Guo et al., "RobustDiCE," ASP-DAC 2024.
- Multi-device and multi-stage partitioning:
  - [57] Zeng et al., "CoEdge," IEEE/ACM ToN 29(2), 2020.
  - [84] Mohammed et al., "Distributed inference acceleration with adaptive DNN partitioning and offloading," INFOCOM 2020.
  - [99] Na et al., GA partitioning on cooperative edge devices, Applied Sciences 12(20), 2022.
  - [95] Miao et al., "Adaptive DNN partition in edge computing environments," ICPADS 2020.
  - [109] Kayal & Leon-Garcia, "DNNSplit," IEEE Access 2024.
  - [121] Hu & Li, INFOCOM 2022.
  - [32] Feltin et al., "DNN partitioning for inference throughput acceleration at the edge," IEEE Access 11, 2023.
- Streaming and pipelines:
  - [97] Gao et al., "OCDST: Offloading chained DNNs for streaming tasks," GLOBECOM 2021.
  - [98] Wu et al., "PDD: Partitioning DAG-topology DNNs for streaming tasks," IEEE IoT-J 2023.
  - [50] Hu et al., "PipeEdge," DSD 2022.
  - [51] Ma et al., "HPipe," NAACL 2024.
  - [52] Zhang et al., "EdgeShard," IEEE IoT-J 2024.
- Transformers on heterogeneous edge devices:
  - [58] Lee et al., "An autonomous parallelization of transformer model inference on heterogeneous edge devices" (Hepti), ICS 2024.
  - [123] Du et al., IEEE TPDS 2024.
- Multi-user contention:
  - [69] Tang et al., IEEE IoT-J 8(12), 2020.
  - [66] Liu et al., "LoadPart," ICDCS 2022.
  - [77] Duan & Wu, IEEE TMC 23(4), 2023.
  - [65] Li et al., throughput maximisation, IEEE TMC 22(5), 2021.
- Reliability via replication:
  - [87] Liu et al., IEEE/ACM ToN 31(6), 2023.
  - [89] Yan et al., "ASPM," ITSC 2023.
  - [96] Zhao et al., UAV swarm, IWCMC 2022.
- All of the above are listed in [Hao-Survey](https://doi.org/10.1145/3786145) References.

### Inferences
- **[Inference] Relation to CROWDio.** In the survey's taxonomy, CROWDio sits in the **peer-to-peer / one-to-multiple** cells with **multi-layer (3+) partitioning plus offloading** (§3.4), under heterogeneity, reliability and mobility challenges, with a centralised coordinator (Foreman). Its joint cut-and-placement planner (exhaustive oracle + Pareto DP) is an algorithm-based joint optimisation. Two places in the survey mark where CROWDio extends the literature. (1) The unified model has latency, energy, cost, accuracy and reliability, but no reconfiguration or switching term. C6 assumes nodes store the complete model, which phones under single residency cannot. CROWDio's switch-cost term (unload + fetch + load + init) and residency constraint fill this gap. (2) §5.2 explicitly calls for "automating rapid re-partitioning in response to node overload or failure", pointing to NEUKONFIG [134] and CONTINUER [146]. CROWDio's epochs, drain/resume, prefetch, peer fetch and checkpoint/resume are a concrete answer for phones. The survey's toolchain section (§4) catalogues measurement and profiling tools but no artefact pipeline. CROWDio's offline exporter (ONNX shards for every contiguous layer range, plus a sha256 manifest) and Android runtime (ONNX Runtime Mobile via Chaquopy) are a toolchain contribution in a category the survey does not list. NEUKONFIG, CONTINUER, AdaMEC and [85] are the most important follow-up reads for CROWDio's switching-cost related work; I did not read them.

### Gaps
- Several equations in the extracted text lost their symbols (notation for nodes, partitions and so on is blank), so I paraphrased formulas from the prose. I did not read the cited works ([134], [146], [88], [85] in particular), so I cannot say how they model repartition cost.

---

## Q7. Zhang et al. 2025 — AMP4EC

### Takeaway
AMP4EC is a Docker-emulated framework (resource monitor, partitioner, weighted-score scheduler, deployer) that splits a PyTorch model (MobileNetV2) into 2–4 contiguous partitions by equal cumulative cost and assigns them to containers. It reports 78.35% lower latency and 414.73% higher throughput than a single-container monolith. The paper has several internal numerical inconsistencies. Its partition boundaries are "fixed after deployment".

### Cited Findings
**Citation.** Guilin Zhang, Wulan Guo, Ziqi Tan, Hailong Jiang. "AMP4EC: Adaptive Model Partitioning Framework for Efficient Deep Learning Inference in Edge Computing Environments." arXiv:2504.00407v2 [cs.DC], 4 Apr 2025 — [AMP4EC](https://arxiv.org/abs/2504.00407). The arXiv record reportedly notes acceptance at FMEC 2025 — [arXiv listing](https://arxiv.org/html/2504.00407v2). I did not verify this against proceedings; cite it as arXiv unless confirmed. The prompt calls it "Zhang et al."; the first author is Guilin Zhang (George Washington University).

**Problem**
- Prior partitioners "rely on pre-partitioned knowledge and lack real-time resource monitoring and adaptation". They cannot integrate a **new device** or handle a **device going offline** — [AMP4EC](https://arxiv.org/abs/2504.00407) §I.

**System model and method**
- (A) Resource Monitor: per-container CPU %, memory, and network I/O via the Docker stats API. "if a device goes offline ... [it is] excluded from consideration" — [AMP4EC](https://arxiv.org/abs/2504.00407) §III-A.
- (B) Model Partitioner, "RALOS". Layer cost is k_h·k_w·C_in·C_out for conv, N_in·N_out for fc, and the parameter count otherwise. Node score S_i = w_cpu·c_i + w_mem·M_i, with share P_i = S_i/ΣS_j. Balance metric L = (1/n)Σ|L_i − L_avg|. Target cost = total/num_partitions. "Layers are sequentially added to a partition until the cumulative cost meets or exceeds the target". Each partition is saved as a separate model file. The authors claim RALOS reduces "end-to-end latency by 27.5%" — [AMP4EC](https://arxiv.org/abs/2504.00407) §III-B, Eqs. (1)–(6).
- (C) Task Scheduler (Algorithm 1, NSA). Skip nodes with load > 0.8 or high latency. Score = 0.2·S_R + 0.2·S_L + 0.1·S_P + 0.5·S_B, where S_B = 1/(1 + TaskCount·2). It keeps a performance-history cache. Complexity O(m·n). Overhead is ~10 ms (<2% of end-to-end latency) at n = 3–5 and 25 ms at n = 10 — [AMP4EC](https://arxiv.org/abs/2504.00407) §III-C.
- (D) Model Deployer: optimisation with TorchScript or quantisation, transfer to the container's model server, undeploy and cleanup — [AMP4EC](https://arxiv.org/abs/2504.00407) §III-D.

**Evaluation setup**
- macOS with Docker Desktop 3.8 (Engine 24.0.7), image pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime, cgroups v2, bridge networks. MobileNetV2, 100 inference iterations per configuration. Batches of 32 requests. Monolithic baseline: one container, 2 cores, 2 GB. 30 s warm-up + 5 min evaluation, repeated 3 times — [AMP4EC](https://arxiv.org/abs/2504.00407) §IV-A, §IV-B.
- Resource profiles (Table II): High 1.0 CPU / 1 GB; Medium 0.6 / 512 MB; Low 0.4 / 512 MB — [AMP4EC](https://arxiv.org/abs/2504.00407) Table II.

**Key quantitative results (as reported)**
- Table I, AMP4EC+Cache / AMP4EC / Monolithic:

  | Metric | AMP4EC+Cache | AMP4EC | Monolithic |
  |---|---|---|---|
  | Latency | 234.56 ms | 605.32 ms | 1082.53 ms |
  | Throughput | 5.07 req/s | 5.01 req/s | 0.96 req/s |
  | Memory | 1.625 MB | 1.625 MB | 0.625 MB |
  | Network bandwidth | 100.0 MB | 100.0 MB | 0 |
  | Stability | 0.95 | 0.95 | 1.0 |
  | Scheduling overhead | 10 ms | 10 ms | 0 |

  Improvement: −78.35% latency, +414.73% throughput — [AMP4EC](https://arxiv.org/abs/2504.00407) Table I.
- Table II average inference time: High 234.56 ms, Medium 389.27 ms, Low 583.91 ms — [AMP4EC](https://arxiv.org/abs/2504.00407) Table II.
- Partition sizes: 2 parts [116, 25]; 3 parts [108, 16, 17] — [AMP4EC](https://arxiv.org/abs/2504.00407) §IV-D.
- Adaptability scenarios: 3 nodes vs a 2-core baseline, 4 nodes vs 3-core, 2 nodes vs 1-core. Text claims "linear performance scaling with up to three edge nodes" — [AMP4EC](https://arxiv.org/abs/2504.00407) §IV-C, §IV-E.

**Internal inconsistencies (verbatim checks)**
- Throughput improvement is "414%" in the abstract and "415%" in the introduction.
- The text says caching reduces "network bandwidth requirements to zero compared to AMP4EC's 100MB", but Table I lists 100.0 MB for *both* AMP4EC+Cache and AMP4EC.
- "Memory efficiency was notable, with AMP4EC+Cache requiring only 1.625MB compared to the monolithic system's 0.625MB". This is higher, not lower (Table I: +160%).
- §IV-E says High and Medium profiles achieved "22-23ms" and Low "40ms". Table II and the introduction say 234.56 / 389.27 / 583.91 ms.
- §III-B claims the partitioner considers communication, but Eq. (6) and its sequential fill rule use computation cost only.
- All from [AMP4EC](https://arxiv.org/abs/2504.00407) Abstract, §I, §III-B, §IV-B, §IV-E, Tables I–II.

**Stated limitations**
- PyTorch only. "partition boundaries are fixed after deployment, limiting adaptability to runtime changes". Docker lacks specialised edge-device support. Monitoring overhead. Network bottleneck. Limited memory restricts scalability. "we also simplified real-world heterogeneity by assuming all distributed devices share the same architecture and operating system" — [AMP4EC](https://arxiv.org/abs/2504.00407) §V.

### Inferences
- **[Inference] Relation to CROWDio.** AMP4EC's architecture (monitor → partitioner → weighted-score scheduler → deployer) mirrors CROWDio's (telemetry → planner → MCDM/least-loaded schedulers → worker shard loading). Its fixed weighted score (0.2/0.2/0.1/0.5) is a hand-tuned counterpart to CROWDio's entropy-weighted MCDM scheduler, which derives weights from data. AMP4EC does not model transmission in its cut choice (Eq. 6), switching cost, or phone constraints. Its "adaptive" claim covers scheduling over a fixed partition, and its own limitations admit boundaries cannot change at runtime. CROWDio's reconfiguration-aware planner addresses exactly this missing piece. AMP4EC's evaluation (Docker containers on one Mac, MobileNetV2, same OS and architecture) is far from CROWDio's 5 real phones. Given the inconsistencies above, cite AMP4EC as an example of the monitor-partition-schedule pattern, not as quantitative evidence.

### Gaps
- No real edge hardware, no network-variation experiment, and no measurement of the device-offline scenario that motivates the paper.

---

## Q8. Cross-paper synthesis: which gaps does CROWDio fill?

### Takeaway
Across the seven papers, no work prices the cost of *changing* a partition plan. Either the model is fully replicated on each node (Neurosurgeon, Edgent, DNN-Scissor; implicitly DDS; survey constraint C6), or boundaries are fixed after deployment (AMP4EC). The one paper on real phones with churn (CrowdNet) avoids repartitioning by retraining a modular architecture and accepting an accuracy cost. CROWDio's distinctive contribution is a lossless, multi-stage (3+) phone pipeline whose planner explicitly prices reconfiguration under single residency and switches only when it pays off, with hysteresis. The survey's own open-challenge list (§5.2) asks for this.

### Cited Findings
**Gap 1 — No switching or reload cost in the decision model.**
- Neurosurgeon: "both NSmobile and NSserver host complete DNN models", so the split can change per query at zero cost — [Neurosurgeon](https://doi.org/10.1145/3037697.3037698) §5.3.
- DNN-Scissor: "deployed on both local and edge devices", with a per-frame action and no switch penalty in the reward — [DNN-Scissor](https://doi.org/10.3390/make7040117) §3.1, Eq. (4).
- Edgent: models model-loading time (Table 2: device y = 1.33e-6·x + 2.182 ms; edge y = 4.49e-6·x + 842.136 ms) but not switching, and replicates the model — [Edgent](https://doi.org/10.1145/3229556.3229562) §3.2, §4.2.
- DDS: shows that cost is non-smooth where the optimal partition switches, but includes no switching cost — [DDS](https://arxiv.org/abs/2306.12185) §IV-B.
- Survey: C6 says nodes "typically need to store the complete DNN model". It points to repartitioning-overhead work only in passing (AdaMEC [88], server-switching delays [85], NEUKONFIG [134]) — [Hao-Survey](https://doi.org/10.1145/3786145) §2.3, §3.4, §5.2.
- AMP4EC: "partition boundaries are fixed after deployment" — [AMP4EC](https://arxiv.org/abs/2504.00407) §V.
- CrowdNet: 0.12 s reconfiguration over already-resident CellNets; mentions the "cold inference problem" without measuring it — [CrowdNet](https://doi.org/10.1109/TMC.2025.3588100) §V-G-3.

**Gap 2 — Lossless multi-stage (>2) partitioning across phones.**
- Neurosurgeon, Edgent, DDS and DNN-Scissor are all two-party with one cut — [Neurosurgeon](https://doi.org/10.1145/3037697.3037698); [Edgent](https://doi.org/10.1145/3229556.3229562); [DDS](https://arxiv.org/abs/2306.12185); [DNN-Scissor](https://doi.org/10.3390/make7040117).
- AMP4EC: 2–4 partitions, but in Docker containers sharing one OS and architecture — [AMP4EC](https://arxiv.org/abs/2504.00407) §IV-D, §V.
- CrowdNet: multi-phone, but with retrained CellNets. It is not a partition of the pretrained model, and accuracy falls slightly short of vanilla (e.g. SST-2 90.1 vs 91.3 for DistilBERT) — [CrowdNet](https://doi.org/10.1109/TMC.2025.3588100) Table IV.
- DNN-Scissor names multi-device as future work — [DNN-Scissor](https://doi.org/10.3390/make7040117) §8.

**Gap 3 — Volunteer churn with state continuity.**
- CrowdNet handles failures via graceful degradation: failure probability 0.1 by default, up to 0.8 in tests — [CrowdNet](https://doi.org/10.1109/TMC.2025.3588100) §V-A, §V-E.
- DDS mentions devices entering and exiting, but this is not evaluated — [DDS](https://arxiv.org/abs/2306.12185) §I.
- AMP4EC motivates the paper with device offline/online events but does not test them — [AMP4EC](https://arxiv.org/abs/2504.00407) §I.
- The survey says reliability work mostly uses multi-replica strategies with "significant overhead" — [Hao-Survey](https://doi.org/10.1145/3786145) §5.2.
- None of the seven implements checkpoint/resume of in-flight work.

**Gap 4 — Memory and residency limits on phones.**
- Only AMP4EC (memory term in its node score) and the survey (C6 storage) model memory, and both treat it as capacity, not as a residency slot — [AMP4EC](https://arxiv.org/abs/2504.00407) Eq. (3); [Hao-Survey](https://doi.org/10.1145/3786145) §2.3.
- CrowdNet profiles memory to decide CellNet counts per device — [CrowdNet](https://doi.org/10.1109/TMC.2025.3588100) §V-G-4.

**Gap 5 — Realistic phone testbeds.**
- Hardware used by the other papers:
  - Neurosurgeon: Jetson TK1 — [Neurosurgeon](https://doi.org/10.1145/3037697.3037698).
  - Edgent: Raspberry Pi 3 — [Edgent](https://doi.org/10.1145/3229556.3229562).
  - DDS: simulation only — [DDS](https://arxiv.org/abs/2306.12185).
  - DNN-Scissor: laptop + Jetson TX2 — [DNN-Scissor](https://doi.org/10.3390/make7040117).
  - AMP4EC: Docker on a Mac — [AMP4EC](https://arxiv.org/abs/2504.00407).
- Only CrowdNet uses phones (6 Android devices) — [CrowdNet](https://doi.org/10.1109/TMC.2025.3588100).
- The survey's device table lists only Raspberry Pi and Jetson — [Hao-Survey](https://doi.org/10.1145/3786145) Table 14.

**Gap 6 — Myopic vs damped adaptation.**
- DNN-Scissor shows greedy per-frame choice is myopic (−83% delay vs Greedy on YOLOv3) and replaces it with a learned long-horizon policy — [DNN-Scissor](https://doi.org/10.3390/make7040117) Table 2.
- Neurosurgeon adapts per query to bandwidth and server load — [Neurosurgeon](https://doi.org/10.1145/3037697.3037698) §6.4–6.5.
- None of the seven uses hysteresis, cooldown or a switch-only-if-beneficial margin.

**Gap 7 — Toolchain and artefacts.**
- Neurosurgeon and Edgent profile with per-layer regressions — [Neurosurgeon](https://doi.org/10.1145/3037697.3037698) §5.1; [Edgent](https://doi.org/10.1145/3229556.3229562) §3.2.
- AMP4EC saves each partition as a separate PyTorch file — [AMP4EC](https://arxiv.org/abs/2504.00407) §III-B.
- CrowdNet converts to ONNX for Android — [CrowdNet](https://doi.org/10.1109/TMC.2025.3588100) §V-G.
- The survey toolchain tables cover frameworks, datasets, hardware, comms, resource control and FLOP counters — [Hao-Survey](https://doi.org/10.1145/3786145) §4.

### Inferences
- **[Inference] Gap 1.** CROWDio's single-residency switching cost (unload + fetch + load + init; measured ~48 s cold vs ~6 s warm, compared with second-scale inference) is the first-order term that this literature leaves out by assuming replication. That assumption becomes infeasible on phones once models are large or several stages share a device. CROWDio's J(new)+margin < J(current) rule, hysteresis and cooldown, plus prefetch and peer fetch to shrink the "fetch" component, are the direct contribution. Edgent's model-loading regression is the closest precedent to cite, and NEUKONFIG, CONTINUER and AdaMEC (via the survey) are the nearest related work to read next.
- **[Inference] Gap 2.** CROWDio occupies the cell "lossless contiguous-range sharding of a pretrained model across ≥3 heterogeneous phones" (DistilBERT, 3 stages, 5 phones). Among these seven, that cell is filled by none.
- **[Inference] Gap 3.** CROWDio's checkpoint/resume, drain vs resume and epochs offer a systems alternative to both replication (overhead, per the survey) and CrowdNet's retraining (accuracy cost and task-specific training).
- **[Inference] Gap 4.** CROWDio's 4-tier affinity (resident > cached on disk > idle > evict) explicitly models the *state* of a device's memory and storage relative to a shard. No paper here does this.
- **[Inference] Gap 5.** CROWDio's 5-phone measurements add to a thin evidence base. CrowdNet is the only peer on phones, so cite it as the comparison point while noting the differences in workload and architecture.
- **[Inference] Gap 6.** CROWDio sits between DNN-Scissor and Neurosurgeon. It is farsighted like DNN-Scissor, but uses an explicit, interpretable cost model with switching cost instead of a learned policy that needs offline training (a limitation DNN-Scissor states). Its stability mechanisms address the oscillation that DDS's non-smooth cost surface implies.
- **[Inference] Gap 7.** CROWDio's exporter (ONNX shards for every contiguous range, sha256 manifest), JIT deferred shard loading, zlib tensor transport and declarative `submit_job` make up a deployment toolchain category that the survey's §4 tables do not list. CROWDio's EWMA telemetry replaces offline per-layer regression profiling (Neurosurgeon, Edgent).
- **[Inference] Shared ideas CROWDio adopts.** Layer-boundary cuts at small activations (Neurosurgeon). Deadline-constrained objectives (Edgent; survey C3). Contention-aware placement (DDS; survey multiple-to-one). Queue-aware decisions (DNN-Scissor). The monitor-partition-schedule-deploy decomposition (AMP4EC). Crowd/D2D framing and heartbeat failure detection (CrowdNet).

### Gaps
- None of the seven papers measures energy or battery on phones during partitioned inference. Neurosurgeon measures energy on a Jetson TK1. So no external phone-energy baseline exists in this set for CROWDio's battery and thermal telemetry.
- I could not verify the peer-reviewed venues for DDS (only arXiv/CoRR found) or AMP4EC (FMEC 2025 is claimed in the arXiv record but not confirmed).
- I did not read the survey-cited works most relevant to switching cost (NEUKONFIG [134], CONTINUER [146], AdaMEC [88], Tian et al. [85]), so claims about how they model repartition overhead stay unverified here.
