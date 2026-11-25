# Closest prior art for CROWDio's reconfiguration-aware partition planner: full-text reading

Scope: full-text (or best-available) reading of the papers judged most likely to pre-empt CROWDio's claims. Those claims are: on volunteer Android phones, the planner chooses cut points and placement from live EWMA telemetry. Under single residency, the switch cost is unload + (fetch+write if absent) + load + session init (measured cold ~48 s, warm ~6 s). That cost sits inside the cut x placement search (exhaustive oracle and Pareto DP). The controller uses a margin, persistence, cooldown and a switch cap, and refuses voluntary switches on stale estimates. Voluntary switches drain; forced switches resume from a retained boundary activation. The system also does standby-disk prefetch, P2P shard fetch, and direct phone-to-phone activations rather than relay through the server.

Reading method: PDFs were downloaded and converted to text with pdftotext, then the method, cost-model, evaluation and limitations sections were read. Full text was obtained for NEUKONFIG, CONTINUER, AdaMEC, the Luger thesis, NNV12, IONN, the EdgeShard IEEE IoT-J journal version, Tian et al. 2021 (via Europe PMC), Huang/Zhang/Xu 2024, Djuhera/Koch/Binotto 2025, Koch/Djuhera/Binotto 2025, Qiu et al. 2025/26 (CoCaR) and Honeybee-DR 2026. Galaxy was read through its arXiv HTML. Two texts were **inaccessible**: Honeybee-Tx (MDPI returned HTTP 403 to both HTML and PDF requests) and the Deng et al. 2026 arXiv full text (only the abstract page was read).

---

## Q1. NEUKONFIG: does it pre-empt switch-cost-aware repartitioning?

### Takeaway
NEUKONFIG is the earliest paper that measures repartitioning downtime for split DNNs (~6 s Pause-and-Resume). It reduces that downtime mechanically with a second "Dynamic Switching" pipeline, but it does **not** put the switch cost into the partition decision. The authors say it repartitions "whenever there is a change in network speed", with no hysteresis. It pre-empts the observation that repartitioning has a measurable downtime cost and the make-before-break idea (similar to CROWDio's standby prefetch). It does not pre-empt cost-in-search, the gated controller, or the phone setting.

### Cited Findings
- Citation: A. A. Majeed, P. Kilpatrick, I. Spence, B. Varghese, "NEUKONFIG: Reducing Edge Service Downtime When Repartitioning DNNs," IEEE Int. Conf. on Cloud Engineering (IC2E) 2021; arXiv:2106.15689 (29 Jun 2021). The IC2E DOI was not verified. — [arXiv](https://arxiv.org/abs/2106.15689)
- Setting: a two-way edge–cloud split of VGG-19 and MobileNetV2 (Keras/TensorFlow) in Docker containers connected with ZeroMQ. Latency model Tinf = Te + Tt + Tc. The partition is chosen by latency only. — [arXiv PDF](https://arxiv.org/pdf/2106.15689)
- Triggers: the paper examines repartitioning scenarios caused by network-speed changes (20 Mbps to 5 Mbps, emulated with Linux tc) and by CPU/memory stress (stress-ng). Network speed is identified as the key trigger. — [arXiv PDF](https://arxiv.org/pdf/2106.15689)
- Baseline Pause-and-Resume: tdowntime = tupdate, about 6 s. Neither CPU/memory availability nor network conditions changed this downtime. — [arXiv PDF](https://arxiv.org/pdf/2106.15689)
- Dynamic Switching variants:
  - Scenario A keeps a redundant pipeline always running: tdowntime = tswitch, under 0.98 ms, at twice the memory (1526.2 MB vs 763.1 MB) in Case 1.
  - Scenario B initialises a pipeline only on change: tdowntime = tinit + tswitch. Case 1 (new Docker container, base image optimised to 575 MB) gives about 1.9 s. Case 2 (new pipeline inside the existing container) gives about 0.6 s at the same memory as the baseline.
  - — [arXiv PDF](https://arxiv.org/pdf/2106.15689)
- During Dynamic Switching, the old pipeline keeps serving at degraded quality, so frames are processed rather than dropped. The paper measures frame-drop rate at 5–20 FPS. — [arXiv PDF](https://arxiv.org/pdf/2106.15689)
- Testbed: a Raspberry Pi 3B+ "device" (1 GB RAM), an x86 edge server (4 cores, 8 GB) and an x86 cloud server (8 cores, 16 GB). There are no phones. — [arXiv PDF](https://arxiv.org/pdf/2106.15689)
- Limitation, in the authors' own words: "NEUKONFIG repartitions DNN whenever there is a change in network speed", which "may adversely impact the performance efficiency". Future work is to "consider how frequently the DNN must be repartitioned". — [arXiv PDF](https://arxiv.org/pdf/2106.15689)

### Inferences
- NEUKONFIG explicitly names the open problem CROWDio addresses: deciding *whether and how often* to repartition. It leaves that as future work. This is strong supporting evidence for CROWDio's novelty framing and should be cited as the problem's origin.
- Its redundant-pipeline idea is close to CROWDio's "prefetch to standby disk" and drain-on-voluntary-switch. CROWDio should cite it as make-before-break precedent. The difference is that NEUKONFIG keeps a second resident pipeline (2x memory). Under single residency on phones that is impossible, which is why CROWDio has to *price* the switch instead.
- NEUKONFIG's 6 s is container pause/update on x86. CROWDio's ~48 s cold / ~6 s warm on phones is a different cost regime, which strengthens the argument that the cost must be included in the decision.

### Gaps
- The IC2E 2021 DOI and page numbers were not verified.

---

## Q2. CONTINUER: failure-time technique selection with a downtime term

### Takeaway
CONTINUER reacts to edge-node failure by choosing among repartitioning, early-exit and skip-connection. It uses a weighted-sum objective over accuracy, latency and downtime, plus user thresholds. Its "downtime" is *empirical and constant per technique*, essentially decision time plus reconnect time. It is not a per-candidate reload or migration cost. It covers only forced (failure) switches, has no hysteresis and no voluntary re-planning, and runs on x86 only. It partially pre-empts "forced switch handling with a downtime-aware selection", but not CROWDio's cost-in-search planner or its resume-from-boundary-activation mechanism.

### Cited Findings
- Citation: A. A. Majeed, P. Kilpatrick, I. Spence, B. Varghese, "CONTINUER: Maintaining Distributed DNN Services During Edge Failures," IEEE Int. Conf. on Edge Computing and Communications (EDGE) 2022, pp. 143–152. DOI 10.1109/EDGE55608.2022.00029 (from repository metadata). arXiv:2206.05267. — [arXiv](https://arxiv.org/abs/2206.05267); [St Andrews repository](https://research-repository.st-andrews.ac.uk/handle/10023/25431)
- The paper defines "Downtime in this paper refers to the time taken to recover from edge failure". In the method, however, downtime is "the time taken to retrieve the estimated accuracy and latency … and for the Scheduler … to select". Downtime values "are however not estimated, instead empirical values of downtime of each technique are used". — [arXiv PDF](https://arxiv.org/pdf/2206.05267)
- Repartitioning is described as expensive ("both computational and communication overheads"). It cites NEUKONFIG's second pipeline to reduce downtime. Skip-connection adds 0.99 ms to reinstate connections. — [arXiv PDF](https://arxiv.org/pdf/2206.05267)
- Decision method: classic additive weighting (weighted sum of min–max-normalised A, L, D with user weights; a weight of 0 if no threshold is given). An XGBoost per-layer latency predictor and a LightGBM accuracy-from-weights predictor are trained offline. — [arXiv PDF](https://arxiv.org/pdf/2206.05267)
- Reported overhead ("maximum downtime"): repartitioning 3.56 ms (ResNet-32) and 16.16 ms (MobileNetV2). Selection takes at most 16.82 ms. Estimation error is at most 0.28% (accuracy) and 13.06% (latency). Selection accuracy reaches up to 99.86%. — [arXiv PDF](https://arxiv.org/pdf/2206.05267)
- Platforms: Intel i7-8700 (3.2 GHz, 16 GB) and i5-8250U (1.6 GHz, 16 GB). Models: ResNet-32 and MobileNetV2 on CIFAR-10. — [arXiv PDF](https://arxiv.org/pdf/2206.05267)

### Inferences
- CONTINUER's "downtime" for repartitioning (ms) leaves out actual model redeploy/load time. CROWDio measures and models those directly (tens of seconds on phones). This is a clear differentiator.
- CONTINUER picks a *technique*, not a cut x placement. Recovery after a node failure redeploys the whole DNN; there is no retained-activation resume. CROWDio's forced-switch resume from a stage-boundary activation is not pre-empted here.

### Gaps
- None material. The full text was read.

---

## Q3. AdaMEC: avoiding repartitioning overhead by pre-partitioned "atoms"

### Takeaway
AdaMEC identifies the cost of re-partition, re-compilation, re-transmission and re-offloading as the core problem. It avoids it *structurally* by pre-partitioning a model once into operator-level "atoms" (PyTorch .pkl). At runtime it searches combinations when hard constraints are violated, then orders the offloading of atoms by a shortest path whose edge weights are atom **transmission latency**. Transmission cost therefore shapes the *migration order*, not the *decision to switch*. Load, session-init and residency costs are not modelled. There is no margin, persistence or cooldown. AdaMEC is the closest prior art for "migration path priced by shard transfer". It does not pre-empt "switch only if J(new)+margin < J(current)" or single-residency reload pricing.

### Cited Findings
- Citation: B. Pang, S. Liu, H. Wang, B. Guo, Y. Wang, H. Wang, Z. Sheng, Z. Wang, Z. Yu, "AdaMEC: Towards a Context-Adaptive and Dynamically-Combinable DNN Deployment Framework for Mobile Edge Computing," ACM Transactions on Sensor Networks, 2023/2024, DOI 10.1145/3630098; arXiv:2310.16547. — [arXiv](https://arxiv.org/abs/2310.16547); [Crossref](https://api.crossref.org/works/10.1145%2F3630098)
- Problem statement: any change in context affects "DNN re-partition and re-compilation, storage, re-transmission, and re-offloading of partitioned modules, which is costly and intractable". IONN and SPINN "need to rerun both the partition and offloading steps from scratch". — [arXiv PDF](https://arxiv.org/pdf/2310.16547)
- Trigger: "When the deployment context or latency requirements change, the current combination … may no longer satisfy hard constraints. Thus our algorithm is triggered". The search runs from the current combination outward (a Euclidean gap to the constraints). — [arXiv PDF](https://arxiv.org/pdf/2310.16547)
- Offloading plan: a Dijkstra-style search over combination graph edges weighted by "the transmitting latency caused by offloading network structures and parameters". It prioritises the lowest-overhead atom first and avoids offloading atoms the target will not use. — [arXiv PDF](https://arxiv.org/pdf/2310.16547)
- Objective (Formula 4): maximise latency benefit subject to the latency requirement and per-device memory and FLOPs budgets. Switching or transfer cost is not a term in the combination objective. — [arXiv PDF](https://arxiv.org/pdf/2310.16547)
- Devices: Raspberry Pi 4B (Cortex-A72, 2 GB), MinDong smartwatch (Cortex-A53, 3 GB), Jetson AGX Xavier, and Turtlebot3. PyTorch. Datasets CIFAR100, ImageNet and BDD100K. Results: up to 62.14% latency reduction and 55.21% average memory saving. Decision times are compared with Neurosurgeon (428.21 ms), DADS (30.01 ms) and QDMP (32.69 ms). — [arXiv PDF](https://arxiv.org/pdf/2310.16547)

### Inferences
- AdaMEC's "avoid re-partition, only re-offload" approach is the closest structural analogue to CROWDio's pre-exported shard catalogue plus P2P fetch. CROWDio should cite it and differentiate on three points: (a) phones with single ONNX-session residency, where *load + session init* dominates over transfer; (b) cost included in the selection objective, not just the migration order; (c) controller gating.

### Gaps
- AdaMEC does not report measured end-to-end reconfiguration time (transfer + load) in a breakdown comparable to CROWDio's 48 s / 6 s.

---

## Q4. Tian et al.: "server-switching delays" in DNN partitioning

### Takeaway
The paper is Tian, Zhu, Xu, Li, *Sensors* 21(1):229, 2021 (MDPO). A 2026 ACM Computing Surveys paper describes it as incorporating "server-switching delays". In the full text, the "switching" is **base-station handover under known mobility**: a DNN job is split across successive edge servers along a pre-known route. The **model is pre-embedded on every edge server and the device**, so there is no model upload, load or residency cost. It does not pre-empt CROWDio.

### Cited Findings
- Citation: X. Tian, J. Zhu, T. Xu, Y. Li, "Mobility-Included DNN Partition Offloading from Mobile Devices to Edge Clouds," Sensors 21(1):229, 2021, DOI 10.3390/s21010229, PMCID PMC7795226. — [Europe PMC](https://europepmc.org/article/PMC/PMC7795226)
- The survey's characterisation: "[85] extends this by incorporating server-switching delays, enabling mobility-aware offloading". — [Hao et al., ACM Comput. Surv. 58(8):204, 2026](https://dl.acm.org/doi/10.1145/3786145)
- System model: "The UE's movement pattern and base stations' geographical distribution are known in advance". "The DNN model is embedded in the mobile device and edge clouds in advance". The objective is total latency, solved via a shortest path on a DAG. — [Europe PMC full text](https://europepmc.org/article/PMC/PMC7795226)
- Evaluation: the "mobile device is a laptop with a Intel CPU (2.11 GHz) and 2 GB memory"; the edge server has an RTX 2080 Ti. Models: AlexNet, VGG16, GoogLeNet, ResNet, MobileNet. The paper assumes edge server resources and mobile battery are sufficient (both are future work). — [Europe PMC full text](https://europepmc.org/article/PMC/PMC7795226)
- A separate Tian paper exists: "Energy-Efficient DNN Partitioning and Offloading for Task Completion Rate Maximization in Multiuser Edge Intelligence" (Wireless Communications and Mobile Computing, 2023). A search snippet mentions a "migration delay" layer-distribution step, but the **full text was not read**. — [ResearchGate listing](https://www.researchgate.net/publication/368681669_Energy-Efficient_DNN_Partitioning_and_Offloading_for_Task_Completion_Rate_Maximization_in_Multiuser_Edge_Intelligence)

### Inferences
- CROWDio's related-work section should not describe Tian 2021 as "switching-cost-aware partitioning" in CROWDio's sense. Its "switching" is a handover of an in-flight job, and the models are pre-installed everywhere.

### Gaps
- The full text of Tian et al. WCMC 2023 was not read, so its "migration delay" model is unverified.

---

## Q5. IONN (SoCC 2018): model upload cost inside the partition graph

### Takeaway
IONN is the earliest work that puts a **model upload (transfer) cost into the partitioning graph**. Edge weights include per-layer upload time, which is decayed by K < 1 per iteration to produce an incremental upload plan. It models first-time installation on a server, not switching between configurations. There is no load or session-init term, no re-planning controller, and no single-residency limit; the server is assumed to have ample resources. It is prior art for "fetch cost in the search", and CROWDio must cite it. It does not cover unload/load/init or controller gating.

### Cited Findings
- Citation: H.-J. Jeong, H.-J. Lee, C. H. Shin, S.-M. Moon, "IONN: Incremental Offloading of Neural Network Computations from Mobile Devices to Edge Servers," ACM SoCC 2018, pp. 401–411, DOI 10.1145/3267809.3267828. — [PDF (course mirror)](https://cse.buffalo.edu/faculty/tkosar/cse710_spring19/jeong-socc18.pdf); [DBLP](https://dblp.uni-trier.de/rec/conf/cloud/JeongLSM18.html)
- Motivating measurement: uploading AlexNet over ~80 Mbps Wi-Fi takes about 24 s. Local execution on the client takes 1.3 s per query. — [PDF](https://cse.buffalo.edu/faculty/tkosar/cse710_spring19/jeong-socc18.pdf)
- Algorithm: the NN execution graph has edge weights equal to layer execution time plus layer upload time. The algorithm repeatedly finds the shortest path, uploads the newly server-side layers as a partition, multiplies upload time by K, and repeats until the weights are about 0. — [PDF](https://cse.buffalo.edu/faculty/tkosar/cse710_spring19/jeong-socc18.pdf)
- Assumption: "edge servers have plenty of network/computing resources"; multi-client contention is future work. — [PDF](https://cse.buffalo.edu/faculty/tkosar/cse710_spring19/jeong-socc18.pdf)
- Hardware: Odroid XU4 client (ARM big.LITTLE, 2 GB) and x86 + GTX 1080 Ti server. Caffe. — [PDF](https://cse.buffalo.edu/faculty/tkosar/cse710_spring19/jeong-socc18.pdf)

### Inferences
- IONN prices *upload* only; it does not price model load or initialisation on the receiving node, and it never evicts. CROWDio's cost term (unload + fetch/write + load + session init, with residency) is a strict generalisation for a different direction: worker-side shard installation on memory-limited phones.

### Gaps
- The text was not searched for any measurement of server-side model load time beyond upload.

---

## Q6. Deng et al., arXiv 2605.09623 (2026)

### Takeaway
The paper is a small-testbed adaptive three-tier partitioner: Raspberry Pi edge, laptop fog, desktop cloud. It profiles at startup and **periodically re-evaluates** the partition. Only the abstract was read. Nothing in the abstract indicates switching or reload cost modelling, hysteresis or phones. It probably does not pre-empt CROWDio, but this is unverified because the full text was not read.

### Cited Findings
- Citation: A. A. Deng, E. Butkus, A. Lapkovskis, P. K. Donta, "Adaptive AI Task Partitioning and Safe Offloading in Heterogeneous Edge-Cloud Continuum," arXiv:2605.09623 (v1 10 May 2026; v2 18 Aug 2026). A related DOI 10.1007/978-3-032-35576-8_23 (Springer) is listed; the venue was not named. — [arXiv](https://arxiv.org/abs/2605.09623)
- Abstract-level results: VGG16, AlexNet and MobileNetV2; energy reduced 27.09–35.82% and latency reduced 6.34–22.92% versus a static partition baseline. — [arXiv](https://arxiv.org/abs/2605.09623)

### Gaps
- **Full text not read.** Whether its periodic re-evaluation includes any switch cost or damping is unknown. This should be checked before final submission.

---

## Q7. Luger 2023 (TU Wien): cost-aware splitting and dynamic rescheduling

### Takeaway
"Cost" in Luger's work means **monetary AWS cost**, not switching cost. Rescheduling has zero reconfiguration cost by design: the whole model is stored on both sides, and each request carries its split point. The trigger is time-based or threshold-based ("e.g. every two minutes"), and the author calls the trigger outside the scope of the algorithm. It does not pre-empt CROWDio.

### Cited Findings
- Citation: D. Luger, "Cost-Aware Neural Network Splitting and Dynamic Rescheduling for Edge Intelligence," Diploma thesis (Dipl.-Ing., Business Informatics), TU Wien, 26 June 2023. Advisor: I. Brandić; assistance: A. Aral. Partial results appeared at EdgeSys '23 (Luger, Aral, Brandić), DOI 10.1145/3578354.3592871. — [TU Wien repositum PDF](https://repositum.tuwien.at/bitstream/20.500.12708/188349/1/Luger%20Daniel%20-%202023%20-%20Dynamic%20Scheduling%20of%20Neural%20Network%20Splitting.pdf); [record](https://repositum.tuwien.at/handle/20.500.12708/188349)
- Residency: "the model itself must already be stored on the server since only the tensor is offloaded … All possible models have to be stored along with all layers since the splitting point is unknown." — [thesis PDF](https://repositum.tuwien.at/bitstream/20.500.12708/188349/1/Luger%20Daniel%20-%202023%20-%20Dynamic%20Scheduling%20of%20Neural%20Network%20Splitting.pdf)
- Trigger: prediction "can be eighter time triggered, e.g. every two minutes … or based on a threshold … when the bandwidth or clock speed changed drastically". The author says this "does not matter to our proposed rescheduling algorithm". — [thesis PDF](https://repositum.tuwien.at/bitstream/20.500.12708/188349/1/Luger%20Daniel%20-%202023%20-%20Dynamic%20Scheduling%20of%20Neural%20Network%20Splitting.pdf)
- Inputs: edge clock speed, bandwidth, latency, and edge/cloud prices (AWS t3.xlarge Wavelength edge at $0.224/h vs cloud at $0.1664/h). A tunable weight trades latency against cost. Testbed: Raspberry Pi 4 (4 GB) edge and an AWS cloud instance. Result: 60.84% cost reduction at +25.92% latency for AlexNet (rented edge). — [thesis PDF](https://repositum.tuwien.at/bitstream/20.500.12708/188349/1/Luger%20Daniel%20-%202023%20-%20Dynamic%20Scheduling%20of%20Neural%20Network%20Splitting.pdf)
- Limitations: GNN splitting was infeasible for the SWAIN use case, and non-chain DNNs are not supported. — [thesis PDF](https://repositum.tuwien.at/bitstream/20.500.12708/188349/1/Luger%20Daniel%20-%202023%20-%20Dynamic%20Scheduling%20of%20Neural%20Network%20Splitting.pdf)

### Inferences
- Luger's "full model resident on both sides" assumption is exactly what single residency on phones rules out. It is a useful contrast case for CROWDio.

### Gaps
- The EdgeSys '23 paper itself was not read; the thesis supersedes it.

---

## Q8. NNV12 (MobiSys 2023): cold vs warm inference breakdown on devices

### Takeaway
NNV12 gives authoritative evidence that on-device cold inference (read + transform + execute) is much slower than warm inference, 1.5–12.7x on CPU and 85.5–443.5x on GPU. It optimises the cold path itself through kernel selection, caching post-transformed weights on disk, and pipelining. It does not do partitioning or placement. It supports CROWDio's premise that load and init cost dominate, and it is complementary: NNV12-style techniques would shrink CROWDio's switch cost.

### Cited Findings
- Citation: R. Yi, T. Cao, A. Zhou, X. Ma, S. Wang, M. Xu, "Boosting DNN Cold Inference on Devices," ACM MobiSys 2023, Helsinki, DOI 10.1145/3581791.3596842. — [ACM DOI](https://doi.org/10.1145/3581791.3596842)
- The cold-vs-warm gap on vanilla libraries (TFLite/TF/tfjs, ncnn, MNN) is 1.5–12.7x on CPU and 85.5–433.5x on GPU. The abstract and intro say 443.5x, so there is an internal inconsistency in the paper. ResNet-50 on a Kryo 475 CPU: cold at least 511.67 ms vs warm 141.56 ms. — [ACM DOI](https://doi.org/10.1145/3581791.3596842)
- Table 1, ResNet-50 cold breakdown:

  | Stage | Pixel 5 CPU | Jetson TX2 GPU |
  |---|---|---|
  | Weights reading | 36.52 ms | 43.03 ms |
  | GPU preparation | – | 3004.01 ms |
  | Weights transformation | 1135.28 ms | 1616.84 ms |
  | Execution | 190.12 ms | 802.77 ms |
  | Total cold | 1363.23 ms | 5467.48 ms |
  | Warm | 185.82 ms | 137.02 ms |

  The memory-allocation row was not legible in the extraction. — [ACM DOI](https://doi.org/10.1145/3581791.3596842)
- Speedups: up to 15.2x (edge CPU) and 401.5x (edge GPU) over the state of the art. On Meizu 16T CPU, 5.1x/9.5x/3.7x average versus ncnn/TFLite/AsyMo. On the TX2 GPU, up to 58.2x/401.5x versus ncnn/TF. Evaluated on 12 models and 4 devices (2 smartphones, 2 Jetsons). — [ACM DOI](https://doi.org/10.1145/3581791.3596842)
- Motivation cites ">40%" app cold-launch probability among 96 Android users. — [ACM DOI](https://doi.org/10.1145/3581791.3596842)

### Inferences
- In CROWDio's 48 s / 6 s, download and ORT session creation for DistilBERT shards dominate. NNV12 shows that weight transformation (not disk read) dominates CNN cold start on a phone CPU. CROWDio should report its own breakdown (fetch vs write vs session init) to make the cost model credible, and can cite NNV12 for the per-stage decomposition.

### Gaps
- The ACM full text was read from a PDF copy. The memory-allocation row in Table 1 was not recovered.

---

## Q9. Honeybee-Tx (Future Internet 2025) and the Honeybee family

### Takeaway
Honeybee-Tx is a dependency-aware, capability-aware **work-stealing** framework for heterogeneous Android clusters (HAR and video workloads). It is the closest mobile-crowd-computing (MCdC) system prior art for phone clusters with DAG dependencies. Abstract-level evidence shows no DNN partition planning, model-residency costing or switch control. **The full text was inaccessible** (MDPI 403). The successor Honeybee-DR (Sci. Rep. 2026, full text read) handles worker drop-out by re-issuing stolen jobs (stateless re-execution), not by resuming from retained activations. It has no model-load cost model.

### Cited Findings
- Citation: S. Segu Nagesh, N. Fernando, S. W. Loke, et al., "A Dependency-Aware Task Stealing Framework for Mobile Crowd Computing" (Honeybee-Tx), Future Internet 17(10):446, 2025. The title is from a Bohrium index; the full author list and DOI were not verified because MDPI blocked access. — [MDPI](https://www.mdpi.com/1999-5903/17/10/446); [IDEAS/RePEc](https://ideas.repec.org/a/gam/jftint/v17y2025i10p446-d1761063.html); [Bohrium](https://www.bohrium.com/paper-details/a-dependency-aware-task-stealing-framework-for-mobile-crowd-computing/1179666080608550912-21695)
- Abstract-level claims (search summaries of the MDPI page): capability-aware job selection via lightweight profiling and dynamic scoring, dependency-aware stealing, and staged result transfers. Five Android devices, including OnePlus 5T, Pixel 6 Pro and Pixel 7. HAR: up to 4.72x speed-up and 63% per-device energy reduction. Video: 2.06x speed-up, 51.4% energy and 71.6% memory savings. — [MDPI](https://www.mdpi.com/1999-5903/17/10/446)
- Honeybee-DR citation: S. Segu Nagesh, N. Fernando, S. W. Loke, A. Ghari Neiat, P. N. Pathirana, "Honeybee-DR: dynamic dependency management and lightweight reliability for mobile crowd computing," Scientific Reports (article in press, accepted 7 Jul 2026), DOI 10.1038/s41598-026-61849-0. — [DOI](https://doi.org/10.1038/s41598-026-61849-0)
- Honeybee-DR failure model: "Delegator failure terminates the session … Worker drop-out is absorbed by the stateless design: the delegator reissues any job in the STOLEN state". Devices communicate peer-to-peer over Wi-Fi Direct with adaptive batching. Honeybee-Tx's work-stealing scaled to 5 devices at 94.4% of ideal speedup. — [DOI](https://doi.org/10.1038/s41598-026-61849-0)

### Inferences
- The Honeybee family pre-empts "P2P phone clusters with DAG work-stealing and capability scoring". It does not pre-empt DNN cut/placement search, the single-residency switch cost, the gated controller, or boundary-activation resume. Because work-stealing is pull-based, there is no global plan to switch, which is a conceptual contrast with CROWDio's planner.

### Gaps
- **The Honeybee-Tx full text was inaccessible.** Whether it carries model or binary state and whether it measures any task-setup or load cost is unverified.

---

## Q10. Galaxy (INFOCOM 2024) and EdgeShard journal version: do they re-plan?

### Takeaway
Neither re-plans at runtime, and neither models switching or reload cost. Galaxy's planner is "an offline procedure that runs once before deployment". EdgeShard (IEEE IoT-J 2025) profiles offline once and solves a DP. Its "dynamic model loading" is only a profiling trick for memory-limited devices. Both are prior art for the *static* planning half of CROWDio, in particular DP over layers x devices, which is close to CROWDio's Pareto DP. Neither pre-empts the reconfiguration-aware half.

### Cited Findings
- Galaxy citation: S. Ye et al., "Galaxy: A Resource-Efficient Collaborative Edge AI System for In-situ Transformer Inference," IEEE INFOCOM 2024, pp. 1001–1010; arXiv:2405.17245. The journal extension "Resource-Efficient Collaborative Edge Transformer Inference With Hybrid Model Parallelism" appeared in IEEE TMC 2025 (per DBLP; not read). — [arXiv](https://arxiv.org/abs/2405.17245); [DBLP](https://dblp.dagstuhl.de/pid/337/7674.html)
- Galaxy planning: offline profiler, then a two-step heterogeneity- and memory-aware workload split, O(D³), under 1 s for 4 devices. It uses hybrid tensor + sequence parallelism with tile-based communication overlap. Devices: Jetson Nano variants (403 MHz–1.47 GHz, 0.7–1.5 GB budgets). Models: DistilBERT to OPT-XL. Results: 1.3–2.5x latency reduction. There is no runtime re-planning, no device churn handling, and it assumes trusted devices. — [arXiv HTML](https://arxiv.org/html/2405.17245v1)
- EdgeShard citation: M. Zhang, X. Shen, J. Cao, Z. Cui, S. Jiang, "EdgeShard: Efficient LLM Inference via Collaborative Edge Computing," IEEE Internet of Things Journal 12(10):13119–, 15 May 2025, DOI 10.1109/JIOT.2024.3524255. — [DOI](https://doi.org/10.1109/JIOT.2024.3524255)
- EdgeShard: "Profiling is an offline step … and only needs to be done once". Joint device selection and partition is solved by DP for latency or throughput. Testbed: 12 Jetson AGX Orin, 2 Orin NX and 1 RTX 3090 server. Models: Llama2 7B/13B/70B. The Discussion and Future Work section covers incentives, batch size and security, but not runtime re-planning or reconfiguration cost. — [DOI](https://doi.org/10.1109/JIOT.2024.3524255)

### Inferences
- Both works' own discussions mention device-owner incentives, DePIN and volunteer settings without handling churn. This is a natural "gap" citation for CROWDio's volunteer-phone churn plus forced-switch design.

### Gaps
- The Galaxy TMC 2025 journal extension was not read. It may add dynamic adaptation and should be checked.

---

## Q11. 2024–2026 "switching-/reconfiguration-/migration-aware" partitioning papers found

### Takeaway
Three recent works come closest. (a) Huang/Zhang/Xu 2024 is the strongest **conceptual** pre-emption: it explicitly motivates switching cost as "reloading and setting up the DNN models in the device memory" and uses a blocked bandit to limit switches, but it is simulation-only and uses an abstract switching penalty. (b) Djuhera/Koch/Binotto 2025 (and a Koch et al. 2025 companion) describe an orchestrator with **EWMA-latency triggers, thresholds and a 30 s cooldown (Tcool) "to prevent thrashing"**, choosing between migration and re-split. This pre-empts CROWDio's EWMA + cooldown controller elements, but it has no switch cost in C(d) and its results are analytical "expected" KPIs. (c) CoCaR (Qiu et al.) puts model loading time into edge-server caching decisions with measured load times. None combines phones, single residency, a measured reload cost inside the cut x placement search, margin/persistence/cap/staleness gating, and drain/resume epochs.

### Cited Findings
- **Huang, Zhang, Xu, "Learning the Optimal Path and DNN Partition for Collaborative Edge Inference," arXiv:2410.01857 (2 Oct 2024)**, "submitted to IEEE journals":
  - Its motivation states that "switching devices for collaborative edge inference necessitates reloading and setting up the DNN models in the device memory, which introduces overhead … Thus, frequent device and path switching is undesirable."
  - It formulates an adversarial group linear bandit *with switching costs*. The proposed B-EXPUCB uses a block structure to limit switches, with an O(T^{1/3}) switching-cost bound. Evaluation is simulation only.
  - — [arXiv](https://arxiv.org/abs/2410.01857); [PDF](https://arxiv.org/pdf/2410.01857)
- A search snippet attributes to an IEEE TMC vol. 25 no. 2 (Feb 2026) paper the sentence: if the chosen path differs from the previous one, "additional costs due to reloading and setting up the DNN on the nodes of the new path are incurred". This is likely the journal version of the above, but the NSF PAR link returned "Not Found" and **it is unverified**. — [NSF PAR (dead link)](https://par.nsf.gov/servlets/purl/10666339)
- **Djuhera, Koch, Binotto, "Joint Partitioning and Placement of Foundation Models for Real-Time Edge AI," arXiv:2512.01039 (30 Nov 2025)**:
  - ShouldReconfigure triggers on EWMA latency > Lmax (150 ms), max utilisation > Umax (0.85), or min bandwidth < Bmin (50 Mbps). It applies a "Time-to-reconfigure cool-down Tcool 30 s", and "Reconfigurations are further rate-limited by Tcool to prevent thrashing".
  - It tries placement migration first, then split revision. It selects d^ = argmin C(d), where C(d) has no reconfiguration-cost term.
  - Evaluation: "expected" KPIs from an analytical ETSI MEC model (three A100 MEC nodes, Llama3-8B). This is not a measured system.
  - — [arXiv](https://arxiv.org/abs/2512.01039)
- **Koch, Djuhera, Binotto, "Intelligent Orchestration of Distributed Large Foundation Model Inference at the Edge," arXiv:2504.03668 (v3 12 Jul 2025)**: the same Lmax/Umax/Bmin/Tcool controller, with Tcool = 30 s "to prevent thrashing". KPI tables report reduced downtime incidents per hour. — [arXiv](https://arxiv.org/abs/2504.03668)
- **Qiu, Dong, Tan, Zhou, Shen, Lee, Fan, "Joint Optimization of DNN Model Caching and Request Routing in Mobile Edge Computing," arXiv:2511.03159 (v2 12 May 2026)**: it considers "the impact of model loading time on QoE due to the switching of caching schemes". Measured: loading ViT submodel 2 from storage takes 0.877 s versus 0.248 s to switch from submodel 1 to 2. It notes prior work uses load time "only to refine solutions rather than to inform caching decisions". The setting is edge servers / base stations, evaluated in simulation on a Xeon + 8x RTX 3070. — [arXiv](https://arxiv.org/abs/2511.03159)
- Also surfaced, not deeply read: Hao et al. ACM CSUR 2026 survey, which flags "Automating rapid re-partitioning in response to node overload or failure" as an open challenge — [ACM](https://dl.acm.org/doi/10.1145/3786145). Zhang & Matta 2025 survey (arXiv:2510.22909); its abstract does not mention cold start or loading — [arXiv](https://arxiv.org/abs/2510.22909).
- Self-prior-art: CROWDio's own preprint, Manamperi, Pathirana, Pathirana, Premarathna, "Memory-Efficient Partitioned DNN Inference on Resource-Constrained Android Crowds," arXiv:2605.20723 (2026), already publishes a 48±4 s cold-start load (download + ONNX session init), 6±1 s Tier-2 warm loads, the 4-tier affinity (resident/cached/idle/evict) and five Android handsets. Its own text frames "memory-adaptive multi-partition residency" and "aggressive pre-fetching" as natural extensions. — [arXiv](https://arxiv.org/abs/2605.20723)

### Inferences
- **Pre-emption flags:**
  1. "Switching cost = reload + setup, so penalise or limit switches" is stated by Huang/Zhang/Xu 2024. CROWDio cannot claim to be first to recognise reload cost as a switching penalty in collaborative partitioning. It can claim a *measured, decomposed, residency-aware* cost inside a cut x placement search on real phones.
  2. "EWMA-triggered reconfiguration with a cooldown to prevent thrashing" is stated by Djuhera/Koch/Binotto 2025. CROWDio cannot claim cooldown or EWMA triggering as novel in themselves. Remaining differentiators: a cost-aware margin comparison J(new)+margin < J(current), persistence, a switch cap, the staleness refusal, and measured evaluation.
  3. The 48 s / 6 s numbers are already public in CROWDio's own arXiv:2605.20723. They should be cited as such, not presented as new.
- Remaining defensible novelty (no counter-example found): the *conjunction* of (i) single-residency phone workers, (ii) a decomposed switch cost (unload, conditional fetch+write, load, session init) used *inside* the exhaustive or Pareto-DP cut x placement search, (iii) margin + persistence + cooldown + cap + staleness-refusal gating, (iv) drain-on-voluntary vs resume-from-boundary-activation-on-forced epochs, (v) standby-disk prefetch and P2P shard fetch, and (vi) direct phone-to-phone activations vs relay, all evaluated on real handsets.

### Gaps
- No paper was found that applies hysteresis or margin-based switching to on-device (smartphone) DNN partition re-planning. Searches were limited to Google-style web search plus arXiv; Google Scholar and IEEE Xplore were not queried directly.
- The IEEE TMC Feb 2026 version of the path/partition switching-cost paper is unverified.
- Full texts not read: Honeybee-Tx (MDPI 403), Deng et al. 2605.09623, the Galaxy TMC 2025 extension, and Tian et al. WCMC 2023.
