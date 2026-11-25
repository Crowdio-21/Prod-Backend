# Reconfiguration-Aware / Switching-Cost-Aware DNN Partitioning and Placement (CROWDio planner)

Verification legend. [V] means the citation was checked against a primary or near-primary source in this session (arXiv abstract page, USENIX page, or publisher listing). [K] means it is well known and the URL is the canonical arXiv or USENIX ID, but it was NOT re-fetched in this session. Numbers given for [K] items are kept conservative. Check them against the PDFs before submission.

Collision note. CROWDio's own earlier paper is already public: Manamperi, Pathirana, Pathirana, Premarathna, Gunasekera, "Memory-Efficient Partitioned DNN Inference on Resource-Constrained Android Crowds," ICML 2026 Workshop on ML for the Global South, arXiv:2605.20723 (20 May 2026). It covers JIT deferred partition loading, the single-partition-resident constraint, a 4-tier affinity scheduler, zlib tensor transport, and streaming 1:1 dependencies. It reports 43±2 MB peak RSS, about 50 mAh per run, and 34% lower latency than barrier synchronisation on DistilBERT across 5 handsets. Its abstract does not mention replanning or switching cost ([arXiv](https://arxiv.org/abs/2605.20723) [V]). The new paper must position itself as the reconfiguration-aware extension of this one, and should self-cite it to avoid a self-plagiarism or overlap flag. This earlier paper is also the natural source for baseline B1 (network-aware affinity).

## 1. Dynamic / adaptive DNN partitioning under changing bandwidth: do these works model the cost of repartitioning?

### Takeaway
Almost the entire split-computing line re-selects the cut per request or per window from bandwidth estimates. It treats the new cut as free, assuming every candidate partition is already resident on both sides. IONN (2018) is the important exception: it models model-upload cost and builds the partition up incrementally. The newest adaptive work (2026) uses only a fixed improvement margin with no reload cost. LLM-on-edge systems (EdgeShard, Galaxy, HexGen, Helix) plan statically or once per configuration and do not price a re-plan.

### Cited Findings
- **Neurosurgeon.** Kang, Hauswald, Gao, Rovinski, Mudge, Mars, Tang, ASPLOS 2017, DOI 10.1145/3037697.3037698 [K]. Picks the layer split point at runtime using regression-predicted per-layer latency and current bandwidth (greedy search over split points) ([survey summary](https://arxiv.org/pdf/2411.18199) [V]). It models no switching cost and assumes the full model is present on both mobile and server. This is the ancestor of baselines B0 and B2.
- **IONN.** Jeong, Lee, Shin, Moon, "IONN: Incremental Offloading of Neural Network Computations from Mobile Devices to Edge Servers," ACM SoCC 2018, pp. 401–411, DOI 10.1145/3267809.3267828 ([researchr](https://researchr.org/publication/JeongLSM18); [PDF](https://cse.buffalo.edu/faculty/tkosar/cse710_spring19/jeong-socc18.pdf) [V]). It explicitly treats the DNN model upload to the edge server as a non-trivial delay. It partitions the model and uploads partitions in an order chosen by a graph or shortest-path algorithm so that queries benefit before the whole model arrives, and clients can offload queries mid-upload ([course summary](https://anrlutdallas.github.io/resource/courses/AON-F18/files/Ashkan-IONN.pdf) [V]). It **does** model a fetch cost, but as one-shot amortised upload ordering. It has no unload/load/session-init, no residency limit, no switching back and forth, and no hysteresis. **It is the closest prior work to CROWDio's C_fetch term and must be cited.** Code: [ElasticDNN](https://github.com/jinevening/ElasticDNN).
- **DADS.** Hu, Bao, Wang, Zhou, Li, "Dynamic Adaptive DNN Surgery for Inference Acceleration on the Edge," IEEE INFOCOM 2019, DOI 10.1109/INFOCOM.2019.8737614 [K]. Uses a min-cut formulation for DAG-shaped DNNs, switching between lightly loaded (latency) and heavily loaded (throughput) regimes by network condition. It has no switching cost.
- **JALAD.** Li, Hu, Jiang, Wang, Wang, Chen, "JALAD: Joint Accuracy- and Latency-Aware Deep Structure Decoupling for Edge-Cloud Execution," IEEE ICPADS 2018, arXiv:1812.10027 [K]. Uses ILP over cut point plus feature quantisation, re-solved when bandwidth changes. It has no switching cost.
- **Edgent.** Li, Zeng, Zhou, Chen, "Edge AI: On-Demand Accelerating Deep Neural Network Inference via Edge Computing," IEEE TWC 2020, arXiv:1910.05316 [K] (MECOMM 2018 workshop version). Jointly picks the partition point and an early-exit branch. Its dynamic-network version uses a bandwidth-state configuration map. It has no reload cost, and all branches and partitions are assumed deployed.
- **SPINN.** Laskaridis, Venieris, Almeida, Leontiadis, Lane, "SPINN: Synergistic Progressive Inference of Neural Networks over Device and Cloud," ACM MobiCom 2020, arXiv:2008.06402 [K]. Uses a runtime scheduler over split point plus early-exit policy under fluctuating network and server load. Both sides hold the whole model, so there is no switching cost. CROWDio's earlier paper contrasts itself with SPINN and Neurosurgeon on exactly this point ([arXiv 2605.20723](https://arxiv.org/abs/2605.20723) [V]).
- **ANS (Autodidactic Neurosurgeon).** Zhang, Chen, Xu, WWW 2021, arXiv:2102.02638 [K]. Learns the split point online as a contextual bandit (regret-bounded) without profiling. It has no switching cost.
- **DRL-based partitioning (DPTO).** "DNN partitioning and task offloading in mobile edge computing" (DQN/DDQN/PPO), Journal of Cloud Computing 12, 2023, DOI 10.1186/s13677-023-00493-9 ([Springer](https://link.springer.com/article/10.1186/s13677-023-00493-9) [V]). Uses an MDP over energy and delay. No switching term was found. I found **no paper using the acronym "DDPQN"**. It may be a mis-remembered acronym, so do not cite it without the original reference.
- **Lyapunov-guided diffusion RL for vehicular DNN partitioning.** arXiv:2406.06986 ([arXiv](https://arxiv.org/pdf/2406.06986) [V]). Decouples the long-term problem into per-slot deterministic problems. No switching cost was found.
- **Learning to Partition.** "Learning to Partition: Dynamic Deep Neural Network Model Partitioning for Edge-Assisted Low-Latency Video Analytics," Machine Learning and Knowledge Extraction 7(4):117, 2025, DOI 10.3390/make7040117 ([DOI](https://doi.org/10.3390/make7040117) [V, existence only; the MDPI page returned 403]). Whether it models switching cost is **unverified**.
- **Adaptive DNN Partitioning and Offloading in Heterogeneous Edge-Cloud Continuum.** Deng, Butkus, Lapkovskis, Donta (Stockholm Univ.), arXiv:2605.09623, 2026 ([arXiv](https://arxiv.org/html/2605.09623v1) [V, full text read]).
  - Re-evaluates every fixed window of 100 inferences by refitting compute rates, re-probing links, and re-searching the split. It switches only if the new candidate is at least 3% better, and a deadline violation forces a switch.
  - It does **not** model reload or weight-transfer cost and has no explicit hysteresis or persistence.
  - Results versus static splitting: latency falls 6.34% (VGG16), 22.92% (AlexNet), and 14.20% (MobileNetV2); energy falls 27–36%.
  - It is the most recent and closest "margin rule" analogue. CROWDio's break-even rule (J(new)+margin < J(current), with switch cost inside J), plus persistence, cooldown, and stale-estimate refusal, is strictly richer.
- **Cost-Driven Offloading for DNN-based Applications over Cloud, Edge and End Devices.** arXiv:1907.13306 ([arXiv](https://arxiv.org/pdf/1907.13306) [V]). Argues that static partitioning under fixed bandwidth assumptions is unsuitable because wireless and node resources are unstable. Monetary or latency cost only, with no switching cost.
- **Cost-Aware Neural Network Splitting and Dynamic Rescheduling for Edge.** D. Luger, TU Wien diploma thesis, 2023 ([repositum PDF](https://repositum.tuwien.at/bitstream/20.500.12708/188349/1/Luger%20Daniel%20-%202023%20-%20Dynamic%20Scheduling%20of%20Neural%20Network%20Splitting.pdf) [V, existence only]). The title suggests rescheduling-cost awareness. I could not extract the text, so it is a **must-check** for reviewer risk.
- **Survey: DNN Partitioning for Cooperative Inference in Edge Intelligence: Modeling, Solutions, Toolchains.** ACM Computing Surveys, DOI 10.1145/3786145 ([ACM](https://dl.acm.org/doi/10.1145/3786145) [V, via snippet]). Notes that task migration becomes frequent and unpredictable under mobility. Per a search snippet, it describes a graph whose vertices are partition/offloading configurations and whose edges connect neighbouring decisions to reduce repartitioning overhead. This means **the "repartitioning overhead" concept is acknowledged in the survey literature**. Read that section and cite the original paper it summarises.
- **Joint DNN layer placement and splitting on two timescales.** arXiv:2105.13618 ([arXiv](https://arxiv.org/abs/2105.13618) [V]).
  - Layer placement on the wireless device is optimised "in a larger time scale" (driven by retraining).
  - Per-request splitting is an optimal-stopping problem whose horizon is set by the placement.
  - It is conceptually close to CROWDio's split between residency/placement (slow) and execution (fast). Whether placement transfer cost appears in the objective was not confirmed from the abstract.
- **CoEdge.** Cooperative DNN inference with adaptive workload partitioning over heterogeneous edge devices, IEEE/ACM ToN ([NSF PAR](https://par.nsf.gov/servlets/purl/10313790) [V]). Re-partitions the workload (spatial) as bandwidth changes. Models are pre-deployed and there is no switch cost.
- **EdgeShard.** Zhang, Cao, Shen, Cui, "EdgeShard: Efficient LLM Inference via Collaborative Edge Computing," arXiv:2405.14371; IEEE IoT Journal 12(10):13119–13131, 2025 ([arXiv](https://arxiv.org/pdf/2405.14371); [PolyU](https://research.polyu.edu.hk/en/publications/edgeshard-efficient-llm-inference-via-collaborative-edge-computin/) [V]).
  - Joint device selection plus layer partitioning by DP, with separate DPs for latency and throughput, under per-device memory budgets.
  - Up to 50% lower latency and about 2x throughput on Llama2.
  - Static, one-shot plan: no re-plan cost, no residency state. **This is the closest prior for CROWDio's memory filter plus DP over ordered device assignments.**
- **Galaxy.** Ye, Du, Zeng, Ou, Chu, Lu, Chen, "Galaxy: A Resource-Efficient Collaborative Edge AI System for In-situ Transformer Inference," IEEE INFOCOM 2024, arXiv:2405.17245 [K]. Hybrid tensor plus sequence parallelism across home edge devices. Static planning, no switching cost.
- **Asteroid.** Ye, Zeng, Chu, Xing, Chen, "Asteroid: Resource-Efficient Hybrid Pipeline Parallelism for Collaborative DNN Training on Heterogeneous Edge Devices," ACM MobiCom 2024, pp. 312–326, DOI 10.1145/3636534.3649363, arXiv:2408.08015 ([arXiv](https://arxiv.org/html/2408.08015v1) [V]).
  - Training, not inference.
  - Its "fault-tolerant yet lightweight pipeline replay" recovers 14x faster than baselines after device exit or failure.
  - Up to 12.2x faster than conventional parallelism and 2.1x faster than state-of-the-art hybrid parallelism.
  - **It is prior art for CROWDio's forced replan on worker loss**, though it handles failure recovery, not voluntary cost-aware switching.
- **HexGen.** Jiang, Yan, Yao, Chen, Xu, Cui, Fu, Pan, Ma, ... Yuan, "HexGen: Generative Inference of Large Language Model over Heterogeneous Environment," ICML 2024, arXiv:2311.11514 [K]. Uses asymmetric tensor and pipeline parallelism with an evolutionary or constrained scheduling search over heterogeneous decentralised GPUs. Static plan, no switching cost.
- **Helix.** Mei, Zhuang, Miao, Vinayak, Rashmi, Jia, "Helix: Serving Large Language Models over Heterogeneous GPUs and Network via Max-Flow," ASPLOS 2025, arXiv:2406.01566 ([arXiv](https://arxiv.org/abs/2406.01566v2) [V]). Uses max-flow plus MILP for joint placement and per-request routing over 24–42 nodes, with up to 3.3x throughput and up to 66%/24% lower prompt/decode latency. Placement is solved offline. Its runtime adaptivity lies in per-request routing, not re-placement, so it has no switching cost.
- **Splitwise.** Patel, Choukse, Zhang, Shah, Goiri, Maleki, Bianchini, "Splitwise: Efficient Generative LLM Inference Using Phase Splitting," ISCA 2024, arXiv:2311.18677 [K]. Disaggregates prefill and decode onto different machines and transfers KV state between them. This is a state-migration cost, but it is per request, not a model reconfiguration. A different 2025 arXiv paper reuses the name ("Splitwise: Collaborative Edge-Cloud Inference for LLMs via Lyapunov-Assisted DRL," arXiv:2512.23310 [V]). Do not confuse the two.

### Inferences
- Reviewers will probably cite IONN for "model upload cost already considered" and Neurosurgeon/SPINN/DADS for "dynamic partitioning already done." CROWDio's defensible delta is the following combination: (a) a full C_switch with unload, conditional fetch, write, load, and session init, under a **hard single-residency constraint** on each device; (b) a break-even decision rule; and (c) multi-device pipeline placement, not a 2-party split.
- The 2026 Stockholm paper shows the state of the art still uses a percent-improvement margin with no reload cost. It is a good, recent "B2-like" related-work foil.

### Gaps
- Full texts of Learning to Partition (MDPI 403) and the Luger thesis (PDF text not extractable) were not read. Both could contain switching-cost modelling.
- The original paper behind the survey's "configuration graph to reduce repartitioning overhead" was not identified.

## 2. Model loading / cold start / switching cost in serving systems

### Takeaway
Datacenter serving treats model load and switch latency as first-class. It either eliminates the cost (PipeSwitch pipelining, BlitzScale and λScale execute-while-load, DeepFlow pre-warming and NPU-fork) or schedules around it (Clockwork, ServerlessLLM locality-aware placement, AlpaServe). The cost is well characterised (vLLM startup is mostly CPU-bound). None of these systems couple it with layer-cut selection on bandwidth-variable mobile devices.

### Cited Findings
- **PipeSwitch.** Bai, Zhang, Zhu, Jin, "PipeSwitch: Fast Pipelined Context Switching for Deep Learning Applications," OSDI 2020 ([USENIX](https://www.usenix.org/conference/osdi20/presentation/bai) [V]).
  - Pipelines layer-group transfer over PCIe with execution, and adds unified memory management and active-standby workers.
  - Task startup overhead of 3.6–6.6 ms and total overhead of 5.4–34.6 ms, which is 10–50x better than NVIDIA MPS.
  - It models and minimises the switch cost, with residency on the GPU, but it does not decide whether to switch.
  - It is analogous to CROWDio's standby-worker prefetch and B5b (async switching).
- **Clockwork.** Gujarati, Karimi, Alzayat, Hao, Kaufmann, Vigfusson, Mace, "Serving DNNs like Clockwork: Performance Predictability from the Bottom Up," OSDI 2020, arXiv:2006.02464 [K]. A central controller with predictable worker actions, where model LOAD to GPU memory is an explicit scheduled action with a predicted duration. It models load cost and residency (a GPU memory cache of models). This is the closest datacenter analogue to "residency state in the planner."
- **AlpaServe.** Li, Zheng, Zhong, Liu, Sheng, Jin, Huang, Chen, Zhang, Gonzalez, Stoica, "AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving," OSDI 2023, arXiv:2302.11665 [K]. Places partitioned models across GPUs to absorb bursts via simulator-guided search. Placement is periodic or static, and it does not price re-placement.
- **ServerlessLLM.** Fu, Xue, Huang, Brabete, Ustiugov, Patel, Mai, "ServerlessLLM: Low-Latency Serverless Inference for Large Language Models," OSDI 2024, arXiv:2401.14351 ([arXiv](https://arxiv.org/abs/2401.14351) [V]).
  - Multi-tier, loading-optimised checkpoints, live migration of inference, and locality-aware scheduling that picks the server minimising startup time given where checkpoints already reside.
  - 10–200x lower latency.
  - It **models residency (checkpoint locality) and load cost in placement.** This is directly analogous to CROWDio's "fetch only if absent" and best-donor C_fetch.
- **BlitzScale.** Zhang, Wang, Liu, Wei, Shan, Chen, Chen, "BlitzScale: Fast and Live Large Model Autoscaling with O(1) Host Caching," OSDI 2025, arXiv:2412.17246 ([arXiv](https://arxiv.org/abs/2412.17246) [V]). Multicasts parameters GPU-to-GPU over the compute network instead of from host caches. "Live" layer-level scaling lets overloaded instances offload layers to instances still loading. Up to 94% lower tail latency than ServerlessLLM and 49% less GPU time than DistServe/vLLM. **Its peer-to-peer parameter loading from other serving instances is prior art for CROWDio's peer-donor shard fetch** (datacenter RDMA rather than phones).
- **λScale.** "λScale: Enabling Fast Scaling for Serverless Large Language Model Inference," arXiv:2502.09922 ([summary](https://www.emergentmind.com/papers/2502.09922) [V via secondary]). RDMA model multicast plus "execute-while-load" pipelined execution across nodes still receiving weights. Up to 5x tail latency and 31.3% cost reduction. Authors should be checked from the arXiv page.
- **DeepFlow.** "DeepFlow: Serverless Large Language Model Serving at Scale," arXiv:2501.14417 ([deeppaper](https://arxiv.deeppaper.ai/papers/2501.14417) [V via secondary]). The FlowServe engine uses pre-warmed pods, DRAM pre-loading, and NPU-fork to scale to 64 instances in seconds. In production on an Ascend NPU cluster for over a year.
- **Tetris.** Li, Zhao, Yang, Zhan, Li, "Tetris: Memory-efficient Serverless Inference through Tensor Sharing," USENIX ATC 2022, not SC ([USENIX](https://www.usenix.org/conference/atc22/presentation/li-jie) [V]). Runtime- and tensor-level page sharing among colocated instances, saving up to 93% memory and giving 30x function density. Memory/residency-focused, with no switching decision.
- **Kabakibo et al. 2026.** Kabakibo, Trivedi, Wang, "Breaking the Ice: Analyzing Cold Start Latency in vLLM," MLSys 2026, arXiv:2606.07362 ([arXiv](https://arxiv.org/abs/2606.07362); [Paderborn](https://en.cs.uni-paderborn.de/cn/newsfolder/news/paper-accepted-to-mlsys-2026) [V]). Decomposes vLLM startup into six steps, finds it mostly CPU-bound, and provides a lightweight analytical startup-latency predictor. **This is strong support for CROWDio decomposing C_switch into unload/fetch/write/load/session-init and predicting each term** (session init being a CPU-bound analogue on ONNX Runtime Mobile).
- **SwapAdvisor.** Huang, Jin, Li, "SwapAdvisor: Pushing Deep Learning Beyond the GPU Memory Limit via Smart Swapping," ASPLOS 2020, DOI 10.1145/3373376.3378530 [K]. Genetic-algorithm search over operator schedule, memory allocation, and swap decisions. Swap cost is modelled within a single training or inference job (tensor granularity), not across devices.
- **Clipper.** Crankshaw, Wang, Zhou, Franklin, Gonzalez, Stoica, NSDI 2017, arXiv:1612.03079 [K]. **INFaaS.** Romero, Li, Yadwadkar, Kozyrakis, USENIX ATC 2021, arXiv:1905.13348 [K]. INFaaS chooses model variants and accounts for load latency when picking variants and for whether a variant is already loaded (residency-aware variant selection). Clipper uses containerised models with no residency-aware switching.
- **Serverless inference survey.** "A Survey of Serverless Machine Learning Model Inference," arXiv:2311.13587 ([arXiv](https://arxiv.org/pdf/2311.13587) [V]). Use it as a cold-start background reference.
- **Tangram.** "Tangram: Accelerating Serverless LLM Loading through GPU Memory Reuse and Affinity," arXiv:2512.01357 ([arXiv](https://arxiv.org/html/2512.01357v1) [V, existence]). Recent affinity- and residency-aware loading.

### Inferences
- The serving literature validates that load/switch cost matters and is predictable, which supports CROWDio's cost model. The novelty claim must avoid "first to consider model loading cost." The claim should be "first to put a residency-dependent switch cost inside the partition/placement objective and switching rule for a mobile crowd."
- B3 (load-time-aware without residency state) maps naturally to "Clockwork/INFaaS-style load cost without tracking what is resident or cached on disk." B5b (async switching) maps to PipeSwitch, BlitzScale, and λScale-style overlap.

### Gaps
- λScale and DeepFlow author lists and exact venues were not confirmed from primary pages.
- Exact Clockwork, AlpaServe, and INFaaS numbers were not re-verified.

## 3. Reconfiguration with migration/switching cost in distributed DL (elastic training, preemption, live migration)

### Takeaway
The closest conceptual prior art for "re-plan the parallel configuration, explicitly pricing migration" is in preemptible and elastic training and serving: SpotServe (KM matching to minimise migration communication), Parcae (proactive "liveput"), Oobleck, Bamboo, and Varuna. Llumnix prices request-state migration. These are datacenter GPU systems with reliable networks, not phones under single residency.

### Cited Findings
- **SpotServe.** Miao, Shi, Duan, Xi, Lin, Cui, Jia, "SpotServe: Serving Generative Large Language Models on Preemptible Instances," ASPLOS 2024, arXiv:2311.15566 ([arXiv](https://arxiv.org/abs/2311.15566) [V]).
  - Dynamically re-parallelises (DP/TP/PP degrees) as instances come and go.
  - Formulates instance migration as bipartite matching solved with Kuhn–Munkres to minimise communication. Uses stateful inference recovery (KV cache) at token granularity within the grace period.
  - P99 latency is 2.4–9.1x better, and cost is 54% lower than on-demand only.
  - **This is the strongest prior for "reconfiguration that explicitly minimises migration cost and resumes from retained state"**, the analogue of CROWDio's forced switch resuming from the stage-boundary activation. The difference: SpotServe is triggered by preemption, not by a voluntary break-even rule under noisy telemetry.
- **Parcae.** Duan, Song, Miao, Zhang, Shi, Lin, Jia, et al., "Parcae: Proactive, Liveput-Optimized DNN Training on Preemptible Instances," NSDI 2024 ([USENIX](https://www.usenix.org/conference/nsdi24/presentation/duan) [K]). Predicts future availability and picks parallel configurations that maximise "liveput" (expected throughput under preemptions), accounting for migration cost between configurations. **This is a proactive, cost-aware reconfiguration planner. Reviewers may raise it against CROWDio's switching rule.**
- **Oobleck.** Jang, Yang, Zhang, Jin, Chowdhury, "Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates," SOSP 2023, arXiv:2309.08125 [K]. Precomputes pipeline templates so that reconfiguration after failure is fast. This is template-based (precomputed plans), similar in spirit to CROWDio's exhaustive oracle and plan cache.
- **Bamboo.** Thorpe, Zhao, Eyolfson, Qiao, Jia, Zhang, Netravali, Xu, "Bamboo: Making Preemptible Instances Resilient for Affordable Training of Large DNNs," NSDI 2023, arXiv:2204.12013 [K]. Redundant computation in pipeline bubbles to survive preemption.
- **Varuna.** Athlur, Saran, Sivathanu, Ramjee, Kwatra, "Varuna: Scalable, Low-cost Training of Massive Deep Learning Models," EuroSys 2022, arXiv:2111.04007 [K]. Job morphing reconfigures PP/DP on spot VMs, with checkpoint-based reconfiguration.
- **Pollux.** Qiao, Choe, Subramanya, Neiswanger, Ho, Zhang, Ganger, Xing, "Pollux: Co-adaptive Cluster Scheduling for Goodput-Optimized Deep Learning," OSDI 2021, arXiv:2008.12260 [K]. Periodically re-allocates resources to maximise goodput and penalises re-allocation (restart/checkpoint delay) in its fitness to avoid thrashing. **This is a scalarised switch-penalty precedent.**
- **Gandiva.** Xiao et al., OSDI 2018 ([USENIX](https://www.usenix.org/conference/osdi18/presentation/xiao) [K]). Introspective scheduling with job migration, suspend/resume, and time-slicing, with migration cost measured. **Optimus.** Peng, Bao, Chen, Wu, Guo, EuroSys 2018, DOI 10.1145/3190508.3190517 [K]. Performance-model-driven resource allocation with checkpoint-based scaling overhead. Both are pre-2019 background.
- **Llumnix.** Sun, Huang, Zhao, Xiao, Zhang, Li, Lin, "Llumnix: Dynamic Scheduling for Large Language Model Serving," OSDI 2024, arXiv:2406.03243 [K]. Live migration of requests and KV cache across instances, with near-zero downtime via staged copy, for load balancing, de-fragmentation, and priorities. State-migration cost is modelled and hidden. Model placement is not changed.
- **Asteroid** (see Section 1). Fault-tolerant pipeline replay on edge devices, with 14x faster recovery ([arXiv](https://arxiv.org/html/2408.08015v1) [V]).

### Inferences
- The SpotServe, Parcae, Oobleck, and Pollux cluster makes "reconfiguration with migration cost" not novel in general. CROWDio should frame its novelty as follows: (i) the switch cost is dominated by per-device model residency transitions (fetch, write, load, and session init on phones), not network state transfer; (ii) telemetry is noisy and stale, so the controller needs a stale-estimate refusal plus persistence; and (iii) voluntary versus forced epochs have different semantics (drain versus resume from the boundary activation).
- A useful sentence for the paper: "SpotServe and Parcae optimise reconfiguration triggered by preemption in datacenters. CROWDio additionally decides *whether* to voluntarily reconfigure under bandwidth drift, using a residency-aware break-even rule."

### Gaps
- Parcae's exact liveput and migration-cost formulation and Pollux's re-allocation penalty were not re-read in this session. Verify both before claiming the contrast.

## 4. Hysteresis / break-even / ski-rental switching rules and online algorithms with switching costs

### Takeaway
Theory for paying a hitting cost plus a switching cost exists in metrical task systems, SOCO, and ski rental, and has been applied to data-center right-sizing and edge service migration. I found **no paper applying SOCO/MTS/ski-rental with a residency-dependent switch cost to DNN cut and placement selection.** Systems practice in DNN partitioning uses ad-hoc margins (for example, a 3% threshold).

### Cited Findings
- **Metrical task systems.** Borodin, Linial, Saks, "An Optimal On-line Algorithm for Metrical Task System," JACM 39(4), 1992, DOI 10.1145/146585.146588 [K]. Gives a (2n−1)-competitive deterministic bound for n states. CROWDio's discrete plan space with plan-to-plan C_switch is exactly an MTS. This gives a theoretical anchor for the break-even rule, which is a work-function or "switch when accumulated regret exceeds switch cost" style heuristic.
- **Ski rental.** Karlin, Manasse, Rudolph, Sleator, "Competitive Snoopy Caching," Algorithmica 3, 1988 [K]. Buy-versus-rent break-even gives a 2-competitive rule, the canonical justification for "switch once cumulative excess cost ≥ switch cost." CROWDio's persistence requirement approximates this.
- **SOCO.**
  - Lin, Wierman, Andrew, Thereska, "Dynamic Right-Sizing for Power-Proportional Data Centers," INFOCOM 2011 / IEEE/ACM ToN 2013 [K], and Lin and Wierman "Online Optimization with Switching Cost" ([abstract PDF](http://www.sigmetrics.org/mama/2012/abstracts/Lin.pdf) [V]). These cover the hitting cost plus switching cost formulation and the lazy-capacity-provisioning algorithm.
  - Chen, Goel, Wierman, "Smoothed Online Convex Optimization in High Dimensions via Online Balanced Descent," COLT 2018, arXiv:1803.10366 ([PMLR](http://proceedings.mlr.press/v75/chen18b/chen18b.pdf) [V]).
  - Predictions variants: "Online Convex Optimization Using Predictions," arXiv:1504.06681 [V]; "Smoothed Online Optimization with Unreliable Predictions," arXiv:2202.03519 [V].
  - Delayed feedback: IJCAI 2025 ([PDF](https://www.ijcai.org/proceedings/2025/0758.pdf) [V]). This is relevant because CROWDio's telemetry is stale and delayed.
- **Edge service migration with migration cost.**
  - Wang, Urgaonkar, Zafer, He, Chan, Leung, "Dynamic Service Migration in Mobile Edge Computing Based on Markov Decision Process," IEEE/ACM ToN 2019, arXiv:1506.05261 [K]. MDP with explicit migration cost and threshold policies.
  - Ouyang, Zhou, Chen, "Follow Me at the Edge: Mobility-Aware Dynamic Service Placement for Mobile Edge Computing," IEEE JSAC 2018, arXiv:1809.05239 [K]. Lyapunov optimisation with a long-term migration-cost budget.
  - **These are the most direct "placement + switching cost" precedents at the edge**, but they place whole services (VM/container), not DNN shards with cut selection.
- Per the targeted searches, no DNN-partitioning paper combining an explicit switching cost with regret or Lyapunov analysis was found. Hierarchical-inference regret papers (arXiv:2304.00891, arXiv:2508.08985 [V]) study offload-or-not, not partition switching.

### Inferences
- CROWDio can strengthen its rule by citing MTS and ski-rental as motivation and, ideally, stating a simple competitive bound for persistence plus a break-even rule on two plans. Without this, reviewers may call the margin, persistence, and cooldown ad-hoc.
- Stale-estimate refusal links to SOCO with delayed feedback or unreliable predictions. That is a good citation pair.

### Gaps
- No empirical comparison of hysteresis versus SOCO-style algorithms for DNN placement was found in the literature.

## 5. Prefetching models/weights and peer-to-peer model distribution

### Takeaway
Edge model caching (TrimCaching), P2P/multicast weight loading (BlitzScale, λScale), and swarm serving (Petals) exist. None combines standby-disk prefetch on phones, a best-of-server-or-peer donor fetch cost inside a placement objective, and single-residency.

### Cited Findings
- **TrimCaching.** "TrimCaching: Parameter-sharing AI Model Caching in Wireless Edge Networks," ICDCS 2024, arXiv:2405.03990 and arXiv:2404.14204 ([arXiv](https://arxiv.org/html/2404.14204v3) [V]). Places models on storage-limited edge servers to maximise the hit ratio within download-latency requirements, exploiting shared parameters across models.
- **DMCPA.** Joint DNN model caching and processor allocation at edge servers, IWQoS 2022 ([PDF](https://cis.temple.edu/~wu/research/publications/Publication_files/iwqos22final.pdf) [V, existence]).
- **CacheNet.** A model caching framework for DL inference on the edge ([PwC](https://astro.paperswithcode.com/paper/cachenet-a-model-caching-framework-for-deep) [V, existence]).
- **BlitzScale / λScale** (Section 2). GPU-to-GPU and RDMA multicast of parameters from peers already holding them, plus execute-while-load. This is the datacenter analogue of peer donors.
- **Petals.** Borzunov, Baranchuk, Dettmers, Ryabinin, Belkada, Chumachenko, Samygin, Raffel, "Petals: Collaborative Inference and Fine-tuning of Large Models," ACL 2023 Demos, arXiv:2209.01188, and "Distributed Inference and Fine-tuning of Large Language Models Over the Internet," NeurIPS 2023, arXiv:2312.08361 [K].
  - Volunteer servers each hold a contiguous block of layers. Clients route through a swarm and pick servers by measured latency and throughput.
  - Servers periodically **rebalance** which blocks they host to fix throughput bottlenecks, and clients fail over to other servers holding the same blocks.
  - **This is the most similar crowd system to CROWDio.** It has dynamic re-hosting of layer blocks with a load cost and volunteer churn. Read its load-balancing section and position CROWDio explicitly. As far as I recall, Petals' rebalancing is a greedy throughput heuristic, not a break-even rule with switching cost (unverified).
- **IONN** (Section 1). Incremental upload ordering is a form of prefetch ordering.
- **Two-timescale placement** (arXiv:2105.13618, Section 1). Layer placement happens on a slower timescale than splitting.

### Inferences
- "Prefetch to standby disk" is analogous to ServerlessLLM's multi-tier checkpoint locality, DeepFlow's DRAM pre-loading, and PipeSwitch's standby workers. CROWDio's novelty is applying this under phone storage and RAM limits with an explicit absent/present term in C_switch.

### Gaps
- No paper on peer-to-peer phone-to-phone DNN weight distribution for inference was found in the time available (federated-learning P2P model dissemination exists but was not surveyed).

## 6. Pipeline-parallel inference throughput models and DP / Pareto partitioning

### Takeaway
CROWDio's pipeline model (finish recurrence; steady-state period P = max over stages of compute and communication; T(N) = fill + (N−1)P) is standard (GPipe, PipeDream, PipeEdge, EdgeShard). DP over layer compositions × device orders is also standard. Potentially novel elements: exact Pareto DP over (latency, throughput, switch cost), uncertainty bounds from EWMA variance, and switch cost as a DP objective term.

### Cited Findings
- **GPipe.** Huang et al., NeurIPS 2019, arXiv:1811.06965 [K]. Micro-batch pipeline, with bubble overhead of (K−1)/(M+K−1).
- **PipeDream.** Narayanan, Harlap, Phanishayee, Seshadri, Devanur, Ganger, Gibbons, Zaharia, "PipeDream: Generalized Pipeline Parallelism for DNN Training," SOSP 2019, arXiv:1806.03377 [K]. A DP partitioner minimises the slowest stage (compute plus communication) over hierarchical topologies. This is the canonical DP planner CROWDio's DP generalises.
- **PipeEdge.** Hu, Imes, Zhao, Kundu, Beerel, Crago, Walters, "PipeEdge: Pipeline Parallelism for Large-Scale Model Inference on Heterogeneous Edge Devices," Euromicro DSD 2022, pp. 298–307, DOI 10.1109/DSD57027.2022.00048, arXiv:2110.14895 ([ISI](https://www.isi.edu/results/publications/10310/pipeedge-pipeline-parallelism-for-large-scale-model-inference-on-heterogeneous-edge-devices); [arXiv](https://www.arxiv.org/pdf/2110.14895) [V]).
  - DP over layer partitions and heterogeneous devices and links, minimising the bottleneck stage.
  - 11.88x (ViT-Huge) and 12.78x (BERT-Large) speedup with 16 edge devices. Up to 4.16x throughput over PipeDream on heterogeneous devices.
  - Static planning, no switch cost. **It is the most direct prior for CROWDio's DP and throughput model.**
- **EdgeShard** (Section 1). DP for latency and throughput separately, with memory budgets.
- **Alpa.** Zheng, Li, Zhang, Zhuang, Chen, Huang, Wang, Xu, Zhuo, Xing, Gonzalez, Stoica, "Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning," OSDI 2022, arXiv:2201.12023 [K]. Inter-operator DP over stage slicing and submesh assignment, plus intra-operator ILP.
- **Helix** (MILP/max-flow) and **HexGen** (heterogeneous asymmetric partitioning). Heterogeneous placement search, static.
- **Pareto / multi-objective partitioning.** Edgent's latency-accuracy trade-off and JALAD's accuracy-latency ILP are two-objective. I did not find, in this session, a DNN partitioning paper that computes an **exact Pareto front including a reconfiguration-cost objective**. Multi-exit joint partitioning with DRL (Electronics 14(8):1647, 2025, DOI 10.3390/electronics14081647 [V, existence]) uses scalarised rewards.

### Inferences
- The throughput recurrence and DP must be presented as building on GPipe, PipeDream, and PipeEdge, not as novel. Novelty lies in the additional state (residency, cached-on-disk) and the switch-cost dimension in the DP, plus uncertainty bounds.

### Gaps
- A targeted search for "Pareto DP pipeline partition heterogeneous" was not run due to the call budget, so a Pareto-DP precedent may exist (for example, in multi-objective DNN mapping on MPSoCs or accelerators). This is a reviewer risk.

## 7. Gap analysis: what CROWDio does that prior work has not (and where reviewers may push back)

### Takeaway
No single prior system found combines: (1) runtime cut **and** device selection for a multi-stage pipeline on phones; (2) a switch cost that depends on per-device **residency state** (single-residency, on-disk cache, best server or peer donor); (3) a break-even, persistence, cooldown, and switch-cap controller that refuses voluntary switches on stale estimates; and (4) epoch semantics (drain versus resume from the retained boundary activation). Each ingredient individually has strong precedents, so the novelty claim must be compositional and empirical.

### Cited Findings
Prior-art map. "Explicitly models switching cost?" asks whether the work prices switching; "Residency state?" asks whether it tracks what is loaded or cached.

| Work | Dynamic cut/placement? | Explicitly models switching cost? | Residency state? | Gap vs CROWDio |
|---|---|---|---|---|
| Neurosurgeon, DADS, JALAD, Edgent, SPINN, ANS | Yes (2-party cut) | No | No (full model on both ends) | No reload cost; 2 parties only |
| IONN ([PDF](https://cse.buffalo.edu/faculty/tkosar/cse710_spring19/jeong-socc18.pdf)) | Incremental | **Yes**, upload cost (one-shot) | Partial (what has been uploaded) | No unload/reload cycling, no hysteresis, 1 server |
| Deng et al. 2026 ([arXiv](https://arxiv.org/html/2605.09623v1)) | Yes, windowed | No (3% margin only) | No | Margin without cost |
| EdgeShard, PipeEdge, Galaxy, HexGen, Helix | Static or once | No | Memory budget only | No re-plan |
| PipeSwitch, Clockwork, ServerlessLLM, BlitzScale, λScale, INFaaS | Model or instance placement | **Yes** (load/switch cost) | **Yes** (GPU or host cache) | Datacenter; no cut selection under bandwidth drift |
| SpotServe, Parcae, Oobleck, Varuna, Pollux | Reconfigure parallelism | **Yes** (migration or restart) | Partial | Preemption-driven; datacenter; not single-residency phones |
| Asteroid ([arXiv](https://arxiv.org/html/2408.08015v1)) | Replan on device exit | Recovery cost minimised | No | Training; failure-only |
| Petals | Block rebalancing | Load cost implicit (unverified) | Yes (blocks per server) | Closest crowd system; check its rebalancing rule |
| Wang 2019 / Ouyang 2018 service migration | Service placement | **Yes** (migration cost; MDP or Lyapunov) | Yes (where the service is) | Whole service, no DNN cut |
| SOCO / MTS / ski rental | Theory | **Yes** | n/a | Not applied to DNN cut and placement |

Claims that appear novel (no counter-example found in this survey):
1. The switch cost is decomposed into unload + conditional (fetch + write) + load + session init. The structure is supported by Kabakibo et al. 2026's six-step decomposition ([arXiv](https://arxiv.org/abs/2606.07362)), but it is evaluated per (shard, device) under **single residency on phones**.
2. That residency-dependent switch cost is embedded **inside** the cut × placement search (exhaustive oracle, exact Pareto DP, scalarised DP) rather than handled by a separate loading layer.
3. A voluntary-switch controller with break-even + margin + persistence + cooldown + per-window cap + **stale-estimate refusal**, driven by EWMA mean/variance telemetry with age and sample counts.
4. Epoch semantics distinguishing voluntary (drain) from forced (resume from the retained stage-boundary activation if the cut is kept).
5. A baseline ladder (B0–B5) that isolates the value of residency awareness (B3 versus full) and async switching (B5a/b).

Claims reviewers may challenge, with the precedent they will cite:
- "Dynamic partitioning under bandwidth": Neurosurgeon, DADS, SPINN, Edgent, Deng 2026.
- "Model upload/loading cost": IONN, PipeSwitch, Clockwork, ServerlessLLM, BlitzScale, Kabakibo 2026.
- "Reconfiguration with migration cost": SpotServe (KM migration plan, stateful recovery), Parcae (liveput), Pollux (re-allocation penalty), Llumnix (live migration).
- "Hysteresis/threshold switching": service-migration MDP threshold policies (Wang 2019), ski rental, MTS. The 3% margin in Deng 2026.
- "Peer weight fetch / prefetch": BlitzScale, λScale (peer multicast), ServerlessLLM (locality), Petals (swarm), TrimCaching (edge model caching).
- "DP pipeline partitioning on heterogeneous devices": PipeDream, PipeEdge, EdgeShard, Alpa.
- "Resume after failure": Asteroid pipeline replay, SpotServe stateful recovery, Oobleck templates.
- "Single-partition residency / JIT loading": CROWDio's own arXiv:2605.20723. Self-overlap must be disclosed.

### Inferences
- The safest framing: "Switching-cost-aware partitioning has been studied for datacenter GPUs (SpotServe, Parcae, ServerlessLLM) and model-upload ordering for single-server offloading (IONN). We study it where it matters most and has not been studied: volunteer phones that can hold only one live session, where switching costs seconds and is comparable to the latency gains being chased." Back this with measured C_switch on real handsets versus measured gains. That ratio is the empirical core.
- Add a small theoretical hook (MTS/ski-rental competitive argument for the break-even plus persistence rule) to pre-empt "ad-hoc thresholds" criticism.
- Petals and the Luger 2023 TU Wien thesis are the two highest-risk unread items. Read both before submission.

### Gaps
- Full texts not read this session: Petals (rebalancing rule), Parcae (migration-cost model), the Luger thesis, Learning to Partition (MAKE 2025), and the ACM CSUR survey section on repartitioning overhead.
- Items marked [K] (Clockwork, AlpaServe, Llumnix, Parcae, Pollux, Oobleck, Bamboo, Varuna, Gandiva, Optimus, DADS, JALAD, Edgent, SPINN, ANS, Galaxy, HexGen, Splitwise, GPipe, PipeDream, Alpa, Petals, MTS, ski rental, Wang 2019, Ouyang 2018) were cited from established knowledge with canonical IDs and were not re-fetched. Verify DOIs and arXiv IDs in the bibliography pass.
- "DDPQN" and "CrowdNet" could not be found as named systems. Treat them as possibly mis-remembered.
