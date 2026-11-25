# Mobile / Android Worker Side of CROWDio: Literature Survey (2019–2026)

Scope: the CROWDio Android worker. It is a Kotlin foreground service with Chaquopy Python and ONNX Runtime Android / TFLite. It runs DistilBERT in 3 stages across phones, loads shards just in time (JIT), keeps only one partition resident at a time, uses an LRU shard cache, sends zlib+base64 tensors over a WebSocket relay or device-to-device, collects telemetry, and supports checkpoint/resume.
Verification policy: an entry appears under "Cited Findings" only if a search result or full-text fetch in this session confirmed it exists. Items I remember but could not confirm are listed under "Gaps".
Full texts or HTML read beyond the abstract: LinguaLinked (arXiv HTML), LOIP (arXiv HTML, including its related-work table). Abstract pages read: EdgePipe, EdgeShard, NNV12, Jupiter, CROWDio preprint. The NNV12 PDF was downloaded but could not be text-extracted (binary stream). For everything else I relied on search snippets and abstracts.

**Important self-citation finding:** the CROWDio pipeline subsystem already exists as a preprint. Lakshani Manamperi, Disumi Pathirana, Thiwanka Pathirana, Nipun Premarathna, Kutila Gunasekera, "Memory-Efficient Partitioned DNN Inference on Resource-Constrained Android Crowds", arXiv:2605.20723 (20 May 2026), accepted at the ICML 2026 Workshop on ML for the Global South. It reports peak per-device RSS of 43 ± 2 MB and 50 ± 3 mAh per run on 5 handsets × 10 runs. Streaming concurrency cuts batch latency by 34% versus a barrier. Phones in the study have 3.3–7.4 GB RAM. It lists five mechanisms: JIT deferred loading, single-partition residency, a 4-tier affinity scheduler, zlib tensor transport, and a streaming 1:1 dependency model. — [arXiv:2605.20723](https://arxiv.org/abs/2605.20723)

---

## 1. Multi-device / collaborative DNN, transformer and LLM inference across phones and edge devices

### Takeaway
Most collaborative-inference systems run on Jetsons, Raspberry Pis or laptops, not volunteer phones. They also assume a static, trusted, always-on device set with every partition resident in memory. Device churn, model-load and reload cost, and per-device memory bounds are rarely first-class concerns. LinguaLinked is the closest precedent, a multi-Android LLM pipeline, but it does not handle failures or churn. Its reported shard reload costs (0.23–2.45 s) are far below CROWDio's measured ~48 s cold load. That makes reload cost a distinctive CROWDio concern.

### Cited Findings

**Classics (pre-2019, for lineage)**
- **MoDNN.** Mao et al., "MoDNN: Local distributed mobile computing system for Deep Neural Network," DATE 2017. Layer-wise partitioning across WLAN-connected phones. Its MapReduce-like execution moves a lot of intermediate data between devices. One citing reference gives the first author as "Chen", so check the DATE proceedings for author order. — [search results via DeepThings and related citations](http://slam.ece.utexas.edu/pubs/codes18.DeepThings.pdf)
- **DeepThings.** Z. Zhao, K. M. Barijough, A. Gerstlauer, "DeepThings: Distributed Adaptive Deep Learning Inference on Resource-Constrained IoT Edge Clusters," IEEE TCAD 2018 (CODES+ISSS 2018). Uses Fused Tile Partitioning of the early conv layers plus distributed work-stealing. Reports 1.7×–3.5× speedup on 2–6 devices with <23 MB of memory each. It is CNN-only, assumes a static cluster, and does not treat churn. — [PDF](http://slam.ece.utexas.edu/pubs/codes18.DeepThings.pdf)
- **AOFL.** L. Zhou et al., "Adaptive Parallel Execution of Deep Neural Networks on Heterogeneous Edge Devices," ACM/IEEE SEC 2019. Extends DeepThings' layer fusion and dynamically selects the degree of parallelism for heterogeneous devices. I did not find the full text, so this description rests on secondary citations. — [cited in DistrEdge arXiv:2202.01699](https://arxiv.org/pdf/2202.01699)

**CNN-era (2019–2023)**
- **DeepSlicing.** S. Zhang et al., "DeepSlicing: Collaborative and Adaptive CNN Inference With Low Latency," IEEE TPDS 32(9):2175–2187, 2021, DOI 10.1109/TPDS.2021.3058532. Fine-grained model and data partitioning with a Proportional Synchronized Scheduler. Reports up to 5.79× latency reduction. — [PDF](https://cis.temple.edu/~wu/research/publications/Publication_files/DeepSlicing_Collaborative_and_Adaptive_CNN_Inference_With_Low_Latency.pdf)
- **DistrEdge.** X. Hou, Y. Guan, T. Han, N. Zhang, "DistrEdge: Speeding up Convolutional Neural Network Inference on Distributed Edge Devices," **IPDPS 2022** (pp. 1097–1107), arXiv:2202.01699. Note that the task brief said ICDCS 2022, but the correct venue is IPDPS. It uses deep RL to choose CNN splits across heterogeneous edge devices. — [arXiv](https://arxiv.org/abs/2202.01699)
- **CoEdge.** L. Zeng et al., IEEE/ACM Transactions on Networking (≈2020/2021). Partitions workload adaptively, sends only overlapping feature-map regions to neighbors, and varies the number of participating devices. PICO criticizes it for needing per-layer communication, which is costly over wireless. — [PICO related work, arXiv:2206.08662](https://arxiv.org/pdf/2206.08662); [LOIP related work](https://arxiv.org/html/2512.21835)
- **PICO.** X. Yang et al., "PICO: Pipeline Inference Framework for Versatile CNNs on Diverse Mobile Devices," IEEE TMC, 2023 (arXiv:2206.08662). Splits both the CNN DAG and the device set into pipeline stages using graph partition plus many-to-many mapping. Reports 1.8×–6.8× throughput on 2–8 Raspberry Pis. This is the closest stage-pipelining precedent for CNNs, but it uses no phones and has no churn handling. — [arXiv](https://arxiv.org/pdf/2206.08662); [PolyU record](https://research.polyu.edu.hk/en/publications/pico-pipeline-inference-framework-for-versatile-cnns-on-diverse-m/)

**Transformer / LLM pipeline and tensor parallel (2021–2026)**
- **EdgePipe / PipeEdge.** Y. Hu, C. Imes, X. Zhao, S. Kundu, P. A. Beerel, S. P. Crago, J. P. N. Walters. Preprint "Pipeline Parallelism for Inference on Heterogeneous Edge Computing," arXiv:2110.14895 (Oct 2021, system called EdgePipe). Published as "PipeEdge," Euromicro DSD 2022. The partitioning is optimal with respect to compute, memory and bandwidth heterogeneity. On 16 edge devices it reports 10.59× (ViT-Large) and 11.88× (ViT-Huge) speedup, 3.93× over a 4-node baseline that cannot fit the model, and up to 4.16× over PipeDream on a heterogeneous set. It has no churn or failure handling, and LOIP lists it as having "no online adaptation". — [arXiv abs](https://arxiv.org/abs/2110.14895); [LOIP table](https://arxiv.org/html/2512.21835)
- **Galaxy.** S. Ye, J. Du, L. Zeng, W. Ou, X. Chu, Y. Lu, X. Chen, "Galaxy: A Resource-Efficient Collaborative Edge AI System for In-situ Transformer Inference," IEEE INFOCOM 2024, arXiv:2405.17245. Pools idle trusted home devices and uses a hybrid of tensor and sequence parallelism with a heterogeneity-aware planner. Reports up to 2.5× lower latency on Jetson Nano testbeds and is about 1.5k LoC on PyTorch. LOIP notes Galaxy assumes each device holds its TP slice, so it fails under tight memory, and that TP needs frequent synchronization (bad over a WAN relay). — [arXiv](https://arxiv.org/abs/2405.17245); [LOIP](https://arxiv.org/html/2512.21835)
- **EdgeShard.** M. Zhang, J. Cao, X. Shen, Z. Cui, "EdgeShard: Efficient LLM Inference via Collaborative Edge Computing," arXiv:2405.14371 (2024), later in IEEE Internet of Things Journal (per LOIP). Joint device selection and partitioning by dynamic programming over edge devices and cloud, evaluated on Llama2. Reports up to 50% lower latency and 2× throughput. Placement is static, and LOIP says it fails with OOM when slices must stay resident. — [arXiv](https://arxiv.org/abs/2405.14371); [LOIP](https://arxiv.org/html/2512.21835)
- **LinguaLinked (closest phone precedent; full text read).** J. Zhao, Y. Song, S. Liu, I. G. Harris, S. Abdu Jyothi, "LinguaLinked: A Distributed Large Language Model Inference System for Mobile Devices," arXiv:2312.00388 (Dec 2023).
  - Models: BLOOM 1.1B, 1.7B and 3B, at fp32 and int8.
  - Devices: 3× Pixel 7 Pro (12 GB) + 1× CUBOT X30 (8 GB), CPU only.
  - Runtime: ONNX Runtime C++ API in an Android app, with ZeroMQ ROUTER-DEALER messaging.
  - Placement and data flow: a linear program places segments. Activations travel device-to-device in a ring and are "never sent to the server".
  - Load balancing: re-balances on bandwidth, memory and FLOP/s, using an LP that overlaps "movable" sub-modules across adjacent devices. This gives a 1.29× speedup (BLOOM-1.7B fp32, 4.624 → 3.587 s/token) and 1.32× for int8 (2.756 → 2.087 s/token). Overall speedups are 1.11–1.61× single-threaded and 1.73–2.65× multi-threaded.
  - Loading: inactive sub-modules sit on storage and are loaded on demand. Reload per balancing event is ~2.449 s at fp32 versus 0.234 s at int8 (BLOOM-1.7B). It does not quantify initial model loading.
  - Gaps relative to CROWDio: no failure or churn handling, and the devices are trusted, co-located and assumed always on. — [arXiv abs](https://arxiv.org/abs/2312.00388); [HTML full text](https://arxiv.org/html/2312.00388)
- **Hermes / Hermes+.** X. Han, Z. Cai, …, R. Buyya, "Hermes: Memory-Efficient Pipeline Inference for Large Models on Edge Devices," arXiv:2409.04249 (Sep 2024). Its PipeLoad mechanism loads layers in a pipeline that overlaps loading with execution under a memory budget, with a Layer Profiler, Pipeline Planner and Execution Engine. Against PipeSwitch it reports up to 4.24× faster inference and 86.7% lower memory for BERT/ViT, and 2.58× with 90.3% lower memory for GPT-style models (BERT-Large, GPT-2, ViT-L, GPT-J). Evaluation was on a CPU server cluster, not phones. An extended version, "Hermes+: Adaptive Memory-Efficient Pipeline Inference for Large Models on Edge Devices", is in an ACM journal (DOI 10.1145/3845607) and reports up to 4.49× and 81.1% lower memory. This is the closest precedent to CROWDio's load-on-demand plus bounded residency, but it works within a single host. — [arXiv](https://arxiv.org/abs/2409.04249v2); [Hermes+ DOI](https://doi.org/10.1145/3845607)
- **TPI-LLM.** Z. Li et al., "TPI-LLM: Serving 70B-scale LLMs Efficiently on Low-resource Edge Devices," arXiv:2410.00531 (2024), later in IEEE Transactions on Services Computing 2025 (per LOIP). Tensor parallelism plus a sliding-window memory scheduler that overlaps disk I/O with compute. It finds link latency matters more than bandwidth and uses a star allreduce. Reports >80% lower TTFT and token latency than Accelerate, >90% lower than Transformers and Galaxy, and ~3.1 GB peak memory for 70B-scale models (secondary summaries give 11.3 GB/device for Llama-3.1-70B; the numbers conflict). Evaluated on laptops, not phones. — [aitopics/arXiv listing](https://aitopics.org/doc/arxivorg:AC11DC10); [LOIP](https://arxiv.org/html/2512.21835)
- **Jupiter.** S. Ye, B. Ouyang, L. Zeng, T. Qian, X. Chu, J. Tang, X. Chen, "Jupiter: Fast and Resource-Efficient Collaborative Inference of Generative LLMs on Edge Devices," IEEE INFOCOM 2025, arXiv:2504.08242. Uses intra-sequence pipeline parallelism for prefill and outline-based pipeline-parallel decoding with speculative decoding. Reports up to 26.1× lower end-to-end latency. LOIP criticizes it for assuming each partition stays resident and for the extra resident memory of its draft heads. — [arXiv](https://arxiv.org/abs/2504.08242)
- **LOIP (2025–2026, newest).** M. Sun, X. Zhang, S. Qu, Y. Li, M. Xiao, Y. Zheng, Y. Yuan, D. Yu, "LOIP: Collaborative LLM Inference with Offloading-based Pipeline Parallelism on Edge Devices," arXiv:2512.21835 (v3 2026). Combines SSD offloading with interleaved pipeline parallelism, overlapping weight loading with compute and communication. Uses DP plus block-level refinement and online memory adaptation for KV growth and bandwidth swings (tested at 50–250 Mbps). Hardware: Jetson Xavier NX and AGX Orin, running LLaMA2-13B, Qwen3-32B and LLaMA3.3-70B. Reports up to 8.8× (single input) and 20.3× (concurrent input) speedup. It explicitly names "weights stay resident" as the common flawed assumption in Galaxy, Jupiter and EdgeShard, which supports CROWDio's single-residency design. — [arXiv HTML](https://arxiv.org/html/2512.21835)
- **MDI-LLM.** arXiv:2505.18164 (2025). Model-distributed LLM inference over low-power edge nodes using "recurrent pipeline parallelism" to reduce idle time. I only saw the abstract. — [arXiv PDF](https://arxiv.org/pdf/2505.18164)
- **Over-the-air distributed on-device LLM inference.** K. Zhang et al., arXiv:2502.12559 (2025), with a follow-up at arXiv:2503.14882. Tensor parallelism with over-the-air all-reduce reports up to 3× faster generation. It is wireless-PHY oriented. — [arXiv](https://arxiv.org/abs/2502.12559)
- **Voltage.** Hu & Li, ICDCS 2024. Automatic partitioning of transformers for distributed edge inference, cited by LOIP. — [LOIP related work](https://arxiv.org/html/2512.21835)
- **Band.** Jeong et al., MobiSys 2022. Coordinates multiple DNNs across heterogeneous processors within one phone (intra-device, not multi-device). — [LOIP related work](https://arxiv.org/html/2512.21835)
- **Confidant.** Y. Chen, Y. Yan, Q. Yang, Y. Shu, S. He, J. Chen, "Confidant: Customizing Transformer-based LLMs via Collaborative Edge Training," arXiv:2311.13381. Published version: "…via Collaborative Training on Mobile Devices," MobiCom 2025 (DOI 10.1145/3680207.3723487). Splits an LLM into sub-models that each fit one phone's memory and uses pipeline-parallel *training* across commodity phones, assigning attention heads to mobile CPU/GPU. This is a close phone-pipeline precedent, but for fine-tuning rather than inference. — [arXiv](https://arxiv.org/abs/2311.13381); [ACM](https://dl.acm.org/doi/10.1145/3680207.3723487)
- **Survey.** "Embedded Distributed Inference of Deep Neural Networks: A Systematic Review," arXiv:2405.03360 (2024). Useful as a taxonomy citation. — [arXiv](https://arxiv.org/html/2405.03360v1)
- **Survey (2026).** "Network Edge Inference for Large Language Models: Principles, Techniques, and Opportunities," arXiv:2604.22906. — [arXiv](https://arxiv.org/pdf/2604.22906)

**Mobile crowd / volunteer computing**
- **Mobile crowd computing survey.** P. K. D. Pramanik, S. Pal, P. Choudhury, "Mobile crowd computing: potential, architecture, requirements, challenges, and applications," J. Supercomputing (Springer), 2023, DOI 10.1007/s11227-023-05545-0. Describes publicly owned phones forming an opportunistic, dynamic ad-hoc grid. — [ACM/Springer](https://dl.acm.org/doi/10.1007/s11227-023-05545-0)
- **Android volunteer computing security.** Bibi et al., "Secure Distributed Mobile Volunteer Computing with Android," ACM TOIT (DOI 10.1145/3428151). Treats Android malware as the main threat to mobile volunteer computing. — [ACM](https://dl.acm.org/doi/10.1145/3428151)
- **Open-collaboration training.** Diskin et al., "Distributed Deep Learning in Open Collaborations," arXiv:2106.10207 (NeurIPS 2021). Notes few successful applications of volunteer computing to ML, because of communication demands. — [arXiv](https://arxiv.org/pdf/2106.10207)
- **DreamLab.** Imperial College / Vodafone Foundation, 2015. A deployed Android/iOS volunteer app, useful as a practical precedent. — [Wikipedia](https://en.wikipedia.org/wiki/DreamLab)

### Inferences
- In the systems surveyed, the testbeds are Jetson, RPi, laptops and CPU servers. Only LinguaLinked (inference) and Confidant (training) use real Android phones. None treats devices as untrusted, intermittently available volunteers. CROWDio's combination of 3-stage transformer pipelining, a WAN relay through the server, checkpoint/resume and volunteer phones has no direct precedent found.
- LOIP (2026) and Hermes argue that "resident-weights" assumptions break under memory pressure. This supports CROWDio's single-partition-residency design. Unlike CROWDio, both assume a fixed device set and fast local storage or interconnect.
- LinguaLinked's 0.23–2.45 s reload, against CROWDio's ~48 s cold and ~6 s warm, suggests CROWDio's cold path is dominated by download, Chaquopy/Python start-up and session creation rather than raw weight I/O. Profiling would need to confirm this.
- Tensor or sequence parallelism (Galaxy, TPI-LLM, over-the-air) needs low-latency links. That justifies CROWDio's choice of pipeline (stage) parallelism over a server relay.

### Gaps
- **CrowdNet:** I could not verify a paper of that name on multi-phone DNN inference. Do not cite it without a DOI.
- CoEdge: the exact ToN volume, year and DOI were not confirmed from the primary source (known only from secondary citations).
- I found no paper that measures churn or dropout rates of phones in collaborative DNN inference.
- I did not open the TPI-LLM PDF. Its memory figures conflict between secondary summaries.
- I did not search for 2025–2026 MobiSys/MobiCom/SenSys papers on multi-phone LLM inference beyond Confidant. I found no peer-reviewed work splitting one LLM across volunteer phones with churn handling.

---

## 2. On-device memory, model loading, cold start and benchmarks

### Takeaway
"Cold inference" (read + initialize + execute) is a recognized, separately optimized problem. NNV12 (MobiSys 2023) reports up to 15.2× (CPU) and 401.5× (GPU) speedup on it, motivated by models that cannot stay resident. Flash-offloading LLM work (LLM in a flash, PowerInfer-2) treats storage-to-DRAM loading as the bottleneck. **Melon is about on-device training memory, not inference loading.** Cite it only as a memory-wall example.

### Cited Findings
- **NNV12.** R. Yi, T. Cao, A. Zhou, X. Ma, S. Wang, M. Xu, "Boosting DNN Cold Inference on Edge Devices" (ACM title "…on Devices"), ACM MobiSys 2023, arXiv:2206.07446. It describes itself as the first engine optimized for cold inference. It has three knobs: per-operator kernel selection, caching post-transformed weights on disk to skip the transform at load, and pipelined kernel execution across asymmetric (big.LITTLE/GPU) processors, with a heuristic scheduler. Reports up to 15.2× (edge CPU) and 401.5× (edge GPU) faster than state-of-the-art engines with zero accuracy loss. The motivation is that devices cannot keep every DNN in memory. The abstract gives no cold-versus-warm ratio. — [arXiv](https://arxiv.org/abs/2206.07446); [MSR PDF](https://www.microsoft.com/en-us/research/uploads/prod/2023/05/mobisys23-Boosting_DNN_Cold_Inference_on_Edge_Devices.pdf)
- **Melon.** Q. Wang, M. Xu, C. Jin, X. Dong, J. Yuan, X. Jin, G. Huang, Y. Liu, X. Liu, "Melon: Breaking the Memory Wall for Resource-Efficient On-Device Machine Learning," ACM MobiSys 2022. It targets on-device **training**, adapting recomputation and micro-batching with a tensor-lifetime-aware memory pool. Reports up to 4.33× larger batch at the same memory, 1.89× average throughput and up to 49.43% energy saving. Code is at qipengwang/Melon. The DOI differs across sources, so check the ACM DL. — [PDF](https://xumengwei.github.io/files/MobiSys22-Melo.pdf); [AMiner](https://aminer.org/pub/634d808890e50fcafd4e4e7f/melon-breaking-the-memory-wall-for-resource-efficient-on-device-machine-learning)
- **LLM in a flash.** K. Alizadeh, I. Mirzadeh, D. Belenko, …, M. Farajtabar (Apple), "LLM in a flash: Efficient Large Language Model Inference with Limited Memory," ACL 2024 (Long), pp. 12562–12584, DOI 10.18653/v1/2024.acl-long.678, arXiv:2312.11514. Keeps weights in flash and loads them on demand, using "windowing" (reusing recently active neurons) and "row-column bundling" for larger contiguous reads. Runs models up to 2× DRAM size, 4–5× (CPU) and 20–25× (GPU) faster than naive loading (exact numbers vary by version). — [arXiv](https://arxiv.org/abs/2312.11514); [ACL Anthology](https://preview.aclanthology.org/setup/2024.acl-long.678)
- **PowerInfer-2.** Z. Xue et al., "PowerInfer-2: Fast Large Language Model Inference on a Smartphone," arXiv:2406.06282 (2024). Neuron-cluster granularity with I/O–compute pipelining for models larger than phone memory. Deployed on OnePlus 12 and Ace 2, with up to 47B-parameter models. Reports on average 3.94× (up to 4.38×) faster than LLM-in-a-flash and 25.4× (up to 29.2×) faster than llama.cpp. Adds 12K LoC on PowerInfer. — [arXiv](https://arxiv.org/pdf/2406.06282); [project](https://powerinfer.ai/v2/)
- **Neuralink (flash neuron placement).** arXiv:2410.19274. Speeds up single-phone LLM inference by optimizing where neurons are placed in flash. — [arXiv](https://arxiv.org/html/2410.19274v3)
- **sd.npu.** arXiv:2510.15312 (2025). Speculative decoding plus NPU-coordinated execution on Qualcomm phones, up to 3.8× speedup. — [arXiv](https://arxiv.org/html/2510.15312v3)
- **MLPerf Mobile.** V. J. Reddi et al., "MLPerf Mobile Inference Benchmark: An Industry-Standard Open-Source Machine Learning Benchmark for On-Device AI," MLSys 2022 (vol. 4, pp. 352–369), arXiv:2012.02328. An app-based benchmark with 4 models (3 vision + 1 NLP, MobileBERT) across SoC vendors' backends. — [MLSys proceedings](https://proceedings.mlsys.org/paper_files/paper/2022/hash/a2b2702ea7e682c5ea2c20e8f71efb0c-Abstract.html); [arXiv](https://arxiv.org/abs/2012.02328)
- **Smart at what cost?** M. Almeida, S. Laskaridis, A. Mehrotra, L. Dudziak, I. Leontiadis, N. D. Lane, "Smart at what cost? Characterising Mobile Deep Neural Networks in the wild," ACM IMC 2021, arXiv:2109.13963. Analyzes 16k+ Play Store apps, finding DNN models in apps grew from 176 (2018) to 1,666 (Apr 2021). Introduces the gaugeNN measurement tool and characterizes energy. — [arXiv](https://arxiv.org/abs/2109.13963v1)
- **MELTing point.** "MELTing point: Mobile Evaluation of Language Transformers," arXiv:2403.12844 (2024). Benchmarks on-device LLM/transformer execution including energy and thermal behavior. Surfaced in search but not opened. — [arXiv](https://arxiv.org/pdf/2403.12844)
- **OODIn.** S. I. Venieris et al., arXiv:2106.04723. An on-device inference framework for heterogeneous phones. On a Samsung A71 it detects thermal throttling in ~800–1150 ms and migrates to another compute engine. — [arXiv](https://arxiv.org/pdf/2106.04723)

### Inferences
- CROWDio's cold load is ~48 s and warm ~6 s, an ~8× gap. That is consistent with NNV12's premise that cold inference dominates when models are not resident. NNV12's "cache post-transformed weights on disk" idea maps directly onto extending CROWDio's sha256 shard cache to store ORT-optimized models (ORT's `optimized_model_filepath`, or the ORT format), skipping graph optimization at session creation. This is an inference about tooling, not a cited result.
- Flash-offloading LLM systems (LLM in a flash, PowerInfer-2, LOIP, Hermes) all overlap I/O with compute. CROWDio could prefetch stage *k+1*'s shard while stage *k* runs elsewhere, which is the JIT deferred loading idea extended to speculative prefetch.

### Gaps
- I did not verify Ignatov et al., "AI Benchmark: All About Deep Learning on Smartphones in 2019" (ICCVW 2019). I believe it exists, but it was not checked in this session.
- I found no peer-reviewed benchmark directly comparing ONNX Runtime Android, TFLite, MNN, NCNN and ExecuTorch on cold start or session-creation latency. MLPerf Mobile reports steady-state latency only.
- The NNV12 full-text breakdown (read / transform / execute fractions) could not be extracted from the PDF.
- I found no academic work on "model swapping" for multi-tenant phone inference beyond Hermes (PipeSwitch baseline) and NNV12.

---

## 3. Split computing and intermediate-feature compression for transmission

### Takeaway
Split computing literature (Matsubara et al.) centers on device→server head/tail splits with *learned, lossy* bottlenecks (BottleFit: ~77% compression, ≤0.6% accuracy loss). CROWDio instead uses *lossless* zlib over base64 on transformer activations across a chain of >2 phones, which this literature does not cover.

### Cited Findings
- **Split computing survey.** Y. Matsubara, M. Levorato, F. Restuccia, "Split Computing and Early Exiting for Deep Learning Applications: Survey and Research Challenges," ACM Computing Surveys 55(5), Art. 90, Dec 2022, arXiv:2103.04505. Defines the head (mobile) / tail (edge) split and early exiting, compares approaches and lists challenges. It is the canonical citation. — [arXiv](https://arxiv.org/abs/2103.04505v1); [NSF PAR](https://par.nsf.gov/servlets/purl/10472616)
- **BottleFit.** Y. Matsubara, D. Callegaro, S. Singh, M. Levorato, F. Restuccia, "BottleFit: Learning Compressed Representations in Deep Neural Networks for Effective and Efficient Split Computing," IEEE WoWMoM 2022, pp. 337–346, arXiv:2201.02693. Injects a bottleneck with multi-stage training and reports 77.1% compression with ≤0.6% accuracy loss on ImageNet. Reduces power by up to 49% and latency by 89% versus local computing, and by 37%/55% versus edge offloading. The head is 83× smaller than autoencoder approaches. — [arXiv](https://arxiv.org/abs/2201.02693v2); [dblp](https://dblp.uni-trier.de/rec/conf/wowmom/MatsubaraCSLR22.html)
- **DeepThings, CoEdge and PICO** reduce transfer by tile or overlap partitioning instead of compression. PICO's argument is that per-layer exchange is too costly over wireless, which favors coarse stage pipelines like CROWDio's. — [PICO](https://arxiv.org/pdf/2206.08662)
- LinguaLinked sends raw activations with a residual "skip" transfer and measured 0.5839 s (sequential) versus 0.5589 s (residual) total delay. — [HTML](https://arxiv.org/html/2312.00388)

### Inferences
- base64 inflates payloads by about 33% before zlib, and float activations compress poorly losslessly. CROWDio could cite BottleFit and Matsubara as the lossy alternative, and fp16 or int8 activation quantization as a cheap middle ground. That trade-off is unexplored for phone-to-phone transformer pipelines.

### Gaps
- I did not search for lossless float-activation codecs (e.g., zfp, or fp16 transport studies for transformer hidden states) or for 2024–2026 learned feature codecs for transformers. That is a follow-up item.
- I found no numbers on the zlib ratio for transformer hidden states.

---

## 4. Energy, thermal throttling, battery and Android background constraints

### Takeaway
Sustained mobile DNN inference does trigger thermal throttling with sharp latency increases ("Play It Cool"; OODIn detects it in ~1 s). Android Doze limits background network to maintenance windows, and Android 15 caps dataSync/mediaProcessing foreground services at 6 h per 24 h. Academic study of these limits for *crowd computing* workloads is essentially absent.

### Cited Findings
- **Play It Cool.** Y. Zhou, F. Liang, T.-W. Chin, D. Marculescu, "Play It Cool: Dynamic Shifting Prevents Thermal Throttling," arXiv:2206.10849 (2022). Shows continuous full-model inference on phone CPUs causes throttling, after which latency rises sharply. It mitigates by shifting among sub-networks of a dynamic network. A secondary article reports BERT pushing the CPU to 80 °C in ~32 s, but I did not confirm this in the paper. — [arXiv](https://arxiv.org/abs/2206.10849v2); [secondary](https://www.unite.ai/how-to-keep-smartphones-cool-when-theyre-running-machine-learning-models/)
- **OODIn.** Detects throttling on a Samsung A71 within ~800–1150 ms and moves execution to another engine. — [arXiv](https://arxiv.org/pdf/2106.04723)
- **Thermal-Aware Scheduling for Deep Learning on Mobile Devices with NPU.** Tan & Cao, IEEE TMC. GPU-versus-NPU scheduling with throttling avoidance as an objective. Seen in search only, with no DOI checked. — [search result](https://par.nsf.gov/servlets/purl/10582109)
- **Joint heat and memory constraints.** "Impact of Joint Heat and Memory Constraints of Mobile Device in Edge-Assisted On-Device Artificial Intelligence" (Elsevier, 2024, DGIST). Uses a measurement study to motivate threshold-based offloading, DVFS and model-size adaptation. — [DGIST Pure](https://dgist.elsevierpure.com/en/publications/impact-of-joint-heat-and-memory-constraints-of-mobile-device-in-e/)
- **Android battery-optimization API study.** "How do open source app developers perceive API changes related to Android battery optimization? An empirical study," Software: Practice and Experience, 2021, DOI 10.1002/spe.2928. Apps show unexpected behaviors after Doze and background-limit changes until code is updated. — [Wiley](https://onlinelibrary.wiley.com/doi/10.1002/spe.2928)
- **Official Android limits.** Doze restricts network and defers jobs and alarms to maintenance windows, and App Standby buckets limit jobs. — [Android Developers: power management resource limits](https://developer.android.com/topic/performance/power/power-details)
- Android 15 caps dataSync and mediaProcessing foreground services at 6 h per 24 h. Freezing cached apps after about 6 h, or about 4 h on Android 16 devices under 8 GB, comes from a vendor blog of low reliability. — [qccbot blog (vendor)](https://blog.qccbot.com/blog/android-background-restrictions-killing-cloud-phone-tasks/)
- **Energy measurement precedent.** Almeida et al. (IMC'21) measure DNN energy with gaugeNN. BottleFit reports up to 49% power reduction from splitting. — [arXiv 2109.13963](https://arxiv.org/abs/2109.13963v1); [BottleFit](https://arxiv.org/abs/2201.02693v2)

### Inferences
- CROWDio's thermal and battery telemetry plus foreground service is the right design. The literature supports making the scheduler thermal-aware (OODIn and Play It Cool-style migration or shrinking). Choosing a foreground-service type matters given Android 15 per-type caps. This needs confirming against the current developer docs for the service type CROWDio declares.
- 50 mAh per run is about 1–1.5% of a typical 4,000–5,000 mAh battery. That is an inference, not a cited figure.

### Gaps
- I found no peer-reviewed measurement of Doze or foreground-service limits on volunteer or crowd distributed computing.
- I did not confirm the exact Android 15 6-hour cap from the primary developer.android.com page in this session (seen only in search snippets).

---

## 5. Python on mobile (Chaquopy, Kivy/BeeWare, Pyodide)

### Takeaway
I found no peer-reviewed academic treatment of Chaquopy (or of Kivy/BeeWare) for distributed computing or DNN execution on Android. Evidence is limited to documentation, GitHub and practitioner articles. This is a documentable gap and novelty point for CROWDio, and it may also explain part of the cold-start overhead.

### Cited Findings
- Chaquopy is "the Python SDK for Android". It runs Python from Java/Kotlin without NDK work, and the team builds ARM native wheels for packages (e.g., NumPy) because upstream often lacks Android builds. — [GitHub chaquo/chaquopy](https://github.com/chaquo/chaquopy); [ProAndroidDev tutorial](https://proandroiddev.com/chaquopy-using-python-in-android-apps-dd5177c9ab6b)
- Practitioner sources report NumPy, SciPy and scikit-learn running in-process on Android via Chaquopy. They also say Google Play's 16 KB page-size requirement forces Chaquopy ≥17.0 and Python ≥3.13 (practitioner claim, not verified on chaquo.com). — [droidcon 2023](https://www.droidcon.com/2023/03/30/chaquopy-using-python-in-android-apps/); [mindster](https://mindster.com/?p=37784)

### Inferences
- Because no academic baseline exists, CROWDio should itself report the Chaquopy interpreter start-up and import cost (numpy, onnxruntime-python if used) as a component of the 48 s cold load. That would be a small but novel measurement contribution.

### Gaps
- I found no academic papers on Kivy, BeeWare (Briefcase) or Pyodide for mobile distributed compute, and no measured Python-on-Android performance study. Further searching of Google Scholar or the ACM DL might find workshop papers.
- The 16 KB page-size and Chaquopy version requirement needs checking against official Chaquopy release notes.

---

## Cross-cutting: gaps CROWDio addresses (supported by the above)
1. **Volunteer, untrusted, churning phones.** No surveyed collaborative-inference system handles device churn or failures. LinguaLinked, EdgePipe, Galaxy, EdgeShard and Jupiter assume static, trusted devices. CROWDio's checkpoint/resume and reassignment fill this gap. — [LinguaLinked](https://arxiv.org/html/2312.00388); [LOIP critique](https://arxiv.org/html/2512.21835)
2. **Single-residency memory bound on real phones.** LOIP (2026) and Hermes (2024) show that the "all weights resident" assumption breaks, but they evaluate Jetson boards and CPU servers. CROWDio's one-ORT-session-per-device bound (43 MB RSS) is the phone-crowd analogue. — [Hermes](https://arxiv.org/abs/2409.04249v2); [LOIP](https://arxiv.org/html/2512.21835); [CROWDio preprint](https://arxiv.org/abs/2605.20723)
3. **Shard (re)load cost as a scheduling input.** NNV12 shows cold inference is a first-class cost. LinguaLinked measured reloads of only 0.23–2.45 s within a pre-provisioned app, while CROWDio's 48 s cold versus 6 s warm makes cache-affinity scheduling essential. — [NNV12](https://arxiv.org/abs/2206.07446); [LinguaLinked](https://arxiv.org/html/2312.00388)
4. **Server-relayed stage pipeline over WAN with lossless compression.** Galaxy and TPI-LLM rely on low-latency LANs for tensor parallelism. Split computing work uses lossy learned bottlenecks. CROWDio's relay plus optional device-to-device transfer with zlib is a distinct point in the design space. — [Galaxy](https://arxiv.org/abs/2405.17245); [BottleFit](https://arxiv.org/abs/2201.02693v2)
5. **Python-on-Android execution substrate.** Chaquopy has no academic precedent (Section 5).
