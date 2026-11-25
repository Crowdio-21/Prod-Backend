# CROWDio SDK / Programming-Model Literature (2019–2026, with classics)

Verification legend: [V] = existence and key details confirmed by a web search or fetch in this session. [K] = standard, well-known citation, with the URL or DOI taken from bibliographic knowledge and not re-fetched this session. The report writer should spot-check [K] DOIs before final publication.

## Q1. Python programming models for distributed / volunteer / mobile / edge computing, and function shipping

### Takeaway
Python task APIs already offer map/submit/future/decorator styles: Ray, Dask, PyWren/Lithops, Parsl, Fiber and Celery. None of them targets heterogeneous Android phones plus PCs over one WebSocket control plane. Older mobile-crowd frameworks (Hyrax, Misco, Honeybee, CWC) target phones but use Java or Android code, not a Python function-shipping API. The CROWDio paper (arXiv 2604.19363) fills this gap with a single-annotation declarative SDK. Shipping functions with cloudpickle carries a documented arbitrary-code-execution risk that CROWDio's design must address.

### Cited Findings
**CROWDio (the focal paper) [V]**
- Manamperi, L., Pathirana, D., Pathirana, T., Premarathna, N., Gunasekara, K. "CROWDio: A Practical Mobile Crowd Computing Framework with Developer-Oriented Design, Adaptive Scheduling, and Fault Resilience." arXiv:2604.19363 (cs.DC), submitted 21 Apr 2026, 10 pp. — [arXiv](https://arxiv.org/abs/2604.19363)
- The system is a centralized MCdC platform with three subsystems: (1) a declarative SDK in which a single function annotation turns sequential code into a distributed workload, with data-parallel map dispatch and aggregation done transparently; (2) tiered checkpointing; (3) a pluggable multi-criteria (MCDM) scheduler driven by live telemetry. — [arXiv abs](https://arxiv.org/abs/2604.19363); [HTML](https://arxiv.org/html/2604.19363)
- Checkpoint tiers: Base (full snapshot), Delta (only variables changed since the last snapshot), and Compacted (the delta chain is consolidated every k=50 by default, which bounds recovery replay to O(k)). Task functions are instrumented at source level before deployment. Instrumentation adds recovery at function entry, loop ranges that skip completed iterations, persistence calls after mutations, and a thread-safe state manager. — [HTML](https://arxiv.org/html/2604.19363)
- Scheduler: telemetry covers CPU, free memory, battery, network latency and thermal state. Criterion weights come from Shannon-entropy weighting. Pluggable strategies are EDAS, ARAS, MABAC and WRR, with FIFO as the baseline. — [HTML](https://arxiv.org/html/2604.19363)
- Results on 6 heterogeneous Android phones (Galaxy A34/A32/A51/S6 Lite/A9+, Moto E40): up to 56.9% lower execution time than round-robin/FIFO (Monte Carlo, 100M trials); checkpoint overhead bounded at 2–3 s per task; Jain's fairness index 0.889. Workloads were CPU-bound, AI/NLP inference and data-parallel. — [arXiv abs](https://arxiv.org/abs/2604.19363)
- Limitations the authors state: only 6 devices; a lab setting without mobility, user interruption or voluntary participation; no energy measurement; distributed image processing was slower than a single device because of coordination overhead on small batches. — [HTML](https://arxiv.org/html/2604.19363)
- Transport is described as persistent event-driven connections, citing RFC 6455 (WebSocket). The paper does not describe the API names (crowdio_map and so on), Chaquopy, model-shard export, or framework-aware deltas. Those features exist only in the current codebase and are not yet published. — [HTML](https://arxiv.org/html/2604.19363)
- Related work the paper compares in its Table 1: Hyrax/Misco, Honeybee, BOINC, and cloud frameworks (Ray, Spark). — [HTML](https://arxiv.org/html/2604.19363)

**General Python distributed frameworks**
- Ray [K]: Moritz, P. et al. "Ray: A Distributed Framework for Emerging AI Applications." OSDI 2018. Ray unifies stateless tasks and stateful actors behind `@ray.remote` with futures (`ray.get`). Its scheduler is bottom-up and distributed, and lineage is stored in a sharded GCS. Relation to CROWDio: the closest API precedent (decorator plus futures, similar to submit/get). Ray assumes trusted datacenter nodes with full CPython and a reliable network; it has no Android workers and no battery or thermal-aware placement. — [USENIX](https://www.usenix.org/conference/osdi18/presentation/moritz); [arXiv 1712.05889](https://arxiv.org/abs/1712.05889)
- Dask [K]: Rocklin, M. "Dask: Parallel Computation with Blocked algorithms and Task Scheduling." SciPy 2015. Dask builds dynamic task graphs, offers `client.map`/`submit` futures in dask.distributed, and provides collection APIs. Relation: a direct precedent for the CROWDio `submit`/`map`/`get` style, but it targets cluster or laptop workers. — [SciPy proceedings DOI 10.25080/Majora-7b98e3ed-013](https://doi.org/10.25080/Majora-7b98e3ed-013)
- PyWren [K]: Jonas, E., Pu, Q., Venkataraman, S., Stoica, I., Recht, B. "Occupy the Cloud: Distributed Computing for the 99%." ACM SoCC 2017. PyWren serializes Python functions with cloudpickle and runs a `map` over AWS Lambda, with stateless workers and S3 for I/O. Relation: the canonical "ship a function, map it over stateless remote workers" model, and CROWDio's `crowdio_map` follows the same idea. PyWren's workers are homogeneous cloud functions, not volatile phones. — [arXiv 1702.04024](https://arxiv.org/abs/1702.04024); [DOI 10.1145/3127479.3128601](https://doi.org/10.1145/3127479.3128601)
- Lithops [V]: Sampé, J., Sánchez-Artigas, M., Vernik, G., Yehekzel, I., García-López, P. "Outsourcing Data Processing Jobs With Lithops." IEEE Transactions on Cloud Computing 11 (2023), published online 2021. Lithops is a multi-cloud serverless framework that continues PyWren. It offers a `concurrent.futures`-like API and a multiprocessing-style API, and moving between clouds only needs a config change. Backends include AWS Lambda, IBM Cloud Functions, Google Cloud Functions and Knative. Relation: a backend-agnostic API precedent. CROWDio extends the "same code, different backend" idea to phones. — [Haifa CRIS record](https://cris.haifa.ac.il/en/publications/outsourcing-data-processing-jobs-with-lithops/); [IBM: Toward Multicloud Access Transparency in Serverless Computing (IEEE Software 2021)](https://www.research.ibm.com/publications/toward-multicloud-access-transparency-in-serverless-computing)
- Parsl [V]: Babuji, Y. et al. "Parsl: Pervasive Parallel Programming in Python." HPDC '19. Parsl uses `@python_app`/`@bash_app` decorators that return futures, builds a dynamic dependency graph, and supports pluggable executors, elastic scaling and fault tolerance. On Blue Waters it reports executor overhead as low as about 5 ms and scaling to more than 250,000 workers on more than 8,000 nodes. Relation: the closest precedent for a decorator-based declarative model with pluggable executors, comparable to CROWDio's `@CROWDio.task` and pluggable scheduler. Parsl targets HPC. — [DOI 10.1145/3307681.3325400](https://doi.org/10.1145/3307681.3325400); [arXiv 1905.02158](https://arxiv.org/abs/1905.02158)
- Fiber [V]: Zhi, J., Wang, R., Clune, J., Stanley, K. O. "Fiber: A Platform for Efficient Development and Distributed Training for Reinforcement Learning and Population-Based Methods." arXiv:2003.11164 (2020). Fiber is a master–worker model that reproduces the Python `multiprocessing` API (Pool.map) on clusters. Its GitHub repository is now archived. Relation: a precedent for "multiprocessing-compatible API, distributed backend"; CROWDio's `crowdio_map` gives a similar experience. — [arXiv](https://arxiv.org/abs/2003.11164)
- Celery [K]: a distributed task queue with `@app.task` decorators, a broker (RabbitMQ/Redis), retries and `AsyncResult`. Relation: an industry baseline for decorator tasks with retries. Celery workers must have the code installed, so it does not ship functions. — [Celery docs](https://docs.celeryq.dev/)
- Python `multiprocessing`/`concurrent.futures` [K]: the standard library's `Pool.map` and `Executor.submit/map` define the de facto map/submit idiom that Fiber, Lithops and CROWDio imitate. — [concurrent.futures docs](https://docs.python.org/3/library/concurrent.futures.html)

**Volunteer and mobile crowd computing**
- BOINC [K]: Anderson, D. P. "BOINC: A Platform for Volunteer Computing." Journal of Grid Computing 18:99–122 (2020). BOINC's server/client middleware uses a C/C++ API (`boinc_init`, `boinc_checkpoint_completed`, `boinc_fraction_done`), result validation by replication, and an Android client. Apps are native binaries pre-registered per platform. Relation: the main volunteer-computing precedent. BOINC's API is a low-level native-code API and its jobs are batch work units, whereas CROWDio offers Python-function shipping and interactive async results. — [DOI 10.1007/s10723-019-09497-9](https://doi.org/10.1007/s10723-019-09497-9)
- Honeybee [V]: Fernando, N., Loke, S. W., Rahayu, W. "Honeybee: A Programming Framework for Mobile Crowd Computing." MobiQuitous 2012, LNICST 120 (Springer 2013). Honeybee is an Android framework that uses work stealing to load-balance independent jobs across mobile nodes not known in advance, which may join or leave at random. Its successors are Honeybee-T (2022) and Honeybee-Tx (Future Internet 2025, dependency-aware). Relation: the direct MCdC programming-framework predecessor. Honeybee is Java-centric and uses work stealing; CROWDio is Python, uses a central Foreman and MCDM scheduling. — [Springer DOI 10.1007/978-3-642-40238-8_19](https://doi.org/10.1007/978-3-642-40238-8_19); [Honeybee-Tx, MDPI Future Internet 17(10):446](https://www.mdpi.com/1999-5903/17/10/446)
- Hyrax (Marinelli 2009, CMU tech report) and Misco (Dou et al., MobiSys 2010) [K, cited by CROWDio]: early MapReduce frameworks for phones. — cited in [CROWDio HTML](https://arxiv.org/html/2604.19363)
- CWC [K]: Arslan, M. Y. et al. "Computing While Charging: Building a Distributed Computing Infrastructure Using Smartphones." ACM CoNEXT 2012. CWC schedules jobs only on smartphones that are charging. Relation: a precedent for charging- and battery-aware participation. — [DOI 10.1145/2413176.2413204](https://doi.org/10.1145/2413176.2413204)

**Function shipping security (cloudpickle / pickle)**
- Python's pickle docs warn that unpickling can execute arbitrary code, so it should never be used on untrusted data. cloudpickle, which serializes closures and lambdas by value, has the same property [K]. — [pickle docs](https://docs.python.org/3/library/pickle.html); [cloudpickle README](https://github.com/cloudpipe/cloudpickle)
- MLflow advisories in 2024 (HiddenLayer): `cloudpickle.load` on a malicious sklearn model artifact executes code. The issue is rated CVSS 8.8 (CWE-502), with CVE-2024-37052…37057 covering sklearn, pyfunc, pmdarima, LightGBM and Keras loaders. — [HiddenLayer advisory](https://hiddenlayer.com/sai-security-advisory/2024-06-mlflow-10); [HiddenLayer overview](https://hiddenlayer.com/2024-06-mlflow)
- These are vendor advisories, not peer-reviewed studies. They recommend formats such as safetensors or JSON for untrusted data. — [HF PoC page](https://huggingface.co/01data-ai/mlflow_f003_sklearn_pickled_model_path_traversal_rce)

### Inferences
- The SDK space divides into three groups: (a) cluster/cloud Python frameworks (Ray, Dask, Parsl, Lithops, Fiber), which are developer-friendly but assume trusted homogeneous nodes; (b) volunteer/MCdC systems (BOINC, Honeybee, CWC), which are device-aware but either native-code or Java; (c) FL SDKs (Q5). CROWDio places a group-(a) style API (map/submit/pipeline, decorator) on group-(b) infrastructure. That combination is its main novelty claim.
- Because CROWDio ships serialized code to volunteer phones, the trust direction is the reverse of the MLflow case. Workers execute code that clients send, so a malicious job submitter threatens volunteers. The literature found here does not address sandboxing shipped Python on Android (for example via Chaquopy). This is a gap CROWDio should at least discuss: authentication of submitters, an allow-list, or isolation.
- The declarative-annotation claim lacks a user study. No developer-friendliness study of MCdC SDKs (for example SUS scores or LoC comparisons) turned up.

### Gaps
- No peer-reviewed empirical study of cloudpickle-based function shipping security in distributed frameworks was found. Only advisories were found.
- No published Chaquopy-based distributed computing system was found. Chaquopy's use in CROWDio could not be confirmed from the arXiv paper, which does not mention it.
- No developer-usability study (SUS, task-completion time) of Ray, Dask or MCdC SDKs was located within the search budget.
- The Ray, Dask, PyWren, BOINC and CWC DOIs are [K] and were not re-fetched.

## Q2. Declarative / intent-based ML serving and SLO/objective-driven inference

### Takeaway
Since INFaaS (ATC'21), "model-less" serving has been an established idea: the developer states latency or accuracy goals and the system picks the variant and hardware. Clipper, Nexus and Clockwork contribute SLO-aware batching and predictability inside datacenters. CROWDio's `submit_job(model_id, inputs, objective="min_latency_with_deadline", deadline_ms)`, where the system picks both the partition cut and the volunteer workers and returns plan_epoch and timings, carries the model-less idea to split inference over volatile phones. No prior work found does that.

### Cited Findings
- INFaaS [V]: Romero, F., Li, Q., Yadwadkar, N. J., Kozyrakis, C. "INFaaS: Automated Model-less Inference Serving." USENIX ATC 2021, pp. 397–411 (Best Paper). Developers specify performance and accuracy requirements per query, and INFaaS chooses the model variant and hardware. On AWS EC2 it reports 1.3× higher throughput, 1.6× fewer SLO violations, and up to 21.6× (8.5× on average) cost savings versus prior systems. Relation: the key conceptual precedent for CROWDio's objective-based `submit_job`. CROWDio's decision variables (cut point and worker set) are different. — [USENIX](https://www.usenix.org/conference/atc21/presentation/romero); [arXiv 1905.13348](https://arxiv.org/abs/1905.13348)
- Clipper [K]: Crankshaw, D. et al. "Clipper: A Low-Latency Online Prediction Serving System." NSDI 2017. Clipper provides a model-abstraction layer with adaptive batching against a latency SLO, caching, and model selection via bandits. Relation: an early example of a latency-objective API. — [USENIX](https://www.usenix.org/conference/nsdi17/technical-sessions/presentation/crankshaw)
- Nexus [K]: Shen, H. et al. "Nexus: A GPU Cluster Engine for Accelerating DNN-Based Video Analysis." SOSP 2019. Nexus does SLO-aware squishy bin-packing of models onto GPUs and supports query-level latency SLOs. — [DOI 10.1145/3341301.3359658](https://doi.org/10.1145/3341301.3359658)
- Clockwork [K]: Gujarati, A. et al. "Serving DNNs like Clockwork: Performance Predictability from the Bottom Up." OSDI 2020. Clockwork's central controller schedules on predictable per-request latencies and drops requests that cannot meet their deadline. Relation: a precedent for deadline-driven admission and for returning per-request timings. — [USENIX](https://www.usenix.org/conference/osdi20/presentation/gujarati)
- KServe [K]: a Kubernetes CRD (`InferenceService`) that declares a model URI and runtime, with autoscaling and the Open Inference Protocol (V2). It is declarative about deployment but not about per-request objectives. — [KServe docs](https://kserve.github.io/website/)
- Neurosurgeon [K]: Kang, Y. et al. "Neurosurgeon: Collaborative Intelligence Between the Cloud and Mobile Edge." ASPLOS 2017. Neurosurgeon picks a layer-granularity cut point automatically to minimize latency or energy from per-layer predictions. Relation: the classic automatic cut-selection precedent for "the system picks the cut". It covers one phone and one server, not a fleet. — [DOI 10.1145/3037697.3037698](https://doi.org/10.1145/3037697.3037698)

### Inferences
- The objective-API lineage runs Clipper → Nexus/Clockwork → INFaaS (model-less) → CROWDio (partition-less and placement-less over phones). INFaaS is the closest framing to cite.
- Returning `plan_epoch` per request is unusual. In the serving literature found, the plan or version is typically hidden. Exposing it supports reproducibility when reconfiguration happens, which is a novel API affordance worth highlighting.

### Gaps
- No model-less or SLO-API serving system over volunteer or mobile workers was found.
- The Clipper, Nexus, Clockwork, KServe and Neurosurgeon citations are [K].

## Q3. Model partitioning toolchains, export formats, manifests and verification

### Takeaway
ONNX's `extract_model` and Polygraphy provide the primitives for cutting a graph at named tensors. Research systems (PipeEdge, EdgeShard) partition transformers across edge devices with custom PyTorch code. Surveys (Hao et al., ACM CSUR 2026; Matsubara et al., CSUR 2022) catalogue the toolchains. No tool found pre-exports every contiguous layer-range shard, verifies the chained output numerically against the source model, enforces a storage budget, and emits a sha256 manifest. That pipeline is crowdio-export's contribution.

### Cited Findings
- Hao et al. [V]: "DNN Partitioning for Cooperative Inference in Edge Intelligence: Modeling, Solutions, Toolchains." ACM Computing Surveys 58(8), June 2026, DOI 10.1145/3786145. The survey gives a unified mathematical framework for architectures, models and optimization problems, and classifies strategies by partition count and granularity. It also summarizes common experimental setups and tools, and discusses open issues including security, privacy and partitioning large models. The full author list and the contents of the toolchain section were not retrieved. — [Crossref](https://api.crossref.org/works/10.1145%2F3786145); [DUT faculty page](https://faculty.dlut.edu.cn/2005011019/en/lwcg/722816/content/362811.htm)
- Matsubara, Levorato, Restuccia [V]: "Split Computing and Early Exiting for Deep Learning Applications: Survey and Research Challenges." ACM CSUR 55(5), Art. 90, Dec 2022, DOI 10.1145/3527155. The survey covers head/tail split computing and early exits. It cites bottleneck-injection work such as BottleFit (arXiv:2201.02693) for compressing intermediate representations. — [arXiv 2103.04505](https://arxiv.org/abs/2103.04505)
- ONNX `onnx.utils.extract_model` / `Extractor` [V]: these extract a sub-model defined by exact input and output tensor names. A boundary must not cut through If/Loop subgraphs. Models over 2 GB use external data. — [ONNX utils docs](https://onnx.ai/onnx/api/utils.html)
- 3GPP SA4 split-inference study (FS_AI4Media) [V]: splitting a single-branch model is straightforward with `extract_model`; multi-branch graphs need parsing to find the cross-cut tensors. — [3GPP TR 26.927 §6.3.3 (iTecSpec)](https://www.itecspec.com/3gpp/26.927/s/6.3.3)
- Other ONNX splitting tools [V]: NVIDIA Polygraphy `surgeon extract` writes a subgraph from given I/O metadata; ssp4onnx splits by weight size; onnx2tf can stop conversion at named outputs to produce subgraphs. — [Polygraphy extract doc](https://pulsar2-docs.readthedocs.io/en/latest/other_tools/polygrapy_extract.html); [ssp4onnx PyPI](https://pypi.org/project/ssp4onnx/)
- torch.fx [K]: Reed, J. et al. "torch.fx: Practical Program Capture and Transformation for Deep Learning in Python." MLSys 2022. torch.fx provides symbolic tracing to an IR, which enables graph splitting (for example `split_module`). It underpins PiPPy and `torch.distributed.pipelining` stage splitting. — [arXiv 2112.08429](https://arxiv.org/abs/2112.08429); [torch.distributed.pipelining docs](https://pytorch.org/docs/stable/distributed.pipelining.html)
- PipeEdge [V]: Hu, Y., Imes, C., Zhao, X., Kundu, S., Beerel, P. A., Crago, S. P., Walters, J. P. "PipeEdge: Pipeline Parallelism for Large-Scale Model Inference on Heterogeneous Edge Devices." Euromicro DSD 2022, pp. 298–307 (arXiv 2110.14895). PipeEdge uses dynamic programming to partition transformer layers across heterogeneous edge devices. It reports 11.88× (ViT-Huge) and 12.78× (BERT-Large) speedup on 16 devices with no accuracy loss, and its code is open source. Relation: the closest prior work for layer-range sharding of BERT. PipeEdge splits PyTorch modules at runtime and does not export verified ONNX shards. — [arXiv](https://arxiv.org/abs/2110.14895); [ISI record](https://www.isi.edu/results/publications/10310/pipeedge-pipeline-parallelism-for-large-scale-model-inference-on-heterogeneous-edge-devices)
- EdgeShard [V]: Zhang, M., Cao, J., Shen, X., Cui, Z. "EdgeShard: Efficient LLM Inference via Collaborative Edge Computing." arXiv:2405.14371 (May 2024). EdgeShard jointly selects devices and partitions an LLM into shards across edge and cloud. On Llama2 it reports up to 50% lower latency and 2× throughput versus on-device and cloud-edge baselines. — [arXiv](https://arxiv.org/abs/2405.14371)
- MLflow Model Registry [K]: versioned models with stage transitions. It is the Databricks-origin registry pattern; integrity checking is not its focus, and the 2024 RCE advisories above concern its pickle flavors. — [MLflow registry docs](https://mlflow.org/docs/latest/model-registry.html); Zaharia, M. et al. "Accelerating the Machine Learning Lifecycle with MLflow." IEEE Data Eng. Bull. 41(4), 2018 [K]
- safetensors [K]: Hugging Face's format avoids pickle code execution, and the Hub stores file sha256 via Git LFS pointers. — [safetensors docs](https://huggingface.co/docs/safetensors/index)

### Inferences
- Correctness verification of split models is rarely made explicit. PipeEdge and EdgeShard report "no accuracy loss" from end-to-end task metrics, not per-shard numeric equivalence. crowdio-export's check (max|diff| < 1e-4 when chaining shards versus PyTorch) is a reusable engineering contribution.
- Exporting all contiguous layer-atom ranges, rather than just the chosen cut, lets the runtime re-plan cuts without re-export. That design choice supports reconfiguration-aware planning; a budget then prunes the O(L²) set.
- A sha256 manifest plus a checksum-verified ShardStore follows the safetensors/LFS integrity pattern and fits untrusted volunteer storage. No partitioning paper found treats shard integrity.

### Gaps
- The text of Hao et al.'s toolchain section could not be retrieved, so its tool list cannot be quoted.
- No paper was found that systematically verifies ONNX split-shard numerical equivalence.
- The torch.fx, MLflow and safetensors citations are [K].

## Q4. Checkpointing APIs for user code

### Takeaway
Checkpointing ranges from transparent system-level checkpointing (CRIU, DMTCP), which is not possible on stock Android for Python-in-app processes, to framework-level checkpointing (PyTorch DCP, CheckFreq, Check-N-Run) that handles tensors and deltas. CROWDio's `@task(checkpoint=True)` sits in between. It is a declarative, source-instrumented application-level checkpoint with delta and compaction tiers, close to BOINC's app-driven checkpoint API but automated.

### Cited Findings
- DMTCP [K]: Ansel, J., Arya, K., Cooperman, G. "DMTCP: Transparent Checkpointing for Cluster Computations and the Desktop." IEEE IPDPS 2009. DMTCP checkpoints unmodified multi-process applications at user level. — [DOI 10.1109/IPDPS.2009.5161063](https://doi.org/10.1109/IPDPS.2009.5161063)
- CRIU [K]: Linux Checkpoint/Restore In Userspace, which needs kernel features and privileges unavailable to ordinary Android apps. — [criu.org](https://criu.org/)
- CheckFreq [K]: Mohan, J., Phanishayee, A., Chidambaram, V. "CheckFreq: Frequent, Fine-Grained DNN Checkpointing." USENIX FAST 2021. CheckFreq tunes checkpoint frequency automatically and splits checkpointing into a snapshot phase and a persist phase. — [USENIX](https://www.usenix.org/conference/fast21/presentation/mohan)
- Check-N-Run [K]: Eisenman, A. et al. "Check-N-Run: A Checkpointing System for Training Deep Learning Recommendation Models." NSDI 2022. Check-N-Run uses differential (incremental) checkpoints plus quantization to shrink checkpoint size. Relation: the key precedent for CROWDio's delta checkpoints. — [USENIX](https://www.usenix.org/conference/nsdi22/presentation/eisenman)
- PyTorch Distributed Checkpoint (DCP) [V]: `torch.distributed.checkpoint` offers sharded save/load and `async_save`, which returns a future so storage I/O overlaps training. Customizable staging (`AsyncStager`), process-based async and plan caching are described in PyTorch blogs. — [DCP docs](https://docs.pytorch.org/main/distributed.checkpoint.html); [PyTorch forum/TorchTitan post](https://discuss.pytorch.org/t/distributed-w-torchtitan-optimizing-checkpointing-efficiency-with-pytorch-dcp/211250)
- BOINC application API [K]: apps call `boinc_time_to_checkpoint()` and `boinc_checkpoint_completed()`. The application, not the system, serializes its own state. — [BOINC paper](https://doi.org/10.1007/s10723-019-09497-9)
- CROWDio's measured overhead is 2–3 s per task, independent of frequency, with compaction every k=50 deltas. — [CROWDio HTML](https://arxiv.org/html/2604.19363)

### Inferences
- CROWDio's position: as declarative as Parsl/Ray retries, more fine-grained than task re-execution, and Android-compatible where CRIU and DMTCP are not. The delta-plus-compaction scheme mirrors Check-N-Run's differential approach, applied to Python variables instead of embedding tables.

### Gaps
- No published application-level checkpointing system for Python on Android was found.
- The DMTCP, CRIU, CheckFreq and Check-N-Run citations are [K]. Gemini (Wang et al., SOSP 2023, in-memory checkpointing) is another candidate citation that was not verified.

## Q5. Federated / on-device learning SDKs as device-fleet comparison points

### Takeaway
FL frameworks are the most mature "Python server, phone clients" SDKs. Flower is framework-agnostic and has Android/TFLite clients; FedML and PySyft are other examples, and Google's production FL system design addresses dropout and eligibility (charging, idle, unmetered network). Their client API, however, is "implement fit/evaluate", with code pre-installed on the device. CROWDio instead ships arbitrary functions and model shards on demand.

### Cited Findings
- Flower [V]: Beutel, D. J. et al. "Flower: A Friendly Federated Learning Research Framework." arXiv:2007.14390 (2020; v5 2022). Flower is described as open-source and mobile-friendly. It supports large-scale experiments with heterogeneous devices, and its abstractions let mobile workloads be ported regardless of language or ML framework. The repository includes Android/TFLite quickstart examples. — [arXiv](https://arxiv.org/abs/2007.14390)
- FedML [K]: He, C. et al. "FedML: A Research Library and Benchmark for Federated Machine Learning." arXiv:2007.13518 (2020). FedML supports simulation, distributed and on-device training (Android/IoT). — [arXiv](https://arxiv.org/abs/2007.13518)
- PySyft [K]: Ryffel, T. et al. "A Generic Framework for Privacy Preserving Deep Learning." arXiv:1811.04017 (2018). — [arXiv](https://arxiv.org/abs/1811.04017)
- TensorFlow Federated [K]: a Python FL API (simulation-focused). — [tensorflow.org/federated](https://www.tensorflow.org/federated)
- Bonawitz, K. et al. "Towards Federated Learning at Scale: System Design." MLSys 2019 [K]. The system selects only eligible devices (charging, idle, unmetered), over-selects participants to absorb dropouts, uses pace steering, and reports diurnal participation patterns. Relation: the best-documented production fleet design for volatile phones. — [arXiv 1902.01046](https://arxiv.org/abs/1902.01046)

### Inferences
- Lessons CROWDio can borrow: eligibility gating (battery, charging, thermal; CROWDio already collects this telemetry), over-provisioning or straggler replication, and pace steering. Where CROWDio differs: code is shipped per job rather than compiled into the app, and the workload is general compute and split inference rather than only training.

### Gaps
- No FL SDK was found that supports arbitrary user-function shipping to phones.
- The FedML, PySyft, TFF and Bonawitz citations are [K].

## Q6. Tensor serialization / transport and activation compression

### Takeaway
CROWDio's zlib+base64 self-describing JSON payloads over WebSocket favor simplicity and Chaquopy compatibility. Base64 adds about 33% overhead, and binary formats (Arrow IPC, protobuf/gRPC, msgpack) would avoid it. The split-computing literature compresses activations with learned bottlenecks or quantization rather than generic zlib.

### Cited Findings
- WebSocket (RFC 6455) is the transport cited by CROWDio. It supports binary frames, so base64 is not required by the protocol. — [RFC 6455](https://www.rfc-editor.org/rfc/rfc6455) [K]; [CROWDio HTML](https://arxiv.org/html/2604.19363)
- Apache Arrow [K]: a columnar in-memory format with IPC/Flight for zero-copy tensor and table transport. — [arrow.apache.org](https://arrow.apache.org/)
- gRPC/protobuf [K]: HTTP/2 RPC with a binary schema. KServe's Open Inference Protocol defines both REST and gRPC tensor encodings. — [gRPC](https://grpc.io/); [KServe](https://kserve.github.io/website/)
- Activation compression in split computing: bottleneck injection and learned compression (BottleFit, arXiv:2201.02693), surveyed in Matsubara et al. CSUR 2022. — [arXiv 2103.04505](https://arxiv.org/abs/2103.04505)

### Inferences
- Base64 inside JSON inflates payloads by 4/3 before compression gains. For BERT hidden states (seq × 768 float32), switching to binary WebSocket frames plus fp16 or int8 quantization is a low-risk improvement. Learned bottlenecks would change the model, which conflicts with the exact-verification guarantee.

### Gaps
- No head-to-head WebSocket vs gRPC measurement on Android edge devices was found within the budget.
- No measured compression ratios of zlib on transformer activations were found.
