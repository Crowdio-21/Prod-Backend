# Distributed / Edge Large-Model Inference and Cold Start: Full-Text Close Reading of 8 Papers for the CROWDio Literature Review

Scope: full-text reading of the reference-stripped text files body_425, 426, 508, 525, 534, 539, 546, 559 (user's Zotero export). Every number below was taken from the paper text unless it is marked **[inference]**. Figures and tables whose values were lost in text extraction are flagged as gaps. Web searches were used only to confirm venue metadata.

CROWDio features checked against each paper: (a) pricing reconfiguration or model-load cost in the placement objective, (b) re-partition triggers with hysteresis or cooldown, (c) churn and fault tolerance, (d) phones versus GPUs or SBCs, (e) worker-to-worker activation routing, (f) peer shard fetch and prefetch, (g) the pipeline time model T_total(N) = finish_K + (N-1)·max(t_comp, t_comm).

---

## Q1. EdgeShard (Zotero key "Zhang2024EdgeLLM"): what does it do and how does it relate to CROWDio?

### Takeaway
"Zhang2024EdgeLLM" is **EdgeShard: Efficient LLM Inference via Collaborative Edge Computing** (M. Zhang, X. Shen, J. Cao, Z. Cui, S. Jiang; IEEE Internet of Things Journal 12(10):13119–13130, 15 May 2025; DOI 10.1109/JIOT.2024.3524255; published online 31 Dec 2024). The Fig. 3 caption still calls the framework "EdgeLLM", which explains the Zotero key. EdgeShard is an offline, profile-once dynamic program for joint device selection and layer partitioning on Jetsons plus one cloud GPU. It has no runtime re-partitioning, no churn handling, and no switching or model-load cost term.

### Cited Findings
- **Problem:** LLMs exceed edge memory. Llama2-7B at full precision needs at least 28 GB, against smartphones at 6–12 GB and Jetson Orin NX at 8–16 GB. Llama2-70B at 4-bit still needs at least 35 GB, and at full precision it needs about 280 GB. Quantization costs accuracy, and cloud–edge splits suffer from high, unstable latency. — [EdgeShard, IEEE IoT-J 2025](https://doi.org/10.1109/JIOT.2024.3524255)
- **Workflow:** three stages: (1) offline profiling, run once, of per-layer execution time on each device (the prefill and decode times are averaged), activation size, per-layer memory, device memory, and pairwise bandwidth; (2) scheduling optimization; (3) collaborative inference. For devices that cannot hold the full model during profiling, a "dynamic model loading" technique loads layers sequentially in shards. This loading is used only for profiling and is not costed at runtime. — [EdgeShard](https://doi.org/10.1109/JIOT.2024.3524255)
- **KV-cache management:** each device greedily preallocates KV space of size N_a × B_b × (I_m + G_m), where N_a is the number of allocated layers, B_b the batch size, I_m the maximum input length and G_m the maximum generated length. — [EdgeShard](https://doi.org/10.1109/JIOT.2024.3524255)
- **Latency formulation:** binary X_{i,j} assigns layer i to node j. Communication cost is O_{i-1}/B_{k,j}, or 0 if k = j. The objective is the sum of compute plus communication. Constraints: privacy (layer 0 on source node 0), per-device memory, and unique assignment. The DP is DP(i,j) = min_k[DP(i-1,k) + t_comp + t_comm + 1(i = N-1)·t_comm(j→0)], which includes the return of the token to the source node. Complexity is O(N·M²). — [EdgeShard](https://doi.org/10.1109/JIOT.2024.3524255)
- **Throughput formulation:** for pipeline parallelism, each device's latency is max(t_comp, t_comm), and the objective is to minimize the slowest device. The DP is g(m, S', j) over (layers processed, set of used devices, last device), with complexity O(N²·2^M·M²). — [EdgeShard](https://doi.org/10.1109/JIOT.2024.3524255)
- **EdgeShard-No-Bubbles:** this variant reorders token generation across micro-batches so that a micro-batch begins decoding immediately instead of waiting for all micro-batches to finish prefill. — [EdgeShard](https://doi.org/10.1109/JIOT.2024.3524255)
- **Testbed:** 15 devices: 12 Jetson AGX Orin, 2 Orin NX and 1 cloud server. The text references an RTX 3090 as the server GPU. Devices are connected at 1000 Mb/s, and Linux tc shapes the links. Models are Llama2-7B, 13B and 70B at full precision. The workload is WikiText-2 with 32 input tokens and 96 generated tokens. The source-to-cloud link is set to 1 Mb/s and the other links to 50 Mb/s ±20%. — [EdgeShard](https://doi.org/10.1109/JIOT.2024.3524255)
- **Llama2-7B results:** latency is 75.88 ms/token, about 1.85× faster than Edge-Solo and Cloud-Edge-Opt and about 3× faster than Cloud-Edge-Even. Throughput is 52.45 tokens/s at maximum batch 8, about 2.2× Edge-Solo and Cloud-Edge-Opt and about 7× Cloud-Edge-Even. — [EdgeShard](https://doi.org/10.1109/JIOT.2024.3524255)
- **Llama2-13B results:** latency is 45.7% lower than Cloud-Edge-Even and 28.8% lower than Cloud-Edge-Opt; throughput is 2.23× and 2.2× higher. For Llama2-70B, all baselines go out of memory; EdgeShard-Even selects 11 AGX Orin plus 1 RTX 3090. — [EdgeShard](https://doi.org/10.1109/JIOT.2024.3524255)
- **Bandwidth sweep (1–50 Mb/s cloud–source):** latency drops sharply from 1 to 10 Mb/s and then saturates. Above 10 Mb/s, EdgeShard produces the same plan as Cloud-Edge-Opt. For 13B at 10 Mb/s, EdgeShard gets about 2× the throughput of Cloud-Edge-Opt, because spreading layers reduces per-device memory and allows batch 8 instead of 4. Under Cloud-Edge-Opt the RTX 3090 and the AGX source reached 95% and 98% memory. — [EdgeShard](https://doi.org/10.1109/JIOT.2024.3524255)
- **Source-node effect:** switching the source node from AGX Orin to Orin NX opens a gap of about 60 ms under Cloud-Edge-Opt but only about 5 ms under EdgeShard. The throughput ratio AGX:NX is 6× under Cloud-Edge-Opt but only 2× under EdgeShard. — [EdgeShard](https://doi.org/10.1109/JIOT.2024.3524255)
- **No-Bubbles gains:** for 7B, +0.36 tokens/s under Cloud-Edge-Even and +6.96 tokens/s under EdgeShard. For 13B, +1.69, +1.89 and +5.21 tokens/s for Even, Cloud-Edge and EdgeShard respectively. Prefill time per token is "usually 10×" that of decode, yet prefill, generation and average profiling give similar plans. — [EdgeShard](https://doi.org/10.1109/JIOT.2024.3524255)
- **Stated limitations and future work:** incentive mechanisms, batch-size-aware optimization (the DP ignores batch size), and the risk of activation leakage to honest-but-curious devices (DP or TEE suggested). It also names DePIN and Petals-style consumer-GPU inference as targets. — [EdgeShard](https://doi.org/10.1109/JIOT.2024.3524255)

### Inferences
- **[inference] Relation to CROWDio:** CROWDio adopts the same contiguous-layer DP family; its Pareto DP and EdgeShard's latency and throughput DPs are close relatives. CROWDio extends EdgeShard on every runtime axis EdgeShard leaves out. EdgeShard is static: there is no telemetry loop (profiling is done once), no model-load cost (weights are assumed resident after deployment), no re-partition trigger, no handling of device departure, no phones in the testbed (only Jetsons and a GPU server), and no artifact distribution or prefetch.
- **[inference]** EdgeShard's per-stage max(t_comp, t_comm) bottleneck matches the steady-state term in CROWDio's T_total(N). EdgeShard does not model the fill term finish_K or the batch count N explicitly. Its "privacy constraint" (first layer pinned to the source) is analogous to a pinned-stage constraint CROWDio could express.
- **[inference]** EdgeShard is an encoder-free, decoder-only autoregressive design. CROWDio's DistilBERT workload is encoder-only and single-pass, so EdgeShard's KV-cache and No-Bubbles machinery does not apply directly. Its layer-wise DP does.

### Gaps
- The numeric contents of Table I (memory), Table III (device specifications), Table IV (full latency and throughput grid) and Figs. 7–11 were lost in text extraction. Only the numbers quoted in the prose are reported above.

---

## Q2. arXiv 2110.14895v1 ("EdgePipe"): who wrote it, where was it published, and does it report "affinity-aware placement cuts inter-stage transfer by 43%"?

### Takeaway
arXiv 2110.14895v1 is **"Pipeline Parallelism for Inference on Heterogeneous Edge Computing"** by Yang Hu, Connor Imes, **Xuanang Zhao (third author, Google)**, Souvik Kundu, Peter A. Beerel, Stephen P. Crago and John Paul N. Walters (USC / USC-ISI), posted 28 Oct 2021. The system inside it is called EdgePipe. The published version was renamed **PipeEdge** and appeared at the 25th Euromicro Conference on Digital System Design (DSD 2022), pp. 298–307, DOI 10.1109/DSD57027.2022.00048. **It is not an INFOCOM 2022 paper**, and I found no INFOCOM listing. **The preprint contains no "43%" figure and no notion of "affinity-aware placement"**, and it reports no inter-stage transfer reduction at all. The user's citation "Zhao et al., EdgePipe, INFOCOM 2022 … cuts inter-stage transfer by 43%" is therefore wrong on the first author, the venue and the claim, as far as this text shows.

### Cited Findings
- **Metadata (preprint):** title, author list and 28 Oct 2021 date as above. — [arXiv 2110.14895v1](https://arxiv.org/abs/2110.14895v1)
- **Metadata (published version):** "PipeEdge: Pipeline Parallelism for Large-Scale Model Inference on Heterogeneous Edge Devices", DSD 2022, pp. 298–307, same seven authors. — [USC ISI publication record](https://www.isi.edu/results/publications/10310/pipeedge-pipeline-parallelism-for-large-scale-model-inference-on-heterogeneous-edge-devices); DOI per [PipeEdge GitHub](https://github.com/YiyunZheng/PipeEdge)
- **Check of the "43%" claim:** a full-text search of body_426.txt for "43", "affinit" and "INFOCOM" found only the pseudocode line number "43:" in Algorithm 1 and the throughput value "2.43 images per second" (ViT-Large on 16 RCC-VE boards). The paper's metrics are throughput (images/s) and speedups. It never measures transferred bytes or inter-stage traffic reduction. — [arXiv 2110.14895v1](https://arxiv.org/abs/2110.14895v1)
- **System model:** a transformer of L layers with per-layer transfer size P_j and runtime memory M_j, and heterogeneous devices with pairwise bandwidth b_{u,v}. T_comm = P_j/b_{u,v}. The authors assume asynchronous communication and **perfect overlap**, so T_period = max(T_comp, T_comm), and the objective minimizes the slowest stage. The DP state h(i, S, u) covers the layers processed, the used-device set and the next device. Complexity is O(2^D·L²·D²), reducible to O(∏(n_i+1)·L²·N²) when devices fall into N identical types. The number of pipeline stages equals the number of participating devices. — [arXiv 2110.14895v1](https://arxiv.org/abs/2110.14895v1)
- **Solver runtime** (ViT-Base, 3 device types × 3 devices, 1.6 GHz i5): their DP takes 0.01 s, a naive DP 18.6 s, and brute force 71 min. — [arXiv 2110.14895v1](https://arxiv.org/abs/2110.14895v1)
- **Setup:** DCompTB testbed with MinnowBoard boards (4× Atom E3845, 2 GB) and RCC-VE boards (4× Atom C2558, 8 GB), both on 1 Gbps links. Heterogeneity is emulated with cpulimit, ulimit and tc, plus a fixed 20 ms added latency. Software is PyTorch 1.9.0 with RPC over TensorPipe. Models are ViT-Base, ViT-Large and ViT-Huge (HuggingFace 4.10) on ImageNet. Baselines are GPipe (even partition, re-implemented) and the PipeDream partitioner (forward pass only, one-level network), each with its optimal microbatch. — [arXiv 2110.14895v1](https://arxiv.org/abs/2110.14895v1)
- **Homogeneous clusters:**
  - MinnowBoard: ViT-Base reaches 0.63 img/s on 4 devices (1.98×). ViT-Large reaches 1.95 img/s on 16, 7.48× over the 2-stage baseline (ideal 8×). ViT-Huge reaches 0.77 img/s on 16, 3.93× over the 4-stage baseline (ideal 4×).
  - RCC-VE: ViT-Base reaches 0.82 img/s on 4 (1.99×). ViT-Large and ViT-Huge reach 2.43 and 1.01 img/s on 16, which are 10.59× and 11.88× over a single device.
  - ViT-Base stage imbalance is about 25% at 2 stages and 80% at 4 stages, caused by the second dense layer of transformer layer 11 (attributed to weight sparsity).
  — [arXiv 2110.14895v1](https://arxiv.org/abs/2110.14895v1)
- **Heterogeneous Cases 1–6 versus PipeDream average:**
  - Case 4: 1.90× (ViT-L) and 1.54× (ViT-H).
  - Case 5: 1.55×, 3.75× and 3.84× (B/L/H), using only 4, 7 and 7 devices.
  - Case 6: 1.98×, 3.98× and 4.16×, using 4, 12 and 14 devices.
  - EdgePipe avoids the lowest-performing devices, and GPipe/PipeDream vary widely across 10 random device orders.
  — [arXiv 2110.14895v1](https://arxiv.org/abs/2110.14895v1)
- **Bandwidth (5–120 Mbps):** throughput does not drop significantly until bandwidth falls below 30 Mbps. At 5 Mbps, ViT-L and ViT-H still get 2.69× and 3.67× over a single RCC-VE. — [arXiv 2110.14895v1](https://arxiv.org/abs/2110.14895v1)
- **Microbatch (2 stages, MinnowBoard, ViT-Base):** GPipe peaks at about 0.34 img/s and EdgePipe at 0.48 img/s, both at microbatch 12. — [arXiv 2110.14895v1](https://arxiv.org/abs/2110.14895v1)
- **Compression (DeiT on RCC-VE):** DeiT-Base gives 0.62 img/s on 1 device and 0.95 on 4 (ViT-Base gives 0.82). DeiT-Small and DeiT-Tiny reach up to 5.55 and 17.23 img/s. — [arXiv 2110.14895v1](https://arxiv.org/abs/2110.14895v1)
- **Published-version numbers:** the DSD abstract reportedly states 11.88× (ViT-Huge) and 12.78× (BERT-Large) on 16 devices. I saw this only in search summaries, not in the full DSD text. — [ISI record](https://www.isi.edu/results/publications/10310/pipeedge-pipeline-parallelism-for-large-scale-model-inference-on-heterogeneous-edge-devices)

### Inferences
- **[inference] Relation to CROWDio:** PipeEdge is the cleanest prior statement of CROWDio's steady-state pipeline term max(t_comp, t_comm) and of DP-based contiguous partitioning with device selection on heterogeneous CPU boards. CROWDio's Android workers are CPU-bound, as PipeEdge's Atom boards were. PipeEdge is fully static: it has no telemetry-driven replanning, no model-load or switching cost (shards are assumed in place), no failure handling, no phones, and it relies on PyTorch RPC rather than a foreman/worker protocol.
- **[inference]** CROWDio's reported ~48 s cold load is the cost that PipeEdge and EdgeShard silently treat as zero. That is the core differentiation argument.
- **[inference] On the 43% figure:** it may come from a different paper, from a secondary source, or from a confusion with Xuanang Zhao's other work. It should be removed or re-sourced before submission.

### Gaps
- The full DSD 2022 text was not available, so I cannot rule out that the published version reports some 43% number. Nothing in the abstracts I saw suggests it does. The Fig. 1 per-device baseline values are garbled in extraction.

---

## Q3. Vaswani et al. 2017, "Attention Is All You Need": what matters for partitioning transformers?

### Takeaway
NIPS 2017 (31st Conference on Neural Information Processing Systems, Long Beach), arXiv 1706.03762 (v7, 2 Aug 2023). A transformer is a stack of N identical layers. Every sub-layer, including the embeddings, outputs width d_model, so the activation crossing any layer boundary is always an n × d_model tensor. That makes contiguous layer-range cuts uniform in transfer size.

### Cited Findings
- **Layer structure:** the encoder has N = 6 identical layers, each with multi-head self-attention plus a position-wise FFN. Each sub-layer is wrapped as LayerNorm(x + Sublayer(x)). The decoder has N = 6 layers with an added encoder–decoder attention sub-layer and masked self-attention. All sub-layers and embeddings output d_model = 512. — [Vaswani et al., arXiv 1706.03762](https://arxiv.org/abs/1706.03762)
- **Base dimensions:** h = 8 heads with d_k = d_v = d_model/h = 64. FFN(x) = max(0, xW1 + b1)W2 + b2 with inner size d_ff = 2048. The big model uses d_model = 1024, d_ff = 4096, h = 16, about 213M parameters; the base model has about 65M. — [Vaswani et al.](https://arxiv.org/abs/1706.03762)
- **Per-layer cost:** self-attention is O(n²·d) per layer with O(1) sequential operations, versus O(n·d²) for recurrent layers. Positional encodings are sinusoidal and added at the bottom of both stacks. The embedding matrix is shared with the pre-softmax linear layer. — [Vaswani et al.](https://arxiv.org/abs/1706.03762)
- **Results:** 28.4 BLEU on EN-DE. For EN-FR the abstract says 41.8 BLEU but §6.1 says 41.0, an internal inconsistency in the text. Training took 3.5 days on 8 P100 GPUs for the big model. — [Vaswani et al.](https://arxiv.org/abs/1706.03762)

### Inferences
- **[inference]** For CROWDio's DistilBERT (encoder-only, 6 layers, d_model 768 per DistilBERT's own specification, which is not in this paper), every inter-stage activation is batch × seq × d_model regardless of the cut point. Cut-point choice is therefore driven by per-stage compute and memory balance, not activation size. The exception is the embedding and head ends, where token IDs or logits differ in size. This justifies CROWDio exporting every contiguous layer range.
- **[inference]** The residual-plus-LayerNorm wrapping means a cut must fall between whole layers. Cutting inside a sub-layer would require shipping the residual stream as well.

### Gaps
- None material to CROWDio. Training details (§5) and parsing results were read but are not relevant.

---

## Q4. Petals (Borzunov et al. 2023): how does it route, balance load and tolerate churn?

### Takeaway
**PETALS: Collaborative Inference and Fine-tuning of Large Models**, by A. Borzunov, D. Baranchuk, T. Dettmers, M. Ryabinin, Y. Belkada, A. Chumachenko, P. Samygin and C. Raffel. Published in ACL 2023 System Demonstrations, pp. 558–568, anthology ID 2023.acl-demo.54; arXiv 2209.01188v2. Among the eight papers, Petals is the closest analogue to CROWDio's volunteer and churn setting. It has DHT-announced block ownership, client-side routing, server rebalancing when peers leave, and replay-based recovery of attention state. It runs on consumer GPUs, not phones, and gives no explicit pricing of the layer-switching (reload) cost.

### Cited Findings
- **Architecture:** servers hold contiguous Transformer blocks, as many as GPU memory allows. Clients hold the embeddings locally (for BLOOM-176B, 3.6B embedding parameters, needing at least 12 GB client RAM). Clients form a chain of servers covering all blocks, send hidden states and receive outputs. Servers keep attention KV for the session. **Clients store their past inputs to each server, so a failed server can be replaced and its KV rebuilt by replaying those inputs.** — [Petals, ACL 2023 demo](https://aclanthology.org/2023.acl-demo.54) / [arXiv 2209.01188](https://arxiv.org/abs/2209.01188)
- **Requirements:** servers need at least 16 GB CPU RAM, 100 Mbit/s and a GPU with at least 8 GB. Clients are recommended at least 25 Mbit/s. — [Petals](https://arxiv.org/abs/2209.01188)
- **Load balancing:** each server periodically announces its active blocks to a DHT. A joining server picks the contiguous interval with the worst throughput; the interval is "always contiguous, since splitting it would harm the inference latency". All nodes periodically check whether rebalancing "would significantly improve the overall throughput" and, if so, switch layers until throughput is near-optimal. This closes gaps when every server for some block leaves. — [Petals](https://arxiv.org/abs/2209.01188)
- **Client routing:** during generation, latency dominates, so clients ping nearby servers and run beam search for the minimum-time path. During fine-tuning, batches are split across servers. On failure, the client drops the server and reruns routing. — [Petals](https://arxiv.org/abs/2209.01188)
- **Compression:** 8-bit weights (LLM.int8 mixed decomposition, about 0.1% fp16 outliers) cut BLOOM's node count from 44 to 22. The authors say this "reduces latency in half and decreases the probability of a failure". Dynamic blockwise quantization of hidden states halves bandwidth. Quantization overhead is about 5% at batch 1. Table 2 generation throughput on 8× A100 at batch 1/8/32 is 4.18/31.3/100.6 tokens/s in 16-bit and 3.95/29.4/95.8 in 8-bit. — [Petals](https://arxiv.org/abs/2209.01188)
- **Benchmarks (BLOOM-176B, Table 3; single-batch steps/s at sequence 128/2048):**
  - 3× A100: 1.71/1.54 at 1 Gbit/s <5 ms; 1.66/1.49 at 100 Mbit/s; 1.23/1.11 at 100 Mbit/s with 100 ms RTT.
  - 12 virtual servers: 1.24/1.06; 1.24/1.05; 0.57/0.53 under the same three network settings.
  - 14 real servers across Europe and North America (RTX 3060, 2080Ti, 3090, A4000, A5000; 100–1000 Mbit/s; 4 behind firewalls via the libp2p circuit relay): 0.83/0.79 steps/s.
  - Parallel forward with 14 real servers: 32.6 and 179.4 tokens/s at batch 1 and 64.
  - Offloading upper bound on 1× A100: 0.18 steps/s.
  - Inference is mostly latency-sensitive. 8 concurrent clients (12 servers, 100 Mbit/s, 100 ms) each see a 20% slowdown.
  — [Petals](https://arxiv.org/abs/2209.01188)
- **Offloading comparison:** offloading 176B 8-bit parameters over PCIe 4.0 ×16 (256 Gbit/s) takes 5.5 s per pass, or 11 s with two GPUs behind a switch. The introduction cites at least 5.5 s per token for RAM offloading and 22 s for SSD offloading. Offloading is about 10× slower than Petals for single-batch inference. — [Petals](https://arxiv.org/abs/2209.01188)
- **Stated limitations:** an incentive imbalance between servers and clients; privacy (first-layer servers can recover input tokens); security (faulty or malicious servers, with a pledge plus hash-verification idea); and model versioning. — [Petals](https://arxiv.org/abs/2209.01188)

### Inferences
- **[inference] Relation to CROWDio:**
  - *Shared ground:* volunteer devices that join and leave, contiguous-block hosting, and recovery after a server loss. Petals' "contiguous interval" rule matches CROWDio's contiguous shard export.
  - *Differences:*
    1. Petals' rebalancing threshold ("significantly improve") is qualitative. It does not price the time a server spends loading new blocks. CROWDio's J(new) + margin < J(current) rule, with an explicit unload + fetch + load + init cost, is a quantitative version of this.
    2. Recovery: Petals replays inputs to rebuild KV. CROWDio checkpoints and resumes or drains tasks.
    3. Routing: Petals routing is client-driven. The client sends hidden states to servers and receives them back; whether servers forward to each other is not stated (**[inference]**: client-mediated). CROWDio has a direct worker-to-worker activation path plus foreman orchestration.
    4. Hardware: Petals targets ≥8 GB consumer GPUs. CROWDio targets phones running ONNX Runtime Mobile, where load and init dominate (48 s cold vs 6 s warm).
    5. Weight sourcing: Petals has no peer-to-peer weight fetch; servers download weights themselves. CROWDio's peer shard fetch is new relative to Petals.

### Gaps
- The paper gives no measured block-reload time and no rebalancing frequency, convergence time or overhead. Offloading rows in Table 3 are partly ambiguous in column order after extraction, so I cite only the clear 0.18 steps/s value.

---

## Q5. Kafetzis, Khalili & Koutsopoulos 2025: does attention-head partitioning with migration price reconfiguration?

### Takeaway
**Large Language Model Partitioning for Low-Latency Inference at the Edge** (AUEB; Huawei European Research Center), arXiv 2505.02533v1, 5 May 2025. I found no peer-reviewed venue, only the arXiv record. Of the eight papers, this is the only one whose per-interval objective explicitly adds a **migration delay** to inference delay. That migration delay is only the state-transfer time m_i/R_jk; it covers no load or initialization cost. The method is a myopic, simulation-only, single-decoder-layer study.

### Cited Findings
- **Model:** one controller. Time is divided into intervals of a fixed number of tokens. In each interval the controller collects available memory M_j(τ), compute C_j(τ) ≤ W_j and link bandwidth R_jk(τ), all assumed constant within the interval or treated as predictions. The blocks are the h attention heads (each co-located with its KV cache), the projection and the FFN. Memory m_i(τ) and compute b_i(τ) grow with τ. — [Kafetzis et al., arXiv 2505.02533](https://arxiv.org/abs/2505.02533)
- **Migration cost:** D_mig(i, j→k, τ) = m_i(τ−1)/R_jk(τ), i.e. footprint including KV divided by bandwidth. At most one migration per head per interval is allowed, "to avoid back-and-forth overhead". The per-interval objective is min D_T(τ) + D_mig_total(τ) subject to memory, where D_T is the max over heads of (input transfer + head compute + head→proj transfer) plus proj→ffn transfer. — [Kafetzis et al.](https://arxiv.org/abs/2505.02533)
- **Myopic heuristic:**
  - The score is S(i,j,τ) = max(m_i/M_j, b_i/C_j, CommFactor). Blocks are sorted by demand, each goes to the argmin-score device, and devices with S ≤ 1 count as individually feasible.
  - When a device is overloaded, ResolveResourceOverload migrates other blocks, then BacktrackForResourceViolations runs. The search is bounded by a time limit T_max and U = |B|·|V| iterations, returning INFEASIBLE if either is exceeded.
  - Complexity is O(|B|²|V|). The interval is chosen as "a few seconds" to leave time for migrations.
  — [Kafetzis et al.](https://arxiv.org/abs/2505.02533)
- **Evaluation:** a custom discrete-event Python simulator; the algorithm runs every token (interval = 1, which the authors call a worst case for migration). One layer with h = 32, D = 2048 and L0 = 64. Memory is 2–8 GB and compute 5–50 GFLOPS (log-normal); links are 1–10 Gbps with full connectivity. The medium scale uses 25 devices, up to N = 1000 tokens, with background load injected. — [Kafetzis et al.](https://arxiv.org/abs/2505.02533)
- **Results:**
  - Small scale (3–5 devices, N = 4): within 15–20% of an exhaustive optimum. Greedy and the other baselines lag by 40–60%.
  - 25 devices: at the final token, EdgeShard exceeds 1000 s, Galaxy reaches about 400–600 s, and the proposed method stays under 200 s.
  - Memory: EdgeShard and Galaxy exceed 7 GB by n = 100, versus about 6 GB for the proposed method. The paper claims "up to 9–10 times speedup".
  — [Kafetzis et al.](https://arxiv.org/abs/2505.02533)
- **Stated future work:** multi-layer models, limited foresight or prediction, and real testbeds with energy constraints. — [Kafetzis et al.](https://arxiv.org/abs/2505.02533)

### Inferences
- **[inference]** The figure values (>1000 s vs <200 s) imply about 5× or more against EdgeShard at the final token. The "9–10×" headline is not directly traceable to a stated figure value.
- **[inference] Relation to CROWDio:**
  - *Shared ground:* CROWDio shares the core idea that a placement change has a cost that must be traded against inference gain.
  - *Gaps CROWDio fills:*
    1. Kafetzis prices only bytes over a link. There is no unload, no storage fetch, and no session load or init, which on phones is the dominant term (CROWDio measured ~48 s cold vs ~6 s warm).
    2. There is no hysteresis margin or cooldown, only a one-migration-per-head cap. The myopic min(D_T + D_mig) can oscillate when D_mig is small.
    3. It is simulation-only, with Gbps links and GFLOPS devices, not phones.
    4. It has no churn or failure model; devices persist and only their capacities vary.
    5. Head-level partitioning requires fine-grained execution graphs, unlike CROWDio's ONNX layer-range shards.
  - *Positioning for CROWDio:* "migration-aware, but migration = state transfer only".

### Gaps
- No numeric tables were given; results are read from Figs. 3–4 and the prose. The baselines' parameters are under-specified, and there is no real-hardware validation.

---

## Q6. LOIP (Sun et al.): how does offloading-based interleaved pipeline parallelism handle weight-loading cost?

### Takeaway
**LOIP: Collaborative LLM Inference with Offloading-based Pipeline Parallelism on Edge Devices**, by M. Sun, X. Zhang, S. Qu, Y. Li, M. Xiao, Y. Zheng, Y. Yuan and D. Yu (Shandong University). arXiv 2512.21835v3, with the v3 stamp dated 22 Sep 2026; it uses the IEEE journal template and I found no venue. This is the paper that treats **repeated weight loading as a first-class latency term**, and it hides that cost by overlapping it with computation. Its runtime adaptation (block-level offload thresholds, KV transfer) deliberately avoids re-running the scheduler. It also avoids reallocating layers because, in the authors' words, re-allocating "would incur excessive overhead". That is an implicit argument for CROWDio's switching-cost pricing.

### Cited Findings
- **Problem:** on UMA Jetsons (CPU and GPU share memory), weights that do not fit must be loaded from SSD each decoding step, creating "bubbles for loading". Fluctuating bandwidth and KV growth compound this. The memory example: LLaMA3.3-70B needs at least 130 GB, while a Jetson Xavier NX has 16 GB. — [LOIP, arXiv 2512.21835](https://arxiv.org/abs/2512.21835)
- **Cost model (per decoding step):** T_total = T_comp + #Stage·|D|·h_size/bw_net + #Stage·max_i max(load(L̃_i)/#Stage − overlap window, 0). The overlap window covers the device's own resident compute, the other devices' compute and communication. The memory constraint is resident layers·l_size + largest offloaded unit per stage + M_KV(n) ≤ Mem_i, with #Stage ≥ 2. The model rearranges to T_total = max(T_comp + T_comm, max_i b_i). — [LOIP](https://arxiv.org/abs/2512.21835)
- **Scheduler:** for each stage count s, a threshold-constrained DP F(l,i) over whole layers finds q_i(k), the minimum number of offloaded layers. Theorem 1 states this is optimal over whole-layer allocations under equal layer cost and zero initial KV. A greedy block refinement then keeps MLP or MHA blocks resident on the bottleneck device. The worst-case bound is O(D·L⁴); in practice it finishes in under 2 s for 80-layer LLaMA3.3-70B on 4 devices. — [LOIP](https://arxiv.org/abs/2512.21835)
- **UMA-aware loading:** a single pinned host buffer mapped into GPU address space, sized to the largest slice and filled with io_uring. Cached "proxy modules" hold byte offsets rather than tensors, and a custom CUDA op feeds the mapped address to cuBLAS. CUDA events gate the overlap of loading and compute, so no second buffer is needed. — [LOIP](https://arxiv.org/abs/2512.21835)
- **Online memory-aware planner:** device-specific token thresholds TS_ij, where TS_i1 is the largest n with M_KV(n) ≤ Mem_i − M_W. At each threshold an extra MHA or MLP block is offloaded (Eq. 9 minimizes the added loading). The authors state "the switching overhead is bounded by a single block (MHA or MLP)". — [LOIP](https://arxiv.org/abs/2512.21835)
- **Bandwidth-sensitive KV transfer:** a low-threshold device parks part of its KV on a paired peer and fetches it back before the stage reruns. The budget is V_trans = sgn(Δ_i)·⌊bw·|Δ_i|/κ_i⌋. When bandwidth drops, the volume is reduced immediately. **When bandwidth rises, the adjustment is deferred unless the next threshold is imminent, and a fluctuation threshold n_ts filters minor changes.** This is a hysteresis-like rule. — [LOIP](https://arxiv.org/abs/2512.21835)
- **Setup:**
  - Hardware: Jetson Xavier NX 16 GB, AGX Orin 32 GB and AGX Orin 64 GB at 1000 Mbps, shaped with tc to 100 or 200 Mbps (the dynamic run samples 50–250 Mbps).
  - Models: LLaMA2-13B, Qwen3-32B and LLaMA3.3-70B on WikiText-2.
  - Request patterns: sporadic (batch 1) and bursty (|D| concurrent).
  - Baselines: PP, PP + offloading, EdgeShard, Galaxy, TPI-LLM and TPI-LLM + offloading. Jupiter is excluded because it assumes full residency.
  - Failure labels: OOT means more than 5 s/token (sporadic) or 2 s/token (bursty).
  — [LOIP](https://arxiv.org/abs/2512.21835)
- **Results:**
  - Headline: up to 8.8× (sporadic, single input) and 20.3× (bursty, concurrent input).
  - KV-constrained runs: 2.5–20.9× and 2.7–13.4× over TP baselines (bursty, at 100 and 200 Mbps); 2.2–11.6× and 2.5–8.1× (sporadic). Over PP baselines: 1.4–8.7× and 1.5–9.6× (bursty); 1.4–5.5× and 1.4–6.2× (sporadic).
  - Dynamic bandwidth: 3.9–21.3× (bursty) and 1.7–8.8× (sporadic).
  - Severe-memory runs: EdgeShard and plain PP go out of memory, and Galaxy always goes out of memory.
  — [LOIP](https://arxiv.org/abs/2512.21835)
- **Ablation** (70B on 2× Orin 64, 1× Orin 32, 1× NX): full LOIP gives 1533.9 ms/token sporadic and 407.1 bursty.
  - Without KV transfer: 1614.2 / 455.6 (+5.2% / +11.9%).
  - Without the scheduler (uniform allocation): 1939.8 / 482.2 (+26.5% / +18.4%).
  - Without UMA loading: 2251.7 / 488.4 (+46.8% / +20.0%).
  — [LOIP](https://arxiv.org/abs/2512.21835)

### Inferences
- **[inference] Relation to CROWDio:**
  - *Shared mechanisms:* LOIP's "load from storage" cost is the closest technical analogue to CROWDio's fetch + load + init. Its single reusable buffer resembles CROWDio's single-residency constraint (one ONNX session per phone).
  - *Different regime:* LOIP streams weights within a fixed plan every decoding step and overlaps the cost. CROWDio switches the resident shard occasionally and prices the switch against expected gain over the remaining work.
  - *What LOIP lacks:* any device churn, failure or checkpoint handling; worker selection beyond the given device set; phones. All experiments use NVIDIA Jetson with CUDA kernels, which do not transfer to ONNX Runtime Mobile or NNAPI.
  - *Overlap with CROWDio:* LOIP's deferred-on-improvement bandwidth rule parallels CROWDio's hysteresis and cooldown. LOIP's KV parking on a peer parallels CROWDio's peer-to-peer data paths, but for state rather than weights.
- **[inference]** LOIP's explicit statement that re-running allocation would be too costly supports CROWDio's position. It also shows a gap: LOIP never quantifies when re-planning would pay off, and CROWDio's J(new) + margin < J(current) test does exactly that.

### Gaps
- The Fig. 6–8 per-bar latencies were lost in extraction (garbled axis labels), so only the prose ranges are reported. Absolute load-time measurements per slice are not reported in text.

---

## Q7. Koch, Djuhera & Binotto 2025: what runtime re-split and re-placement framework do they propose, and what evidence backs it?

### Takeaway
**Intelligent Orchestration of Distributed Large Foundation Model Inference at the Edge** (Florida Atlantic University; TU Munich; Carl Zeiss). arXiv 2504.03668v3 (12 Jul 2025); v1 (Mar 2025) was titled "Adaptive Orchestration for Inference of Large Foundation Models at the Edge". It is also listed in *Computer Networks and Communications* 3(2):111–128, Sep 2025, DOI 10.37256/cnc.3220256807. It is a **position and architecture paper derived from IBM patent US20250071069A1, with no original experiments**. Its trigger design is strikingly close to CROWDio's: EWMA latency, utilization, bandwidth floor and cooldown, trying migration before re-split. Its cost function has **no reconfiguration or model-load cost term**.

### Cited Findings
- **Architecture:** four modules: Monitoring & Capacity Profiling (adds layer-granular latency, activation size and privacy tags); Adaptive Orchestrator (keep the split, redistribute sub-splits, or fully re-split); Split Revision; and Reconfiguration Broadcast (pushes "on-the-fly–generated weight shards and graph rewiring commands"). It is positioned against Kubernetes, KubeEdge, Ray Serve, Triton and InferLine, which treat models as opaque, and against EdgeShard, which the authors call manual and static. — [Koch et al., arXiv 2504.03668](https://arxiv.org/abs/2504.03668); [journal listing](https://ojs.wiserpub.com/index.php/CNC/article/view/6807)
- **Formulation:** capacity profile CP(n_j, t) = {CPU, GPU, Mem, NetCap}, partitions S_1…S_k, and placement x_ij. The objective is Φ = αL + βU + γP (latency, imbalance, privacy penalty), subject to unique assignment, capacity and trusted-node privacy constraints. Split revision is min over (S ∈ Σ, x) of Φ. — [Koch et al.](https://arxiv.org/abs/2504.03668)
- **Algorithm 1 and triggers (Table 3 defaults):**
  - Triggers: EWMA latency > L_max = 150 ms; max GPU/CPU utilization > U_max = 0.85; minimum edge-link bandwidth < B_min = 50 Mbps; a privacy-policy violation.
  - Reconfiguration happens only if the best mapping d̂ ≠ d_t and t − t_last ≥ T_cool = 30 s.
  - If multiple triggers fire, the orchestrator tries placement migration first and re-splits only if migration fails.
  — [Koch et al.](https://arxiv.org/abs/2504.03668)
- **Motivating numbers:** a 7B LLM takes about 25 ms on an RTX A6000 versus more than 250 ms on a Jetson. Backhaul ranges from under 1 ms (mmWave) to 30 ms (congested Wi-Fi). — [Koch et al.](https://arxiv.org/abs/2504.03668)
- **"Expected results" (Tables 4–5, Fig. 3):** these are explicitly compiled from prior studies and empirical estimates, not measured by the authors. Static versus adaptive: 500–1000 ms vs 100–300 ms latency; ≤1 vs ≥5 req/s; 50–60% vs 80–95% utilization; 60–70% vs 95–99% SLA hit-rate (400 ms budget); 5–10 vs 0–2 downtime incidents per hour. They also claim monitoring costs about 10 ms per cycle. — [Koch et al.](https://arxiv.org/abs/2504.03668)
- **Future work:** RL-based decisions, SMPC and homomorphic encryption, network-aware partitioning and standard benchmarks. — [Koch et al.](https://arxiv.org/abs/2504.03668)

### Inferences
- **[inference] Relation to CROWDio:**
  - *Overlap:* Koch et al. is conceptually nearest to CROWDio's reconfiguration-aware planner (EWMA telemetry, cooldown against thrashing, migrate before re-split), and it is a useful citation that the problem is recognized.
  - *Where CROWDio goes further:*
    1. Koch triggers on threshold violations and accepts any lower-cost mapping (C(d̂) < C(d_t)). It does not subtract the cost of getting there. CROWDio requires J(new) + margin < J(current), with switching cost priced as unload + fetch + load + init under single residency.
    2. Koch's Reconfiguration Broadcast generates shards on the fly. CROWDio pre-exports every contiguous range with a sha256 manifest and prefetches or peer-fetches them, which removes generation from the critical path.
    3. Koch has no churn or checkpoint semantics, and "failover" appears only in the compiled table.
    4. Koch has no implementation. CROWDio has a 5-phone measurement.
  - *Caveat for citing:* Tables 4–5 are secondary aggregates, and the review should not cite them as measured evidence.

### Gaps
- No implementation, simulation or measured reconfiguration cost. The sources behind Tables 4–5 ([28, 43, 45–50]) could not be checked because references were stripped.

---

## Q8. Kabakibo, Trivedi & Wang 2026, "Breaking the Ice": what does vLLM cold start consist of, and can it be predicted?

### Takeaway
**Breaking the Ice: Analyzing Cold Start Latency in vLLM** (Paderborn University; IBM Research Europe). Proceedings of the 9th MLSys Conference, Bellevue WA, 2026, according to the paper's own footer; arXiv 2606.07362v1 (5 Jun 2026). It is a GPU-server measurement study. vLLM startup (about 20 s for Llama3.2-3B) splits into six steps that are mostly CPU-bound, and **weight loading is only about 7–10% of startup**. Each step scales near-linearly with a simple parameter, so a per-step linear-regression predictor estimates total startup well. The finding that "load bytes" is a poor proxy for cold start is directly relevant to CROWDio's switching-cost model.

### Cited Findings
- **Definition and setup:** startup runs from engine initialization until vLLM is ready to serve ("Application startup complete"). 22 models (Table 1) on H100 NVL and L40S, using vLLM v0.10.1.1, PyTorch 2.7.1 and CUDA 12.6/12.8. Each experiment is the average of 5 runs. Container start, remote storage and network are deliberately excluded. — [Kabakibo et al., arXiv 2606.07362](https://arxiv.org/abs/2606.07362)
- **Version variance:** OPT-6.7B on H100 shows more than 4× variance across 9 vLLM releases (values between 7.8 s and 36.41 s appear in Fig. 1), with a 2× reduction from v0.9 to v0.10. — [Kabakibo et al.](https://arxiv.org/abs/2606.07362)
- **Six steps (Llama3.2-3B, total 20.32 s):**
  1. Framework bootstrap: platform detection, imports, model info, worker initialization. It is model-independent. Get Model Info dropped from about 4.47 s to 0.12 s with metadata caching in v0.11, which was not used in their runs.
  2. Tokenizer initialization: linear in tokenizer file size (PCC 0.99), 0.08–0.29 s.
  3. Model loading: structure initialization takes about 0.1 ± 0.05 s regardless of size; weight loading is linear in model size (PCC 1.00), about 0.5–1 s for small models and nearly 5 s for DeepSeek-V2-Lite-16B, measured with a warm page cache.
  4. torch.compile: Dynamo transformation is linear in compiled-graph size (PCC 0.96; e.g. Llama2-7B 4.86 s, Llama2-13B 5.97 s). Loading cached compiled graphs is linear too (PCC 0.95), from 2.89 s for Qwen-MoE-14.3B (808 KB) to 5.66 s for Yi-9B (1.69 MB).
  5. KV-cache profiling: a dummy forward pass, linear in size for non-MoE models (PCC 0.92), 0.67–0.79 s; MoE models take 0.94–0.97 s.
  6. CUDA-graph capture: linear in model size (PCC 0.99, 0.91–1.51 s) and in the number of batch sizes (PCC 1.00; 0.33 s for 3 sizes to 1.8 s for 60).
  — [Kabakibo et al.](https://arxiv.org/abs/2606.07362)
- **CPU dominance:** going from L40S to H100 gives about no speedup except about 1.2× in graph capture. Changing the CPU (EPYC 9354 vs Xeon 8568Y+) has a larger and inconsistent per-step effect. At any moment at least one core is at 100%, and rarely more than two. — [Kabakibo et al.](https://arxiv.org/abs/2606.07362)
- **SSD cold reads** (4× PCIe 5.0, 25 GB/s): only model loading is affected (normalized 1.52 in Fig. 13), and total startup changes by only about 1.04×, because loading is about 7–10% of total. Loader backends matter: CoreWeave Tensorizer loads in 53–60% of the Safetensors time, and the Run:ai streamer gives moderate gains. — [Kabakibo et al.](https://arxiv.org/abs/2606.07362)
- **Without the compile cache** (VLLM_DISABLE_COMPILE_CACHE=1): graph storing takes 11–21 s, versus 3–6 s for loading cached graphs. — [Kabakibo et al.](https://arxiv.org/abs/2606.07362)
- **Predictor:** a white-box, per-step linear regression (training data collection takes about hours, training about milliseconds, prediction about 100s of ms), applied to non-MoE models. Validation on Falcon-11B, Gemma-7B, Mistral-7B, Llama3-3B and Qwen-7B gives "MSE of 2.42 secs" with a maximum error of 2.08 s (Llama3-3B). On v0.11 the "MSE" is 2.62 s. The authors also say it can estimate vLLM sleep-mode (fast re-initialization) trends. — [Kabakibo et al.](https://arxiv.org/abs/2606.07362)
- **Limitations:** engine-only scope; linearity fails for MoE, SSM-hybrid and diffusion models (affecting KV profiling only); the predictor is hardware-specific and must be retrained; vLLM log granularity introduces up to hundreds of ms of error. Related cold-start systems are cited but not evaluated: ServerlessLLM, ParaServe, Medusa, TIDAL, and CSGO (edge-distributed cold start). — [Kabakibo et al.](https://arxiv.org/abs/2606.07362)

### Inferences
- **[inference]** The paper reports "MSE" in seconds, which suggests it means RMSE or MAE. Cite the error as "about 2 s on about 22–33 s startups" rather than as an MSE.
- **[inference]** The mapping of the individual Fig. 2 bar segments (1.37, 4.86, 1.66, 0.92, 4.57, 3.96, 1.52, 1.00 s) to step names is ambiguous after extraction. Cite only the 20.32 s total and the per-step ranges given in the prose.
- **[inference] Relation to CROWDio:**
  - *Direct analogy:* this is the best citation for decomposing cold start into stages and modeling each linearly. CROWDio's switching cost (unload + fetch + load + init) is the same idea applied to an ONNX Runtime Mobile session on phones.
  - *Methodological lesson:* initialization costs that are not proportional to bytes (session creation, graph optimization, the first-run dummy pass) can dominate. That matches CROWDio's cold ~48 s vs warm ~6 s gap, and it argues for CROWDio to profile its own init term rather than derive it from shard size alone.
  - *Differences:* the study covers single-node GPU servers in a serverless context. It has no partitioning, no phones, and no use of the predictor inside a placement optimizer; integration into orchestration is listed as future work.
  - *CROWDio's contribution relative to this paper:* it closes that loop by using a startup-cost estimate inside the reconfiguration decision.

### Gaps
- The individual bar values in Figs. 1, 2, 10, 11 and 13 are partly garbled. Measurements do not cover edge or mobile runtimes, ONNX Runtime, or TFLite.

---

## Q9. Cross-paper synthesis: where is CROWDio's novelty relative to these eight papers?

### Takeaway
Across the eight papers, **no work combines all four of CROWDio's distinctive elements**:
1. an explicit, measured model-(re)load and initialization cost in the placement objective under single residency;
2. a hysteresis and margin test (J_new + margin < J_cur) with a cooldown;
3. churn handling through checkpoint, resume or drain;
4. a phone runtime, with peer shard fetch and direct worker-to-worker activations.

Each element appears separately and partially in one or two of the papers.

### Cited Findings

**Static DP partitioners.** EdgeShard and PipeEdge/EdgePipe treat weights as resident after a one-time deployment. Neither has a load-cost term or runtime replanning.
- [EdgeShard](https://doi.org/10.1109/JIOT.2024.3524255); [arXiv 2110.14895](https://arxiv.org/abs/2110.14895v1)

**Steady-state pipeline throughput.** Both papers model steady-state throughput as the slowest stage, max(t_comp, t_comm), which is CROWDio's per-batch term.
- [EdgeShard](https://doi.org/10.1109/JIOT.2024.3524255); [arXiv 2110.14895](https://arxiv.org/abs/2110.14895v1)

**Migration or reconfiguration costs.**
- Kafetzis et al. price only the state-transfer time m_i/R_jk.
- LOIP prices per-step weight loading inside a fixed plan, and its block-level switch is "bounded by a single block".
- Koch et al. and Petals reconfigure without any cost term: Koch's test is "C(d̂) < C(d_t)" and Petals rebalances if it "would significantly improve".
- Sources: [Kafetzis](https://arxiv.org/abs/2505.02533); [LOIP](https://arxiv.org/abs/2512.21835); [Koch](https://arxiv.org/abs/2504.03668); [Petals](https://arxiv.org/abs/2209.01188)

**Anti-thrashing mechanisms.**
- Koch: T_cool = 30 s.
- Kafetzis: at most one migration per head per interval.
- LOIP: deferred upward bandwidth adjustment plus a fluctuation threshold n_ts.
- None uses a cost-margin rule.
- Sources: [Koch](https://arxiv.org/abs/2504.03668); [Kafetzis](https://arxiv.org/abs/2505.02533); [LOIP](https://arxiv.org/abs/2512.21835)

**Churn and fault tolerance.** Only Petals handles devices leaving: DHT announcements, client re-routing, KV rebuilt by replaying stored inputs, and server rebalancing to close gaps.
- [Petals](https://arxiv.org/abs/2209.01188)

**Cold-start anatomy.** Kabakibo et al. show that engine cold start is mostly CPU-bound initialization rather than weight I/O, with weight loading about 7–10% of startup.
- [Kabakibo](https://arxiv.org/abs/2606.07362)

**Hardware in the testbeds.**
- Jetsons plus an RTX 3090: EdgeShard.
- Jetsons: LOIP.
- Intel Atom boards: PipeEdge.
- A100 and consumer GPUs: Petals.
- H100 and L40S: Kabakibo.
- Simulation only: Kafetzis.
- No experiments: Koch.
- None uses smartphones, although EdgeShard and Kafetzis name smartphones as motivating devices.
- Sources: all eight papers above.

### Inferences
- **[inference] Gaps CROWDio can claim, phrased cautiously as "among the surveyed works":**
  - (i) Reconfiguration decisions that price the full switch: unload + artifact fetch (including from peers) + session load + init.
  - (ii) A margin-gated switch rule with cooldown and epochs, rather than threshold-triggered or myopic cost minimization.
  - (iii) A real multi-phone deployment showing a cold/warm init ratio of about 8× (48 s vs 6 s), the mobile counterpart to Kabakibo's server-side breakdown.
  - (iv) Checkpoint, resume and drain semantics on worker disconnect, as opposed to Petals' replay-based KV rebuild.
  - (v) Peer-to-peer shard distribution with prefetch to disk, which no surveyed paper has. Petals servers download weights themselves; Koch broadcasts "on-the-fly–generated shards".
- **[inference] Encoder versus decoder workloads:** most surveyed work targets autoregressive decoders, where KV-cache growth is the main dynamic. CROWDio's DistilBERT pipeline is encoder-only and batch-streamed, so its dominant dynamic is device availability and residency, not KV growth. Reviewers may ask how CROWDio extends to decoder LLMs; LOIP and Kafetzis are the relevant pointers.
- **[inference] Citation corrections for the user's paper:**
  - Replace "Zhao et al., EdgePipe, INFOCOM 2022" with "Hu et al., PipeEdge, DSD 2022", or cite arXiv 2110.14895 as Hu et al. 2021.
  - Remove the "43% inter-stage transfer" claim; it is not in this text.
  - Cite Koch et al.'s quantitative table only as an aggregate of prior work.
  - Treat the Kafetzis "9–10×" figure as a simulation result.

### Gaps
- I could not verify a peer-reviewed venue for Kafetzis et al. or LOIP; they are arXiv-only as far as the searches showed.
- The DSD 2022 PipeEdge full text was not read.
- No surveyed paper reports on-phone ONNX or TFLite load and init measurements, so there is no external baseline for CROWDio's 48 s / 6 s numbers.
- The references were stripped from all files, so claims each paper makes about third-party work (e.g. Koch's Tables 4–5 sources, CSGO in Kabakibo) could not be traced.
