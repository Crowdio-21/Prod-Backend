# Backend / Orchestrator Literature for CROWDio (MCC frameworks, scheduling, bandwidth-awareness, fault tolerance, DAGs, evaluation)

Verification legend used throughout:
- [V] = existence and bibliographic details confirmed by a web source fetched or retrieved in this research session (URL given).
- [K] = well-known classic cited from domain knowledge; NOT re-fetched in this session. The report writer should treat bibliographic details as high-confidence but spot-check DOIs before final publication.
- NEW = 2024–2026 work.

Research budget note: about 16 search/fetch calls were made. The MDPI full text for Honeybee-Tx returned HTTP 403, so its numbers come from abstracts and indexes, not the full paper.

## Q1. Mobile crowd computing, smartphone grids, volunteer and opportunistic computing frameworks (incl. newest)

### Takeaway
The MCC framework line runs from the 2009–2013 prototypes (Hyrax, Misco, Honeybee, Serendipity, CWC) to a small set of 2024–2026 systems: Honeybee-Tx (Future Internet 2025), Kiridana et al.'s mobile-agent work-stealing system (MERCon 2024), and CROWDio itself (arXiv 2604.19363, April 2026). Nearly all prior systems are decentralised work stealing or MapReduce research prototypes. None combines a centralised telemetry-driven MCDM scheduler, network-cost-aware dispatch, pipeline DAGs and framework-aware delta checkpointing. That combination is CROWDio's niche.

### Cited Findings
**CROWDio's own preprint (NEW, 2026)**
- [V] L. Manamperi, D. Pathirana, T. Pathirana, N. Premarathna, K. Gunasekara, "CROWDio: A Practical Mobile Crowd Computing Framework with Developer-Oriented Design, Adaptive Scheduling, and Fault Resilience," arXiv:2604.19363 (cs.DC), submitted 21 Apr 2026. It has a Coordinator–Worker design, a declarative SDK (one function annotation), tiered checkpointing and a pluggable MCDM scheduler driven by live telemetry. On 6 heterogeneous Android devices, capability-aware scheduling cut total execution time by up to 56.9% vs. round-robin. Checkpointing overhead was bounded at 2–3 s per task regardless of frequency, and Jain's fairness index was 0.889 — [arXiv abs](https://arxiv.org/abs/2604.19363)
- [V] The CROWDio preprint's stated limitations are a 6-device scale (20–100 devices untested), no modelling of mobility, user interruption or voluntary participation, EDAS/ARAS/MABAC vs. WRR at scale left as future work, energy measurement left as future work, and multi-stage DNN pipeline inference as future work. The preprint does NOT discuss bandwidth-aware scheduling or simulators, so the current backend branch (bandwidth-aware scheduler, Sufferage batch mapping, trace-driven simulator, virtual fleet) goes beyond the published paper — [arXiv HTML](https://arxiv.org/html/2604.19363)
- [V] The CROWDio preprint's Table 1 claims it is the only compared system that is mobile-targeted with MCDM scheduling, tiered checkpointing and an SDK. Its references include Hyrax, Misco, Honeybee, Kiridana 2024, BOINC 2020, Saleh & Shastry 2022, Pramanik 2021/2024, Stan et al. 2021, Ray (OSDI 2018) and Spark (CACM 2016) — [arXiv HTML](https://arxiv.org/html/2604.19363)

**Direct competitors and precedents (newest first)**
- [V] NEW. S. Segu Nagesh, N. Fernando, S. W. Loke, A. Ghari Neiat, P. N. Pathirana, "A Dependency-Aware Task Stealing Framework for Mobile Crowd Computing," *Future Internet* 17(10):446, Oct 2025, DOI 10.3390/fi17100446. Honeybee-Tx adds capability-aware job selection, static dependency-aware work stealing and staged result transfers. On video processing it reports 2.06× speed-up vs. monolithic execution, 51.4% energy reduction, 71.6% memory savings and 42% less network traffic than dependency-oblivious approaches. Another summary reports up to 4.72× speed-up and 63% lower per-device energy for human-activity-recognition workloads on five Android devices. Relation to CROWDio: this is the closest 2025 competitor. It is decentralised (work stealing) with static dependencies. CROWDio is centralised and handles dynamic upstream-result injection, streaming 1:1 vs. barrier edges, and network-cost-aware placement — [MDPI](https://www.mdpi.com/1999-5903/17/10/446); [IDEAS/RePEc](https://ideas.repec.org/a/gam/jftint/v17y2025i10p446-d1761063.html); [Bohrium summary](https://www.bohrium.com/paper-details/a-dependency-aware-task-stealing-framework-for-mobile-crowd-computing/1179666080608550912-21695)
- [V] S. Segu Nagesh, N. Fernando, S. W. Loke, et al., "Opportunistic Mobile Crowd Computing: Task-dependency Based Work-Stealing," ACM MobiCom 2022 (demo). It introduced Honeybee-T, which shares work by device computational ability and task dependencies. It is the precursor of Honeybee-Tx — [Deakin DRO](https://dro.deakin.edu.au/articles/conference_contribution/Opportunistic_mobile_crowd_computing_Task-dependency_based_work-stealing/21601323); [dblp](https://dblp1.uni-trier.de/rec/conf/mobicom/NageshFLNP22.html)
- [V] NEW. T. B. Kiridana, S. Kapukotuwa, B. S. Dissanayake, K. Gunasekera, "Mobile Crowd Computing with Mobile Agents and Work Stealing," Moratuwa Engineering Research Conf. (MERCon) 2024, IEEE, pp. 97–102, DOI 10.1109/MERCon63886.2024.10688742. It uses mobile agents plus work stealing for load balancing across heterogeneous smart mobile devices. It comes from the same University of Moratuwa group (Gunasekera) as CROWDio and is CROWDio's in-house predecessor; full-text results were not retrieved — [UoM repository](https://dl.lib.uom.lk/handle/123/25436)
- [V] N. Fernando, S. W. Loke, W. Rahayu, "Computing with nearby mobile devices: a work sharing algorithm for mobile edge-clouds," *IEEE Trans. Cloud Computing* (2019). Honeybee work stealing balances independent jobs across heterogeneous phones and tolerates nodes joining and leaving. On an Android prototype it reached up to 4× speed-up with 7 devices and up to 71% energy savings with 8 devices — [Deakin DRO](https://dro.deakin.edu.au/articles/journal_contribution/Computing_with_nearby_mobile_devices_a_work_sharing_algorithm_for_mobile_edge-clouds/20870146)
- [V] N. Fernando, S. W. Loke, W. Rahayu, "Honeybee: A Programming Framework for Mobile Crowd Computing," MobiQuitous 2012, LNICST 120, Springer 2013, DOI 10.1007/978-3-642-40238-8_19. This is the original work-stealing MCC programming framework — [Springer](https://rd.springer.com/chapter/10.1007/978-3-642-40238-8_19)
- [V] (cited in CROWDio) E. E. Marinelli, "Hyrax: Cloud Computing on Mobile Devices using MapReduce," CMU-CS-09-164, 2009; A. Dou et al., "Misco: A MapReduce Framework for Mobile Systems," PETRA 2010 — [CROWDio refs](https://arxiv.org/html/2604.19363)
- [V] D. P. Anderson, "BOINC: A Platform for Volunteer Computing," *J. Grid Computing* 18(1):99–122, 2020. This is the reference volunteer-computing middleware, with heterogeneity and churn handled mainly for desktops. It is a precedent for CROWDio's pull-based dispatch and replication thinking — [CROWDio refs](https://arxiv.org/html/2604.19363)
- [K] M. Y. Arslan, I. Singh, S. Singh, H. V. Madhyastha, K. Sundaresan, S. V. Krishnamurthy, "CWC: A Distributed Computing Infrastructure Using Smartphones," *IEEE TMC* 14(8), 2015 (earlier "Computing while charging" at IEEE PerCom 2012). This centralised scheduler used phones only while they were charging, with task profiling and bandwidth-aware assignment of small jobs. It is the closest classic precedent to CROWDio's centralised and bandwidth-aware design.
- [K] C. Shi, V. Lakafosis, M. Ammar, E. Zegura, "Serendipity: Enabling Remote Computing among Intermittently Connected Mobile Devices," ACM MobiHoc 2012. It distributes PNP-DAG jobs over opportunistic contacts and is a precedent for DAG jobs on intermittently connected phones.
- [K] E. Miluzzo, R. Cáceres, Y.-F. Chen, "Vision: mClouds – Computing on Clouds of Mobile Devices," ACM MCS Workshop (MobiSys) 2012. This is a position paper on mobile-device clouds.

**Surveys**
- [V] NEW. P. K. D. Pramanik, S. Pal, P. Choudhury, "Mobile crowd computing: potential, architecture, requirements, challenges, and applications," *J. Supercomputing* 80(2):2223–2318, 2024 (online 2023), DOI 10.1007/s11227-023-05545-0. This is a 95-page overview defining MCC, architecture models and requirements, and comparing it with desktop grids, clusters, mobile grid/cloud, ad hoc mobile cloud and crowdsourcing. It is the canonical framing reference for CROWDio's introduction — [dblp](https://dblp.org/rec/journals/tjs/PramanikPC24); [OUCI](https://ouci.dntb.gov.ua/works/4EVkYrql)
- [V] NEW. A related 2024 *J. Supercomputing* paper (vol. 80, issue 16) from the same group: "Sustainable edge computing with mobile crowd computing: a proof-of-concept with a smart HVAC use case" — [search index record](https://bvbvf1.bib-bvb.de/vufind/PrimoRecord/cdi_proquest_journals_3095599272)
- [V] N. Fernando, S. W. Loke, W. Rahayu, "Mobile cloud computing: A survey," *FGCS* 29(1):84–106, 2013. It identifies the lack of developer-facing abstractions as a persistent gap — [CROWDio refs](https://arxiv.org/html/2604.19363)
- [V] A. Ray, C. Chowdhury, S. Bhattacharya, S. Roy, "A survey of mobile crowdsensing and crowdsourcing strategies for smart mobile device users," *CCF Trans. Pervasive Computing and Interaction* 5:98–123, 2023 — [CROWDio refs](https://arxiv.org/html/2604.19363)

**Smartphone/dew-computing clusters (Hirsch, Mateos, Zunino line)**
- [V] M. Hirsch, C. Mateos, A. Zunino, T. A. Majchrzak, T.-M. Grønli, H. Kaindl, "A Task Execution Scheme for Dew Computing with State-of-the-Art Smartphones," *Electronics* 10(16):2006, 2021. It evaluates job-scheduling heuristics using real battery traces and benchmarks, plus small clusters of mid- and low-end phones running TensorFlow object recognition — [TU Wien PDF](https://repositum.tuwien.at/bitstream/20.500.12708/138785/1/Hirsch-2021-Electronics-vor.pdf)
- [V] A companion paper, "A Simulation-based Performance Evaluation of Heuristics for Dew Computing" (2021), ranks devices by last-reported battery level combined with performance scores. Sanabria et al. (*Wireless Communications & Mobile Computing* 2021) cover hybrid battery- and mains-powered scheduling evaluated with the DewSim trace-based battery simulator. These are precedents for CROWDio's battery-aware ranking and trace-driven simulation — [Kristiania repo](https://kristiania.brage.unit.no/kristiania-xmlui/handle/11250/3005677); [CONICET record](https://repositoriosdigitales.mincyt.gob.ar/vufind/Record/CONICETDig_c4db7b492140765984d7acc931fc2d78)

### Inferences
- The most recent peer-reviewed MCC framework (Honeybee-Tx, Oct 2025) is still decentralised work stealing over static dependencies and does not model per-link bandwidth or cached artifacts. CROWDio's bandwidth-aware completion-time model with a "not-yet-cached args+code+model" term has no direct MCC precedent found. CWC (classic) is the closest centralised analogue.
- CROWDio's MCDM scheduler builds directly on the Pramanik 2021 line but moves it from offline ranking to an online dispatcher. This is the main novelty claim against Pramanik.

### Gaps
- Full text of Honeybee-Tx (MDPI returned 403), so its fault-tolerance and device-leave handling are unverified.
- No 2025–2026 MCC frameworks beyond Honeybee-Tx and CROWDio were found. A search for DroidCloud, an mCloud journal version and BOINC-on-Android performance papers (2019+) was not done for lack of budget.

## Q2. Resource selection and scheduling in MCC/edge: MCDM, energy, churn/availability, incentives

### Takeaway
The MCDM-for-MCC literature is dominated by Pramanik et al. (EDAS, ARAS, MABAC, COPRAS, MARCOS on static smartphone specs). Energy-aware smartphone scheduling uses battery traces (Hirsch/Mateos). Availability prediction relies on charging and diurnal patterns. None of these feeds live telemetry (RTT, temperature) into an online entropy-weighted MCDM dispatcher, which is CROWDio's approach.

### Cited Findings
- [V] P. K. D. Pramanik, S. Biswas, S. Pal, D. Marinković, P. Choudhury, "A Comparative Analysis of Multi-Criteria Decision-Making Methods for Resource Selection in Mobile Crowd Computing," *Symmetry* 13(9):1713, 2021, DOI 10.3390/sym13091713. Resource selection is posed as MCDM, and five methods (EDAS, ARAS, MABAC, COPRAS, MARCOS) are compared with runtime measured on a Windows laptop and an Android phone, plus sensitivity and rank-order analysis. A Mendeley dataset of smartphone parameters accompanies it (DOI 10.17632/m8r4zt246z.1). This is the direct precedent for CROWDio's EDAS/ARAS/MABAC set — [MDPI](https://www.mdpi.com/2073-8994/13/9/1713); [TU Berlin repo](https://depositonce.tu-berlin.de/items/7e2cad6a-12ee-438f-92d0-0ca1c6725a64/full); [dataset](https://data.mendeley.com/datasets/m8r4zt246z/1)
- [V] MCDM method originals (cited by CROWDio): EDAS — Keshavarz Ghorabaee et al., *Informatica* 26(3):435–451 (2015 in print; CROWDio lists 2017); ARAS — Zavadskas & Turskis, *Technological and Economic Development of Economy* 16(2):159–172, 2010; MABAC — Pamučar & Ćirović, *Expert Systems with Applications* 42(6):3016–3028, 2015; entropy weighting — Shannon 1948 — [CROWDio refs](https://arxiv.org/html/2604.19363). Note: the EDAS Informatica 26(3) paper is generally dated 2015; CROWDio's reference list says 2017, so the writer should check this.
- [V] R. Stan et al., "Evaluation of Task Scheduling Algorithms in Heterogeneous Computing Environments," *Sensors* 21(18):6035, 2021. Conventional policies degrade significantly on low-capacity edge nodes, which motivates capability-aware selection — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8433676/)
- [V] E. Saleh, C. Shastry, "A new approach for global task scheduling in volunteer computing systems," *Int. J. Information Technology*, 2022. It schedules globally on observed per-device performance and is a precedent for CROWDio's learned per-worker compute times — [CROWDio refs](https://arxiv.org/html/2604.19363)
- [V] Availability/churn: Javadi, Matawie, Anderson model volunteer host availability with Markovian arrival processes and report reasonable-to-excellent accuracy; Longo, Mateos, Zunino (Springer 2018) predict device energy availability hours ahead from charging and radio history; a Szeged measurement study models smartphone churn as alternating availability intervals with diurnal patterns — [Western Sydney](https://researchers.westernsydney.edu.au/en/publications/modeling-and-analysis-of-resources-availability-in-volunteer-comp/); [Springer chapter](https://springerlink.bibliotecabuap.elogim.com/chapter/10.1007/978-3-319-77028-4_47); [Szeged PDF](https://publicatio.bibl.u-szeged.hu/4420/1/all.pdf)

### Inferences
- CROWDio's entropy-weighted EDAS/ARAS/MABAC over live CPU/RAM/battery/RTT/temperature is an online extension of Pramanik's offline, spec-based comparison. Including RTT and thermal state as criteria appears new in the MCC MCDM literature, although no exhaustive search was done to prove the absence.
- CROWDio has no churn or availability predictor. It reacts via checkpoint/resume, which is a gap relative to the availability-modelling line.

### Gaps
- No 2023–2026 MCC paper on incentive-aware scheduling was retrieved (not searched for budget reasons).
- No 2024–2026 smartphone-churn prediction paper was found. Results were 2013–2018.

## Q3. Network/bandwidth-aware and data-locality scheduling

### Takeaway
CROWDio's completion-time estimate (RTT + uncached bytes/downlink + compute + result/uplink) with Sufferage batch mapping is a mobile-crowd instantiation of the classic MCT/Sufferage heterogeneous-mapping family plus WAN/data-locality schedulers (Corral, Iridium, Tetrium). Recent edge work (Cotter et al. 2025) uses active ping-based bandwidth probing with EWMA, which is very close to CROWDio's passive+active probing. No MCC framework was found that combines artifact-cache-aware transfer cost, shared-Wi-Fi max-min fair share and Sufferage.

### Cited Findings
- [V] NEW. J. Cotter, I. Castiñeiras, V. Cionca (Munster Tech. Univ.), "Accuracy vs Performance: An Abstraction model for deadline constrained offloading at the mobile-edge," arXiv:2510.01885, 2 Oct 2025. The controller runs an initial iperf3 baseline, then periodically has a random device send 10 × 1400-byte pings to peers, converts RTTs to bit/s, and smooths with EWMA (α=0.3). Probing is every 30 s in the main experiments, with 1.5–30 s tested. Testbed: 4 Raspberry Pi 2 boards on 802.11n running TFLite YOLOv2. The new scheduler (RAS) allocates in under 6 ms vs. 140–205 ms for the baseline (WPS), and pre-empts in under 100 ms vs. 250 ms or more. Frequent probing adds congestion, and 75%-duty background traffic cuts frame completion by 18%. Relation: a precedent for CROWDio's active probing and for the probe-overhead trade-off — [arXiv HTML](https://arxiv.org/html/2510.01885v1)
- [V] Zhang, Cao, Hao, Sufferage-based offloading across multiple edge servers choosing the server with the highest bandwidth efficiency, *KSII Trans. Internet and Information Systems* 16(11), Nov 2022. This is the closest found precedent for "Sufferage + bandwidth" at the edge — [KSII TIIS](https://itiis.org/journals/tiis/digital-library/38002)
- [V] ENTS: An Edge-native Task Scheduling System for Collaborative Edge Computing, arXiv:2210.07842 (2022). It jointly schedules compute and network resources across collaborating edge nodes — [arXiv](https://arxiv.org/pdf/2210.07842)
- [V] NEW. Lou et al., *IEEE Trans. Mobile Computing* vol. 23 (2024), on startup-aware dependent task scheduling on heterogeneous edge nodes, with bandwidth as a hard constraint during task (image/code) startup, using list scheduling. This is conceptually close to CROWDio's "code+model not yet cached" term; details come from a search snippet only — [search result context](https://arxiv.org/pdf/2210.07842)
- [K] Classic heuristics: M. Maheswaran, S. Ali, H. J. Siegel, D. Hensgen, R. F. Freund, "Dynamic Mapping of a Class of Independent Tasks onto Heterogeneous Computing Systems," *JPDC* 59(2):107–131, 1999, which introduced Sufferage and compared it with Min-Min/MCT. T. D. Braun et al., "A Comparison of Eleven Static Heuristics for Mapping a Class of Independent Tasks onto Heterogeneous Distributed Computing Systems," *JPDC* 61(6):810–837, 2001, found Min-Min/GA strong and Max-Min weaker on its workloads. H. Topcuoglu, S. Hariri, M.-Y. Wu, "Performance-Effective and Low-Complexity Task Scheduling for Heterogeneous Computing" (HEFT/CPOP), *IEEE TPDS* 13(3):260–274, 2002.
- [K] Data-locality/WAN analytics: V. Jalaparti, P. Bodik, I. Menache, S. Rao, K. Makarychev, M. Caesar, "Network-Aware Scheduling for Data-Parallel Jobs: Plan When You Can" (Corral), ACM SIGCOMM 2015. Q. Pu, G. Ananthanarayanan, P. Bodik, S. Kandula, A. Akella, P. Bahl, I. Stoica, "Low Latency Geo-distributed Data Analytics" (Iridium), SIGCOMM 2015. C.-C. Hung, G. Ananthanarayanan, L. Golubchik, M. Yu, M. Zhang, "Wide-area Analytics with Multiple Resources" (Tetrium), EuroSys 2018. K. Ousterhout, P. Wendell, M. Zaharia, I. Stoica, "Sparrow: Distributed, Low Latency Scheduling," SOSP 2013. M. Zaharia et al., "Delay Scheduling," EuroSys 2010.
- [V] Ray (Moritz et al., OSDI 2018), cited by CROWDio, includes object-locality-aware scheduling. It is a design-philosophy precedent but is not built for mobile connectivity — [CROWDio refs](https://arxiv.org/html/2604.19363)

### Inferences
- CROWDio's "uncached artifact" term is effectively data/code locality (as in Delay Scheduling, Sparrow and Ray) applied to phones, where the downlink is the bottleneck. Max-min fair-share on a shared Wi-Fi AP resembles Tetrium/Corral's joint network and compute planning, scaled down to one collision domain. This framing is useful for positioning.
- Cotter et al. show that probe frequency trades accuracy against congestion. CROWDio's passive (piggy-backed on task transfers) plus active probing is a defensible answer, and should be evaluated against probe-only baselines.

### Gaps
- No 2023–2026 paper evaluating Sufferage specifically on smartphones or Wi-Fi edge clusters was found.
- 802.11 shared-medium capacity modelling papers (e.g., airtime fairness) were not searched.

## Q4. Fault tolerance in mobile/edge distributed computing

### Takeaway
Delta/incremental checkpointing is well established for container migration at the edge (about 1–4% of full checkpoint size transferred) and for DL training (CheckFreq, Check-N-Run). Smartphone-specific checkpointing literature is mostly old (2010–2014) and trigger-based (battery low, cell exit). Framework-aware (PyTorch/TF/NumPy) delta checkpoints that resume on a different phone after a WebSocket disconnect were not found in MCC work.

### Cited Findings
- [V] Ngo et al., "Coordinated Container Migration and Base Station Handover in Mobile Edge Computing," IEEE GLOBECOM 2020, arXiv:2009.05682. Delta checkpointing cut migrated state to about 4% and 0.9% of the pre-migration checkpoint for two services, with 30–40% lower downtime and 13–22% lower end-to-end delay. It uses Docker on an edge testbed, not phones. It is a precedent for CROWDio's base+delta scheme — [arXiv](https://arxiv.org/pdf/2009.05682)
- [V] Lee, Chin, Gil (2010), mobile-grid task migration with user-defined checkpoints triggered when a device is about to leave its cell or run out of battery. It reports shorter job completion. It is a precedent for battery-triggered checkpointing — [Springer Professional](https://www.springerprofessional.de/en/efficient-resource-management-and-task-migration-in-mobile-grid-/3388244)
- [V] Tuli & Kumar (2011) survey of mobile checkpointing. It names low wireless bandwidth, lack of stable storage and battery as core constraints, which supports CROWDio storing checkpoints server-side (file-backed payload store) — [arXiv 1108.6195](https://arxiv.org/pdf/1108.6195)
- [V] Honeybee (TCC 2019) tolerates devices joining and leaving via work stealing (re-stealing lost work) rather than checkpointing — [Deakin DRO](https://dro.deakin.edu.au/articles/journal_contribution/Computing_with_nearby_mobile_devices_a_work_sharing_algorithm_for_mobile_edge-clouds/20870146)
- [K] DL checkpointing: J. Mohan, A. Phanishayee, V. Chidambaram, "CheckFreq: Frequent, Fine-Grained DNN Checkpointing," USENIX FAST 2021. A. Eisenman et al., "Check-N-Run: A Checkpointing System for Training Deep Learning Recommendation Models," USENIX NSDI 2022, which uses differential/incremental checkpoints plus quantisation. These are precedents for CROWDio's framework-aware deltas.
- [K] Stragglers and replication: G. Ananthanarayanan et al., "Reining in the Outliers in Map-Reduce Clusters using Mantri," OSDI 2010; G. Ananthanarayanan et al., "Effective Straggler Mitigation: Attack of the Clones" (Dolly), NSDI 2013; M. Zaharia et al., "Improving MapReduce Performance in Heterogeneous Environments" (LATE), OSDI 2008.

### Inferences
- CROWDio's CROWDio-paper result (2–3 s bounded overhead per task) can be positioned against Ngo et al.'s delta-size savings. The writer should note that CROWDio's novelty is applying framework-aware deltas on phones with orchestrator-side storage, not delta checkpointing per se.
- CROWDio's starvation guard and learned per-worker times overlap conceptually with LATE/Mantri speculative execution. Adding speculative replication would close a gap against BOINC-style redundancy.

### Gaps
- No 2023–2026 smartphone-specific checkpointing paper was found in one targeted search. This should be stated as a literature gap, with the caveat that the search was limited.

## Q5. DAG / pipeline workflow scheduling on heterogeneous edge devices

### Takeaway
DAG scheduling at the edge is active (2024–2026), mostly DRL- or list-scheduling-based offloading to edge servers. Work targeting personally owned, sporadically available devices (IBDASH 2022; arXiv 2409.10839, 2024) is closest to CROWDio. Honeybee-Tx is the only MCC framework with dependencies, and it handles them statically.

### Cited Findings
- [V] NEW. "Dynamic DAG-Application Scheduling for Multi-Tier Edge Computing in Heterogeneous Networks," arXiv:2409.10839, 2024. It schedules DAG apps across personally owned (sporadically available) and commercial edge devices with static partitioning, using each task node as the unit — [arXiv](https://arxiv.org/pdf/2409.10839)
- [V] IBDASH (Purdue DCSL, IEEE SRDS 2022) orchestrates DAG applications over mixed commercial and personal edge devices, accounting for memory and network bandwidth and modelling device failure/interference probability — [PDF](https://engineering.purdue.edu/dcsl/publications/papers/2022/ibdash-srds22.pdf)
- [V] NEW. "Partition-Aware Scheduling for Mobile Heterogeneous Inference Co-Execution," arXiv:2609.14213 (2026). It argues that HEFT/PEFT treat DAG nodes as indivisible and splits operators across CPU/GPU. It is relevant to CROWDio's mobile-first DNN placement — [arXiv](https://arxiv.org/pdf/2609.14213)
- [V] NEW. Multi-edge DRL online scheduling of dependency-constrained DAG tasks (*Tsinghua Science and Technology*, 2026), reporting up to 63.9% makespan reduction vs. baselines — [SciOpen PDF](https://www.sciopen.com/local/article_pdf/10.26599/TST.2026.9010046.pdf)
- [V] NEW. NCSAC: deterministic-latency DAG scheduling for mobile edge networks via MDP and dual GCNs (*J. China Univ. Posts & Telecom.*, 2026) — [JCUPT](https://journal.hep.com.cn/jcupt/EN/10.19682/j.cnki.1005-8885.2026.1006)
- [V] Honeybee-Tx: static dependency-aware stealing with staged result transfers, 42% less network traffic than dependency-oblivious approaches — [IDEAS](https://ideas.repec.org/a/gam/jftint/v17y2025i10p446-d1761063.html)
- [K] Serendipity (MobiHoc 2012) and HEFT (TPDS 2002), as in Q1/Q3.

### Inferences
- CROWDio's distinction between streaming 1:1 and barrier edges with upstream result injection corresponds to pipelined vs. bulk-synchronous dependencies. Most edge DAG papers assume a full DAG known up front with data sizes per edge. CROWDio can claim practical runtime support (actual execution on phones), whereas most cited DAG work is simulation-only (DRL/MDP papers).

### Gaps
- No dedicated 2024–2026 survey of DAG offloading on heterogeneous edge devices was found in one search.

## Q6. Evaluation methodology: simulators and device-fleet emulation

### Takeaway
CloudSim-derived simulators (iFogSim/iFogSim2, EdgeCloudSim) and Python YAFS dominate. Smartphone-specific trace-based simulation exists (DewSim, battery traces). None of these simulates a WebSocket orchestrator with real telemetry, which justifies CROWDio's custom trace-driven simulator and virtual fleet, but the writer should cite them and ideally calibrate against one.

### Cited Findings
- [V] YAFS (Lera, Guerrero, Juiz), "YAFS: A simulator for IoT scenarios in fog computing," arXiv:1902.01091 (IEEE Access 2019). It is Python/SimPy based, unlike Java CloudSim extensions — [ar5iv](https://ar5iv.labs.arxiv.org/html/1902.01091)
- [V] NEW. arXiv:2604.23122 (2026), "Source-Code Analysis of iFogSim for Simulating Distributed IoT Architectures: Coverage, Challenges, and Enhancements." It compares iFogSim2, YAFS and EdgeCloudSim. EdgeCloudSim is described as a Java/CloudSim simulator specialised for MEC with mobility and cellular network models, and native fault-tolerance coverage is weak — [arXiv](https://arxiv.org/pdf/2604.23122)
- [V] NEW. "Navigating the Edge-Cloud Continuum: A State-of-Practice Survey," arXiv:2506.02003 (2025). It notes EdgeCloudSim's mobility and link characterisation — [arXiv](https://arxiv.org/pdf/2506.02003)
- [V] An iFogSim2 vs. FogNetSim++ comparison found iFogSim2 used up to 80% less energy with longer execution times. A 2023 MJPAS overview notes most Java simulators extend CloudSim — [search summary](https://www.bohrium.com/paper-details/a-comparative-analysis-of-simulators-for-the-cloud-to-fog-continuum/812709037253591040-2560); [MJPAS](https://mjpas.uomustansiriyah.edu.iq/index.php/mjpas/article/download/55/48/57)
- [V] DewSim, extended by Sanabria et al. (2021), models battery behaviour from real-device traces for smartphone-cluster scheduling. It is the closest precedent to CROWDio's virtual device fleet — [CONICET record](https://repositoriosdigitales.mincyt.gob.ar/vufind/Record/CONICETDig_c4db7b492140765984d7acc931fc2d78)
- [K] R. N. Calheiros et al., "CloudSim," *Software: Practice & Experience* 41(1), 2011. H. Gupta et al., "iFogSim," *SPE* 47(9), 2017. R. Mahmud et al., "iFogSim2," *J. Systems and Software* 190, 2022. C. Sonmez, A. Ozgovde, C. Ersoy, "EdgeCloudSim," *Trans. Emerging Telecommunications Technologies* 29(11), 2018. H. Casanova et al., "Versatile, Scalable, and Accurate Simulation of Distributed Applications and Platforms" (SimGrid), *JPDC* 74(10), 2014. SimGrid has validated flow-level network models, which is relevant if CROWDio's Wi-Fi fair-share model needs validation.

### Inferences
- The research gap CROWDio fills is a simulator that replays the real scheduler code paths (same Python scheduler objects) against virtual devices with telemetry, RTT and bandwidth traces. Generic simulators require reimplementing the policy in Java. This "same code in sim and production" argument is a strong methodological claim and should be stated explicitly.

### Gaps
- No public smartphone fleet trace dataset (CPU, bandwidth, availability) from 2023–2026 was located. Mobile churn traces found were 2013-era.

---
### Consolidated research gaps CROWDio addresses (synthesis, inference-level)
1. Centralised, online MCDM dispatch over live telemetry (RTT, temperature, battery). Prior MCDM-MCC work (Pramanik 2021) ranks offline from specs.
2. Network-cost-aware completion-time estimation including artifact/model caching, Sufferage batch mapping and shared-Wi-Fi fair share. MCC frameworks (Honeybee/-T/-Tx, Kiridana 2024) use work stealing without explicit link models. The closest analogues are CWC (classic) and Cotter et al. 2025 (Raspberry Pi, not phones).
3. Pipeline DAGs with streaming vs. barrier edges and runtime result injection, executed on real phones. Honeybee-Tx is static. Edge DAG papers are mostly simulation/DRL.
4. Framework-aware delta checkpointing with cross-device resume. Delta checkpointing exists for containers (Ngo 2020) and DL training (CheckFreq, Check-N-Run), but not for MCC.
5. Evaluation: a trace-driven simulator plus a virtual fleet sharing the production scheduler code, vs. CloudSim-family reimplementation.
Remaining gaps for CROWDio (from its own preprint and this survey): scale beyond 6 devices, energy measurement, churn/availability prediction, incentives, speculative replication.
