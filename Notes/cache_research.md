# Caching and Cache Optimisation for Diffusion Language Models

## Errata & Corrections (2026-10-18)

> Citation-only audit (abstract pages fetched from arxiv.org on 2026-10-18). Numbers and analysis in this note are otherwise unchanged.

- **E-1 — TEAM is not a coreset method; one figure unverified.** arXiv:2602.08404 is "TEAM: Temporal-Spatial Consistency Guided Expert Activation for MoE Diffusion Language Model Acceleration": it exploits routing consistency across steps and neighbouring positions (conservative expert activation for decoded/masked tokens + speculative exploration). Its abstract confirms "up to 2.2×" but gives **no** expert-reduction percentage, so "35–39% fewer experts" is unverified against the abstract. Only **DES** (arXiv:2602.00879, "Dynamic Expert Sharing: Decoupling Memory from Parallelism in Mixture-of-Experts Diffusion LLMs") matches the "shared coreset" mechanism in Addendum A. `[Sourced]`
- **E-2 — "Quantized Keys Steal Attention" (arXiv:2605.26266) is a VIDEO-diffusion paper** ("…Bias Correction for KV-Cache Compression in Video Diffusion"; chunk-wise autoregressive video models MAGI-1, SkyReels-V2, HY-WorldPlay). It supports the softmax mechanism only by analogy and **must not be cited as dLLM evidence**. `[Sourced]`
- **E-3 — Confirmed as cited** (no change): arXiv:2510.14973 "Attention Is All You Need for KV Cache in Diffusion LLMs" (Elastic-Cache); arXiv:2605.08134 DARE ("DARE: Diffusion Language Model Activation Reuse for Efficient Inference"); arXiv:2603.08026 is **DyLLM** ("…Saliency-based Token Selection and Partial Attention"), not DARE — Addendum A's pairing is correct, but downstream docs had it swapped; arXiv:2610.06996 "Mask-Guided KV Cache Eviction in Block Diffusion Language Models" (MaskAhead / Q-MaskAhead: eviction + selection, low-bit variant — not a step-0 asymmetric-KV method); arXiv:2602.14209 MAGE (step-0 all-`[MASK]` top-k prefix-KV index reuse); arXiv:2608.01784 REFLEX (~15% average expert-compute reduction); arXiv:2602.05305 FlashBlock; arXiv:2402.02750 KIVI. `[Sourced]`

## Scope and answer

As of **17 October 2026**, this search found **41 papers whose main contribution directly concerns caching, cached-state reuse, cache refresh/eviction/compression, cache I/O/offloading, or an architecture explicitly designed to make diffusion-language-model inference cacheable**. It also found **12 closely adjacent papers** where caching is an important enabling component, cache-preserving constraint, or reused attention pattern, but not the principal contribution.

“Every paper” is necessarily bounded here to publicly discoverable academic work available by the cutoff date. The inventory covers arXiv, OpenReview, ACL/Findings, ICLR, ICML, NeurIPS, AAAI and relevant systems venues; it excludes image/video-only Diffusion Transformer caching, ordinary autoregressive-LLM KV-cache papers, papers that merely compare against cached dLLMs, and sampling-only acceleration with no substantive cache mechanism. Recent surveys describe the same central distinction: diffusion-aware caching must manage stale state across refinement steps, unlike append-only autoregressive caching, and creates an explicit memory–speed trade-off.[^1]

### Regime labels

- **Full-sequence**: a non-block masked/bidirectional DLM, even if the inference wrapper introduces a sliding window or temporary blocks.
- **Block diffusion**: autoregressive or causal progression across blocks, with diffusion inside the active block.
- **Causal/hybrid**: the architecture changes attention or state-space structure specifically to regain exact/native cacheability.
- **Both/general**: evaluated across, or designed independently of, block and full-sequence backbones.

This distinction matters because block diffusion makes completed-prefix KV states exact and reusable by construction, whereas full-sequence bidirectional DLMs usually require approximate reuse plus a refresh policy. The original Block Diffusion paper explicitly introduced arbitrary-length generation with KV caching and parallel within-block sampling.[^2]

## Direct papers: 2025

| Date    | Paper                                                                                                                                                    | Regime                                 | Cache contribution                                                                                                                                                 |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 2025-03 | [Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models](https://arxiv.org/abs/2503.09573)                                  | Block diffusion                        | Foundational block-causal formulation: exact reuse of completed-prefix KV while denoising the current block in parallel.[^2]                                       |
| 2025-05 | [dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching](https://arxiv.org/abs/2506.06295)                                       | Full-sequence                          | Long-interval prompt caching plus feature-similarity-guided partial response updates; training-free.[^3]                                                           |
| 2025-05 | [dKV-Cache: The Cache for Diffusion Language Models](https://arxiv.org/abs/2505.15781)                                                                   | Full-sequence                          | Delays insertion of newly decoded tokens until their representations stabilise; Decode and Greedy variants.[^4]                                                    |
| 2025-05 | [FlashDLM: Accelerating Diffusion Language Model Inference via Efficient KV Caching and Guided Diffusion](https://arxiv.org/abs/2505.21467)              | Full-sequence + block wrapper          | **FreeCache** reuses stable KV projections of clean tokens; also explores block-wise caching and AR-guided unmasking.[^5]                                          |
| 2025-05 | [Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding](https://arxiv.org/abs/2505.22618)                    | Full-sequence via block-wise inference | Approximate DualCache for tokens outside the active block, refreshed at block boundaries; coupled with confidence-aware parallel decoding.[^6]                     |
| 2025-05 | [Accelerating Diffusion LLMs via Adaptive Parallel Decoding](https://arxiv.org/abs/2506.00413)                                                           | Full-sequence                          | Adaptive parallel decoding with an auxiliary AR model, additionally enabling KV caching and limiting masked input size.[^7]                                        |
| 2025-06 | [Esoteric Language Models: A Family of Any-Order Diffusion LLMs](https://arxiv.org/abs/2506.01928)                                                       | Causal/hybrid                          | Recasts MDMs through any-order autoregression and causal attention, enabling native KV caching while retaining parallel generation.[^8]                            |
| 2025-08 | [Sparse-dLLM: Accelerating Diffusion LLMs with Dynamic Cache Eviction](https://arxiv.org/abs/2508.02558)                                                 | Full-sequence via block-wise inference | Bidirectional dynamic cache eviction plus one-step delayed cache updates; reduces cache entries and memory.[^9]                                                    |
| 2025-09 | [d²Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching](https://arxiv.org/abs/2509.23094)                                                 | Full-sequence                          | Two-stage token selection: certainty-prior selection for masked tokens and attention-influence selection for prompt/decoded tokens.[^10]                           |
| 2025-09 | [Sequential Diffusion Language Models](https://arxiv.org/abs/2509.24007)                                                                                 | Causal/hybrid                          | Next-sequence prediction that remains KV-cache compatible while varying generation span; bridges token-, block-, and sequence-level generation.[^11]               |
| 2025-09 | [Fast-dLLM v2: Efficient Block-Diffusion LLM](https://arxiv.org/abs/2509.26328)                                                                          | Block diffusion                        | Hierarchical block-level and sub-block KV caches for converted AR backbones.[^12]                                                                                  |
| 2025-10 | [dInfer: An Efficient Inference Framework for Diffusion Language Models](https://arxiv.org/abs/2510.08666)                                               | General/system                         | Modular inference runtime with a dedicated KV-cache manager and cache-aware system optimisations.                                                                  |
| 2025-10 | [Mask Tokens as Prophet: Fine-Grained Cache Eviction for Efficient dLLM Inference](https://arxiv.org/abs/2510.09309)                                     | Full-sequence/long context             | **MaskKV** uses mask-query attention to evict prompt KV per head and adapts budgets across layers and prompt-preferring heads.[^13]                                |
| 2025-10 | [Attention Is All You Need for KV Cache in Diffusion LLMs](https://arxiv.org/abs/2510.14973)                                                             | Full-sequence via sliding/block window | **Elastic-Cache** adaptively decides when to refresh from attention drift and where to refresh from a learned layer boundary; also caches distant masks.[^14][^15] |
| 2025-11 | [Orchestrating Dual-Boundaries: An Arithmetic Intensity Inspired Acceleration Framework for Diffusion Language Models](https://arxiv.org/abs/2511.21759) | Full-sequence via block-wise inference | Optimises repeated cache-refresh prefills and shares decoded-token KV across speculative branches.[^16]                                                            |
| 2025-12 | [WeDLM: Reconciling Diffusion Language Models with Standard Causal Attention for Fast Inference](https://arxiv.org/abs/2512.22737)                       | Causal/hybrid                          | Topological reordering under causal attention makes parallel mask recovery natively prefix-cache compatible.[^17]                                                  |

## Direct papers: 2026

| Date    | Paper                                                                                                                                                 | Regime                                 | Cache contribution                                                                                                                                                  |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-01 | [Window-Diffusion: Accelerating Diffusion Language Model Inference with Windowed Token Pruning and Caching](https://arxiv.org/abs/2601.20332)         | Full-sequence                          | Sliding local window with online active tokens, periodically refreshed cached buffer tokens, and pruned far-field masks.[^18]                                       |
| 2026-01 | [SPA-Cache: Singular Proxies for Adaptive Caching in Diffusion Language Models](https://arxiv.org/abs/2602.02544)                                     | Full-sequence                          | Low-dimensional singular proxy for update-critical token identification plus layer-adaptive update-budget allocation.[^19]                                          |
| 2026-02 | [FlashBlock: Attention Caching for Efficient Long-Context Block Diffusion](https://arxiv.org/abs/2602.05305)                                          | Block diffusion                        | Caches stable block-external attention outputs and normalisers across denoising steps while recomputing block-internal attention.[^20][^21]                         |
| 2026-02 | [Stopping Computation for Converged Tokens in Masked Diffusion-LM Decoding](https://arxiv.org/abs/2602.06412)                                         | Full-sequence                          | **SureLock** locks stabilised positions, skips their query/FFN computation, and retains their cached KV for unlocked tokens.[^22]                                   |
| 2026-02 | [MAGE: All-[MASK] Block Already Knows Where to Look in Block Diffusion LLM](https://arxiv.org/abs/2602.14209)                                         | Block diffusion                        | Reuses the first-step all-mask block’s top-k prefix-KV index set for all remaining denoising steps.[^23]                                                            |
| 2026-03 | [EntropyCache: Decoded Token Entropy Guided KV Caching for Diffusion Language Models](https://arxiv.org/abs/2603.18489)                               | Full-sequence                          | Constant-cost decoded-token entropy trigger for reuse versus full refresh; recomputes the k most recently decoded tokens.[^24][^25]                                 |
| 2026-05 | [DARE: Diffusion Language Model Activation Reuse for Efficient Inference](https://arxiv.org/abs/2605.08134)                                           | Full-sequence                          | DARE-KV reuses key/value activations; DARE-O additionally reuses output activations, using query change to predict redundancy.[^26]                                 |
| 2026-05 | [Polestar: Drift-Aware Cache Calibration and Token Commitment for Efficient Inference of Diffusion LLMs](https://arxiv.org/abs/2607.14107)            | Full-sequence via block-wise inference | Drift-aware sparse layer-wise KV refresh; clustered/quantised hidden states and CPU offload reduce monitoring overhead.[^27]                                        |
| 2026-05 | [Enabling KV Caching of Shared Prefix for Diffusion Language Models](https://arxiv.org/abs/2606.07571)                                                | Full-sequence/cross-request serving    | **BiCache** profiles a safe layer depth based on shared-prefix ratio, reusing shared-prefix KV across requests only in suitable shallow layers.[^28]                |
| 2026-05 | [Dynamic-dLLM: Dynamic Cache-Budget and Adaptive Parallel Decoding for Training-Free Acceleration of Diffusion LLM](https://arxiv.org/abs/2606.26120) | Full-sequence                          | Dynamically allocates per-layer cache-update budgets from token/layer dynamics and couples them with adaptive commitment thresholds.[^29]                           |
| 2026-05 | [WaveFilter: Enhancing the Long-Context Capability of Diffusion LLMs via Wavelet-Guided KV Cache Filtering](https://arxiv.org/abs/2606.00724)         | Full-sequence/long context             | Wavelet decomposition identifies critical long-context tokens from which a sparse KV cache is built.                                                                |
| 2026-06 | [Prefilling-dLLM: Predictive Prefilling for Long-Context Inference in Diffusion Language Models](https://arxiv.org/abs/2606.10537)                    | Full-sequence/long context             | Computes chunked prefix KV once, retrieves top-k chunks, compresses within chunks, and reuses static prefix KV across all denoising steps.[^30]                     |
| 2026-06 | [HERALD: High-Throughput Block Diffusion LLM Serving via CPU-GPU Cooperative KV Cache Retrieval](https://arxiv.org/abs/2606.21633)                    | Block diffusion/system                 | CPU–GPU KV offloading tailored to the block pattern: select once per block and transfer the reused sparse subset once over PCIe.[^31]                               |
| 2026-06 | [Affix Cache for Diffusion Large Language Models](https://arxiv.org/abs/2608.26140)                                                                   | Full-sequence/cross-request serving    | **ACache** shares prefix, infix, or suffix cache objects across requests and recomputes only request-sensitive anchor tokens.[^32]                                  |
| 2026-07 | [Sangam: Efficiently Serving Diffusion LLMs with the AR Stack](https://arxiv.org/abs/2607.04206)                                                      | General/system                         | Serving architecture and deficit-budget scheduler for cached dLLMs’ repeated, indivisible refresh-prefill/decode cycle.[^33]                                        |
| 2026-07 | [LaCache: Exact Caching and Precision-Adaptive Inference for Diffusion Large Language Models](https://arxiv.org/abs/2607.16339)                       | Block/semi-autoregressive              | Lossless memoisation of embeddings, pre-attention RoPE states, and FlashAttention online-softmax statistics; paired with adaptive FP8 FFNs.[^34][^35]               |
| 2026-07 | [GroupKV: Hierarchical KV Cache Management for Long-Context Diffusion LLM Inference](https://arxiv.org/abs/2609.17573)                                | Block diffusion/system                 | Group-to-token coarse-to-fine retrieval, predictive cross-layer prefetch, CPU offload, streaming prefill, and staleness repair.[^36]                                |
| 2026-08 | [Archer: Adaptive Reuse of Cached Hidden States for Efficient Rollback in Diffusion Language Models](https://arxiv.org/abs/2608.08086)                | Full-sequence/revocable decoding       | Reuses cached hidden states during rollback/revision rather than rebuilding the entire state.                                                                       |
| 2026-09 | [Fixed State, Long Reach: What a Constant-Size Cache Buys Block Diffusion at Scale](https://arxiv.org/abs/2609.11998)                                 | Block diffusion/state-space            | Trains attention, Mamba, and hybrid denoisers under a common block-causal objective; Mamba compresses finalised blocks into a constant-size exact state cache.[^37] |
| 2026-09 | [Flash-dLLM: IO-Aware KV Caching and Parallel Decoding for Fast, Memory-Efficient Diffusion LLMs](https://arxiv.org/abs/2609.26796)                   | Full-sequence via sliding window       | Fused QKV projection, RoPE and cache writes; scheduled variable-length FlashAttention; selective influential-token updates and cache-driven verification.[^38]      |
| 2026-09 | [Time-Anchored Diffusion Language Models: Latent-Space Caching for Fast Generation](https://arxiv.org/abs/2609.37924)                                 | Full-sequence/model-level              | Trains cacheable semantic anchor representations; evaluates an expensive anchor network periodically and fuses its stale latent with current state.[^39]            |
| 2026-10 | [Mask-Guided KV Cache Eviction in Block Diffusion Language Models](https://arxiv.org/abs/2610.06996)                                                  | Block diffusion                        | **MaskAhead** uses current masked-block queries for selection and future masked-block probes for eviction; Q-MaskAhead operates on low-bit KV.[^40]                 |

## Cache-adjacent papers

These are worth tracking because they either preserve/cache state as a design constraint, reuse an attention pattern in a cache-like way, compress cross-block state, or are explicitly built on cached dLLM inference. They should not be conflated with papers whose principal novelty is the cache policy itself.

| Date    | Paper                                                                                                                                                 | Regime                    | Why adjacent rather than core                                                                                                                |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 2025-06 | [Accelerating Diffusion Large Language Models with SlowFast Sampling](https://arxiv.org/abs/2506.10848)                                               | Full-sequence             | Sampling schedule is the contribution; it integrates with dLLM-Cache rather than proposing a new cache.[^41]                                 |
| 2025-09 | [Learning to Parallel: Accelerating Diffusion Large Language Models via Learnable Parallel Decoding](https://arxiv.org/abs/2509.25188)                | Full-sequence             | Learnable commitment policy; reports an additional configuration combined with KV cache.[^42]                                                |
| 2026-01 | [d³LLM: Ultra-Fast Diffusion LLM using Pseudo-Trajectory Distillation](https://arxiv.org/abs/2601.07568)                                              | Full-sequence/multi-block | Training method plus entropy multi-block decoding includes a KV-refresh mechanism, but caching is secondary.[^43]                            |
| 2026-02 | [Focus-dLLM: Accelerating Long-Context Diffusion LLM Inference via Confidence-Guided Context Focusing](https://arxiv.org/abs/2602.02159)              | Full-sequence             | Sparse attention, not KV caching; reuses identified sink locations across layers.[^44]                                                       |
| 2026-04 | [LoSA: Locality Aware Sparse Attention for Block-Wise Diffusion Language Models](https://arxiv.org/abs/2604.12056)                                    | Block diffusion           | Sparse prefix attention reduces KV reads but is not itself a cache-refresh or eviction framework.                                            |
| 2026-05 | [PulseCol: Periodically Refreshed Column-Sparse Attention for Accelerating Diffusion Language Models](https://arxiv.org/abs/2605.20813)               | Full-sequence             | Caches/reuses sparsity patterns across denoising iterations rather than KV or hidden states.                                                 |
| 2026-05 | [Orthrus: Memory-Efficient Parallel Token Generation via Dual-View Diffusion](https://arxiv.org/abs/2605.12825)                                       | Causal/hybrid             | Frozen AR prefill builds one exact KV cache shared by AR and diffusion views; architecture/consensus is primary.[^45]                        |
| 2026-06 | [Bifocal Diffusion Language Models: Asymmetric Bidirectional Context for Parallel Generation](https://arxiv.org/abs/2606.27732)                       | Causal/hybrid             | Causal Transformer KV cache plus reverse Mamba right-context sidecar; cacheability is architectural.[^46]                                    |
| 2026-06 | [SW-SpeedDLM: Sliding Window Speculative Decoding for Diffusion Language Models Under Long Context Constraints](https://doi.org/10.3390/math14122137) | Full-sequence/windowed    | Uses cross-segment KV compression into summary tokens, alongside windowed denoising and speculation.[^47]                                    |
| 2026-07 | [FlowBlock: Wavefront-Parallel Decoding for Self-Correcting Diffusion Language Models](https://arxiv.org/abs/2607.17652)                              | Block diffusion           | Main contribution is inter-block wavefront scheduling, constrained to preserve frozen-prefix KV reuse.[^48]                                  |
| 2026-08 | [Retrofitting Linear Attention into Diffusion Language Models](https://arxiv.org/abs/2608.06628)                                                      | Block diffusion/hybrid    | Replaces softmax attention over previous blocks with constant-state linear attention; cache pressure is motivation, not a cache policy.[^49] |
| 2026-09 | [Trajectory-Level Speculative Decoding for Diffusion Language Models](https://arxiv.org/abs/2608.27514)                                               | Block diffusion           | Builds trajectory speculation on Fast-dLLM’s dual-cache infrastructure; speculation is the novel part.[^50]                                  |

## Technical taxonomy

| Family                         | Representative papers                                                               | What is reused                                                                              | Main risk/bottleneck                                             |
| ------------------------------ | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| Fixed/delayed KV reuse         | dKV-Cache, Fast-dLLM, FlashDLM                                                      | Token or outside-block K/V across steps                                                     | Staleness after context changes; refresh-prefill cost            |
| Adaptive refresh               | dLLM-Cache, d²Cache, Elastic-Cache, SPA-Cache, EntropyCache, Dynamic-dLLM, Polestar | Selected token/layer K/V or hidden states                                                   | Proxy cost and whether the proxy truly predicts drift            |
| Eviction/compression           | Sparse-dLLM, MaskKV, WaveFilter, MaskAhead                                          | A selected subset or quantised subset of historical KV                                      | Lost long-range evidence and task-dependent budgets              |
| Attention/activation reuse     | FlashBlock, DARE, LaCache, PulseCol                                                 | Attention outputs/statistics, output activations, operator intermediates, sparsity patterns | Memory bandwidth and kernel integration                          |
| Long-context retrieval/offload | Prefilling-dLLM, HERALD, GroupKV                                                    | Chunk/group KV retrieved from GPU or CPU tiers                                              | PCIe transfer, prefetch errors, stale prefetched entries         |
| Cross-request reuse            | BiCache, ACache                                                                     | Shared prefix or arbitrary shared affix KV                                                  | Request-specific bidirectional contamination                     |
| Architecture-native caching    | Block Diffusion, Eso-LM, SDLM, Fast-dLLM v2, WeDLM, Bifocal, Fixed State            | Exact prefix KV or fixed-size recurrent state                                               | Requires retraining/conversion and changes modelling assumptions |
| Learned latent caching         | Time-Anchored DLM                                                                   | Deep semantic anchor latent across timesteps                                                | Training complexity and stale-latent correction                  |
| Cache systems/kernels          | dInfer, Sangam, Flash-dLLM                                                          | Cache storage, update pipeline, fused writes, refresh scheduling                            | I/O can replace FLOPs as the dominant bottleneck                 |

The field has therefore moved through three stages: first making approximate KV reuse possible, then learning or measuring _when/where_ state is stale, and finally treating cache I/O, offloading, cross-request sharing, and serving schedules as first-class systems problems. Flash-dLLM explicitly reports that naïve caching can become memory-I/O bound, while GroupKV and HERALD target CPU–GPU movement rather than only reducing arithmetic.[^31][^36][^38]

## Block vs non-block

### Block diffusion

Block models have the cleaner cache semantics: finalised preceding blocks are causally frozen, so their KV is exact. Current work therefore concentrates on the remaining scaling costs—repeated reads of a growing prefix, sparse subset selection, offloading, attention-output reuse, sub-block caching, and cache eviction. FlashBlock, MAGE, HERALD, GroupKV, Fixed State, and MaskAhead form the most direct block-specific optimisation line.[^20][^36][^23][^40][^31]

The strongest architectural alternative to a linearly growing attention cache is a recurrent/state-space summary. _Fixed State, Long Reach_ studies this directly under matched block-diffusion training, while the linear-attention retrofit and Bifocal designs pursue related constant-state or asymmetric-context compromises.[^37][^49][^46]

### Non-block diffusion

For full bidirectional DLMs, changing one token can alter representations everywhere, so most cache methods are approximate. The key research question is not merely “what to cache?” but “which token and layer states remain accurate enough, and when must they be refreshed?” dLLM-Cache, dKV-Cache, d²Cache, Elastic-Cache, SPA-Cache, EntropyCache, Dynamic-dLLM and Polestar differ primarily in the proxy used to answer that question: value similarity, delayed stability, certainty/attention, attention drift, low-rank singular projection, entropy, dynamic token budgets, or KL-style representation drift.[^14][^3][^27][^24][^4][^10][^19][^29]

A second line changes the architecture so caching becomes exact or native: causal attention with reordered observed tokens, any-order autoregressive factorisation, block-causal conversion, or a recurrent right-context pathway. These methods may yield cleaner systems behaviour, but they are not drop-in optimisations for an arbitrary pretrained bidirectional dLLM.[^8][^46][^17]

## Critical assessment

Reported speedups are **not directly comparable**. Baselines vary between unoptimised full-sequence inference and strong cached implementations; papers use different GPUs, batch sizes, generation lengths, cache budgets, denoising schedules and accuracy tolerances. Some gains come jointly from caching and parallel decoding, so attributing the full speedup to caching alone is incorrect; the efficiency-evaluation literature has explicitly questioned inconsistent dLLM benchmarking practice.

The most important unresolved issue is **semantic drift under stale caches**. Faster-but-different behaviour has been observed when refresh intervals change, and tighter cache refresh can recover agreement at the cost of speed; this means task accuracy alone may miss content substitutions even when aggregate scores look stable. Approximate-cache papers should therefore report exact-output agreement or trajectory agreement in addition to benchmark accuracy, throughput and FLOPs.[^51]

Memory is not automatically improved by caching. Full hidden/KV reuse may trade compute for substantial persistent state, while aggressive eviction saves memory but can damage long-context retrieval. MaskKV, Prefilling-dLLM, HERALD, GroupKV and MaskAhead are the papers most directly aimed at the memory-capacity side rather than only recomputation.[^36][^13][^40][^30][^31]

## Search protocol

The search used multiple arXiv API queries over exact and synonymous terms, including “diffusion language”, “diffusion LLM”, “masked diffusion”, “block diffusion”, “KV”, “cache”, “caching”, “state reuse”, “activation reuse”, “eviction”, “prefix”, “offload”, “prefill” and “attention caching”. Candidate titles and abstracts were deduplicated by arXiv identifier, then screened manually for text-language relevance.

Forward and backward snowballing used the related-work sections of recent cache papers and current surveys, plus curated diffusion-LLM bibliographies. This matters because papers such as PulseCol, SureLock, Bifocal and FlowBlock can be missed by title-only searches even though reused patterns or cache-preserving execution are technically relevant.[^52][^1]

The cutoff excludes papers first made public after 17 October 2026. The inventory may still miss non-English publications, inaccessible workshop manuscripts, papers with no public abstract, or work whose terminology never mentions caching/state reuse despite implementing it. The split between “direct” and “adjacent” is intentionally conservative and should be retained in downstream bibliometric analysis rather than collapsing all 53 papers into one homogeneous category.

---

## References

1. [Accelerating Masked Diffusion Large Language Models](https://arxiv.org/pdf/2607.12829.pdf) - states are similar, but they introduce a clear memory–speed
   trade-off and can shift the bottleneck t...

2. [Interpolating Between Autoregressive and Diffusion Language Models](https://iclr.cc/virtual/2025/oral/31749) - In this work, we introduce a class of block diffusion language models that interpolate between discr...

3. [Accelerating Diffusion Large Language Models with Adaptive Caching](https://arxiv.org/html/2506.06295v1) - Based on this, we propose dLLM-Cache, a training-free adaptive caching framework that combines long-...

4. [GitHub - horseee/dKV-Cache: [NeurIPS'25] dKV-Cache: The Cache ...](https://github.com/horseee/dKV-Cache) - - Delayed Caching Mechanism: Unlike autoregressive models, dKV-Cache delays the caching of keys and ...

5. [Accelerating Diffusion Language Model Inference via Efficient KV ...](https://arxiv.org/html/2505.21467v1) - We iteratively repeat the denoising process on the current block until the current block is fully un...

6. [Related papers: Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding](https://fugumt.com/fugumt/paper_check/2505.22618v3_enmode) - We introduce Free Draft-and-Verification (FreeDave), a novel fast decoding algorithm tailored forDLL...

7. [Accelerating Diffusion LLMs via Adaptive Parallel Decoding](https://arxiv.org/abs/2506.00413) - The generation speed of LLMs are bottlenecked by autoregressive decoding, where tokens are predicted...

8. [Esoteric Language Models](https://arxiv.org/abs/2506.01928) - Diffusion-based language models offer a compelling alternative to autoregressive (AR) models by enab...

9. [[PDF] Accelerating Diffusion LLMs with Dynamic Cache Eviction - arXiv](https://arxiv.org/pdf/2508.02558.pdf) - First, we cache KV states for tokens outside the current decoding block during early inference step.

10. [d$^2$Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching](https://www.arxiv.org/pdf/2509.23094.pdf) - dkv-cache: The cache for diffusion language models.
    arXiv preprint arXiv:2505.15781, 2025.

11. [github.com · ai4s-research · awesome-diffusion-llmAwesome Diffusion LLM - GitHub](https://github.com/ai4s-research/awesome-diffusion-llm) - - Sequential Diffusion Language Models (SDLM) — Introduces Next-Sequence Prediction to unify next-to...

12. [Fast-dLLM v2: Efficient Block-Diffusion LLM](https://arxiv.org/abs/2509.26328) - Autoregressive (AR) large language models (LLMs) have achieved remarkable performance across a wide ...

13. [Mask Tokens as Prophet: Fine-Grained Cache Eviction for Efficient dLLM Inference](https://web3.arxiv.org/abs/2510.09309) - MaskKV is built on two key innovations: (1) a mask-query guided scoring mechanism that leverages att...

14. [Attention Is All You Need for KV Cache in Diffusion LLMs](https://arxiv.org/html/2510.14973v2) - We presented Elastic-Cache, a training-free, architecture-agnostic policy that makes KV caching in d...

15. [Attention Is All You Need for KV Cache in Diffusion LLMs](https://arxiv.org/pdf/2510.14973.pdf) - First, distant MASK tokens exert negligible influence on unmasking the current token and behave prim...

16. [Orchestrating Dual-Boundaries: An Arithmetic Intensity Inspired Acceleration Framework for Diffusion Language Models](https://www.arxiv.org/pdf/2511.21759.pdf) - Based on the insights from above analysis, we propose ODB-dLLM, a dLLM inference framework that expl...

17. [Track: Oral 5A LLM Training & Inference Efficiency - ICML 2026](https://icml.cc/virtual/2026/session/68663) - A key reason is that many DLLMs rely on bidirectional attention, which breaks standard prefix KV cac...

18. [Window-Diffusion: Accelerating Diffusion Language Model Inference with Windowed Token Pruning and Caching](https://arxiv.org/abs/2601.20332) - Diffusion language models (DLMs) generate text through iterative denoising, but inference requires f...

19. [SPA-Cache: Singular Proxies for Adaptive Caching in Diffusion Language Models](https://arxiv.org/abs/2602.02544) - To address these challenges, we present SPA-Cache that jointly optimizes update identification and b...

20. [FlashBlock: Attention Caching for Efficient Long-Context Block Diffusion](https://arxiv.org/abs/2602.05305) - Generating long-form content, such as minute-long videos and extended texts, is increasingly importa...

21. [Attention Caching for Efficient Long-Context Block Diffusion](https://arxiv.org/html/2602.05305v1) - Based on this observation, we propose FlashBlock, a cached block-external attention mechanism that r...

22. [Stopping Computation for Converged Tokens in Masked Diffusion-LM Decoding](https://arxiv.org/abs/2602.06412) - Masked Diffusion Language Models generate sequences via iterative sampling that progressively unmask...

23. [MAGE: All-[MASK] Block Already Knows Where to Look in Block Diffusion LLM](https://arxiv.org/pdf/2602.14209.pdf) - In addition, MAGE reduces KV cache access time but not its memory footprint;

24. [EntropyCache: Decoded Token Entropy Guided KV Caching for Diffusion Language Models](https://openreview.net/pdf?id=PpfhfFSoJh) - We presented EntropyCache, a KV caching method for diffusion-based large language models that provid...

25. [EntropyCache: Decoded Token Entropy Guided KV ...](https://github.com/mscheong01/EntropyCache) - EntropyCache: Decoded Token Entropy Guided KV Caching for Diffusion Language Models

26. [Diffusion Language Model Activation Reuse for Efficient Inference](https://arxiv.org/abs/2605.08134) - > We introduce DARE, with two complementary mechanisms: DARE-KV, which reuses cached key-value (KV) ...

27. [Polestar: Drift-Aware Cache Calibration and Token ...](https://arxiv.org/html/2607.14107v1) - Polestar comprises two components: Polestar-Cache, which identifies stale KV-cache positions via dri...

28. [Enabling KV Caching of Shared Prefix for Diffusion Language Models](https://arxiv.org/pdf/2606.07571v4.pdf) - BICACHE, for enabling shared prefix KV caching in diffusion language models.

29. [GitHub - TianyiWu233/DYNAMIC-DLLM: Official PyTorch ...](https://github.com/TianyiWu233/DYNAMIC-DLLM) - Dynamic-dLLM consists of two components: **Dynamic Cache Updating (DCU)** reallocates the cache upda...

30. [Prefilling-dLLM: Predictive Prefilling for Long-Context Inference in Diffusion Language Models](https://arxiv.org/pdf/2606.10537.pdf) - Motivated by this observation, we present PREFILLING-DLLM (Prefilling for diffusion LLMs), which com...

31. [HERALD: High-Throughput Block Diffusion LLM Serving via CPU-GPU Cooperative KV Cache Retrieval](https://arxiv.org/abs/2606.21633) - The KV cache dominates GPU memory in long-context LLM serving, crowding out batch capacity and leavi...

32. [Affix Cache for Diffusion Large Language Models](https://arxiv.org/html/2608.26140v1) - # Affix Cache for Diffusion Large Language Models
    ...
    We present ACache, an affix-oriented cache reu...

33. [Sangam: Efficiently Serving Diffusion LLMs with the AR Stack](https://arxiv.org/html/2607.04206v1) - Sangam addresses these differences with a deficit token-budget scheduler that provides amortized sta...

34. [LLM Quant Papers Viewer](https://genggng.github.io/hermes-arxiv-agent/) - 2607.16339 LaCache: Exact Caching and Precision-Adaptive Inference for Diffusion Large Language Mode...

35. [Fugu-MT 論文翻訳(概要): LaCache: Exact Caching and Precision-Adaptive Inference for Diffusion Large Language Models](https://fugumt.com/fugumt/paper_check/2607.16339v2) - 論文の概要: LaCache: Exact Caching and Precision-Adaptive Inference for Diffusion Large Language Models

36. [GroupKV: Hierarchical KV Cache Management for Long-Context Diffusion LLM Inference](https://arxiv.org/html/2609.17573v1) - To address these challenges, we present GroupKV, a lightweight hierarchical KV cache management syst...

37. [How Fixed-State Caches Extend Block Diffusion - CCTest](https://cctest.ai/en/articles/fixed-state-caching-gives-block-diffusion-a-longer-reach) - A sequence mixer summarizes finalized blocks into a reusable state, while a matching block-causal ob...

38. [Flash-dLLM: IO-Aware KV Caching and Parallel Decoding for Fast, Memory-Efficient Diffusion LLMs](https://arxiv.org/html/2609.26796v1) - # Flash-dLLM: IO-Aware KV Caching and Parallel Decoding for Fast, Memory-Efficient Diffusion LLMs

39. [Time-Anchored Diffusion Language Models: Latent-Space Caching for Fast Generation](https://arxiv.org/html/2609.37924v1) - We introduced time-based anchoring, which makes the latent space of a DLM cacheable by learning repr...

40. [[2610.06996] Mask-Guided KV Cache Eviction in Block ...](https://arxiv.org/abs/2610.06996) - by G Molodtsov · 2026 — Abstract page for arXiv paper 2610.06996: Mask-Guided KV Cache Eviction in B...

41. [Accelerating Diffusion Large Language Models with SlowFast Sampling: The Three Golden Principles](https://arxiv.org/abs/2506.10848) - Diffusion-based language models (dLLMs) have emerged as a promising alternative to traditional autor...

42. [Learning to Parallel: Accelerating Diffusion Large Language Models via Learnable Parallel Decoding](https://www.semanticscholar.org/paper/adc3a081078a9267221045e74a79b8c7b422cfb8) - Autoregressive decoding in large language models (LLMs) requires $\mathcal{O}(n)$ sequential steps f...

43. [d3LLM: Ultra-Fast Diffusion LLM using Pseudo-Trajectory Distillation](https://arxiv.org/abs/2601.07568) - Diffusion large language models (dLLMs) offer capabilities beyond those of autoregressive (AR) LLMs,...

44. [Focus-dLLM: Accelerating Long-Context Diffusion LLM Inference via Confidence-Guided Context Focusing](https://arxiv.org/abs/2602.02159) - Diffusion Large Language Models (dLLMs) deliver strong long-context processing capability in a non-a...

45. [Orthrus: Memory-Efficient Parallel Token Generation via Dual-View Diffusion](https://arxiv.org/abs/2605.12825) - We introduce Orthrus, a simple and efficient dual-architecture framework that unifies the exact gene...

46. [Bifocal Diffusion Language Models: Asymmetric Bidirectional ... - arXiv](https://arxiv.org/html/2606.27732v1) - _Causal_ dLLMs such as WeDLM (liu2025wedlm) and CARD (ruan2026card) restore cacheability via topolog...

47. [SW-SpeedDLM: Sliding Window Speculative Decoding for Diffusion Language Models Under Long Context Constraints](https://www.mdpi.com/2227-7390/14/12/2137) - Masked diffusion language models (MDLMs) apply full bidirectional attention at every denoising step,...

48. [FlowBlock: Wavefront-Parallel Decoding for Self-Correcting Diffusion Language Models](https://arxiv.org/abs/2607.17652) - Block-wise diffusion large language models (dLLMs) decode sequentially at the block level, enabling ...

49. [Retrofitting Linear Attention into Diffusion Language Models](https://www.semanticscholar.org/paper/0a614fd09a917c91732a701b9113e31ebe5a9816) - Diffusion language models (dLLMs) offer a promising alternative to autoregressive models by accelera...

50. [Trajectory-Level Speculative Decoding for Diffusion Language Models](https://www.semanticscholar.org/paper/dd7e939d7b5719682ab870794a025072cf39e506) - Diffusion-based language models (dLLMs) enable parallel token generation through iterative denoising...

51. [Faster but Different: Diagnosing and Controlling Content Drift in Accelerated Multimodal Diffusion Language Models](https://arxiv.org/abs/2607.29079) - Training-free acceleration makes diffusion-based multimodal large language models (dMLLMs) more depl...

52. [Awesome-DLMs/README.md at main - GitHub](https://github.com/VILA-Lab/Awesome-DLMs/blob/main/README.md) - [9 Jun 2026] Prefilling-dLLM: Predictive Prefilling for Long-Context Inference in Diffusion Language...

---

## Addendum A: MoE FFN Reuse & Consolidation in Diffusion Language Models (2026-10)

In MoE diffusion LMs (such as `LLaDA2.1-mini`, where routed MoE FFN accounts for **56.3%** of forward wall-clock time vs. **14.3%** for Attention), the FFN can be targeted at six distinct levels:

| Target | Representative Work | Mechanism | Trade-off |
| :--- | :--- | :--- | :--- |
| **FFN outputs** | **DyLLM**, **DARE** ([`arXiv:2603.08026v2`](https://arxiv.org/html/2603.08026v2), [`arXiv:2605.08134`](https://arxiv.org/abs/2605.08134)) | Reuse the previous step's FFN output (or full decoder layer output) for non-salient/stable tokens or shallow layers | Approximate; stale residuals propagate into subsequent layers unless periodically refreshed |
| **Expert branches** | **MoECa** ([`arXiv:2606.15615v1`](https://arxiv.org/pdf/2606.15615v1.pdf)) | Cache individual `(token, expert)` branch outputs $C_{i,l,e} = E_{l,e}(h_{i,l}^{(t_{\text{last}})})$ rather than aggregated MoE output $y_{i,l}^{(t)} = \sum_{e \in R_{i,l}^{(t)}} g_{i,l,e}^{(t)} E_{l,e}(h_{i,l}^{(t)})$ | Finer reuse when router weights/top-$k$ shift independently, but cache indexing and HBM traffic can exceed small-batch GEMM cost |
| **Unique experts** | **DES** ([`arXiv:2602.00879`](https://arxiv.org/abs/2602.00879)), **TEAM** ([`arXiv:2602.08404`](https://arxiv.org/html/2602.08404v1); temporal-spatial-consistency expert activation, not a coreset [E-1]), **dMoE** ([`arXiv:2605.30876`](https://arxiv.org/html/2605.30876)) | Restrict per-token routing to a shared block/sequence-level expert coreset (or reroute cold masked tokens to hot-token experts) | **DES**: >55% fewer unique experts, up to 38% lower latency, 99% accuracy; **TEAM**: ~~35–39% fewer experts,~~ (unverified [E-1]) up to 2.2× speedup; **dMoE**: requires training |
| **Expert budget** | **REFLEX** ([`arXiv:2608.01784`](https://arxiv.org/pdf/2608.01784.pdf)) | Assign fewer experts (variable `top-k`) to already-decoded, cold, or low-priority tokens while keeping full `top-k` for hot frontier tokens | Saves actual FFN FLOPs (~15% fewer expert-token computations on average) without changing router ordering |
| **Expert residency** | **TIDE** ([`arXiv:2605.20179`](https://arxiv.org/html/2605.20179)), **OLED-MoE** ([`arXiv:2609.33385`](https://arxiv.org/abs/2609.33385)), **WiSP** ([`arXiv:2606.21868`](https://arxiv.org/html/2606.21868v2)) | Cache frequently reused expert weights in GPU VRAM across denoising iterations | Lossless (1.4–1.5× in **TIDE** when VRAM-constrained), but offers little gain when all experts already reside in unified memory |
| **FFN precision** | **LaCache** ([`arXiv:2607.16339`](https://arxiv.org/abs/2607.16339)), **ITC-MoE** | Per-group FP8 or low-rank expert weights/activations | Lower risk than output caching when FFN is bandwidth-bound |

---

## Addendum B: Asymmetric KV-Cache Quantization & Diffusion "Confidence Starvation" (2026-10)

### 1. First-Order Perturbation Theory: Why Keys Require Higher Precision Than Values
Attention computes:
$$s = \frac{Q K^\top}{\sqrt{d}}, \qquad p = \operatorname{softmax}(s), \qquad o = p V$$
Under quantization errors $\Delta K$ and $\Delta V$, the logit error is $\Delta s = \frac{Q \Delta K^\top}{\sqrt{d}}$, and to first order:
$$\Delta o \approx J_{\mathrm{softmax}}(s) \Delta s \, V + p \Delta V$$
- **Key error ($\Delta K$)** perturbs the pre-softmax attention logits $s$, passing through the exponential $\operatorname{softmax}(s)$ where $\mathbb{E}[\exp(s + \epsilon)] \neq \exp(s)$ (**Quantized Keys Steal Attention**, [`arXiv:2605.26266`](https://arxiv.org/html/2605.26266v1) — **video diffusion, analogy only, not dLLM evidence** [E-2]) and shifting which prefix positions receive attention mass (**QuantWM**, [`arXiv:2605.03562`](https://arxiv.org/html/2605.03562v1); [`arXiv:2609.26425`](https://arxiv.org/html/2609.26425v2)).
- **Value error ($\Delta V$)** leaves the attention distribution $p$ intact and only perturbs the vectors being convexly averaged ($p \Delta V$).

### 2. Prior Art on Asymmetric K/V Precision & Axis Granularity
- **Quantize What Counts: More for Keys, Less for Values** ([`arXiv:2502.15075`](https://arxiv.org/abs/2502.15075)): Proves `K4V2` (75.2%) dramatically outperforms `K2V4` (54.7%) due to larger spectral/Frobenius norms along the Key path.
- **TurboAngle** ([`arXiv:2603.27467`](https://arxiv.org/abs/2603.27467)) & **KVTuner** ([`arXiv:2502.04420`](https://www.alphaxiv.org/abs/2502.04420)): Evaluate 8-bit key norms / 4-bit value norms and layer/head-specific K/V precision pairs.
- **KIVI** ([`arXiv:2402.02750`](https://arxiv.org/html/2402.02750v2)) & **KVQuant** ([`arXiv:2401.18079`](http://arxiv.org/pdf/2401.18079.pdf)): Show that Key outliers are concentrated in *persistent channels* (**per-channel quantization** for Keys, or pre-RoPE quantization), whereas Value outliers are *token-specific* (**per-token quantization** for Values).
- **QAQ** & **KVLinC** ([`arXiv:2510.05373`](https://arxiv.org/abs/2510.05373)): Apply learned logit corrections or Hadamard rotations specifically to compensate Key softmax error.

### 3. The Diffusion-Specific Feedback Loop: "Confidence Starvation"
In confidence-thresholded block/masked dLLM decoding ($c_i = \max_x p_\theta(x_i = x \mid x_t) \ge \tau_{\text{mask}}$; **Fast-dLLM**, [`arXiv:2505.22618`](https://arxiv.org/html/2505.22618v1); **ReQAT**, [`arXiv:2606.15682`](https://arxiv.org/abs/2606.15682); **Polestar**, [`arXiv:2607.14107`](https://arxiv.org/html/2607.14107)):
1. **FP4 Key noise** perturbs $Q K^\top / \sqrt{d}$ attention scores over the committed prefix.
2. Attention distributions flatten or misdirect, reducing hidden-state and vocabulary-logit margins.
3. Fewer masked positions cross $\tau_{\text{mask}} = 0.70$, forcing the engine onto the `numToTransfer = 1` fallback (**starvation steps**).
4. Tokens remain `[MASK]` for more denoising steps (**+14.4% steps** in `cache-fp4`), and each extra step repeatedly consumes the distorted prefix cache (**-22.7% TPS**).
5. **Decisive $2 \times 2$ Ablation Matrix (COMPLETED 2026-10-18 — see [`Plans/wiki-drafts/track1-asymmetric-kv-confidence-starvation.md`](../Plans/wiki-drafts/track1-asymmetric-kv-confidence-starvation.md))**:
   - Comparing equal-byte `K8V4` (`1.947 MiB`, `-59.38%`) vs `K4V8` (`1.947 MiB`, `-59.38%`) confirmed that `K4V8` inflates steps from `284` to `321` (`+37` steps), inflates 1-token starvation steps from `130` to `152`, drops TPS from `15.67` to `13.25 tok/s` (`-15.4%`), and collapses token parity from `73.0%` (`4/12` exact prompts) to `41.5%` (`0/12` exact prompts).
   - **Block-Aligned `KIVI` (`kivi4-clean`, 4-bit per-channel Keys along 32-token blocks + 4-bit per-token Values, `-70.31%` size, `1.423 MiB`)** completely cures 4-bit Confidence Starvation (`333 -> 280` steps, `167 -> 108` starvation steps, `15.46 tok/s`, `4/12` exact prompt matches).
6. **Self-Contained Implementation Blueprint for Tracks 2–4**: Persisted in [`Plans/wiki-drafts/tracks-2-to-4-implementation-handoff.md`](../Plans/wiki-drafts/tracks-2-to-4-implementation-handoff.md).

