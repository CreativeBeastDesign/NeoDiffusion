# Self-Contained Implementation Handoff: Tracks 2, 3, and 4 (Diffusion Caching & MoE Reuse)

**Date**: 2026-10-18  
**Status**: Ready-to-Execute Blueprint (Zero Context Recapture Required)  
**Prerequisite Completed Work**:
- Base Cache Optimisation Campaign (`2026-10-17`): [`Plans/cache-optimisation-logbook.md`](../cache-optimisation-logbook.md) (§1–§6) & [`Plans/wiki-drafts/dual-caching-and-kv-compression.md`](dual-caching-and-kv-compression.md)
- Track 1 Asymmetric KV-Cache Quantization & "Confidence Starvation" (`2026-10-18`): [`Plans/wiki-drafts/track1-asymmetric-kv-confidence-starvation.md`](track1-asymmetric-kv-confidence-starvation.md) & [`Plans/cache-optimisation-logbook.md`](../cache-optimisation-logbook.md) (§8)
- Literature Survey & Addenda: [`Notes/cache_research.md`](../../Notes/cache_research.md) (Addenda A & B)

## Errata & Corrections (2026-10-18)

> Campaign-number evidence: `python3 Tools/analysis/cache_campaign_aggregate.py` (CPU-only, stdlib) and [`Plans/cache-optimisation-logbook.md`](../cache-optimisation-logbook.md) § "Errata & Corrections (2026-10-18)". Citations checked against arxiv.org abstract pages on 2026-10-18.

- **E-1 — §0.1 presets are not "shipped in `main`" and have open quality gates (status change; fix in progress 2026-10-18, see code-fix section).** The `CachePreset` code is not in commit HEAD `1ff0053` (working tree only). Per the standing rule (no preset ships while its quality gate is open), `k8v4-clean`, `kivi4-clean`, `fp8-clean`, `dual-k8v4` are **experimental, default-off, quality gate OPEN**. `exact-fast` stays default-on (token-identical, 716/716) with **Studio BF16 parity OPEN**. `dual-k8v4` set `dualCachingEnabled = true` (⇒ FlashBlock on) and was never benchmarked; its measured arm (`cache-dual-k8v4`) is CONFOUNDED by the clean-commit skip gate not excluding CSA2 (stale K/V could be committed). `[Sourced]` / mechanism `[Inferred]`
- **E-2 — tok/s figures in §0.1 are mean-of-per-prompt ratios.** Aggregate (Σtok/Σs, Track 1 file, 96/96 valid): FP16 clean-commit 12.50, `k8v4` 12.75, `kivi4` 12.90, `fp8` 13.04 tok/s. KV quantization does not change ms/forward (+0.2–0.9%); differences are trajectory effects. `[Sourced]`
- **E-3 — `fp8`/`fp4`/`k8v4`/… are affine integer formats** (MLX `quantized` default affine mode, scale + bias per group): read `fp8` as "`fp8` (affine int8)". `[Sourced: KVCacheCompression.swift]`
- **E-4 — Forward budget provenance.** The 56.3% / 42.9% / 14.3% / 1.7% figures are WP-6c measurements on the **Mac Studio M2 Ultra**, not M1. "Shared expert + combine ~11.7%" is a **remainder** (56.3 − 42.9 − 1.7); WP-6c's own breakdown lists the shared expert at 4.2%. "Dense Layer 0 + RMSNorms + residuals ~26.4%" is also a **remainder** (100 − 56.3 − 14.3 − 3.0); no measurement of it exists in `Plans/gather_qmm_handoff.md`, and the master list's measured remainder is ~21% (norms + embed + sampler + loop). `[Inferred: remainder]`
- **E-5 — Expert count.** LLaDA2.1-mini has **256** routed experts (`models/llada2-1-mini/config.json` `num_experts: 256`, 8 per token), not 64/128. WP-6g's 57.3 distinct experts per layer per forward (chat suite, `gather_qmm_handoff.md` §8.1) is **~22%** of the experts — routing is concentrated, not "almost the entire expert weight matrix". `[Sourced]`
- **E-6 — DARE citation.** DARE is **arXiv:2605.08134**, "DARE: Diffusion Language Model Activation Reuse for Efficient Inference" (DARE-KV reuses cached K/V activations, DARE-O reuses output activations; query change predicts redundancy; up to 1.20× per-layer latency, up to 87% of attention activations reused, average accuracy drop 2.0% / 1.2%). arXiv:2603.08026 is **DyLLM** ("Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention"). `[Sourced]`
- **E-7 — REFLEX budget.** REFLEX (arXiv:2608.01784, "Rethinking MoE Inference as Refinement-Aware Compute Allocation in Diffusion Language Models") reports a **~15% average** reduction in allocated expert compute (on LLaDA-MoE and LLaDA2.0-mini). The 50% cut / −21.5% of the forward in Exp-2B1 is this handoff's own assumption, roughly 3× the paper's figure, and is `[Speculative]`. `[Sourced]`
- **E-8 — TEAM / DES.** TEAM (arXiv:2602.08404) is "Temporal-Spatial Consistency Guided Expert Activation for MoE Diffusion Language Model Acceleration" (conservative expert activation for decoded/masked tokens + speculative exploration; up to 2.2× speedup). It is **not** an expert-coreset/consolidation method. DES (arXiv:2602.00879) is "Dynamic Expert Sharing: Decoupling Memory from Parallelism in Mixture-of-Experts Diffusion LLMs" — the coreset mechanism in Exp-2C comes from DES only. `[Sourced]`
- **E-9 — MaskAhead vs MAGE.** arXiv:2610.06996 is "Mask-Guided KV Cache Eviction in Block Diffusion Language Models": **MaskAhead** ranks KV entries with current-block mask queries (selection) and upcoming-block probes (eviction); **Q-MaskAhead** is its low-bit-KV variant (up to 20.1× KV-memory reduction on long-prompt QA). The "step-0 all-`[MASK]` top-k index set reused for the rest of the block" mechanism is **MAGE** (arXiv:2602.14209, "All-[MASK] Block Already Knows Where to Look in Block Diffusion LLM"). `[Sourced]`
- **E-10 — "Confidence starvation"**: as measured (1-transfer unmask steps) it is 49.6% of FP16 steps; read as reduced per-step commit rate. `[Inferred]`

---

## 0. Current Baseline & Shipped State (`[Sourced]`)

### 0.1 ~~Shipped~~ Feature Flags & Presets ~~in `main`~~ (working tree; experimental unless noted) (`[Sourced]`) [E-1]
- [`Packages/DiffusionGeneration/Sources/GenerationParams.swift`](../../Packages/DiffusionGeneration/Sources/GenerationParams.swift):
  - `skipRedundantCommitForward: Bool = true` (Default-On; saves `34` redundant 20-layer `captureAndCommit` forwards across the 12-prompt suite with `100.0%` bit-for-bit token parity).
  - `GenerationParams.CachePreset` (`--cache-preset` in `diffusion-bench` and `diffusion-server`):
    - `exact-fast` (`kvQuantization = .none`, `skipRedundantCommitForward = true`): `4.792 MiB`, `15.86 tok/s` (`K=1`; 12.50 aggregate [E-2]), `100.0%` token parity (`12/12` exact prompts; Studio BF16 parity OPEN [E-1]).
    - `k8v4-clean` (**experimental, quality gate OPEN** [E-1]; `kvQuantization = .k8v4` (8-bit affine int K / 4-bit V [E-3]), `skipRedundantCommitForward = true`): **`1.947 MiB` (`-59.38%`)**, `15.67 tok/s` (`K=1`; 12.75 aggregate [E-2]), **`73.0%` token match (`4/12` exact prompts)**.
    - `kivi4-clean` (**experimental, quality gate OPEN** [E-1]; `kvQuantization = .kivi4`, `skipRedundantCommitForward = true`): **`1.423 MiB` (`-70.31%`)**, `15.46 tok/s` (`K=1`; 12.90 aggregate [E-2]), **`58.7%` token match (`4/12` exact prompts)** — cures 4-bit confidence starvation [E-10] (`280` steps vs `333` for `fp4`).
    - `fp8-clean` (**experimental, quality gate OPEN** [E-1]; `kvQuantization = .fp8` (affine int8 [E-3]), `skipRedundantCommitForward = true`): `2.546 MiB` (`-46.88%`), `15.99 tok/s` (`K=1`; 13.04 aggregate [E-2]), `70.2%` token match (`4/12` exact prompts).
    - `dual-k8v4` (`kvQuantization = .k8v4`, `skipRedundantCommitForward = true`, `csa2Enabled = true`, `boundary=4, stride=2, stepStride=2, minStep=2`; **also `dualCachingEnabled = true` ⇒ FlashBlock on, until the 2026-10-18 fix**): `1.947 MiB` (`-59.38%`), `14.64 tok/s` (`K=1`, `21.30 tok/s` on `code`), `60.2%` token match — **numbers from a CONFOUNDED arm without FlashBlock; preset never benchmarked; quality gate OPEN** [E-1].
    - `legacy` (`kvQuantization = .none`, `skipRedundantCommitForward = false`): `4.792 MiB`, `11.94 tok/s` (`K=4`) / `13.77 tok/s` (`K=1`).
- [`Packages/DiffusionCore/Sources/KVCacheCompression.swift`](../../Packages/DiffusionCore/Sources/KVCacheCompression.swift):
  - `KVCacheQuantization` supports `.none`, `.fp8`, `.k8v4`, `.k4v8`, `.fp4`, `.fp4g32`, `.kivi4` (`perChannel = true` along the 32-token block axis for Keys, `perChannel = false` for Values), and `.k8v4s4`.

### 0.2 In-Situ Forward Cost Breakdown on `LLaDA2.1-mini` (`[Sourced]` from `WP-6c` & `Step 3`; **measured on the Mac Studio M2 Ultra** [E-4])
Before starting Track 2, keep the measured forward budget in view (`Plans/gather_qmm_handoff.md` §5.6–§7):
- **Engine loop outside forward**: `~5.2%` (`forward = 94.8%` of step wall-clock).
- **Inside the 20-layer forward pass (`B=32` active tokens)**:
  - **Routed MoE (`19` MoE layers, `L1..L19`)**: **`56.3%`** (`gatherQuantizedMM` expert GEMMs alone = **`42.9%`**, router = **`1.7%`**, shared expert + combine = **`~11.7%`** `[Inferred: remainder; WP-6c lists shared expert at 4.2%]` [E-4]).
  - **Attention (including KV cache read/write + QKV/O projections)**: **`14.3%`**.
  - **LM Head (`[2048 -> 157184]`)**: **`~3.0%`**.
  - **Dense Layer 0 + RMSNorms + residuals**: **`~26.4%`** `[Inferred: remainder = 100 − 56.3 − 14.3 − 3.0; not measured]` [E-4].

> **Key Takeaway (`[Inferred]`)**: Any within-block optimization that only touches Attention (`14.3%`) is capped at a `1–3%` wall-clock gain. To achieve a `>10%` single-step speedup on `LLaDA2.1-mini`, an optimization **must reduce `gatherQuantizedMM` work inside `LLaDA2SparseMoEBlock` (`42.9%` of the forward)** without triggering CPU-GPU synchronization (`numActive.item()`) and without inflating denoising steps by more than the `ms/fwd` savings.

---

## Track 2: MoE FFN & Layer-Output Reuse on `LLaDA2.1-mini`

### 2.1 `Exp-2A`: `DARE-O` / `DyLLM` Whole-Layer or MoE-FFN Output Caching (`[Speculative]`)

#### Hypothesis & Mechanism
- **Papers**: `DARE` (~~`arXiv:2603.08026v2`~~ **`arXiv:2605.08134`** [E-6]) and `DyLLM` (**`arXiv:2603.08026`** [E-6]; `Notes/cache_research.md` Addendum A).
- Instead of caching only active-block K/V (`CSA2`, which saved `<1%` `ms/fwd` because K/V projection is tiny), cache the **MoE FFN output $\Delta h_{\text{FFN}}^{(\ell, t)} \in \mathbb{R}^{1 \times 32 \times 2048}$** (or the entire decoder layer output) and reuse it on selected steps/layers.
- **Critical Lesson from `CSA2` (`Plans/cache-optimisation-logbook.md` §3.1)**:
  - **Never** cache or reuse on `stepInBlock == 0` (all `[MASK]` input) or `stepInBlock == 1` (first step with unmasked anchor tokens). Reuse must only activate at `stepInBlock >= 2` with periodic refresh (`stepStride = 2`: Steps 0, 1, 3, 5... compute fresh MoE FFN; Steps 2, 4, 6... reuse or partially reuse).

#### Two Sub-Variants to Test in `Exp-2A`
1. **`Exp-2A1` (Layer-Periodic MoE FFN Caching on Shallow/Alternating Layers)**:
   - On MoE layers matching a schedule (e.g., shallow MoE layers `1..<5`, or alternating MoE layers `1, 3, 5, 7` with `minStep = 2, stepStride = 2`), store `cachedFFNOutput: MLXArray?` on `LLaDA2SparseMoEBlock` (or `LayerActiveCache`).
   - On reuse steps (`stepInBlock >= 2 && stepInBlock % 2 == 0`), skip `mlp(ffnInput)` completely and add `cachedFFNOutput` to `hidden`.
   - **Expected `ms/fwd` savings**: Skipping 4 of 19 MoE layers on 40% of steps saves `~4/19 × 56.3% × 0.40 ≈ 4.7%` of total denoising compute; skipping 9 alternating MoE layers on 40% of steps saves `~10.7%` of compute.
   - **Success Gate**: Net wall-clock TPS gain require step inflation `< 4%` (`<= +11` steps across the 12-prompt suite).
2. **`Exp-2A2` (Token-Selective MoE FFN Reuse for Already-Decoded Tokens + `LayerJotCache` KV Hold)**:
   - When a token in the 32-token block has already been unmasked on an earlier step (or frozen via `JOT`), its hidden state changes very little on subsequent refinement steps (`DyLLM`).
   - If combined with `Exp-2B` (`REFLEX`) below, decoded tokens can either reuse their cached FFN output or use fewer experts.

#### Exact Code Hooks for `Exp-2A`
- **File 1**: [`Packages/DiffusionCore/Sources/LLaDA2DecoderLayer.swift`](../../Packages/DiffusionCore/Sources/LLaDA2DecoderLayer.swift#L56-L129)
  - Add `private var cachedFFNOutput: MLXArray? = nil` (and `public func resetFFNCache()`) to `LLaDA2DecoderLayer`, or store it in `ExactPrefixCache` per layer.
  - In `callAsFunction(_ x:cos:sin:cache:...)`, when `ffnReuse == true` and `cachedFFNOutput != nil` (and shape matches `x.shape`), set `ffnOutput = cachedFFNOutput!`; otherwise compute `ffnOutput` and store `cachedFFNOutput = ffnOutput`.
- **File 2**: [`Packages/DiffusionGeneration/Sources/DiffusionEngine+Entry.swift`](../../Packages/DiffusionGeneration/Sources/DiffusionEngine+Entry.swift#L130-L195)
  - Pass `stepInBlock` into the cached forward path (already wired when `csa2Config != nil`) and reset `cachedFFNOutput` at the start of each block (`stepInBlock == 0`).

---

### 2.2 `Exp-2B`: `REFLEX` Step-Adaptive & Role-Adaptive Expert Budget (`[Speculative]`)

#### Hypothesis & Mechanism
- **Paper**: `REFLEX` (`arXiv:2608.01784`, `Notes/cache_research.md` Addendum A; reports ~15% average expert-compute reduction [E-7]).
- `LLaDA2.1-mini` routes every token in the 32-token block to **`topK = 8` routed experts** (`256` expert-token evaluations per layer) on every single denoising step — even when 25 of the 32 tokens are already decoded and the step only unmasks 1–2 remaining tokens!
- **Hardware Constraint on Apple Silicon (`[Sourced]` from `WP-3a` & `WP-6f`)**:
  - `gatherQuantizedMM(x, weight, scales, biases, rhsIndices: indices)` expects a rectangular `indices` tensor of shape `[T, k]`.
  - If `k` varies *per token* within the same `gatherQuantizedMM` call, one must either pad to `max(k)` (saving zero FLOPs) or partition tokens into buckets.
  - **HOWEVER**, varying `k` **per denoising step** (`k_t \in \{8, 6, 4, 2\}` depending on `stepInBlock` or remaining mask ratio) keeps `indices` strictly rectangular `[32, k_t]` with **zero CPU-GPU sync, zero gather/scatter overhead, and a direct linear reduction in `gatherQuantizedMM` dispatches and FLOPs!**

#### Two Zero-Sync Variants to Test in `Exp-2B`
1. **`Exp-2B1` (Step-Progress `REFLEX`: Full `topK = 8` on Early Steps, Reduced `topK = 4` on Refinement Steps)**:
   - On `stepInBlock < reflexStepBoundary` (e.g., `stepInBlock < 2` or `stepInBlock < 3`, where bulk unmasking happens), run `LLaDA2MoEGate` with full `effectiveTopK = 8`.
   - On `stepInBlock >= reflexStepBoundary` (late refinement steps where most of the block is already unmasked and only 1–3 tail tokens are being resolved), slice the router's already-sorted top-$k$ indices to `effectiveTopK = 4` (or `6`) and re-normalize `weights` over those `effectiveTopK` experts:
     ```swift
     let kEff = effectiveTopK ?? topK
     let indicesRaw = argSort(-maskedScores, axis: -1)[0..., ..<kEff] // [T, kEff]
     ```
   - **Why this is a prime candidate on M1 & Studio (`[Inferred]`)**:
     - `SwitchGLU` immediately runs `gatherQuantizedMM` on `[32, 4]` instead of `[32, 8]` — **cutting `gatherQuantizedMM` work by 50% on every step $\ge \text{boundary}$** (`-21.5%` of the entire model forward! — `[Speculative]`; the REFLEX paper itself reports only ~15% average expert-compute reduction [E-7]) with **zero GPU-CPU sync, zero scatter/gather, and preserving the router's exact top-4 expert ordering per token!**
2. **`Exp-2B2` (Two-Bucket Token-Role `REFLEX` without CPU Sync)**:
   - Partition the 32 tokens on the GPU (via `argSort` on `isMasked` or previous-step margin, exactly like `LLaDA2SparseMoEBlock`'s static-capacity path at lines 420–438 of `LLaDA2MoE.swift`) into two static-sized buckets:
     - **Hot / Masked Bucket (`16` tokens)**: `topK = 8` (`128` expert-token evaluations)
     - **Cold / Already-Decoded Bucket (`16` tokens)**: `topK = 2` (`32` expert-token evaluations)
   - Total expert-token evaluations drop from `32 × 8 = 256` to `128 + 32 = 160` (**`-37.5%` expert GEMM FLOPs**).

#### Exact Code Hooks for `Exp-2B`
- **File 1**: [`Packages/DiffusionCore/Sources/LLaDA2MoE.swift`](../../Packages/DiffusionCore/Sources/LLaDA2MoE.swift#L247-L311)
  - Add `effectiveTopK: Int? = nil` parameter to `LLaDA2MoEGate.callAsFunction(_ x:ablation:reuseRouter:effectiveTopK:)` and `LLaDA2SparseMoEBlock.callAsFunction`.
  - At line 293 of `LLaDA2MoE.swift`:
    ```swift
    let kEff = min(topK, max(1, effectiveTopK ?? topK))
    let indicesRaw = argSort(-maskedScores, axis: -1)[0..., ..<kEff]  // [T, kEff] descending
    let indices = indicesRaw.asType(.int32)
    var weights = takeAlong(scores, indices, axis: -1)
    if kEff > 1 {
        weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
    }
    weights = weights * routedScalingFactor
    ```
- **File 2**: [`Packages/DiffusionGeneration/Sources/GenerationParams.swift`](../../Packages/DiffusionGeneration/Sources/GenerationParams.swift)
  - Add `reflexEnabled: Bool = false`, `reflexMinStep: Int = 2`, `reflexLateTopK: Int = 4`.

---

### 2.3 `Exp-2C`: `DES` / `TEAM` Block-Level Expert Coreset Consolidation (`[Speculative]`)

#### Hypothesis & Mechanism
- **Papers**: `DES` (Dynamic Expert Sharing, `arXiv:2602.00879`) and `TEAM` (`arXiv:2602.08404`; temporal-spatial consistency guided expert activation — not a coreset method [E-8]).
- `WP-6g` (`Plans/gather_qmm_handoff.md` §8) measured that a single 32-token forward in `LLaDA2.1-mini` activates **57.3 distinct experts per layer** (out of ~~64/128~~ **256** [E-5]), ~~meaning almost the entire expert weight matrix is touched on every layer~~ **i.e. ~22% of the expert weights — routing is already concentrated (~4.5 picks per expert)** [E-5].
- `DES` aggregates router scores across the 32-token block (`blockMass = scoresForRouting.sum(axis: 0)`), selects a block-level coreset of size $M \in \{16, 24, 32\}$, masks non-coreset experts to `-Float.infinity`, and then selects each token's `top-8` from within that shared coreset.
- **Important Hardware Caveat from `WP-6f` (`[Sourced]`)**:
  - In `WP-6f` (`Plans/gather_qmm_handoff.md` §10), the `.moeFixedExperts` probe forced all 32 tokens onto `8` distinct experts without sorting and ran **25% slower** on M2 Ultra because unsorted threads hitting the same expert caused cache/bank conflicts, whereas `T >= 128` sorted dispatch (`sortedIndices: true` at line 457 of `LLaDA2MoE.swift`) sped up `gatherQuantizedMM`.
  - Therefore, if testing `DES` (`Exp-2C`), test it **both** with unsorted `gatherQuantizedMM` and with `sortedIndices: true` (GPU-sorted expert indices) so threads hitting the same coreset expert coalesce!

---

## Track 3: Long-Context Sparse Prefix Block Selection (`LLaDA2.1-mini` at $L \ge 2048$)

### 3.1 `Exp-3A`: `Q-MaskAhead` / `MAGE` Step-0 Top-$k$ Prefix Block Selection (`[Speculative]`)

#### Hypothesis & Mechanism
- **Papers**: `MAGE` (`arXiv:2602.14209`; source of the step-0 all-`[MASK]` top-k reuse idea) and `Q-MaskAhead` (`arXiv:2610.06996`, "Mask-Guided KV Cache Eviction in Block Diffusion Language Models" — mask-query selection + eviction, low-bit variant) [E-9] (`Notes/cache_research.md`).
- At short contexts (`64–192` tokens), the entire prefix KV cache is only `1.42–1.95 MiB` (`kivi4` / `k8v4`), so reading and dequantizing the full prefix takes microseconds.
- At long contexts ($L = 2048\text{–}8192$ tokens = `64–256` committed blocks), dequantizing and attending to all `2048–8192` prefix tokens on every denoising step of every block scales as $O(L)$.
- **Mechanism**:
  1. Store committed prefix KV in **per-block `CompressedKVTensor` chunks of 32 tokens** (or slice the packed `CompressedKVTensor` along axis 2 in multiples of 32 tokens, since both `.k8v4` and `.kivi4` are block-aligned along axis 2!).
  2. On **Step 0** of a new block (`stepInBlock == 0`, when all 32 active tokens are `[MASK]`), attend to the full prefix (or block-pooled mean Keys $\bar{K}_b \in \mathbb{R}^{1 \times n_{KV} \times D}$ for each prefix block $b \in \{0,\dots,N_{\text{blk}}-1\}$) and rank prefix blocks by cumulative attention score from the 32 `[MASK]` queries.
  3. Always retain **Block 0** (attention sink) + **most recent 2 blocks** (local window) + **top-$k$ highest-scoring prefix blocks** ($k \in \{8, 16\}$).
  4. On **Steps $1 \dots S$** of that block, gather and dequantize **only** the selected $k+3$ blocks from `CompressedKVTensor` before running `scaledDotProductAttention`!
- **Why this composes cleanly with Track 1 (`[Inferred]`)**:
  - In `CompressedKVTensor` ([`KVCacheCompression.swift`](../../Packages/DiffusionCore/Sources/KVCacheCompression.swift)), axis `2` of `packed`, `scales`, and `biases` (for per-token `.k8v4`) has length `L` (or `L / 32` for `.kivi4` scales/biases). Thus, selecting a subset of 32-token prefix blocks **before calling `.dequantize()`** avoids even dequantizing unselected prefix blocks!

---

## Track 4: Faithful `Elastic-Cache` Algorithm 1 on Full-Canvas Bidirectional `Sumi-7B`

### 4.1 Why `Sumi-7B` Is the Ideal Target for `Elastic-Cache` (`[Sourced]` & `[Inferred]`)

1. **Architectural Contrast with `LLaDA2.1-mini` (`[Sourced]`)**:
   - `LLaDA2.1-mini` is a **block-causal** diffusion model (`B=32`) where `ExactPrefixCache` already drops the prefix from per-step QKV/FFN and block-causal masking already drops all future blocks (`> B`).
   - **`Sumi-7B`** ([`SumiEngine.swift`](../../Packages/DiffusionGeneration/Sources/SumiEngine.swift) and [`Plans/sumi-logbook.md`](../sumi-logbook.md)) is a **1.0-style full-canvas bidirectional diffusion model** (`28` layers, `hiddenSize = 3584`, `intermediateSize = 18944` dense SwiGLU, `28` Q heads, `4` KV heads, `headDim = 128`).
   - Per [`Plans/sumi-logbook.md`](../sumi-logbook.md) (§4.2):
     - `Sumi-7B` runs with `use_cache = false` across a fixed `canvasLength = 1024` (or `2048`) on every single step (`~14.6 TFLOP/step`, **`8.3 s/step` on M1**).
     - Even when generating `maxNewTokens = 64` with a `~40`-token prompt (`denoiseEnd = promptLength + 64 + 2 ≈ 106`):
       - Tokens `0 ..< promptLength` (`~40` tokens) are **pinned prompt tokens** (never change).
       - Tokens `promptLength ..< denoiseEnd` (`64` tokens) are the **active denoising window** (and under `freezeCommitted = true`, committed positions within this window also freeze progressively!).
       - Tokens `denoiseEnd ..< canvasLength` (`~918` tokens out of `1024`!) are the **frozen post-anchor tail** (`activeMask == false`: their input token IDs never change across all `numDenoisingSteps` steps!).
     - Currently, `SumiModel.logits(forTokens: z)` runs `QKV`, `scaledDotProductAttention`, and the massive **dense 7B SwiGLU FFN (`[1024, 3584] -> [1024, 18944] -> [1024, 3584]`)** over all **1,024 tokens** on every step!

2. **Expected Speedup from Faithful `Elastic-Cache` Algorithm 1 on `Sumi-7B` (`[Inferred]`)**:
   - In `Elastic-Cache` Algorithm 1 (`Resources/elastic-cache.md`), on cache-reuse steps:
     - **The active window $W = [\text{promptLength}, \text{denoiseEnd})$ (`64` tokens) is ALWAYS recomputed** (`Line 18` of Alg. 1 — avoiding the `WP-1a` flaw where active tokens were given stale K/V!).
     - **The static exterior $E = [0, \text{promptLength}) \cup [\text{denoiseEnd}, \text{canvasLength})$ (`~960` tokens)** has constant input token IDs; its $K_E^\ell, V_E^\ell$ are cached from the most recent refresh step and concatenated with the freshly computed $K_W^\ell, V_W^\ell$ (`64` tokens).
   - Consequently, on cache-reuse steps in layers $\ell \ge \ell^\star$:
     - `inputLayernorm`, `Q/K/V` projections, `O` projection, `postAttentionLayernorm`, and the **dense 7B SwiGLU FFN** execute on **`|W| = 64` tokens instead of `1024` tokens** — a **`16.0x` reduction in QKV & FFN FLOPs and weight-bandwidth reads**!
     - Bidirectional attention runs with `Q` of length `64` against `K, V` of length `1024` (`[1, 28, 64, 1024]`), which is **`16.0x` fewer attention FLOPs** than `[1, 28, 1024, 1024]`!
   - Even with periodic full-canvas refreshes every $R = 4$ or $R = 8$ steps (or attention-aware drift refresh $\sigma_t^\ell < \gamma$ from shallow boundary $\ell^\star \in \{2, 4\}$), the theoretical per-step speedup is **`3.5x–7.0x`** (`8.3 s/step -> ~1.2–2.4 s/step` on M1)!

### 4.2 Exact Implementation Blueprint for `Sumi-7B` `Elastic-Cache` (`Exp-4A`)

1. **Where `SumiModel` Runs Layers**:
   - Inspect `Packages/DiffusionModel/Sources/SumiModel.swift` (and its decoder layer / attention classes).
   - Currently, `SumiModel` applies `RoPE` using sequential positions `0 ..< canvasLength` and passes `hidden` `[1, 1024, 3584]` through all 28 layers.
2. **Precomputing Position Split (`W` vs `E`)**:
   - Given `promptLength` and `denoiseEnd = promptLength + budget + reserve`:
     - Active window range: `W = promptLength ..< denoiseEnd` (contiguous slice of length `|W| ≈ 64`!).
     - Because `W` is a **single contiguous slice** `[promptLength ..< denoiseEnd]`, slicing `hidden` to `W` at layer $\ell^\star$ is a **zero-copy contiguous slice** `hidden[0..., promptLength ..< denoiseEnd, 0...]` (`[1, |W|, 3584]`), and its RoPE `cos`/`sin` slice is simply `cos[promptLength ..< denoiseEnd]`, `sin[promptLength ..< denoiseEnd]`!
3. **Per-Layer Cache Structure (`SumiLayerElasticCache`)**:
   - For each layer $\ell \in \{0,\dots,27\}$, store:
     - `cachedKeysPrefix: MLXArray?` (`[1, nKV, promptLength, D]` — or compressed via `CompressedKVTensor` with `.k8v4` / `.kivi4`!)
     - `cachedValuesPrefix: MLXArray?`
     - `cachedKeysSuffix: MLXArray?` (`[1, nKV, canvasLength - denoiseEnd, D]`)
     - `cachedValuesSuffix: MLXArray?`
4. **Refresh vs. Reuse Step Execution**:
   - **On Refresh Steps** (`step == 0`, or `step % refreshInterval == 0`, or when cosine similarity $\sigma_t^{\ell^\star} < \gamma$):
     - Run full `[1, 1024, 3584]` through all layers $0 \dots 27$ (or from $\ell^\star \dots 27$ if layers $0 ..< \ell^\star$ always run on `[1, 1024, 3584]`, or even simpler: since tokens outside $W$ are 100% static at layer 0 input, **layer 0's exterior K/V is 100% constant across the entire generation** and only changes in layers $\ell \ge 1$ due to bidirectional attention into $W$!).
     - Notice that! In `Sumi-7B`:
       - At **Layer 0** ($\ell = 0$), the input embeddings for $E = [0, \text{promptLength}) \cup [\text{denoiseEnd}, 1024)$ **never change at any step $t$**! So at Layer 0, $K_E^{(0)}, V_E^{(0)}$ are **100% mathematically invariant across all steps $t \in \{0,\dots,T-1\}$**!
       - Only at **Layers $\ell \ge 1$** do exterior tokens' hidden states drift slightly as active-window tokens in $W$ change.
     - Save `K[0..., 0..., 0 ..< promptLength, 0...]` and `K[0..., 0..., denoiseEnd ..< canvasLength, 0...]` (and `V`) into `SumiLayerElasticCache`.
   - **On Reuse Steps** (for layers $\ell \ge \ell^\star$, where $\ell^\star$ can even be `0` between refreshes!):
     - Slice `hidden` to `hidden_W = hidden[0..., promptLength ..< denoiseEnd, 0...]` (`[1, |W|, 3584]`).
     - Compute `Q_W, K_W, V_W` on `hidden_W` (`|W| = 64` tokens) with RoPE offset `promptLength`.
     - Form full `K = concatenated([cachedKeysPrefix, K_W, cachedKeysSuffix], axis: 2)` (`[1, nKV, 1024, D]`) and `V` (`[1, nKV, 1024, D]`).
     - Run `scaledDotProductAttention(queries: Q_W, keys: K, values: V, scale: scale, mask: .none)` $\rightarrow$ `[1, nH, |W|, D]`.
     - Run `oProj`, residual add, `postAttentionLayernorm`, and **dense SwiGLU FFN on `[1, |W|, 3584]` (`64` tokens)**!
     - At the final layer (`27`), `lmHead` only needs logits for `W` (`[1, |W|, vocabSize]`) because `SumiEngine` only updates tokens where `activeMask == true` (which is strictly a subset of `W`)!
5. **Benchmark Command for `Exp-4A` (Once Implemented)**:
   ```bash
   ./.build/release/diffusion-bench sumi \
     --arms sumi-baseline,sumi-elastic-r4,sumi-elastic-r8,sumi-elastic-k8v4 \
     --json scratch/track4_sumi_elastic.jsonl
   ```

---

## Track 5: Additional High-Value Caching Experiments (`Exp-1D`, `Exp-5A`, `Exp-5B`, `Exp-5C`)

### 5.1 `Exp-1D`: Pre-RoPE `KIVI` (`KVQuant` + `KIVI` Hybrid: Quantizing Keys *Before* Rotary Embeddings) (`[Inferred]` / `[Speculative]`)

#### Motivation & Mechanism
- In Track 1 (`scratch/track1_kv_starvation_k1.jsonl`), post-RoPE `kivi4` (`4-bit per-channel Keys along 32-token blocks + 4-bit per-token Values`) completely cured 4-bit Confidence Starvation (`333 -> 280` steps, `15.46 tok/s`, `-70.31%` KV cache size, `4/12` exact prompt matches), but its overall token match (`58.7%`) still trailed `k8v4` (`73.0%`).
- **Why (`[Inferred]`)**: In [`LLaDA2Attention.swift`](../../Packages/DiffusionCore/Sources/LLaDA2Attention.swift), `cache.pendingKeys` currently captures Keys **after** `applyRotaryPosEmb(k, cos: cos, sin: sin)`. Because `RoPE` rotates channel pairs $(d, d+D/2)$ by a position-dependent angle $m\theta_d$ across token positions $m \in \{0,\dots,31\}$ within a block, a persistent outlier in channel $d$ is trigonometrically mixed into channel $d+D/2$ as $m$ varies across the 32-token block, slightly smearing the per-channel dynamic range.
- **Implementation Blueprint**:
  1. In [`LLaDA2Attention.swift`](../../Packages/DiffusionCore/Sources/LLaDA2Attention.swift), store **pre-RoPE** `k` (post-`keyLayernorm`, pre-`applyRotaryPosEmb`) in `cache.pendingKeys` when `quantization == .kivi4PreRope` (alongside the block's `cos`/`sin` slice, or recomputing `cos`/`sin` from `0 ..< committedLength`).
  2. Quantize pre-RoPE `k` per-channel (`groupSize = 32` along the 32-token block axis), where channel outliers are **100% position-stationary**.
  3. Upon dequantization in `LayerKVCache.keys`, apply `applyRotaryPosEmb` to the dequantized pre-RoPE Keys.
  4. **Target Gate**: Lift 4-bit token match from `58.7%` (`kivi4` post-RoPE) to **`>73%` (matching or beating `k8v4`)** while retaining the full **`-70.31%` (`1.423 MiB`)** 4-bit memory reduction.

---

### 5.2 `Exp-5A`: Fused Quantized Prefix Attention (Eliminating On-the-Fly FP16 Dequantization Buffers) (`[Inferred]`)

#### Motivation & Mechanism
- Currently, `LayerKVCache.keys` and `LayerKVCache.values` call `CompressedKVTensor.dequantize()`, which materializes a temporary uncompressed FP16 `[1, nKV, L_prefix, 128]` tensor in unified memory on every step of every layer before calling `MLXFast.scaledDotProductAttention`.
- At short contexts (`L_prefix <= 128`, `4.79 MiB`), dequantizing on the fly takes `< 0.1 ms`, which is why `k8v4` and `kivi4` match `FP16` speed (`15.46–15.67 tok/s` vs `15.86 tok/s`) rather than exceeding it. At long contexts ($L_{\text{prefix}} = 4096\text{–}16384$ tokens), however, materializing temporary FP16 K/V buffers on every denoising step wastes memory bandwidth.
- **Implementation Blueprint**:
  - **Option A (`MLX.quantizedMatmul`)**: For per-token quantized `K` and `V` (`k8v4` / `fp8`, affine int formats [E-3]), compute $s_{\text{prefix}} = Q \widehat{K}_{\text{prefix}}^\top / \sqrt{D}$ directly via `MLX.quantizedMatmul(q, compressedKeys.packed, scales: compressedKeys.scales, biases: compressedKeys.biases, transpose: true, groupSize: 64, bits: 8)` without materializing $\widehat{K}_{\text{prefix}}$ in FP16!
  - **Option B (Custom Metal Flash-Decoding Kernel)**: Dequantize packed `uint32` K/V tiles directly inside threadgroup/SIMD registers during the prefix attention loop, combining online softmax (`logsumexp`) across the compressed prefix and the uncompressed 32-token active block.

---

### 5.3 `Exp-5B`: Cross-Turn Persistent Prefix KV Cache (`SessionPrefixCache` for Multi-Turn Chat & System Prompts) (`[Inferred]`)

#### Motivation & Mechanism
- Currently, `ExactPrefixCache` is allocated inside `DiffusionEngine.generateCached(prompt:params:...)` ([`DiffusionEngine+Entry.swift`](../../Packages/DiffusionGeneration/Sources/DiffusionEngine+Entry.swift#L225)) and discarded when the generation finishes.
- In multi-turn chat (`Tools/diffusion-server` and `App/Diffusion`), every user turn re-prefills the system prompt and the entire prior conversation history from token 0 (`prefillSeconds`).
- Because `LLaDA2.1-mini` uses **strict block-causal attention (`blockLength = 32`)**, any leading sequence of complete 32-token blocks whose token IDs match the previous turn's token IDs has **100% mathematically identical K/V states**.
- **Implementation Blueprint**:
  1. Add `committedBlockTokens: [[Int]]` (array of 32-token arrays) to `ExactPrefixCache`.
  2. Allow `DiffusionEngine.generateCached` to accept and return a reusable `sessionCache: ExactPrefixCache?`.
  3. At the start of `generateCached(prompt:...)`, compare the new `prompt`'s 32-token blocks `0, 1, ...` against `sessionCache.committedBlockTokens`.
  4. Retain the longest matching prefix of $M$ 32-token blocks (`M * 32` tokens) in `sessionCache` (truncating any diverged tail blocks via slice along axis 2 of `CompressedKVTensor`), and only prefill the new blocks from `M * 32 ..< prefillBlocks * 32`!
  5. With `k8v4-clean` (`16,640 B/tok`) or `kivi4-clean` (`12,160 B/tok`), caching a 2,048-token multi-turn conversation takes only **`23.8–32.5 MiB`** of RAM and reduces multi-turn prefill time (`TTFT`) by **80–95%**.

---

### 5.4 `Exp-5C`: Mac Studio M2 Ultra (192 GB) & Long-Sequence (`genLength = 256–512`) Backfill (`[Sourced]` per `AGENTS.md` Rule 3.4)

#### Motivation & Command
- Per `AGENTS.md` Rule 3.4, all M1 caching winners (`exact-fast`, `k8v4-clean`, `kivi4-clean`, `fp8-clean`) should be verified in a single **genuinely interleaved (prompt-major, not arm-blocked — the M1 campaigns were blocked)** process on the target 192 GB Mac Studio M2 Ultra at both standard (`genLength = 128`) and long (`genLength = 512`) lengths:
  ```bash
  ./.build/release/diffusion-bench llada \
    --arms q-cached-legacy,cache-clean-commit-k1,cache-fp8-clean-k1,cache-k8v4-clean-k1,cache-kivi4-clean-k1 \
    --speculation-k 1 --suites chat,reasoning,code --runs 3 --gen-length 128 --warmup-once \
    --json scratch/studio_cache_winners_k1_gen128.jsonl
  ```

---

## 6. Master Checklist for Resuming Work Next Week

| Track / Exp | Target Model | Experiment Summary | Files to Touch | Primary Metric Gate |
| :--- | :---: | :--- | :--- | :--- |
| **Track 1 Follow-Up (`Exp-1D`)** | `LLaDA2.1-mini` | **Pre-RoPE `KIVI` (`kivi4PreRope`)**: quantize Keys per-channel *before* `RoPE` so channel outliers don't rotate across token positions | `LLaDA2Attention.swift`, `KVCacheCompression.swift`, `LayerKVCache.swift` | `-70.3%` KV size (`1.423 MiB`), `<= 285` steps, lift token match from `58.7%` to **`>73%`** |
| **Track 2 (`Exp-2B1`)** | `LLaDA2.1-mini` | **Step-Adaptive `REFLEX`** [E-7]: `topK=8` on steps `0..<2`, `topK=4` on steps `>=2` (rectangular `[32, 4]`, zero CPU sync) | `LLaDA2MoE.swift`, `GenerationParams.swift`, `LLaDABench.swift` | `-20%` `ms/fwd` on steps $\ge 2$, `< +3%` step inflation, **`>= +8%` net TPS** |
| **Track 2 (`Exp-2A1`)** | `LLaDA2.1-mini` | **Periodic MoE FFN Output Caching (`DARE-O` / `DyLLM`)**: `minStep=2, stepStride=2` on shallow MoE layers `1..<5` | `LLaDA2DecoderLayer.swift`, `DiffusionEngine+Entry.swift` | `-5%` to `-9%` `ms/fwd`, `< +3%` step inflation, net positive TPS |
| **Track 2 (`Exp-2C`)** | `LLaDA2.1-mini` | **`DES` ~~/ `TEAM`~~ Block-Level Expert Coreset** [E-8] ($M \in \{16, 24, 32\}$) + GPU-sorted dispatch (`sortedIndices: true`) | `LLaDA2MoE.swift` | `-40%` to `-55%` unique experts per layer, net positive `ms/fwd` & TPS |
| **Track 3 (`Exp-3A`)** | `LLaDA2.1-mini` | **`Q-MaskAhead` / `MAGE` Top-$k$ Prefix Block Selection** on `k8v4` / `kivi4` before `.dequantize()` at $L \ge 2048$ | `KVCacheCompression.swift`, `LayerKVCache.swift`, `ExactPrefixCache.swift` | $2\times\text{–}4\times$ faster prefix attention & dequantization at $L \in \{2048, 4096\}$ |
| **Track 4 (`Exp-4A`)** | `Sumi-7B` | **Faithful `Elastic-Cache` Alg. 1**: Active Window `W=64` always recomputed; Exterior `E=960` cached with periodic refresh | `SumiModel.swift`, `SumiEngine.swift`, `SumiBench.swift` | **`3x–7x` wall-clock speedup** (`8.3 s/step -> ~1.2–2.4 s/step` on M1), $\ge 90\%$ token agreement |
| **Track 5 (`Exp-5A`)** | `LLaDA2.1-mini` | **Fused Quantized Prefix Attention** (`quantizedMatmul` / Metal kernel reading packed KV directly without temporary FP16 buffer) | `LLaDA2Attention.swift`, `KVCacheCompression.swift` | Eliminates temporary FP16 dequant buffer; speeds up long-context attention at $L \ge 2048$ |
| **Track 5 (`Exp-5B`)** | `LLaDA2.1-mini` | **Cross-Turn Persistent `SessionPrefixCache`** in `DiffusionEngine`, `diffusion-server`, and macOS `App` | `ExactPrefixCache.swift`, `DiffusionEngine+Entry.swift`, `Tools/diffusion-server` | **80–95% reduction in multi-turn prefill `TTFT`** for shared 32-token prefix blocks |
| **Track 5 (`Exp-5C`)** | `LLaDA2.1-mini` | **Mac Studio M2 Ultra (192 GB) Backfill** of `exact-fast`, `k8v4-clean`, `kivi4-clean`, `fp8-clean` at `gen-128` & `gen-512` | `LLaDABench.swift` | Studio wall-clock TPS verification per `AGENTS.md` Rule 3.4 |

