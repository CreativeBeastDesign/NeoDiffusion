# NeoDiffusion Phase 1 — Conceptual Design

**Status**: draft, awaiting review
**Last updated**: 2026-07-04
**Scope**: Conceptual architecture of the inference engine. No implementation detail beyond what is needed to fix interfaces and phase boundaries. Phase 2 (Swift/Metal implementation guide) and Phase 3 (optimisation roadmap) build on this document after review.
**Provenance discipline**: claims are tagged **sourced** (traceable to a paper/model card/wiki source note), **inferred** (reasoned from sourced facts), or **speculative**.

---

## 1. Decision record

Fixed by interview on 2026-07-04; changing any of these invalidates parts of this document.

| Decision | Choice | Consequence |
|---|---|---|
| Substrate | MLX-Swift arrays + lazy eval; custom Metal kernels on hot paths via MLX custom-op interface | No raw command-buffer management in Phases 1–2; abstractions must not preclude a later raw-Metal path |
| Serving hardware | Mac Studio M2 Ultra 192 GB | BF16 fits trivially; bandwidth ~800 GB/s class (**inferred**, Apple spec-sheet class figure — verify in Phase 2) |
| Dev hardware | MacBook Pro M1 16 GB | 4-bit quantization is the *first* weight format, not an afterthought |
| Precision order | 4-bit (MLX affine quant) first, BF16 later | Correctness baseline is initially quant-confounded; BF16 parity check happens on the Studio |
| Model scope | LLaDA2.1-mini first; abstraction sized for the LLaDA family, not all dLLMs | MoE support is baseline infrastructure, not an optimisation |
| Interface | Swift library + `diffusion-bench` CLI; OpenAI-compatible HTTP server on top | Streaming semantics must be defined for block-diffusion output (§7) |
| Workload | Interactive chat/reasoning ≤4k context first; coding later | Long-context eviction methods (MaskKV, Sparse-dLLM) are Phase 3 extras, not core; but interfaces must not hardcode short context |
| Phase 3 priority | Elastic-Cache ≈ **MultiBD (co-first, confirmed 2026-07-04)** ≥ Spiffy ≈ Streaming-dLLM > d²Cache > LocalLeap ≈ FreeDave > Sparse-dLLM | §8 maps extension points in this order; full roadmap in `phase-3-optimisation-roadmap.md` |

## 2. Target model: LLaDA2.1-mini

Facts from the HF model card and `config.json` (**sourced**, huggingface.co/inclusionAI/LLaDA2.1-mini, retrieved 2026-07-04):

| Property | Value |
|---|---|
| Architecture | `LLaDA2MoeModelLM` (MoE diffusion LM, draft-and-edit) |
| Total params (non-embedding) | 16 B |
| Layers | 20 (layer 0 dense FFN, layers 1–19 MoE; `first_k_dense_replace=1`) |
| Hidden size | 2048 |
| Attention | GQA: 16 query heads, 4 KV heads, head_dim 128 |
| RoPE | partial (rotary_dim 64 of 128, `partial_rotary_factor=0.5`), θ = 600 000 |
| Dense FFN intermediate | 5120 (SiLU) |
| MoE | 256 routed experts (intermediate 512) + 1 shared expert; 8 active/token; sigmoid score, group-limited routing (8 groups, top-4 groups), `routed_scaling_factor=2.5`, expert bias enabled, router in FP32 |
| Vocab | 157 184; embeddings untied |
| Context | 32 768; norm RMSNorm (ε=1e-6) |
| Recommended decoding | `block_length=32`, temperature 0.0; Q Mode: `threshold=0.7, editing_threshold=0.5`; S Mode: `threshold=0.5, editing_threshold=0.0`; `max_post_steps=16`; output length up to 16 384 |

Decoding paradigm (**sourced**, [[llada2-1-tech-report]]): block diffusion with draft-and-edit. Per step, tokens with confidence above `τ_mask` are unmasked ([[mask-to-token-m2t]]); already-drafted tokens whose replacement confidence exceeds `τ_edit` are edited ([[token-to-token-t2t]]). Dual thresholds give S Mode / Q Mode. Attention is block-wise causal ([[block-wise-causal-attention]]): clean prefix blocks are causal context; the active block attends bidirectionally.

Implication for the engine (**inferred**): block-causality makes prefix KV exactly reusable — the KV cache for completed blocks is *valid by construction*, unlike full-bidirectional dLLMs where all caching is approximate. This is the single most engine-shaping property of the model. It is also why Spiffy-style verification is exactly lossless on block-causal models (**sourced**, [[spiffy]]).

## 3. Why an AR engine design does not transfer

AR decode is memory-bandwidth-bound: one token per forward pass, weights streamed per token. dLLM block decode runs a full forward over the active block (+ suffix representation) per denoising step, amortizing weight reads over `block_length` positions but paying multiple steps per block (TPF on the model card: ~2–12 tokens per forward, task-dependent, **sourced**). Consequences (**inferred**):

- The optimisation target is *steps per block × cost per step*, two independent levers. Phase 3 methods split cleanly: fewer steps (threshold calibration, early exit, Spiffy) vs cheaper steps (Elastic-Cache, suffix pruning, MoE kernels).
- Per-step CPU↔GPU synchronization is proportionally more damaging than in AR engines because each step is a *decision point* (which tokens crossed threshold?) that naively forces a readback. §6 makes the decode loop GPU-resident.
- KV storage is small (GQA, 4 KV heads: 2 × 4 × 128 × 20 layers × 2 B = 40 KB/token BF16 → ~168 MB at 4k context, **inferred** from config). Cache *recomputation policy*, not cache *size*, is the cost centre at our workload. Long-context coding workloads change this later.

## 4. System architecture

Keeps the existing four-package split, adds two products. Hexagonal reading: `DiffusionGeneration` is the application core; kernels, model IO, tokenizer, and server are adapters behind ports.

```
┌────────────────────────────────────────────────────────┐
│  NeoDiffusionServer (OpenAI-compat HTTP)   diffusion-bench (CLI)
├────────────────────────────────────────────────────────┤
│  DiffusionGeneration — application core                 │
│   • DenoisingLoop (block state machine, GPU-resident)   │
│   • DecodingPolicy (τ_mask/τ_edit, S/Q mode, sampler)   │
│   • CacheManager port  • Scheduler port  • Streamer     │
├────────────────────────────────────────────────────────┤
│  DiffusionModel — model adapter                         │
│   • config parsing, safetensors mmap loading            │
│   • quantization (4-bit load path), tokenizer glue      │
│   • LLaDAFamilyModel: builds the layer stack            │
├────────────────────────────────────────────────────────┤
│  DiffusionCore — computation blocks                     │
│   • BlockCausalAttention (GQA, partial RoPE)            │
│   • MoELayer (sigmoid grouped router + gathered GEMM)   │
│   • RMSNorm, embeddings, logits head                    │
├────────────────────────────────────────────────────────┤
│  DiffusionKernels — Metal adapter                       │
│   • MLX custom ops; later: fused kernels per §8         │
└────────────────────────────────────────────────────────┘
```

Package responsibilities match the existing `*Content.md` stubs; the two additions are the server target and making `CacheManager` a port owned by Generation rather than a detail inside Core.

### 4.1 DiffusionModel
- Weights: mmap safetensors; 4-bit MLX affine quantization (group size 64 initial guess, **speculative** — quality/speed sweep is a Phase 2 task) applied at load or offline-converted. Router weights stay FP32 (**sourced**, `router_dtype: fp32` in config); embeddings and norms stay BF16/FP16 (**inferred**, standard practice — validate).
- Tokenizer: `swift-transformers` `Tokenizers` with the model's chat template. Detokenization must handle *revision*: an edited token invalidates previously produced text (§7).
- `LLaDAFamilyModel` protocol: config-driven layer stack so LLaDA2.1-flash (and future family members) differ by config, not code. Non-LLaDA dLLMs (Dream etc.) are out of scope by decision; the protocol is not designed for them.

### 4.2 DiffusionCore
- **BlockCausalAttention**: one attention implementation with three input classes — cached prefix KV (read-only), active-block KV (recomputed per step), suffix representation (initially: full masked-suffix attention as the reference; the suffix window/pruning variants are Phase 3 swaps). Partial RoPE applied to the first 64 dims of each head (**sourced**, config).
- **MoELayer**: FP32 sigmoid router with group-limited top-k (8 groups over 256 experts, top-4 groups, top-8 experts total, + shared expert, scaling 2.5 — all **sourced** from config); expert compute via MLX quantized gathered matmul in Phase 2, custom fused dispatch in Phase 3. Routing is per-token — a 32-token block step routes 32×8 expert calls; batching tokens by expert is the known hot spot (**inferred**; the [[alpha-moe-megakernel]] source note reaches the same conclusion for CUDA).
- Layer 0's dense FFN is a separate code path (**sourced**, `first_k_dense_replace=1`).

### 4.3 DiffusionGeneration
- **DenoisingLoop**: owns the per-block state machine (§5) and the GPU-residency rules (§6).
- **DecodingPolicy**: pure function of (logits, state) → (unmask set, edit set); threshold defaults from the model card; the Phase 3 calibration cluster (one-shot/dynamic thresholding) replaces this implementation behind the same interface.
- **CacheManager port**: owns *two distinct caches with different correctness contracts* — this two-tier split is well anchored in the wiki, it is Fast-dLLM's [[hierarchical-caching]] structure ([[block-level-cache]] / [[sub-block-cache]], **sourced**):
  - **ExactPrefixCache** (prompt + committed blocks): exact by construction of block-causal attention; append-only at block commit; never subject to staleness policy.
  - **ActiveBlockCache** (within-block generation state): approximate by nature — tokens are still changing, so any reuse across denoising steps is a policy decision, not a guarantee. This is the *only* cache the Phase 3 staleness machinery (Elastic-Cache drift test, token-stability, refresh depth) ever touches.
  The loop code sees one port; the two caches are separate types so an optimisation cannot accidentally apply an approximate policy to the exact tier.
- **Streamer**: block-commit semantics (§7).

### 4.4 Products
- `diffusion-bench`: fixed prompt sets; reports TPS, TPF (tokens per forward), steps/block, wall-clock per phase, peak memory, and a sync-point count (number of forced CPU readbacks per generated block — the metric for §6's goal). Harness metrics deliberately mirror the wiki's [[elastic-cache-metal-kernel]] proposal so Phase 3 ablations reuse it.
- **NeoDiffusionServer**: `/v1/chat/completions` (+streaming). Single-request first (`max-running-requests 1`, matching the SGLang reference deployment for this model, **sourced**); batching is out of scope until Phase 3 proves single-stream performance.

## 5. The denoising loop (conceptual)

Per generation: prefix pass over the prompt fills the prefix KV cache once ([[prompt-caching]] in its trivial form). Then per block:

1. Initialize active block as 32 `[MASK]` tokens appended after committed context.
2. Repeat until block complete or step budget:
   a. Forward pass: active block (+ suffix representation) against cached prefix KV.
   b. Confidence extraction from logits (temperature 0 → argmax prob).
   c. M2T: unmask tokens with confidence > τ_mask (at minimum the single most-confident token, guaranteeing progress — **sourced** behaviour of threshold decoding, [[configurable-threshold-decoding]]).
   d. T2T: re-predict already-drafted tokens; replace where replacement confidence > τ_edit (Q Mode; τ_edit=0 disables in S Mode) — bounded by `max_post_steps` (**sourced** decoding params).
   e. Update block state in place.
3. Commit block: append its KV to prefix cache; emit tokens to the Streamer.
4. EOS handling: high-confidence EOS terminates remaining blocks (cheap early exit; the fuller [[early-exit-block-diffusion]] variant is Phase 3).

**Resolved (review rounds 1–2, 2026-07-04, verified against `modeling_llada2_moe.py`)**: the reference computes both selection sets from *one* forward per step — Γ (masked positions, confidence > `threshold`, minimum `num_to_transfer=1` via top-k fallback) and Δ (unmasked non-prompt positions in the active block, confidence > `editing_threshold` AND predicted token ≠ current token) from the same sampled `x0/x0_p`. M2T-only mode = τ_edit disabled. `max_post_steps` semantics (**sourced**, reference code): `post_steps` counts iterations in which the active block has no remaining masks; the loop breaks when `post_steps > max_post_steps` or when a step produces no unmasks and no edits. Despite the docstring's word "global", refinement only ever touches the active block (`cur_x[:, -block_length:]`) — committed blocks are never revisited, so block-boundary streaming (§7) is safe by construction.

Multi-block editing ([[multi-block-editing-mbe]]) is *not* in the Phase 1 loop: it invalidates committed prefix KV and breaks streaming commitments. Deferred; noted as required if coding workloads later demand it (**inferred** trade-off, [[llada2-1-tech-report]] reports MBE matters most for code/math).

**Loop shape amendment (2026-07-04, from [[mbd-lms]])**: the state machine above is *buffer-shaped* rather than single-block-shaped. The loop owns a fixed array of block slots with lifecycle `dummy → active → toCache → inCache` ([[block-buffer]]); `N_buf = 1` reproduces the reference SingleBD algorithm exactly and is the parity mode for all Phase 2 gates. `N_buf = 2` enables training-free [[multi-block-diffusion-decoding]] in Phase 3 — measured on LLaDA2.1-Mini itself at TPF +44% / −0.59pp (math), the strongest directly-applicable optimization number we hold (**sourced**, [[mbd-lms]]). This costs one level of indirection now and avoids restructuring the loop later; front-block in-order commit keeps §7's streaming semantics intact.

## 6. GPU residency and hop minimisation

Design rule: **the CPU never learns anything about a block until it commits, except an occasional tiny "done yet?" flag.**

- Steps 2a–2e run entirely as MLX graph operations; thresholding, argmax, unmask-set construction, and state update are array ops, never Swift-side loops over logits (**inferred** from MLX's lazy-evaluation model; standard MLX practice).
- Loop control without per-step readback: run K steps speculatively in-graph, read back a single "block complete" scalar asynchronously (MLX async eval) every K steps, K tuned (~2–4, **speculative**). Wasted steps on overshoot are bounded and cheap relative to a synchronous stall each step.
- Unified memory means a readback is a sync/stall problem, not a copy problem — the cost is pipeline drain, not bytes (**inferred**, Apple Silicon UMA). Hence the metric is *sync-point count*, which `diffusion-bench` reports per block.
- Detokenization and streaming happen at block commit only.
- MLX already batches ops into few command buffers; the residual hop cost after this design is kernel-launch and graph-evaluation overhead, which is what Phase 3 fusion attacks (per-model fused kernels, matching your stated goal). The abstractions here (DecodingPolicy and CacheManager as pure array-op functions) keep every candidate fusion inside one graph region — that is the property that must survive review.

## 7. Streaming semantics for an editing model

A dLLM with T2T cannot stream like an AR model: emitted tokens may be revised. Policy:

- **Commit horizon = block boundary.** Only committed blocks are streamed (T2T and `max_post_steps` operate within the active block; once committed, tokens are final by construction of the Phase 1 loop — MBE would break this, which is another reason it is deferred).
- Optional "draft preview" channel: **decided nice-to-have, deferred** (review round 1). Technically it is feasible over HTTP — OpenAI streaming is SSE, and SSE permits custom event types, so revisable-token events could ride the same connection without breaking OpenAI-compat clients (which ignore unknown events). Out of scope for Phases 1–2; the Streamer interface keeps the active-block state observable so it can be added without loop changes.
- Latency shape: bursty, ~32 tokens per burst — fine for chat. Recommended output lengths for coding (16k, **sourced** model card) make burst streaming *more* attractive, not less.

## 8. Extension points for Phase 3 (ordered by your priority)

Each is an interface commitment now, an implementation later. All names are wiki concepts.

1. **Elastic-Cache cluster** — `CacheManager` port exposes per-layer, per-token refresh decisions: `shouldRefresh(layer, tokens, signals) -> RefreshPlan`. The three axes (what: [[sliding-window-attention]]; when: [[most-attended-drift]] vs [[token-stability]]; where: [[layer-wise-kv-dynamics]]/[[depth-aware-refresh]]) map to independent strategy objects, per the wiki's [[elastic-cache-metal-kernel]] proposal, whose Phase-2/3 ablation slots in here unchanged.
2. **Training-free MultiBD** (added 2026-07-04, [[mbd-lms]]) — activate `N_buf = 2` in the [[block-buffer]]-shaped loop: concurrent refinement of two blocks, τ_add/τ_semi activation gating added to DecodingPolicy. Directly measured on LLaDA2.1-Mini (TPF +44%, −0.59pp math); the TPF→TPS conversion is roofline-dependent and must be re-measured on the M2 Ultra (**sourced** numbers, **inferred** portability). Priority: **co-first with Elastic-Cache (confirmed by André, 2026-07-04)** — evidence quality comparable to S2D2, measured on the target model. Composes: per-slot ActiveBlockCache for the staleness machinery; τ_add for chat/reasoning needs tuning (source only has math 0.10 / code 0.90).
3. **Spiffy** — the loop's step 2 generalizes to *batched candidate states*: forward B draft states, verify, accept longest valid path. Requires (a) batch dim in the forward path from day one (cheap now, painful later), (b) offline calibration artefact (draft graph) loadable per model (**sourced** mechanism, [[spiffy]]). Note the precision caveat: exact losslessness holds for block-causal models but was precision-sensitive in bf16 batched-attention on LLaDA (**sourced**) — flagged for Metal numerics testing.
4. **Streaming-dLLM cluster** — suffix representation is already an interface in BlockCausalAttention (§4.2): full suffix (reference) → windowed suffix + final-token positional cue ([[attenuation-guided-suffix-modeling]]); DecodingPolicy already owns thresholds → adaptive τ(t) ([[dynamic-confidence-aware-decoding]]); early exit slots into step 4.
5. **d²Cache / LocalLeap / FreeDave / Sparse-dLLM** — all reachable through the same two ports (CacheManager for the caches, DecodingPolicy + batched forward for FreeDave's draft-verification). No additional interface commitments needed now (**inferred**).
6. **MoE kernels** — required baseline (§4.2) but the *fused* dispatch (routing + gather + GEMM + scatter in fewer launches) is Phase 3; the [[alpha-moe-megakernel]] notes are the starting point, portability explicitly unvalidated on Metal (**sourced** caveat in the wiki note).

Deliberately excluded from interface commitments: training-time concepts (EBPO, MTF), DID family, text-VAE family, hybrid AR — matching `Optimisations.md`'s out-of-scope list.

## 9. Memory and feasibility estimates (all **inferred** — verify in Phase 2)

| Item | 4-bit (M1 16 GB) | BF16 (M2 Ultra 192 GB) |
|---|---|---|
| Weights (~16.6 B incl. embeddings) | ~9–10 GB (4.5 bits/weight eff. with group scales) | ~33 GB |
| KV cache @ 4k ctx | ~0.17 GB (FP16; KV stays 16-bit even with 4-bit weights) | ~0.17 GB |
| Activations/scratch (block 32) | ~1–2 GB | ~2–4 GB |
| Verdict | Tight but feasible; requires raised wired-memory limit, no headroom for Spiffy batch>2 | Comfortable; Spiffy batching and BF16 parity work happen here |

**Quantization decision (review round 1, delegated)**: MLX affine 4-bit, group size 64 as primary format; router FP32 (**sourced**, config), embeddings/norms/shared-expert 16-bit. Rationale: it's MLX's native, kernel-supported format — zero custom quant tooling in Phase 2. Risk (**speculative**): 4-bit group-64 quality on a fine-grained MoE (256 small experts, intermediate 512) is less charted than on dense models. Mitigation is a fixed Phase 2 sweep, in order: group-64 all-quant → group-32 on routed experts → 6-bit routed experts; accept the first configuration that passes §10's task-level quality bar. No new format design under any outcome.

## 10. Correctness and validation plan

1. **Reference anchor**: HF `modeling_llada2_moe.py` + weights (to be downloaded), greedy, fixed prompts. BF16 NeoDiffusion (on the Studio) must match reference token-for-token over full generations before any optimisation is trusted. 4-bit path is compared for task-level quality, not token parity.
2. **Unit level**: each Core block (attention with partial RoPE, router, expert GEMM, RMSNorm) tested against MLX-Python or PyTorch reference outputs on random tensors.
3. **Loop level**: S/Q mode reproduce the model card's qualitative TPF ordering (S ≈ 5.3 avg TPF > Q ≈ 3.1, **sourced** card averages) — directional check, not exact.
4. **Perf harness**: `diffusion-bench` metrics from §4.4, baselined *before* Phase 3 so every optimisation has a delta. dInfer's published numbers serve as directional cross-platform reference only (different hardware).
5. **Sync audit**: Metal debugger / Instruments capture confirming the decode loop produces no per-step blocking readbacks.

## 11. Open questions

Resolved in review round 1 (2026-07-04): T2T interleaving (§5 — both sets from one forward, dual mode), draft preview (§7 — deferred, SSE-feasible), quantization (§9 — group-64 primary with fixed fallback sweep), server framework (§13 — Hummingbird).

Resolved in review round 2 (2026-07-04, against `modeling_llada2_moe.py`):

- `sliding_window: 4096` is *wired through* — `LLaDA2MoeAttention` reads it and passes it to the attention interface (André's reading confirmed). Nuance: the eager path ignores the kwarg, FlashAttention-2 is disabled (`_supports_flash_attn_2 = False`), generic SDPA doesn't consume it, and at ≤4k context a 4096-token window is a no-op regardless. Verdict: live parameter, dormant effect at our workload; the engine implements it as a mask-equivalence only if/when >4k contexts become a goal (**sourced** wiring, **inferred** verdict).
- `max_post_steps` semantics: resolved, see §5. Per-block, not global; streaming safe.
- Raw-Metal escape hatch: trigger agreed, decision deferred to Phase 3 measurements (André, round 2).

Remaining (carried into Phase 2):

- `use_qk_norm` — the modeling code branches on it but `config.json` doesn't set it; the default in `configuration_llada2_moe.py` must be checked before building the attention block.
- Authoritative decoding defaults: `generate` signature defaults (threshold 0.95 / editing_threshold 0.9) differ from the model card's recommended modes (Q: 0.7/0.5, S: 0.5/0.0); check `generation_config.json` and treat the card's modes as the served defaults.
- KV-commit cleanliness: captured block KV is only final-token-consistent when the loop exits via "no changes"; exit via post-step budget requires one extra forward at commit (§5 of the Phase 2 guide).

## 12. Relation to the wiki

This document is the engineering counterpart of wiki proposal [[neodiffusion-inference-engine]] (04-Proposals). Concept-level claims live in the wiki; this file owns engineering decisions. The wiki's [[elastic-cache-metal-kernel]] proposal becomes a Phase 3 work package unchanged.

## 13. Addendum — review round 1 (2026-07-04)

### 13.1 Raw-Metal-first vs MLX + custom kernels: trade-off record

Question raised in review; decision unchanged (MLX + custom kernels, raw-Metal path kept open). The trade-offs, recorded so the Phase 3 escape-hatch decision is made against criteria rather than mood:

**What raw-Metal-first would buy** (all **inferred** from Metal platform capabilities):
- *GPU-driven loop control*: indirect command buffers let the GPU encode its own next denoising step, eliminating even the K-step async flag readback of §6 — the strongest possible version of the hop-reduction goal. MLX does not expose ICBs.
- *Full fusion freedom*: whole-step megakernels (forward → confidence → threshold select → state update in one or two dispatches), threadgroup-memory and simdgroup-matrix tuning per kernel, no graph-evaluation or op-scheduling overhead at all.
- *Residency and memory control*: MTLHeaps, explicit wired residency for weights, no allocator behavior you don't own.
- *No dependency risk*: no mlx-swift API churn; profiling traces map 1:1 to code you wrote.

**What it would cost**:
- *Rebuilding the substrate*: quantized GEMM (MLX's 4-bit kernels are heavily tuned — matching them is months, not weeks), gathered/grouped GEMM for 256-expert MoE, attention, RoPE, RMSNorm, softmax, safetensors loading, a quantization format and converter, broadcasting/layout infrastructure. All before the first end-to-end token.
- *Correctness isolation*: with MLX, a wrong output is diffed against MLX-Python ops layer by layer; raw-Metal has no on-device reference substrate, so every numerical bug is investigated from scratch.
- *Delayed validation*: Phase 3's research value (which wiki concepts actually pay off on this hardware) is gated on a working engine; raw-Metal-first pushes that back by the substrate-rebuild time.

**Why the middle path holds**: MLX custom ops *are* raw Metal kernels — MSL you write, with MLX handling buffers, streams, and everything around them. Kernel-level control (fusion within a dispatch, threadgroup strategy, simdgroup ops) is available now; what is genuinely unreachable inside MLX is ICB-style GPU-driven control flow and total residency control. **Escape-hatch trigger (proposed)**: if after Phase 3 kernel fusion the sync audit shows loop-control overhead (graph eval + flag readbacks) still >10–15% of step time on the M2 Ultra, port the decode loop (loop control only, not the substrate) to raw Metal with ICBs, keeping MLX for weights/GEMM via shared MTLBuffers (**speculative** — feasibility of buffer sharing to be checked in Phase 2).

### 13.2 Server framework: Hummingbird

Decision (delegated): **Hummingbird 2** — SwiftNIO-based, structured-concurrency-native, materially lighter than Vapor (no ORM/template/session machinery we won't use), actively maintained, supports SSE streaming responses. Hand-rolled NIO rejected: HTTP parsing/keep-alive/TLS plumbing is undifferentiated effort. Vapor remains the fallback if Hummingbird's ecosystem turns out to miss something specific (unlikely for two endpoints). (**inferred** recommendation; framework performance differences are noise next to model inference cost — "light-weight" is the operative criterion.)
