# NeoDiffusion — Claude Code Handoff

Swift/Metal inference engine for diffusion language models on Apple Silicon, built on MLX-Swift. First target: **LLaDA2.1-mini** (`inclusionAI/LLaDA2.1-mini`), served via an OpenAI-compatible HTTP server from a Mac Studio M2 Ultra. This file is the entry point; it summarizes what is decided, what is dangerous, and where everything else lives.

## Current status (2026-07-10)

Planning is **complete**; implementation is **underway**. **M0–M5 are implemented and green on the dev toy-config FP32 path.** All design decisions below are settled with André — do not relitigate them, do raise contradictions if you find the code/reality disagreeing with them.

**M6 is dev-complete; the dev baseline is frozen** (2026-07-10, campaign record `Plans/m6-logbook.md` — read it before touching performance): 4-bit artefact verified bit-perfect and running (`models/llada2-1-mini-4bit`; BF16 source local at `models/llada2-1-mini`), `diffusion-bench llada` implements the M6 metric set. Headline numbers (M1, 4-bit, gen-128): chat ≈ 5–12 tok/s, reasoning ≈ 9–17, code ≈ 12–22; steady forward ≈ 0.34 s; peak 9.57 GB. Key facts: under memory pressure, macOS **paging** — not inference compute — causes a ~50 s/forward pathology (the bench auto-labels every JSONL row `envValid` with swap/free/thermal telemetry; conclusions come from valid rows only); `gatherQuantizedMM` is the fastest MoE dispatch *under the measured conditions* (M1, mlx-swift 0.31.6, real shapes — re-run `LLaDAMoEDispatchBench` after any MLX/GPU/shape change before trusting the ranking); per-process warmup ≈ 19 s (rows carry a `warmupIncluded` flag); `.strict` vs `.referenceBias` quality evidence is smoke-level qualitative (not a gate — M8 promotes it to blind paired scoring). **M7 (server) is implemented** (André, 2026-07-10). **M8's M1-scoped sweep is COMPLETE** (2026-07-10, record: `Plans/m8-logbook.md`): shipped default stays the **uniform 4-bit g64 artefact** — 4-bit drift is real (~2.7% confident-position flips vs a streamed-BF16 reference, measured with its own noise-floor control) but none of the swept axes buys it back: lm_head 4-bit REJECTED (fails the margin gate — opposite of Sumi; never port quantization verdicts between models), g32/6-bit experts REJECTED (no improvement ⇒ experts aren't the driver). lm_head stays 16-bit — the §2.7 open item is closed. Outstanding M8 item: André's blind scores for the strict-vs-referenceBias sheet (`scratch/m8_blind/sheet.md`; evaluate with `Tools/m8_blind_sheet.py --score`). Toolchain quirks: run `Tools/seed-metallib.sh` after builds (this toolchain's `swift build` does not produce the Cmlx metallib), and never partially delete `.build` (`build.db` poisoning → silent mis-links). Next: **Phase 3, dev-only under the §0.1 protocol** in the phase-3 roadmap (hardware-independent metrics decide; wall-clock is host-scoped; Studio backfills recorded arms). **Phase 3 status (2026-07-10 night): WP-1a Elastic-Cache CLOSED — negative result** (`Plans/elastic-cache-logbook.md`; active-KV reuse has ~zero ceiling on block-causal LLaDA2.x and inflates steps/block monotonically; serving-path drift-instrumentation leak fixed, 660a9a7); **WP-1b MultiBD underway** (branch `wp-1b-multibd`, record `Plans/wp1b-logbook.md`). Studio debts: BF16 parity (M4/M5), M6 both-machines gate, M8 scored-set comparison. **Read `Plans/handoff-post-M5.md` first** for file locations and gotchas.

## Required reading (in this order)

| Document | What it holds | Read when |
|---|---|---|
| `Plans/handoff-post-M5.md` | **Start here if resuming.** Post-M5 handoff: where every file lives (incl. the cached reference `.py`), facts verified this milestone, what `DiffusionEngine.Output` already gives M6, and the concrete gotchas/backtracks (MLX operator traps, speculation-vs-cache-commit, step-count off-by-one) | Before M6, or any real-weight run |
| `Plans/phase-2-implementation-guide.md` | **The build plan.** Pinned reference algorithm (parity target), per-module contract with HF weight names + dtype rules, milestones M0–M8 with acceptance gates (M4/M5 entries carry current status), exhaustive deviations-from-reference list | Before writing any code |
| `Plans/phase-1-conceptual-design.md` | Architecture rationale: decision record, package responsibilities, denoising-loop state machine, GPU-residency rules, streaming semantics, cache design, §13 addendum (raw-Metal trade-off, escape-hatch trigger) | Before M3–M5; whenever a design question arises |
| `Plans/phase-3-optimisation-roadmap.md` | Post-baseline optimisation tiers, composability matrix, kernel-fusion track | Only after M6's baseline is frozen; do not start Phase 3 work early |
| `Plans/Optimisations.md` | Historical concept survey — **superseded by phase-3 doc** for ordering; keep for context only | Optional |

External references (fetch, don't trust memory):
- `https://huggingface.co/inclusionAI/LLaDA2.1-mini` — `config.json`, `configuration_llada2_moe.py`, `modeling_llada2_moe.py` (the ground-truth reference implementation; the phase-2 doc §1 pins its `generate()` semantics), `model.safetensors.index.json` (weight-name verification, M1 gate), `generation_config.json` + `tokenizer_config.json` (special-token audit, M2 gate).
- André's research wiki (separate Obsidian vault, "Diffusion") — concept/proposal pages behind the design, notably `04-Proposals/neodiffusion-inference-engine.md`, `04-Proposals/elastic-cache-metal-kernel.md`, `02-Sources/mbd-lms.md`. Not required for Phase 2 coding; required before Phase 3 work packages. Ask André for access if needed.

## Fixed decisions (summary — details in phase-1 §1 and §13)

- **Substrate**: MLX-Swift arrays + lazy eval; hot paths later as custom Metal kernels *via MLX custom ops*. No raw command-buffer management in Phase 2. A raw-Metal port of loop control only happens if a measured trigger fires (phase-1 §13.1: loop overhead >10–15% of step time after Phase 3 fusion).
- **Hardware**: dev = MacBook Pro M1 16 GB (4-bit only, toy-config unit tests); correctness + serving = Mac Studio M2 Ultra 192 GB (BF16 parity runs live here).
- **Precision**: 4-bit MLX affine quant (group 64) is the primary format; BF16 is the correctness anchor. Never quantize: router (FP32), embeddings/norms/shared experts (16-bit). `lm_head` quantization is an open sweep item (M8), not a default.
- **Model scope**: LLaDA-family config-driven abstraction. Not a general dLLM framework.
- **Server**: Hummingbird 2, `/v1/chat/completions` (+SSE), single-request, block-commit streaming. Draft-preview streaming is deferred (feasible via SSE custom events; don't build it unasked).
- **Loop shape**: the denoising loop is **Block-Buffer-shaped from day one** — fixed slot array, states `dummy → active → toCache → inCache`, `N_buf = 1` hardwired through all of Phase 2. N_buf=2 (MultiBD) is Phase 3.
- **Two caches, two contracts**: `ExactPrefixCache` (prompt + committed blocks; exact by block-causality; append-only; never policy-managed) vs `ActiveBlockCache` (within-block; approximate; the only tier Phase 3 staleness policies may touch). Keep them separate *types*.
- **Excluded from scope**: multi-block editing (MBE), training-time concepts (MultiTF, EBPO, MTF), DID/text-VAE families, hybrid AR decoding, batching across requests (until Phase 3 proves single-stream).

## Target model quick facts (sourced from config.json, 2026-07-04)

20 layers (layer 0 dense FFN int. 5120; layers 1–19 MoE) · hidden 2048 · GQA 16 Q / 4 KV heads, head_dim 128 · **qk-norm ACTIVE** (per-head RMSNorm on Q,K before RoPE — default `use_qk_norm=True` from `configuration_llada2_moe.py`, not overridden by config.json) · partial RoPE (first 64 of 128 dims, θ=600000, FP32 freqs, absolute positions) · MoE: 256 experts (int. 512) + 1 shared, 8 active/token, FP32 sigmoid router + expert-bias, group-limited top-k (8 groups, top-4, top-2-sum group score), weight-normalize then ×2.5 · vocab 157184, embeddings untied · RMSNorm ε=1e-06 · `mask_id=156895`, `eos_id=pad=156892` · decoding: block 32; Q Mode τ=0.7/0.5, S Mode τ=0.5/0.0, `max_post_steps=16`.

## Critical gotchas (each one has bitten or will bite)

1. **FP32 is load-bearing** in: router matmul + sigmoid, RMSNorm internals, attention softmax, RoPE frequency computation, final logits (+ confidence softmax). Confidence thresholds were tuned against fp32 softmax — do not fp16 these paths.
2. **Config defaults**: keys absent from `config.json` take the Python-class defaults from `configuration_llada2_moe.py`. The Swift config struct must encode the full default table (e.g. `use_qk_norm=True` applies because it's absent; `rms_norm_eps` default 1e-05 is overridden to 1e-06).
3. **Fused QKV**: one `query_key_value` weight, split as [16 Q, 4 K, 4 V] heads on the head axis after reshape to `[B, L, 24, 128]` — not three projections.
4. **Reference quirks not to copy**: `generate()` forces `output_attentions=True` (drops HF to eager attention — ignore, use SDPA); the MoE `moe_infer` does `.cpu().numpy()` per step (a CPU sync — use `gather_qmm`); `steps`/`minimal_topk` params are dead code; the reference has **no KV cache at all** (recomputes full prefix each step). Every deviation must appear in phase-2 §5's list.
5. **Parity gates**: BF16 on the Studio must match the reference **token-for-token** (temp 0, ≥20 prompts, S/Q modes, eos_early_stop on/off) with the prefix cache *disabled*, then *identical* with it enabled, before anything else lands. The 4-bit path is held to task-level quality only, never token parity.
6. **Commit cleanliness**: block KV captured at commit is final-token-consistent only if the loop exited via "no changes". If it exited via the post-step budget (`post_steps > max_post_steps` right after an edit), run one extra forward before capturing KV.
7. **M2T/T2T in one forward**: Γ (masked, conf > τ_mask, ≥1 guaranteed via top-k fallback) and Δ (unmasked non-prompt in active block, conf > τ_edit AND token changed) come from the *same* sampled `x0/x0_p`. `max_post_steps` counts iterations where the block has no masks left; refinement never touches committed blocks.
8. **Prompt tail shares a block with generation start** (`prefill_blocks = prompt_len // 32`); prompt positions inside that block are edit-protected; position ids are absolute over the padded total length.
9. **No per-step CPU readbacks**: selection sets, thresholding, state updates are MLX array ops; loop control uses K-step speculative execution with an async "done" flag. `diffusion-bench` counts sync points — the budget is ≤1 blocking readback per K steps + 1 per block commit.
10. **Signature defaults ≠ served defaults**: reference `generate` defaults are 0.95/0.9; the served modes are the model card's Q (0.7/0.5) and S (0.5/0.0). Audit `generation_config.json` in M2.
11. **`sliding_window: 4096`** is wired through but inert (eager ignores it, FA2 disabled, ≤4k context). Implement nothing; revisit only for >4k goals.

## Working rules

- **Milestones in order, gates are hard**: M0 plumbing → M1 config/weights/quant → M2 tokenizer → M3 core blocks → M4 forward parity → M5 loop+cache → M6 bench baseline → M7 server → M8 quant sweep. Acceptance criteria are in phase-2 §4; do not start Mn+1 with Mn red.
- **Fixture-based testing**: a one-off Python script on the Studio dumps reference intermediates/logits/generation traces (incl. per-step Γ/Δ) as safetensors; Swift tests diff against fixtures and never require PyTorch at runtime.
- **Provenance discipline** (house style, from the wiki): mark claims **sourced / inferred / speculative** in docs and non-trivial code comments; unsourced performance assumptions are bugs.
- **Record negative results**: failed approaches go into the Plans docs (and the wiki via André), not into deletion.
- **André's preferences**: Swift 6 / SwiftPM; hexagonal architecture (Generation = application core; kernels/model-IO/tokenizer/server = adapters behind ports — this mapping is phase-1 §4); ask questions rather than assume when information is missing; concise communication.
- **When you deviate from the plans**: update the relevant Plans doc in the same change, and flag it to André. The Plans docs are the source of truth, not this summary — on conflict, phase-2 wins for implementation detail, phase-1 for architecture intent.

## Environment notes

- `Package.swift`: swift-tools 6.0; deps mlx-swift ≥0.29.0, swift-transformers ≥1.1.1, swift-numerics. **To add in M0**: Hummingbird 2 (server target only); **to remove**: iOS platform entry (non-target).
- Weights: both live locally on the dev M1 now — BF16 source `models/llada2-1-mini` (~33 GB, storage only; BF16 *inference* still needs the Studio) and the converted 4-bit artefact `models/llada2-1-mini-4bit` (9.5 GB, verified bit-perfect 2026-07-10, 16-bit tensors stored F16 — see phase-2 §3 status). Conversion script: `Tools/convert_weights_streaming.py` (streaming, dev-M1-safe; quant predicate per the keep-list above).
- `Tools/diffusion-bench` implements the M6 metric set (TPS, TPF honest/logical, steps/block, post-steps/block, **sync-point count**, peak memory, per-phase wall-clock) as `diffusion-bench llada`, plus the Sumi-campaign mode; prompt suites are checked in under `PromptSuites/`. See its README (incl. the release-build metallib copy quirk).
- Tests live in `Tests/` (per-package) — currently empty scaffolds.
