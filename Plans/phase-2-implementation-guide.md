# NeoDiffusion Phase 2 — Implementation Guide (Swift/Metal via MLX)

**Status**: draft
**Last updated**: 2026-07-04
**Prerequisite**: `phase-1-conceptual-design.md` (all §1 decisions + §13 addendum assumed fixed).
**Primary source**: `modeling_llada2_moe.py` and `config.json` from `inclusionAI/LLaDA2.1-mini` (retrieved 2026-07-04). Everything in §1–§2 below is **sourced** from that code unless marked otherwise.

---

## 1. The reference algorithm, pinned

What `generate()` actually does (this is the parity target — the engine must reproduce it token-for-token at temperature 0 before deviating):

```
steps = min(steps, gen_length // minimal_topk)        # computed, then NEVER USED — dead code
total_length = ceil((prompt_len + gen_length) / 32) * 32
x = [mask_id] * total_length;  x[:prompt_len] = prompt
attn_mask = block-lower-triangular(num_blocks) expanded to token level
            # within-block bidirectional, cross-block causal, block granularity 32
position_ids = arange(total_length)                    # absolute positions

for block in prefill_blocks .. num_blocks:             # prefill_blocks = prompt_len // 32
    cur_x = x[: (block+1)*32]
    post_steps = 0
    loop:
        active_mask = (last 32 tokens of cur_x == mask_id)
        if !any(active_mask): post_steps += 1
        if post_steps > max_post_steps: break

        logits = forward(cur_x, attn_mask, position_ids)   # FULL prefix recomputed every step
        x0, x0_p = sample(logits[last 32])                 # temp 0: argmax + softmax prob

        # M2T (Γ): masked positions with x0_p > threshold;
        #          if fewer than num_to_transfer (=1), take top-k by confidence instead
        # T2T (Δ): unmasked AND non-prompt positions in active block,
        #          x0_p > editing_threshold AND x0 != current token
        cur_x[last 32][Γ ∪ Δ] = x0[Γ ∪ Δ]

        if !any(active_mask) and Δ empty: break            # settled

    commit cur_x into x
    if eos_early_stop and block fully unmasked and contains eos_id: break

trim output at first eos_id
```

Facts with engineering consequences:

- **One forward, both selection sets** — Γ and Δ computed from the same `x0/x0_p`. Matches the review-round-1 decision exactly.
- **Refinement is per-block, not global.** Docstring says "global refinement iterations"; code edits only `cur_x[:, -block_length:]`. Committed blocks are immutable → block-boundary streaming is exact.
- **No KV cache anywhere.** The reference recomputes the entire prefix each step (`use_cache: false` in config; the `DynamicCache` plumbing in `forward` is unused by `generate`). ExactPrefixCache is NeoDiffusion's first deviation — mathematically exact under the block-causal mask, but it must be *proven* equivalent in M5, not assumed.
- **`generate` passes `output_attentions=True`** and never uses the attention weights. On HF this forces the eager attention path (SDPA is bypassed when attention weights are requested) — a reference inefficiency we do not copy. Numerically eager and SDPA agree to float tolerance; parity tests must therefore compare with tolerance-aware token matching, not bitwise logits.
- **Special tokens**: `mask_id = 156895`, `eos_id = 156892` (equal to `pad_token_id`). Verify against `generation_config.json`/`tokenizer_config.json` in M2 (**open**).
- **Signature defaults ≠ card recommendations**: 0.95/0.9 in code vs Q Mode 0.7/0.5, S Mode 0.5/0.0 on the card. The served defaults are the card's modes; the bench must be able to set all of (`threshold`, `editing_threshold`, `max_post_steps`, `num_to_transfer`, `eos_early_stop`).
- **Prompt tail shares a block with generation start** (`prefill_blocks = prompt_len // 32`, remainder lives in the first generated block; `prompt_mask_in_block` shields prompt positions from editing). The engine must reproduce this, including absolute `position_ids` over the padded `total_length`.
- **`sliding_window`** is passed to the attention interface but has no effect in the eager path, FA2 is disabled, and a 4096 window at ≤4k context is inert. Implement nothing in Phase 2; revisit if >4k contexts become a target.

## 2. Model internals checklist (per-module contract)

Each module below lists: computation, dtype rules, and the HF weight names to map. Weight names are from the module tree in `modeling_llada2_moe.py`; confirm exact keys against `model.safetensors.index.json` at download time (M1 acceptance).

### 2.1 Embeddings
- `model.word_embeddings.weight` [157184, 2048]; `pad_token_id=156892` as padding idx. Untied from output head.
- Keep 16-bit (Phase 1 §9 decision).

### 2.2 RMSNorm (used as input/post-attention/final norm and optional qk-norm)
- Compute in **FP32 internally** (cast in, rsqrt(mean(x²)+1e-6), cast back, then scale) — matches reference exactly; do not use a fused fp16 variant that skips the fp32 upcast.
- Names: `model.layers.{i}.input_layernorm.weight`, `.post_attention_layernorm.weight`, `model.norm.weight`.

### 2.3 Attention
- **Fused QKV**: `model.layers.{i}.attention.query_key_value.weight` [(16+2·4)·128 = 3072, 2048], no bias (`use_qkv_bias=false`). Split order: 16 Q heads, 4 K heads, 4 V heads (dim −2 after reshape to [B, L, 24, 128]).
- **QK-norm: ACTIVE** (**resolved 2026-07-04**, André read `configuration_llada2_moe.py`: `use_qk_norm=True` default; `config.json` omits the key, so the default applies). Per-head-dim RMSNorm (dim 128, eps `rms_norm_eps`) applied to Q and K *before* RoPE. Expect `model.layers.{i}.attention.query_layernorm.weight` / `.key_layernorm.weight` in the checkpoint; treat their absence in `model.safetensors.index.json` as a contradiction to resolve, not a reason to skip the norm.
- **Partial RoPE**: rotary on the first 64 dims of each 128-dim head (llama-style rotate-half on the 64-dim slice, passthrough on the rest). Frequencies computed in FP32 (`θ=600000`, dim 64); cos/sin cast to activation dtype after. Absolute position ids.
- **Scores**: scale = 128^-0.5; additive block-diffusion mask; **softmax in FP32** (reference upcasts); GQA via 4→16 KV head expansion (MLX SDPA handles GQA natively — verify it reproduces `repeat_kv` semantics).
- Output: `model.layers.{i}.attention.dense.weight` [2048, 2048], no bias.
- Quantize QKV/dense per §9 policy (group-64).

### 2.4 Dense FFN (layer 0 only)
- `model.layers.0.mlp.{gate_proj,up_proj,down_proj}.weight`, intermediate 5120, SiLU: `down(silu(gate(x)) * up(x))`.

### 2.5 MoE block (layers 1–19)
- **Router** (`model.layers.{i}.mlp.gate.weight` [256, 2048] + `.gate.expert_bias` [256], a buffer):
  1. logits = x·Wᵀ in **FP32** (both operands cast to fp32 — keep this weight unquantized fp32);
  2. scores = sigmoid(logits);
  3. routing scores = scores + expert_bias (bias affects *selection only*);
  4. group-limited top-k: reshape scores to [tokens, 8 groups, 32]; group score = **sum of top-2 per group**; keep top-4 groups; mask others to −inf; top-8 experts over the masked scores;
  5. weights = gather(scores *without* bias, top-8 indices), normalize to sum 1 (+1e-20), multiply by 2.5.
- **Experts**: `model.layers.{i}.mlp.experts.{e}.{gate_proj,up_proj,down_proj}.weight`, intermediate 512. Implementation: MLX quantized gathered matmul (`gather_qmm`) over the top-8 indices; do *not* port the reference's sort-by-expert CPU loop (`tokens_per_expert.cpu().numpy()` — a per-step CPU sync we must not reproduce).
- **Shared expert**: `model.layers.{i}.mlp.shared_experts.{gate_proj,up_proj,down_proj}.weight`, intermediate 512·1; always-on, added to routed output. Keep 16-bit (§9).
- Residual adds in activation dtype.

### 2.6 Output head
- `lm_head.weight` [157184, 2048], no bias; **logits cast to FP32** before confidence computation (reference does `logits.float()` — confidence thresholds were tuned against fp32 softmax, don't cheap out here).

### 2.7 Numerics summary

| FP32 | 16-bit (BF16) | Quantized 4-bit g64 |
|---|---|---|
| Router matmul + sigmoid + selection; RMSNorm internals; attention softmax; RoPE freqs; final logits + softmax/confidences | Embeddings, norms' stored weights, shared experts, activations, KV | QKV, attention dense, dense FFN (layer 0), routed expert weights, lm_head* |

*lm_head quantization is on the M8 sweep list — with a 157k vocab it is ~322 M params (~0.64 GB BF16 → ~0.18 GB at 4-bit), but confidence quality is sensitive to it; sweep before committing (**speculative**).

## 3. Environment and assets

- macOS 15+, Xcode 16+, Swift 6 toolchain (Package.swift already declares `swift-tools-version: 6.0`, platforms macOS 15/iOS 18 — drop iOS from platforms, it's a non-target).
- Dependencies: pin `mlx-swift` (≥0.29.0, already declared), `swift-transformers` (≥1.1.1, already declared), add `hummingbird` (2.x) for the server target only.
- Model: `huggingface-cli download inclusionAI/LLaDA2.1-mini` (~33 GB BF16) onto the Studio; dev machine gets only the converted 4-bit artefact (~9–10 GB) plus tokenizer files.
- Conversion: one-off Python script (mlx-lm's convert as starting point, custom quant predicate implementing §2.7's keep-list) producing MLX-format 4-bit weights + a JSON manifest recording the quant config. Swift loader consumes the converted artefact; loading raw HF BF16 directly is a Studio-only path.

  **Status (2026-07-10): done and verified.** `Tools/convert_weights_streaming.py` (streaming, one tensor in flight — runs on the 16 GB dev M1) produced `models/llada2-1-mini-4bit` (9.5 GB single-file safetensors, quant config recorded in its `config.json`). Verified against the BF16 source (which now also lives locally at `models/llada2-1-mini`): keep-list matches §2.7 exactly (router gate + `expert_bias` FP32; embeddings/norms/qk-norms/shared experts/`lm_head` 16-bit; rest 4-bit g64 affine); 12 category spot-checks **bit-perfect** vs `mx.quantize`/`astype` of the source; 0 missing / 0 unexpected tensors vs the source index. Two artefact facts to know: (1) 16-bit tensors are stored **F16, not BF16** (deviation from §2.7's wording) — lossless for these tensors (max |x| = 7.7 ≪ F16 range, no inf/nan, **sourced**: measured on the artefact) and the sensible dev dtype on M1-family Metal (§6 note), but re-emit BF16 if a path ever wants bitwise-BF16 weights; (2) packed 4-bit weights are safetensors `U8 [out, in/2]` while MLX wants `U32 [out, in·bits/32]` — `DiffusionModel.sanitize` byte-reinterprets on load (Sumi campaign quirk 3), covered by `LLaDA2QuantizedLoaderTests` (bit-exact view round-trip + toy artefact→logits round-trip).
- Reference fixtures: a Python script (runs once on the Studio, PyTorch CPU or MPS) that dumps, for a fixed set of inputs: per-module intermediate tensors (§2 unit tests), full-forward logits (M4), and full `generate` traces incl. per-step Γ/Δ sets (M5). Stored as safetensors; tests never require PyTorch at runtime.

## 4. Milestones

Each milestone has acceptance criteria; do not start the next before they pass. Estimated relative sizes given as S/M/L.

### M0 — Plumbing (S)
Package.swift: add server executable target + Hummingbird dep; remove iOS platform; test targets already exist. CI-style `swift test` green on both machines.
**Accept**: builds + empty tests pass on M1 (dev) and Studio.

### M1 — Config, weights, quantization (M)
`LLaDA2MoeConfig` Codable struct mirroring `config.json` (all fields §2 relies on). Fields absent from `config.json` take the Python-class defaults — notably `use_qk_norm=true` (confirmed); encode the full default table from `configuration_llada2_moe.py` in the Swift struct so absent-key semantics match HF exactly (e.g. `rms_norm_eps` default 1e-05 is *overridden* to 1e-06 by config.json — the struct must distinguish default from explicit). Safetensors reader (MLX built-in) + name-mapping table from §2 verified against `model.safetensors.index.json`. Conversion script + quantized-artefact loader.
**Accept**: all tensors mapped (zero unmatched keys in either direction); loaded 4-bit footprint ≤10.5 GB; config round-trips.

### M2 — Tokenizer (S)
swift-transformers tokenizer + chat template application; special-token audit (`mask_id=156895`, `eos_id=pad=156892` — cross-check `generation_config.json`, `tokenizer_config.json`).
**Accept**: encode/decode parity with HF tokenizer on a 100-case suite (multilingual, code, chat-template round-trips).

### M3 — Core blocks (L)
§2 modules in DiffusionCore, each with a fixture-diff test (rtol 1e-2 / atol 1e-3 at BF16 for 16-bit paths; fp32 paths tighter, 1e-5). Includes the block-diffusion mask builder (analytic, not the reference's O(total²) materialized tril-expansion) and MoE with `gather_qmm`.
**Accept**: every module passes fixture diffs on random-weight toy config (M1 machine) AND real weights (Studio, BF16).

### M4 — Full forward parity (M)
Stack assembly; single forward over a mixed prompt+mask sequence with block mask. **Parity is measured against a corrected `.strict` (0/-inf) reference baseline, never stock `generate()`** (§6 mask decision, deviation 8): the fixture dumper must build the model's attention mask with `-inf` on disallowed pairs — either patch the reference mask construction or drive `model.forward` directly with a hand-built mask, bypassing `generate()`. `generate_core_fixtures.py` already emits strict masks for the module fixtures; extend the same correction to the full-forward dump.
**Accept**: BF16 logits vs corrected reference fixtures: top-1 token agreement 100% over fixture set; max softmax-prob deviation within tolerance (define after first run; expect ~1e-2 at BF16). 4-bit path: top-1 agreement ≥ *measured and recorded* (no hard gate — it feeds M8's quality decision).

**Status (2026-07-07): implemented + green on the dev toy-config FP32 path.** Stack assembly wired in `LLaDA2MoeInnerModel`/`LLaDA2MoeModel` (`callAsFunction` + `logits(forTokens:blockLength:)` convenience that builds the `.strict` mask + absolute position ids). `generate_core_fixtures.py` extended with a `forward.*` dump that drives the whole stack under a strict 0/-inf mask by monkeypatching `create_bidirectional_mask` (the reference inner model unconditionally rebuilds its mask at line 877, so passing a 4D mask alone is insufficient — this is the "patch the reference mask construction" option). `ForwardParityTests.testFullForwardParity` gates top-1 100% + max |Δprob|. **First-run measurement** (toy config, seed 0, FP32, 3 blocks × 16, prompt_len 20): top-1 48/48 = 100%; max |Δprob| 8.4e-9; max |Δlogit| 2.2e-6 — i.e. fused SDPA + FP32 paths reproduce the corrected reference to ~float-epsilon. Test bound left at |Δprob| ≤ 1e-2 for BF16 headroom. The parity test is **config-driven from the fixture's `manifest.json`** (model config + block length; nothing hardcoded to the toy shape) with a `NEODIFFUSION_FIXTURE_DIR` override, so the *same* test covers both M4 paths — dev toy FP32 and Studio BF16 real weights. **Still open (Studio-only, can't run on the dev M1)**: generate the BF16 real-weight fixtures (`python3 Tools/generate_core_fixtures.py --config <hf_dir>/config.json --weights <hf_dir> --dtype bfloat16 --out <dir>`) and run `NEODIFFUSION_FIXTURE_DIR=<dir> swift test`; and the 4-bit top-1 measurement for M8.

### M5 — Denoising loop + ExactPrefixCache (L)
The §1 algorithm as MLX array ops (Γ/Δ set construction on-GPU; K-step async readback per Phase 1 §6). **Build the loop Block-Buffer-shaped from day one** (Phase 1 §5 amendment, 2026-07-04): a fixed slot array with `dummy/active/toCache/inCache` states, `N_buf = 1` hardwired for this milestone. All parity gates run at N_buf=1, which reduces exactly to the reference algorithm; the τ_add/τ_semi activation logic and N_buf=2 are Phase 3 work, but the data structures and the per-slot state machine exist now so enabling them is a config change, not a rewrite. ExactPrefixCache: committed-block KV appended at commit; **commit-cleanliness rule**: if the loop exited via "no changes", the last forward's block KV is final-token-consistent — capture it; if it exited via post-step budget (an edit happened in the final iteration), run one extra forward to capture clean KV (**inferred** from §1 semantics; verify by trace comparison).
**Accept**: (a) with cache *disabled*, temp-0 generations match the **corrected** (`.strict`-mask) reference traces token-for-token on ≥20 prompts across S/Q modes and `eos_early_stop` on/off — the reference trace dumper bypasses stock `generate()`'s 0/1 mask (§6, deviation 8); (b) with cache *enabled*, outputs identical to (a) — this is where ExactPrefixCache's block-causality exactness is proven, and it holds *because* we ship `.strict`; (c) sync audit: ≤1 blocking readback per K steps + 1 per block commit.

**Status (2026-07-07): implemented + green on the dev toy-config FP32 path.** The loop lives in `DiffusionEngine` (`DiffusionGeneration` package), driven by `GenerationParams` (S/Q presets, gotcha 10). Both selection sets come from one forward/step; Γ's top-`numToTransfer` fallback is **branchless** (`numToTransfer == 1`, the live value — a `precondition` guards >1, which needs index-exact top-k tie handling deferred to an M6 bench extension). Loop control is **K-step speculative with per-step snapshot rollback** (`speculationK`, default 4): up to K steps are built into one graph, a single stacked break flag is read back per batch, and overshoot past the break point is discarded — so output is exact for any K ≥ 1 (`testSpeculationInvariance` proves K=1 ≡ K=4). The `post_steps` accumulator is carried in-graph (no per-step readback). The loop is **Block-Buffer-shaped** (`BlockBuffer`, `nBuf == 1` hardwired via `precondition`; `dummy → active → toCache → inCache`, front-block in-order commit; `BlockBufferTests`).

ExactPrefixCache + ActiveBlockCache are **separate types** (`ExactPrefixCache` = per-layer `LayerKVCache`, append-only; `ActiveBlockCache` = the recompute-every-step Phase-2 contract, stateless stub for Phase 3 staleness policies). KV is threaded through cache-aware overloads on `LLaDA2Attention`/`LLaDA2DecoderLayer`/`LLaDA2Moe*Model` (`callAsFunction(_:cos:sin:cache:)` etc.) that forward the **active block only** against committed K/V with **no mask** (every committed key is in an allowed ≤-current block) — the existing uncached signatures are untouched, so M3/M4 stay green. **Commit-cleanliness** is realized as an unconditional **capture forward** over the final committed tokens at each commit (a strict superset of the "extra forward on budget exit" rule — it also serves as the per-block prefill and makes speculative overshoot harmless, since stale `pending` K/V are overwritten before commit). Prompt blocks are prefilled uniformly as pre-settled blocks.

**Fixtures/tests** (`Tools/generate_loop_fixtures.py`, `DenoisingLoopParityTests`): the dumper transcribes reference `generate()` verbatim under a monkeypatched strict `create_bidirectional_mask` (same technique as the M4 forward dump) and emits per-block commits, final padded `x`, trimmed output, and per-step Γ/Δ diagnostics for 16 cases (prompt lengths 8/16/20/33 → prefill_blocks 0/1/1/2 and the tail-share case, × S/Q × eos on/off). **First-run measurement** (toy config, seed 0, FP32, block 16): (a) 16/16 token-for-token; (b) ExactPrefixCache 16/16 identical to cache-disabled *and* to the reference; (c) sync budget respected (worst syncs/step 0.367 at K=4 vs ideal 0.25 — the excess is the +1 commit readback, which the budget accounts for). **Still open (Studio-only, can't run on the dev M1)**: BF16 real-weight run — `python3 Tools/generate_loop_fixtures.py --config <hf_dir>/config.json --weights <hf_dir> --dtype bfloat16 --out <dir>` then `NEODIFFUSION_LOOP_FIXTURE_DIR=<dir> swift test --filter DenoisingLoopParityTests`. (`generateCached` currently re-runs one capture forward per commit unconditionally; conditioning it on the clean-exit case is a Phase-3 micro-opt, not a correctness item.)

### M6 — diffusion-bench (S)
Metrics: TPS, TPF, steps/block, post-steps/block, sync-point count, peak memory, per-phase wall-clock (prefill/denoise/commit). JSON lines output; fixed prompt suites (chat, reasoning, code) checked into `Tools/diffusion-bench`.
**Accept**: stable numbers across 3 runs (<5% variance) on both machines; this freezes the Phase 3 baseline.
**Also record here** (first point the full engine runs): end-to-end generation quality under `.strict` vs `.referenceBias` on a handful of real prompts (per-prompt output + a perplexity/quality proxy), to confirm the strict mask is not just cleaner but not worse than stock-`generate()` numerics (§6 mask decision, item 4). Diagnostic only — not a gate.

**Status (2026-07-10): tooling implemented; measurement runs still to do.** What landed (the handoff §3 list, all five items):
- **Engine instrumentation** — `DiffusionEngine.Output.metrics` now carries per-phase wall-clock (`prefill`/`denoise`/`commit` seconds), `postStepsPerBlock` (the `post_steps` counter at loop exit, read from the already-evaluated speculative batch — no extra sync), and `forwardsEvaluated` (honest count: speculative overshoot + per-commit capture forwards + prompt prefill). Phase clocks are *real* only with `DiffusionEngine(instrument: true)`, which `eval`s at phase boundaries (≤1 extra sync per commit/prefill block; default `false` keeps M5 behavior — `DenoisingLoopParityTests` unchanged and green, and instrumentation-preserves-tokens is tested).
- **`generate(maskSemantics:)`** — `.referenceBias` is runnable cache-off only (precondition guards the cached path), feeding the §6-item-4 diagnostic (`diffusion-bench llada --mask-diagnostic`).
- **`diffusion-bench llada`** — full metric set as JSONL per (arm, suite, prompt, run) + <5% cross-run variance gate; default arms `q-cached`/`s-cached`; custom arm axes: mode, cached/uncached, mask, `--speculation-k`. Prompt suites (chat/reasoning/code, 4 chat-templated cases each) checked into `Tools/diffusion-bench/PromptSuites/`. The legacy `DiffusionGeneration` stub is deleted (bench/server no longer referenced it).
- **TPF bookkeeping**: honest TPF = tokens / `forwardsEvaluated`; logical TPF = tokens / denoising steps (run `--speculation-k 1` for reference-identical step economy). `stepsPerBlock` keeps its M5 semantics (+1 vs the reference trace on budget-break exits — handoff §4); documented in the bench README rather than changed.
- **4-bit real-weight path is live on the dev M1** (artefact + loader, see §3 conversion status), so the M6 dev-baseline arms can run locally. Reminder: those numbers are the *dev* baseline; the Phase-3 frozen baseline re-freezes on the Studio, and the 4-bit path stays task-level-only (never parity — gotcha 5).

**First real-weight run (2026-07-10, dev M1, debug build, `LLaDARealWeightSmokeTests` — opt-in `NEODIFFUSION_LLADA_REAL=1`)**: the full engine (artefact loader → chat template → `generateCached` Q mode → eos trim) produced *"The capital of Japan is Tokyo."* — correct, fluent, and **eos-trimmed** (2 blocks committed of 5 possible, `eos_early_stop` fired; the eos path, untestable at toy scale per handoff §4, works with real weights). Accounting all consistent: 4 logical steps / 10 forwards evaluated (8 denoise at K=4 + 2 captures; prompt 25 < 32 → 0 prefill blocks), 4 sync points (2 K-batches + 2 commits — budget respected), peak 9.57 GB (fits the 16 GB M1 with headroom, ≤10.5 GB M1-gate satisfied at runtime too). Wall-clock (495 s) is a **debug-build + cold-kernel artifact** — not a performance datum; all timed numbers must come from release-build bench arms (memory-noted).

**Perf anomaly resolved (2026-07-10, campaign record: `Plans/m6-logbook.md`).** The initially observed ~50 s/forward (509.6 s release sanity run) was **machine-state memory pressure**, not engine or kernel cost: on an unloaded machine the same binary does **256 tok / ~70 s ≈ 3.6 tok/s, ~0.34 s per evaluated forward** (steady-probe, 2.4% repeat deviation) — within ~2× of the naive bandwidth floor. Micro-benches (`LLaDAMoEDispatchBench`, gated `NEODIFFUSION_M6_BENCH=1`) additionally **discharge the §6 `gather_qmm`-on-M1 risk**: the production `gatherQuantizedMM` (4.5 ms/op at [256×512×2048]) beats both the dense-all-experts upper bound (0.4×) and dequant+`gatherMM` (0.2×) — the documented segmented-qmm fallback would be a regression; do not build (negative result kept as the gated bench). Forward budget: ~63% MoE, ~10% lm_head. Two operational rules: per-process warmup ≈ 19 s lands in the first block (amortized in multi-prompt runs; exclude for single-prompt timing), and steps/block is strongly content-dependent (2.0 on trivial QA vs 20.9 on essay — TPS figures are meaningless without their prompt suite).

**Dev baseline FROZEN (2026-07-10, full record + operational rules: `Plans/m6-logbook.md` §5).** Clean-run (warmup-excluded, André-approved convention) suite totals over 12 chat-templated prompts at gen-128: q-cached 108.2/109.7/129.3 s, s-cached 112.5/114.7/120.7/120.7 s. Per-suite steady rates: **chat ≈ 5–12 tok/s (≈13 steps/blk), reasoning ≈ 9–17 tok/s (≈6), code ≈ 12–22 tok/s (≈5)**; hard-content tail ≈ 3.6 tok/s (essay, 20.9 steps/blk). Peak 9.57 GB; sync budget always respected; S vs Q ≈ 10–15% fewer steps/block. **Variance gate**: within-process ≤1.0% (excellent); cross-process +6–10% thermal drift ⇒ formal <5%: s-cached **PASS** (4.0%), q-cached **FAIL** (11.7%; 6.4% with the one +117% straggler excised) — the understood laptop-thermal cause (Sumi quirk 4), not engine noise. **§6-item-4 mask diagnostic done**: 4 chat prompts uncached, `.strict` vs `.referenceBias` — 8/8 outputs coherent and of comparable quality (eyeball-tier, texts in `scratch/llada_bench.jsonl`); `.strict` ships with no visible quality cost.

**Remaining for M6 full acceptance**: the Studio re-run (both-machines gate + definitive cross-run variance off laptop thermals) — same commands, recorded in the logbook. Dev-side M6 is otherwise complete.

### M7 — Server (M)
Hummingbird 2; `POST /v1/chat/completions` (blocking + SSE streaming), `GET /v1/models`. Block-commit streaming (one SSE chunk per committed block); model loaded once at startup; single request at a time (queue, 503 or wait on overflow — pick during implementation); request-level params mapped to engine params (temperature, max_tokens→gen_length; mode selection via extension field `neodiffusion_mode: "s"|"q"` defaulting to Q).
**Accept**: OpenAI Python/JS client works unmodified against the Studio over LAN; streamed and blocking answers identical.

### M8 — Quantization sweep + BF16 baseline (M)
Per Phase 1 §9: group-64 all-quant → group-32 routed experts → 6-bit routed experts (+ lm_head quant on/off axis). Quality harness = bench prompt suites + a small scored set (e.g. GSM8K subset) at fixed seeds, compared against the Studio BF16 run.
**Accept**: first config meeting the quality bar (task-level, per Phase 1 §10) becomes the shipped default; all results recorded in the wiki as an experiment page (`05-Experiments/`).

**M1-scoped plan (agreed with André 2026-07-10 — M7 done, only dev-runnable M8 work proceeds; Studio items deferred, not dropped):**
1. **Environment validity is a precondition** (m6-logbook §5 rule 1, automated in the bench): M8 conclusions come from `envValid` rows only; every case runs on an explicitly unloaded machine with an immediate same-process rerun as the contamination check; cold-start (per-process warmup ≈ 19 s) reported separately, never averaged (rules 2/5).
2. **Deterministic 4-bit logit-drift measurement** on a compact fixed corpus (~a few hundred positions from real mid-generation canvases, per the Sumi §3.3 lesson — not synthetic probes): record top-1 agreement, top-k overlap, max/mean probability drift, and **confidence-margin changes** (the quantity the Γ/Δ thresholds actually consume) against the best M1-runnable reference: a **layer-streamed BF16 forward** (the Sumi §2 technique — stream the ~33 GB checkpoint layer-by-layer through the 16 GB machine; slow, one-off fixture dump). The M4 gate already requires 4-bit top-1 to be *measured and recorded*, not thresholded — this discharges it dev-side; the Studio re-dump later cross-checks the streamed reference itself.
3. **Strict-mask evaluation promoted beyond the M6 eyeball tier**: paired `.strict` vs `.referenceBias` outputs on the same prompts (uncached, fixed seeds), **blind-scored** (source labels stripped/shuffled; André scores coherence, instruction adherence, factual consistency where applicable) + scripted checks where objective (code executability, early-EOS/degeneration/length anomalies). Rationale: `.strict` deliberately corrects the reference script's suspected future-block masking bug (§6 deviation 8), so its quality claim deserves better than "8 coherent outputs". Current status of that claim: **smoke-level qualitative evidence, explicitly not a gate.**
4. **Confidence/trajectory diagnostics** in the engine (small `Output.metrics` extension, sync-budget-neutral — fold into existing eval batches): per-step Γ/Δ transfer/edit counts, committed-token confidence summaries, EOS block index. Purpose: distinguish "quantization changes final text" from "quantization changes denoising *dynamics*" (steps/post-steps distributions per row already land in JSONL since the telemetry extension).
5. **Sweep axes runnable on M1**: lm_head 4-bit on/off (in-memory via `MLXNN.quantize` post-load, no artefact rebuild — Sumi quirk 6), group-32 routed experts, 6-bit routed experts (both need conversion-script variants; disk and RAM fit). Each axis: one variable at a time vs the frozen g64 artefact, logit-drift metrics (item 2) + bench suites + trajectory diagnostics (item 4).
**Deferred to Studio**: the BF16 *interactive* baseline and scored-set comparison at full speed; final shipped-default decision if the M1 evidence is ambiguous.

**Status (2026-07-10): M1-scoped sweep COMPLETE — shipped default stays the uniform 4-bit g64 artefact.** Full evidentiary record: `Plans/m8-logbook.md`. Headlines: (1) 4-bit quantization drift is real and measured — ~2.7% of high-confidence (margin ≥ 0.20) positions flip vs a layer-streamed BF16 reference on 1 408 real mid-generation positions, with a three-way control design that measured the methodology's own noise floor (0 confident flips); most-sensitive population = fully-masked (step-0) windows, i.e. where Γ selection happens; late-step windows 94.4% top-1. (2) All three sweep axes REJECTED: lm_head 4-bit fails the margin-calibrated gate (largest flipped margin 0.32 > noise 0.23 — the *opposite* of Sumi's head result: never port quantization conclusions between models); g32 experts and 6-bit experts (+0.5 / +3.4 GB) leave confident flips unchanged — the routed experts are **not** the drift driver (suspicion: QKV/dense paths or fully distributed). (3) §2.6/§2.7's "lm_head quantization is an open sweep item" is hereby CLOSED: keep 16-bit. (4) E8 blind mask-quality scoring: pairs + sealed-key sheet generated, scripted degeneration checks clean in both arms; awaiting André's blind scores — the `.strict` quality claim stays smoke-level-qualitative until then. Also landed en route: trajectory diagnostics in `Output.metrics` (Γ/Δ per step, transfer confidence, eos block), mixed-quant conversion + loader support, and two toolchain quirks (metallib seeding via `Tools/seed-metallib.sh`; never partially delete `.build` — `build.db` poisoning). Wiki experiment page: to be created by André from the logbook (`05-Experiments/`).

## 5. Deviations from reference (exhaustive list)

Tracked so parity failures have a known suspect list:

1. ExactPrefixCache (reference recomputes prefix every step) — exact by block-causality; M5(b) proves it.
2. SDPA instead of forced-eager attention — tolerance-level numeric difference.
3. Analytic block mask instead of materialized O(total²) tril expansion — same shape as the reference's materialized array, but see deviation 8 for the numeric-semantics difference.
4. `gather_qmm` MoE dispatch instead of CPU sort-loop — removes a per-step CPU sync; identical math.
5. 4-bit weights (dev path) — *not* parity-comparable; only the BF16 path anchors correctness.
6. Dead `steps`/`minimal_topk` parameters not implemented; `num_to_transfer` kept (it is live).
7. `output_attentions` never requested.
8. **Strict block-causal mask (0/-inf) instead of the reference's 0/1 additive-bias mask** (resolved 2026-07-07, §6). This is a deviation *from the published inference script, not from the paper's algorithm* — the reference `generate()` applies its block-tril as a `+1` score bias that fails to mask future blocks (a suspected bug); `.strict` implements the cross-block-causal attention the paper specifies. Parity is therefore measured against a **corrected** reference baseline (strict mask), never stock `generate()`. `.referenceBias` is retained only as a diagnostic. This is the deviation that keeps ExactPrefixCache (deviation 1) mathematically exact.

## 6. Risks / opens carried into implementation

- ✅ **Mask semantics — reference `generate()` is NOT block-causal (found in M3, resolved by André 2026-07-07).** §1 pins the mask as "block-lower-triangular … within-block bidirectional, cross-block causal." The reference *does* build that tril, but as a **0/1 float array** passed to `generate()` as a ready-made 4D mask. transformers returns 4D masks as-is, and eager/SDPA then **add** it to the scores — allowed pairs get `+1`, future pairs get `+0`; nothing is `-inf`, so future blocks are *not* masked out. Verified on the toy config: perturbing a future block shifts an early block's logits ~0.32 under the reference 0/1 mask, exactly 0.0 under a strict 0/-inf mask.

  **Decision (André):**
  1. **Ship `.strict` as NeoDiffusion's real semantics.** This is a **deviation from the published inference script, not from the paper's specified algorithm** — the paper calls for cross-block-causal / within-block-bidirectional attention, which `.strict` implements. The stock `generate()` 0/1 mask is treated as a reference bug we deliberately do not copy. (See §5 deviation 3 and 8.)
  2. **`.referenceBias` stays in `BlockDiffusionMask` as a diagnostic/compat mode only,** clearly labeled as reproducing the suspected bug. Never shipped.
  3. **M4/M5 parity does NOT freeze against stock `generate()` output.** Build a *corrected* reference baseline instead — patch the reference repo's mask construction to emit `-inf` on disallowed pairs (or drive the model with a hand-written attention loop that bypasses `generate()`), and parity-test against that. The `generate_core_fixtures.py` block-mask already emits strict 0/-inf; the M4/M5 full-`generate` trace dumper (per §3) must apply the same correction.
  4. **Measure end-to-end quality under both masks** on a few real prompts once the engine runs (M6+): confirm `.strict` is at least as good as `.referenceBias` (generation quality / perplexity), documenting that the strict mask is not just cleaner but not worse. Tracked as an M6/M8 quality-harness item.

  ExactPrefixCache's "exact by block-causality" argument (deviation 1) is **sound under `.strict`** and stands; it would have been unsound under `.referenceBias`. Not an M3 blocker (blocks are mask-agnostic; fixtures use `.strict`).
- ~~`use_qk_norm` default unknown~~ **resolved**: `True` (see §2.3).
- MLX `/` promotes integer operands to float (bit me in the analytic mask: `arange/blockLength` became fractional, silently degrading the block mask to a token-level causal one). `BlockDiffusionMask` uses `floorDivide`; watch for the same promotion elsewhere index math is done on-GPU.
- MLX-Swift fused SDPA (`MLXFast.scaledDotProductAttention`) **does** reproduce manual `repeat_kv` + FP32-softmax attention to 0.0 on the 0/-inf mask (M3 verified, `testSDPAvsManual`) — §2.3's "verify SDPA reproduces repeat_kv semantics" is discharged. The forward path uses the fused kernel; `LLaDA2Attention.attendReference` is retained as the validation oracle.
- ✅ ~~MLX-Swift `gather_qmm` availability/perf for [256 experts × 512 inter] shapes on M1-generation GPUs (**speculative**) — fallback: segmented qmm over expert groups.~~ **Resolved 2026-07-10 (M6 campaign, `LLaDAMoEDispatchBench`), scope-conditioned**: under the measured conditions (M1 GPU, mlx-swift 0.31.6, these exact shapes, 4-bit g64 affine) `gatherQuantizedMM` is the *fastest* dispatch (4.5 ms/op; 0.4× dense-all-experts, 0.2× dequant+gather) — implementing the fallback would be speculative regression work; it is preserved only as the gated negative regression test. Dispatch rankings can flip with MLX version, GPU generation, expert count, or quant layout — **re-run the bench** on any of those changes (Studio included).
- BF16 on M1-family Metal: MLX supports bf16 storage but some older-GPU paths are fp16-optimized; if BF16 underperforms on the dev machine it doesn't matter (parity runs live on the Studio), but record it.
- Tokenizer chat template fidelity in swift-transformers vs HF (jinja edge cases) — M2's suite exists to catch this.
- `generation_config.json` may override special ids/thresholds — audit in M2.

## 7. What Phase 3 inherits

M6's frozen baseline + the two ports (CacheManager with Exact/Active split, DecodingPolicy) + batched forward + the Block-Buffer-shaped loop (N_buf ready to raise). Phase 3 work packages per the Phase 1 §8 ordering: [[elastic-cache-metal-kernel]] (wiki proposal) against the ActiveBlockCache, and training-free MultiBD ([[mbd-lms]]: N_buf=2, τ_add/τ_semi gating — directly measured on LLaDA2.1-Mini, proposed co-first pending André's priority call); then Spiffy (batch dim already paid for), then the Streaming-dLLM cluster.
