# NeoDiffusion — Handoff after M5 (denoising loop + ExactPrefixCache)

**Written**: 2026-07-07, at the close of M5.
**For**: whoever picks up **M6 (diffusion-bench)** next — and the still-open **Studio BF16 real-weight parity** for M4/M5.
**Status**: M5 is implemented and green on the dev toy-config FP32 path (26/26 tests). The only M5/M4 work left is the BF16 real-weight run, which needs a machine the ~33 GB checkpoint fits on (not the dev M1). See the download question answer at the bottom.

This document is scoped to *things you'd otherwise have to rediscover*. The authoritative design/detail lives in `phase-2-implementation-guide.md` (§4 M5/M6 entries updated with this milestone) and `phase-1-conceptual-design.md`.

> **Update 2026-07-10 — the §3 M6 groundwork below has landed.** The 4-bit artefact exists and is verified (`models/llada2-1-mini-4bit`; BF16 source local at `models/llada2-1-mini`); the loader handles its U8-packed layout; `DiffusionEngine.Output.metrics` now carries per-phase wall-clock, post-steps/block and the honest forward count (items 2–4 of §3); the legacy stub is deleted and `diffusion-bench llada` implements the M6 metric set with checked-in prompt suites (items 1 and 5). §5's answer is partially overtaken: download + streaming quantisation *did* work on the dev M1 (`Tools/convert_weights_streaming.py`) — but its core point stands: **BF16 parity remains Studio-blocked**, and the 4-bit path is task-level only. Current state and remaining M6 work: phase-2 guide §3 conversion status + §4 M6 status.

---

## 1. Where things are (files you'll want, code + cached)

### Reference implementation (the ground truth), already in the HF cache
The pinned reference `.py` files are downloaded and cached locally — **you do not need network access to read them**:

```
/Users/andrebarlocher/.cache/huggingface/hub/models--inclusionAI--LLaDA2.1-mini/snapshots/20e64e2ad21644d0e5248586ed9c942cdd45de0f/
    modeling_llada2_moe.py            # the reference model + generate(); generate() at line 1244
    configuration_llada2_moe.py       # the default table (use_qk_norm=True etc.)
    config.json                       # real model config (20 layers, 256 experts, ...)
```

`generate()` is lines **1244–1457**; `_sample_with_temperature_topk_topp` (temp-0 = argmax + softmax prob) at **1212**. The fixture dumpers fetch these same files via `huggingface_hub` (cache hit), so they run offline.

### NeoDiffusion source added/changed in M5
- **Loop**: `Packages/DiffusionGeneration/Sources/DiffusionEngine.swift` — `generate` (cache-disabled) + `generateCached` (ExactPrefixCache); `Output` struct exposes `tokens`, `finalSequence`, `blockCommits`, `stepsPerBlock`, `syncPoints`.
- **Params**: `Packages/DiffusionGeneration/Sources/GenerationParams.swift` — S/Q mode presets.
- **State machine**: `Packages/DiffusionGeneration/Sources/BlockBuffer.swift`.
- **Caches**: `Packages/DiffusionGeneration/Sources/{ExactPrefixCache,ActiveBlockCache}.swift`; per-layer KV store `Packages/DiffusionCore/Sources/LayerKVCache.swift`.
- **Cache-aware forward overloads** (added alongside the existing signatures — the uncached ones are untouched): `LLaDA2Attention`, `LLaDA2DecoderLayer` (`callAsFunction(_:cos:sin:cache:)`), `LLaDA2MoeInnerModel` / `LLaDA2MoeModel` (`callAsFunction(_:positionIds:caches:)`, plus `layerCount`).

### Fixtures + dumpers
- **M5 loop fixtures**: `Tools/generate_loop_fixtures.py` → `scratch/loop_fixtures/{weights.safetensors,traces.json,manifest.json}` (16 cases). Consumed by `Tests/DiffusionGenerationTests/DenoisingLoopParityTests.swift`.
- **M3/M4 fixtures**: `Tools/generate_core_fixtures.py` → `scratch/core_fixtures/` (consumed by `CoreFixtureTests`, `ForwardParityTests`).
- **Toy HF config/weights** the dumpers default to: `scratch/dummy_hf/{config.json,model.safetensors}` (made by `Tools/generate_dummy_weights.py`).
- **4-bit conversion**: `Tools/convert_weights.py` (shard-by-shard `mx.quantize`, keep-list matches §2.7).

Both dumpers accept `--config <hf>/config.json --weights <hf_dir> --dtype bfloat16 --out <dir>` for the Studio real-weight runs; the Swift tests read the fixture dir from `NEODIFFUSION_FIXTURE_DIR` (M4) / `NEODIFFUSION_LOOP_FIXTURE_DIR` (M5), else default to `scratch/…`.

---

## 2. Verified facts (confirmed this milestone; adapt-noted where I diverged)

**Reference `generate()` semantics** (read from the cached source, matches the §1 pin):
- `num_blocks = ceil((prompt_len + gen_length) / block_length)`, `total_length = num_blocks * block_length`, `prefill_blocks = prompt_len // block_length`.
- `post_steps` **increments only on a mask-free iteration**, checked *before* the forward; the budget break (`post_steps > max_post_steps`) does **no forward** that iteration.
- **Both break conditions imply the block is mask-free at commit** → a committed block never contains `mask_id`. (This is why the engine's eos check only tests "does the generated region contain eos", never "is it unmasked".)
- Γ fallback: if `#{masked, conf>τ_mask} ≥ num_to_transfer` use those, else top-`min(num_to_transfer, #available)` by confidence. `num_to_transfer` **live value = 1** (top-1 = argmax).
- Δ: `editable = ~active_mask & ~prompt_mask_in_block`; keep where `conf > τ_edit` **and** `x0 != current`.
- Trim: first `eos_id` in the generated region `[prompt_len, prompt_len+gen_length)`, **inclusive**; if none, `gen_length` (the return slice `+1` then clamps → output length == `gen_length`).

**Mask**: stock `generate()` bakes a **0/1 soft-bias** mask (not block-causal — the suspected bug, §6). All fixtures use the corrected **strict 0/-inf** mask, forced by monkeypatching `create_bidirectional_mask` — the reference inner model *unconditionally rebuilds its mask* (`modeling_llada2_moe.py` ~line 877), so passing a 4D mask argument alone is **not** enough. Both dumpers already do this patch.

**Cache exactness** (M5b, deviation 1): a committed block's K/V equals what an uncached full-window forward computes for those positions, so the active block attends to `[committed ++ active]` with **no mask** and gets bitwise-identical logits. Proven: 16/16 cached == cache-disabled == reference (incl. heavy-editing S-mode blocks that exit via the post-step budget — the commit-cleanliness scenario).

**Toy config** (dev-runnable, `scratch/dummy_hf/config.json`): vocab 1000, hidden 128, **2** layers, block 16, 4 experts / 2 active, n_group 2 / topk_group 1, first_k_dense 1, moe_inter 64, 4 Q / 1 KV heads, head_dim 32. Special ids are remapped for the tiny vocab: **toy `mask_id=999`, `eos_id=998`** (real model: 156895 / 156892).

**Dev Python env**: torch 2.12.1, transformers 5.2.0, safetensors, huggingface_hub — all present; the reference dumpers run on the dev machine at toy scale.

**Real model size**: ~16 B params → **~33 GB BF16**. Does **not** fit the dev M1's 16 GB. Disk is fine (132 GB free).

**MLX-Swift API facts** (these are easy to get wrong — see §4):
- Element-wise logical ops are **`.&&` / `.||`**, prefix-not is **`.!`** (there is *also* a scalar-returning `&&`/`!` — don't grab those by accident).
- 0-dim scalar `MLXArray`s **cannot** be `concatenated(axis: 0)` — reshape to `[1]` first.
- Empty array: `MLXArray.zeros([1, 0], dtype: .int32)`.
- `.max(axis:)`, `.argMax(axis:)`, `.any()` exist as methods; `MLXArray(Int32(a) ..< Int32(b))` for ranges.
- MLX `/` promotes int operands to float — index math on-GPU must use `floorDivide` (already true in `BlockDiffusionMask`; watch it anywhere new).

---

## 3. What's in place for M6 (diffusion-bench)

M6 wants: TPS, TPF, steps/block, post-steps/block, **sync-point count**, peak memory, per-phase wall-clock (prefill/denoise/commit), JSON-lines output, fixed prompt suites.

**Already handed to you by `DiffusionEngine.Output`**:
- `stepsPerBlock` — denoising steps per block (see the caveat in §4 about the budget-break off-by-one and speculation).
- `syncPoints` — the blocking-readback count (the M5c metric); already asserted within budget.
- `finalSequence` / `blockCommits` / `tokens` — for correctness + quality-proxy scoring.

**What M6 must add**:
1. **Rewrite the stubs.** `Tools/diffusion-bench/Sources/main.swift` and the `DiffusionGenerationTests.testDenoisingGeneration` test **still use the legacy `DiffusionGeneration` stub class** (random-latent placeholder) and `DiffusionModelConfig`. `DiffusionGeneration` (the stub) is kept only so bench/server compile; M6/M7 replace it with `DiffusionEngine`. Delete the stub once both callers move over.
2. **Timing/phase instrumentation** is *not* in the engine yet — add per-phase wall-clock (prefill vs denoise vs commit) and TPS/TPF at the bench layer (wrap `generate`/`generateCached`, `eval` at phase boundaries so timings are real, not lazy).
3. **post-steps/block** is not separately surfaced — it's folded into `stepsPerBlock`. Expose the mask-free-iteration count if you want it as its own metric.
4. **Wasted-forward count**: the K-step speculative loop *evaluates* K forwards per batch but only `stepsPerBlock` of them are "logical". For an honest TPF, report wasted forwards too (or run bench at `speculationK: 1` for reference-comparable step economy — output is identical).
5. Prompt suites (chat/reasoning/code) get checked into `Tools/diffusion-bench`.

The §6-mask quality diagnostic (`.strict` vs `.referenceBias`) is also first-runnable at M6 — `BlockDiffusionMask.Semantics.referenceBias` exists for exactly this.

---

## 4. Things that bit me / would bite you (backtracks, gotchas)

- **Logical operators.** I first wrote `.&`/`.|`/`.logicalNot()` in the engine → compile error. Correct is `.&&`/`.||`/`.!` (element-wise). If a boolean expression silently type-checks but behaves oddly, check you didn't get the scalar `&&`/`!` overloads.
- **Scalar concat fatal error.** `anyMask.any()` and the break flags are 0-dim; `concatenated(breakFlags, axis: 0)` **crashes at runtime** ("Axis 0 out of bounds for array with 0 dimensions"). Fix: `breakFlag = (...).reshaped([1])` before stacking. Generally: reshape scalars to `[1]` before combining for a single readback.
- **Speculation vs cache commit — the subtle one.** The K-step speculative loop overwrites each layer's `pending` K/V every step, so after finding the break, `pending` holds the *last speculative* step's K/V, **not** the committed tokens'. I do **not** try to reuse denoise-step `pending`. Instead `generateCached` runs an **unconditional capture forward** over the final committed tokens at each commit (this also doubles as the per-block prefill). It's one extra forward/block (perf, not correctness). The guide's "only re-forward on a budget exit" optimization is a **Phase-3 micro-opt** — don't prematurely add it; it's easy to get wrong against speculation.
- **`stepsPerBlock` is not reference-exact on budget-break blocks.** The reference doesn't count the budget-break iteration (it breaks before the forward); my counter counts `firstBreak+1` including it, so a block that exits via the post-step budget reads **+1** vs the reference. Tokens are unaffected and no test asserts step-count parity. If M6 wants steps/block to match the reference trace's `per_block_steps`, subtract 1 on budget-break exits (or assert against the fixture's `per_block_steps` and reconcile there).
- **eos-trim is only structurally exercised at toy scale.** Random toy weights never emit `eos_id`, so all 16 cases return the full `gen_length` and eos-on/off produce identical output. The trim + `eos_early_stop` logic is faithfully transcribed but only *lightly* exercised — worth a targeted check on the real-weight run, where eos can actually fire.
- **`numToTransfer > 1` is not implemented** — the branchless Γ is `== 1` only, guarded by a `precondition`. >1 needs index-exact top-k tie handling (argmax picks first-max on ties; a threshold-based top-k would over-select on ties). Deferred to an M6 bench extension if the sweep needs it.
- **Fixtures are toy-only right now.** `scratch/loop_fixtures` was generated with seeded random weights; parity is *algorithmic* agreement, which is the strong claim. The BF16 real-weight run is what closes M4/M5 formally.

---

## 5. Your question: download + quantise on the dev M1?

Short answer: **download — sure; quantise — only if you babysit memory; but neither unblocks the parity gate you actually need next.**

- **Downloading** the ~33 GB checkpoint on the M1 is fine (132 GB disk free). It's just storage.
- **Quantising on the M1 is fragile.** `convert_weights.py` accumulates the *entire* converted output (~10 GB) in a Python dict **before** writing, plus one ~5 GB shard in flight → peak comfortably over 16 GB. It'll swap hard and may OOM. If you must, first refactor `convert_weights.py` to stream (write/free per shard) — happy to do that.
- **The blocker is BF16, not 4-bit.** The remaining M4/M5 acceptance is **BF16 token-for-token parity**, which needs (a) the PyTorch reference dumping BF16 fixtures and (b) the Swift engine running BF16 inference — both a ~33 GB working set. Neither fits 16 GB. And the 4-bit path is **explicitly never parity-comparable** (CLAUDE.md gotcha 5 / §2.7): quantising locally gives you something to *run*, not something to *validate* against.

**Recommendation**: do the BF16 fixture dump, the BF16 parity run, **and** the 4-bit conversion on a machine the 33 GB fits on — the Studio, or (since you lack access) a rented cloud box with ≥ ~48–64 GB RAM (a cloud Mac, or a Linux host for the PyTorch side). Then copy just the ~10 GB 4-bit artefact + tokenizer files back to the M1 for local task-level runs. This is exactly the split CLAUDE.md's "Environment notes" already prescribes. Until a big machine is available, **M6/M7 can be built and tested against the toy fixtures** (all green today) — only their *real-weight numbers* wait on the checkpoint.
