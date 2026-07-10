# NeoDiffusion — LLaDA-MoE-7B-A1B (LMOE) Implementation & Optimisation Guide

**Status**: draft, for André's review
**Written**: 2026-07-07, between M5 and M6
**Companion documents**: `handoff-post-M5.md` (start here for the current engine state), `phase-2-implementation-guide.md` (LLaDA2.1-mini parity target), `phase-3-optimisation-roadmap.md` (post-baseline optimisations).
**Primary sources (retrieved 2026-07-07)**:
- LMOE model card, [`inclusionAI/LLaDA-MoE-7B-A1B-Instruct`](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct)
- LMOE [`config.json`](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct/raw/main/config.json), [`configuration_lladamoe.py`](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct/raw/main/configuration_lladamoe.py), [`modeling_lladamoe.py`](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct/raw/main/modeling_lladamoe.py), [`generation_config.json`](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct/raw/main/generation_config.json)
- LMOE paper: [Zhu et al., "LLaDA-MoE: A Sparse MoE Diffusion Language Model", arXiv:2509.24389](https://arxiv.org/abs/2509.24389)
- [dInfer reference engine (`inclusionAI/dInfer`)](https://github.com/inclusionAI/dInfer)

Every load-bearing claim below is either **sourced** (traceable to one of the above), **inferred** (reasoned from sourced facts), or **speculative** — flagged inline.

---

## TL;DR — the honest take before the plan

LMOE is **not the same shape** as LLaDA2.1-mini. It looks like a sibling but it deviates on almost every axis that Phase 2 optimises for. Before starting the port, be aware of these five hard facts (all **sourced** from `config.json` + `modeling_lladamoe.py`):

1. **Attention is fully bidirectional, everywhere.** The reference sets `is_causal = False` in every attention path, and its inner-model `forward` asserts `attention_mask is None`. There is no block-causal / cross-block-causal mask baked into `forward`. Block-causal shape is imposed *only* by the sampling loop's chunked unmasking — not by the mask fed to the attention kernel. **This directly invalidates the "ExactPrefixCache is mathematically exact" argument you rely on for LLaDA2.1-mini** (Phase 2 §5 deviation 1, `handoff-post-M5.md` §2). Prompt+committed KV under bidirectional attention is *not* invariant when a new mask block appears — every masked token is a key that attends bidirectionally to every other, so the committed prefix's K/V changes with future context. Under LMOE, `ExactPrefixCache` is *approximate*, not exact.
2. **The sampling algorithm is classic LLaDA-1.x, not LLaDA2.x draft-and-edit.** No T2T (Δ set), no `editing_threshold`, no `max_post_steps`, no S/Q modes. It's `low_confidence_remask` with a `num_transfer_tokens` schedule and top-k selection per step — plus optional CFG. `GenerationParams` and the `Γ ∪ Δ` selection code in `DiffusionEngine` do not carry over as-is.
3. **The router is softmax + top-k, no groups, no expert bias, no scaling factor.** LLaDA2.1-mini's sigmoid + group-limited + `expert_bias` + `routed_scaling_factor=2.5` FP32 router is not what LMOE ships. LMOE hardcodes `norm_topk_prob = False` (weights are NOT renormalised), and `config.json` sets `routed_scaling_factor: 1` (i.e. no scaling), `router_num_group: null`, `router_topk_group: null`, `moe_router_enable_expert_bias: false`. Your `LLaDA2MoEGate` needs a second implementation, not a config toggle.
4. **QKV is split, not fused. Attention uses full (not partial) RoPE. Norm names are `q_norm/k_norm`, not `query_layernorm/key_layernorm`.** All parity-blocking if you copy the LLaDA2.1-mini weight-map table.
5. **Weights are 3 safetensors shards, ~14.7 GB BF16**. Fits in a Studio's 192 GB comfortably; **does not fit** the dev M1's 16 GB even quantised naïvely (~4 GB 4-bit routed experts + full FP16 shared/norm/embed/lm_head is ~5–6 GB before activations, but the quantise step itself needs the full BF16 in RAM — see §7). LMOE therefore has **similar dev-machine feasibility to LLaDA2.1-mini's 33 GB checkpoint**: convert on a bigger box, run inference locally.

The good news: the differences are *narrower and simpler* than LLaDA2.1-mini (fewer layers, no groups, no shared expert, no partial RoPE, no editing). Once the abstraction is right, LMOE ends up as **less** code than LLaDA2.1-mini, not more. It is a genuinely useful second target because it isolates whether your engine is "one model masquerading as a family" or actually family-shaped.

Fair warning: at the dev M1's 16 GB, **you gain unblocking on LMOE, not on your BF16 parity blocker for M4/M5**. That blocker (33 GB BF16 for LLaDA2.1-mini) is unchanged. LMOE at 4-bit routed experts fits — see §7's memory table — but it doesn't help you close M4/M5 without a bigger machine.

---

## 0. Recommended reading order

1. This §0–§2 (delta and refactor overview).
2. §3–§4 (multi-model abstraction and MLX sketches). Then decide with André whether M6 or LMOE-as-second-target lands first.
3. §5 (Metal sketches). Skip until §4 lands.
4. §6 (which Resources optimisations apply to LMOE). Skim as a menu for Phase 3.
5. §7 (memory, quantisation, fixture logistics) — needed before you download anything.

---

## 1. LMOE model quick facts (sourced from `config.json` + `modeling_lladamoe.py`, 2026-07-07)

Side-by-side against LLaDA2.1-mini so the *deltas* are visible. Everything under LMOE is **sourced** unless flagged.

| Property | LLaDA2.1-mini (current target) | LLaDA-MoE-7B-A1B (LMOE) |
|---|---|---|
| Total params | ~16 B | ~7 B (activated: ~1.4 B) |
| Layers | 20 (layer 0 dense, 1–19 MoE) | 16 (**all** MoE — `moe_layer_freq = [1]*16`) |
| Hidden size | 2048 | 2048 |
| Heads (Q / KV) | 16 / 4 (**GQA**) | 16 / 16 (**MHA, no GQA**) |
| Head dim | 128 | 128 (implicit: 2048/16) |
| QKV projection | **fused** `query_key_value` [3072, 2048] | **split** `q_proj`/`k_proj`/`v_proj` |
| Attention bias | none (`use_qkv_bias=false`) | none (`attention_bias=false`) |
| Output proj name | `attention.dense` | `self_attn.o_proj` |
| Q/K norm | `query_layernorm`/`key_layernorm` on head_dim | `q_norm`/`k_norm` on head_dim (`qk_layernorm=true`) |
| RoPE | **partial** (first 64 of 128 dims), θ=600 000 | **full** (`partial_rotary_factor=1`), θ=50 000 |
| Max position | 4 096 (wired 4 096 sliding window, inert) | 8 192 (`max_position_embeddings`), sliding window absent |
| Dense FFN | layer 0 only, intermediate 5120 | none (all layers MoE) |
| MoE experts | 256 routed + 1 shared, intermediate 512 | 64 routed, **no shared**, intermediate 1024 |
| Active experts / token | 8 | 8 |
| Router | **sigmoid**, group-limited (8×32 groups, top-4 groups), FP32 matmul, +`expert_bias`, ×2.5 scale, weights renormalised | **softmax**, plain top-8 (no groups), FP32 dtype in softmax only, no bias, no scaling factor, `norm_topk_prob=False` (weights NOT renormalised) |
| Shared expert intermediate | 512 (`num_shared_experts=1`) | `shared_expert_intermediate_size: null` → **none** |
| lm_head | untied (`tie_word_embeddings=false`) | untied (`tie_word_embeddings=false`), `_tied_weights_keys=["lm_head.weight"]` is dead metadata |
| Vocab | 157 184 | 157 184 (**same tokenizer family** — verify in M2') |
| RMSNorm ε | 1e-6 | 1e-5 (**default, not overridden**) |
| Sampling method | draft-and-edit: M2T + T2T under (τ_mask, τ_edit), `max_post_steps`, S/Q modes | classic LLaDA: `low_confidence_remask` + `num_transfer_tokens` schedule + top-k selection, optional CFG |
| `mask_id` | 156 895 | 156 895 (same) |
| `eos_id` / `pad_id` | 156 892 / 156 892 | 156 892 / 156 892 (same) |
| Recommended block | 32 | 32 (readme code uses block 32, steps 128, gen 128) |
| Recommended threshold | Q 0.7 / S 0.5 | dInfer defaults to `threshold=0.9`, single threshold (no editing threshold) |
| Weights on disk | ~33 GB BF16 | ~14.7 GB BF16 (3 shards of 5 GB + 5 GB + 4.72 GB) |
| Cache in reference | recomputes full prefix; `use_cache=false` in config; inner-model asserts `not use_cache` | recomputes full prefix; `use_cache=false`; asserts identical to LLaDA2.1-mini |
| Attention mask shape | reference passes 4D block-tril but as 0/1 bias (§6-mask bug); NeoDiffusion ships `.strict` | reference sets `attention_mask=None` and `is_causal=False`; **truly bidirectional over the padded sequence** |

**dInfer decoding parameters** (**sourced**, README): `ThresholdParallelDecoder(0, threshold=0.9)`, `BlockWiseDiffusionLLM(model, decoder, BlockIteratorFactory(True), cache_factory=KVCacheFactory('dual'))`, `block_length=32`, `gen_length=128`, `temperature=0.`, `cfg_scale=0.`, `remasking='low_confidence'`. dInfer's `KVCacheFactory('dual')` and `BlockIteratorFactory(True)` are **not documented** in the README beyond the constructor invocation — the mechanism (dual = prompt cache + block cache? or dual = clean/noisy?) has to be read from dInfer's source. Flagged as an M0' investigation item, not a parity dependency. The **transformers-side** sampling function in the README (reproduced in §2.1) is the parity target for the baseline path; dInfer is the ceiling target, not the correctness anchor.

**dInfer performance claim** (**sourced**, model card): >1 000 TPS average, 1 100+ TPS on HumanEval batch=1, on 8× H800. These are the ceiling; do not treat as portable. The Apple Silicon target is single-request TPS on M2 Ultra 192 GB — the ratio to dInfer will be whatever it is, and reporting it honestly is a Phase 3 output, not a Phase 2 promise.

---

## 2. The reference sampling algorithm, pinned

Analogous to Phase 2 §1 for LLaDA2.1-mini: this is the parity target — the engine must reproduce it token-for-token at temperature 0 before deviating.

### 2.1 The transformers-based sampling function (verbatim from the model card)

```python
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=156895):
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length:
                                   prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            # (CFG branch omitted — cfg_scale=0 default)
            logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

            # Mask future blocks OUT of the current step's selection candidates
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            # Pick the top-num_transfer_tokens[j, i] positions BY CONFIDENCE this step
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x
```

Facts with engineering consequences (all **sourced** from the code above unless noted):

- **`steps` is a *per-block* budget**, not global (`steps = steps // num_blocks` on entry). At the README's defaults (steps=128, gen=128, block=32) that is **1 step per block × 32 tokens = fixed unmask-all-in-one-step**. That is *not* how the block-diffusion literature typically frames things and I'd want André to confirm this is the intended default before optimising against it (**inferred**: at temp 0 with 1 step/block LMOE effectively behaves as full-context bidirectional argmax — the "diffusion" aspect only kicks in when `steps > block_length` at block_length=32; e.g. steps=64, block=16). The dInfer `ThresholdParallelDecoder(0, threshold=0.9)` chooses a **variable** transfer count per step by threshold, not a scheduled count — a genuinely different algorithm.
- **`num_transfer_tokens[j, i]` is fixed at loop entry** by dividing the block's mask count evenly across steps + putting the remainder in the first few steps. It is a *schedule*, not a policy.
- **Selection is per-batch, per-step top-k by confidence** — Python `for` loop over the batch dimension. On Metal this is one `argSort` per token-batch dim slice (or `topK` if MLX exposes it), not a real problem, but it *is* a batch-dim readback in the reference (`num_transfer_tokens[j, i]` is a Python int).
- **The reference is fully bidirectional over the entire padded sequence every step.** `model(x)` — no attention mask, no cache. Every step recomputes L=prompt+gen forward through 16 layers, with full self-attention over all L positions. Under LMOE's `is_causal=False`, prompt tokens attend bidirectionally to mask tokens and vice versa. This is the correctness-anchoring shape and must be reproduced verbatim in the M4'/M5' fixture-diff tests.
- **Future-block masking is a *soft* selection guard, not an attention mask.** The line `x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf` prevents future-block tokens from being *selected for unmasking* this outer-loop iteration; it does **not** prevent them from *influencing* current-block logits via attention. That is the block-causal-shape-imposed-by-loop semantics I mentioned in the TL;DR.
- **CFG (`cfg_scale > 0`)** doubles the forward's batch dim (prompt+full, prompt-masked); at `cfg_scale=0.` this branch is dead code. The Phase 2 engine has no CFG hook — see §3.5.
- **`add_gumbel_noise(logits, temperature=0.)`** is a no-op (`if temperature == 0: return logits`). Sampling at temperature 0 is `argmax(logits)`, same as LLaDA2.1-mini.
- **No `eos_early_stop`**, no `Δ` / editing, no `max_post_steps`. The block loop always runs exactly `steps // num_blocks` iterations, then moves on. Simpler than LLaDA2.1-mini's inner loop.

### 2.2 Differences that break LLaDA2.1-mini parity code

Every one of these will silently fail your existing tests if you just point them at LMOE weights:

1. `DiffusionEngine.generateCached`'s ExactPrefixCache: unsafe under bidirectional attention (see §3.4).
2. `LLaDA2Attention` fused-QKV split by `[16 Q, 4 KV, 4 KV]` heads: wrong for LMOE — needs three separate projections and 16-Q-16-KV.
3. `PartialRotaryEmbedding` slicing first 64 dims: LMOE rotates the full 128.
4. `LLaDA2MoEGate` sigmoid + group logic + `expert_bias`: **wrong routing** on LMOE — needs softmax + plain top-k branch.
5. `LLaDA2SparseMoEBlock` shared-expert add: absent on LMOE (`shared_expert_intermediate_size: null`).
6. Weight name mapping `attention.query_key_value.weight`, `attention.dense.weight`, `query_layernorm.weight` etc.: LMOE uses `self_attn.{q,k,v,o}_proj.weight`, `self_attn.{q,k}_norm.weight`.
7. `LLaDA2MoeConfig`'s `firstKDenseReplace: 1` default vs LMOE's all-MoE (via `moe_layer_freq`): needs a per-layer `mlp_type` decision, not a scalar.
8. `GenerationParams` (Q/S modes with τ_mask, τ_edit): irrelevant to LMOE's sampling — no editing threshold.
9. `DiffusionEngine`'s Γ ∪ Δ selection: LMOE is Γ-only, and Γ is scheduled (`num_transfer_tokens`), not thresholded. dInfer's `ThresholdParallelDecoder` is a *third* selection policy (thresholded, no schedule). Three algorithms, one interface.

---

## 3. Multi-model refactor: from "LLaDA2.1-mini engine" to "LLaDA-family engine"

Your Phase 1 §1 decision already says *"LLaDA-family config-driven abstraction. Not a general dLLM framework."* — LMOE is exactly the second family member that decision was reserved for. The right moment to do this is **now, before M6 freezes the baseline** — every extra Phase 3 optimisation lands twice if the abstraction slips later.

### 3.1 What to keep, what to split

Reuse verbatim:
- `DiffusionCore/LayerKVCache.swift`, `LLaDA2RMSNorm.swift` (RMSNorm is architecture-agnostic; only ε differs, already parameterised).
- `DiffusionGeneration/BlockBuffer.swift` — the `dummy → active → toCache → inCache` state machine is model-independent.
- `DiffusionGeneration/ExactPrefixCache.swift` and `ActiveBlockCache.swift` — types stay; **semantics do not carry** to LMOE (§3.4). The types themselves are fine.
- Server + tokenizer plumbing (LMOE shares the tokenizer family — same vocab, same `mask_id`/`eos_id`).

Split into per-family variants:
- `DiffusionCore/LLaDA2Attention.swift` → `DiffusionCore/Attention/{FusedQKVGQAAttention, SplitQKVMHAAttention}.swift` behind a common `Attention` protocol.
- `DiffusionCore/PartialRotaryEmbedding.swift` → parameterise `rotaryDim` and `theta` at init; a `PartialRotaryEmbedding(rotaryDim: 64, theta: 600000)` and `RotaryEmbedding(rotaryDim: headDim, theta: 50000)` are the same code with different constants. The "rotate first N of D dims + passthrough" pattern is already what your code implements — just don't hardcode 64.
- `DiffusionCore/LLaDA2MoE.swift` → split router into two implementations (`SigmoidGroupedRouter` for LLaDA2.1-mini, `SoftmaxTopKRouter` for LMOE), keep `SwitchGLU` / `SwitchLinear` / `QuantizedSwitchLinear` intact (the gathered-qmm dispatch is model-agnostic). `LLaDA2SparseMoEBlock` needs a variant without `sharedExperts` — an optional field is enough, LMOE just passes `nil`.
- `DiffusionCore/LLaDA2DecoderLayer.swift` → per-family, but only because the sub-module type varies (fused vs split QKV, shared-expert or not). If the sub-modules are protocols, the decoder layer is one struct.
- `DiffusionModel/LLaDA2MoeConfig.swift` → generalise to a `LLaDAFamilyConfig` protocol; concrete types `LLaDA21MiniConfig` (current struct) and `LLaDAMoEConfig` (new).

Delete / retire on LMOE:
- `GenerationParams.SQMode` (LMOE has no editing).
- `DecodingPolicy`'s τ_edit branch (LMOE is Γ-only; the Δ branch stays dark).
- `max_post_steps` (LMOE runs a fixed schedule).

Add:
- A `SamplingPolicy` protocol that owns the outer-loop / inner-step algorithm. Three concrete implementations: `DraftAndEditPolicy` (LLaDA2.1-mini, existing code), `ScheduledLowConfidencePolicy` (LMOE README code, §3.5), `ThresholdParallelPolicy` (LMOE dInfer-compatible, §6.4). This is the biggest structural change and it *has* to happen before M6 baselines — otherwise M6's numbers freeze one algorithm's baseline as "the" baseline.

### 3.2 Suggested Package layout

```
Packages/
  DiffusionCore/
    Sources/
      LayerKVCache.swift                 # unchanged
      RMSNorm.swift                       # renamed from LLaDA2RMSNorm; ε is a param
      BlockDiffusionMask.swift            # unchanged
      Rope/
        PartialRotaryEmbedding.swift      # parameterised (rotaryDim, theta)
      Attention/
        Attention.swift                   # protocol
        FusedQKVGQAAttention.swift        # ex-LLaDA2Attention (LLaDA2.1-mini)
        SplitQKVMHAAttention.swift        # new (LMOE)
      MoE/
        SwitchLinear.swift                # unchanged
        SwitchGLU.swift                   # unchanged
        SigmoidGroupedRouter.swift        # ex-LLaDA2MoEGate (LLaDA2.1-mini)
        SoftmaxTopKRouter.swift           # new (LMOE)
        SparseMoEBlock.swift              # optional shared-expert field
      DecoderLayer.swift                  # generalised (per-family sub-module types)
      LLaDA2MLP.swift                     # unchanged; used by LLaDA2.1-mini layer 0
  DiffusionModel/
    Sources/
      LLaDAFamilyConfig.swift             # protocol
      LLaDA21MiniConfig.swift             # existing struct, renamed
      LLaDAMoEConfig.swift                # new
      DiffusionModel.swift                # dispatch by config type
      DiffusionTokenizer.swift            # unchanged
  DiffusionGeneration/
    Sources/
      SamplingPolicy.swift                # protocol (new)
      DraftAndEditPolicy.swift            # existing DiffusionEngine.generate/generateCached inner loop
      ScheduledLowConfidencePolicy.swift  # LMOE README parity
      ThresholdParallelPolicy.swift       # LMOE dInfer-style
      DiffusionEngine.swift               # thin driver on top of the policy
      BlockBuffer.swift                   # unchanged
      ExactPrefixCache.swift              # unchanged type; usage predicated on mask semantics
      ActiveBlockCache.swift              # unchanged
```

**Concrete refactor sequencing** (single sprint if done in this order):
1. Rename `LLaDA2RMSNorm.swift` → `RMSNorm.swift`, generalise ε. **Tests stay green.**
2. Parameterise `PartialRotaryEmbedding` on (`rotaryDim`, `theta`). **Tests stay green** (defaults reproduce LLaDA2.1-mini).
3. Extract `Attention` protocol, rename `LLaDA2Attention` → `FusedQKVGQAAttention` implementing it. **Tests stay green.**
4. Introduce `LLaDAFamilyConfig` protocol; make `LLaDA2MoeConfig` conform. **Tests stay green.**
5. Extract `SamplingPolicy` protocol from `DiffusionEngine`. **Tests stay green** (`DraftAndEditPolicy` is the current `generate/generateCached` inner loop as-is).
6. **Only then** land `SplitQKVMHAAttention`, `SoftmaxTopKRouter`, `LLaDAMoEConfig`, `ScheduledLowConfidencePolicy`. New tests pass; existing ones untouched.

If step 5 (the SamplingPolicy split) is too big to slip before M6, the pragmatic alternative is: **land steps 1–4 now, defer 5–6 to a second milestone (call it M5.5)**. M6 then benches on `DraftAndEditPolicy` only; LMOE lands in an M5.5 that establishes its own baseline before rejoining M6's `diffusion-bench` in a widened form. This is what I'd honestly recommend if you and Claude Code find step 5 non-trivial — a second bench baseline is cheaper than a rushed abstraction.

### 3.3 Weight-map table for LMOE (M1' acceptance criterion)

Confirm every one of these against `model.safetensors.index.json` at download time. Format matches the LLaDA2.1-mini table in `phase-2-implementation-guide.md` §2.

| HF key | Shape | Dtype | Swift target |
|---|---|---|---|
| `model.embed_tokens.weight` | `[157184, 2048]` | BF16 | `LLaDAMoEModel.embedTokens` (keep 16-bit) |
| `model.norm.weight` | `[2048]` | BF16 | `LLaDAMoEModel.norm` |
| `lm_head.weight` | `[157184, 2048]` | BF16 | `LLaDAMoEModel.lmHead` |
| `model.layers.{i}.input_layernorm.weight` | `[2048]` | BF16 | `DecoderLayer.inputLayerNorm` |
| `model.layers.{i}.post_attention_layernorm.weight` | `[2048]` | BF16 | `DecoderLayer.postAttentionLayerNorm` |
| `model.layers.{i}.self_attn.q_proj.weight` | `[2048, 2048]` | BF16 (→ 4-bit) | `SplitQKVMHAAttention.qProj` |
| `model.layers.{i}.self_attn.k_proj.weight` | `[2048, 2048]` | BF16 (→ 4-bit) | `SplitQKVMHAAttention.kProj` |
| `model.layers.{i}.self_attn.v_proj.weight` | `[2048, 2048]` | BF16 (→ 4-bit) | `SplitQKVMHAAttention.vProj` |
| `model.layers.{i}.self_attn.o_proj.weight` | `[2048, 2048]` | BF16 (→ 4-bit) | `SplitQKVMHAAttention.oProj` |
| `model.layers.{i}.self_attn.q_norm.weight` | `[128]` | BF16 | `SplitQKVMHAAttention.qNorm` (per-head-dim RMSNorm) |
| `model.layers.{i}.self_attn.k_norm.weight` | `[128]` | BF16 | `SplitQKVMHAAttention.kNorm` |
| `model.layers.{i}.mlp.gate.weight` | `[64, 2048]` | **FP32 (keep unquantised)** | `SoftmaxTopKRouter.weight` |
| `model.layers.{i}.mlp.experts.{e}.gate_proj.weight` | `[1024, 2048]` × 64 → stacked `[64, 1024, 2048]` | BF16 (→ 4-bit) | `SwitchGLU.gateProj` (via `ExpertWeightStacking.stack`) |
| `model.layers.{i}.mlp.experts.{e}.up_proj.weight` | `[1024, 2048]` × 64 → stacked | BF16 (→ 4-bit) | `SwitchGLU.upProj` |
| `model.layers.{i}.mlp.experts.{e}.down_proj.weight` | `[2048, 1024]` × 64 → stacked | BF16 (→ 4-bit) | `SwitchGLU.downProj` |

Explicit absences vs LLaDA2.1-mini (verify all *do not* appear in `model.safetensors.index.json`):
- No `expert_bias` buffer.
- No `shared_expert.*` / `shared_experts.*` weights.
- No `mlp.gate_proj/up_proj/down_proj` (LMOE has no dense layers — all `mlp_type == 'moe'`).
- No `query_key_value.weight` — split QKV only.

**M1' acceptance**: zero unmatched keys in either direction. Estimated 4-bit footprint at the LLaDA2.1-mini keep-list (router FP32, embeddings/norms 16-bit, routed experts + attention QKV/o + lm_head 4-bit): ~4.5 GB routed experts + ~0.9 GB embeddings/lm_head/norms + ~0.3 GB attention 4-bit + ~50 MB router FP32 → ≈ **5.7 GB on disk** (**inferred**, verify at M1'). Well within the M1's 16 GB.

### 3.4 ExactPrefixCache under LMOE — the correctness question

This is the single most important design decision for LMOE. Read carefully.

**LLaDA2.1-mini** ships (in NeoDiffusion, per your §6 mask decision) a `.strict` 0/-inf block-causal mask: within-block bidirectional, cross-block causal. Under that mask, when you commit block *b*, positions in block *b* attend only to blocks 0..*b* — they do *not* see block *b+1*'s masks. Therefore committed K/V at block *b* is invariant across future blocks. That's what makes `ExactPrefixCache` exact (Phase 2 §5 deviation 1, `handoff-post-M5.md` §2).

**LMOE**'s reference `forward` sets `attention_mask=None` and `is_causal=False` — attention is truly bidirectional over the entire padded (prompt + generation) sequence. **Every token attends to every other token**, including the still-masked future tokens. That means when you commit block *b*, block *b*'s K/V *do* depend on block *b+1*'s masks — a bunch of `mask_id` tokens whose K/V will change once *b+1* enters denoising and starts unmasking. **`ExactPrefixCache` under LMOE's shipped mask is not exact**, it is approximate. Using it will diverge from the reference at some point in generation.

You have four options; pick with André.

- **(a) Ship LMOE with `ExactPrefixCache` disabled.** No cache reuse across block commits — every step re-runs full L-length attention over the whole padded window. This is what the reference does. It's slow (that's the point of dInfer's `KVCacheFactory('dual')`) but *exact* by construction. Fine for correctness anchoring; fine for baseline TPS reporting; not what you want to ship in production.
- **(b) Ship LMOE with a *strict* mask that NeoDiffusion imposes (analogous to your LLaDA2.1-mini §6 decision).** Argue that the correct semantics is block-causal / within-block-bidirectional — the same argument the LLaDA2.1 paper makes for that model — and force it via a 0/-inf mask, then use `ExactPrefixCache` as before. This is a genuine deviation from LMOE's shipped inference behaviour and needs its own evidence: LMOE's paper (arXiv:2509.24389) may or may not specify block-causal attention as the algorithmic intent; the code just implements bidirectional. This is *not* the same situation as LLaDA2.1-mini's 0/1 bias bug — LMOE's code is internally consistent, it just picks bidirectional. If the paper endorses bidirectional, you cannot argue this away.
- **(c) Ship LMOE with `ActiveBlockCache` only, no `ExactPrefixCache`.** Recompute prompt+committed prefix every step (approximate cache freshness for the recent-blocks region via Elastic-Cache-style policies from Phase 3). This is the honest "the model is bidirectional; embrace it" answer. Slower than (b) but doesn't cheat.
- **(d) Prompt-only prefix cache.** Cache K/V for the prompt tokens once at generation start (they never carry a mask, since prompts have no `mask_id`), refresh committed-block K/V approximately (Phase 3), never treat committed-block K/V as exact. Compromise between (a) and (b): the prompt cache is *actually* exact (prompt tokens' K/V under bidirectional attention over `prompt + mask*N` is stationary iff the mask-token embeddings are — which they are, `mask_id` embedding is constant — but the K/V *depends on the presence and count of mask tokens in the sequence*; recomputing shows they DO change when total_length is fixed and only content beyond the prompt changes step-to-step. Actually, **the prompt K/V is not exact either under bidirectional attention** — the mask tokens the prompt attends to change as denoising progresses. Cross that one off.)

**My honest opinion**: (b) is the wiki-consistent answer *if* the LMOE paper claims block-causal semantics; (a) is the honest answer if it doesn't. Do **not** go with your current default of "assume ExactPrefixCache works" — under LMOE it silently doesn't, and the failure mode is quality drift that only shows up on real prompts, not on toy fixtures. Read arXiv:2509.24389 §2 (algorithm) or wherever the sampling procedure is specified before choosing. Track this as an M1'/M2' resolution item.

If you choose (b), you are effectively saying NeoDiffusion has a stronger opinion about diffusion-LM attention semantics than the reference implementation — you already do for LLaDA2.1-mini (§6 mask), and doing it a second time for LMOE for the same reason is defensible. It also means the entire Phase 3 Elastic-Cache track continues to attack `ActiveBlockCache` staleness only, which is a nicer story.

### 3.5 ScheduledLowConfidencePolicy — the LMOE sampling loop, MLX pseudocode

```swift
public struct ScheduledLowConfidencePolicy: SamplingPolicy {
    public let blockLength: Int          // 32 by README default
    public let stepsPerBlock: Int        // stepsTotal / numBlocks, precomputed at init
    public let temperature: Float        // 0.0 → argmax path
    public let cfgScale: Float           // 0.0 → skip CFG doubling
    public let maskId: Int32             // 156_895
    public let eosId: Int32              // 156_892

    public func generate(
        model: LLaDAFamilyModel,
        prompt: MLXArray,        // [B, promptLen], Int32 token ids
        genLength: Int
    ) -> Output {
        let B = prompt.dim(0), P = prompt.dim(1)
        precondition(genLength % blockLength == 0)
        let numBlocks = genLength / blockLength
        let totalLen = P + genLength

        // x: [B, totalLen] filled with maskId then prompt written back
        var x = MLXArray(repeating: maskId, [B, totalLen])
        x[.stride(), 0..<P] = prompt
        let promptIndex = (x .!= maskId)      // [B, totalLen] Bool

        for blockIdx in 0..<numBlocks {
            let blockStart = P + blockIdx * blockLength
            let blockEnd = blockStart + blockLength

            // maskIndex for THIS block only (mask outside block ignored by selection guard below)
            let blockMask = (x[.stride(), blockStart..<blockEnd] .== maskId)
            let numTransfer = getNumTransferTokens(blockMask, steps: stepsPerBlock)  // [B, stepsPerBlock]

            for step in 0..<stepsPerBlock {
                let maskIndex = (x .== maskId)                             // [B, totalLen] Bool

                // Full-context forward — no KV cache under option (a)/(c); use ExactPrefixCache under (b)/(d).
                let logits = model(x)                                      // [B, totalLen, V]

                // temp 0 → argmax; nonzero → gumbel path (rarely used at inference)
                let x0 = argmax(logits, axis: -1)                          // [B, totalLen]

                // Confidence = softmax(logits)[argmax]
                let p = softmax(logits, axis: -1, precise: true)           // FP32
                let x0p = takeAlong(p, x0.expandedDimensions(axis: -1),
                                    axis: -1).squeezed(axis: -1)           // [B, totalLen]

                // Selection guard: mask future blocks out of this iteration's candidates
                var confidenceGuard = x0p
                confidenceGuard[.stride(), blockEnd..<totalLen] = -Float.infinity

                // Only masked positions are candidates
                let confidence = which(maskIndex, confidenceGuard, MLXArray(-Float.infinity))

                // Top-K per batch, K = numTransfer[b, step].
                // KEY POINT: `numTransfer[b, step]` is a Python-int in the reference, i.e. a readback.
                // We can keep it GPU-resident: precompute cumulative schedule once and pick via scatter+sort.
                // See §4.2 for the GPU-resident selection kernel.
                let transferMask = topKMaskGPUResident(
                    confidence, kPerRow: numTransfer[.stride(), step])     // [B, totalLen] Bool

                // Write x0 into x at transferMask positions; leave promptIndex untouched (it never has maskId anyway)
                x = which(transferMask, x0, x)
            }
        }

        return Output(tokens: x[.stride(), P..<(P + genLength)], /* … */)
    }
}
```

Notes on the sketch (all **inferred**, verify in M4'):
- The `numTransfer` schedule is a Python-int table in the reference — you want it as an MLX `Int32` array `[B, stepsPerBlock]` computed once at block start; per-step selection is then `topK(confidence, k=numTransfer[..., step])`. MLX-Swift's `argSort` gives ordered indices; extract the `numTransfer[b, step]`-largest positions via a scatter into a boolean mask. See §4.2 for the concrete kernel.
- `argmax` + `softmax(logits)[argmax]` computed together shares the FP32 upcast — do them under one autoreleased graph region and `.eval()` at the end of each step (not per op).
- No block-diffusion mask helper is needed — the reference passes no mask. If you go with option (b), the mask helper is the same `BlockDiffusionMask.strict` you already have.
- The engine's `Output.blockCommits` still fires once per outer-loop iteration (block-boundary streaming stays valid).

### 3.6 ThresholdParallelPolicy — the dInfer-style variant

Sketch, per the model card's dInfer snippet (**sourced** the invocation, **inferred** the mechanism from the class name):

```swift
public struct ThresholdParallelPolicy: SamplingPolicy {
    public let blockLength: Int          // 32
    public let maxStepsPerBlock: Int     // safety bound; e.g. blockLength
    public let threshold: Float          // 0.9 per dInfer default
    public let maskId: Int32
    // ... same fields as ScheduledLowConfidencePolicy

    // Per step: instead of "select top-K by confidence", select all positions with x0_p > threshold.
    // Guaranteed progress: if none exceed threshold, unmask the single top-1 by confidence.
    // Loop terminates when the block has no maskId tokens left OR maxStepsPerBlock reached.
}
```

This is much closer in spirit to the LLaDA2.1-mini Q Mode (thresholded selection with a top-1 fallback) than to `ScheduledLowConfidencePolicy` — the Γ branch of your existing `DecodingPolicy` is almost directly reusable, minus the Δ branch and minus `max_post_steps` (LMOE has no editing loop). The dInfer `KVCacheFactory('dual')` is speculative naming — I would not port it until dInfer's source has been read; leaving it as `ExactPrefixCache` (option b) or `ActiveBlockCache`-only (option c) is fine for a first landing.

---

## 4. MLX-Swift implementation sketches

Both the basic (fixture-diff-passing) and the optimised (Metal-custom-op-fused) variants. All snippets are pseudocode grounded in the existing NeoDiffusion patterns (see `/home/user/workspace/LLaDA2MoE.swift` — same style).

### 4.1 SplitQKVMHAAttention — basic MLX

```swift
public final class SplitQKVMHAAttention: Module, Attention {
    @ModuleInfo(key: "q_proj") public var qProj: Linear
    @ModuleInfo(key: "k_proj") public var kProj: Linear
    @ModuleInfo(key: "v_proj") public var vProj: Linear
    @ModuleInfo(key: "o_proj") public var oProj: Linear
    @ModuleInfo(key: "q_norm") public var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") public var kNorm: RMSNorm

    public let numHeads: Int
    public let headDim: Int
    public let rope: PartialRotaryEmbedding    // rotaryDim = headDim → full RoPE

    public init(config: LLaDAMoEConfig) {
        self.numHeads = config.numAttentionHeads
        self.headDim = config.hiddenSize / config.numAttentionHeads
        // No GQA on LMOE: numKeyValueHeads == numAttentionHeads
        precondition(config.numKeyValueHeads == config.numAttentionHeads,
                     "LMOE is MHA, not GQA. If this fires, the config drifted.")
        self._qProj = ModuleInfo(wrappedValue:
            Linear(config.hiddenSize, config.numAttentionHeads * headDim, bias: false),
            key: "q_proj")
        self._kProj = ModuleInfo(wrappedValue:
            Linear(config.hiddenSize, config.numKeyValueHeads * headDim, bias: false),
            key: "k_proj")
        // ... v_proj, o_proj similarly
        self._qNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: headDim, eps: config.rmsNormEps),
                                 key: "q_norm")
        self._kNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: headDim, eps: config.rmsNormEps),
                                 key: "k_norm")
        // Full rotary on LMOE
        self.rope = PartialRotaryEmbedding(rotaryDim: headDim, theta: config.ropeTheta)
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,                // [B, L, H]
        positionIds: MLXArray,        // [L]
        mask: MLXArray? = nil         // [B, 1, L, L] additive; nil under LMOE default (option a/c)
    ) -> MLXArray {
        let B = x.dim(0), L = x.dim(1)
        // Project
        var q = qProj(x).reshaped(B, L, numHeads, headDim)             // [B, L, H_q, D]
        var k = kProj(x).reshaped(B, L, numHeads, headDim)             // [B, L, H_kv, D] (H_kv == H_q here)
        var v = vProj(x).reshaped(B, L, numHeads, headDim)

        // q/k norm applied per-head-dim BEFORE RoPE, in FP32 internally (RMSNorm handles that already).
        q = qNorm(q.reshaped(-1, headDim)).reshaped(B, L, numHeads, headDim)
        k = kNorm(k.reshaped(-1, headDim)).reshaped(B, L, numHeads, headDim)

        // Full RoPE (rotaryDim == headDim), positions absolute, freqs FP32.
        let (cos, sin) = rope.cosSin(positionIds: positionIds, dtype: x.dtype)
        q = rope.apply(q, cos: cos, sin: sin)                          // [B, L, H, D]
        k = rope.apply(k, cos: cos, sin: sin)

        // Transpose to [B, H, L, D]
        q = q.swappedAxes(1, 2); k = k.swappedAxes(1, 2); v = v.swappedAxes(1, 2)

        // Fused SDPA (softmax internally FP32). No mask under LMOE-default option (a)/(c);
        // pass a strict block-causal mask under option (b).
        let ctx = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0 / sqrt(Float(headDim)), mask: mask
        )                                                              // [B, H, L, D]

        return oProj(ctx.swappedAxes(1, 2).reshaped(B, L, -1))         // [B, L, H]
    }
}
```

Optimised variant (§5): fuse `qProj`, `kProj`, `vProj` into a single 3H·D-wide GEMM at load time (the weights concatenate along the output dim; the reference happens to store them split, but you can concatenate on load into a single MLX array and drop two of the three `Linear` modules). This is a **load-time** optimisation, not a runtime one — you still call SDPA the same way. Saves two kernel launches per attention block per step, i.e. `2 × 16 layers × stepsPerBlock` launches per block. On the M1 dev machine that's ~64–128 fewer launches/block — measurable in Instruments; probably negligible in wall-clock vs the MoE dispatch cost. Land it after the router fusion, not before.

### 4.2 SoftmaxTopKRouter — basic MLX

```swift
public final class SoftmaxTopKRouter: Module, MoERouter {
    public let topK: Int
    public let numExperts: Int
    public let weight: MLXArray                          // [numExperts, hiddenSize], FP32-forced

    public init(hiddenSize: Int, numExperts: Int, topK: Int) {
        self.topK = topK
        self.numExperts = numExperts
        self.weight = MLXArray.zeros([numExperts, hiddenSize], dtype: .float32)
        super.init()
    }

    // Returns (indices [T, topK] Int32, weights [T, topK] FP32, logits [T, numExperts] FP32)
    public func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let logits = matmul(x.asType(.float32), weight.transposed())          // [T, numExperts]
        // Reference: F.softmax(router_logits, dim=1, dtype=torch.float) then topk
        let probs = softmax(logits, axis: -1, precise: true)                  // FP32
        // Plain top-K (no groups) — MLX exposes `top(_ :k :axis:)` for values; use argSort for indices.
        let sortedIdx = argSort(-probs, axis: -1)[.stride(), ..<topK]         // [T, topK], descending
        var weights = takeAlong(probs, sortedIdx, axis: -1)                   // [T, topK]
        // norm_topk_prob is hardcoded False in the reference — DO NOT renormalise.
        // routed_scaling_factor is 1 — no multiplication needed.
        return (sortedIdx, weights, logits)
    }
}
```

Differences vs `LLaDA2MoEGate` you can verify by diffing against `/home/user/workspace/LLaDA2MoE.swift`:
- No `expert_bias` field or add.
- No `nGroup`/`topkGroup` reshape or group-score computation.
- No `argSort → mask → -inf → argSort` two-pass.
- No `/ (weights.sum(...) + 1e-20)` renormalisation (`norm_topk_prob=False` in the reference, hardcoded).
- No `× routedScalingFactor` (config value is 1).
- Softmax instead of sigmoid on the logits.

### 4.3 SparseMoEBlock — basic MLX (no shared expert)

```swift
public final class LLaDAMoESparseMoEBlock: Module {
    @ModuleInfo(key: "gate") public var gate: SoftmaxTopKRouter
    @ModuleInfo(key: "experts") public var experts: SwitchGLU

    public init(config: LLaDAMoEConfig) {
        self._gate = ModuleInfo(wrappedValue:
            SoftmaxTopKRouter(hiddenSize: config.hiddenSize,
                              numExperts: config.numExperts,
                              topK: config.numExpertsPerTok),
            key: "gate")
        self._experts = ModuleInfo(wrappedValue:
            SwitchGLU(hiddenSize: config.hiddenSize,
                      intermediateSize: config.expertIntermediateSize,   // 1024
                      numExperts: config.numExperts),
            key: "experts")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let flat = x.reshaped(-1, shape.last!)               // [T, H]
        let (indices, weights, _) = gate(flat)               // [T, k], [T, k]
        let expertOut = experts(flat, indices: indices)      // [T, k, H]
        let combined = (expertOut.asType(.float32) * weights.expandedDimensions(axis: -1))
            .sum(axis: 1)
            .asType(x.dtype)
        // No shared-expert add on LMOE.
        return combined.reshaped(shape)
    }
}
```

### 4.4 GPU-resident top-K selection for the sampling policy

This is the piece of `ScheduledLowConfidencePolicy` that must NOT hit CPU. The reference uses a Python `for j in range(B)` loop with `torch.topk(confidence[j], k=num_transfer_tokens[j, i])` — `k` is a Python int, i.e. a readback. On the M1 you have B=1 so the loop is trivial, but the `k` readback still forces a sync.

Sketch (**speculative** — I have not benchmarked MLX's `argSort` at these shapes on M1 GPU):

```swift
/// Given confidence [B, L] and kPerRow [B] Int32 (values in [0, L]), returns a boolean mask
/// [B, L] True at the top-kPerRow[b] positions of confidence[b], False elsewhere.
///
/// Approach: precompute ranked indices once (argSort descending), then compare each position's
/// rank against kPerRow[b] with a broadcast. Zero CPU readbacks.
func topKMaskGPUResident(_ confidence: MLXArray, kPerRow: MLXArray) -> MLXArray {
    let B = confidence.dim(0), L = confidence.dim(1)
    let rank = argSort(-confidence, axis: -1)              // [B, L] each row: indices of top-1, top-2, …
    // Convert rank → per-position rank (inverse permutation)
    let arange = MLXArray(0 ..< Int32(L))
    let perPositionRank = MLXArray.zeros([B, L], dtype: .int32)
    // scatter: perPositionRank[b, rank[b, r]] = r
    perPositionRank.scatterAlong(rank, values: arange.broadcasted(to: [B, L]), axis: -1)
    // Compare per-position rank < kPerRow[b]
    let k = kPerRow.reshaped([B, 1])                        // [B, 1]
    return perPositionRank .< k                             // [B, L] Bool
}
```

The `scatterAlong` API is illustrative — MLX-Swift's exact scatter operator name will differ; check `MLXArray.scattered(...)` / `mx.scatter`. The alternative if scatter is awkward is:

```swift
// Alternative: sort once, then set mask via take_along.
let sortedIdx = argSort(-confidence, axis: -1)             // [B, L]
let rowKMask = (MLXArray(0..<Int32(L)).broadcasted(to: [B, L])) .< kPerRow.reshaped([B, 1])
// rowKMask is [B, L] Bool: True in first kPerRow[b] positions of the *sorted* order.
// Scatter back to original positions:
var mask = MLXArray.zeros([B, L], dtype: .bool)
mask.scatterAlong(sortedIdx, values: rowKMask, axis: -1)
return mask
```

Either way: `argSort`, `broadcast <`, `scatter`. Three ops on GPU, no readback. Verify with a `syncPoints` counter (M5-style) that this variant does not add a sync — it shouldn't.

### 4.5 The forward pass under LMOE

```swift
public final class LLaDAMoEInnerModel: Module {
    public let config: LLaDAMoEConfig
    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo public var layers: [DecoderLayer]
    @ModuleInfo public var norm: RMSNorm

    // ... init wires 16 DecoderLayers, all with MoE (no first_k_dense special case)

    public func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        var h = embedTokens(inputIds).asType(.float32)   // per Phase 2 §2.7 policy
        let L = inputIds.dim(1)
        let positionIds = MLXArray(0 ..< Int32(L))
        for layer in layers {
            h = layer(h, positionIds: positionIds, mask: nil)   // no mask under LMOE default
        }
        return norm(h)                                    // [B, L, H]
    }
}
```

The `LLaDAMoEModel` top-level wraps this with `lm_head`. `logits(forTokens:)` (the M4-parity convenience) drops the block-mask argument entirely under LMOE default — no mask is passed to attention.

### 4.6 What optimised looks like — before Metal

Before dropping to Metal, the following MLX-only optimisations apply to LMOE and are cheap:
- **Fused QKV load-time concatenation** (§4.1 note): saves 2 kernel launches / layer / step.
- **Router matmul + softmax + top-K in one graph region**: `mx.eval()` at the end of the router, not inside it. Already the pattern in your existing gate; keep it.
- **`gather_qmm` for routed experts**: already implemented in `SwitchLinear`/`QuantizedSwitchLinear` (see `LLaDA2MoE.swift` lines 78–105). LMOE with 64 experts × intermediate 1024 is a *larger per-expert* GEMM than LLaDA2.1-mini's 256×512, which on Metal means each dispatch is more compute-bound and the launch overhead is a smaller fraction of the total — LMOE will look *closer* to the roofline than LLaDA2.1-mini even without fusion. (**inferred**, verify with Instruments.)
- **Skip the shared-expert add** — LMOE has none. This alone removes one full FFN forward per layer per step vs LLaDA2.1-mini's structure. It's the single biggest reason LMOE will run faster than LLaDA2.1-mini on the same hardware in absolute terms.
- **Prompt+block precomputed position ids**: LMOE forward runs over the full padded sequence every step, so position ids are `arange(0, totalLen)` and identical across steps within a block. Compute once at `generate` start, not per step. Trivial; skip if MLX already caches this.

---

## 5. Metal sketches — where the sandbox actually lives

Two kernels are worth writing by hand for LMOE. Both are speculative-payoff, and both should wait until §4 has landed and been profiled — **do not write Metal code before Instruments says the corresponding MLX region is a hot path**. That is Phase 3 discipline; I'm mentioning them here so the abstraction supports them.

### 5.1 Fused routed-MoE dispatch (the LMOE version of Alpha-MoE)

Aleph Alpha's Alpha-MoE (see `Resources/alpha-moe-megakernel.md`) fuses Up-Proj + Gate GEMM → SwiGLU → activation quant → Down-Proj + local combine into one persistent kernel. On Hopper. On Metal you cannot copy it — no WGMMA, no persistent producer/consumer pipelines, no FP8 W8A8. But the *principle* — keep intermediates in threadgroup memory instead of writing them back to global — is directly applicable.

For LMOE specifically, the fused kernel wants to compute, for one token *t* against one expert *e* (looped over the top-8 experts for that token):

```
// pseudo-Metal, threadgroup-per-token, one thread per intermediate lane
threadgroup float gate_out[intermediate_size];    // 1024 × sizeof(float) = 4 KB — fits in 32 KB threadgroup
threadgroup float up_out[intermediate_size];      // another 4 KB
threadgroup float acc[hidden_size];               // 8 KB — LMOE hidden 2048 in fp32; use half if precision allows

for e in top_8_experts_for_token(t):
    // Load expert e's gate_proj[e], up_proj[e] slices; do W·x → gate_out, up_out
    // (dequant on load if 4-bit: sample scale group, multiply, accumulate)
    threadgroup_barrier(mem_flags::mem_threadgroup);
    gate_out[l] = silu(gate_out[l]) * up_out[l];  // SwiGLU in-place
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Down_proj[e] · gate_out → acc, weighted by router_weight[t, e_slot]
    acc[h] += router_weight * dot(down_proj[e][h, :], gate_out);
```

Contrast with the Alpha-MoE 8-row-interleaved WGMMA trick: it does not port. The Metal-native equivalent is (**speculative**):
- **Threadgroup memory for `gate_out * up_out`** (~4 KB per token) — this is the "shared memory residency" translation.
- **One kernel per (token × top-K) tile, iterated on top-K inside the kernel** — one dispatch per token, top-K expert visits looped internally. Removes 7 dispatches out of every 8 for the routed MoE. (**speculative** on payoff — worth measuring against MLX's `gather_qmm` baseline.)
- **Weight-load coalescing** — pull expert weights via `[[threadgroup]]` argument buffers if possible; otherwise texture-cache them. LMOE's 64 experts × 3 projections × 1024 × 2048 = ~800 MB per layer at BF16, ~200 MB at 4-bit, per layer — no way to keep multiple layers' experts resident, but *one* layer's worth (200 MB × 16 layers = 3.2 GB) can arguably stay in unified memory pinned to GPU. On the M2 Ultra 192 GB, this is trivial; on the M1 16 GB, the routed 4-bit experts total ~4.5 GB — no residency issues, everything's in UMA already.
- **Skip the Alpha-MoE SwiGLU-interleaving trick** — it requires WGMMA-shape weight layouts. It won't help you and the layout rewrite is real cost.

**When to write this kernel**: only if Instruments shows the routed-expert `gather_qmm` dispatch dominating the step (>40% of step wall-clock on the Studio at BF16), AND after the router+attention epilogue fusions (§5.2). Otherwise the MLX baseline is enough.

### 5.2 Fused attention epilogue

Under LMOE the attention post-processing is: `SDPA output → oProj → residual add → post-attention RMSNorm → MoE`. The residual + norm combo is a classic Metal fusion target (norm reads the residual anyway).

```
// Kernel: fused_residual_rmsnorm(h_out_of_attn: half*, residual: half*, gamma: half*, out: half*)
// One threadgroup per token, one thread per hidden lane.
threadgroup float sq_sum = 0;
for h in hidden_lanes:
    float x = residual[h] + h_out_of_attn[h];  // fused add
    scratch[h] = x;
    sq_sum += x * x;                            // reduction
sq_sum = threadgroup_reduce(sq_sum);
float rms = rsqrt(sq_sum / H + eps);
for h in hidden_lanes:
    out[h] = scratch[h] * rms * gamma[h];
```

Applies verbatim to LLaDA2.1-mini too. Land it in the shared codebase, not the LMOE-specific package. This is Phase 3 §4-cross-cutting-track item 2 (attention epilogue fusion).

### 5.3 What NOT to write in Metal

- **A block-diffusion mask kernel.** Under LMOE default, no mask is passed. Under option (b) with `.strict`, `BlockDiffusionMask.strict` is already efficient — MLX broadcasts the analytic form; no Metal savings to expect.
- **A fused softmax+top-K router.** The router is 8-way top-K over 64 experts (LMOE), i.e. `argsort` over 64 elements per token. It's fast enough in MLX; a Metal port is not worth the maintenance burden.
- **Anything that needs FP8.** Apple Silicon doesn't natively support FP8; software-emulated FP8 buys you nothing on unified memory (§6.2). `Resources/per-block-fp8-quantization.md`'s ceiling doesn't port.

### 5.4 The raw-Metal escape hatch (§13.1)

Phase 1 §13.1 gates the raw-Metal decode-loop port on loop-control overhead being >10–15% of step time after Phase 3 fusion. LMOE at 16 layers × 8 experts × ~1024 intermediate is a substantially *more* compute-per-step model than LLaDA2.1-mini at 20 layers × 8 experts × ~512 intermediate — the per-step compute is roughly `2×` (LMOE experts are 2× wider, 1× less deep, same active count). If the escape-hatch gate depends on loop overhead being a nontrivial fraction of step time, LMOE will trip that gate *later* than LLaDA2.1-mini, not earlier. Do not expect LMOE to force a raw-Metal port sooner. (**inferred**, verify with a step-time breakdown once M6' lands.)

---

## 6. Which Resources optimisations apply to LMOE

Cross-reference against `Plans/Optimisations.md` and `Plans/phase-3-optimisation-roadmap.md`. For each Resources concept: does it apply to LMOE? What changes vs LLaDA2.1-mini?

Legend: ✅ = applies unchanged; ⚙️ = applies but needs re-tuning / a variant; ❌ = does not apply; ❓ = depends on the mask-semantics decision (§3.4).

### 6.1 Base / structural (Phase 2 territory)

| Concept | LLaDA2.1-mini | LMOE | Note |
|---|---|---|---|
| `block-wise-causal-attention` | ✅ (via `.strict` mask) | ❓ (option b) / ❌ (option a/c) | If you don't ship a strict mask, LMOE isn't block-causal. |
| `mask-to-token-m2t` | ✅ (Γ set) | ⚙️ | LMOE Γ is *scheduled* (`num_transfer_tokens`), not thresholded. Same idea, different selection rule. |
| `token-to-token-t2t` | ✅ (Δ set) | ❌ | LMOE has no editing. |
| `block-diffusion` | ✅ | ✅ | Both models chunk generation into blocks; block-boundary streaming valid. |
| `editable-state-evolution` | ✅ | ❌ | Editing state doesn't exist on LMOE. |
| `speedy-mode-s-mode` / `quality-mode-q-mode` | ✅ | ❌ | Threshold presets require T2T; LMOE has none. dInfer's threshold=0.9 is the *only* mode analogue. |
| `block-level-cache` / `prompt-caching` | ✅ (exact) | ❓ | Under LMOE default: prompt cache is *approximate* (§3.4). Under option (b): exact. |

### 6.2 Caching optimisations

| Concept | LLaDA2.1-mini | LMOE | Note |
|---|---|---|---|
| `adaptive-kv-caching` | ✅ (Phase 3) | ✅ | Model-agnostic policy layer over `ActiveBlockCache`. |
| `elastic-cache` / `elastic-cache-v2` | ✅ (WP-1a) | ⚙️ | The full pipeline (sliding window β=16, drift test γ=0.9, depth boundary ℓ★) applies verbatim to LMOE — LMOE's 16 layers give you fewer ℓ★ candidates to sweep than LLaDA2.1-mini's 20, so tuning is cheaper. **But**: under LMOE default (bidirectional over full sequence), `elastic-cache`'s premise (that KV drift comes primarily from *recent* denoising steps, so shallow layers stabilise fast) still holds — it's a claim about denoising dynamics, not about attention mask shape. This is actually a good fit. |
| `block-wise-mask-caching` | ✅ | ✅ | Cache distant-MASK KV in blocks, reuse when the block enters the active window. Under LMOE bidirectional, this is even more valuable — the "distant MASK tokens are length-bias only" observation is stronger when they ARE contributing to the forward via bidirectional attention. |
| `layer-wise-kv-dynamics` / `depth-aware-refresh` | ✅ | ✅ | Depth-monotonic KV drift is a structural claim; LMOE's 16 layers should still show the same monotonicity. |
| `selective-layer-refresh` | ✅ | ✅ | Same as depth-aware. |
| `sliding-window-attention` (as an Elastic-Cache instantiation) | ✅ | ✅ | Restrict prediction to β active MASK positions. Works under bidirectional attention just as under block-causal — it's a *decoding* window, not an *attention* window. |
| `denoising-step-importance` | ✅ | ✅ | Model-agnostic; step importance is a policy signal. |

### 6.3 Sampling / decoding

| Concept | LLaDA2.1-mini | LMOE | Note |
|---|---|---|---|
| `hierarchical-decoding` | ✅ | ✅ | Model-agnostic tree/schedule over blocks. |
| `configurable-threshold-decoding` | ✅ (baseline) | ⚙️ | LMOE's ThresholdParallelDecoder is exactly this. The dInfer variant lives here. |
| `denoising-step-importance` | ✅ | ✅ | See above. |
| `iteration-smoothing` (from `Resources/iteration-smoothing.md`, not read here — flag) | ? | ? | Not evaluated. Read before committing. |
| `most-attended-drift` (Elastic-Cache signal) | ✅ | ✅ | Attention-based signal; model-agnostic given the attention interface. |

### 6.4 Quantisation

| Concept | LLaDA2.1-mini | LMOE | Note |
|---|---|---|---|
| Group-64 affine 4-bit (current default) | ✅ | ✅ | Same policy: router FP32, embeddings/norms 16-bit, experts/QKV/o_proj/lm_head 4-bit. LMOE has no shared expert so the "keep shared-expert 16-bit" line vanishes. |
| `per-block-fp8-quantization` | ❌ (Apple Silicon, no native FP8) | ❌ | Same reason. Both models are BF16 upstream; software FP8 gains nothing on UMA. |
| Group-32 on routed experts (M8 fallback) | ✅ | ✅ | If LMOE's 4-bit quality drops too far — LMOE's 64 experts × intermediate 1024 might tolerate coarser quantisation better than LLaDA2.1-mini's 256 × 512 (larger per-expert matrices → group scales average over more values). (**speculative**, sweep.) |

### 6.5 Speculation / MultiBD

| Concept | LLaDA2.1-mini | LMOE | Note |
|---|---|---|---|
| Training-free MultiBD (`N_buf = 2`) | ✅ (WP-1b, co-first) | ⚙️ | The `[[mbd-lms]]` numbers were measured on LLaDA2.1-Mini specifically (TPF 4.50→6.50, −0.59pp math). LMOE has no published number; do not assume portability. But the mechanism (`τ_add` activation gating + `τ_semi` semi-completion fallback + per-slot `ActiveBlockCache`) is architecture-agnostic. LMOE's simpler sampling loop actually makes MultiBD *easier* to slot in — no `max_post_steps` interaction to worry about. |
| Spiffy (auto-speculation) | ⚙️ | ⚙️ | Spiffy's losslessness argument depends on block-causal attention. Under LMOE-default bidirectional, the argument breaks — verifying a draft state is no longer trivially block-causal. Under LMOE option (b) with `.strict`, Spiffy's losslessness restored. Same as LLaDA2.1-mini for the numeric-precision caveat (bf16 batched attention). |
| S2D2 (self-speculative decoding) | ⚙️ | ⚙️ | Same story. |
| FreeDave (draft-verification) | ⚙️ | ⚙️ | Same story. |

### 6.6 Kernel-level

| Concept | LLaDA2.1-mini | LMOE | Note |
|---|---|---|---|
| `alpha-moe-megakernel` (Aleph Alpha) | ⚙️ (Metal-inspired) | ⚙️ (same) | Direct port impossible on Metal (no WGMMA, no FP8). The *principle* (fuse Up+Gate → SwiGLU → Down + local combine, keep intermediates in threadgroup memory) ports; see §5.1. LMOE's larger intermediate (1024 vs 512) is arguably a *better* fit for Metal threadgroup residency — 4 KB fits comfortably in the 32 KB shared limit. |
| Fused attention epilogue (residual + RMSNorm) | ✅ (Phase 3 §4-cross-cutting-item-2) | ✅ | Model-agnostic. |

### 6.7 Out of scope (unchanged)

`in-place-chain-of-thought`, `multi-block-editing-mbe`, `multi-turn-forward-mtf` — training-time / editing-family concepts. Both models exclude them per Phase 1 §1.

---

## 7. Memory, quantisation, fixture logistics

### 7.1 Footprint estimates (LMOE)

All **inferred** — verify at M1'.

| Item | 4-bit (M1 dev, 16 GB) | BF16 (Studio, 192 GB) |
|---|---|---|
| Routed experts (16 × 64 × 3 projections × ~2M params each = ~6.3 B weights) | ~3.6 GB @ 4.5 bpw eff. | ~12.6 GB |
| Attention QKV+o (16 × 4 × 4M each = ~256 M) | ~0.15 GB | ~0.5 GB |
| Embeddings (157 184 × 2048) | ~0.6 GB (kept 16-bit) | ~0.6 GB |
| lm_head (157 184 × 2048, quantised @ 4-bit unless M8 sweep says otherwise) | ~0.18 GB | ~0.6 GB |
| Norms + q/k_norm | ~1 MB | ~1 MB |
| Router (16 × 64 × 2048 FP32) | ~8 MB (kept FP32) | ~8 MB |
| KV cache @ 4k padded (approx: 16 layers × 16 heads × 128 × 2 (K,V) × 4096 tokens × 2B BF16) | ~536 MB (much bigger than LLaDA2.1-mini's ~168 MB — no GQA, so KV is 4× larger per token) | ~536 MB |
| Activations / scratch (block 32, full-context forward L=4096) | ~1–2 GB | ~2–4 GB |
| **Total** | **~5.1 GB + ~1.5 GB activations = ~6.6 GB peak** | **~14.7 GB + ~4 GB = ~19 GB peak** |

**The KV cache size is the number to double-check.** LMOE is MHA (no GQA) — every KV head is a Q head — so per-token KV is 16 heads × 128 × 2 × 2B = 8 KB/token, vs LLaDA2.1-mini's 4 heads × 128 × 2 × 2B = 2 KB/token. At the max_position 8192 LMOE advertises: 8 KB × 8192 × 16 layers = **1 GB just for full-context KV**. Still trivial on the Studio; nontrivial on the M1 if you push to 8k contexts.

**LMOE fits on the M1 at 4k contexts. It probably fits at 8k too but with less headroom.** This is a genuinely useful dev-machine target — you can do end-to-end runs, not just toy-config runs. That's a real unblocking result versus LLaDA2.1-mini.

### 7.2 Conversion path (M1' logistics)

- Downloading the 3 × 5 GB shards on the M1 is fine (still ~120 GB free assuming the LLaDA2.1-mini checkpoint is not on the same disk).
- **Quantising on the M1 is fragile** for the same reason `convert_weights.py` is fragile for LLaDA2.1-mini (accumulating the whole converted dict in memory before writing). Refactor `convert_weights.py` to stream shard-by-shard *before* running LMOE conversion, then it'll be fine at ~15 GB peak.
- BF16 fixture dumps for LMOE need the full BF16 model in memory + PyTorch forward — that's ~15 GB weights + ~4 GB activations + Python overhead → **~20–24 GB peak**. Doesn't fit the M1 either. Needs the Studio or a rented cloud box.
- 4-bit LMOE artefact + tokenizer files copied back to the M1: ~5 GB — fine for local task-level runs and `diffusion-bench` numbers.

**Recommendation**: mirror the LLaDA2.1-mini flow. BF16 fixtures + 4-bit conversion on a bigger machine, copy the ~5 GB 4-bit artefact + tokenizer to the M1 for local runs. If a Studio isn't available, a Linux box with ≥24 GB RAM covers the PyTorch fixture-dump side; the 4-bit conversion needs MLX (macOS/Metal), so *that* step still wants a Mac. A cloud Mac (e.g. MacStadium hourly) with 32+ GB unified memory does both. This is unchanged from the LLaDA2.1-mini gate.

### 7.3 Fixture plan (M4' / M5' analogues)

Same shape as M4/M5 for LLaDA2.1-mini:
- `Tools/generate_core_fixtures.py` extended to take LMOE weights + config; produce per-module intermediate + full-forward logits.
- `Tools/generate_loop_fixtures.py` extended to transcribe LMOE's `generate()` verbatim (§2.1) and emit per-step selection sets + final `x`.
- Toy config for M1-runnable unit tests: vocab 1000, hidden 128, **2** layers, **4** experts / 2 active, no groups, no shared expert, intermediate 128, block 16, 4 heads MHA. Special ids remapped (`mask_id=999`, `eos_id=998`) as with LLaDA2.1-mini.
- The parity gates in `Tests/DiffusionGenerationTests/` are `SamplingPolicy`-parameterised: same fixture-driven test runs three policies × two models with the right (policy, model) fixture dir.

### 7.4 Studio BF16 real-weight parity (M4/M5 blocker) is unchanged

LMOE landing does **not** unblock the outstanding M4/M5 BF16 parity gate for LLaDA2.1-mini. That still needs a Studio or equivalent for the 33 GB BF16 forward. LMOE lets you run *LMOE* end-to-end on the M1; it does not shrink the LLaDA2.1-mini checkpoint.

---

## 8. Milestones (analogous to Phase 2 §4)

Suggested slot: land after M5's remaining BF16 parity closes on the Studio, **before** M6 freezes the diffusion-bench baseline. Rationale: M6 baselines whatever model + policy is in place, and you want both models + all three policies benched from the same starting line.

| Milestone | Content | Size | Depends on |
|---|---|---|---|
| **M5.1 — Family refactor** | §3.2 refactor steps 1–5 (rename, parameterise, protocolise; no new families yet). All 26/26 existing tests stay green. | S | M5 done |
| **M5.2 — LMOE config + weight load** | `LLaDAMoEConfig` + weight-name map (§3.3) + streaming `convert_weights.py`. Zero unmatched keys. | M | M5.1 |
| **M5.3 — LMOE core blocks** | `SplitQKVMHAAttention`, `SoftmaxTopKRouter`, `SparseMoEBlock (no shared)`, `LLaDAMoEInnerModel`. Fixture-diff tests on toy config (M1 machine) + real weights (Studio, BF16). | L | M5.2 |
| **M5.4 — LMOE forward parity** | Full forward against a **strict-mask baseline OR bidirectional-baseline** depending on §3.4 decision. 100% top-1 agreement; max |Δprob| < BF16 tolerance. | M | M5.3 |
| **M5.5 — LMOE sampling policies** | `ScheduledLowConfidencePolicy` (§3.5) + fixture-based parity against the README code. Optionally `ThresholdParallelPolicy` (§3.6). Sync-point budget ≤1 per block × K-step speculation baseline. | L | M5.4 |
| **M6' — Widened diffusion-bench** | Extend `Tools/diffusion-bench` to run { LLaDA2.1-mini, LMOE } × { policies }. Freeze two model baselines and three policy baselines. | M | M5.5, M6 |
| **M7' — Server model swap** | `/v1/models` returns both; request-level `model` field routes; unchanged block-commit streaming. | S | M6', M7 |
| **M8' — LMOE quant sweep** | Same axis order as M8 (group-64 → group-32 experts → 6-bit experts → lm_head quant on/off). BF16 anchor on Studio. | M | M5.5, M8 |

Everything above `M5.1` is optional if you decide to defer LMOE until after M6 baselines LLaDA2.1-mini. My honest read: **do M5.1 (the refactor) before M6 regardless**, and slot M5.2–M5.5 either before or after M6 as availability allows. The refactor is small if you sequence it as steps 1–5 in §3.2; it's much bigger later.

---

## 9. Risks / opens carried into implementation

1. **§3.4 mask decision is unresolved.** Must be resolved with André at M5.2, before writing forward code. If the LMOE paper (arXiv:2509.24389) specifies block-causal semantics, take option (b); if it endorses bidirectional, take option (a) or (c). I have not read the paper.
2. **dInfer's `KVCacheFactory('dual')` and `BlockIteratorFactory(True)` mechanisms are unknown.** Read `dInfer` source before implementing `ThresholdParallelPolicy`; it may reveal that dInfer already uses a strict block-causal mask internally (which would settle §3.4 in favour of option b) or that it embraces bidirectional and pays the cost (option c).
3. **The README `generate()` at steps=128, gen_length=128, block=32 is *1 step per block*.** Confirm that this is a sensible default and not a demo shortcut; if you want fine-grained denoising, LMOE probably wants steps ≫ block_length, e.g. `steps = block_length × num_blocks` (i.e. 1 step per token). Ask André / re-read the LMOE paper.
4. **CFG (`cfg_scale > 0`) is not wired in NeoDiffusion.** LMOE's default is 0. If you ever want CFG, the engine's forward path needs to double the batch dim — the abstraction cost is small if you add it now, painful later.
5. **`ThresholdParallelDecoder(0, threshold=0.9)`'s first constructor arg is `0`.** Unknown meaning (start_step? min_transfer? warm-up steps?). Verify against dInfer source before assuming.
6. **`_tied_weights_keys = ["lm_head.weight"]` is dead metadata in `LLaDAMoEModelLM`** — `tie_word_embeddings=False` in config, and I saw no actual weight-sharing code. Treat embeddings and lm_head as separate weights (they are, in the safetensors index). Flag if the actual safetensors are shared — I'd be surprised.
7. **`is_causal = False` combined with `attention_mask = None` in the reference means the LMOE forward will happily attend past the current outer-loop block's end.** This is what motivates the `x0_p[:, blockEnd:] = -inf` selection guard. Reproduce this guard faithfully — it is a *selection* correction, not an *attention* correction.

---

## 10. What to say to Claude Code

Copy-paste-friendly kick-off objective:

> Read `/Users/andrebarlocher/Documents/Swift/NeoDiffusion/Plans/lmoe-implementation-guide.md` in full. Then execute **M5.1 only** (§3.2 refactor steps 1–5): rename LLaDA2RMSNorm → RMSNorm, parameterise PartialRotaryEmbedding on rotaryDim+theta, extract an `Attention` protocol and rename `LLaDA2Attention` → `FusedQKVGQAAttention`, introduce a `LLaDAFamilyConfig` protocol with `LLaDA2MoeConfig` conforming, and extract a `SamplingPolicy` protocol from `DiffusionEngine` with the existing loops packaged as `DraftAndEditPolicy`. All 26 existing tests must stay green after each of the 5 steps — verify with `swift test` between steps. Do NOT start §3.3+ (LMOE weight map, new modules) in this milestone. Update `Plans/phase-2-implementation-guide.md` with an M5.1 status entry mirroring the M4/M5 status style. If you find any of the 5 refactor steps requires touching more than one package's public API, stop and ask before proceeding.

If that comes back green, follow up with M5.2 by naming §3.3 and pointing at the LMOE HF config URL. Everything downstream depends on M5.1 landing without regressing LLaDA2.1-mini.

---

## 11. What I'm *not* certain about (be sceptical here)

- The 5.7 GB 4-bit LMOE footprint estimate assumes MLX's affine quant at group 64 reaches ~4.5 bpw effective on 64 experts × 1024 intermediate. That's a **speculative** extrapolation from mlx-lm's usual results on dense LLaMA-class models; MoE fine-grained experts might compress worse. Verify at M5.2 conversion.
- The claim "LMOE will look closer to the roofline than LLaDA2.1-mini even without fusion" (§4.6) rests on the intuition that larger per-expert GEMMs are more compute-bound. On Metal specifically, at 4-bit with `gather_qmm`, I have not measured this. Treat as a hypothesis to test in M6', not a promise.
- The **Elastic-Cache fit for LMOE** section (§6.2) argues that the depth-monotonic KV drift claim ports to LMOE's 16 layers. This is a structural claim from `layer-wise-kv-dynamics.md`; empirically it was measured on LLaDA-1.5 / LLaDA2.x. LMOE's paper likely reports similar dynamics but I haven't checked.
- The **§3.4 approximate-vs-exact ExactPrefixCache argument for LMOE** hinges on a careful reading of what "bidirectional attention over prompt + mask sequence" implies for cache reuse across block commits. I am confident in the direction (cache is not exact under bidirectional attention when the mask population changes step-to-step) but the specific engineering choice (a/b/c/d) needs the paper's algorithmic intent to settle cleanly. Read arXiv:2509.24389 §2 before committing.
- The **dInfer perf ceiling doesn't port**: `1000+ TPS on 8× H800` is a distributed inference number with 8-way tensor parallelism, fused FP8 MoE, and CUDA-graphed sampling. NeoDiffusion is single-node, MLX, 4-bit, on Apple Silicon. Expect a fraction of that number; report the fraction honestly. Do not pick a fraction to target — measure what you get.
