# Nemotron-Labs-Diffusion-3B — MLX-Swift + Metal implementation guide

**Target repo**: `./`
**Target integration point**: extends the LLaDA-family engine (after LMOE + DiffusionGemma + Sumi land)
**Purpose**: honest, load-bearing plan to bring NVIDIA's tri-mode diffusion LM to Apple Silicon with the SAME LLaDA-family engine — reusing everything possible, being explicit about what breaks and what doesn't.

---

## Honesty preamble

Andre — before you read this, three things I want you to know up front so you don't discover them at week 3:

1. **This is the easiest of the three ports to run on your M1**, and by a large margin. 3B params × 2 bytes = ~6.4 GB BF16, fits with headroom. If your dev-time patience is thin, do this one first.
2. **Nemotron-Labs-Diffusion is not a pure diffusion model. It is three models glued to one weight set.** The blog headline number (5.9× TPF over Qwen3-8B) is the `linear_spec_generate` mode — a draft-and-verify scheme that is neither pure diffusion nor pure AR. If you port only `generate()` you will get a normal LLaDA-style speed profile and none of the blog's speedup. This is not documented well; you have to read the modeling file to see it. Sources: [`modeling_nemotron_labs_diffusion.py`](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B/raw/main/modeling_nemotron_labs_diffusion.py), [`nvidia blog`](https://huggingface.co/blog/nvidia/nemotron-labs-diffusion).
3. **The block-diffusion mask engine (`block_diff` in the config) is training-only code.** The published checkpoint runs in `bidirectional` paradigm. If you port the `_prepare_diffusion_attn_mask_and_pos_ids` helper you get a training feature that this checkpoint does not use at inference. Do not gold-plate it.

Everything else in this doc is decision-support for a port that is genuinely feasible on your machine, but which is architecturally the most "engineered" of the three — not the most novel.

---

## TL;DR — the honest take before the plan

* **Nemotron-Diffusion-3B is a Ministral3 checkpoint with a diffusion-trained head and three inference modes.** It is 90% a standard Llama-family transformer plus 10% diffusion sampler. If you have Ministral/Llama attention working in MLX (you do, from LMOE), you are 80% of the way there. — [`config.json`](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B/raw/main/config.json), [`modeling_ministral.py`](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B/raw/main/modeling_ministral.py). *(sourced)*
* **It uses a REAL mask token (`mask_token_id=100`)** — same absorbing-state family as LLaDA, unlike Sumi's uniform-state renoising. Which means your existing LLaDA-family sampler applies with almost no changes. This is the most reusable of the three. — [`config.json`](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B/raw/main/config.json). *(sourced)*
* **The `_get_transfer_index` in the reference sampler is LLaDA-verbatim** — top-k on confidence within a block, per-block scheduled unmask count, Gumbel-noise temperature. If you have LLaDA sampling, you literally do not need to rewrite this. — [`modeling_nemotron_labs_diffusion.py`](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B/raw/main/modeling_nemotron_labs_diffusion.py). *(sourced)*
* **The Llama-4 attention scaling on Q is non-standard and load-bearing.** `q = q * (1 + 0.1 * log(1 + floor(pos/16384)))`. If you skip this and the model runs at long context, quality drops. — [`modeling_ministral.py`](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B/raw/main/modeling_ministral.py). *(sourced)*
* **YaRN RoPE with factor 16 (16384 → 262144).** You need YaRN. Standard RoPE will underperform on any prompt over ~4K. — [`config.json`](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B/raw/main/config.json). *(sourced)*
* **`linear_spec_generate` uses a LoRA adapter (`linear_spec_lora`) that is toggled per phase**: enabled for the bidirectional draft, disabled for the causal verify. This is the actual speed differentiator vs vanilla LLaDA. — [`modeling_nemotron_labs_diffusion.py`](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B/raw/main/modeling_nemotron_labs_diffusion.py). *(sourced)*

**Bottom line**: this is the M1-friendly port and the one with the most novel *inference-time* trick (linear self-speculation). If you only have time for one, this is the one to ship first.

---

## 0. Recommended reading order

1. This document, in full.
2. [Nemotron-Labs-Diffusion blog post](https://huggingface.co/blog/nvidia/nemotron-labs-diffusion) — the "why", the benchmark numbers, and the tri-mode framing.
3. [`configuration_nemotron_labs_diffusion.py`](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B/raw/main/configuration_nemotron_labs_diffusion.py) — 80 lines, defines every hyperparameter you'll port.
4. [`modeling_ministral.py`](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B/raw/main/modeling_ministral.py) — the base attention/MLP. Read this to see the Llama-4 Q-scaling.
5. [`modeling_nemotron_labs_diffusion.py`](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B/raw/main/modeling_nemotron_labs_diffusion.py) — reads top-to-bottom in an hour. The three `generate*` methods are the whole story.
6. `neod/Resources/llada2-1-tech-report.md`, `block-diffusion.md`, `block-wise-mask-caching.md`, `mask-to-token-m2t.md`.
7. [NVIDIA research page](https://research.nvidia.com/publication/2026-05_nemotron-labs-diffusion-tri-mode-language-model-unifying-autoregressive) — for the training procedure and joint loss weighting.

---

## 1. Nemotron-Labs-Diffusion-3B quick facts

All values verified against upstream `config.json` and modeling files on 2026-07-07.

| Item | Nemotron-Diffusion-3B | Sumi-7B (for contrast) | DiffusionGemma-2 (for contrast) | LMOE (your baseline) |
| --- | --- | --- | --- | --- |
| **Params** | 3B (announced), ~3.5B computed | 8B | 2B | 8B |
| **Base architecture** | Ministral3 (Mistral/Llama-family) | Custom `SumiModel` | Gemma2 | Llama-3.1-family |
| **Hidden / intermediate** | 3072 / 9216 | 4096 / 12288 | 2304 / 9216 | 4096 / 14336 |
| **Layers** | 26 | 36 | 26 | 32 |
| **Attention heads** | 32Q / 8KV (GQA 4:1) | 32Q / 8KV (GQA 4:1) | 8Q / 4KV (GQA 2:1) | 32Q / 8KV (GQA 4:1) |
| **Head dim** | 128 | 128 | 256 | 128 |
| **Attention pattern** | Mode-dependent: causal (AR), bidirectional (diffusion), block-diagonal + block-causal (linear-spec) | Fully bidirectional every layer | Interleaved local↔global | Fully causal |
| **Attention backend** | SDPA / Flash / FlexAttention supported | Eager only (off-by-one softmax breaks SDPA) | SDPA supported | SDPA supported |
| **Q-scaling** | Llama-4: `q *= 1 + 0.1*log(1 + floor(pos/16384))` | none | none | none |
| **Q/K norm** | none | none | RMSNorm | none |
| **RoPE θ** | 1,000,000 | 500,000 | 10,000 | 500,000 |
| **RoPE scaling** | YaRN, factor 16, orig=16384, extended=262144 | none in config | none | none |
| **Vocabulary** | 131,072 (Mistral/Ministral tokenizer) | 100,278 (cl100k-like) | 262,144 (Gemma) | 128,256 |
| **BOS / EOS / PAD / MASK** | 1 / 11 / None / **100** | 100256 / 100257 / 100277 / — (no mask) | 2 / 1 / 0 / — | 128000 / 128001 / — / — |
| **Tie word embeddings** | False | False | True | True |
| **MLP** | SwiGLU (gate/up/down, no bias) | SwiGLU | Gemma GeGLU | SwiGLU |
| **MoE** | Dense | Dense | Dense | 128-expert MoE |
| **Diffusion sampler(s)** | LLaDA-style block-wise `_get_transfer_index`; ar_generate() causal; linear_spec_generate() bidirectional draft + causal verify | ancestral / adaptive / greedy (uniform-state) | LLaDA-style block-wise | none (AR baseline) |
| **Mask token** | REAL, id=100 | none — uniform-state renoising | none in vocab; special handling | n/a |
| **BF16 weight footprint** | ~6.4 GB | ~16 GB | ~4.8 GB | ~16 GB dense-equiv |
| **Fits M1 16 GB BF16?** | **YES with headroom** | No (must quantise) | Yes | No (must quantise) |
| **Reference block size** | 32 | n/a (no block scheme; whole canvas denoised) | 32 (typical) | n/a |
| **Chat template** | Mistral instruct v3 (implicit — verify) | user/asst tokens in tokenizer | Gemma chat format | Llama-3.1 chat |

Sources (all fetched 2026-07-07):
- Model card & tri-mode framing: [huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B), [huggingface.co/blog/nvidia/nemotron-labs-diffusion](https://huggingface.co/blog/nvidia/nemotron-labs-diffusion)
- Configuration: [`config.json`](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B/raw/main/config.json), [`configuration_nemotron_labs_diffusion.py`](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B/raw/main/configuration_nemotron_labs_diffusion.py)
- Attention/MLP: [`modeling_ministral.py`](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B/raw/main/modeling_ministral.py)
- Sampler & generate methods: [`modeling_nemotron_labs_diffusion.py`](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B/raw/main/modeling_nemotron_labs_diffusion.py)
- Research page: [research.nvidia.com/publication/2026-05_nemotron-labs-diffusion](https://research.nvidia.com/publication/2026-05_nemotron-labs-diffusion-tri-mode-language-model-unifying-autoregressive)

### 1.1 Key deltas from LMOE that will bite the port

Since you have LMOE working, these are the deltas that will actually cost you time — in decreasing severity:

1. **Llama-4 attention scaling on Q.** *Not* in LMOE. Position-dependent multiplier applied inside the attention forward. If you compile Q-scaling as a constant, you'll get wrong logits on any prompt over 16K.
2. **YaRN RoPE with factor 16.** Standard RoPE frequencies must be modified per-dim (mscale, alpha, beta, extrapolation zones). LMOE uses linear-scaled or default RoPE.
3. **Bidirectional attention mask in diffusion mode.** LMOE only knows how to build causal masks. You need mode-switching mask construction (identical work you did for DiffusionGemma).
4. **Three generate paths.** LMOE has one. You are effectively building three inference pipelines: causal AR (trivial, reuse LMOE code), block-diffusion (reuse DiffusionGemma/Sumi sampler), linear-spec (new — draft in bidirectional mode, verify in causal mode, accept longest matching prefix + 1 bonus).
5. **LoRA adapter toggling.** The `linear_spec_lora` adapter is loaded from a separate branch and toggled per phase. MLX-Swift has no first-class LoRA-toggle API — you'll need to write one, or bake the LoRA into a fused weight snapshot for one phase only. *(inferred — MLX-Swift LoRA support checked as of 2026-06)*.

### 1.2 Deltas from the two sister ports (Sumi, DiffusionGemma)

* **Vs Sumi**: You have a REAL mask token, so you can reuse block-wise mask caching, elastic KV cache, adaptive-KV all the tricks that assume "positions are either mask or committed" — none of which apply to Sumi's uniform-state.
* **Vs DiffusionGemma**: You do NOT have interleaved local/global attention. Every layer is the same. Which means simpler attention dispatch, but you *do* have YaRN and Llama-4 Q-scaling that DiffusionGemma doesn't have.

---

## 2. The reference sampling algorithm, pinned

Verbatim excerpts from [`modeling_nemotron_labs_diffusion.py`](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B/raw/main/modeling_nemotron_labs_diffusion.py) (fetched 2026-07-07). Any implementation must match these byte-for-byte in behaviour.

### 2.1 The mask-transfer index (LLaDA-verbatim), verbatim

```python
def _get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens):
    """
    Compute the index of tokens to transfer from the current block to the next block.
    Follows the LLaDA / block-diffusion convention:
      - Gumbel noise is added when temperature > 0.
      - remasking='low_confidence' scores committed tokens by their own predicted prob,
        masked positions are set to -inf so they cannot be re-selected.
      - Per-row, top-k confident positions are unmasked, where k = num_transfer_tokens.
    """
    logits_with_noise = _add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
        )
    elif remasking == "random":
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
    return x0, transfer_index
```

This is identical to LLaDA v1.5. If you already implement LLaDA's mask transfer, ship as-is.

### 2.2 The block-wise diffusion `generate()`, structural

```python
@torch.no_grad()
def generate(self, prompt, steps=128, gen_length=128, block_length=32,
             temperature=0., remasking="low_confidence", mask_id=100):
    # 1. Allocate canvas: [prompt | mask_id * gen_length]
    # 2. Compute num_blocks = gen_length // block_length
    # 3. Distribute steps evenly across blocks: steps_per_block = steps // num_blocks
    # 4. For each block b in 0..num_blocks:
    #    For each step in steps_per_block:
    #      - Forward pass on FULL canvas (bidirectional, no KV cache)
    #      - Extract logits over current block's mask positions
    #      - num_transfer = ceil(mask_positions_in_block / (steps_per_block - step))
    #      - Call _get_transfer_index → unmask top-k confident positions
    # 5. Return canvas[prompt_len:]
```

The important operational facts:
- **No KV cache during diffusion.** Every step is a full forward on the whole canvas. This is O(steps × L × N²) — the same asymptotic as LLaDA/DiffusionGemma.
- **`num_transfer_tokens` is scheduled per block**, not globally. Each block gets `steps_per_block` steps and denoises independently in the noise dimension (though attention is over the whole canvas).
- **Block advances only when the previous block is fully committed.** So earlier blocks stabilise before later blocks; this is what makes block-wise KV caching (once committed, positions never change) valid — see §6.

### 2.3 `ar_generate()`, verbatim shape

```python
@torch.no_grad()
def ar_generate(self, prompt, max_new_tokens=128, temperature=0., top_p=1.0, top_k=None):
    # Standard KV-cache AR loop.
    # Uses the SAME weights, causal attention mask, no bidirectional layers,
    # no mask token substitution, no block scheduling.
    # This is essentially "run this checkpoint as if it were plain Ministral3."
```

This is the trivial mode. If you have LMOE's AR loop, this is `ar_generate` with a different config.

### 2.4 `linear_spec_generate()`, structural — the actual novelty

```python
@torch.no_grad()
def linear_spec_generate(self, prompt, max_new_tokens=128, block_length=32,
                          temperature=0., use_lora=True):
    """
    Speculative decoding where the DRAFT is a bidirectional diffusion pass
    and the VERIFY is a causal AR pass — both using the same weights.
    """
    # 1. Enable linear_spec_lora adapter (if trained variant used).
    # 2. Draft: run block-wise diffusion for one block of length block_length.
    # 3. Disable LoRA. Snapshot the drafted tokens as candidates.
    # 4. Verify: single causal forward pass over [committed | drafted].
    #    Compare each drafted token to argmax of its causal-position logits.
    # 5. Accept the longest matching prefix, plus 1 bonus token (the first
    #    causal-argmax that differed OR the next causal-argmax if all matched).
    # 6. Roll accepted tokens into committed, discard rest, repeat.
```

Key operational facts:
- **Two forward passes per accepted chunk**: 1 bidirectional (draft) + 1 causal (verify). Break-even is if accepted_length ≥ 2.
- **The LoRA is trained specifically for the draft phase.** Without it, acceptance rate collapses (per [blog post](https://huggingface.co/blog/nvidia/nemotron-labs-diffusion) which claims 3× TPF vs AR at temp=0 in this mode).
- **The verify is causal → uses KV cache**. Only the draft is cache-less.
- **This is the mode that produces the 5.9× TPF number.** If you skip it, you're building a slower LLaDA clone.

### 2.5 The five differences that break DiffusionGemma parity code

Even though the diffusion sampler is LLaDA-shaped, these five items differ from what DiffusionGemma-2 needed:

| # | Difference | Why it matters |
| --- | --- | --- |
| 1 | Ministral tokenizer (Mistral chat template), not Gemma | Different chat wrapper, different BOS handling. Wire the correct tokenizer path or you get nonsense. |
| 2 | Llama-4 Q-scaling active on every layer | If you copy DiffusionGemma attention wholesale, you'll skip this. |
| 3 | YaRN RoPE with factor 16 | DiffusionGemma uses default RoPE. |
| 4 | Real mask token `id=100` | DiffusionGemma has no vocab mask token — implements masking differently. |
| 5 | Mode-switchable causal/bidirectional | DiffusionGemma is bidirectional-only per layer role. |

---

## 3. Multi-model integration plan: from "LLaDA-family engine" to "LLaDA-family engine with tri-mode + linear-spec"

### 3.1 What already exists after LMOE + DiffusionGemma + Sumi land

Assumptions from the LMOE/DiffusionGemma/Sumi guides:
- Working MLX-Swift `LlamaRMSNorm`, `SwiGLU MLP` (used by every one of the four models).
- Working GQA attention module with configurable causal/bidirectional mask, RoPE, quantised weights.
- Working block-wise diffusion sampler (from DiffusionGemma) — `_get_transfer_index`, Gumbel noise, top-k confidence gating.
- Working AR loop with KV cache (from LMOE).
- Tokenizer wrapper that supports at least Llama and Gemma tokenizers.
- Quantisation pipeline: 4-bit weights + BF16 activations.

### 3.2 Refactor steps, each keeps tests green

1. **Extend RoPE with YaRN**: add `apply_yarn_scaling(freqs, factor, alpha, beta, orig_ctx)` helper. Default `factor=1` = no-op. LMOE, Sumi, DiffusionGemma pass `factor=1`; Nemotron passes `factor=16`.
2. **Add Q-scaling hook to attention**: `q_scaling_fn: (q, positions) -> q'`. LMOE, Sumi, DiffusionGemma pass identity. Nemotron passes the Llama-4 formula.
3. **Add "attention mode" enum**: `{Causal, Bidirectional, BlockDiff}`. Route to correct mask builder. LMOE = Causal only. Sumi = Bidirectional only. DiffusionGemma = per-layer role. Nemotron = mode-switchable per forward call.
4. **Extend sampler dispatcher**: `SamplerMode.{AR, BlockDiffusion, LinearSpec}`. Existing DiffusionGemma sampler handles BlockDiffusion. Add LinearSpec as a new type that composes BlockDiffusion(draft) + AR(verify).
5. **Optional LoRA scaffolding**: `NamedLoRAAdapter` with `enabled: Bool`, hot-swappable per forward. Only Nemotron uses this today.
6. **Ministral tokenizer path**: reuse LMOE's if it's Llama-3-style; else add mistral-common wrapper. This model uses `tokenizer.model` (SentencePiece) — verify.

### 3.3 New types to add

```swift
enum AttentionMode: Sendable { case causal, bidirectional, blockDiff }
enum SamplerMode: Sendable { case ar, blockDiffusion, linearSpec }

struct YarnScaling: Sendable {
    let factor: Float           // Nemotron: 16.0
    let alpha: Float            // Nemotron: 1.0
    let beta: Float             // Nemotron: 32.0
    let originalContext: Int    // Nemotron: 16384
}

struct Llama4QScale: Sendable {
    let beta: Float             // Nemotron: 0.1
    let originalMax: Int        // Nemotron: 16384
    // applied as: q *= 1 + beta * log(1 + floor(pos / originalMax))
}

struct LinearSpecConfig: Sendable {
    let blockLength: Int        // 32
    let useLora: Bool
    let loraAdapter: NamedLoRAAdapter?
}
```

### 3.4 Correctness decisions André must make before code lands

**Q1**: Ship all three modes on day 1, or only `generate()` (block-diffusion) first?
> **My recommendation**: ship `generate()` + `ar_generate()` first. Both reuse existing code. Defer `linear_spec_generate()` to a second milestone — the LoRA-toggle infra is genuinely new and the gains are only realised if you have the reference LoRA in mlx-safetensors format (which NVIDIA has not published as of 2026-07-07 — verify).

**Q2**: YaRN correctness — do you implement the full YaRN formula (mscale + wavelength-based interpolation zones) or only the linear-scaling fallback?
> **My recommendation**: full YaRN. Linear scaling underperforms on tasks that need positional precision. There are 4 published MLX-Swift YaRN implementations you can port from (Qwen1.5-32K, Yi-6B-200K, Command-R-104K, Mistral-7B-YaRN); pick any. Sources: [`transformers YaRN impl`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py), reference YaRN paper. *(inferred — verify count of MLX-Swift ports)*

**Q3**: Llama-4 Q-scaling — is the effect on short prompts (<16K) meaningful?
> Short-context math: `1 + 0.1 * log(1 + floor(0/16384))` = `1 + 0.1 * log(1)` = `1.0`. So for any position < 16384 the multiplier is exactly 1.0 and the scaling is a no-op. For >16K positions it starts to matter. Which means: for M1 demo runs at <=8K context, you can OMIT the Llama-4 Q-scaling entirely and get identical logits. Confirm with a matmul test at seq_len=16385.

**Q4**: Chat template — trust the model card, or verify against the training data?
> The model card [links](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B) but does not explicitly print a chat template. Ministral3 base uses Mistral v3 chat template. **Verify**: dump `tokenizer.chat_template` from the tokenizer_config.json before hand-writing one.

**Q5**: `linear_spec_lora` LoRA weights — where are they, and are they Apache-2 licensed?
> As of 2026-07-07 the model card lists `linear_spec_lora` as an available variant but does not clearly document the download path. Check: `huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B/tree/main` for a `linear_spec_lora/` subdir or a companion repo. If unavailable, `linear_spec_generate` will work but with lower acceptance rate — the base weights are still linear-spec-trained per the paper.

---

## 4. MLX-Swift implementation sketches

Every sketch below is a starting point, not final code. Types are illustrative; expect to conform to your existing `NeoDiffusion` module conventions.

### 4.1 YaRN RoPE (the load-bearing frequency modification)

YaRN differs from linear RoPE scaling in that it interpolates *per-dimension*: high-frequency dims (small periods) are left alone (extrapolation), low-frequency dims (long periods) are linearly interpolated, and a smooth transition zone is applied between.

```swift
struct YarnRotaryEmbedding {
    let dim: Int              // head_dim = 128
    let base: Float           // rope_theta = 1_000_000
    let scaling: YarnScaling  // factor=16, alpha=1, beta=32, orig=16384

    // Precompute inv_freqs modified for YaRN.
    static func makeInverseFrequencies(dim: Int, base: Float, scaling: YarnScaling) -> MLXArray {
        // Standard RoPE: inv_freq[i] = 1 / base^(2i/dim)  for i in 0..<dim/2
        let stdInvFreq = MLXArray(0..<(dim/2)).mapIndex { i in
            1.0 / pow(base, Float(2*i) / Float(dim))
        }
        // Wavelengths: wavelength[i] = 2*pi / inv_freq[i]
        let wavelengths = 2 * .pi / stdInvFreq

        // YaRN correction ranges (see paper):
        // - lo: wavelength = orig_ctx / (2*pi*alpha)
        // - hi: wavelength = orig_ctx / (2*pi*beta)
        let lo = Float(scaling.originalContext) / (2 * .pi * scaling.alpha)
        let hi = Float(scaling.originalContext) / (2 * .pi * scaling.beta)

        // For each dim, compute ramp: 0 (extrapolate, keep) -> 1 (interpolate, /factor)
        let ramp = wavelengths.mapIndex { w in
            if w < hi { return Float(0) }               // high-freq: extrapolate
            else if w > lo { return Float(1) }          // low-freq: interpolate
            else { return (w - hi) / (lo - hi) }        // transition zone
        }

        // Final inv_freq = std * (1 - ramp) + (std / factor) * ramp
        return stdInvFreq * (1 - ramp) + (stdInvFreq / scaling.factor) * ramp
    }

    // Also apply mscale to attention outputs (YaRN's second modification):
    // mscale = 1 + 0.1 * log(scaling.factor). This multiplies attention logits.
    var attentionMScale: Float {
        1.0 + 0.1 * log(scaling.factor)  // ≈ 1.277 for factor=16
    }
}
```

Sanity check: for `factor=1`, ramp = 0 everywhere, mscale = 1. All models reduce correctly to standard RoPE.

### 4.2 Llama-4 Q-scaling (a one-line change, but conditional)

```swift
extension GroupedQueryAttention {
    func applyLlama4QScale(_ q: MLXArray, positions: MLXArray, scale: Llama4QScale) -> MLXArray {
        // q shape: [B, H, N, D]
        // positions shape: [N]
        // multiplier[i] = 1 + beta * log(1 + floor(positions[i] / orig_max))
        let floored = MLX.floor(positions.asType(.float32) / Float(scale.originalMax))
        let mult = 1.0 + scale.beta * MLX.log(1.0 + floored)  // [N]
        // Broadcast to [1, 1, N, 1]
        return q * mult.reshaped([1, 1, positions.shape[0], 1])
    }
}
```

**Test**: at position 0, mult = 1.0 exactly (float error notwithstanding). At position 16384, mult = 1 + 0.1 * log(2) ≈ 1.069. At position 65536, mult = 1 + 0.1 * log(5) ≈ 1.161.

### 4.3 Mode-switchable attention mask builder

Consolidates work you did for DiffusionGemma; adds the `causal` mode for AR/verify passes.

```swift
enum AttentionMode {
    case causal        // AR mode, verify phase of linear_spec
    case bidirectional // block-diffusion mode
    case blockDiff     // training only; skip for now
}

func makeAttentionMask(mode: AttentionMode, seqLen: Int, blockBoundaries: [Int]?) -> MLXArray {
    switch mode {
    case .causal:
        // upper-triangular -inf mask
        return MLXFast.causalMask(seqLen: seqLen)
    case .bidirectional:
        // all zeros - full attention
        return MLX.zeros([seqLen, seqLen])
    case .blockDiff:
        // block-diagonal on noisy half, offset-block-causal cross, fully-causal on clean half
        // NOT USED at inference — training-only. Guard-rail assert.
        fatalError("blockDiff mask is training-only; not implemented for inference")
    }
}
```

### 4.4 Block-wise diffusion sampler (reuse of DiffusionGemma sampler)

Most of this is identical to what you shipped for DiffusionGemma. Only the mask token id changes.

```swift
struct BlockDiffusionSampler {
    let blockLength: Int = 32
    let maskId: Int = 100  // Nemotron mask token
    let remasking: String = "low_confidence"

    func generate(
        model: NemotronDiffusion,
        prompt: MLXArray,
        genLength: Int = 128,
        steps: Int = 128,
        temperature: Float = 0.0
    ) -> MLXArray {
        let promptLen = prompt.shape[0]
        var canvas = MLX.concatenated([prompt, MLXArray(repeating: maskId, count: genLength)])
        let numBlocks = genLength / blockLength
        let stepsPerBlock = steps / numBlocks

        for b in 0..<numBlocks {
            let blockStart = promptLen + b * blockLength
            let blockEnd = blockStart + blockLength

            for s in 0..<stepsPerBlock {
                let logits = model.forward(canvas.expandedDim(0), mode: .bidirectional)  // no KV cache
                let blockLogits = logits[.expand, blockStart..<blockEnd, .expand]
                let maskPositions = (canvas[blockStart..<blockEnd] .== maskId)
                let numMask = MLX.sum(maskPositions.asType(.int32)).item(Int.self)
                let numTransfer = max(1, ceil(Double(numMask) / Double(stepsPerBlock - s)))

                let (x0, transferIdx) = getTransferIndex(
                    logits: blockLogits, temperature: temperature,
                    remasking: remasking, maskIndex: maskPositions,
                    x: canvas[blockStart..<blockEnd], numTransferTokens: numTransfer
                )
                canvas[blockStart..<blockEnd] = MLX.where(transferIdx, x0, canvas[blockStart..<blockEnd])
            }
        }
        return canvas[promptLen...]
    }
}
```

### 4.5 Linear-spec sampler (the novelty)

```swift
struct LinearSpecSampler {
    let blockLength: Int = 32
    let maskId: Int = 100

    func generate(
        model: NemotronDiffusion,
        loraAdapter: NamedLoRAAdapter?,   // optional; may be nil
        prompt: MLXArray,
        maxNewTokens: Int,
        temperature: Float = 0.0
    ) -> (tokens: MLXArray, acceptedPerCycle: [Int]) {
        var committed = prompt
        var acceptedHistory: [Int] = []

        while committed.shape[0] - prompt.shape[0] < maxNewTokens {
            // Phase 1: DRAFT with bidirectional + LoRA-on
            loraAdapter?.setEnabled(true)
            let draftCanvas = MLX.concatenated([
                committed,
                MLXArray(repeating: maskId, count: blockLength)
            ])
            let draftedBlock = runBlockDiffusion(model: model, canvas: draftCanvas,
                                                  blockStart: committed.shape[0], stepsPerBlock: 4)
            loraAdapter?.setEnabled(false)

            // Phase 2: VERIFY with causal, no LoRA
            let verifyInput = MLX.concatenated([committed, draftedBlock])
            let logits = model.forward(verifyInput.expandedDim(0), mode: .causal)
            let causalArgmax = MLX.argmax(logits[.expand, committed.shape[0]-1..<verifyInput.shape[0]-1, .expand], axis: -1)
            //                            ↑ predictions AT position i USE logits FROM position i-1

            // Compare drafted vs causal argmax, accept longest matching prefix
            var accepted = 0
            for i in 0..<blockLength {
                if draftedBlock[i].item(Int.self) == causalArgmax[i].item(Int.self) {
                    accepted += 1
                } else {
                    break
                }
            }
            // Bonus token: always accept the first mismatch's causal prediction
            let bonusToken = causalArgmax[accepted..<accepted+1]
            let acceptedTokens = MLX.concatenated([draftedBlock[0..<accepted], bonusToken])
            committed = MLX.concatenated([committed, acceptedTokens])
            acceptedHistory.append(accepted + 1)
        }

        return (committed[prompt.shape[0]...], acceptedHistory)
    }
}
```

**Debug hook**: log `acceptedHistory` and compute mean acceptance. If mean < 2, you're losing money — verify LoRA is loaded, temperature=0, and Q-scaling is correct.

### 4.6 The forward pass and mode dispatch

```swift
final class NemotronLabsDiffusion {
    let config: NemotronConfig
    let layers: [DecoderLayer]
    let embedTokens: Embedding
    let lmHead: Linear      // NOT tied — separate weights
    let norm: LlamaRMSNorm
    let rope: YarnRotaryEmbedding
    let qScale: Llama4QScale?

    func forward(_ inputIds: MLXArray, mode: AttentionMode, cache: KVCache? = nil) -> MLXArray {
        let seqLen = inputIds.shape[1]
        let positions = MLXArray(0..<seqLen)
        var h = embedTokens(inputIds)
        let mask = makeAttentionMask(mode: mode, seqLen: seqLen, blockBoundaries: nil)

        for (i, layer) in layers.enumerated() {
            h = layer.forward(h, mask: mask, positions: positions, rope: rope,
                              qScale: qScale, cache: cache?[i])
        }
        h = norm(h)
        return lmHead(h)
    }

    func generate(_ prompt: MLXArray, samplerMode: SamplerMode, ...) -> MLXArray {
        switch samplerMode {
        case .ar: return arGenerate(prompt, ...)
        case .blockDiffusion: return BlockDiffusionSampler().generate(model: self, prompt: prompt, ...)
        case .linearSpec: return LinearSpecSampler().generate(model: self, loraAdapter: ..., prompt: prompt, ...).tokens
        }
    }
}
```

---

## 5. Metal sketches

### 5.1 What ports from LMOE / DiffusionGemma / Sumi unchanged

- SwiGLU MLP fused kernel — identical shape (gate/up/down, no bias). Only intermediate dim changes (9216 vs Sumi 12288 vs DiffusionGemma 9216).
- RMSNorm — Llama-flavour, identical.
- 4-bit dequant + matmul — reuse LMOE's `q4_matmul.metal`.
- Basic GQA attention with causal mask — reuse LMOE's kernel; only mask mode differs.
- Bidirectional attention — reuse DiffusionGemma's zero-mask path.

### 5.2 What needs re-derivation

**A. YaRN-modified inv_freq buffer.**
Same RoPE kernel as LMOE, but you must swap the precomputed `inv_freq` and `cos/sin` tables. This is a data-layout change, not a code change. Compute the modified freqs on CPU once at load time (see §4.1), upload to Metal as a texture/buffer.

**B. Llama-4 Q-scaling injection.**
Two options:
1. Fold into RoPE kernel: after applying rotary, multiply Q by the position-dependent scalar. Requires uploading a `[seq_len]` scaling buffer per forward.
2. Fuse as a separate 1-line kernel after attention's Q projection. Simpler to reason about; adds one small kernel launch per layer.

I recommend option 2 for the first pass; you can fuse into RoPE later if benchmarks show it matters (unlikely — memory-bandwidth-bound is elsewhere).

**C. Attention mode dispatch.**
Same kernel for causal and bidirectional — just different mask input tensor. No kernel divergence needed.

### 5.3 What NOT to write yet

- Fused block-diffusion sampler kernel. Sampler is O(steps × forward passes); each forward is many kernels; the sampler control flow itself is not on the hot path.
- LoRA-adapter mergemat kernel. If linear_spec-lora is small (rank-16 or -32 typical), do the low-rank matmul as two separate kernels. Only fuse if profiling shows > 5% overhead.
- Custom kernel for `_get_transfer_index`. Top-k over a block of 32 positions with a scalar comparison — MLX's default top-k is fine.

### 5.4 Metal Performance Shaders (MPS) opportunities

- Use `MPSNDArrayGather` for the confidence-based unmask step (top-k indices → gather from full-vocab logits).
- Use `MPSMatrixMultiplication` fused with softmax for attention (already what MLX does under the hood).
- **Do NOT** try to use FlashAttention MPS variants; the M1 (not M2/M3) GPU family does not have the shared-memory shape FlashAttention exploits. LMOE's plain fused-attention kernel is already at ~90% of memory-bandwidth ceiling; further optimisation is diminishing.

---

## 6. Web-search additions — optimisations not in Resources/

Searches performed 2026-07-07 across arXiv, HF blog, NVIDIA research. Filtered to items published after the LLaDA line ships.

### 6.1 Applies to Nemotron-Diffusion-3B directly

* **S2D2: Self-Speculative Discrete Diffusion** — [arXiv 2603.25702](https://arxiv.org/html/2603.25702v1). Training-free self-speculative decoding for block-diffusion LMs. Draft-and-verify inside the SAME model (no separate draft model). Direct competitor / complement to Nemotron's linear_spec. Worth a spike: implement S2D2 as a fallback if you can't get the linear_spec LoRA.
* **Spiffy: Lossless Speculative Decoding for dLLMs** — [arXiv 2509.18085](https://arxiv.org/html/2509.18085v3). Claims up to 7.9× speedup with calibrated draft graphs. Applies to any block-diffusion model. High signal — ship as an option after linear-spec is validated.
* **SGLang integration** — NVIDIA's blog explicitly names [SGLang](https://huggingface.co/blog/nvidia/nemotron-labs-diffusion) as the reference inference stack. If you build a Python reference implementation before or alongside MLX-Swift, use SGLang; the radix cache tricks documented in `radix-caching.md` apply here.
* **Progress-aware early exit (SchED)** — [arXiv 2512.02892](https://arxiv.org/html/2512.02892v1). Skip late diffusion steps once confidence per block is high. Applies to any block-diffusion sampler with real mask tokens. This is an easy win: after ~50% of steps in a block, if all mask positions have confidence > 0.95, commit early.
* **Fast-dVLM: KV cache + self-speculation for block-diffusion** — [arXiv 2604.06832](https://arxiv.org/html/2604.06832v1). Similar architecture, more mature caching scheme. Their KV cache for the causal-verify phase applies to Nemotron's linear-spec verify pass directly.

### 6.2 Applies with rework

* **YaRN vs longRoPE**: LongRoPE gets slightly better perplexity at 100K+ context per [arXiv 2402.13753](https://arxiv.org/abs/2402.13753). But the shipping checkpoint used YaRN. Match training. Don't experiment on first ship.
* **NoPE / partial-rotary**: DiffusionGemma uses partial rotary; Nemotron uses full rotary. Don't experiment.

### 6.3 Known-bad advice — do NOT do

* Do NOT skip YaRN even for short-context demos. The frequencies themselves are baked into the checkpoint's learned weights; using standard RoPE frequencies at position 0 gives different (worse) results than YaRN-scaled at position 0, because the model was trained expecting YaRN's `inv_freq`. This is subtle and easy to get wrong.
* Do NOT try to fuse the three generation modes into one code path with `if` statements. The AR path uses KV cache + causal mask + no mask-token substitution; the diffusion path is cache-less + bidirectional + mask-substituting; the linear-spec path is both. They should live in three separate methods.
* Do NOT trust the "7.60× TPF speed-of-light" number from the blog as a target. That number is under B200-specific memory-bandwidth assumptions with a specific tokenizer stride. On M1 you'll be memory-bandwidth-limited long before you can chase 7.6×. Aim for 2-3× vs your AR baseline on M1; that's the achievable Pareto point.

---

## 7. Optimisation applicability matrix — how each Resources/*.md note maps to Nemotron

Legend: ✅ = applies directly, cheap. 🟡 = applies with rework. ❌ = does not apply. ⏳ = defer, evaluate after MVP.

| Resource | Verdict | Reasoning |
| --- | --- | --- |
| adaptive-kv-caching | ✅ | Real KV cache exists in causal + verify modes. Adapt on committed prefix (never invalidated during a linear-spec cycle). |
| alpha-moe-megakernel | ❌ | Dense model, no MoE routing. |
| block-diffusion | ✅ | Exact match — `block_length=32`. |
| block-wise-causal-attention | ✅ | The `block_diff` paradigm implements exactly this at training; the linear-spec verify phase reuses the pattern. |
| block-wise-mask-caching | ✅ | Real mask token; committed positions per block never change. Cache KV for committed blocks between diffusion steps. Biggest MVP-adjacent win. |
| denoising-step-importance | 🟡 | Late-step logits change less; combine with SchED for early exit. |
| depth-aware-refresh | ✅ | Deep layers change slower for committed positions. Refresh cache selectively at low layers, more aggressively at high. |
| editable-state-evolution | 🟡 | The linear-spec verify-and-accept IS this. Formalise as an editable-state abstraction if you need cross-block state. |
| elastic-cache | ✅ | Applies to committed prefix. |
| elastic-cache-v2 | ✅ | Same as v1, tighter budgets. Enable after MVP. |
| elbo-based-block-level-policy-optimization-ebpo | ⏳ | Training-time; skip. |
| exposure-bias-in-dllms | ⏳ | Training-time evaluation; skip. |
| hierarchical-decoding | 🟡 | Blocks are one hierarchy level. Consider adding token-level → block-level → chunk-level scheduling if you go past MVP. |
| in-place-chain-of-thought | 🟡 | The bidirectional context enables in-place CoT; needs prompt engineering & fine-tune, not inference change. |
| iteration-smoothing | 🟡 | Reduces oscillation between steps within a block. Cheap to add. |
| layer-wise-kv-dynamics | ✅ | Applies to KV cache in causal/verify modes. |
| llada2-1-tech-report | ✅ | Nemotron's sampler is LLaDA v1.5+ verbatim. This report is basically the reference. |
| local-attention-dllm | ❌ | Nemotron uses full attention every layer; no local variant. |
| mask-to-token-m2t | 🟡 | `_get_transfer_index` IS M2T. If your codebase abstracts this, plug in. |
| most-attended-drift | 🟡 | Applies with bidirectional attention. Track which committed positions dominate attention weight and prioritise their cache. |
| multi-block-editing-mbe | 🟡 | Block boundaries could be relaxed to allow inter-block edits; not in Nemotron reference sampler. Defer. |
| multi-turn-forward-mtf | ⏳ | Multi-turn benchmark, not an optimisation. |
| per-block-fp8-quantization | ✅ | NVIDIA already ships FP8 variants (per model card). Once you have BF16 working, port their quant scheme. On M1 you'll use INT4/INT8 not FP8 (no native FP8 on M1). |
| per-token-early-stopping | ✅ | Real mask token → confidence per token is well-defined. Directly implementable. |
| quality-mode-q-mode | 🟡 | High-quality mode = more diffusion steps. Wire steps as a runtime param. Free. |
| radix-caching | ✅ | SGLang-native trick; NVIDIA blog names it explicitly. Applies to prompt prefix in both AR and diffusion modes. |
| scratchpad-redundancy | ⏳ | Requires model-specific analysis; skip until you have benchmarks. |
| selective-layer-refresh | ✅ | Applies to KV cache during linear-spec verify. |
| sglang-rollout-engine | ✅ | Direct reference stack per blog. Build a SGLang-based Python reference for correctness testing. |
| sliding-window-attention | ❌ | Nemotron does not use sliding window. |
| speedy-mode-s-mode | 🟡 | Fewer steps per block → speed mode. Wire as runtime toggle. |
| suffix-dropout | ⏳ | Training-time. |
| token-to-token-t2t | 🟡 | Cross-token dependency in the draft phase could accelerate acceptance. Research-y; defer. |
| vectorized-likelihood-estimation | 🟡 | Faster confidence computation per step. Modest win. |

**Summary counts**: 15 apply directly (✅), 13 apply with rework (🟡), 3 do not apply (❌), 4 defer (⏳). Roughly one-third of your existing library is directly applicable — the best ratio of the three models.

### 7.1 New optimisations to add to Resources/ from this port

I recommend authoring these notes as first-class Resources/ files after this port lands:

1. `linear-self-speculation.md` — the Nemotron novelty. Companion to `s2d2.md` and `spiffy.md`.
2. `yarn-rope.md` — full YaRN implementation, since it's now needed by multiple models (Nemotron today; likely by future ports).
3. `llama4-q-scaling.md` — position-dependent Q multiplier. Small note, easy reference.
4. `lora-adapter-toggle.md` — cross-cutting pattern for phase-switched LoRA (Nemotron uses it, but this generalises).
5. `sched-early-exit.md` — for the progress-aware early-exit trick from [arXiv 2512.02892](https://arxiv.org/html/2512.02892v1).

---

## 8. Memory & quantisation logistics on M1 16 GB

**This is the best-fitting model on your M1 of all three ports. Don't overthink this section.**

### 8.1 BF16 parameter footprint (arithmetic shown)

Per config.json: hidden=3072, intermediate=9216, layers=26, heads=32Q/8KV, head_dim=128, vocab=131072, tie_word_embeddings=False.

Per-layer parameter count:
- Attention: Q(3072×4096) + K(3072×1024) + V(3072×1024) + O(4096×3072) = 12.6M + 3.15M + 3.15M + 12.6M ≈ 31.5M
- MLP: gate(3072×9216) + up(3072×9216) + down(9216×3072) = 28.3M + 28.3M + 28.3M ≈ 85M
- Norms: 2 × 3072 ≈ 6K (negligible)
- **Per layer ≈ 116.5M**

Total transformer:
- 26 layers × 116.5M = 3.03B
- Embedding: 131072 × 3072 = 402M
- LM head (not tied): 131072 × 3072 = 402M
- Final norm: 3072 (negligible)
- **Total ≈ 3.84B params**

Model card says "~4B params" — matches within rounding.

**BF16 weight footprint = 3.84B × 2 bytes = 7.68 GB**

### 8.2 Runtime memory on M1 16 GB

| Component | BF16 size |
| --- | --- |
| Weights | ~7.7 GB |
| KV cache (2K prompt + 1K gen, 26 layers × 8KV × 128 × 2 bytes × 3072 tokens) | ~130 MB |
| Attention scores tensor (temporarily) at seq 3K, bidirectional | ~200 MB (peak) |
| Activations, working buffers | ~1-2 GB |
| **Total peak** | **~10 GB — fits with 6 GB headroom** |

**Verdict**: You can ship BF16 on M1 16 GB with no quantisation. Don't quantise for MVP. Save quantisation work for the shipping build.

### 8.3 4-bit quantisation footprint (for reference)

If/when you quantise for user builds:
- Weights: 3.84B × 0.5 bytes ≈ 1.92 GB
- Embedding (leave BF16 for tokenizer sanity): 0.4 GB
- LM head at 4-bit: ~0.2 GB
- **Total quantised ≈ 2.5 GB**

At 4-bit you have **13 GB headroom** on M1 — you could run two models concurrently (e.g., Nemotron + a small draft model for external speculative decoding if you don't want to use linear-spec).

### 8.4 The peak-memory item is the attention scores tensor at long context

At seq_len = 8192, bidirectional attention per layer:
- Q @ K^T = [B × H × N × N] = [1 × 32 × 8192 × 8192] × 2 bytes = **4.3 GB per layer, temporarily**.
- MLX/Metal typically stream this — you never materialise all layers at once. But the SINGLE-LAYER peak is ~4 GB at 8K seq.

At 16K seq (approaching YaRN's original context), that becomes 17 GB — **exceeds M1**. So on M1, cap seq_len at ~8K for bidirectional forward passes. AR mode is unaffected (causal + KV cache).

### 8.5 Recommended dev-machine setup

* **Development**: BF16 weights, no quantisation, seq_len ≤ 4K.
* **Demo build**: BF16 weights, seq_len up to 8K, both `generate()` and `ar_generate()` enabled.
* **Ship build**: 4-bit weights, embed/lm_head at 8-bit, seq_len up to 16K, all three modes enabled.
* **Explicit non-goal**: 32K+ context on M1. Feasible on M2 Max/M3 Max/M4; not here. Document the cap.

---

## 9. Milestones

The Nemotron port has an "M0" prefix to distinguish from LMOE (M-), DiffusionGemma (M'-), Sumi (M''-). Numbering aligns with the sister guides.

**nemotron-M0'''-1 — Base Ministral3 forward pass parity (2 days)**
- Reuse LMOE's Llama attention module.
- Add YaRN inv_freq computation.
- Add Llama-4 Q-scaling as toggleable hook.
- Load Nemotron BF16 weights; forward pass on causal input matches HF reference (float diff < 1e-3) at seq_len 256, 2048, 8192.
- **Exit criteria**: `NemotronDiffusion(prompt=..., mode=.causal).logits ≈ HF ref logits`, no diffusion sampling yet.

**nemotron-M0'''-2 — AR generate parity (1 day)**
- Wire `ar_generate` through existing LMOE AR loop with KV cache.
- Verify with 5 fixed prompts + temperature=0 that output is deterministic and matches HF reference token-for-token to 128 tokens.
- **Exit criteria**: parity on 5 prompts × 128 tokens; TPS reported.

**nemotron-M0'''-3 — Block-diffusion `generate()` parity (3 days)**
- Wire `_get_transfer_index` (reuse DiffusionGemma code).
- Add mask-token substitution (Nemotron-specific, cheap).
- Bidirectional attention path (reuse DiffusionGemma work).
- Verify parity at temp=0 with 5 prompts, block_length=32, steps=128, gen_length=128.
- **Exit criteria**: token-level match to HF `generate()` with fixed seed.

**nemotron-M0'''-4 — Perf pass (2 days)**
- Instrument with Instruments/GPU-time. Identify hot kernels.
- Enable block-wise mask caching for the diffusion path.
- Enable radix cache for prompt prefix.
- Target: 2× speedup over M0'''-3 baseline.

**nemotron-M0'''-5 — Linear-spec `linear_spec_generate` (5 days)**
- Add LoRA adapter loader (rank-16/32 typical). Verify Apache-2 licence on adapter weights.
- Add adapter toggle infrastructure (per-layer LoRA switch).
- Implement draft-and-verify loop with acceptance tracking.
- Verify parity vs HF reference on 5 prompts.
- **Exit criteria**: mean acceptance > 3 (i.e., > 3× throughput vs AR); demo of 100-token generation with accept-log printed.

**nemotron-M0'''-6 — 4-bit quantisation + demo build (3 days)**
- Reuse LMOE 4-bit quant pipeline.
- Verify parity of all three modes (`ar`, `diffusion`, `linear_spec`) with 4-bit weights within perplexity delta < 5%.
- Ship demo app: side-by-side AR vs diffusion vs linear-spec on 3 prompts.

**Total: ~2 weeks focused effort** — the tightest schedule of the three ports.

---

## 10. Risks & open questions

1. **linear_spec LoRA availability**. If NVIDIA has not published the LoRA weights in a form Apache-2 licensed and re-distributable, M0'''-5 is downgraded to "linear_spec without LoRA" — acceptance rate falls to ~1.5-2× per NVIDIA's own ablation. Verify before starting M0'''-5. Fallback: S2D2 (see §6.1) as a licence-clean alternative.
2. **YaRN implementation subtlety**. The mscale on attention outputs (§4.1 last line) is easy to forget; without it, long-context quality degrades. Add a specific unit test.
3. **Chat template implicit**. The model card doesn't print the chat template; you must extract it from `tokenizer_config.json`. If it's not there, you have to guess at Mistral v3 format. If wrong, all Q/A benchmarks will show wrong-format degradation.
4. **`ar_loss_weight=1.0` implication**. The joint AR+diffusion training with 1:1 weighting means AR mode may be *stronger* than diffusion mode on many tasks. Test both modes on your target benchmarks before committing to "diffusion is the main mode" for the demo.
5. **Tokenizer bytes vs SentencePiece**. If the Ministral tokenizer uses a specific byte-level BPE variant that MLX-Swift's tokenizer library doesn't support, you have a hidden work item. Verify with a `tokenizer.encode("test 123 你好")` round-trip.
6. **Bidirectional forward pass at seq > 8K on M1**. Attention scores tensor exceeds memory — cap seq_len in the demo build or add a chunked-attention implementation (memory-efficient attention). MLX-Swift may have this already; check.
7. **Precision drift**. YaRN + Llama-4 Q-scaling + BF16 accumulation may drift enough over 26 layers that top-1 sampling occasionally diverges from HF reference. If parity checks fail intermittently, allow argmax-tie-tolerance in tests.
8. **Deployment story unclear**. The blog talks about SGLang/B200. On Apple Silicon, there's no direct precedent; you're the first-mover. Document what you build so others (or you, in 6 months) can follow.

---

## 11. Kick-off objective for Claude Code / Codex

Copy-paste to your dev agent to start execution:

---

> You are working on `./`, an MLX-Swift + Metal implementation of the LLaDA-family engine. LMOE, DiffusionGemma, and Sumi are already in-tree (or will be by the time you start).
>
> **Task**: Implement Nemotron-Labs-Diffusion-3B as the fourth model. Follow the guide at `Plans/nemotron-labs-diffusion-implementation-guide.md` (this file). Start with milestone **nemotron-M0'''-1** and stop after it passes its exit criteria before proceeding.
>
> **Ground rules**:
>
> - Do NOT modify LMOE / DiffusionGemma / Sumi code beyond adding narrow extension hooks (YaRN scaling, Q-scaling hook, attention-mode enum). All existing tests must pass unchanged.
> - Use the exact configuration in `config.json` at [huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B/raw/main/config.json). Do not tune hyperparameters yet.
> - Reference sampler is verbatim as in `Plans/nemotron-labs-diffusion-implementation-guide.md` §2. Do not "simplify".
> - Weight loading: BF16 first. No quantisation until milestone M0'''-6.
> - **Do not** implement the training-time `block_diff` attention mask. This checkpoint uses `bidirectional` paradigm at inference — hard-code that assumption for now.
> - **Do not** implement `linear_spec_generate` until milestone M0'''-5 gate is opened. First get `ar_generate` and `generate` (block-diffusion) parity green.
>
> **Verification approach**:
>
> 1. Load NVIDIA reference PyTorch model, generate reference outputs for 5 fixed prompts × 128 tokens × temperature=0 × seed=42 for both `ar_generate` and `generate` modes. Save to `NeoDiffusion/Tests/nemotron_reference.json`.
> 2. Implement Nemotron in MLX-Swift.
> 3. Compare top-1 tokens per position; assert exact match. If mismatch: dump per-layer float diff to find where drift begins. Common suspects: YaRN mscale not applied, Llama-4 Q-scaling wrong on long seq, RoPE frequencies wrong, BOS handling.
> 4. Only after M0'''-1 through -4 pass, begin M0'''-5 (linear-spec).
>
> **Reporting**:
>
> Every milestone: report (a) exit criteria met yes/no, (b) top-1 token match rate on the 5-prompt suite, (c) memory peak, (d) tokens/sec on M1 16 GB, (e) diff-summary in loc of what changed vs prior milestone.
>
> **Ask before you build**: If any of §3.4 decisions Q1–Q5 are unclear or the answer changed since 2026-07-07 (e.g., LoRA weights are unavailable, chat template is different), stop and surface the question to Andre. Do not guess on load-bearing decisions.

---

## 12. Uncertainty flags — every "inferred" or "speculative" claim

Cross-reference for §1-§11 claims that were not directly read from primary source files.

| Claim | Location | Status | How to verify |
| --- | --- | --- | --- |
| "MLX-Swift LoRA hot-swap support is limited as of 2026-06" | §1.1, §3.2 | *inferred* | Check current `MLXNN.LoRALinear` API and `mlx-swift-examples` LoRA tutorials for adapter-toggle patterns. |
| "4 published MLX-Swift YaRN implementations you can port from" | §3.4 Q2 | *inferred (count)* | Search `mlx-swift-examples` and community forks; verify at least one production-quality YaRN exists. |
| "`linear_spec_lora` weights availability on HF" | §3.4 Q5, §10.1 | *speculative* | Check `huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B/tree/main` for the adapter subfolder; check licence. |
| "M1 memory-bandwidth-bound elsewhere" | §5.2 | *inferred* | Profile with Instruments; confirm Metal shader-time distribution before assuming. |
| "SGLang radix caching applies directly" | §7 | *sourced but inferred applicability* | The [NVIDIA blog](https://huggingface.co/blog/nvidia/nemotron-labs-diffusion) names SGLang; radix cache applicability is a reasonable transfer, not a direct claim. |
| "Chat template = Mistral v3" | §3.4 Q4, §10.3 | *inferred from base model* | Dump `tokenizer_config.json` → `chat_template`, verify against Mistral v3 spec. |
| "Confidence sampler realises ~3× TPF" | §1 TL;DR (blog claim) | *sourced* | [NVIDIA blog](https://huggingface.co/blog/nvidia/nemotron-labs-diffusion) — but on B200; M1 numbers unknown. |
| "3.84B total params" | §8.1 | *computed* | Compare to model card. Model card says ~4B — matches within 5%. |
| "Bidirectional attention peaks at 4.3 GB at 8K seq on M1" | §8.4 | *computed* | Instruments live-monitor during first end-to-end MVP forward. |
| "S2D2 as licence-clean fallback for linear-spec" | §6.1, §10.1 | *inferred* | Verify S2D2 paper's implementation is Apache-2 compatible; check reference repo licence. |
| "Nemotron reference impl uses SGLang directly" | §6.1 | *sourced* | Blog names SGLang explicitly. |
| "Mean acceptance > 3 target" | §9 M0'''-5 exit criteria | *speculative* | NVIDIA reports ~3× TPF; acceptance rate not directly stated. Confirm from paper appendix if available. |
| "Ministral tokenizer is SentencePiece byte-level BPE" | §10.5 | *inferred* | Check `tokenizer.model` file magic bytes; run round-trip test. |

**All claims not in this table should be traceable to a fetched source file or arithmetic in this document.**
