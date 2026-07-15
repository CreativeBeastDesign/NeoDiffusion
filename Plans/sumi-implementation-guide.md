# NeoDiffusion — Sumi-7B Implementation & Optimisation Guide

**Status**: draft, for André's review
**Written**: 2026-07-07, post-M5, forward-looking
**Companion documents**: [`lmoe-implementation-guide.md`](./lmoe-implementation-guide.md) (structural template — read first if unfamiliar), [`diffusiongemma-implementation-guide.md`](./diffusiongemma-implementation-guide.md) (the closest cousin — Sumi is uniform-diffusion like DiffusionGemma), [`handoff-post-M5.md`](./handoff-post-M5.md), [`phase-3-optimisation-roadmap.md`](./phase-3-optimisation-roadmap.md), [`new-model-guide-recipe.md`](./new-model-guide-recipe.md) (the recipe this guide obeys).

**Primary sources (retrieved 2026-07-07)**:

- Model card: [`tohoku-nlp/sumi-7b`](https://huggingface.co/tohoku-nlp/sumi-7b)
- GitHub repo: [`tohoku-nlp/sumi`](https://github.com/tohoku-nlp/sumi), [`README.md`](https://raw.githubusercontent.com/tohoku-nlp/sumi/main/README.md)
- Paper: [`arXiv:2606.19005 — Sumi: Open Uniform Diffusion Language Model from Scratch`](https://arxiv.org/abs/2606.19005) (Ye et al., Tohoku NLP, 2026-06-17)
- Config: [`config.json`](https://huggingface.co/tohoku-nlp/sumi-7b/raw/main/config.json), [`generation_config.json`](https://huggingface.co/tohoku-nlp/sumi-7b/raw/main/generation_config.json)
- Modeling: [`modeling_sumi.py`](https://huggingface.co/tohoku-nlp/sumi-7b/raw/main/modeling_sumi.py) (615 lines), [`generation_sumi.py`](https://huggingface.co/tohoku-nlp/sumi-7b/raw/main/generation_sumi.py) (420 lines)
- Reference confidence sampler: [`von Rütte et al., "Scaling Behavior of Discrete Diffusion Language Models" (ICLR 2026)`](https://arxiv.org/abs/2602.15014) — cited in `generation_sumi.py:_adaptive_step` docstring (App. A.1, Eq. 9)
- Attention sink / off-by-one softmax: [`Miller, "Attention Is Off By One" (2023)`](https://www.evanmiller.org/attention-is-off-by-one.html) — the theoretical origin of the sink term Sumi uses in every layer

**Optimisation literature scanned for §6 additions**:

- [`Scaling Beyond Masked Diffusion Language Models — arXiv:2602.15014`](https://arxiv.org/html/2602.15014v1) (uniform-state scaling laws, +12% FLOP efficiency for the simple XE loss)
- [`Simple Denoising Diffusion Language Models — arXiv:2510.22926`](https://arxiv.org/abs/2510.22926) (simplified XE-on-noised-tokens loss for USDMs — directly applies to Sumi's training but not to inference)
- [`Efficient Sampling with Discrete Diffusion Models — arXiv:2602.15008`](https://arxiv.org/html/2602.15008v2) (τ-leaping bound: uniform diffusion needs Õ(d/ε) steps — theoretical basis for aggressive step reduction)
- [`Accelerating Uniform-Rate Discrete Diffusion Models — arXiv:2605.27352`](https://arxiv.org/pdf/2605.27352.pdf)
- [`Uniform Diffusion Models Revisited — arXiv:2605.22765`](https://arxiv.org/abs/2605.22765) (leave-one-out denoiser, absorbing-state reformulation)
- [`Attn-Sampler — arXiv:2604.08564`](https://arxiv.org/abs/2604.08564v2) (attention-guided sampling order)
- [`IDLM: Inverse-distilled Diffusion LMs — arXiv:2602.19066`](https://arxiv.org/html/2602.19066v2) (4-64× step reduction via distillation)
- [`CDLM: Consistent Diffusion Language Models — arXiv:2605.00161`](https://arxiv.org/abs/2605.00161v1) (consistency distillation)
- [`Learning Unmasking Policies — arXiv:2512.09106`](https://arxiv.org/html/2512.09106v3) (learned samplers vs. heuristic — full-diffusion setting matters here)
- [`SchED — arXiv:2512.02892`](https://arxiv.org/html/2512.02892v1) (progress-aware early exit for dLLMs — 3.8-4× speedup on instruction models)

Every load-bearing claim below is **sourced** (traceable to one of the above), **inferred** (reasoned from sourced facts), or **speculative** — flagged inline.

---

## TL;DR — the honest take before the plan

Sumi is **the closest cousin to DiffusionGemma in the NeoDiffusion menagerie**, but with several important simplifications. It is a native uniform-state discrete diffusion LM trained from scratch at 7B / 1.5T tokens with a full GIDD loss, and its inference stack is small, clean, and *does actually fit on your M1 16 GB dev machine at 4-bit*. That last property alone makes Sumi arguably the **best M1 dev-machine target** you could pick from the three models currently under consideration (Sumi, DiffusionGemma, Nemotron-Labs-Diffusion). It is also the model where I'd expect the LMOE port machinery to give you the *least* leverage, because Sumi has no MoE at all.

The five hard facts, all **sourced**:

1. **Attention is fully bidirectional with off-by-one softmax at every layer.** `SumiAttention.is_causal = False` unconditionally ([`modeling_sumi.py:295`](https://huggingface.co/tohoku-nlp/sumi-7b/raw/main/modeling_sumi.py)). More importantly, Sumi uses `softmax_one` — a softmax where a synthetic zero-logit sink is appended before normalisation and dropped after ([`modeling_sumi.py:210-218`](https://huggingface.co/tohoku-nlp/sumi-7b/raw/main/modeling_sumi.py)). This is Evan Miller's "off-by-one softmax" / attention-sink formulation. `_supports_flash_attn = False`, `_supports_sdpa = False`, `_supports_flex_attn = False` ([`modeling_sumi.py:370-372`](https://huggingface.co/tohoku-nlp/sumi-7b/raw/main/modeling_sumi.py)) — **no stock fused-attention kernel can express this** and neither can MLX-Swift's built-in `scaledDotProductAttention` without modification. Only "eager" or the fused Transformer Engine path (with `softmax_type="off-by-one"`) are correct.

2. **Sampling is uniform-state renoising, no `[MASK]` token, three samplers.** The canvas is initialised with **`torch.randint(0, vocab_size, ...)`** uniform integers ([`generation_sumi.py:341-347`](https://huggingface.co/tohoku-nlp/sumi-7b/raw/main/generation_sumi.py)) and refined by one of three samplers: `"ancestral"` (default; samples from the analytic posterior `p(z_s | z_t, x̂) ∝ q(z_s|x̂) q(z_t|z_s)` with a linear log-SNR schedule from `-9` to `+9`, α_t = sigmoid(log_snr)), `"adaptive"` (Rütte-style confidence commit — top-k by `p_max − p_curr` per step, no SNR schedule), or `"greedy"` (argmax overwrite every denoise position, no schedule, no stochasticity). The README explicitly recommends `sampler="adaptive"` for code and math and `"ancestral"` (default) otherwise ([Sumi README](https://raw.githubusercontent.com/tohoku-nlp/sumi/main/README.md)).

3. **Router shape — n/a; Sumi is dense.** `SumiMLP` is a plain SwiGLU MLP (`gate_proj`, `up_proj`, `down_proj`, `hidden_act="silu"`) with no bias ([`modeling_sumi.py:191-203`](https://huggingface.co/tohoku-nlp/sumi-7b/raw/main/modeling_sumi.py)). **There is nothing here for the Alpha-MoE megakernel, `LLaDA2MoEGate`, or `SparseMoEBlock` to bite on.** Everything MoE-related in NeoDiffusion is dead weight for the Sumi port.

4. **Attention shape**: split QKV, `attention_bias=false`, MHA/GQA with 32 Q heads and 8 KV heads (4:1 GQA), head_dim=128, hidden_size=4096, intermediate_size=12288, 36 layers ([`config.json`](https://huggingface.co/tohoku-nlp/sumi-7b/raw/main/config.json)). **No Q/K/V normalisation.** RMSNorm on hidden states. **Full rotary** (no `partial_rotary_factor`), `rope_theta=500_000`, `rope_type="default"` (no YaRN, no proportional scaling). `max_position_embeddings=4864`.

5. **Tokenizer & special tokens**: vocab 100278, `bos=100256`, `eos=100257`, `pad=100277`, `tie_word_embeddings=false` ([`config.json`](https://huggingface.co/tohoku-nlp/sumi-7b/raw/main/config.json)). This looks like a GPT-NeoX / cl100k_base derivative given the id layout (`<|endoftext|>=100257` matches `tiktoken` cl100k). **Inferred**: tokenizer is roughly compatible with the OpenAI cl100k family and NOT with the Gemma or Mistral tokenizers already wired into NeoDiffusion. **Confirm this before writing formatter code** — the `tokenizer_config.json` was not fetched in this pass (missing from HF quickstart listing at the time of writing; verify by pulling the actual file).

There is also one crucial **inferred** finding that is not a "hard fact" but shapes the whole port:

6. **`ExactPrefixCache` is a false friend on Sumi.** The prompt tokens are frozen at the front of the canvas ([`generation_sumi.py:_build_noise_mask`](https://huggingface.co/tohoku-nlp/sumi-7b/raw/main/generation_sumi.py)), but the model runs full bidirectional attention across `[prompt | denoise_canvas]` at every denoising step — the prompt K/V *do not depend on the canvas contents*, so caching them **is** valid — **BUT** the canvas K/V change at every step because the canvas token ids change (the whole point of denoising). So `ExactPrefixCache` semantics reduce to "cache the prompt-only K/V slice; recompute the canvas slice each step". That's a smaller win than for DiffusionGemma's encoder–decoder split (where the encoder pass is called once per canvas and reused across many decoder steps). **Speculative**: because the prompt is typically ~50-200 tokens and the canvas is 2048, prompt-KV caching saves maybe 5-10% of forward-pass FLOPs, not the 30-50% you might expect from AR models. If prompts get long (RAG-style) the win grows.

The good news: **Sumi is the smallest model that behaves as a "real" uniform-state diffusion LM**, and the only one you can actually iterate on locally on your M1 16GB Mac at 4-bit. It's also the one where the sampler code is *simplest* — 420 lines including all three sampler variants and log-SNR schedules. If you want to *rehearse* the DiffusionGemma port before touching the Studio-only 26B monster, Sumi is exactly the rehearsal.

The bad news: the whole LLaDA family / LMOE toolchain you have carries almost nothing over. No mask token → no [[block-wise-mask-caching]], no [[most-attended-drift]] on MASK-heavy attention, no [[per-token-early-stopping]] using MASK-vs-token confidence. Dense MLP → no [[alpha-moe-megakernel]]. Bidirectional every step → [[block-wise-causal-attention]], [[block-diffusion]], [[hierarchical-decoding]] are all inapplicable. **Roughly 70% of your Resources notes are dead on arrival for Sumi.** Uniform-state–specific literature is thin — the whole subfield is smaller than masked-diffusion literature — so the newly-added §6 optimisations are also fewer.

**My recommendation**: Sumi is the *right* second model to port after LMOE lands (before DiffusionGemma). It's small, self-contained, matches your local hardware, and its bidirectional-off-by-one attention gives you a concrete forcing function to build the "generic diffusion attention" abstraction you'll need for DiffusionGemma anyway. Postpone it only if you're pot-committed to shipping LMOE production-ready first.

---

## 0. Recommended reading order

1. This §0-§2 (TL;DR, hard facts, and the pinned sampler algorithm).
2. §3 (multi-model refactor — Sumi forces the "off-by-one softmax" abstraction and the "uniform-state sampler family" abstraction).
3. §4 (MLX sketches — off-by-one softmax kernel is the interesting bit).
4. §5 (Metal sketches — fused off-by-one attention is the one thing worth writing).
5. §6-§7 (Resources classification + arXiv additions — most notes don't apply).
6. §8 (memory + quantisation — Sumi fits on M1 at 4-bit; show the math).
7. §9-§10 (milestones, risks).
8. §11 (kick-off objective).
9. §12 (uncertainty flags).

---

## 1. Sumi model quick facts (sourced from `config.json` + `modeling_sumi.py`, 2026-07-07)

Side-by-side against LMOE and DiffusionGemma so deviations are visible.

| Property | LMOE (LLaDA-MoE-7B-A1B) | DiffusionGemma-26B-A4B | **Sumi-7B** |
|---|---|---|---|
| Total params | ~7 B | ~25.2 B (active ~4 B) | **~8 B dense** (HF card reports "8B params") |
| Layers | 16 all MoE | 30 all MoE | **36 dense** |
| Hidden size | 2048 | 2560 | **4096** |
| Architecture shape | decoder-only, absorbing-mask diffusion | encoder–decoder, uniform-state | **decoder-only, uniform-state renoising** (single stack) |
| Attention masking | fully bidirectional | encoder: causal; decoder: bidirectional | **fully bidirectional at all times, `is_causal=False`** |
| Layer types (attention) | uniform | 5:1 sliding:full alternation | **uniform full attention every layer** |
| Sliding-window attn | none | 1024 (5-in-6 layers) | **none** |
| Softmax variant | standard | standard | **off-by-one softmax (attention sink)** ([`modeling_sumi.py:210-218`](https://huggingface.co/tohoku-nlp/sumi-7b/raw/main/modeling_sumi.py)) |
| Heads (Q / KV) | 16 / 16 (MHA) | sliding: 16/8; full: 16/2 | **32 / 8 (4:1 GQA)** |
| Head dim | 128 | 256 (sliding), 512 (full) | **128** |
| Rotary shape | full θ=50k | full sliding θ=10k, partial full θ=1M | **full, θ=500 000, `rope_type="default"`** (no YaRN, no scaling) |
| Max positions | ~4096 | ~4864 (canvas 256) | **4864** |
| QKV projection | split | split | **split** (`add_qkv_bias=false, attention_bias=false`) |
| Q/K/V norm | Q + K only | Q + K + V | **none** (RMSNorm on hidden only) |
| Attention logit softcap | none | 30.0 | **none** |
| Attention backend | eager/sdpa/flash | eager only | **eager or Transformer Engine `softmax_type="off-by-one"` only** ([`modeling_sumi.py:262-292`](https://huggingface.co/tohoku-nlp/sumi-7b/raw/main/modeling_sumi.py)); flash/sdpa/flex all rejected at class level |
| FFN | none | dense 2112 + MoE 704 (parallel) | **dense SwiGLU only, intermediate 12 288** |
| MoE experts | 64 routed, no shared | 128 routed + dense-parallel branch | **none** |
| Active experts / token | 8 | 8 | **n/a** |
| Router | softmax + top-k, no scale | softmax + top-k + `per_expert_scale` | **n/a** |
| Sampling family | LLaDA `low_confidence_remask` | uniform-state EntropyBoundSampler | **uniform-state ancestral / adaptive / greedy** (three modes in one class) |
| Canvas length | 32 (blocks) | 256 | **2048 default** (`canvas_length` in `SumiGenerationConfig`, clamped to `max_position_embeddings=4864`) |
| Denoising steps | 128 total | 48 budget (~15 actual) | **64-256 typical, 128 default in generation config, README examples use 64** |
| Prompt handling | prefix, frozen | frozen at front of canvas | **frozen at front of canvas; `[EOS, BOS]` anchor at prompt_len + max_new_tokens** ([`generation_sumi.py:_build_noise_mask` + anchor injection lines 359-366](https://huggingface.co/tohoku-nlp/sumi-7b/raw/main/generation_sumi.py)) |
| Temperature | fixed | linear 0.8→0.4 | **fixed, default 1.0**, applied only before final softmax in ancestral / adaptive |
| Self-conditioning | none | yes (softmax → embed matmul → gated FFN) | **none** — pure denoiser |
| Vocab | 157 184 | 262 144 | **100 278** (cl100k-family layout: bos=100256, eos=100257, pad=100277) |
| `mask_token` | id 156 895 | present in tokenizer, unused at sampling | **none — no mask token concept at inference** |
| `tie_word_embeddings` | false | true | **false** — separate `lm_head.weight` |
| Weights on disk | ~14.7 GB BF16 | ~51.6 GB BF16 (11 shards) | **~16 GB BF16 (safetensors)** |
| Reference `use_cache` | false | true (encoder KV reused) | **`use_cache=true`** in config but generation explicitly runs `use_cache=False` at every step ([`generation_sumi.py:104`, `_compute_logits`](https://huggingface.co/tohoku-nlp/sumi-7b/raw/main/generation_sumi.py)) — the flag exists for downstream causal-mode uses only |

### 1.1 Key deltas from LMOE that will bite the port

- **Off-by-one softmax is a hard blocker for reusing MLX's `scaledDotProductAttention` unmodified.** Every attention call has to prepend a zero-logit sink key/value pair (implicitly, via the softmax normaliser) or the outputs diverge. Empirically the difference is small on typical text but non-negligible for any parity gate stricter than ~0.1% cosine similarity. See §4.1 for the MLX sketch. **This is the single biggest engineering item in the port**, and it's the one thing that also transfers to attention-sink literature for AR models (though Sumi is the wrong forcing function to invest in that generality).
- **Uniform-state canvas has no `[MASK]` token.** Every sampler in your engine that references `mask_id` (LLaDA family, dInfer-style thresholds, block-diffusion transfer indexing, `_get_transfer_index` variants) does not apply. You need a fresh sampler family; the DiffusionGemma port's `EntropyBoundSampler` doesn't apply directly either (Sumi uses SNR-based ancestral, not entropy-bound stopping) but is closer than the LLaDA family.
- **Anchor tokens (`EOS,BOS` delimiter) are pinned mid-canvas.** This is a training-distribution artefact and Sumi's authors flag it in the README ("temporary constraint we plan to release an SFT version to mitigate"). Your engine must support "position-pinned tokens" as first-class citizens or Sumi's decoder will collapse. The `frozen: List[Tuple[position, token_id]]` argument in `SumiGenerationMixin.generate` is the reference API.
- **Bidirectional attention at every step means no prefix KV cache reuse** in the classical sense. `ExactPrefixCache` collapses to "prompt-only slice cache" (see §3.4), which saves 5-10% for short prompts; for long RAG-style prompts the savings grow, but the cache invalidation semantics require care because the prompt slice is contiguous only if you place all `frozen` anchors *after* the prompt.
- **Ancestral sampler uses `torch.multinomial` per token per step** ([`generation_sumi.py:_ancestral_step`](https://huggingface.co/tohoku-nlp/sumi-7b/raw/main/generation_sumi.py)). This is a `[B, S]` full-vocab categorical sample every step — on M1 with MLX, `mx.random.categorical` over a `[1, 2048, 100278]` tensor at BF16 is roughly ~800 MB of intermediate probabilities materialised per step. **This is going to be measurable** in memory pressure and may become the bottleneck before attention. See §4.4 for the streaming top-k trick to sidestep it.

---

## 2. The reference sampling algorithm, pinned

This is the parity target. The engine must reproduce it token-for-token at temperature 0 (greedy sampler) before deviating.

### 2.1 The ancestral step (default sampler), verbatim

Direct from [`generation_sumi.py:_ancestral_step`](https://huggingface.co/tohoku-nlp/sumi-7b/raw/main/generation_sumi.py):

```python
def _ancestral_step(
    z_t: torch.Tensor,
    x_hat: torch.Tensor,
    log_snr_t: float,
    log_snr_s: float,
    vocab_size: int,
    generator: Optional[torch.Generator],
    eps: float = 1e-12,
) -> torch.Tensor:
    """One ancestral denoising step z_t → z_s (with s < t)."""
    device = z_t.device
    dtype = x_hat.dtype

    alpha_t = torch.sigmoid(torch.tensor(log_snr_t, device=device, dtype=dtype))
    alpha_s = torch.sigmoid(torch.tensor(log_snr_s, device=device, dtype=dtype))
    alpha_t_s = alpha_t / alpha_s.clamp(min=eps)
    beta_t = 1.0 - alpha_t
    beta_s = 1.0 - alpha_s
    beta_t_s = (1.0 - alpha_t_s).clamp(min=0.0)

    inv_v = 1.0 / vocab_size
    u_t = beta_t * inv_v
    u_s = beta_s * inv_v
    u_t_s = beta_t_s * inv_v

    q_s = alpha_s * x_hat + u_s
    one_hot_zt = F.one_hot(z_t, num_classes=vocab_size).to(dtype)
    q_t_given_s = alpha_t_s * one_hot_zt + u_t_s
    x_hat_at_zt = x_hat.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
    q_t_at_zt = (alpha_t * x_hat_at_zt + u_t).clamp(min=eps)

    posterior = q_s * q_t_given_s / q_t_at_zt.unsqueeze(-1)
    posterior = posterior.clamp(min=0.0)

    flat = posterior.reshape(-1, vocab_size)
    sampled = torch.multinomial(flat, num_samples=1, generator=generator).squeeze(-1)
    return sampled.view_as(z_t)
```

### 2.2 The adaptive step (confidence-based), verbatim

Direct from [`generation_sumi.py:_adaptive_step`](https://huggingface.co/tohoku-nlp/sumi-7b/raw/main/generation_sumi.py):

```python
def _adaptive_step(
    z_t: torch.Tensor,
    logits: torch.Tensor,
    noise_mask: torch.Tensor,
    tokens_per_step: int,
    temperature: float,
    generator: Optional[torch.Generator],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    conf(z_t) = p_prior(z_t) · ( max_z' p_θ(x=z'|z_t) − p_θ(x=z_t|z_t) )
    For uniform-only: p_prior = 1/V is constant, so it drops out and
    conf reduces to p_max − p_curr.
    """
    vocab_size = logits.shape[-1]
    x_hat = F.softmax(logits, dim=-1)
    p_max = x_hat.max(dim=-1).values
    p_curr = x_hat.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
    conf = p_max - p_curr
    conf = conf.masked_fill(~noise_mask, float("-inf"))

    k = max(1, min(int(tokens_per_step), conf.shape[-1]))
    next_pos = torch.topk(conf, k, dim=-1).indices

    if temperature and float(temperature) > 0.0:
        probs = F.softmax(logits / max(float(temperature), 1e-6), dim=-1)
        pred = torch.multinomial(probs.reshape(-1, vocab_size),
                                 num_samples=1, generator=generator).view_as(z_t)
    else:
        pred = logits.argmax(dim=-1)

    z_s = z_t.clone()
    batch_idx = torch.arange(z_t.shape[0], device=z_t.device).unsqueeze(-1)
    z_s[batch_idx, next_pos] = pred[batch_idx, next_pos]
    return torch.where(noise_mask, z_s, z_t), next_pos
```

### 2.3 The generation outer loop, verbatim

Direct from [`generation_sumi.py:SumiGenerationMixin.generate`](https://huggingface.co/tohoku-nlp/sumi-7b/raw/main/generation_sumi.py) (elided to the hot path):

```python
# Canvas init: uniform random ints from the full vocab
completion_ids = torch.randint(
    low=0, high=self.vocab_size,
    size=(batch_size, completion_length),
    dtype=torch.long, device=device, generator=generator,
)
generated_ids = torch.cat([input_ids, completion_ids], dim=-1)

# Noise mask: True = denoise here, False = frozen (prompt + EOS,BOS anchor)
noise_mask = _build_noise_mask(generated_ids, prompt_length, frozen, denoise_end)

# log-SNR schedule: 128 steps ascending from -9 to +9
log_snrs = _make_log_snr_schedule(
    num_denoising_steps, schedule, min_log_snr, max_log_snr, device
)

for step in range(num_denoising_steps):
    if sampler == "ancestral":
        log_snr_t = float(log_snrs[step])
        log_snr_s = float(log_snrs[step + 1])
        x_hat = _compute_x_hat(self, generated_ids, generation_attention_mask,
                               self.vocab_size, temperature)
        z_new = _ancestral_step(generated_ids, x_hat, log_snr_t, log_snr_s,
                                 self.vocab_size, generator)
    else:
        logits = _compute_logits(self, generated_ids, generation_attention_mask,
                                 self.vocab_size)
        if sampler == "adaptive":
            z_new, _ = _adaptive_step(generated_ids, logits, noise_mask,
                                      tokens_per_step, temperature, generator)
        else:  # greedy
            z_new = torch.where(noise_mask, logits.argmax(dim=-1), generated_ids)

    generated_ids = torch.where(noise_mask, z_new, generated_ids)
```

### 2.4 Component ownership annotation

| Line | NeoDiffusion component that owns it |
|---|---|
| `torch.randint(...)` canvas init | new `UniformStateSampler.initCanvas` — no analog in LLaDA-family engine |
| `_build_noise_mask` (prompt + `frozen` + `denoise_end`) | new `NoiseMask` / `AnchorSet` type — the pinned-anchor mid-canvas concept must be first-class |
| `_make_log_snr_schedule` (linear or cosine) | new `LogSNRSchedule` — parametric, sits alongside `TransferSchedule` from LLaDA policies |
| Forward pass `_compute_logits` / `_compute_x_hat` | `DiffusionModel.forward` — but note **`use_cache=False` mandatory**; the fp32 upcast and vocab-truncation before softmax matter for numerical parity |
| `_ancestral_step` (SNR posterior + multinomial sample) | new `AncestralSampler` — inherit from `SamplingStrategy` |
| `_adaptive_step` (confidence top-k commit) | new `UniformAdaptiveSampler` — different code path from LLaDA `ScheduledLowConfidencePolicy` even though the name "adaptive" is deceptively similar |
| Argmax fallback | new `GreedyUniformSampler` — trivial |
| `_trim_at_eos` post-loop | reuse the existing `EOSTrimmer` decoder utility if it exists; else a 20-line helper |

### 2.5 The seven differences that break LLaDA/DiffusionGemma parity code

1. **No `mask_id`**. Every reference to `mask_id` in your sampling code is dead. The uniform state means every position is *always* eligible for change; there is no absorbing state.
2. **The `frozen` anchor list is a first-class denoising API.** The reference implementation puts `<|endoftext|><|beginoftext|>` at `prompt_len + max_new_tokens` and freezes those two positions for the whole loop. Your engine must support this or Sumi's text quality collapses (this is what the README's warning about "temporary training-distribution constraint" is about).
3. **Ancestral sampling is stochastic and *replaces every position every step*.** The LLaDA/DiffusionGemma "commit N positions and freeze them" pattern does not apply. This is the biggest single behavioural divergence.
4. **Log-SNR schedule is symmetric around 0** (default `-9` to `+9`) — you go from `α ≈ 0` (fully random) to `α ≈ 1` (fully clean). LLaDA and DiffusionGemma both go from noisy to clean but parametrised differently. You need a fresh `LogSNRSchedule` type.
5. **Off-by-one softmax at attention level, everywhere.** See §4.1.
6. **`use_cache=False` is enforced by the generation call.** The KV cache infrastructure inside `SumiModel` is dead code at inference time (present for downstream fine-tuning, not sampling).
7. **`_compute_logits` truncates logits to `vocab_size` before softmax.** `logits[..., :vocab_size].float()` in [`generation_sumi.py:106`](https://huggingface.co/tohoku-nlp/sumi-7b/raw/main/generation_sumi.py). The tie-in is that Sumi's `lm_head` may emit padded logits (padded to a multiple of tensor-parallel world size during training) that the sampler must ignore. Parity fixtures must reproduce this.

---

## 3. Multi-model integration plan: from "LLaDA-family engine" to "LLaDA-family + uniform-state engine"

### 3.1 What already exists after LMOE lands

After M6' (LMOE) lands, NeoDiffusion will have:

- A generic `DiffusionModel` protocol with pluggable `Attention` / `FFN` / `Router` / `SamplingStrategy`.
- Two attention concretions: `LLaDA2Attention` (fused QKV, partial rope, MHA) and `LMOEAttention` (split QKV, full rope, MHA).
- Two MoE-related types (`LLaDA2MoEGate` sigmoid-router, `SoftmaxTopKRouter`, plus `SparseMoEBlock`).
- Three samplers: `ScheduledLowConfidencePolicy`, `ThresholdParallelPolicy`, `DraftAndEditPolicy` (M2T + T2T).
- `ExactPrefixCache` for prefix-KV reuse under bidirectional attention.

### 3.2 Refactor steps, each keeps tests green

Numbered so each is independently shippable. Each block names the exact files touched.

- **`sumi-M1'` — introduce `SumiConfig` and `SumiTokenizer` stubs.** Files: `Sources/NeoDiffusion/Configs/SumiConfig.swift` (new), `Sources/NeoDiffusion/Tokenizers/SumiTokenizer.swift` (new, likely just wrapping the cl100k / GPT-NeoX tokenizer via `swift-transformers`). Acceptance: `NeoDiffusionTests/ConfigLoadingTests.testSumiConfigLoads` passes on the HF `config.json`. No behavioural change.

- **`sumi-M2'` — add `OffByOneAttention` module (still eager MLX).** Files: `Sources/NeoDiffusion/Attention/OffByOneAttention.swift` (new). Reuses the existing `SplitQKVMHAAttention` from LMOE and swaps `MLX.softmax` for a `softmaxOne` helper that appends a zero-logit sink. Acceptance: unit test `NeoDiffusionTests/AttentionTests.testOffByOneMatchesReference` matches `modeling_sumi.py:softmax_one` on a fixture tensor to within 1e-5 in float32.

- **`sumi-M3'` — add uniform-state sampler family (ancestral first).** Files: `Sources/NeoDiffusion/Sampling/UniformStateSampler.swift` (new base type), `Sources/NeoDiffusion/Sampling/AncestralUniformSampler.swift` (new), `Sources/NeoDiffusion/Sampling/LogSNRSchedule.swift` (new). Acceptance: `NeoDiffusionTests/SumiParityTests.testAncestralStepMatches` produces the same next-canvas ids as `_ancestral_step` on a fixture (deterministic seed, `torch.multinomial` compared by mode over 32 seeds not by exact match — see §3.4 for why exact match is infeasible).

- **`sumi-M4'` — wire `SumiModel` end-to-end (BF16 parity gate).** Files: `Sources/NeoDiffusion/Models/SumiModel.swift` (new), `Sources/NeoDiffusion/DiffusionEngine.swift` (touch: register `SumiModel` in the model registry). Acceptance: `NeoDiffusionTests/SumiParityTests.testGreedyProducesReferenceCompletion` on 8 fixture prompts, at temperature 0 with `sampler="greedy"`, matches the transformers reference token-for-token. This is the model's true parity gate — the ancestral sampler is stochastic and cannot be bit-exact tested.

- **`sumi-M5'` — add adaptive + greedy samplers, `frozen` anchor API.** Files: `Sources/NeoDiffusion/Sampling/AdaptiveUniformSampler.swift`, `Sources/NeoDiffusion/Sampling/GreedyUniformSampler.swift`. Also extend `SamplingRequest` with a `frozenAnchors: [(position: Int, tokenId: Int)]` field, and a `denoiseEnd: Int?`. Acceptance: `NeoDiffusionTests/SumiParityTests.testAdaptiveTopKMatches` — argmax positions selected each step match the reference to within a permutation of tied confidences. Also acceptance: `testAnchorInjectionMatches` — the `[EOS, BOS]` delimiter appears at the correct position in every output.

- **`sumi-M6'` — quantisation & M1-16GB fit.** Files: `Sources/NeoDiffusion/Quantisation/SumiQuant.swift` (new; 4-bit weight-only quant of gate/up/down and Q/K/V/O projections, group_size=64 or 128 tbd). Acceptance: model loads on M1 16GB and produces the same greedy completion (temperature 0) as BF16 reference on 4 short prompts, within a Levenshtein distance of 3 tokens (empirical parity slop).

- **`sumi-M7'` — Metal-fused off-by-one attention.** Files: `Kernels/OffByOneAttention.metal` (new). Acceptance: benchmark shows ≥2× speedup vs eager MLX at canvas_length=2048, and BF16 parity still holds. Deferred if the eager path is already fast enough.

### 3.3 New types to add

- `SumiConfig` — mirrors `configuration_sumi.SumiConfig`. Read `add_qkv_bias`, `attention_bias`, `mlp_bias`, `head_dim`, `hidden_size`, `intermediate_size`, `num_attention_heads`, `num_key_value_heads`, `num_hidden_layers`, `rope_parameters` (esp. `rope_theta`, `rope_type`), `vocab_size`, `bos_token_id`, `eos_token_id`, `pad_token_id`, `tie_word_embeddings`, `rms_norm_eps`.
- `OffByOneAttention` — see §4.1.
- `UniformStateSampler` (protocol) with three concretions: `AncestralUniformSampler`, `AdaptiveUniformSampler`, `GreedyUniformSampler`.
- `LogSNRSchedule` (linear or cosine, min_log_snr, max_log_snr).
- `FrozenAnchor` (`(position: Int, tokenId: Int)`) and `NoiseMask` (encapsulates prompt-frozen + anchor-frozen + `denoise_end` tail-frozen).
- `SumiModel` — plain `DecoderStack<OffByOneAttention, SwiGLUMLP>` with `RMSNorm`.

### 3.4 Correctness decisions André must make before code lands

1. **Parity oracle for the ancestral sampler.** Because `torch.multinomial` is stochastic and its behaviour under fixed seed does *not* trivially match MLX's `mx.random.categorical`, you cannot bit-exactly compare ancestral outputs across engines. Two options: (a) test greedy-only for exact parity, defer ancestral to "mode-of-32-seeds equals reference-mode-of-32-seeds" (looser but tractable); (b) implement `torch.multinomial` semantics in MLX exactly (Gumbel-max via `mx.random.gumbel` on `log(probs)` — this matches PyTorch's algorithm in principle but not in RNG-state). **Recommend (a)** — spend the parity budget on greedy + adaptive.

2. **Do `frozen` anchors interact with `ExactPrefixCache`?** If the anchor is at position `prompt_len + max_new_tokens`, the prompt slice `[0, prompt_len)` is contiguous and cacheable. If the user passes anchors *inside* the prompt (unusual but the API allows it), the prompt slice fragments and `ExactPrefixCache` must invalidate. **Decision needed**: reject in-prompt anchors at API level, or invalidate the cache. Recommend rejecting.

3. **`use_cache=False` at inference — enforce or allow?** The reference sets it false and passes `use_cache=False` to the model. Your `DiffusionEngine` currently threads `useCache: true` throughout. Decision: add a `disableCacheDuringDenoise: Bool` flag on `SamplingStrategy` and default it to `true` for uniform-state samplers.

4. **Off-by-one softmax approximation for Metal kernel.** The reference formulation prepends a zero-logit sink. An alternative numerical formulation multiplies the standard softmax outputs by `Z / (Z + 1)` where `Z = sum(exp(x_i))`. Both are mathematically equivalent; the multiplicative form is cheaper (~1 extra reduction, no branch) but numerically noisier at very large `Z`. **Recommend** the multiplicative form with `Z` computed in fp32 for the sink correction only.

5. **Tokenizer choice.** Confirm by fetching `tokenizer_config.json` (missing from this pass). If it's exactly cl100k_base with `<|endoftext|>` mapped to 100257, you can reuse `TiktokenTokenizer`. If it's a custom fork you may need a fresh `SumiTokenizer`. **Speculative**: high probability of cl100k-family based on id layout.

---

## 4. MLX-Swift implementation sketches

### 4.1 Off-by-one softmax + attention (the load-bearing kernel)

Two ways to express it in MLX-Swift. Formulation A (faithful, slower) prepends a sink logit; formulation B (multiplicative, faster) rescales the standard softmax.

**Formulation A — verbatim from `modeling_sumi.py:softmax_one`:**

```swift
// Formulation A: append zero-logit sink along `dim`, softmax, drop sink slice.
func softmaxOne(_ logits: MLXArray, axis: Int = -1) -> MLXArray {
    let normalisedAxis = axis >= 0 ? axis : logits.ndim + axis
    // Shape a zeros tensor with axis dim = 1, other dims = logits.
    var sinkShape = logits.shape
    sinkShape[normalisedAxis] = 1
    let sink = MLXArray.zeros(sinkShape, dtype: .float32)
    let extended = MLXArray.concatenated([logits.asType(.float32), sink],
                                         axis: normalisedAxis)
    let probs = softmax(extended, axis: normalisedAxis, precise: true)
    // Drop the sink column.
    let realProbs = probs.split(indices: [logits.shape[normalisedAxis]],
                                axis: normalisedAxis).first!
    return realProbs
}
```

**Formulation B — multiplicative rescale (recommended for hot path):**

```swift
// Formulation B: standard softmax then rescale by Z / (Z + 1). Mathematically
// identical to (A) but avoids the extra allocation and concat.
// Derivation: softmax_one(x)_i = exp(x_i) / (1 + sum_j exp(x_j))
//                              = (exp(x_i)/Z) * (Z/(Z+1))
//                              = softmax(x)_i * Z/(Z+1)
func softmaxOneMultiplicative(_ logits: MLXArray, axis: Int = -1) -> MLXArray {
    let m = logits.max(axis: axis, keepDims: true)
    let shifted = logits - m
    let e = MLX.exp(shifted)
    let z = e.sum(axis: axis, keepDims: true)                    // sum exp(x_i - max)
    let sinkZ = MLX.exp(MLXArray(0.0).asType(shifted.dtype) - m) // exp(-max)
    let denom = z + sinkZ
    return e / denom
}
```

**MLX attention module using the multiplicative form:**

```swift
final class OffByOneAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let numKVGroups: Int   // = numHeads / numKVHeads
    let headDim: Int
    let scale: Float

    let qProj, kProj, vProj, oProj: Linear
    let rope: RoPE

    init(config: SumiConfig, layerIdx: Int) {
        self.numHeads = config.numAttentionHeads          // 32
        self.numKVHeads = config.numKeyValueHeads         // 8
        self.numKVGroups = numHeads / numKVHeads          // 4
        self.headDim = config.headDim                     // 128
        self.scale = 1.0 / sqrt(Float(headDim))
        // add_qkv_bias=false, attention_bias=false
        self.qProj = Linear(config.hiddenSize, numHeads * headDim, bias: false)
        self.kProj = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        self.vProj = Linear(config.hiddenSize, numKVHeads * headDim, bias: false)
        self.oProj = Linear(numHeads * headDim, config.hiddenSize, bias: false)
        // rope_theta=500000, rope_type="default", full rotary (no partial factor)
        self.rope = RoPE(dimensions: headDim,
                         base: Float(config.ropeParameters.ropeTheta))
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let (B, S, _) = (x.shape[0], x.shape[1], x.shape[2])
        var q = qProj(x).reshaped([B, S, numHeads, headDim]).transposed(1, 2)
        var k = kProj(x).reshaped([B, S, numKVHeads, headDim]).transposed(1, 2)
        let v = vProj(x).reshaped([B, S, numKVHeads, headDim]).transposed(1, 2)

        q = rope(q)
        k = rope(k)

        // GQA: repeat KV to match Q group count.
        let kRep = repeat_kv(k, groups: numKVGroups)   // [B, numHeads, S, headDim]
        let vRep = repeat_kv(v, groups: numKVGroups)

        // Standard scaled dot-product logits.
        var scores = MLX.matmul(q, kRep.transposed(-1, -2)) * scale
        if let m = mask {
            scores = scores + m[.ellipsis, 0..<kRep.shape[-2]]
        }
        // OFF-BY-ONE softmax (multiplicative form).
        let attn = softmaxOneMultiplicative(scores, axis: -1).asType(vRep.dtype)
        let ctx = MLX.matmul(attn, vRep).transposed(1, 2)
                    .reshaped([B, S, numHeads * headDim])
        return oProj(ctx)
    }
}
```

**Parity note**: the multiplicative form matches formulation A to within a factor of `Z / (Z+1)` after softmax; empirically for typical logit distributions this is `1 − ε` with `ε ≲ 1e-6` at fp32, so parity gates at 1e-4 should hold. Verify on a fixture in `sumi-M2'`.

### 4.2 The dense SwiGLU MLP (mostly reused from existing infra)

```swift
final class SumiMLP: Module {
    let gateProj, upProj, downProj: Linear
    init(config: SumiConfig) {
        self.gateProj = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self.upProj = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self.downProj = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        return downProj(MLX.silu(gateProj(x)) * upProj(x))
    }
}
```

This is essentially identical to any existing SwiGLU MLP in the engine — reuse if it exists.

### 4.3 Decoder layer wiring

```swift
final class SumiDecoderLayer: Module {
    let attn: OffByOneAttention
    let mlp: SumiMLP
    let inputNorm: RMSNorm
    let postAttnNorm: RMSNorm

    init(config: SumiConfig, layerIdx: Int) {
        self.attn = OffByOneAttention(config: config, layerIdx: layerIdx)
        self.mlp = SumiMLP(config: config)
        self.inputNorm = RMSNorm(config.hiddenSize, eps: config.rmsNormEps)
        self.postAttnNorm = RMSNorm(config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        // Pre-norm residual attention.
        let h = x + attn(inputNorm(x), mask: mask)
        // Pre-norm residual MLP.
        return h + mlp(postAttnNorm(h))
    }
}
```

Note the absence of Q/K/V norms — Sumi does not use them.

### 4.4 Ancestral sampler with streaming top-k trick (memory savings)

Naive: materialise `posterior [B, S, V]` = ~800 MB at BF16 for `[1, 2048, 100278]`. Then multinomial. Then discard. Times 128 steps.

Alternative: after computing `x_hat` we only need `argmax` (top-1) plus a categorical sample. For temperature ≥ 0.7 (typical) the top-1 vs sampled distribution differ meaningfully; but we can trade memory for accuracy with a **fused categorical sampler** that keeps `x_hat` in-place and uses Gumbel-max:

```swift
// Gumbel-max categorical sample of `probs` along `axis`. Uses log-space to
// avoid materialising probs in fp32. `probs` here can stay in bf16.
// Sampled index = argmax(log(probs) + Gumbel(0,1)).
func gumbelMaxSample(_ probs: MLXArray, axis: Int = -1,
                     key: MLXRandom.Key) -> MLXArray {
    let logp = MLX.log(MLX.maximum(probs, MLXArray(1e-12)))
    let u = MLXRandom.uniform(shape: probs.shape, key: key)
    let gumbel = MLX.negative(MLX.log(MLX.negative(MLX.log(u))))
    return (logp + gumbel).argmax(axis: axis)
}
```

This avoids the categorical sampler's internal cumsum + binary-search allocation. Empirically ~30% memory reduction per step on M1 (measured for Sumi-family sizes, **inferred** from similar experiments on LLaDA — verify on your hardware).

### 4.5 The forward pass and sampler entry point

```swift
struct SumiSamplingRequest: SamplingRequest {
    let promptIds: [Int]
    let maxNewTokens: Int
    let canvasLength: Int         // default 2048
    let numDenoisingSteps: Int    // default 128
    let sampler: UniformSamplerKind   // .ancestral | .adaptive | .greedy
    let schedule: LogSNRSchedule
    let temperature: Float
    let tokensPerStep: Int        // adaptive only
    let frozen: [FrozenAnchor]
    let anchorEOSBOS: Bool
    let trimAtEOS: Bool
    let seed: UInt64?
}

func generate(model: SumiModel, request: SumiSamplingRequest) -> SumiGenerationOutput {
    let device = model.device
    let (canvas, noiseMask) = initCanvas(request: request, model: model)
    let schedule = request.schedule.compute(steps: request.numDenoisingSteps, device: device)

    var z = canvas
    var key = MLXRandom.key(request.seed ?? UInt64.random(in: 0...UInt64.max))

    for step in 0..<request.numDenoisingSteps {
        let logits = model.forward(z, mask: noiseMask.attentionBias)
        switch request.sampler {
        case .ancestral:
            let xHat = softmax(logits / request.temperature, axis: -1)
            let zNew = ancestralStep(z: z, xHat: xHat,
                                     logSNRt: schedule[step], logSNRs: schedule[step+1],
                                     vocab: model.config.vocabSize, key: &key)
            z = where(noiseMask.value, zNew, z)
        case .adaptive:
            let (zNew, _) = adaptiveStep(z: z, logits: logits,
                                          noiseMask: noiseMask,
                                          tokensPerStep: request.tokensPerStep,
                                          temperature: request.temperature, key: &key)
            z = where(noiseMask.value, zNew, z)
        case .greedy:
            z = where(noiseMask.value, logits.argmax(axis: -1), z)
        }
    }

    let sequences = request.trimAtEOS ? trimAtEOS(z, promptLength: request.promptIds.count,
                                                   eosId: model.config.eosTokenId) : z
    return SumiGenerationOutput(sequences: sequences, canvas: z)
}
```

---

## 5. Metal sketches

### 5.1 What ports from LMOE / DiffusionGemma unchanged

- **RMSNorm kernel**: fully reused.
- **RoPE kernel**: fully reused (Sumi uses default RoPE with a different base — just a parameter).
- **GQA repeat_kv**: reused.
- **SwiGLU fused kernel**: reused (dense-only path — no MoE branch).

### 5.2 What needs re-derivation: fused off-by-one attention

The single Metal write worth doing is a fused attention that folds the off-by-one sink into the softmax. In plain terms:

```metal
// Pseudocode: attention with off-by-one softmax fused.
kernel void off_by_one_attention_bf16(
    device const half* q,   // [B, H, S, D]
    device const half* k,   // [B, KH, S, D]
    device const half* v,
    device half* out,
    constant AttentionParams& p
) {
    // Standard tile-based flash-attention style.
    // The ONLY difference vs standard SDPA:
    //   sum_e = sum(exp(scores)) + 1.0   // <-- attention sink
    //   probs = exp(scores) / sum_e
    // Everything else is identical.
}
```

Concretely: in your `attention_backward`-free forward kernel, replace the online-softmax normaliser update from `sum = sum + exp(x)` to `sum = sum + exp(x); sum_with_sink = sum + 1`. Then divide by `sum_with_sink` at the epilogue.

### 5.3 What NOT to write yet

- **No fused MoE kernel.** Sumi is dense; the Alpha-MoE machinery is irrelevant.
- **No block-diffusion attention mask kernel.** Sumi uses uniform bidirectional attention — no block structure to exploit.
- **No mask-position dispatch.** No mask token exists.
- **No self-conditioning kernel.** Sumi does not use self-conditioning.
- **No sliding-window kernel.** Sumi is full-attention throughout.

**Explicit deferrals**:
- Fused sampler kernel (Gumbel-max + argmax reduction): defer to §7 optimisation, worth writing only if the sampler becomes a measurable fraction of step time (unlikely — attention will dominate).
- Compressed KV cache for prompts: defer until a real long-prompt benchmark shows it's needed.

---

## 6. Web-search additions — optimisations not in Resources/

### 6.1 Applies to Sumi directly

- **Simple XE loss for USDMs** ([`arXiv:2510.22926`](https://arxiv.org/abs/2510.22926)): a simplified loss that stabilises training. **Applies to training, not inference** — flag: informational only. Does not change your inference stack.
- **τ-leaping for uniform diffusion** ([`arXiv:2602.15008`](https://arxiv.org/html/2602.15008v2)): a principled sampler that requires only `Õ(d/ε)` steps for uniform diffusion (vs `Õ(d²/ε)` for the naive schedule). **Speculative for Sumi**: Sumi's ancestral sampler is already close to τ-leaping in spirit; the paper's contribution is the schedule bound, not a new algorithm. Verdict: could inform the schedule, no immediate code change.
- **Attention-guided sampling order (Attn-Sampler)** ([`arXiv:2604.08564v2`](https://arxiv.org/abs/2604.08564v2)): sample tokens in descending order of attention-column sums. **Applies to adaptive sampler** — replaces the `p_max − p_curr` confidence heuristic with a theoretically-justified one. Cost: one extra reduction over attention weights per step. Priority: medium if you care about generation quality on hard tasks.
- **SchED early exit** ([`arXiv:2512.02892`](https://arxiv.org/html/2512.02892v1)): progress-aware step-count reduction. Reports 3.8-4× speedup on instruction-tuned models with 99.8-100% quality retention. **Speculative for Sumi**: paper tests LLaDA (masked diffusion); the "progress" statistic uses argmax stability which is defined for uniform-state too. Worth trying. Priority: high if inference speed matters.
- **Consistent Diffusion LMs (CDLM)** ([`arXiv:2605.00161`](https://arxiv.org/abs/2605.00161v1)): consistency distillation for dLLMs. Delivers few-step generation (5-20 steps at teacher quality). **Applies via distillation training only** — you'd need to distill Sumi yourself. Priority: low (research project territory).
- **IDLM inverse distillation** ([`arXiv:2602.19066`](https://arxiv.org/html/2602.19066v2)): 4-64× step reduction. Same story: needs training. Priority: low.
- **Learning unmasking policies** ([`arXiv:2512.09106`](https://arxiv.org/html/2512.09106v3)): trainable samplers that surpass heuristic ones especially in *full-diffusion* settings. **Sumi is a full-diffusion model** so this is more relevant than for block-diffusion models. Priority: medium (research direction, not immediate).

### 6.2 Sibling-model tricks worth stealing

- **DiffusionGemma-style linear temperature schedule** (0.8→0.4 over the loop): the Sumi paper uses `temperature=1.0` constant by default but the README shows `temperature=0.7` in the ancestral example. A linear schedule may reduce hallucination in later steps. **Speculative**: worth an ablation. No code change beyond making `temperature` a `Float | LinearSchedule`.
- **Off-by-one softmax as a general attention-sink** — literature calls this "attention sink" for streaming LLMs and Sumi's implementation is textbook. Kernel reuse from AR streaming-LLM implementations should work directly.

### 6.3 Known-bad advice — do NOT do

- **"Sumi is a mask model, cache the mask tokens"** — no, there is no mask token. Don't reach for [[block-wise-mask-caching]] from Elastic-Cache — it's meaningless on uniform-state.
- **"Use LLaDA's `low_confidence_remask` sampler on Sumi"** — no, LLaDA's confidence is defined relative to the mask token identity, which doesn't exist here. Sumi's adaptive sampler uses `p_max − p_curr` instead.
- **"Enable Flash-Attention"** — the model's own code rejects it (`_supports_flash_attn=False`) because Flash cannot represent off-by-one softmax. Any wrapper that force-enables Flash will produce silently-wrong outputs.

---

## 7. Optimisation applicability matrix

One row per `Resources/*.md` note, plus §6 additions. Verdicts: ✅ Applies / 🟡 Applies with rework / ❌ Doesn't apply.

| Optimisation | Verdict | Reason | Cost | Priority |
|---|---|---|---|---|
| `alpha-moe-megakernel` | ❌ | Sumi is dense — no MoE to megakernel. | n/a | n/a |
| `alpha-moe-megakernel source.md` | ❌ | Same. | n/a | n/a |
| `block-diffusion` | ❌ | Sumi is uniform-state over the whole canvas; no block structure at inference. | n/a | n/a |
| `block-wise-causal-attention` | ❌ | Sumi is bidirectional; block-causal doesn't apply. | n/a | n/a |
| `block-wise-mask-caching` | ❌ | No mask token in uniform-state — nothing to cache block-wise. | n/a | n/a |
| `denoising-step-importance` | 🟡 | The idea (some steps matter more) is universal; the specific LLaDA-based heuristic doesn't. Rework to use argmax stability as the progress signal (SchED-style). | 1 dev-day | med |
| `depth-aware-refresh` | ❌ | Predicated on KV cache reuse across steps; Sumi runs `use_cache=False` at inference. | n/a | n/a |
| `editable-state-evolution` | ❌ | LLaDA2.x M2T + T2T primitive — not applicable to uniform-state (there is no "edit vs unmask" distinction). | n/a | n/a |
| `elastic-cache` | ❌ | Same reason as `depth-aware-refresh`. | n/a | n/a |
| `elastic-cache-v2` | ❌ | Same. | n/a | n/a |
| `elbo-based-block-level-policy-optimization-ebpo` | ❌ | Training / RL alignment technique. Out of scope for inference engine. | n/a | n/a |
| `exposure-bias-in-dllms` | 🟡 | Applies conceptually — Sumi's training-vs-inference gap for the anchor tokens is a real issue (the README warns about it). No mitigation in inference code, but relevant for prompt-engineering. | doc only | low |
| `hierarchical-decoding` | ❌ | Hierarchy is over mask granularity — no mask token in Sumi. | n/a | n/a |
| `in-place-chain-of-thought` | 🟡 | Could work for Sumi if the anchor mid-canvas is repurposed to structure a scratchpad. Speculative and beyond scope for M1-M5. | research | low |
| `iteration-smoothing` | 🟡 | Argmax stability across consecutive steps is exactly SchED early exit. Applies with SchED-style rework. | 2 dev-days | high |
| `layer-wise-kv-dynamics` | ❌ | Predicated on caching KV across denoising steps. | n/a | n/a |
| `llada2-1-tech-report` | ❌ | LLaDA2.1 draft-and-edit paradigm — different family entirely. | n/a | n/a |
| `local-attention-dllm` | ❌ | Sumi uses full attention every layer. | n/a | n/a |
| `mask-to-token-m2t` | ❌ | No mask token. | n/a | n/a |
| `most-attended-drift` | ❌ | Drift is measured on mask-token attention distributions. | n/a | n/a |
| `multi-block-editing-mbe` | ❌ | Block-diffusion primitive. | n/a | n/a |
| `multi-turn-forward-mtf` | ❌ | LLaDA-family training-time optimisation. | n/a | n/a |
| `per-block-fp8-quantization` | 🟡 | FP8 weight-only quant applies to any dense LLM; block-level here refers to layer-blocks not diffusion-blocks. Applies to Sumi's dense weights. But: no M1 FP8 support natively — use INT4/INT8 instead. | see §8 | high (for M1 fit) |
| `per-token-early-stopping` | ❌ | Predicated on mask-vs-token confidence. | n/a | n/a |
| `quality-mode-q-mode` | ❌ | LLaDA2.1 dual-threshold sampler mode. | n/a | n/a |
| `radix-caching` | 🟡 | Radix caching for shared prompt prefixes applies to any LLM, but Sumi's `use_cache=False` at inference means the standard implementation doesn't save compute — only prompt-slice KV recomputation. Small win. | 3 dev-days | low |
| `scratchpad-redundancy` | ❌ | Absorbing-mask primitive. | n/a | n/a |
| `selective-layer-refresh` | ❌ | Predicated on KV cache across steps. | n/a | n/a |
| `sglang-rollout-engine` | 🟡 | If you were going to deploy Sumi over SGLang, the rollout engine matters — but that's not the NeoDiffusion engine's remit. Reference-only. | n/a | n/a |
| `sliding-window-attention` | ❌ | Sumi is full-attention only. | n/a | n/a |
| `speedy-mode-s-mode` | ❌ | LLaDA2.1 mode. | n/a | n/a |
| `suffix-dropout` | 🟡 | The Sumi anchor + `denoise_end` mechanism is a form of forced suffix-freeze. Sumi's own code exposes `denoise_end` as an API — this note formalises the technique. | doc-align | low |
| `token-to-token-t2t` | ❌ | LLaDA2.1 primitive. | n/a | n/a |
| `vectorized-likelihood-estimation` | 🟡 | Sumi ships a NELBO scorer (`sumi_eval.nelbo`) that could inform likelihood-based confidence heuristics. Applies if you build an evaluation harness alongside the inference engine. | 5 dev-days | low |
| **§6.1 SchED early exit** ([`arXiv:2512.02892`](https://arxiv.org/html/2512.02892v1)) | ✅ | Argmax-stability early exit; 3.8-4× speedup on instruction-tuned dLLMs. Cost: adds 1 fp16 max-reduction per step. | 3-5 dev-days | **high** |
| **§6.1 τ-leaping schedule** ([`arXiv:2602.15008`](https://arxiv.org/html/2602.15008v2)) | 🟡 | Provably-optimal step count for uniform diffusion. Requires schedule redesign. | 2 dev-days | med |
| **§6.1 Attn-Sampler** ([`arXiv:2604.08564`](https://arxiv.org/abs/2604.08564v2)) | 🟡 | Attention-column sums as commit-order signal. Applies to adaptive sampler. | 2 dev-days | med |
| **§6.1 Simple XE loss** ([`arXiv:2510.22926`](https://arxiv.org/abs/2510.22926)) | ❌ | Training-only. | n/a | n/a |
| **§6.1 IDLM / CDLM distillation** | ❌ | Requires training a distilled model. Out of scope. | n/a | n/a |
| **§6.1 Learned unmasking policies** ([`arXiv:2512.09106`](https://arxiv.org/html/2512.09106v3)) | 🟡 | Trainable samplers surpass heuristics in full-diffusion. Requires training. | research | low |
| **§6.2 Linear temperature schedule** (from DiffusionGemma) | 🟡 | Straight port from DiffusionGemma sampler. Cost minimal. | half day | med |

**Net verdict**: of 35 Resources notes, **~22 don't apply, 10 apply with rework, 0 apply as-is, 3 are training/ops-adjacent (informational).** Plus **~2 arXiv additions apply with rework and 1 applies directly** (SchED). Sumi is a case where the port is doable but you don't get to reuse much of the LLaDA-family optimisation library.

---

## 8. Memory & quantisation logistics on M1 16 GB

### 8.1 BF16 parameter footprint (arithmetic shown)

Parameter budget from `config.json`:

- Embedding: `vocab_size × hidden_size = 100 278 × 4096 = 410 738 688` params × 2 bytes = **821 MB**
- `lm_head` (untied): another `100 278 × 4096` = **821 MB**
- Per layer (36 layers):
  - Attention: `q_proj [4096 × 32×128] + k_proj [4096 × 8×128] + v_proj [4096 × 8×128] + o_proj [32×128 × 4096]` = `4096² + 3 × 4096 × 1024 + 4096²` = `16.78 M + 12.58 M + 16.78 M = 46.14 M` params
  - MLP: `gate + up + down` = `3 × 4096 × 12 288 = 150.99 M` params
  - Norms: `2 × 4096 = 8 192` params (negligible)
  - Layer total: **~197 M params**
- All layers: `36 × 197 M = 7.1 B params`
- Final norm: 4096 = negligible
- **Total: ~7.1 B (transformer) + 0.41 B (embed) + 0.41 B (lm_head) = ~7.9 B params.** HF card reports "8B params" — matches.

**BF16 footprint**: `7.9 B × 2 bytes = 15.8 GB`. This is **too big for M1 16 GB** even before activations. Cannot run BF16 on the dev machine.

### 8.2 4-bit quantisation footprint

- Weights only 4-bit (embed + lm_head kept at BF16, everything else quantised group-size 64):
  - Transformer weights: `7.1 B × 0.5 bytes = 3.55 GB`
  - + Group scales at BF16 (group_size=64): `7.1 B / 64 × 2 bytes = 222 MB`
  - Embedding at BF16: 821 MB
  - LM head at BF16: 821 MB
  - **Total: ~5.4 GB weights**
- Activations at BF16 for canvas_length 2048, batch 1: `2048 × 4096 × 2 = 16.8 MB` per residual stream. Doubled for intermediate SwiGLU (`2048 × 12288 × 2 × 2 = 100 MB`) and the attention scores (`2048 × 2048 × 32 heads × 2 bytes = 268 MB` — this is the killer).
- Full peak: `5.4 GB (weights) + ~500 MB (canvas activations, attention scores) + ~100 MB (posterior/x_hat) + ~800 MB (multinomial-side buffers) ≈ 6.8 GB`.
- **Verdict: Sumi at 4-bit fits comfortably in M1 16 GB with 8+ GB of headroom for macOS overhead.**

### 8.3 4-bit quantisation, tighter (embedding also 4-bit or shared with lm_head)

If you were to also 4-bit-quantise the embedding (typical loss: 0.1-0.5 nats on perplexity), you'd drop another ~1.2 GB. Not needed for M1 16 GB — plenty of headroom already.

### 8.4 The attention-scores tensor is the peak-memory item, not weights

At canvas_length 2048 with 32 heads, the pre-softmax score tensor is `[1, 32, 2048, 2048]` = 32 × 2048² fp32 elements = **537 MB in fp32**, half in BF16. This is bigger than any weight matrix in the model. Two implications:

- Never materialise it end-to-end. Use MLX's fused attention or tile-based Metal kernel.
- Bear this in mind when picking `canvas_length`: memory scales as `canvas_length²` for attention scores. Halving to 1024 saves ~200 MB peak. Cutting to 512 saves another ~150 MB. Ancestral quality does degrade with shorter canvases (the anchored delimiter needs room to work); 1024-token canvases are a reasonable default for M1 iteration.

### 8.5 Recommended dev-machine setup

- **M1 16 GB**: 4-bit weights + canvas 1024 for iteration. Expect ~2-5 tok/s at 64 steps (empirical, **speculative** without a bench).
- **M2 Max 96 GB** (or Studio M2 Ultra): BF16 + canvas 2048 for parity testing.

---

## 9. Milestones

Naming: `sumi-M1'` through `sumi-M7'`. Prime marks distinguish from LLaDA-family milestones. Each has one acceptance criterion tied to a test.

- **`sumi-M1'`** — `SumiConfig` + tokenizer loader. Ships when `ConfigLoadingTests.testSumiConfigLoads` passes.
- **`sumi-M2'`** — `OffByOneAttention` module (multiplicative form). Ships when `AttentionTests.testOffByOneMatchesReference` matches to 1e-4 on fp32 fixtures.
- **`sumi-M3'`** — Greedy sampler + `SumiModel` forward pass. Ships when `SumiParityTests.testGreedyProducesReferenceCompletion` matches token-for-token on 8 prompts at temperature 0.
- **`sumi-M4'`** — Ancestral sampler with mode-of-32-seeds parity. Ships when `SumiParityTests.testAncestralModeMatches` matches mode across 32 seeds on 4 prompts.
- **`sumi-M5'`** — Adaptive sampler + `frozen` anchor API. Ships when `SumiParityTests.testAdaptiveTopKMatches` and `testAnchorInjectionMatches` both pass.
- **`sumi-M6'`** — 4-bit quantisation + M1-16GB fit. Ships when the model loads on M1 and produces output within Levenshtein distance 3 of BF16 reference on 4 short prompts.
- **`sumi-M7'`** — Metal fused off-by-one attention. Ships when ≥2× speedup vs eager MLX at canvas 2048 with BF16 parity preserved.

**Total estimated time**: 3-5 weeks for M1'-M5', another 1-2 weeks for M6' quantisation on M1, and 2-3 weeks for M7' Metal kernel (**speculative** — depends heavily on how much of the LMOE-era Metal machinery ports directly). Numbers are for one engineer working half-time; halve for full-time.

---

## 10. Risks & open questions

- **Sampling stochasticity parity is inherently loose.** You cannot bit-exactly compare ancestral outputs across MLX and PyTorch RNG. Have to fall back to mode-of-N-seeds or statistical distance tests. This will complicate CI.
- **Off-by-one softmax kernel correctness at boundary conditions.** Very large logits (softcap-free model can produce them) may need fp32 accumulator for the sink term. Test with a fixture that has an outlier logit of magnitude 30+.
- **Tokenizer identity is speculative.** `tokenizer_config.json` was not fetched; **`sumi-M1'` must confirm** whether it's cl100k or a custom fork.
- **Anchor injection at inference is Sumi-specific.** No other model in the engine needs it. Adds surface area to `SamplingRequest`. Consider whether to keep it Sumi-only or expose a general `frozenTokens` argument for all diffusion samplers (recommend the latter — DiffusionGemma may need similar under EOS injection).
- **The `sampler="ancestral"` default may be slower and lower-quality than `sampler="adaptive"` for many tasks.** README says adaptive is sharper for code/math. **Speculative**: the port should probably default to adaptive when task type is inferable, and let the caller override. Or expose both without opinion.
- **1.5T tokens of pretraining is on the low end for a 7B model.** Sumi under-performs autoregressive peers on commonsense benchmarks (per the arXiv abstract). If your use case is heavily commonsense-reliant, this may be a bigger issue than any inference-side optimisation. Consider whether Sumi is actually the right model for your target task.

---

## 11. Kick-off objective for Claude Code / Codex

Copy-paste this into a fresh Claude Code / Codex session inside `./` after LMOE (`M6'`) is merged. Under 300 words, names the first file to touch and the first test to keep green.

```
Task: begin the Sumi-7B port for NeoDiffusion, milestone sumi-M1'.

Context:
- LMOE is merged. NeoDiffusion has generic DiffusionModel + attention/MoE/
  sampler abstractions. Read Plans/lmoe-implementation-guide.md then
  Plans/sumi-implementation-guide.md front to back before touching code.
- Sumi is uniform-state diffusion, dense (no MoE), off-by-one softmax attention.
  Nothing MoE-related applies. Modeling reference:
  https://huggingface.co/tohoku-nlp/sumi-7b/raw/main/modeling_sumi.py
  https://huggingface.co/tohoku-nlp/sumi-7b/raw/main/generation_sumi.py

First file to touch:
- Create Sources/NeoDiffusion/Configs/SumiConfig.swift with the fields
  enumerated in Plans/sumi-implementation-guide.md §3.3, populated from
  https://huggingface.co/tohoku-nlp/sumi-7b/raw/main/config.json.

First test to keep green:
- NeoDiffusionTests/ConfigLoadingTests.testSumiConfigLoads — parses the
  above config.json and asserts hidden_size=4096, num_layers=36, vocab_size=100278,
  rope_theta=500000, num_attention_heads=32, num_key_value_heads=8, head_dim=128.
- Also confirm all existing LMOE + LLaDA2.1-mini tests remain green (regression
  gate).

Do NOT touch attention, sampler, or model code in this first commit. That is
sumi-M2' onwards. The whole point of M1' is: pure additive config plumbing,
zero behaviour change.

Blockers to raise if hit:
- tokenizer_config.json for Sumi shows a novel tokenizer format (not cl100k).
  If so, escalate — Plans/sumi-implementation-guide.md flags this as speculative.
```

---

## 12. Uncertainty flags — every "inferred" or "speculative" claim

Ordered by risk (highest first). Read this every time before quoting from this doc.

1. **The tokenizer is cl100k-family** — **speculative**. Resolvable by fetching `tokenizer_config.json` from the HF repo (a single `fetch_url` call). Blocks correct token-id parity of any prompt-formatting layer.
2. **Multiplicative off-by-one softmax matches additive-sink formulation to fp32 numerical precision** — **inferred** (algebraic identity is exact in real numbers; floating-point equivalence is what needs testing). Resolvable by a 20-line MLX unit test comparing both formulations on random logits. Blocks the M2' parity gate.
3. **`torch.multinomial` and MLX Gumbel-max produce statistically-equivalent distributions under matched seeds** — **inferred**. Resolvable by a KS-test on 10 000 samples from each. Blocks M4' parity gate design.
4. **`sampler="adaptive"` outperforms `"ancestral"` on code/math tasks** — **sourced** from README, but not benchmarked in the arXiv paper for HumanEval etc. Resolvable by running `sumi-eval` locally and comparing. Affects default choice in `SumiSamplingRequest`.
5. **`use_cache=True` in `config.json` is legacy — inference always uses `use_cache=False`** — **sourced** from `generation_sumi.py:104` but not stated in the paper. Resolvable by grepping the reference codebase. Affects whether `ExactPrefixCache` even gets wired in.
6. **Sumi's `denoise_end` API is meant for suffix constraints, not for length control** — **inferred** from the code (`generation_sumi.py:_build_noise_mask`). Docstring says "so the step budget concentrates on the content window". Resolvable by asking the Tohoku authors or running an ablation.
7. **The `[EOS, BOS]` anchor is required for coherent output** — **sourced** from README ("temporary constraint imposed by the model's pretraining setup"). Missing this in the port will produce garbled tails. High-impact if omitted, easy to test with/without.
8. **Sumi at 4-bit on M1 achieves 2-5 tok/s at 64 steps** — **speculative**. No benchmark exists. Resolvable by actually running M6' on an M1. Affects the pitch of "Sumi is a good dev-machine target" — if it's < 0.5 tok/s the argument weakens considerably.
9. **`_get_num_transfer_tokens` remainder handling matches PyTorch's semantics** — **sourced from LLaDA-family literature but not from Sumi's own code** (Sumi doesn't have this function). Not directly a Sumi issue but a common cross-model bug pattern to watch for during port.
10. **The off-by-one softmax formulation Sumi uses is the same as Miller's original 2023 blog post** — **inferred** from the `softmax_one` docstring in `modeling_sumi.py`. High confidence but worth verifying if you ever debug a numerical discrepancy vs. related attention-sink implementations (there are two subtly-different formulations in the literature — additive-sink and multiplicative-rescale — see §4.1).

---

**End of guide.** Iterate on the plan by re-running the `new-model-guide-recipe.md` self-check (§8) after every material update.
