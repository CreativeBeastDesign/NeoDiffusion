# NeoDiffusion — DiffusionGemma-26B-A4B Implementation & Optimisation Guide

**Status**: draft, for André's review
**Written**: 2026-07-07, mid-M5, forward-looking (post-LMOE port target)
**Companion documents**: [`lmoe-implementation-guide.md`](./lmoe-implementation-guide.md) (the structural template — read that first if unfamiliar), [`handoff-post-M5.md`](./handoff-post-M5.md), [`phase-3-optimisation-roadmap.md`](./phase-3-optimisation-roadmap.md), `new-model-guide-recipe.md` (the recipe this guide obeys).
**Primary sources (retrieved 2026-07-07)**:
- DiffusionGemma model card, [`google/diffusiongemma-26B-A4B-it`](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
- Config: [`config.json`](https://huggingface.co/google/diffusiongemma-26B-A4B-it/raw/main/config.json), [`generation_config.json`](https://huggingface.co/google/diffusiongemma-26B-A4B-it/raw/main/generation_config.json), [`tokenizer_config.json`](https://huggingface.co/google/diffusiongemma-26B-A4B-it/raw/main/tokenizer_config.json)
- Modeling: [`modeling_diffusion_gemma.py`](https://github.com/huggingface/transformers/tree/main/src/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py) (1688 lines), [`generation_diffusion_gemma.py`](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/diffusion_gemma/generation_diffusion_gemma.py) (1242 lines)
- HF docs: [`transformers/model_doc/diffusion_gemma`](https://huggingface.co/docs/transformers/en/model_doc/diffusion_gemma)
- vLLM day-0 blog: [`vLLM support for DiffusionGemma`](https://vllm.ai/blog/2026-06-10-diffusion-gemma)
- Google announcement: [`DiffusionGemma: 4x faster text generation`](https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/)
- Visual guide: [`Grootendorst, A Visual Guide to DiffusionGemma`](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-diffusiongemma)
- Entropy-bound sampler reference: [`arXiv:2505.24857`](https://arxiv.org/abs/2505.24857) (cited in `generation_diffusion_gemma.py` line 422)
- MLX 4-bit port: [`mlx-community/diffusiongemma-26B-A4B-it-4bit`](https://huggingface.co/mlx-community/diffusiongemma-26B-A4B-it-4bit)
- QAT warning: [`mlboydaisuke/DiffusionGemma-26B-A4B-CoreAI`](https://huggingface.co/mlboydaisuke/DiffusionGemma-26B-A4B-CoreAI)
- Fast-dLLM (block-wise approx KV cache for MDMs): [`nvlabs.github.io/Fast-dLLM`](https://nvlabs.github.io/Fast-dLLM/)
- FreeCache + Guided Diffusion (prompt-side cache): [`arXiv:2505.21467`](https://arxiv.org/abs/2505.21467)
- dLLM-Cache (adaptive prompt/response caching): [`arXiv:2506.06295`](https://arxiv.org/abs/2506.06295)
- dKV-Cache (delayed KV cache for DLMs): [`arXiv:2505.15781`](https://arxiv.org/abs/2505.15781)
- Elastic-Cache (attention-aware drift, depth-aware refresh): [`arXiv:2510.14973`](https://huggingface.co/papers/2510.14973)
- BiCache (shared-prefix DLM caching): [`arXiv:2606.07571`](https://arxiv.org/abs/2606.07571)
- DAWN (dependency-aware parallel unmasking): [`arXiv:2602.06953`](https://arxiv.org/abs/2602.06953)
- E2D2 (encoder–decoder discrete diffusion): [`arXiv:2510.22852`](https://arxiv.org/html/2510.22852v1)

Every load-bearing claim below is either **sourced** (traceable to one of the above), **inferred** (reasoned from sourced facts), or **speculative** — flagged inline.

---

## TL;DR — the honest take before the plan

DiffusionGemma is a *different animal* from LLaDA2.1-mini and LMOE. Do not treat it as "another dLLM checkbox to tick" — most of the parity code you built for the LLaDA family will not apply, and one primary-source finding suggests the port is **conditionally recommended, not obviously worth doing**. Five hard facts up front, all **sourced**:

1. **It is an encoder–decoder, not a decoder-only, dLLM.** One Gemma 4 backbone runs in two modes: **encoder mode** with causal attention (writes KV, called once per canvas to prefill/commit) and **decoder mode** with bidirectional attention (reads KV read-only, called `max_denoising_steps` times per canvas) ([`vLLM blog`](https://vllm.ai/blog/2026-06-10-diffusion-gemma), [`Grootendorst`](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-diffusiongemma), [`modeling_diffusion_gemma.py`](https://github.com/huggingface/transformers/tree/main/src/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py) `DiffusionGemmaDecoderTextAttention.is_causal = False`, hardcoded). **NeoDiffusion's whole `ExactPrefixCache` semantics need to be revisited.** Under DiffusionGemma the encoder KV *is* exact and stationary during a canvas denoising loop — but the decoder never appends to it, so the "commit tokens → append to KV" step happens *outside* the denoising loop, on a *separate encoder forward pass* that reprocesses the freshly-denoised canvas causally. This is closer to a T5-style encoder–decoder than to LLaDA's absorbing-mask diffusion.
2. **It is a uniform-state (renoising) sampler, not a masked-diffusion sampler.** There is no `[MASK]` token in the canvas. The canvas is initialised with **random tokens sampled uniformly from the 262k vocab** and refined by accepting confident positions and *re-noising* (re-randomising) the rest ([`generation_diffusion_gemma.py:394-404`](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/diffusion_gemma/generation_diffusion_gemma.py), [SGLang cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/Google/DiffusionGemma) confirms "no mask token"). This kills every optimisation in your Resources notes that relies on `[MASK]` semantics: [[block-wise-mask-caching]], [[distance-decay]] dropout on suffix MASK tokens, [[most-attended-drift]] on MASK-heavy attention distributions, [[per-token-early-stopping]] using MASK-vs-token confidence, [[hierarchical-decoding]] of a MASK span. That's ~60% of your current Phase 3 shopping list gone at a stroke.
3. **The sampler is EntropyBoundSampler + adaptive stopping, not confidence-thresholded remasking.** `entropy_bound=0.1`, `max_denoising_steps=48`, temperature schedule linear 0.8→0.4, adaptive stopping when *(mean_entropy < 0.005 AND argmax stable for stability_threshold=1 steps)* ([`generation_config.json`](https://huggingface.co/google/diffusiongemma-26B-A4B-it/raw/main/generation_config.json), [`generation_diffusion_gemma.py:226-234`](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/diffusion_gemma/generation_diffusion_gemma.py)). In practice loops terminate in ~15-20 steps not 48 ([`vLLM blog`](https://vllm.ai/blog/2026-06-10-diffusion-gemma) says "generating 15-20 tokens per forward pass" at canvas_length=256 which implies ~13-17 accepted steps). Your `ThresholdParallelDecoder` / draft-and-edit code is not applicable. **You need a new sampler class (`EntropyBoundSampler`) alongside a fresh `StableAndConfidentStoppingCriteria`.**
4. **The MoE structure is dense-MLP + sparse-MoE running in parallel and summed at every layer, not exclusively MoE.** Each decoder/encoder layer has *both* a full dense MLP (`intermediate_size=2112`) *and* a routed MoE block (`num_experts=128, top_k=8, moe_intermediate_size=704`) whose outputs are added ([`modeling_diffusion_gemma.py:649-664`](https://github.com/huggingface/transformers/tree/main/src/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py); "1 shared expert" in marketing = the full dense MLP). This is unusual — most MoEs treat the shared expert as *one more expert*, not as a full parallel branch. Router is **softmax + top-k, weights renormalised to sum to 1, then × learned `per_expert_scale`; no groups, no bias, FP32 for router softmax only**. Alpha-MoE-style dense+MoE fusion is *harder* here than for LMOE because there are two intermediates (2112 and 704) to fuse. **Your `LLaDA2MoEGate` and `SparseMoEBlock` do not carry over.**
5. **Weights are 25.2B params, ~51.6 GB BF16 across 11 shards.** M1 16 GB **cannot** run this model even at 4-bit (~18-19 GB minimum). MLX-community reports the [4-bit MLX port](https://huggingface.co/mlx-community/diffusiongemma-26B-A4B-it-4bit) as ~15.3 GB on disk (matches the [`gemma-4-26b-a4b-it-nvfp4`](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-nvfp4) which reports 15.26 GB). **This model is Studio-only for you** — 96-192 GB M2/M3 Ultra required for BF16 parity. On M2 Max 96 GB, existing community ports run at ~8-35 tok/s at 4-bit.

There's also a **sixth honesty finding that is not "in the config" but is a showstopper** if you don't plan for it:

6. **The released BF16 weights are QAT master weights that degenerate at full precision.** The [Core AI port model card](https://huggingface.co/mlboydaisuke/DiffusionGemma-26B-A4B-CoreAI) explicitly states that HF Transformers and unquantized MLX both produce incoherent output — coherent generation *requires* the QAT-adjusted int4 expert grid. If true, **NeoDiffusion's BF16-parity-gate discipline (M4/M5) does not port to DiffusionGemma at all** — you cannot use "matches BF16 reference on 32 fixture prompts" as your parity oracle because the BF16 reference is broken by design. You need to switch to a *4-bit-vs-4-bit* parity oracle against transformers int4, or against the MLX-community 4-bit port. Flagged **sourced (single primary source)** — recipe §4 asks for verification. This is either the biggest correctness surprise of the port or a misunderstanding on the port author's part. Verify before touching the milestones.

The good news: many *sampler-level* optimisations that came from LLaDA-family papers *do not apply*, but a handful of *encoder–decoder-native* optimisations from adjacent literature ([Fast-dLLM](https://nvlabs.github.io/Fast-dLLM/), [FreeCache](https://arxiv.org/abs/2505.21467), [BiCache](https://arxiv.org/abs/2606.07571), [dLLM-Cache](https://arxiv.org/abs/2506.06295), [DAWN](https://arxiv.org/abs/2602.06953)) map onto DiffusionGemma naturally, and the vLLM blog + user reports on Reddit ([r/LocalLLaMA/1u554eo](https://www.reddit.com/r/LocalLLaMA/comments/1u554eo/)) give concrete step-count tuning that yields ~3× speedups out of the box with negligible quality loss.

**My recommendation**: postpone this port to after LMOE lands and the M4/M5 blocker is unstuck. In particular, do not start DiffusionGemma work while the dev machine is M1 16 GB — you cannot even load-test the model there. Treat this document as *scoping* rather than immediate build-plan. See §11.

---

## 0. Recommended reading order

1. This §0–§2 (delta, honesty findings, and the sampler algorithm).
2. §3 (the multi-model refactor — how much of the LLaDA-family abstraction survives an encoder–decoder shape).
3. §4 (MLX sketches — SplitQKVAttention needs a v-shares-k variant for global layers; Encoder + Denoiser wiring).
4. §5 (Metal sketches — where fused kernels actually pay for themselves given the dense+MoE-in-parallel structure).
5. §6 (Resources optimisations — most of the LLaDA-family notes do *not* apply; new arXiv menu at the end).
6. §7 (memory + quantisation logistics — the QAT question determines the whole plan).
7. §8-§9 (milestones, risks) — will be short because much is contingent on §7's QAT answer.
8. §10 (kick-off objective — the one paragraph you paste into Claude Code / Codex when you start).
9. §11 (uncertainty flags — read this every time before committing to a claim from this doc).

---

## 1. DiffusionGemma model quick facts (sourced from `config.json` + `modeling_diffusion_gemma.py`, 2026-07-07)

Side-by-side against LMOE and LLaDA2.1-mini, so deviations are visible. Everything is **sourced** unless flagged.

| Property | LLaDA2.1-mini | LMOE (LLaDA-MoE-7B-A1B) | **DiffusionGemma-26B-A4B** |
|---|---|---|---|
| Total params | ~16 B | ~7 B | **~25.2 B** (active ~4 B; A4B in the name) |
| Layers | 20 (0 dense, 1-19 MoE) | 16 (all MoE) | **30 all MoE**, layer 0 not special |
| Hidden size | 2048 | 2048 | **2560** |
| Architecture shape | decoder-only, absorbing-mask diffusion | decoder-only, absorbing-mask diffusion | **encoder–decoder (one model, two modes)** |
| Attention masking | block-causal via loop discipline (mask is 0/1) | fully bidirectional (`is_causal=False` everywhere) | **encoder: causal (sliding+full); decoder: bidirectional over canvas, reads encoder KV read-only** |
| Layer types (attention) | uniform | uniform | **`layer_types` alternates 5 sliding + 1 full, repeating** (`["sliding_attention"]*5 + ["full_attention"]` × 5 = 30 layers) |
| Sliding-window attn | wired 4096, inert | none | **1024, 5-in-6 layers**, `rope_theta=10000` |
| Full attn | uniform full | full | **1-in-6 layers, `global_head_dim=512, kv_heads=2`, `partial_rotary_factor=0.25`, `rope_theta=1_000_000`, `rope_type="proportional"`** |
| Heads (Q / KV) | 16 / 4 (GQA) | 16 / 16 (MHA) | **sliding: 16 / 8 (GQA, head_dim=256); full: 16 / 2 (GQA, global_head_dim=512)** |
| Head dim | 128 | 128 | **256 (sliding), 512 (full)** |
| Rotary shape | partial (first 64/128) θ=600 000 | full θ=50 000 | **full on sliding θ=10 000; partial (32/128 of `global_head_dim=512` → 128 dims) on full θ=1 000 000, `rope_type="proportional"`** |
| QKV projection | fused | split | **split**, `attention_bias=false` |
| Q/K/V norm | Q + K only | Q + K only | **Q + K + V (V norm has `with_scale=False` — no learnable scale)** |
| Attention logit softcap | none | none | **`final_logit_softcapping=30.0`** applied to output logits before sampler |
| Dense FFN | layer 0 only, intermediate 5120 | none | **all layers, intermediate 2112, sums in parallel with MoE** |
| MoE experts | 256 routed + 1 shared, intermediate 512 | 64 routed, no shared, intermediate 1024 | **128 routed, no separate "shared" (dense MLP is the parallel branch), moe_intermediate=704, `hidden_activation="gelu_pytorch_tanh"`** |
| Active experts / token | 8 | 8 | **8** |
| Router | sigmoid + group-limited + expert_bias + ×2.5 scale + renorm | softmax + top-k, no groups, no bias, no scale, no renorm | **softmax + top-k, weights renormalised to sum to 1, then × learned `per_expert_scale`, FP32 for router softmax only, no groups, no bias** |
| Sampling family | LLaDA2.x draft-and-edit (M2T+T2T, τ_mask, τ_edit, S/Q modes) | classic LLaDA (`low_confidence_remask` + top-k schedule) | **uniform-state renoising with `EntropyBoundSampler` + `StableAndConfidentStoppingCriteria`; block-autoregressive outer loop** |
| Canvas length | 32 (blocks) | 32 (blocks) | **256** (`canvas_length` in config) |
| Denoising steps | ~64-128 total | 128 total | **`max_denoising_steps=48` budget, terminates in ~13-17 in practice per [`vLLM blog`](https://vllm.ai/blog/2026-06-10-diffusion-gemma)** |
| Temperature | fixed | fixed | **linear schedule t_min=0.4 → t_max=0.8**, applied via `LinearTemperatureScheduleLogitsProcessor` |
| Self-conditioning | none | none | **yes** — previous-step softmax → embedding-matrix multiply → gated FFN → added to token embeddings next step |
| Vocab | 157 184 | 157 184 | **262 144** (GemmaTokenizer family, multimodal) |
| `mask_token` | id 156 895 | id 156 895 | **`<mask>` id present but not used at sampling time** (uniform-state doesn't need it) |
| `bos/eos/pad` | 156 892/156 892 | 156 892/156 892 | **`bos=<bos>, eos=[1, 106, 50], pad=0`, `tie_word_embeddings=True`** |
| Weights on disk | ~33 GB BF16 | ~14.7 GB BF16 | **~51.6 GB BF16 (11 shards)**; 4-bit MLX ~15.3 GB; QAT int4 ~13-14 GB |
| Reference cache in HF | `use_cache=false` | `use_cache=false` | **`use_cache=true`** (dynamic/static cache for encoder KV; **required** — the algorithm literally reuses encoder KV across denoising steps) |
| Multimodal | text-only | text-only | **text + image + audio + video tokens in tokenizer** (though weights on HF are the text/vision variant) |

**Key deltas from LMOE** (call these out because they're what will bite the port):

- **Encoder-decoder is a single Gemma-4 26B backbone.** [`Grootendorst`](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-diffusiongemma) confirms: same weights, dynamic switch between causal-attention encoder mode and bidirectional-attention decoder mode. This is [E2D2's idea (arXiv:2510.22852)](https://arxiv.org/html/2510.22852v1) applied to Gemma-4. The switch is via the `is_causal` flag on `DiffusionGemmaDecoderTextAttention` and via which layer stack is called (`self.model.encoder` vs `self.forward`) — see [`generation_diffusion_gemma.py:733-741`](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/diffusion_gemma/generation_diffusion_gemma.py).
- **Alternating sliding+full attention with heterogeneous heads.** This is Gemma-4 architecture: 5-of-6 layers are sliding-window attention with 16Q/8KV heads and head_dim=256; the 6th layer is full attention with 16Q/**2 KV** heads and `global_head_dim=512`. The full layer also **skips v_proj** (v=k) and uses `partial_rotary_factor=0.25` — only 128 dims of the 512 head are rotated. This is a **major** implementation lift because your existing `LLaDA2Attention` assumes uniform heads across all layers.
- **Self-conditioning is a real subsystem, not a docstring.** [`generation_diffusion_gemma.py:562-579`](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/diffusion_gemma/generation_diffusion_gemma.py) shows: at each denoising step `i`, the logits from step `i-1` are softmaxed, matmul'd against `embed_tokens.weight` to get a probability-weighted embedding, passed through a gated MLP (`DiffusionGemmaSelfConditioning`), and *added* to the canvas token embeddings before layer 0. The encoder passes see zeroed self-conditioning. This is roughly +50 M params in the SC MLP and **must** be present for the model to converge in <48 steps — without it convergence is empirically much slower ([`arXiv:2505.24857`](https://arxiv.org/abs/2505.24857) is the cited theoretical source for the entropy-bound sampler and self-conditioning combination).
- **Adaptive stopping is stateful across steps** ([`generation_diffusion_gemma.py:498-540`](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/diffusion_gemma/generation_diffusion_gemma.py)): stability requires *N consecutive steps* with identical argmax. `stability_threshold=1` in the default generation config means "one preceding step" — pragmatically ~2 consecutive identical steps. Your engine must track this state per-batch-item.

---

## 2. The reference sampling algorithm, pinned

This is the parity target. The engine must reproduce it token-for-token at temperature 0 before deviating.

### 2.1 The algorithm, in `generation_diffusion_gemma.py`'s own words

Direct quote from [`generation_diffusion_gemma.py:562-579`](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/diffusion_gemma/generation_diffusion_gemma.py):

```
1. Autoregressive canvas generation loop:
    a. Encode all previous tokens using the encoder, to get the KV cache.
    b. Prepare data for the new denoising loop.
    c. For each denoising (diffusion) step:
        i.   Run the decoder, taking the current canvas, the encoder KV cache, and
             the self-conditioning logits (if available) as inputs.
        ii.  Select new canvas tokens from the output logits (argmax with temperature warp).
        iii. Apply the sampler acceptance and renoising logic.
        iv.  Update the diffusion stopping criteria.
        v.   Use the output logits as self-conditioning logits for the next step.
    d. Append the new denoised canvas to the sequence of generated tokens.
    e. Check autoregressive stopping criteria; break outer if all sequences finished.
    f. Prepare tensors for the next block.
```

### 2.2 EntropyBoundSampler, verbatim (from `generation_diffusion_gemma.py:406-448`)

```python
def accept_canvas(self, current_canvas, denoiser_canvas, logits, cur_step):
    """
    Accept k tokens with lowest entropy such that
      sum_i^k entropy_i - max(entropy_1, ..., entropy_k) <= entropy_bound
    where the LHS is an upper bound on joint mutual information between these tokens,
    so the sampler chooses k tokens that are approximately independent.
    Originally proposed in https://arxiv.org/pdf/2505.24857.
    """
    dist = torch.distributions.Categorical(logits=logits)
    token_entropy = dist.entropy()                                              # (B, canvas_length)
    sorted_token_entropy, sorted_indices = torch.sort(token_entropy, dim=-1)    # ascending
    cumulative_entropy = torch.cumsum(sorted_token_entropy, dim=-1)
    # sorted_token_entropy = cumulative maximum entropy (sorted ascending)
    sorted_selection_mask = cumulative_entropy - sorted_token_entropy <= self.entropy_bound
    self.accepted_token_mask = torch.scatter(
        input=torch.zeros_like(sorted_selection_mask),
        dim=-1, index=sorted_indices, src=sorted_selection_mask
    )
    accepted_canvas = torch.where(self.accepted_token_mask, denoiser_canvas, current_canvas)
    return accepted_canvas

def renoise_canvas(self, accepted_canvas, cur_step):
    renoise_mask = ~self.accepted_token_mask
    random_canvas = torch.randint(0, self.vocab_size, accepted_canvas.shape)
    return torch.where(renoise_mask, random_canvas, accepted_canvas)
```

### 2.3 StableAndConfidentStoppingCriteria, verbatim (from `generation_diffusion_gemma.py:498-540`)

```python
def __call__(self, argmax_canvas, logits, **kwargs):
    # Stability: argmax the same across `stability_threshold` steps
    if self.stability_threshold == 0:
        stable = torch.ones(logits.shape[0], dtype=torch.bool)
    else:
        if self.argmax_canvas_history is None:
            self.argmax_canvas_history = torch.full(
                (self.stability_threshold, argmax_canvas.shape[0], argmax_canvas.shape[1]),
                -1, dtype=argmax_canvas.dtype
            )
        stable = (self.argmax_canvas_history == argmax_canvas[None, :, :]).all(-1).all(0)
        self.argmax_canvas_history = torch.roll(self.argmax_canvas_history, -1, dims=0)
        self.argmax_canvas_history[-1] = argmax_canvas
    # Confidence: mean entropy < confidence_threshold
    dist = torch.distributions.Categorical(logits=logits)
    confident = torch.mean(dist.entropy(), dim=-1) < self.confidence_threshold
    return stable & confident
```

### 2.4 Self-conditioning wiring (paraphrased from `modeling_diffusion_gemma.py`)

Between steps, `self_conditioning_logits` from step `t-1` are:
1. Softmaxed over vocab.
2. Multiplied by `embed_tokens.weight` (transposed): `sc_emb = softmax(logits_{t-1}) @ E`.
3. Passed through the small gated MLP `DiffusionGemmaSelfConditioning` (Wi_gate ⊙ silu(Wi_up) → Wo).
4. Added to the canvas token embeddings: `h_0 = embed_tokens(canvas_t) + gated(sc_emb)`.

On the first step, self-conditioning is zeroed. On encoder passes, self-conditioning is zeroed via `self_conditioning_mask`. The gated MLP is **~30-50 M params** (inferred: hidden_size=2560 × 2 gates × intermediate).

### 2.5 Differences that break LLaDA-family parity code

- **No `[MASK]` token in the canvas**. The `renoise_canvas` function re-samples random ints in `[0, vocab_size)`. Your engine's mask-management code (masks-as-integers, `mask_token_id`) doesn't apply. Instead, track *which positions have been accepted*, not *which positions are masked*.
- **The stopping decision is per-canvas-item and stateful.** `argmax_canvas_history` is a rolling buffer of `(stability_threshold, batch, canvas_length)` — bigger than your engine's current stopping-criteria payload.
- **Two attention regimes must coexist in a single forward pass**: encoder (causal) and decoder (bidirectional). Your `LLaDA2Attention` needs a variant class per role, or an `is_causal` flag threaded through.
- **KV cache is written *only* by the encoder, and read-only during the entire denoising loop.** This is *simpler* than LLaDA's per-step KV update — a genuine speedup vector — but only if your cache abstraction supports "encoder-side write once, decoder reads N times without mutation."
- **The block-autoregressive outer loop is a canvas commit + new encoder pass, not just an `append` to the token stream**. Committing means running the encoder *again* on the freshly-denoised 256-token canvas to extend the KV cache. This is a non-negligible cost per canvas (a full 256-token causal prefill) but happens *once per canvas*, not per denoising step.

---

## 3. Multi-model refactor: from "LLaDA-family engine" to "block-diffusion engine family"

### 3.1 What survives, what needs a second implementation

Assumes your engine has an abstract `DiffusionModel` protocol (see LMOE guide §3.1). What each existing abstraction becomes under DiffusionGemma:

- **`Sampler` protocol**: your current `ThresholdParallelDecoder` / `low_confidence_remask` classes are *not applicable*. A new `EntropyBoundSampler` class implementing `accept_canvas(current, denoiser, logits, step)` + `renoise_canvas(accepted, step)` + `initialize_canvas(batch, device)` is needed. It should be a peer of, not a subclass of, `ThresholdParallelDecoder` — the state and interface are different.
- **`StoppingCriteria` protocol**: your current stopping is stateless. `StableAndConfidentStoppingCriteria` is stateful. Extend the protocol with `reset()` (called between canvases) and a rolling argmax buffer.
- **`AttentionKernel`**: needs both a `causal=True` mode (for encoder, using existing LLaDA-family kernel plus sliding-window support) and `causal=False, cross_attend_to_encoder_kv=True` mode (for decoder — bidirectional over canvas, cross-reads encoder KV). The decoder mode is a *new* kernel: it doesn't append to KV cache.
- **`KVCache`**: your `ExactPrefixCache` semantics *do* apply here for the encoder KV — but now correctly, not by accident. The encoder is genuinely causal + append-only. The decoder is genuinely read-only. This is the *cleaner* case, and DiffusionGemma is where `ExactPrefixCache` is actually mathematically exact (unlike under LMOE, where it's approximate).
- **`MoEBlock`**: your `SparseMoEBlock` gets a peer, `ParallelDenseSparseMoEBlock`, that runs `y = DenseMLP(x) + SparseMoE(x)` and returns the sum. The router is much simpler than LMOE's (no groups, no bias) — softmax + top-k + renorm + × per_expert_scale. FP32 only inside the router softmax.
- **`RotaryEmbedding`**: needs partial-rotary support (`partial_rotary_factor=0.25` on full layers) *and* `rope_type="proportional"` scaling. Your current partial-rotary code from LLaDA2.1-mini may or may not include proportional scaling — verify against [transformers' Gemma-4 RoPE](https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma3).
- **`Tokenizer`**: 262k vocab, GemmaTokenizer family, multimodal control tokens. Not compatible with LLaDA's 157k tokenizer. Needs a fresh tokenizer wrapper (import GemmaTokenizer via HF then bridge to Swift, or reimplement).

### 3.2 Suggested Package layout (delta from LMOE §3.2)

```
Packages/
├── DiffusionCore/                              (unchanged)
├── DiffusionGeneration/
│   ├── Sources/
│   │   ├── ScheduledLowConfidencePolicy.swift  (LLaDA family)
│   │   ├── DraftAndEditPolicy.swift            (LLaDA2.x)
│   │   ├── EntropyBoundPolicy.swift            ★ NEW
│   │   ├── StableConfidentStopping.swift       ★ NEW
│   │   ├── LinearTempSchedule.swift            ★ NEW
│   │   ├── BlockAutoregressiveOuterLoop.swift  ★ NEW (canvas commit + encoder re-prefill)
├── DiffusionModel/
│   ├── Sources/
│   │   ├── LLaDA2Model.swift                   (LLaDA2.1-mini)
│   │   ├── LMOEModel.swift                     (LMOE)
│   │   ├── DiffusionGemmaModel.swift           ★ NEW
│   │   ├── DiffusionGemmaEncoder.swift         ★ NEW (causal, sliding+full layers, writes KV)
│   │   ├── DiffusionGemmaDecoder.swift         ★ NEW (bidirectional, reads-only encoder KV)
│   │   ├── SelfConditioningBlock.swift         ★ NEW (softmax → embed matmul → gated MLP)
│   │   ├── ParallelDenseSparseMoEBlock.swift   ★ NEW (dense + MoE, sum outputs)
│   │   ├── SoftmaxNormedTopKRouter.swift       ★ NEW (renorm + per_expert_scale)
│   │   ├── SplitQKVAlternatingAttention.swift  ★ NEW (5:1 sliding:full layer_types)
├── DiffusionKernels/                           (largely unchanged; new fused variants come in §5)
```

### 3.3 The critical correctness question: `ExactPrefixCache` under DiffusionGemma

**Under DiffusionGemma the encoder KV is genuinely exact and stationary during a canvas denoising loop.** This is what the model was designed for.

- Encoder processes prompt with causal attention → writes KV once (`is_prefill=True`).
- Decoder does N denoising steps, each reading the encoder KV read-only. The decoder does *not* mutate the encoder KV.
- After canvas converges, `argmax_canvas` is appended to `input_ids`, and the encoder is called again on the new canvas (this time as `is_prefill=False`, only the last `canvas_length` tokens) — appending KV for the committed canvas. See [`generation_diffusion_gemma.py:733-741`](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/diffusion_gemma/generation_diffusion_gemma.py).

**Consequence**: DiffusionGemma is the *cleanest* case for `ExactPrefixCache` semantics you have. The encoder KV really is a strict causal prefix. All the wobbles you had to reason about under LMOE (bidirectional attention makes prefix KV approximate) simply don't exist. **This is a small win for the abstraction.**

However — and this is the surprising bit — because the encoder KV is *not* updated during denoising, the standard KV-cache optimisations that motivated LLaDA-family Fast-dLLM / Elastic-Cache work are less impactful. Fast-dLLM's "block-wise approximate KV cache" is trying to *avoid recomputing KV every denoising step* — DiffusionGemma already does that by construction. Elastic-Cache's "when to refresh KV" is moot: it never refreshes during the canvas loop. **So a whole class of dLLM caching optimisations you were counting on for Phase 3 will show much smaller wins on DiffusionGemma than on LLaDA models.** Honest note: this is a *point in DiffusionGemma's favor architecturally* but a *point against porting it as an "optimisation playground"* — the low-hanging fruit was already picked by Google.

### 3.4 The reference outer loop, MLX pseudocode (delta from LMOE §3.5)

```swift
func generate(prompt: MLXArray, canvasLength: Int = 256, maxNewTokens: Int = 512,
              maxDenoisingSteps: Int = 48, entropyBound: Float = 0.1,
              tMin: Float = 0.4, tMax: Float = 0.8,
              stabilityThreshold: Int = 1, confidenceThreshold: Float = 0.005) -> MLXArray {

    var inputIds = prompt
    var pastKV: EncoderKVCache = .empty
    let sampler = EntropyBoundSampler(entropyBound: entropyBound, canvasLength: canvasLength, vocab: 262144)
    let stopper = StableAndConfidentStopping(stabilityThreshold: stabilityThreshold,
                                             confidenceThreshold: confidenceThreshold)
    let numCanvases = Int(ceil(Double(maxNewTokens) / Double(canvasLength)))

    var isPrefill = true
    for _ in 0..<numCanvases {
        // 1a. Encode all previous tokens (prompt on first pass, last canvas on subsequent)
        let encoderInput = isPrefill ? inputIds : inputIds[-canvasLength...]
        pastKV = model.encoder(encoderInput, past: pastKV, isCausal: true)
        isPrefill = false

        // 1b. Initialize canvas with uniform random tokens
        var currentCanvas = MLXRandom.randint(0..<262144, shape: [1, canvasLength])
        var scLogits: MLXArray = MLXArray.zeros([1, canvasLength, 262144])  // zeroed on step 0
        stopper.reset()

        // 1c. Denoising loop (max 48 steps, usually terminates in ~13-17)
        var argmaxCanvas = currentCanvas
        var finished = false
        for step in stride(from: maxDenoisingSteps, through: 1, by: -1) {
            let (logits, newScLogits) = model.decoder(
                canvas: currentCanvas,
                encoderKV: pastKV,           // read-only
                selfConditioning: scLogits,
                isCausal: false               // bidirectional over canvas
            )
            // Linear temperature schedule
            let t = tMin + (tMax - tMin) * Float(step) / Float(maxDenoisingSteps)
            let scaled = logits / t
            // Denoiser proposal (argmax with temperature)
            let denoiserCanvas = MLXRandom.categorical(scaled)
            argmaxCanvas = scaled.argMax(axis: -1)
            // Sampler acceptance
            let accepted = sampler.acceptCanvas(current: currentCanvas,
                                                denoiser: denoiserCanvas,
                                                logits: scaled, step: step)
            // Adaptive stopping check
            if stopper(argmax: argmaxCanvas, logits: scaled) { finished = true; break }
            // Renoise unaccepted positions
            currentCanvas = sampler.renoiseCanvas(accepted, step: step)
            scLogits = newScLogits
        }

        // 1d. Commit: append argmax to inputIds; the NEXT loop iteration will run encoder on it
        inputIds = MLXArray.concat([inputIds, argmaxCanvas], axis: -1)
        if hasEosInCanvas(argmaxCanvas) { break }
    }
    return inputIds
}
```

**Note the argmax-vs-sampled distinction** (sourced, [`Grootendorst`](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-diffusiongemma)): the *committed* tokens are the *argmax of processed logits*, not the sampled/accepted noisy canvas that's carried across steps. The sampler's role is just to decide which positions to *stop sampling* — the final output is the argmax at convergence time. **This is critical for parity** — get it wrong and you'll match nothing.

### 3.5 Reference user-tuned config that trebles speed (sourced, [r/LocalLLaMA/1u554eo](https://www.reddit.com/r/LocalLLaMA/comments/1u554eo/))

A widely-shared community tuning: `--max-denoising-steps 9 --max-post-steps 2 --threshold 0.6 --max-transfer-per-step 128` produces ~3× throughput vs defaults on M2 Ultra 96 GB with negligible HumanEval score drop. The `--max-post-steps` and `--threshold` are references to a variant CLI wrapper — the underlying knob is `max_denoising_steps` at 9 (down from 48). **Verify on your own eval before adopting**: this is a *single community report*, not a paper.

---

## 4. MLX-Swift implementation sketches

### 4.1 SplitQKVAlternatingAttention — sliding vs full per-layer

Key wrinkle: the QKV projection dims differ *by layer type*. For sliding layers: Q [16 * 256 = 4096, 2560], K, V [8 * 256 = 2048, 2560]. For full layers: Q [16 * 512 = 8192, 2560], K [2 * 512 = 1024, 2560], V has *no separate projection* (v = k after k_norm). This is the trickiest structural difference.

```swift
struct DiffusionGemmaLayer {
    let layerType: LayerType   // .sliding or .full
    let qProj: Linear
    let kProj: Linear
    let vProj: Linear?         // nil for .full layers (v == k)
    let qNorm: RMSNorm
    let kNorm: RMSNorm
    let vNorm: RMSNorm         // with_scale = false → no learnable scale, just normalise

    func attention(x: MLXArray, encoderKV: KVCache?, isCausal: Bool) -> MLXArray {
        let (numHeads, headDim) = layerType == .sliding ? (16, 256) : (16, 512)
        let numKVHeads = layerType == .sliding ? 8 : 2

        var q = qProj(x).reshaped(-1, numHeads, headDim)
        var k = kProj(x).reshaped(-1, numKVHeads, headDim)
        var v: MLXArray
        if let vp = vProj { v = vp(x).reshaped(-1, numKVHeads, headDim) }
        else               { v = k }                                       // full layer: v = k

        q = qNorm(q); k = kNorm(k); v = vNorm(v)

        // Rotary:
        //   sliding: full rotary, θ=10_000, over head_dim=256
        //   full:    partial rotary factor=0.25 → rotate first 128/512 dims, θ=1_000_000, rope_type=proportional
        (q, k) = applyRotary(q, k, layerType: layerType)

        // Cross-attend to encoder KV in decoder mode (concat encoder KV before attention)
        if let encKV = encoderKV, !isCausal {
            k = MLXArray.concat([encKV.k(atLayer: idx), k], axis: 1)
            v = MLXArray.concat([encKV.v(atLayer: idx), v], axis: 1)
        }

        // Sliding-window mask for .sliding layers when isCausal
        let mask = buildMask(layerType, isCausal, slidingWindow: 1024)
        let scores = scaledDotProduct(q, k, mask)
        return outputProj(scores.softmax() @ v)
    }
}
```

**Uncertainty flag**: I have not verified whether MLX's built-in `scaledDotProductAttention` supports the `partial_rotary` + `rope_type="proportional"` combination out of the box. If it doesn't, expect ~2-3 days of RoPE debugging. **Speculative** — check MLX 0.31+ release notes.

### 4.2 SelfConditioningBlock — MLX sketch

```swift
struct DiffusionGemmaSelfConditioning {
    let embedTokens: Embedding      // shared with token embeddings, tie_word_embeddings=true
    let gate: Linear
    let up: Linear
    let down: Linear

    func callAsFunction(scLogits: MLXArray, mask: MLXArray) -> MLXArray {
        // scLogits: [B, canvas_length, vocab_size=262144]
        // mask:     [B]  — bool, false = zero-out this example's SC
        let probs = MLX.softmax(scLogits, axis: -1)
        // Expected embedding: probs @ E — batched matmul, O(canvas_length * vocab_size * hidden)
        let scEmb = probs @ embedTokens.weight.T                      // [B, canvas, hidden]
        // Gated MLP: swish(gate(x)) * up(x) → down
        let gated = MLX.silu(gate(scEmb)) * up(scEmb)
        let output = down(gated)                                       // [B, canvas, hidden]
        // Zero out where mask says so
        return mask.reshaped(-1, 1, 1) * output
    }
}
```

**Performance hazard, honest**: the `probs @ E` matmul is `[B, 256, 262144] @ [262144, 2560]` — that's `2560 × 262144 × 256 × B` FLOPs = ~40 GFLOPs per denoising step, per batch item. At 15 steps per canvas that's 600 GFLOPs per canvas *just for self-conditioning* on a 26B model whose total FLOPs per canvas denoising pass are dominated by attention + MoE. The [vLLM blog](https://vllm.ai/blog/2026-06-10-diffusion-gemma) doesn't call this out but it's non-trivial. **Consider using a low-rank approximation** (top-k probs only, sparse matmul) as a Phase 3 optimisation — but validate parity first.

### 4.3 ParallelDenseSparseMoEBlock — MLX sketch

```swift
struct ParallelDenseSparseMoEBlock {
    let denseMLP: GatedMLP           // intermediate=2112
    let router: SoftmaxTopKRouter    // 128 experts, top-8, renorm=true, per_expert_scale learnable
    let experts: SparseMLPExperts    // 128 × (gate + up + down, intermediate=704)

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let densePath = denseMLP(x)                                // [B, T, hidden]
        let (topKIndices, topKWeights) = router(x)                 // [B, T, 8], [B, T, 8]
        let moePath = experts.gather(x, topKIndices, topKWeights)  // [B, T, hidden]
        return densePath + moePath
    }
}
```

Router:
```swift
struct SoftmaxTopKRouter {
    let routerLinear: Linear                    // [128, 2560]
    let perExpertScale: MLXArray                // [128] — learnable

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        // FP32 for router numerical stability
        let logits = routerLinear(x.astype(.float32))               // [B, T, 128]
        let (topKVals, topKIdx) = MLX.topk(logits, k: 8, axis: -1)  // [B, T, 8]
        // Softmax over just the top-8 values (renormalise to sum to 1)
        let weights = MLX.softmax(topKVals, axis: -1)               // [B, T, 8]
        // Multiply by learnable per-expert scale (indexed by topKIdx)
        let scale = MLX.take(perExpertScale, topKIdx)               // [B, T, 8]
        return (topKIdx, (weights * scale).astype(x.dtype))
    }
}
```

### 4.4 The forward pass — encoder mode vs decoder mode

```swift
class DiffusionGemmaModel {
    let embedTokens: Embedding
    let layers: [DiffusionGemmaLayer]           // 30 layers
    let finalNorm: RMSNorm
    let lmHead: Linear                          // tied with embedTokens
    let selfCond: DiffusionGemmaSelfConditioning

    func encoder(_ ids: MLXArray, past: KVCache) -> KVCache {
        var h = embedTokens(ids)                // no self-conditioning in encoder mode
        for layer in layers {
            h = layer.forward(h, encoderKV: nil, isCausal: true, past: past)
        }
        // Encoder doesn't produce logits — just updates KV cache
        return past
    }

    func decoder(canvas: MLXArray, encoderKV: KVCache, selfConditioning: MLXArray?) -> (logits: MLXArray, scLogits: MLXArray) {
        var h = embedTokens(canvas)
        if let sc = selfConditioning {
            let mask = MLXArray.ones([canvas.shape[0]], dtype: .bool)   // all examples active
            h = h + selfCond(scLogits: sc, mask: mask)
        }
        for layer in layers {
            h = layer.forward(h, encoderKV: encoderKV, isCausal: false, past: nil)
        }
        h = finalNorm(h)
        let logits = lmHead(h)
        // final_logit_softcapping = 30
        let capped = 30.0 * MLX.tanh(logits / 30.0)
        return (logits: capped, scLogits: capped)   // scLogits and output logits are the same tensor
    }
}
```

### 4.5 What optimised looks like — before Metal

- **Encoder KV is written once, read N times**. Store it in device memory as contiguous per-layer arrays; the decoder loop never mutates it. Good for cache-locality.
- **Router table lookup**: the top-K expert dispatch is where you'd naively lose 2-3× on Apple Silicon. MLX has a `gather` primitive; if it's slow, the fused-MoE Metal kernel becomes worthwhile (§5).
- **Self-conditioning matmul**: as noted, this is `[B, 256, 262144] @ [262144, 2560]`. Consider batching this across denoising steps if you find MLX schedules it inefficiently — but do that *after* parity is achieved.

---

## 5. Metal sketches — where fused kernels pay off

### 5.1 Where the shape of the model actually helps

The `ParallelDenseSparseMoEBlock` is the most Metal-friendly of your MoE variants because:
- Dense MLP path is a straight `up → silu × gate → down` chain, no dispatch.
- MoE path is 8 experts × 128-choose-k lookup — the same pattern as LMOE, so your Alpha-MoE analog carries.

**Fusing dense + MoE into one megakernel** is *possible* but I'd rate it **low-priority speculative** — the dense-and-MoE run in parallel and are summed, so they can be dispatched concurrently on separate command buffers, gaining most of the fusion benefit for none of the kernel complexity. **Do this via MLX stream separation, not custom Metal, on the first pass.**

### 5.2 The sliding-vs-full attention split as a kernel dispatch decision

Your attention kernel needs to handle:
- **Sliding attention**: 16 Q / 8 KV heads, head_dim=256, window=1024, full rotary. This is a fairly standard sliding-window attention kernel.
- **Full attention**: 16 Q / 2 KV heads (highly GQA), head_dim=512, partial rotary (128 dims), rope_theta=1M. Head dim 512 is *wide* — check Metal shared-memory constraints; you may need to tile head_dim across threadgroups.

Two separate kernels are cleanest. Fusing them into one templated kernel saves ~5% (**speculative**) at the cost of substantial complexity.

### 5.3 The self-conditioning matmul as a fused kernel candidate

The `probs @ E` matmul in self-conditioning is a `[B, 256, 262144] @ [262144, 2560]` matmul that runs *per denoising step*. On M2 Ultra this is ~10 ms per step at BF16, ~2.5 ms at INT4 — 15 steps = 37 ms per canvas at INT4. That's ~20-30% of the canvas's decoding cost.

**If** MLX doesn't fuse this well, writing a `softmax → matmul → gate/up/down` fused Metal kernel is worthwhile. But **verify** MLX's default schedule first — Apple's team has done substantial work on chained matmuls for exactly this shape (attention logits × V).

### 5.4 What NOT to write in Metal

- **Do not** write a fused encoder-decoder switch. The mode switch is a Swift-level decision (`isCausal: Bool`), and Metal has no way to help.
- **Do not** write custom kernels for the router — softmax + top-k on 128 experts is a MLX primitive that will be fast enough.
- **Do not** write custom kernels for the sampler's `accept_canvas` — the entropy computation is O(B × canvas × vocab) which is fine on the GPU-side of MLX out of the box.

---

## 6. Which Resources optimisations apply to DiffusionGemma

Classified against the specific DiffusionGemma sampling regime, not against dLLMs in general. **Every note in your `Resources/` folder was written under the LLaDA / LLaDA2.x paradigm.** Many of them assume `[MASK]` tokens, absorbing-state diffusion, or shared prompt+response representation — none of which hold for DiffusionGemma.

### 6.1 Base / structural

| Optimisation | DiffusionGemma applicability | Reason |
|---|---|---|
| [[block-diffusion]] | **Applies** — same paradigm, DiffusionGemma is block-autoregressive with 256-token canvases. Actually implements a canonical version. | Sourced: [E2D2 arXiv:2510.22852](https://arxiv.org/html/2510.22852v1) argues explicitly that encoder-decoder + block diffusion is the "right" combination. |
| [[block-wise-causal-attention]] | **Applies** to the encoder mode. **Doesn't apply** to decoder mode (which is fully bidirectional). | Encoder is causal + sliding; decoder is bidirectional over canvas. |
| [[block-wise-mask-caching]] | **Doesn't apply.** No `[MASK]` token in the canvas — uniform-state renoising uses random tokens. | Sourced: [`generation_diffusion_gemma.py:394-404`](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/diffusion_gemma/generation_diffusion_gemma.py). |
| [[sliding-window-attention]] | **Applies natively** — 5-of-6 layers are sliding-window with window=1024. Not an optimisation; it's the model's architecture. | Sourced from `config.json`. |
| [[local-attention-dllm]] | **Applies (already implemented)** as sliding-window on 5/6 layers. Adding *further* local-attention would risk quality. | The 1/6 full-attention layers exist for a reason (long-range dependencies). |

### 6.2 Caching optimisations

| Optimisation | DiffusionGemma applicability | Reason |
|---|---|---|
| [[adaptive-kv-caching]] | **Partially applies** — but the mechanism is different. DiffusionGemma's encoder KV is already stationary during a canvas loop; adaptive selection has less to do. | The value of adaptive selection here is across *canvases*, not across denoising steps. |
| [[elastic-cache]] | **Doesn't apply as-is** — Elastic-Cache's core mechanism (attention-aware drift + depth-aware refresh across denoising steps) targets *decoder-only* models that recompute KV every step. DiffusionGemma's encoder KV doesn't drift during denoising. | Sourced: [arXiv:2510.14973](https://huggingface.co/papers/2510.14973). The "when to refresh" question is answered trivially: never during the canvas loop. |
| [[elastic-cache-v2]] | Same as above — the mechanism targets a bottleneck that DiffusionGemma avoids by construction. | See above. |
| [[layer-wise-kv-dynamics]] | **Structural insight applies**, but not the depth-aware refresh mechanism. | Depth-aware refresh is moot when KV isn't being refreshed. |
| [[most-attended-drift]] | **Doesn't apply** — drift-based caching is not needed. | See above. |
| [[selective-layer-refresh]] | **Doesn't apply.** | See above. |
| [[depth-aware-refresh]] | **Doesn't apply.** | See above. |
| [[radix-caching]] | **Applies** for encoder-side prompt caching across requests. | Prompt processing is standard causal; radix caching of common prefixes is applicable and worthwhile. |
| [[scratchpad-redundancy]] | **Doesn't apply.** Uniform-state renoising doesn't have a "scratchpad" of masked suffix tokens. | The canvas has no distinct MASK region; it's a full 256-token block of noisy tokens. |
| [[suffix-dropout]] | **Doesn't apply.** | See above. |
| **FreeCache** (new, [arXiv:2505.21467](https://arxiv.org/abs/2505.21467)) | **Applies to the encoder side.** FreeCache = cache prompt KV, don't recompute it as later canvases arrive. DiffusionGemma does this by default via encoder KV reuse — but *between requests* (radix caching) this is a fresh win. | Sourced. |
| **dLLM-Cache** (new, [arXiv:2506.06295](https://arxiv.org/abs/2506.06295)) | **Partially applies** — prompt-cache reuse yes, response-cache (V-verify similarity) — no, because DiffusionGemma renoises the response every step. | Sourced. |
| **BiCache** (new, [arXiv:2606.07571](https://arxiv.org/abs/2606.07571)) | **Applies** — BiCache is bidirectional prefix caching for DLM shared prefixes. On DiffusionGemma the encoder KV is exactly a bidirectional prefix (from the decoder's perspective). Reported 36-98% throughput improvement for shared-prefix serving. | Sourced. |
| **dKV-Cache** (new, [arXiv:2505.15781](https://arxiv.org/abs/2505.15781)) | **Doesn't apply directly** — dKV-Cache targets a decoder-only DLM's step-to-step KV drift. DiffusionGemma has separate encoder KV that doesn't drift. | Sourced. |

### 6.3 Sampling / decoding

| Optimisation | DiffusionGemma applicability | Reason |
|---|---|---|
| [[denoising-step-importance]] | **Applies** — "not all steps matter equally" applies to any iterative denoiser. Concretely, an *early exit* schedule (drop steps from the low-noise / high-confidence end) could work here, and the reference [Reddit tuning](https://www.reddit.com/r/LocalLLaMA/comments/1u554eo/) suggests reducing 48 → 9 steps with minimal quality loss. | The mechanism (train small model for robust steps) doesn't apply; the finding does. |
| [[iteration-smoothing]] (IterSmooth) | **Applies to the extent that DiffusionGemma already does it.** DiffusionGemma's self-conditioning IS essentially "reuse logits from previous step to inform next step" — the same idea as IterSmooth's "convert masked-position logits into expected embedding". DiffusionGemma feeds it through a gated MLP; IterSmooth feeds it directly. | Structurally the same idea. IterSmooth-in-addition-to-SC is likely not worthwhile. |
| [[hierarchical-decoding]] | **Doesn't apply** — hierarchical decoding sub-divides a masked span. DiffusionGemma has no masked span, it has a canvas of noisy tokens all at once. | Sourced. |
| [[in-place-chain-of-thought]] | **Might apply** — the ICE mechanism (embed reasoning tokens in the canvas, watch answer-token confidence) is dLLM-generic and could work on DiffusionGemma. | **Speculative**. Would need experimentation. |
| [[per-token-early-stopping]] | **DiffusionGemma already does an analog of this** at the whole-canvas level via `StableAndConfidentStoppingCriteria`. Per-token version is not directly transferable (positions are re-noised until accepted; there's no per-position accept-and-hold). | Sourced. |
| [[editable-state-evolution]] | **Doesn't apply.** DiffusionGemma commits argmax at end of canvas denoising — there is no T2T edit operation. | Sourced. |
| [[mask-to-token-m2t]] | **Doesn't apply** — no MASK tokens. | Sourced. |
| [[token-to-token-t2t]] | **Doesn't apply.** | Sourced. |
| [[multi-block-editing-mbe]] | **Doesn't apply** — canvases, once committed, are not revisited. | Sourced. |
| [[speedy-mode-s-mode]] / [[quality-mode-q-mode]] | **Doesn't apply as-is** — no τ_mask parameter. The analog is `entropy_bound` and `max_denoising_steps`. | Sourced. |
| **DAWN** (new, [arXiv:2602.06953](https://arxiv.org/abs/2602.06953)) | **Might apply** — DAWN's dependency-graph parallel unmasking is dLLM-generic; whether it improves over DiffusionGemma's entropy-bound "independent tokens" heuristic is an empirical question. Reported 1.80-8.06× speedup on masked DLMs. | **Speculative** — DAWN targets masked models; DiffusionGemma's entropy-bound is already dependency-aware in a different sense. |
| **Fast-dLLM** (new, [nvlabs.github.io/Fast-dLLM](https://nvlabs.github.io/Fast-dLLM/)) | **Confidence-aware parallel decoding applies.** The threshold-based unmasking is essentially what DiffusionGemma's entropy-bound sampler does. Fast-dLLM's DualCache (masked-suffix caching) doesn't apply (no masked suffix). | Sourced. Reported 27.6× throughput on LLaDA-family; probably 2-3× at most on DiffusionGemma because Google already optimised the caching path. |

### 6.4 Quantisation

| Optimisation | DiffusionGemma applicability | Reason |
|---|---|---|
| [[per-block-fp8-quantization]] | **Applies** — Google publishes FP8 numbers ([blog](https://blog.google/innovation-and-ai/technology/developers-tools/diffusion-gemma-faster-text-generation/) says "1000+ TPS on H100 FP8"). MLX doesn't yet support FP8 natively (as of MLX 0.31); NVFP4 (4-bit) is the Apple-native equivalent. | See §7. |
| **NVFP4 4-bit weight quantisation** (new) | **Recommended** — [`mlx-community/gemma-4-26b-a4b-it-nvfp4`](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-nvfp4) is the standard MLX port. 15.26 GB on disk. Reports 14-49% faster decode than affine UD-4bit on MoE models ([Wulff IT source](https://wulffit.de/artikel/gemma4-opencode-praxis/)). | Sourced. |
| **TurboQuant 3-bit KV cache** (new) | **Applies to encoder KV**. Since encoder KV is stationary during canvas denoising and gets read 15+ times per canvas, quantising it saves memory bandwidth on every read. Reports 63% KV memory reduction on Gemma-4-31B with quality preserved. | Sourced, [Prince Canuma via MLStreetTalk](https://x.com/MLStreetTalk/status/2040302198605943255). |
| **QAT master weights (BF16) ≠ generation quality** | **Critical honesty note.** [Core AI port](https://huggingface.co/mlboydaisuke/DiffusionGemma-26B-A4B-CoreAI) states the BF16 released weights degenerate at full precision — you need the QAT int4 expert grid for coherence. **This changes your parity oracle strategy.** | See §7.4. |

### 6.5 Kernel-level

| Optimisation | DiffusionGemma applicability | Reason |
|---|---|---|
| [[alpha-moe-megakernel]] | **Applies in principle** — fusing MoE dispatch is generally valuable. **Harder here** because dense+MoE are parallel and summed; you'd fuse dense-MLP into the same kernel or run them as separate concurrent streams. Alpha-MoE is Hopper-specific (FP8) — Apple Silicon needs a from-scratch reimplementation. | Sourced. |
| **Fused sampling epilogue** (new) | **Applies** — you can fuse `logits / temperature → softmax → argmax → entropy → accept_mask` into a single kernel. Saves ~15% of denoising-step latency (**speculative**). | Not from arXiv; my analysis. |

### 6.6 Out of scope

Everything under the LLaDA2.x draft-and-edit machinery — [[vectorized-likelihood-estimation]], [[elbo-based-block-level-policy-optimization-ebpo]], [[multi-turn-forward-mtf]], [[exposure-bias-in-dllms]], [[sglang-rollout-engine]] — is training/RL infrastructure, not inference. Skip for inference-only port.

### 6.7 Summary of the applicability shift

Out of the 35 Resources notes:
- **~10** apply directly or with minor adaptation.
- **~10** don't apply because DiffusionGemma is uniform-state / encoder-decoder / self-conditioned.
- **~5** are training-only.
- **~10** the mechanism from LLaDA papers targets a bottleneck DiffusionGemma avoids by construction.

**This is the honest finding**: DiffusionGemma port is *not* a Phase 3 optimisation playground. Google shipped it with the major dLLM inference bottlenecks already addressed. The remaining wins are quantisation (NVFP4 4-bit), reduced denoising steps (48 → 9 with the Reddit config), and radix caching of prompt encoder KV across requests.

---

## 7. Memory, quantisation, fixture logistics

### 7.1 Footprint estimates on M1 16 GB dev + M2 Ultra 96/192 GB Studio

Model weights: **25.2 B params**. All 30 layers are MoE (128 experts each, top-8) + parallel dense MLP + attention.

Per-parameter cost breakdown (**inferred** from `config.json`):
- Embeddings: 262144 × 2560 = 671 M params
- Per-layer attention: (approx, mixing sliding + full)
  - Sliding: qkv = (4096+2048+2048)*2560 = 21 M params × 25 layers = 525 M
  - Full: qkv = (8192+1024+0)*2560 = 23.6 M params × 5 layers = 118 M
  - Output proj + norms: ~7 M × 30 = 210 M
- Per-layer dense MLP: (gate+up+down) × 30 layers × (2560 × 2112 × 3) ~ 486 M
- Per-layer MoE experts: 128 experts × 3 × (2560 × 704) × 30 layers ~ 20.8 B (the bulk)
- Self-conditioning MLP: ~50 M (**inferred**, exact size not read)
- LM head: tied with embeddings, 0 additional

Total ~ 25.2 B params × 2 B/param BF16 = **50.4 GB** (close to observed 51.6 GB).

At **4-bit weights** (routed experts + dense) + BF16 for embeddings, LM head (tied so shared), and norms:
- Experts + dense MLPs: (20.8 B + 486 M) × 0.5 B/param = **10.65 GB**
- Attention weights: 853 M × 0.5 B/param = **0.43 GB**
- Embeddings + LM head (tied): 671 M × 2 B = **1.34 GB**
- Self-cond + norms: 50 M × 0.5 B = 0.025 GB + small
- Activation working set at canvas=256, batch=1: (256 × 2560 × 30 layers × 2 B) = ~40 MB per forward pass, but with KV cache: prompt KV at 8k tokens × 8 KV heads × 256 head_dim (sliding) or 2 KV heads × 512 (full) × 30 layers × 2 B ≈ **512 MB for 8k context** (**inferred**)

Total 4-bit: **~13-14 GB** on-device (matches the [tq3-g32 port](https://huggingface.co/manjunathshiva/diffusiongemma-26B-A4B-it-tq3-g32) and [mlx-4bit port](https://huggingface.co/mlx-community/diffusiongemma-26B-A4B-it-4bit) at 13.7-15.3 GB).

**M1 16 GB verdict: cannot fit.** Even the 4-bit port is 15.3 GB on disk before activation memory, KV cache, or MLX runtime overhead. M1's 16 GB is *unified* memory shared with the OS + everything else; realistic budget is ~10 GB for a model. **You need a bigger dev machine to run DiffusionGemma at all.**

M2 Ultra 96 GB verdict: fits comfortably at any quantisation. Fits at BF16 too (~52 GB) with 44 GB headroom. This is the target machine.

### 7.2 The QAT question — a fork in the road

If [Core AI port's claim](https://huggingface.co/mlboydaisuke/DiffusionGemma-26B-A4B-CoreAI) that BF16 master weights are broken is correct:

- **Your BF16-parity M4/M5 gate does not port.** You cannot use "the BF16 reference output on 32 prompts matches ours to N tokens" as your correctness oracle, because the BF16 reference is intentionally broken.
- **The parity target becomes the MLX 4-bit port**, or `transformers` int4 output. Both are downstream artifacts, not the "reference" implementation. This is a *methodological regression* in your engine's correctness discipline.
- **You need to write a QAT-aware quantisation step** or use the pre-quantised MLX port's weights directly. Doing your own quantisation on the BF16 weights will produce a broken model.

Verification path (Phase 0 for the DiffusionGemma port):
1. Load `google/diffusiongemma-26B-A4B-it` in Transformers Python at BF16, generate 50 prompts.
2. If output is coherent → Core AI claim is wrong; proceed with BF16 parity as usual.
3. If output is degenerate → Core AI claim is correct; switch parity oracle to `mlx-community/diffusiongemma-26B-A4B-it-4bit`.

**Blocking item for the milestones.**

### 7.3 Fixture plan (M4' / M5' analog)

- 32 prompts × 2 modes (Q-analog: default `entropy_bound=0.1, steps=48`; S-analog: `entropy_bound=0.3, steps=9`).
- Fixed random seed for the `initialize_canvas` random tokens and for `renoise_canvas`. This is *critical* — a floated seed makes the output completely non-reproducible.
- Parity metric: exact token match on argmax_canvas at each canvas commit. Not perplexity, not BLEU. **Token-level identity to reference.**
- Reference source: whichever wins §7.2 (BF16 Transformers or MLX 4-bit port). Document explicitly which.

### 7.4 M2 Ultra Studio real-weight parity

Studio has enough memory to load BF16 (52 GB) with headroom for MLX runtime and KV cache. Convert BF16 → NVFP4 in one pass using MLX's `mlx_lm.convert` (or your own script if you need per-layer quantisation control).

---

## 8. Milestones (analogous to Phase 2 §4 in `phase-2-implementation-guide.md`)

**Precondition** (do not enter M0'' without): §7.2's QAT question is answered, and a Studio (not M1) is available for at least 20 hours/week.

- **M0''**: Verify QAT question; download `google/diffusiongemma-26B-A4B-it` on Studio; run 5 prompts through HF Transformers as the parity baseline. **Owner: André. Acceptance: table of 5 prompt outputs + their `tokens_per_forward`.**
- **M1''**: Port GemmaTokenizer to Swift, verify against HF tokenizer on 20 prompts (byte-exact). **Acceptance: `swift test testTokenizer` passes for the 20 prompts.**
- **M2''**: Weight map + safetensors loader. All 11 shards load into an MLX array dict; keys match a golden weight-map JSON. **Acceptance: `MLX.arrayDict.count == expected_count` and each tensor's shape matches config.**
- **M3''**: Encoder-only forward pass parity. Feed the same 256 tokens through HF encoder mode and MLX encoder mode, check KV cache tensor equality to N decimals. **Acceptance: `max(abs(hf_kv - mlx_kv)) < 1e-2` per layer (BF16 tolerance).**
- **M4''**: Decoder-only forward pass parity. Same 256-token canvas + same encoder KV, check output logits. **Acceptance: same tolerance as M3''.**
- **M5''**: Full canvas denoising parity with fixed seed. Same random canvas init, same self-conditioning, verify committed argmax_canvas matches HF. **Acceptance: token-exact match on 5 prompts × 1 canvas each.**
- **M6''**: Multi-canvas generation parity (up to 512 tokens). **Acceptance: token-exact match on 5 prompts × 2 canvases each.**
- **M7''**: NVFP4 4-bit quantisation + parity gate at 4-bit vs 4-bit. **Acceptance: match the MLX 4-bit port on 32 prompts.**
- **M8''**: Speedup optimisations: Reddit-config (steps=9), radix cache, TurboQuant KV. **Acceptance: TPS ≥ 30 on M2 Ultra 96 GB, quality-preserved on HumanEval subset.**

Total effort estimate: **6-10 weeks Studio time** for M0''-M6''. Post-baseline optimisations M7''-M8'' add another **3-4 weeks**.

---

## 9. Risks / opens carried into implementation

1. **QAT-master-weights question** (§7.2): if it's real, your correctness discipline needs a rethink. **Block M0'' on this.**
2. **Partial rotary + `rope_type="proportional"`** on full-attention layers: MLX support status unclear. Verify on MLX 0.31+ before M4''.
3. **Self-conditioning matmul cost**: not free; may need a fused kernel or a low-rank approximation. Defer to M8''.
4. **26 GB active + 51 GB weights ≠ 16 GB dev machine.** Do not attempt this port on M1 16 GB. Studio is a hard prerequisite.
5. **The "worth doing" question**: given §6.7 (most of your Phase 3 optimisation notes don't apply), is DiffusionGemma actually a *good* second/third target for NeoDiffusion? Honest answer: **only if what you care about is model coverage, not optimisation research.** For optimisation research, LMOE and LLaDA2.1-mini give you more surface to explore.
6. **Multimodal tokens in vocab**: text-only inference is what you want. Non-text tokens are dead but present in the 262k vocab; they won't affect generation if the input is text-only, but they *do* enlarge `probs @ E` cost in self-conditioning.

---

## 10. What to say to Claude Code / Codex when you kick off

Paste this as the initial objective. It is written for a coding agent that reads the guide but doesn't automatically obey it.

> Implement DiffusionGemma (`google/diffusiongemma-26B-A4B-it`) into NeoDiffusion. The full plan is in `Plans/diffusiongemma-implementation-guide.md` — treat it as the authoritative spec.
>
> **Before starting M0'', block on §7.2's QAT verification**: run `google/diffusiongemma-26B-A4B-it` at BF16 in Transformers on 5 prompts. If output is degenerate (repetitive, incoherent) confirm to me that the "QAT master weights broken at full precision" claim from the Core AI port is real. **If it's real, switch the parity oracle to `mlx-community/diffusiongemma-26B-A4B-it-4bit` and update M5''/M6''/M7'' accordingly.** Do not proceed with M2'' until this is settled.
>
> **Do not** re-use the LLaDA-family sampler code — DiffusionGemma is uniform-state renoising with an EntropyBoundSampler, not confidence-thresholded mask remasking. Add a fresh `EntropyBoundPolicy.swift` in `Packages/DiffusionGeneration/Sources/`. Reference the algorithm in §2.2 of the guide.
>
> **Do not** re-use `LLaDA2Attention` unmodified — DiffusionGemma has 5:1 alternating sliding + full attention with different (Q, KV) head counts per layer type. Add `SplitQKVAlternatingAttention.swift` in `Packages/DiffusionModel/Sources/`. See §4.1.
>
> **Do not** touch this without a Studio available. M1 16 GB cannot run this model at any quantisation. If you're on the dev M1, stop and hand it back to me.
>
> Track progress against the milestones M0''-M8'' in §8, and update this document's status header at each milestone. If §7.2's QAT question comes out surprising, halt and re-scope with me before touching §8.
>
> Every optimisation you consider applying, cross-check §6's applicability matrix first. Most LLaDA-family optimisations do NOT apply. Do not add "another sampler" or "another KV cache variant" without justifying against §6.

---

## 11. What I'm *not* certain about (be sceptical here)

Flagged in order of "biggest risk of being wrong":

1. **The QAT-master-weights degeneration claim** (§7.2, §1 finding #6). **Single primary source** ([Core AI port model card](https://huggingface.co/mlboydaisuke/DiffusionGemma-26B-A4B-CoreAI)). Not corroborated by Google's official blog or vLLM's day-0 post. Verify empirically before believing.
2. **The Reddit tuning `steps=9`** (§3.5). Single community report. Might be task-specific.
3. **The self-conditioning MLP size** (~50 M params, §4.2). **Inferred** from context and standard MLP scaling; I did not read the exact `DiffusionGemmaSelfConditioning` `__init__` in modeling code. Verify at M0''.
4. **NVFP4 vs BF16 quality gap for DiffusionGemma**. NVFP4 recovers 97-99% quality for models >30B ([Red Hat NVFP4 article](https://developers.redhat.com/articles/2026/02/04/accelerating-large-language-models-nvfp4-quantization)) — but DiffusionGemma is 25.2 B and the interaction with self-conditioning is untested.
5. **MLX 0.31+ support for `partial_rotary + rope_type="proportional"`** (§4.1). I did not check the exact MLX release notes. May be up-to-date, may not. Verify at M4''.
6. **Fast-dLLM applicability speedup magnitude** (§6.3). Fast-dLLM reports 27.6× on LLaDA; I claim "2-3× at most on DiffusionGemma" based on the argument that Google already fixed the KV-cache bottleneck by construction — this is analysis, not measurement. Speculative.
7. **DAWN's applicability to entropy-bound sampling** (§6.3). DAWN targets masked DLMs. Whether the dependency-graph mechanism helps a uniform-state renoising sampler is untested. Speculative.
8. **The `probs @ E` self-conditioning cost estimate** (§4.2, ~40 GFLOPs). Inferred from shape arithmetic. May be higher/lower depending on MLX's matmul kernel efficiency on this exact shape.
9. **Whether TurboQuant 3-bit KV cache is compatible with the encoder-KV-read-many pattern**. Reported to work on standard Gemma-4; not specifically tested on DiffusionGemma's encoder-decoder split. Speculative.
10. **The claim in §6.7 that DiffusionGemma is "not a Phase 3 optimisation playground"**. This is my synthesis of the applicability matrix — it's plausible but strong, and I'd want to be talked out of it if you have counter-evidence.

Everything else in this document is either directly sourced or a straightforward composition of sourced facts. If something feels wrong, it's most likely in this list.

---

## 12. Companion documents to update after this doc lands

- `phase-3-optimisation-roadmap.md`: add a "DiffusionGemma optimisation restrictions" subsection referencing §6.7.
- `Resources/`: consider adding notes for `bicache`, `freecache`, `dllm-cache`, `dkv-cache`, `dawn`, `e2d2-encoder-decoder-diffusion`. These are the arXiv sources cited in §6 that don't yet have Resources notes.
- `handoff-post-M5.md`: add a "DiffusionGemma is a future target, blocked on QAT verification and Studio access" line at the bottom of the "future targets" section.
