# Block Diffusion

**Summary**: A diffusion framework where generation proceeds in blocks rather than token-by-token or full-sequence. Each block is denoised jointly, and blocks are processed autoregressively. Enables efficient AR-to-dLLM adaptation. Also applies when the denoised object is a compressed continuous latent block rather than a discrete token block.  
**Aliases**: block-wise diffusion, chunked diffusion, semi-autoregressive diffusion, block-causal prior (latent variant)  
**Status**: draft  
**Last updated**: 2026-07-02  
**Sources**:
- [[fast-dllm]]
- [[cola-dlm]]

---

## Definition

Block diffusion is a diffusion framework that generates text in fixed-size blocks (e.g., 16-32 tokens) rather than:
- Token-by-token (AR models)
- Full sequence at once (full-sequence diffusion)

Each block undergoes multiple denoising steps, and blocks are generated autoregressively (one block after another).

## Why it matters

Block diffusion offers a practical balance:
- More parallelism than AR (multiple tokens per step).
- More stable than full-sequence diffusion (smaller search space).
- Enables efficient adaptation of AR models to diffusion paradigm.

## Mechanism

### Block Processing

1. **Block selection**: Choose which block to generate next.
2. **Block denoising**: Apply multiple denoising steps to the block.
3. **Block commitment**: Once block is stable, commit and move to next.
4. **Repeat**: Process subsequent blocks autoregressively.

### Complementary Attention Mask

Fast-dLLM uses a special attention mask:
- Prefix attends backward to all previous tokens.
- Current block attends to prefix + within block.
- Future blocks are masked (not yet generated).

This enables bidirectional context within block while maintaining AR objectives.

### Hierarchical Caching

Block-level cache: Stores context from previously generated blocks.
Sub-block cache: Tracks stability within current block.

### Continuous-Latent Variant (Cola DLM)

Block diffusion is not limited to discrete token blocks. Cola DLM ([[cola-dlm]]) applies the same block-causal factorization to a compressed *continuous latent* sequence instead of raw tokens: `p_ψ(z0) = p_ψ(z0^(1)) · Π_b p_ψ(z0^(b) | z0^(<b))`, with bidirectional attention within a latent block and causal dependence across blocks. A block-causal DiT learns this latent prior via Flow Matching, and a separate conditional decoder (not the diffusion process) realizes the actual text — see [[continuous-latent-prior-transport]]. This decouples block size in *token* terms from block size in *diffusion-network* terms: with DiT block size 16 and VAE patch size `p`, one denoising block realizes `p × 16` text tokens after decoding (**sourced**, Section 5.3 of the Cola DLM paper). This is a materially different mechanism from Fast-dLLM's discrete complementary-attention-mask block diffusion, even though the block-causal *structure* (bidirectional-within-block, causal-across-block) is the same shape.

### Denoising Steps per Block — Diminishing Returns

Cola DLM's step-count ablation (Section 4.4.2, Fig. 9a) found a consistent pattern that generalizes to block-diffusion tuning broadly: quality improves sharply from 1–2 to 4–8 steps, then saturates after ~16–32 steps, with most of the practical gain already captured at 8–10 steps. At DiT block size 16, this corresponds to an **idealized 1.6–2.0× reduction in sequential generation depth versus AR decoding** (16 tokens generated in 8–10 sequential steps) — "idealized" because this ignores per-step overhead and any CFG cost (see [[classifier-free-guidance-scale-tradeoff]]), which the paper applies on top (best config: 16 steps, CFG=7). **Sourced**, but specific to Cola DLM's latent block-diffusion setting — not verified for Fast-dLLM's discrete-token block diffusion.

## Trade-offs

- **Block size**: Larger = more parallelism but harder to denoise correctly.
- **Block steps**: More steps = better quality but slower; diminishing returns set in early (see above) — tuning step count is a genuine efficiency lever, not just a quality knob to maximize.
- **Context window**: Limited to prefix + current block (not full context).
- **Token block vs. latent block**: operating on a compressed latent block (Cola DLM) versus a discrete token block (Fast-dLLM) changes what "block size" costs — latent block size trades off against VAE patch size and decoder cost, not directly against attention/KV-cache cost the way token block size does.

## Apple Silicon implications

- Block size should align with Metal memory hierarchy.
- Hierarchical caching maps well to L1/L2/L3 cache.
- Enables efficient on-device inference.
- Compatible with LLaDA2.1's MBE approach.
- For the continuous-latent variant: a small latent block (d=16–128) is cheap to move through unified memory per denoising step relative to a full hidden-dimension block, but the decoder's per-token vocabulary projection cost is untouched by block size — see [[continuous-latent-prior-transport]] for the caveat that this may not reduce end-to-end latency as much as it reduces prior-network cost alone. **Speculative** until measured.

## Related concepts

- [[multi-block-editing-mbe]] (LLaDA's block approach)
- [[complementary-attention-mask]] (Fast-dLLM's mask)
- [[hierarchical-caching]] (Fast-dLLM's caching)
- [[semi-autoregressive]] (related concept)
- [[continuous-latent-prior-transport]] (Cola DLM's latent-block variant)
- [[text-vae-latent-interface]] (patch size, the other lever on tokens-per-block for the latent variant)

## Open questions

- What is optimal block size for Apple Silicon?
- Can block size adapt based on task?
- How does it compare to token-level diffusion?
- Does the 8–10 step diminishing-returns pattern found for Cola DLM's latent blocks also hold for discrete-token block diffusion (Fast-dLLM), or is it specific to the smoother continuous latent trajectory?

## Related pages

- [[fast-dllm]] (source)
- [[llada2-1-tech-report]] (LLaDA2.1 uses similar block approach)
- [[multi-block-editing-mbe]] (related block concept)
- [[cola-dlm]] (continuous-latent block-causal variant)