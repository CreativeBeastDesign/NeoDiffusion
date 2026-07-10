# Elastic-Cache

**Summary**: An adaptive KV cache management strategy for diffusion LLMs that uses attention-aware drift detection and layer-selective recomputation to maximize accuracy while minimizing decoding latency.  
**Aliases**: ElasticCache, elastic cache  
**Status**: stable  
**Last updated**: 2026-07-03  
**Sources**:
- [[elastic-cache]]
- [[elastic-cache-v2]]

---

## Version note

This page is sourced from two ingested notes covering different arXiv revisions of the same paper ("Attention Is All You Need for KV Cache in Diffusion LLMs", VILA-Lab; web-verified arXiv:2510.14973, ICLR 2026 — **sourced via web search, distinct from the two ingested PDF-derived source notes**). The v1 source note ([[elastic-cache]]) and the v2 source note ([[elastic-cache-v2]]) report different throughput numbers for overlapping benchmarks — v1 reports 45.1× on GSM8K at 512 tokens; v2 reports 8.7× on GSM8K at 256 tokens and 45.1× on "longer sequences." This page's mechanism description and the "Performance highlights" numbers below follow the v1 source note; the discrepancy with v2's numbers has not been reconciled against the original paper and is recorded here rather than silently resolved, per the wiki's contradiction-handling rule. Six of v2's named sub-mechanisms — [[attention-aware-drift-test]], [[depth-aware-refresh]], [[most-attended-drift]], [[block-wise-mask-caching]], [[layer-wise-kv-dynamics]], [[selective-layer-refresh]] — now have their own atomic concept pages that cross-reference this page for full mechanism detail rather than duplicating it.

## Definition

Elastic-Cache is a training-free, architecture-agnostic method forKV cache management in diffusion language models. Unlike fixed-period refresh schemes, it dynamically decides **when** to refresh (via an attention-aware drift test on the most-attended token) and **where** to refresh (via a depth-selective schedule that recomputes from a learned boundary layer onward). Combined with sliding window decoding and block-caching of distant MASK tokens, it achieves up to 45.1× speedup on GSM8K while maintaining or improving accuracy.

## Why it matters

Standard diffusion LLM decoders recompute QKV for all tokens at every step, leading to massive redundancy because KV states change little across most steps, especially in shallow layers. Fixed-period schemes (e.g., every k steps) waste compute when nothing has changed and miss updates when semantic revisions occur. Elastic-Cache aligns recomputation with actual state changes, reducing redundant QKV work and making diffusion LLMs practical for deployment.

## Mechanism

Elastic-Cache integrates three components:

### 1. Sliding Window Decoding

Active prediction is restricted to a contiguous window of size β (default 16) of masked tokens closest to the left. Distant MASK tokens are excluded from the current forward pass and their KV states are cached block-wise.

Let:
- `M_t`: masked positions at step `t`
- `M_t^β = M_t[:β]`: active window
- `D_{<t}`: decoded tokens up to step `t-1`

The set of tokens processed is `Q_t = D_{<t} ∪ M_{t-1}^β`.

### 2. Attention-Aware Cache Update

For each layer ℓ, determine if the cache has become stale:

- Identify the **most-attended token** among decoded positions:
  ```
  T_t^ℓ = argmax_{k ∈ D_{<t}} ∑_{q ∈ M_{t-1}^β} S_t^ℓ[q,k]
  ```
  where `S_t^ℓ` are attention weights.

- Compute **cosine similarity** between consecutive attention vectors of this token:
  ```
  σ_t^ℓ = cosine_similarity(S_{t-1}^ℓ[T_{t-1}^ℓ], S_t^ℓ[T_{t-1}^ℓ])
  ```

- If `σ_t^ℓ < γ` (threshold, default 0.9), the layer's cache is stale; set boundary `ℓ★ = ℓ`.

The threshold `γ` controls update frequency: higher → fewer updates → more speed but potential accuracy loss.

### 3. Layer-Aware Cache Update

- For layers `ℓ > ℓ★`: **recompute** KV cache from scratch (full cache for all tokens)
- For layers `ℓ ≤ ℓ★`: **reuse** cached KV for queries in `Q_t`; full cache for `I` (all tokens) when needed

This exploits the monotonic increase of KV drift with layer depth: shallow layers stabilize quickly, deep layers continue evolving.

### Block-Caching of Distant MASK Tokens

MASK tokens outside the sliding window are never recomputed; their KV states are cached in blocks (size 32) and reused when they eventually enter the window.

### Full Algorithm (Algorithm 1)

```
Require: prompt, window β, threshold γ, generation length N
Initialize: x0 = [prompt, MASK×N]; cache K,V for all tokens, all layers
t = 1
while M_t ≠ ∅:
  M_t^β = M_t[:β]
  Q_t = D_{<t} ∪ M_{t-1}^β
  ℓ★ = ∞
  for ℓ = 1 .. L:
    if ℓ > ℓ★:   # Cache Update
      H̃_t,ℓ[I], K̃_t,ℓ[I], Ṽ_t,ℓ[I] = cache_all(I)
      Q_t,ℓ[I], K_t,ℓ[I], V_t,ℓ[I] = FFN(H̃_t,ℓ[I])
      H_t,ℓ+1[I], S_t,ℓ[T_{<t}] = MHA(Q_t,ℓ[I], K̃_t,ℓ[I], Ṽ_t,ℓ[I])
    else:         # Cache Reuse
      H̃_t,ℓ[Q_t], K̃_t,ℓ[Q_t], Ṽ_t,ℓ[Q_t] = cache(Q_t)
      Q_t,ℓ[Q_t], K_t,ℓ[Q_t], V_t,ℓ[Q_t] = FFN(H̃_t,ℓ[Q_t])
      H_t,ℓ+1[Q_t], S_t,ℓ[T_{<t}] = MHA(Q_t,ℓ[Q_t], K̃_t,ℓ[I], Ṽ_t,ℓ[I])
      compute σ_t^ℓ = cosine_similarity(S_{t-1}^ℓ[T_{t-1}^ℓ], S_t^ℓ[T_{t-1}^ℓ])
      if σ_t^ℓ < γ: ℓ★ = ℓ; H̃_t,ℓ+1[I] = H_t,ℓ+1[Q_t]  # propagate full states
    T_t^ℓ = argmax_{k∈D_{<t}} ∑_{q∈M_{t-1}^β} S_t^ℓ[q,k]
  Decode new tokens: x_{t+1}, D_{t+1} = decode(x_t, M_t^β)
  Update: M_{t+1} = M_t \ D_{t+1}; T_t = ⋃_ℓ T_t^ℓ; t = t+1
return x_{t-1}
```

## Trade-offs

- **γ threshold**: Higher values (e.g., 0.95) preserve accuracy (83.0% on GSM8K) but reduce throughput (98.4 t/s). Lower values (e.g., 0.7) maximize throughput (138.6 t/s) but drop accuracy (77.6%). Default 0.9 balances both.
- **Window size β**: Larger β increases parallel tokens per step (fewer iterations) but reduces cacheable MASK tokens, raising per-step compute. Optimal β task-dependent (16–32).
- **Top-k tracking**: Monitoring Top-1 most-attended token is cheapest; Top-k (10–15) improves accuracy slightly but adds overhead.
- **Memory**: Block-caching reduces peak memory (18.11 GB vs 19.62 GB baseline on LLaDA-1.5, 512 tokens) by avoiding intermediate state storage for all layers.

## Apple Silicon implications

- **Metal implementation**: The attention-aware drift test (cosine similarity on attention weights) can be fused into the attention kernel or computed in a lightweight post-pass using Metal compute shaders.
- **Layer-selective recomputation**: On Apple Silicon's unified memory, recomputing deeper layers while reusing shallow caches could reduce memory traffic. The boundary ℓ★ could be determined empirically (e.g., last 1/3 of layers).
- **Sliding window**: Fixed-size window maps well to Metal threadgroups; optimal β likely differs for Apple Silicon's L1/L2 cache sizes (e.g., 32 vs 16).
- **Block-caching**: Aligns with Metal's resource buffers; each block stored contiguously for efficient reuse.
- **Memory savings**: Demonstrated ~1.4 GB reduction enables on-device deployment with limited GPU memory.
- **Theoretical guarantees**: Proofs of layer-wise KV drift monotonicity and most-attended token stability provide principled foundations for porting to Apple Silicon, where hardware characteristics may shift optimal hyperparameters.

## Open questions

- Reconcile the v1 vs v2 source note throughput numbers (45.1× GSM8K-512 vs 8.7× GSM8K-256 / 45.1× "longer sequences") against the original paper directly — currently unresolved in this wiki.
- How does optimal `γ` vary on Apple Silicon vs CUDA? Could it be auto-tuned per device?
- Can the layer boundary ℓ★ be predicted statically from architecture or must it be learned online? Does it generalize across prompts?
- What is the latency overhead of cosine similarity computation on Metal? Could approximate dot product suffice?
- Does Elastic-Cache combine synergistically with other dLLM optimizations like Iteration Smoothing or Credit Decoding? Both alter attention patterns; would drift detection need adjustment?
- How does block-caching interact with per-block quantization (FP8) to further reduce memory bandwidth on Apple Silicon?
- Could the sliding window limit maximum parallel decoding degree compared to hierarchical decoding? How does TPF compare?
- Can the theoretical analysis be extended to bound final output quality as a function of cache staleness?
- How does performance scale with long contexts (>8k tokens)? Does the sliding window need dynamic resizing?
- What is the impact on multimodal models like LLaDA-V? The paper uses lower `γ=0.7`; why?

## Related concepts
- [[attention-aware-drift-test]] (atomic breakout of the "when to refresh" mechanism, from the v2 source note)
- [[depth-aware-refresh]] (atomic breakout of the "where to refresh" mechanism, from the v2 source note)
- [[most-attended-drift]] (atomic breakout of the observation justifying the drift test)
- [[block-wise-mask-caching]] (atomic breakout of the distant-MASK caching mechanism)
- [[layer-wise-kv-dynamics]] (atomic breakout of Theorem A.8's depth-drift monotonicity)
- [[selective-layer-refresh]] (atomic breakout of the combined when+where policy)
- [[vicinity-kv-cache-refresh]] (dInfer's selective recompute; Elastic-Cache is adaptive via attention)
- [[sliding-window-decoding]] (core component of Elastic-Cache)
- [[attention-aware-decoding]] (uses attention patterns to drive decisions)
- [[layer-wise-computation]] (selective layer recomputation)
- [[block-caching]] (caching distant MASK tokens in blocks)
- [[kv-drift]] (the metric being detected)
- [[most-attended-token]] (key signal for stale cache detection)
- [[fast-dllm]] (Dual Cache baseline)
- [[dkv-cache]] (fixed-period refresh baseline)
- [[deepcache]] (fixed-interval baseline)
- [[confidence-aware-decoding]] (used in experiments, not part of Elastic-Cache)

## Related pages

- [[elastic-cache]] (source note)
- [[dinfer-framework]] (alternative KV cache approach)

## Performance highlights

- **GSM8K (LLaDA-1.5, 512 tokens)**: 81.35% accuracy, 117.2 t/s, **45.1× speedup** over baseline (2.6 t/s), outperforming Fast-dLLM (80.82%, 36.8 t/s, 14.2×) and DeepCache (83.1%, 60.9 t/s, 23.4×).
- **Memory**: 18.13 GB peak vs 19.62 GB baseline and 21.42 GB Fast-dLLM.
- **Scaling**: Throughput increases with generation length (256: 58.0 t/s; 512: 117.2 t/s; 1024: 169.8 t/s), opposite to Fast-dLLM which degrades.
- **Cache update frequency**: ~10% of layers refreshed on average (`γ=0.9`), demonstrating high selectivity.
- **Multimodal (LLaDA-V, MathVerse 512)**: 29.2% accuracy, 32.3 t/s (single-token), 42.2 t/s (parallel).

## Hyperparameters

- `β` (window size): default 16 (32 for some GSM8K configs)
- `γ` (attention threshold): default 0.9 (lower for more updates)
- Block size: 32 tokens
- `ε` (confidence threshold for parallel decoding): 0.9 (used in experiments but external to Elastic-Cache)

## Ablation insights

- **Window size**: β = 16 optimal for most tasks; β = 8 gives higher accuracy but lower throughput; β > 32 degrades accuracy (caching too many MASKs that later become relevant).
- **Threshold γ**: Lower γ increases throughput but reduces accuracy; higher γ does opposite. γ = 0.9 provides best balance.
- **Top-k tracking**: Top-15 yields +0.3% accuracy over Top-1 but reduces throughput; Top-1 is recommended for most deployments.
- **Block-caching ablations**: Removing block-caching reduces throughput by 30–40% with minimal accuracy impact, confirming its importance.
- **AdaBlock integration**: Adaptive window sizing hurts performance; fixed β suffices with attention-aware updates.

## Theoretical foundations

- **Layer-wise KV drift monotonicity** (Theorem A.8): Under assumptions of bounded representations and progressive unmasking, expected KV drift increases with layer depth. Justifies layer-selective refresh.
- **Attention concentration and drift** (Theorem A.9): The most-attended token's KV drift is bounded by the average drift plus `O(√(dk R_ℓ / N))`. Validates using it as a conservative staleness indicator.
