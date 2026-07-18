# Mega-Kernel B: Hybrid Selection Kernel

**Goal**: Increase cache precision by requiring tokens to pass both feature similarity AND KV drift tests  
**Status**: idea  
**Last updated**: 2026-04-16  
**Depends on concepts**:  
- [[feature-similarity-caching]] (dLLM-Cache)  
- [[kv-drift-test]] (Elastic-Cache)  
- [[token-stability]] (both)  
- [[cosine-similarity]]  

**Grounded in sources**:  
- [[dllm-cache]] - feature similarity as stability indicator  
- [[elastic-cache]] - KV drift testing  
- [[elastic-cache-v2]] - depth-aware refinement  

---

## Problem

Single-criterion caching has false positives:
- **dLLM-Cache** (similarity threshold): Tokens may have similar features but different KV trajectories
- **Elastic-Cache** (drift threshold): Tokens may have low drift but different semantic content

Neither method alone provides sufficient precision for aggressive caching.

## Proposed Design

```
┌─────────────────────────────────────────────────────────────────┐
│                  HYBRID SELECTION KERNEL                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  For each token i at step t:                                    │
│                                                                  │
│    ┌─────────────────────────────────────────┐                  │
│    │  similarity_i = cosine(KV_current, KV_cached)             │
│    │  drift_i = ||KV_current - KV_cached|| / ||KV_current||     │
│    │                                          │                  │
│    │  IF similarity_i > θ_sim AND drift_i < ε_drift:            │
│    │      cache_token(i)  # Reuse cached KV                     │
│    │  ELSE:                                                      │
│    │      recompute_token(i)  # Fresh computation               │
│    └─────────────────────────────────────────┘                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Combined Decision Logic

```
CacheToken(token) = (similarity > θ_sim) AND (drift < ε_drift)
```

**Truth table**:

| Similarity | Drift | Action |
|------------|-------|--------|
| High (>θ_sim) | Low (<ε_drift) | **Cache** ✓ |
| High | High | Recompute (semantic change) |
| Low | Low | Recompute (feature change) |
| Low | High | Recompute (clearly unstable) |

Only the first row results in caching.

### Why AND Instead of OR?

Using OR would be too permissive (any single criterion triggers caching).
Using AND requires **both** conditions to be met:
- High similarity = semantically similar (content preserved)
- Low drift = numerically stable (KV state preserved)

This increases precision at the cost of recall (fewer tokens cached).

### Threshold Calibration

From dLLM-Cache: θ_sim = 0.95-0.99 typically works
From Elastic-Cache: ε_drift = 0.01-0.05 typically works

Calibration needed per model architecture.

## Expected Benefits

- **Higher precision**: Fewer false positives than either method alone
- **Balanced criteria**: Catches both semantic drift and numerical instability
- **Simpler than Mega-Kernel A**: Only two criteria, no eviction
- **Moderate speedup**: Trade some speed for quality assurance

## Assumptions

- Both metrics can be computed efficiently
- Thresholds are task-agnostic (transfer across tasks)
- Similarity and drift are approximately independent signals

## Risks and Unknowns

- **Lower recall**: AND operation means fewer tokens pass both tests
- **Double overhead**: Computing both similarity AND drift adds cost
- **Threshold coupling**: Optimal thresholds may depend on using both

## Apple Silicon / Metal Mapping

- **Threadgroup strategy**: Shared memory for KV comparisons
- **Memory hierarchy**: KV stored in threadgroup for fast access
- **Synchronization**: Warp-level reductions for similarity/drift
- **Fusion opportunities**: 
  - Fused kernel: compute similarity + drift in single pass
  - Use Accelerate for vector operations
- **Likely bottlenecks**: Double metric computation

## Validation Plan

1. **Threshold sweep**: 2D grid over θ_sim × ε_drift
2. **Precision/recall curve**: Measure cache hit rate vs quality
3. **Ablation**: Compare to dLLM-Cache alone, Elastic-Cache alone
4. **Apple Silicon benchmarks**: Profiling on M-chips

## Related proposals

- [[mega-kernel-a-unified-cache-manager]] - More comprehensive, adds eviction
- [[mega-kernel-d-depth-aware-elastic]] - Layer-aware version of drift-only
- [[mega-kernel-c-multi-stage-pipeline]] - Sequential staging approach