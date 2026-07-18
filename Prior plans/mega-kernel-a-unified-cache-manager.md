# Mega-Kernel A: Unified Cache Manager

**Goal**: Maximize cache hit rate and minimize recomputation by combining three orthogonal caching selection dimensions  
**Status**: idea  
**Last updated**: 2026-04-16  
**Depends on concepts**:  
- [[block-freezing]] (FreeCache)  
- [[kv-drift-test]] (Elastic-Cache)  
- [[token-stability]] (both)  
- [[mask-query-attention]] (MaskKV)  
- [[cache-eviction]] (MaskKV)  
- [[layer-wise-caching]] (Elastic-Cache)  

**Grounded in sources**:  
- [[freecache]] - block stability detection  
- [[elastic-cache]] - KV drift testing and layer-aware caching  
- [[maskkv]] - importance-based cache eviction  

---

## Problem

Existing caching methods each optimize one dimension:
- **FreeCache**: Temporal stability (step count)
- **Elastic-Cache**: KV drift (change detection)
- **MaskKV**: Token importance (attention-based)

Each method has blind spots:
- FreeCache may cache unstable tokens that change frequently
- Elastic-Cache may miss tokens that drift but are semantically similar
- MaskKV may evict recently-stable tokens in favor of high-importance tokens

No method handles **cache eviction** when capacity is exceeded.

## Proposed Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    UNIFIED CACHE MANAGER                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ STAGE 1      │    │ STAGE 2      │    │ STAGE 3      │      │
│  │ Block        │───▶│ Drift        │───▶│ Eviction     │      │
│  │ Stability    │    │ Testing      │    │ Manager      │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│        │                   │                   │               │
│        ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Blocks       │    │ Layer-wise   │    │ Importance   │      │
│  │ eligible     │    │ KV refresh   │    │ Threshold    │      │
│  │ for caching  │    │ decisions    │    │ Enforcement  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Stage 1: Block Stability Detection (FreeCache)

```
if block_stable_for_n_steps(block, n=3):
    mark_eligible_for_cache(block)
```

- Apply to each block independently
- Uses step counter per block
- Low computational overhead

### Stage 2: Layer-Aware Drift Testing (Elastic-Cache)

```
# Test most-attended token drift
drift = compute_kv_drift(most_attended_token, cached_kv)

if drift < ε_drift:
    # Shallow layers: reuse cache
    for layer in layers[:ℓ*]:
        use_cached_kv(layer)
else:
    # Refresh from ℓ* onward
    recompute_from_layer(ℓ*)
```

- Test once per step (not per-token)
- Conservative lower bound from most-attended token
- Layer boundary ℓ* learned or tuned

### Stage 3: Importance-Based Eviction (MaskKV)

```
when cache_budget_exceeded:
    # Evict lowest importance tokens
    for token in tokens_sorted_by_mask_query_score(ascending):
        if token not in attention_sinks:
            evict_from_cache(token)
            if budget_satisfied:
                break
```

- Attention sinks are never evicted
- Only when cache exceeds capacity

## Expected Benefits

- **Orthogonal selection**: Three independent criteria reduce false positives
- **High cache precision**: Only tokens passing all filters are cached
- **Eviction guarantee**: Never fail due to cache overflow
- **Layer awareness**: Shallow layers benefit from aggressive caching
- **Expected speedup**: 30-40× on long sequences

## Assumptions

- Attention sinks can be identified (typically 8-32 tokens)
- Layer boundary ℓ* is consistent across inputs
- Cache budget is known and fixed
- Mask-query attention weights can be extracted

## Risks and Unknowns

- **Complexity**: Three stages add latency overhead
- **Cascading decisions**: Failure in Stage 1 affects Stage 2
- **Threshold tuning**: ε_drift, ℓ*, budget all need calibration
- **Performance**: Combined overhead may offset gains

## Apple Silicon / Metal Mapping

- **Threadgroup strategy**: Per-block cache state in threadgroup memory
- **Memory hierarchy**: 
  - L1: Current step KV (hot)
  - L2: Cached stable KV (warm)
  - DRAM: Historical blocks
- **Synchronization**: Atomic counters for cache state
- **Fusion opportunities**: 
  - Drift test fused with forward pass
  - Mask-query extraction fused with attention
- **Likely bottlenecks**: Mask-query computation, eviction sorting

## Validation Plan

1. **Microbenchmarks**: Measure overhead of each stage independently
2. **Ablation study**: Remove each stage, measure quality/speedup tradeoff
3. **Threshold sweep**: Grid search over ε_drift, ℓ*, budget
4. **Apple Silicon profiling**: Identify Metal kernel bottlenecks
5. **Quality metrics**: Perplexity, downstream task accuracy

## Related proposals

- [[mega-kernel-b-hybrid-selection]] - Simpler two-method combination
- [[mega-kernel-c-multi-stage-pipeline]] - Sequential staging approach
- [[mega-kernel-d-depth-aware-elastic]] - Elastic-Cache focused

## Meta notes

**Why combine three methods?**
- FreeCache provides coarse block-level filtering (computational efficiency)
- Elastic-Cache provides fine layer-level precision (quality preservation)
- MaskKV provides overflow protection (reliability)

Each method addresses a different failure mode of the others.