# Mega-Kernel D: Depth-Aware Elastic Kernel

**Goal**: Maximize efficiency on long sequences using Elastic-Cache v2's three empirical observations plus eviction  
**Status**: plausible  
**Last updated**: 2026-04-16  
**Depends on concepts**:  
- [[block-diffusion]] (Fast-dLLM)  
- [[kv-drift-test]] (Elastic-Cache)  
- [[depth-aware-refresh]] (Elastic-Cache v2)  
- [[maskkv-eviction]] (MaskKV)  
- [[anchor-tokens]] (LocalLeap)  

**Grounded in sources**:  
- [[elastic-cache-v2]] - depth-aware refresh + attention-aware drift  
- [[fast-dllm]] - block diffusion + hierarchical caching  
- [[localleap]] - anchor tokens + local determinism  
- [[maskkv]] - mask-query eviction  

---

## Problem

Elastic-Cache v2's three observations are powerful but:
1. Each observation requires different handling
2. Long sequences benefit from combined approach
3. Cache eviction still needed for memory management

Elastic-Cache v2 alone provides adaptive refreshing but not eviction.

## Proposed Design

```
┌─────────────────────────────────────────────────────────────────┐
│            DEPTH-AWARE ELASTIC KERNEL                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  STEP 1: BLOCK PRE-FILTER (Fast-dLLM)                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Identify blocks for progressive caching                  │   │
│  │  • Recent blocks: Full computation                        │   │
│  │  • Middle blocks: Cached + drift tested                   │   │
│  │  • Old blocks: Checked only when drift detected           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                     │
│                            ▼                                     │
│  STEP 2: BLOCK STABILITY CHECK (Elastic-Cache v2)               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Use block-wise distant MASK caching                      │   │
│  │  • Blocks stable for N steps → Aggressive caching         │   │
│  │  • Blocks unstable → Refresh + retest                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                     │
│                            ▼                                     │
│  STEP 3: DRIFT TEST + REFRESH (Elastic-Cache v2)                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Most-attended token drift test:                          │   │
│  │  • If drift < ε: Shallow layers reuse cache               │   │
│  │  • If drift ≥ ε: Full refresh from ℓ* onward              │   │
│  │  Depth-aware selective refresh:                           │   │
│  │  • Shallow layers (ℓ < ℓ*): Aggressive cache              │   │
│  │  • Deep layers (ℓ ≥ ℓ*): Conservative cache               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                     │
│                            ▼                                     │
│  STEP 4: IMPORTANCE EVICTION (MaskKV)                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  When cache budget exceeded:                              │   │
│  │  • Compute mask-query importance scores                   │   │
│  │  • Evict lowest importance (non-sink) tokens              │   │
│  │  • Preserve anchor tokens always                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Step 1: Block Pre-Filter (Fast-dLLM)

```
def block_prefilter(blocks, step):
    for block in blocks:
        distance_from_tip = step - block.last_unmasked_step
        
        if distance_from_tip <= BLOCK_THRESHOLD:
            block.cache_mode = FULL_COMPUTE
        elif distance_from_tip <= MIDDLE_THRESHOLD:
            block.cache_mode = CACHED_TESTED
        else:
            block.cache_mode = CHECK_ON_DRIFT
    
    return blocks
```

### Step 2: Block Stability Check (Elastic-Cache v2)

```
def check_block_stability(block, step):
    consecutive_stable = count_consecutive_unmasked(block, step)
    
    if consecutive_stable >= N_STABLE_STEPS:
        block.stability = STABLE
        # Eligible for aggressive caching
    else:
        block.stability = UNSTABLE
        # Requires refresh + retest
```

### Step 3: Drift Test + Depth-Aware Refresh (Elastic-Cache v2)

```
def drift_test_and_refresh(token, cached_kv, current_kv):
    drift = compute_drift(most_attended_token(current_kv), cached_kv)
    
    if drift < ε_drift:
        # Test most-attended token only (conservative lower bound)
        for layer in layers[:ℓ*]:
            use_cached_kv(layer)  # Shallow: reuse
        for layer in layers[ℓ*:]:
            use_current_kv(layer)  # Deep: conservative
    else:
        # Full refresh from ℓ* onward
        refresh_from_layer(ℓ*)
        
        # Record new anchor if stability detected
        if stability_detected(token):
            update_anchor_tokens(token)
```

### Step 4: Importance Eviction (MaskKV)

```
def evict_if_needed(cached_tokens, budget):
    if len(cached_tokens) <= budget:
        return cached_tokens
    
    # Compute importance scores
    for token in cached_tokens:
        if token.is_attention_sink:
            continue
        score = mask_query_attention(token)
    
    # Sort ascending, evict lowest
    sorted_tokens = sort_by_importance(cached_tokens)
    while len(cached_tokens) > budget:
        token = sorted_tokens.pop(0)
        if not token.is_anchor:
            evict(token)
    
    return cached_tokens
```

## Expected Benefits

- **Long-sequence optimized**: Three observations specifically target long sequences
- **Empirically grounded**: Elastic-Cache v2 has strong 45.1× speedup on long seq
- **Depth awareness**: Different strategies for shallow vs deep layers
- **Anchor preservation**: Never evict stable anchor tokens
- **Expected speedup**: 40-50× on long sequences

## Assumptions

- Elastic-Cache v2 thresholds transfer to LLaDA2.1
- Block boundaries align with LLaDA2.1 block size
- Anchor tokens can be reliably identified

## Risks and Unknowns

- **Complexity**: Four steps add coordination overhead
- **Threshold tuning**: ε_drift, ℓ*, N_stable all need calibration
- **Memory usage**: Highest cache utilization may exceed VRAM on Apple Silicon

## Apple Silicon / Metal Mapping

- **Threadgroup strategy**: Per-block state + layer budgets in threadgroup
- **Memory hierarchy**: 
  - L1: Anchor tokens (hot, always in cache)
  - L2: Stable block KV (warm)
  - L3/SLC: Recently tested blocks
  - DRAM: Historical blocks
- **Synchronization**: Pipeline with early exit on drift detection
- **Fusion opportunities**: 
  - Steps 2+3 can be fused (stability + drift test)
  - Step 4 eviction is separate priority queue
- **Likely bottlenecks**: Mask-query computation, depth-aware branching

## Validation Plan

1. **Observation ablation**: Remove each Elastic-Cache v2 observation
2. **Threshold sweep**: ε_drift, ℓ*, N_stable grid search
3. **Long-sequence benchmarks**: 2K, 4K, 8K token sequences
4. **Apple Silicon profiling**: Memory bandwidth vs compute tradeoff
5. **Quality metrics**: Perplexity, downstream task accuracy

## Related proposals

- [[mega-kernel-a-unified-cache-manager]] - Three parallel stages
- [[mega-kernel-b-hybrid-selection]] - Simpler AND logic
- [[mega-kernel-c-multi-stage-pipeline]] - Sequential staging (DPad → FreeCache → D2Cache)