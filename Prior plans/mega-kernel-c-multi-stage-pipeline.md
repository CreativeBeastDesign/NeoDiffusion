# Mega-Kernel C: Multi-Stage Inference Pipeline

**Goal**: Progressive narrowing from coarse attention restriction to fine-grained cache management  
**Status**: idea  
**Last updated**: 2026-04-16  
**Depends on concepts**:  
- [[suffix-dropout]] (DPad)  
- [[sliding-window-attention]] (DPad)  
- [[block-freezing]] (FreeCache)  
- [[layer-wise-caching]] (D2Cache)  
- [[adaptive-eviction]] (D2Cache)  
- [[attention-decay-tracking]] (D2Cache)  

**Grounded in sources**:  
- [[dpad]] - sliding window + distance-decay dropout  
- [[freecache]] - block-based temporal freezing  
- [[d2cache-dual-adaptive-cache]] - dual adaptive caching  
- [[localleap]] - localized processing  

---

## Problem

Single-stage approaches have limitations:
- DPad reduces attention scope but doesn't manage cache lifecycle
- FreeCache freezes blocks but doesn't handle fine-grained token eviction
- D2Cache manages eviction but doesn't reduce attention computation

A **pipeline** approach can combine the strengths of each.

## Proposed Design

```
┌─────────────────────────────────────────────────────────────────┐
│            MULTI-STAGE INFERENCE PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  STAGE 1: ATTENTION RESTRICTION (DPad)                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Apply sliding window + distance-decay dropout           │   │
│  │  • Window size: W tokens                                 │   │
│  │  • Distance-decay: probability decays with distance      │   │
│  │  • Output: Restricted attention scope                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                     │
│                            ▼                                     │
│  STAGE 2: BLOCK FREEZING (FreeCache)                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Freeze blocks stable for N consecutive steps            │   │
│  │  • Frozen blocks: No KV computation                      │   │
│  │  • Active blocks: Full computation                       │   │
│  │  • Output: Reduced token set for processing              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                     │
│                            ▼                                     │
│  STAGE 3: FINE-GRAINED MANAGEMENT (D2Cache)                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Token-level: Top-K importance selection                  │   │
│  │  Layer-level: Per-layer cache budget allocation           │   │
│  │  Eviction: Attention decay tracking                       │   │
│  │  Output: Efficient cache utilization                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Stage 1: Attention Restriction (DPad)

```
def apply_attention_restriction(attention_scores):
    window_size = 64  # tokens
    for i in range(seq_len):
        for j in range(seq_len):
            distance = j - i
            if distance > window_size:
                attention_scores[i,j] = 0  # Drop
            else:
                # Distance-decay weighting
                attention_scores[i,j] *= exp(-distance * decay_rate)
```

Benefits:
- Reduces attention FLOPs before expensive computation
- Simple mask modification
- Compatible with prefix caching (prefix always attended)

### Stage 2: Block Freezing (FreeCache)

```
def freeze_stable_blocks(blocks, step):
    for block in blocks:
        if block.unmasked_for_consecutive_steps(step, n=3):
            block.frozen = True
            # Skip all KV computation for this block
    
    # Active blocks continue processing
    return [b for b in blocks if not b.frozen]
```

Benefits:
- Eliminates computation for frozen blocks
- Block granularity reduces per-token overhead
- Good fit for LLaDA-style block processing

### Stage 3: Fine-Grained Cache Management (D2Cache)

```
def manage_cache(active_tokens, layer_idx):
    # Token-level: Keep top-K important
    top_k = select_top_k_importance(active_tokens, k=cache_size)
    
    # Layer-level: Adjust budget by depth
    if layer_idx < ℓ_boundary:
        budget = large_cache_budget
    else:
        budget = small_cache_budget
    
    # Eviction: Track attention decay
    evict_low_decay_tokens(top_k, budget)
    
    return cached_tokens
```

Benefits:
- Maximizes cache efficiency within budget
- Layer-aware (deep layers get smaller budgets)
- Automatic eviction prevents overflow

## Expected Benefits

- **Progressive narrowing**: Coarse → fine filtering
- **Modular**: Each stage can be tuned independently
- **Comprehensive**: Addresses attention scope, temporal stability, cache eviction
- **Good for short sequences**: Stage 1 (DPad) shines on shorter context
- **Expected speedup**: 20-30× (best for short sequences)

## Assumptions

- DPad window size works for target sequence lengths
- Block freezing is compatible with LLaDA2.1 architecture
- D2Cache parameters transfer across tasks

## Risks and Unknowns

- **Stage ordering**: Is this the optimal order?
- **Overhead accumulation**: Three stages may add latency
- **Complexity**: Harder to debug and tune

## Apple Silicon / Metal Mapping

- **Threadgroup strategy**: 
  - Stage 1: Mask buffer in constant memory
  - Stage 2: Block state in threadgroup memory
  - Stage 3: Priority queue for eviction
- **Memory hierarchy**: Hierarchical caching mirrors Metal hierarchy
- **Synchronization**: Pipeline barriers between stages
- **Fusion opportunities**: 
  - Stage 1+2 can be fused (mask + freeze check)
  - Stage 3 eviction is separate kernel
- **Likely bottlenecks**: Stage 3 eviction sorting

## Validation Plan

1. **Stage ablation**: Remove each stage, measure incremental impact
2. **Order permutation**: Test Stage 2↔3 swap
3. **Parameter tuning**: Window size, freeze threshold, cache budget
4. **Apple Silicon profiling**: Pipeline vs fused kernel comparison

## Related proposals

- [[mega-kernel-a-unified-cache-manager]] - Unified cache manager (different architecture)
- [[mega-kernel-b-hybrid-selection]] - Simpler parallel approach
- [[mega-kernel-d-depth-aware-elastic]] - Elastic-Cache focused