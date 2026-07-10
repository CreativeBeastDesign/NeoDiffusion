# Suffix Dropout in dLLMs

**Summary**: A technique that deterministically drops distant suffix tokens from attention computation based on distance, reducing computation on redundant low-entropy tokens. Part of DPad's approach.  
**Aliases**: distance-decay dropout, local attention, scratchpad pruning  
**Status**: draft  
**Last updated**: 2026-04-16  
**Sources**:
- [[dpad]]

---

## Definition

Suffix dropout is a deterministic attention mask modification that drops distant suffix tokens based on their distance from the current position. Unlike random dropout, suffix dropout is:
- **Deterministic**: Same tokens dropped every time given same positions.
- **Distance-based**: Tokens farther from current position have higher dropout probability.
- **Suffix-specific**: Only applies to future (suffix) tokens, not prefix.

## Why it matters

In dLLMs, suffix tokens (future positions) act as a "scratchpad" collecting signals from prefix. However:
- Most suffix tokens are low-entropy and redundant.
- They contribute diminishing semantic value with distance.
- Computing attention with all suffix tokens is wasteful.

Suffix dropout eliminates this waste while preserving fidelity.

## Mechanism

### Distance-Decay Function

For position i attending to position j (j > i):
```
keep_prob = f(distance = j - i)
```

Common decay functions:
- **Linear**: keep_prob = max(0, 1 - distance/max_distance)
- **Exponential**: keep_prob = exp(-distance * rate)
- **Step**: keep_prob = 1 if distance < window_size else 0

### Sliding Window Variant

DPad uses a fixed-length sliding window:
- Only tokens within window participate in attention.
- Tokens outside window are dropped.
- Simple to implement and tune.

## Trade-offs

- **Window size**: Larger = more accuracy but less speedup.
- **Decay function**: Different functions give different trade-offs.
- **Task dependence**: Some tasks may need larger windows.

## Apple Silicon implications

- Deterministic mask can be precomputed.
- Simple modification to attention kernel.
- Compatible with prefix caching.
- Could be combined with other caching methods.

## Related concepts

- [[local-attention-dllm]] (broader category)
- [[sliding-window-attention]] (specific variant)
- [[distance-decay]] (the decay mechanism)
- [[scratchpad-redundancy]] (what it addresses)

## Open questions

- What is optimal window/decay for Apple Silicon?
- Can decay be learned or must it be hand-tuned?
- Does it work for all task types?

## Related pages

- [[dpad]] (source)
- [[maskkv]] (related attention selection)