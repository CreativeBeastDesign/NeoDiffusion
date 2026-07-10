# Per-Token Early Stopping

**Summary**: A decoding strategy for masked diffusion language models where individual token predictions are finalized as soon as they exhibit sufficient confidence, rather than waiting for a fixed global step count.  
**Aliases**: token-level early exit, adaptive per-position stopping, confidence-gated unmasking  
**Status**: draft  
**Last updated**: 2026-04-16  
**Sources**:
- [[just-on-time-jot]]

---

## Definition

Per-token early stopping is a decoding strategy where each token position is evaluated independently for "convergence" at each diffusion step. Unlike global stopping criteria (which finalize all remaining tokens simultaneously at some step T), per-token stopping allows different positions to exit at different steps based on local confidence signals.

The key components are:
1. **Confidence metric**: A scalar measure of prediction certainty at each position
2. **Threshold**: A cutoff that determines when a prediction is "confident enough"
3. **Stopping decision**: If confidence > threshold, finalize token and remove from further refinement

This is orthogonal to the transfer schedule (which determines which tokens to reveal each step) and can be combined with any masking strategy.

## Why It Matters

- **Reduces unnecessary computation**: Many tokens stabilize well before the final step; continuing to refine them wastes GPU cycles
- **Focuses computation**: Resources are concentrated on tokens that genuinely need refinement
- **Training-free**: No fine-tuning required; uses only existing model predictions
- **Architecture-agnostic**: Applies to any masked diffusion model with a confidence signal

Prior work (Prophet) showed that "early answer convergence" occurs: many positions can be correctly predicted with only half the prescribed steps. Per-token stopping exploits this phenomenon at a finer granularity.

## Mechanism

### Confidence Metrics

**Top-2 Ratio** (used by Jot):
```
ri = p_max / (p_second + ε)
```
- Ratio close to 1: uncertain between alternatives
- Ratio >> 1: strong commitment to top candidate
- Invariant to temperature scaling (uses unscaled softmax)

**Entropy** (used by Entropy-Sum Decoding):
```
H(p) = -Σ_k p_k log p_k
```
- Measures total uncertainty in distribution
- Requires full distribution; more expensive than top-2

**KL Divergence** (used by KLASS):
```
KL(p_t || p_{t-1})
```
- Measures change between consecutive steps
- Large change = unstable; small change = converged

**Top-1 Probability** (simplest):
```
p_max
```
- Just the argmax probability
- Cheaper but less informative than ratio

### Threshold Strategies

**Fixed Threshold**:
```
if confidence > τ: finalize
```
- Simple but may not adapt to position-specific difficulty

**Spatial Modulation** (Jot):
```
τi = τmax - (τmax - τmin) · φi
```
where φi depends on proximity to already-unmasked tokens. Positions near context get lower (more lenient) thresholds.

**Adaptive Schedules**:
- Vary threshold over diffusion steps (stricter early, more lenient later)
- Learn thresholds via reinforcement learning or supervised learning

### Interaction with Transfer Schedule

Per-token stopping operates alongside the transfer schedule:
1. At each step, compute confidence for all masked positions
2. Finalize positions exceeding threshold
3. Apply transfer schedule to remaining masked positions
4. If no positions meet threshold, proceed as normal

This ensures graceful degradation: when predictions are uncertain, behavior matches standard decoding.

## Trade-offs

| Aspect | Pro | Con |
|--------|-----|-----|
| Fixed threshold | Simple, predictable | May exit too early for hard tokens or too late for easy tokens |
| Spatial modulation | Adapts to local context | Requires computing spatial weights |
| Aggressive thresholds | Higher speedup | Risk of quality degradation on hard tasks |
| Conservative thresholds | Better quality | Lower speedup |

### Quality vs Speed Trade-off

- **Aggressive (low τ)**: Up to 19× speedup but -10%+ quality loss on reasoning tasks
- **Balanced (τ~90)**: 5-7× speedup with <3% quality loss
- **Conservative (high τ)**: May exceed baseline quality but minimal speedup

The optimal threshold depends on:
- Task difficulty (math > commonsense)
- Model architecture (Dream vs LLaDA have different sensitivity)
- Acceptable quality degradation

## Apple Silicon Implications

### Memory Bandwidth
- Reduced step count directly reduces memory traffic
- Spatial weight computation is O(L·D) per step; negligible compared to forward pass
- Confidence metric can be fused with mask predictor

### Cache Behavior
- Fewer steps = fewer cache evictions
- Finalized tokens don't participate in attention → reduced attention compute
- Spatial weights small enough to fit in L1 cache (D≤16)

### Threadgroup Strategy
- Per-token independence enables fine-grained parallelism
- Early exit decisions are independent → minimal thread divergence
- Could use warp-level reduction for top-2 extraction

### Fusion Opportunities
- Confidence computation + threshold check can be fused into single kernel
- Spatial weights precomputed once per step; shared across positions
- Early exit predicate can short-circuit computation

### Potential Bottlenecks
- Memory latency for spatial weight lookup
- Thread divergence if finalized vs continuing paths diverge
- These are likely minor compared to forward pass savings

## Related Concepts

- [[confidence-based-decoding]] (umbrella term for confidence-gated strategies)
- [[entropy-sum-decoding]] (provides theoretical foundation for confidence-based control; discusses entropy metrics)
- [[in-place-chain-of-thought]] (confidence-based early exit for reasoning; shares threshold mechanism)
- [[configurable-threshold-decoding]] (LLaDA2.1's τ_mask, τ_edit; similar threshold concept)
- [[speedy-mode-s-mode]] (aggressive thresholds for speed; low τ = early exit)
- [[quality-mode-q-mode]] (conservative thresholds; high τ = delayed exit)
- [[iteration-smoothing]] (both improve decoding efficiency; could compound with early exit)
- [[credit-decoding]] (both track token stability across steps)
- [[hybrid-dlm-ar-decoding]] (CoDiLA's confidence-based verification; analogous to per-token acceptance)
- [[self-speculative-decoding-dlm]] (block-size-1 verification uses acceptance probability; similar confidence gating)

## Implementation Considerations

### For Metal Compute Shaders

1. **Confidence extraction**:
   - Use `simd_reduce` for max/second-max
   - Compute ratio in floating point
   - Fuse with existing softmax if possible

2. **Spatial weights**:
   - Precompute kernel (geometric decay) once per step
   - Use prefix sum or cumulative approach for efficiency
   - Store in threadgroup memory (fast access)

3. **Early exit logic**:
   - Predicate array marking finalized positions
   - Use for masking attention and subsequent forward passes
   - Could use `threadgroup_barrier` to synchronize

### Open Questions

- What confidence metric is best for Apple Silicon (top-2 ratio vs entropy vs top-1)?
  - Partial answer: [[entropy-sum-decoding]] analyzes entropy as confidence metric; provides theoretical bounds on batch size from cumulative entropy
- Can spatial modulation be replaced with a cheaper proxy (e.g., position in sequence)?
  - Related: [[in-place-chain-of-thought]] uses fixed threshold with no spatial modulation; shows simple threshold (τ=0.8/0.9) works well
- How does per-token stopping interact with block-level operations (block MDM)?
  - Related: [[multi-block-editing-mbe]] (block-level operations); [[soft-parallel-decoding]] (iterative refinement at block level)
- Can the method be extended to non-masked diffusion models (SEDD, D3PM)?
  - Related: [[continuous-guided-mdm]] (CRoCoDiL works with continuous latent; could use similar confidence signals)
- What is the minimum viable threshold τ for LLaDA2.1 on Apple Silicon?
  - Related: [[entropy-sum-decoding]] shows theoretical minimum based on ε-accuracy; [[in-place-chain-of-thought]] finds τ=0.8/0.9 works on LLaDA
- Does per-token stopping affect text coherence in long generations?
  - Partial answer: [[hybrid-dlm-ar-decoding]] addresses coherence via AR verification; could be combined with early stopping
- Can early exit decisions be made batched across positions for efficiency?
  - Related: [[iteration-smoothing]] and [[credit-decoding]] show how to batch token-level decisions efficiently in Metal

## Historical Context

- **Prophet (2025)**: First documented "early answer convergence"; uses global confidence gap
- **KLASS (2025)**: KL-adaptive stability sampling for batch unmasking
- **Jot (2026)**: Per-token early stopping with spatial modulation; training-free
- **ICE (2025)**: Confidence-based early exit for chain-of-thought; task-specific

## Comparison to Related Methods

| Method | Granularity | Threshold Type | Training-free |
|--------|-------------|----------------|---------------|
| Prophet | Global | Top-2 gap | Yes |
| KLASS | Batch | KL divergence | Yes |
| Jot | Per-token | Spatial-modulated ratio | Yes |
| ICE | Per-token | Fixed confidence | Yes |
| Entropy-Sum | Batch | Cumulative entropy | Yes |

## References

- [[just-on-time-jot]] (primary source)
- [[in-place-chain-of-thought]] (related; uses similar confidence signals)
- Prophet (arXiv:2508.19982)
- KLASS (arXiv:2511.05664)