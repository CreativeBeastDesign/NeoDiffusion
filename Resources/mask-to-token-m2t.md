# Mask-to-Token (M2T)

**Summary**: The standard discrete diffusion operation where a [MASK] token is replaced with a predicted token from the model.  
**Aliases**: unmasking, mask prediction, diffusion forward step  
**Status**: stable  
**Last updated**: 2026-04-16  
**Sources**:
- [[llada2-1-tech-report]]

---

## Definition

Mask-to-Token (M2T) is the fundamental operation in discrete diffusion language models where a masked position `[MASK]` is replaced with a token sampled from the model's predicted distribution `pθ(vi|xt)`. This is the core "drafting" mechanism that generates new content.

## Why it matters

M2T defines the basic generation dynamics of dLLMs. The confidence threshold `τ_mask` controls how conservative the model is when filling masked positions. Aggressive thresholds (low τ_mask) enable rapid parallel generation but risk introducing errors that propagate through subsequent steps (exposure bias).

## Mechanism

Given a current state `xt` with masked positions, the model computes probabilities for each vocabulary token at each masked location. The top candidate `vi = argmax v pθ(v|xt)` is selected. If `pθ(vi|xt) > τ_mask`, the mask is replaced with `vi`. This operation is applied in parallel across all masked positions meeting the threshold.

## Trade-offs

- **Low τ_mask**: High throughput, more parallel tokens filled per step, but higher error rate.
- **High τ_mask**: Conservative, fewer tokens filled per step, lower error rate, slower generation.
- Parallelism vs. fidelity: M2T's independent sampling can create local inconsistencies that compound.

## Apple Silicon implications

- M2T computations are highly parallel and map well to Metal's GPU execution model.
- Threshold checks and argmax operations can be fused into a single kernel launch.
- Memory bandwidth is critical: probability vectors for all masked positions must be accessible.
- Could leverage threadgroup shared memory for candidate token selection when `τ_mask` is uniform across batch.

## Related concepts
- [[token-to-token-t2t]]
- [[editable-state-evolution]]
- [[configurable-threshold-decoding]]
- [[multi-block-editing-mbe]]

## Open questions
- What is the optimal `τ_mask` for different hardware threadgroup sizes?
- Can branch divergence from threshold checks be minimized on Apple Silicon?