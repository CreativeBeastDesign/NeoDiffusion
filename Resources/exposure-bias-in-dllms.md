# Exposure Bias in dLLMs

**Summary**: The phenomenon where errors in model-generated tokens compound due to conditioning on imperfect predictions, particularly severe in parallel decoding.  
**Aliases**: exposure bias, error accumulation  
**Status**: stable  
**Last updated**: 2026-04-16  
**Sources**:
- [[llada2-1-tech-report]]

---

## Definition

Exposure bias occurs when a model's predictions at later steps are conditioned on its own earlier predictions, which may contain errors. In autoregressive models, this is somewhat mitigated by the ability to self-correct via chain-of-thought. In discrete diffusion language models, parallel decoding amplifies this because many tokens are generated simultaneously, and errors in one position can influence others through attention, creating cascading inconsistencies.

## Why it matters

LLaDA2.1's core innovation (editable state evolution) is designed specifically to combat exposure bias. By allowing tokens to be revised after initial generation, the model can correct early mistakes before they propagate. This is essential for maintaining quality when using aggressive thresholds in S Mode.

## Mechanism

In standard M2T-only diffusion, once a token is sampled at step `t`, it becomes fixed input to future steps. If the token is incorrect, subsequent predictions are conditioned on this error, potentially leading to:
- Reduced confidence in subsequent predictions
- Conservative behavior (slower generation)
- Local inconsistencies that become global

Editing (T2T) breaks this chain by allowing erroneous tokens to be replaced based on later context.

## Trade-offs

- **Without editing**: Exposure bias forces conservative τ_mask to avoid errors → slower generation.
- **With editing**: Can accept lower-confidence initial samples, relying on later correction.
- **Cost**: Editing requires recomputation of attention for edited tokens if they appear in context of other positions.
- **Incomplete correction**: Not all errors may be caught, especially if τ_edit is too high.

## Apple Silicon implications

- The need to recompute attention for edited tokens suggests careful kernel design to avoid full re-execution.
- Could exploit Metal's parallelism to evaluate edits in batch and apply them selectively.
- Memory bandwidth: edited tokens must be written back and potentially re-read by dependent positions.
- Unified memory may simplify the data movement but careful synchronization is needed.

## Related concepts
- [[editable-state-evolution]]
- [[token-to-token-t2t]]
- [[multi-block-editing-mbe]]

## Open questions
- How does the frequency of edits (due to exposure bias) vary across different task domains on Apple Silicon?
- Can we predict which tokens are likely to be edited to prefetch or pre-compute?