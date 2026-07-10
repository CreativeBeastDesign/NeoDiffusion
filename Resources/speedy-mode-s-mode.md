# Speedy Mode (S Mode)

**Summary**: Operating mode that uses aggressively low τ_mask to maximize parallel token generation throughput, relying on T2T editing for correction.  
**Aliases**: fast mode, high-throughput mode  
**Status**: stable  
**Last updated**: 2026-04-16  
**Sources**:
- [[llada2-1-tech-report]]

---

## Definition

Speedy Mode (S Mode) is a configuration of LLaDA2.1's decoding where the mask-to-token confidence threshold τ_mask is set aggressively low. This causes the model to fill many masked positions in parallel at each step, achieving high tokens-per-forward (TPF) and thus high throughput (TPS). The resulting draft may contain errors, but these are expected to be corrected by subsequent T2T editing passes.

## Why it matters

S Mode demonstrates that the speed-accuracy tradeoff in dLLMs can be broken: by decoupling drafting speed from editing corrections, overall throughput can be dramatically increased with only modest quality loss. The paper reports up to 892 TPS on HumanEval+ (100B model) with minimal score degradation.

## Mechanism

- τ_mask set very low (e.g., values not explicitly given but "aggressively lowered")
- Many tokens unmasked per step → fewer diffusion steps needed
- T2T editing with τ_edit at a level sufficient to catch and correct most errors
- May optionally use Multi-Block Editing (MBE) for cross-block refinement

## Trade-offs

- **Speed**: Dramatic TPS gains (2×–3× over Q Mode in many benchmarks)
- **Quality**: Small but noticeable drops in some benchmarks, particularly in non-structured domains
- **Stability**: Lower τ_mask can produce "rough drafts" with stuttering artifacts; editing mitigates but doesn't always eliminate
- **Domain sensitivity**: Best suited for code/math where structure aids correction; general chat degrades more

## Apple Silicon implications

- High parallelism maps well to Apple Silicon's GPU; potential for very high occupancy
- Needs efficient T2T implementation to keep latency low despite correction overhead
- Quantization (per-block FP8) helps maintain throughput while reducing memory bandwidth
- Could be the default mode for code generation tasks on Metal

## Related concepts
- [[quality-mode-q-mode]]
- [[configurable-threshold-decoding]]
- [[exposure-bias-in-dllms]]

## Open questions
- What is the minimal τ_edit needed to maintain quality in S Mode on Apple Silicon hardware?
- Does S Mode benefit more from MBE on Metal than on other platforms?