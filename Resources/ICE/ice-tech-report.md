# ICE: In-Place Chain-of-Thought Prompting with Early Exit

**Type**: paper  
**Canonical source**: 01-Inbox/ICE.pdf  
**Date**: 2025-10-11 (arXiv)  
**Relevance**: Introduces in-place CoT prompting and confidence-aware early exit for dLLMs; evaluated on LLaDA models; significant speedups and accuracy improvements on reasoning tasks. Highly applicable to LLADA2.1's bidirectional architecture.  
**Status**: processed  
**Last updated**: 2026-04-16

## Summary

ICE (In-Place Chain-of-Thought Prompting with Early Exit) reimagines chain-of-thought reasoning for diffusion LLMs by embedding structured reasoning templates directly within masked token positions during iterative refinement, rather than as prefix prompts. The framework leverages dLLMs' bidirectional attention and concurrent answer accessibility to implement a two-phase decoding strategy: a reasoning phase that progressively fills the thinking section while monitoring confidence of the (still-masked) answer tokens, followed by an early exit when answer confidence exceeds a threshold, triggering a single-step parallel decoding of the entire answer. Tested on LLaDA-8B-Instruct and LLaDA-1.5, ICE achieves up to 17.29% absolute accuracy improvement with 4.12× speedup on GSM8K, and up to 276.67× acceleration on MMLU while maintaining competitive accuracy. ICE is compatible with existing acceleration techniques like dLLM-Cache.

## Key claims

- dLLMs' bidirectional attention and iterative refinement enable fundamentally different prompting strategies compared to autoregressive models' prefix-only paradigm.
- ICE's in-place prompting integrates reasoning steps directly into the generation process, transforming reasoning from external preprocessing into an integral component.
- dLLMs exhibit a key empirical pattern: confidence in answer tokens converges rapidly and stabilizes early, while the reasoning section continues to refine; this enables efficient early exit without quality loss.
- The two-phase approach (reasoning → answer generation) with confidence monitoring achieves simultaneous accuracy gains and inference speedups.
- ICE works with existing dLLM acceleration methods (e.g., dLLM-Cache) for cumulative benefits.

## Mechanisms

### Prefix-only vs. In-place prompting

- **Autoregressive (AR) models**: Must generate reasoning tokens sequentially before answer tokens become accessible; CoT is provided as a prefix before the actual answer.
- **Diffusion LLMs (dLLMs)**: Use bidirectional attention over the full sequence; tokens are iteratively refined from a fully masked state. This enables concurrent answer accessibility: answer tokens can be partially revealed and refined alongside reasoning steps.
- ICE exploits this by structuring `ygen = (ythinking, yanswer)` and embedding reasoning step templates `T1, T2, ..., Nt` directly within `ythinking`. The answer section remains fully masked during the reasoning phase.

### ICE algorithm

**Initialization**:
```
y(N) = (yprompt, T1, T2, ..., TNt, [MASK]... [MASK])
```
The thinking section is pre-filled with step templates; the answer section is completely masked.

**Reasoning Phase** (executed for `k = N, N-1, ...`):
1. At step `k`, the model estimates the clean sequence: `ˆy(k)0 = argmax_v fθ(y0 = v | y(k))`.
2. For each position, compute confidence: `confidence(k)i = max_v fθ(y0,i = v | y(k))`.
3. Monitor average confidence of answer tokens: `avg_conf(k)answer = (1/Lanswer) Σ_{i∈answer} confidence(k)i`.
4. If `avg_conf(k)answer ≥ τ` (threshold), **early exit**: transition to answer generation phase.
5. Otherwise, update only the thinking section via selective unmasking: `y(k-1) = S_thinking(ˆy(k)0, y(k), k)`. Answer tokens remain masked.

**Answer Generation Phase**:
- Perform a single-step parallel decoding of all answer tokens: `ˆy_final = argmax_v fθ(yanswer = v | y_current)`.
- Return final sequence.

### Confidence dynamics

Empirically, the answer section's confidence rises quickly and stabilizes within a small number of steps, while the thinking section continues to change. This indicates the model internally determines the correct answer early but continues to refine the reasoning trace. Early exit capitalizes on this by stopping after answer confidence converges, avoiding unnecessary computations spent polishing reasoning.

### Two operational modes

- **ICE-SP** (Speed-Prioritized): Uses lower confidence threshold (e.g., τ=0.8) for more aggressive early exit, achieving maximum speedup with minimal accuracy loss.
- **ICE-PP** (Performance-Prioritized): Uses higher threshold (e.g., τ=0.9) to preserve accuracy, still gaining some speed.

### Compatibility

ICE can be combined with dLLM-Cache (adaptive caching) for additional speedups. The structured reasoning approach remains robust with caching.

## Trade-offs

- **Reasoning step count `Nt`**: More steps (e.g., 4 vs 3) can improve accuracy but increase reasoning phase duration; there is a sweet spot per task domain (e.g., GSM8K optimal at Nt=3, MATH at Nt=4). Too many steps (Nt>6) degrades performance.
- **Confidence threshold τ**: Lower τ → earlier exit → higher speedup but risk of premature termination; higher τ → more reasoning steps → better accuracy but slower. Task-adaptive selection needed.
- **Mask token allocation** across thinking steps: Uniform, front-heavy, or back-heavy strategies affect performance; back-heavy and front-heavy generally outperform uniform.
- **Model choice**: Works with both base and preference-optimized dLLMs; gains are larger for non-preference-aligned models (e.g., LLaDA-8B-Instruct vs LLaDA-1.5).

## Hardware implications

- Early exit reduces total number of denoising steps dramatically, especially on knowledge-intensive tasks (MMLU, GPQA) where answer confidence converges extremely fast (276.67× speedup on MMLU).
- The reasoning phase still requires iterative refinement of the thinking section; this is where most steps are spent.
- dLLM-Cache can further accelerate by caching stable prompt computations; combination yields multiplicative speedups.
- The algorithm imposes negligible overhead: only confidence monitoring (max probability per token) and averaging.
- Memory requirements unchanged from base dLLM; no extra parameters needed.
- The single-step answer generation is a parallel decoding pass over the answer positions; this is already how standard dLLMs operate at the final step.

## Relevance to Apple Silicon

- ICE directly applies to LLADA2.1, which is a masked diffusion LLM with bidirectional attention. The paper's evaluation on LLaDA models confirms applicability.
- The dramatic speedups (particularly on knowledge tasks) could make on-device reasoning with LLADA2.1 feasible within reasonable latency.
- Early exit based on simple confidence thresholds is hardware-friendly; no complex distribution comparisons (like KL in DualDiffusion) are needed.
- Could be combined with Apple-specific optimizations: per-block FP8 quantization, radix caching, and model scheduling (light/heavy models) to accelerate the reasoning phase.
- Open questions:
  - How does confidence threshold τ interact with Apple Silicon's numerical precision (e.g., FP16 vs FP32)? Should thresholds be recalibrated?
  - Can the early exit decision be computed efficiently on the Neural Engine or GPU? The monitoring is lightweight.
  - Does the reasoning phase benefit from hierarchical blocking (MBE) or vicinity KV refresh? Possibly, as the thinking section undergoes multiple refinements.
  - What is the optimal `Nt` for typical on-device tasks (short reasoning, code generation)? Likely lower than paper's 3-4.
  - Could the thinking templates be optimized for Apple's hardware or specific use cases (e.g., shorter templates for low-latency)?
  - Integration with Adaptive Denoising / Model Scheduling: Could use a heavy model for reasoning phase and light model for answer generation? Might compromise quality.

## Extracted concepts

- In-place chain-of-thought prompting (embedding reasoning templates directly into masked generation space)
- Two-phase decoding (reasoning phase + answer generation phase)
- Confidence-aware early exit mechanism (monitor answer token confidence, threshold-based transition)
- Concurrent answer accessibility (dLLM property enabling early answer visibility)
- Thinking/answer section structuring (semantic segmentation of the generation sequence)
- Token-level confidence dynamics (answer confidence converges early; reasoning continues to refine)
- Operational modes: speed-prioritized (SP) vs performance-prioritized (PP)
- Compatibility with dLLM-Cache and other acceleration methods

## Open questions

- What is the optimal confidence threshold τ for different task classes on Apple Silicon? The paper uses 0.8 and 0.9; is there a sweet spot for on-device use?
- How does ICE interact with other dLLM optimizations like Iteration Smoothing, Credit Decoding, or hierarchical decoding? Could multiplicative gains be achieved?
- Could the thinking templates be learned or adapted rather than hand-crafted? The paper uses fixed "Step 1:", "Step 2:" etc.; maybe more sophisticated templates improve quality.
- Does ICE work for open-ended generation (not just question answering)? The paper focuses on reasoning benchmarks; other domains may have different confidence dynamics.
- How does the early exit decision correlate with eventual correctness? Could false positives (early exit with wrong answer) be detected and corrected?
- What is the latency/accuracy trade-off when combined with Model Scheduling? For example, use light model during reasoning phase and heavy model for answer generation (or vice versa).
- Can the reasoning phase be further accelerated by parallelizing across thinking steps? Currently, thinking tokens are refined iteratively; maybe some can be unmasked in parallel within the reasoning phase.
- How does ICE behave with very short reasoning templates (e.g., Nt=1 or 2)? Simpler tasks might need minimal reasoning structure.
- Does the method extend to multimodal dLLMs (e.g., LLADA-V)? The paper mentions lower `γ` for LLADA-V; perhaps confidence dynamics differ.
- What is the memory bandwidth impact of monitoring confidence each step? Likely negligible, but profiling on Apple Silicon would confirm.
- Could the early exit mechanism be implemented as a learned policy (e.g., reinforcement learning) rather than a fixed threshold?
- How does ICE compare to speculative decoding approaches like DualDiffusion? ICE reduces steps via early exit; DualDiffusion reduces steps by using a fast drafter. Could they be combined (use drafter for reasoning phase, then verifier for answer)?
- Does ICE's two-phase approach affect exposure bias? The answer generation is a single step from the final reasoning-phase state; this could differ from standard iterative refinement that alternates mask/unmask across all tokens.

## Related pages

- [[mask-to-token-m2t]] (basic operation used repeatedly)
- [[editable-state-evolution]] (the sequence evolution process)
- [[configurable-threshold-decoding]] (ICE uses confidence threshold τ)
- [[speedy-mode-s-mode]] and [[quality-mode-q-mode]] (ICE-SP and ICE-PP are analogous speed-accuracy trade-off modes)
- [[exposure-bias-in-dllms]] (potential interaction)
- [[multi-block-editing-mbe]] (could accelerate reasoning phase)
- [[iteration-smoothing]] (could smooth confidence monitoring)
- [[credit-decoding]] (another confidence-based technique)
- [[entropy-sum-decoding]] (different threshold-based acceleration; compare)
- [[soft-parallel-decoding]] (DMax uses parallel decoding; ICE uses early exit then parallel answer)
- [[speculative-decoding-dllm]] (DualDiffusion; could combine)
- [[elastic-cache]] (compatible)
- [[model-scheduling]] (potential integration)
- [[dllm-cache]] (not yet a concept page, but ICE is compatible)
- [[llada2-1-tech-report]] (base model used in evaluation)
