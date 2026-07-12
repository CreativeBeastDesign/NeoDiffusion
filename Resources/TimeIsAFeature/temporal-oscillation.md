# Temporal Oscillation (TSE)

**Summary**: The phenomenon where dLLMs frequently produce a correct answer at an intermediate denoising step, only to overwrite it with an incorrect one by the final step — evidenced by a persistent, large gap between final Pass@1 and "ever-pass" rate across multiple benchmarks and models.
**Aliases**: answer overwriting, intermediate-correctness loss
**Status**: draft
**Last updated**: 2026-07-03
**Sources**:
- [[time-is-a-feature]]

---

## Definition

Temporal oscillation is the core phenomenon the TSE paper identifies: standard dLLM decoding relies solely on the sequence predicted at the final denoising step, discarding all intermediate predictions — but a significant fraction of questions are answered *correctly* at some intermediate step and then *incorrectly* revised by the final step. This is formalized via the gap between final Pass@1 (accuracy of the last step's output) and ever-pass rate EverPass@1|t = E_i[max_{k≤t} e_i,k] (the fraction of questions correct at *any* step up to t) (**sourced**, [[time-is-a-feature]]). E.g. on GSM8K length 128 with LLaDA-8B-Instruct: 68.5% final Pass@1 vs. 80.5% ever-pass rate, a 12.0-point gap (**sourced**, [[time-is-a-feature]]).

## Why it matters

This is a fundamentally different kind of finding than most of this wiki's content: nearly every other source here is about *inference speed* (fewer steps, less compute, less memory) under an implicit assumption that the final-step output is the correct target to converge toward faster. Temporal oscillation challenges that assumption directly — it shows the final step is not even reliably the *most accurate* step, meaning standard decoding is leaving accuracy on the table for reasons unrelated to speed. This motivates an entirely different axis of dLLM improvement: exploiting intermediate predictions rather than merely computing them faster.

## Mechanism

The gap is measured across four reasoning benchmarks (GSM8K, MATH500, SVAMP, Countdown) and two models (LLaDA-8B-Instruct, LLaDA-1.5), and persists across all of them (**sourced**, [[time-is-a-feature]]). The paper further breaks questions into three groups — Finally-Correct, Always-Incorrect, and Intermediate-Correct (correct at some step, wrong at the end) — and finds Intermediate-Correct questions on GSM8K begin with *lower* token-level entropy than Always-Incorrect ones, suggesting the model is initially confident but the confidence doesn't hold (**sourced**, [[time-is-a-feature]]). SVAMP shows a distinctive additional pattern: Pass@1 accuracy *declines* between steps 3-20 before recovering after step 20 — a fluctuation trajectory, not just a single overwrite event (**sourced**, [[time-is-a-feature]]).

## Trade-offs

- **A diagnostic finding, not by itself a fix**: temporal oscillation is the *problem statement* the paper's two methods ([[temporal-self-consistency-voting]] and [[temporal-consistency-reinforcement]]) address — the phenomenon itself doesn't prescribe a unique solution, and the paper's own results show its methods reduce but do not eliminate oscillation (ever-pass rate remains above final-pass rate even after RFT).
- **Benchmark-dependent severity**: the gap's size and even its qualitative pattern (SVAMP's dip-then-recover vs. GSM8K/Countdown's steadier trajectories) varies by dataset, meaning "how much oscillation to expect" isn't a fixed property of dLLMs in general but interacts with task difficulty and structure.

## Apple Silicon implications

- **Inferred**: this finding has no direct hardware/kernel implication by itself — its downstream methods ([[temporal-self-consistency-voting]]) do, but the phenomenon itself is a modeling/algorithmic observation, not a systems one. Included here as the load-bearing premise for the two methods that do carry systems implications.

## Related concepts
- [[temporal-semantic-entropy]] (the metric developed to quantify this phenomenon's severity per-trajectory)
- [[temporal-self-consistency-voting]] (the test-time method exploiting this phenomenon)
- [[temporal-consistency-reinforcement]] (the training-time method addressing this phenomenon)
- [[credit-decoding]] (a different, pre-existing wiki concept that also accumulates signal across steps, but to influence *when tokens commit* during generation rather than to diagnose or correct post-hoc answer instability)

## Open questions
- Is temporal oscillation specific to reasoning/math benchmarks (all four evaluated datasets are math-reasoning tasks with parseable numeric answers), or does an analogous phenomenon occur in open-ended generation where "correctness" isn't as cleanly defined?
- Does temporal oscillation's severity correlate with any property measurable *without* running the full generation (e.g., could it be predicted from early-step entropy alone, enabling a cheaper detection signal)?
- How does temporal oscillation interact with aggressive parallel-decoding/caching methods elsewhere in this wiki that reduce the number of denoising steps — does fewer steps mean less opportunity for oscillation (fewer chances to revise), or does it concentrate the same instability into fewer, larger jumps?
