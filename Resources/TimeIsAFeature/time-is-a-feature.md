# Time Is a Feature (TSE)

**Type**: paper
**Canonical source**: Time Is a Feature: Exploiting Temporal Dynamics in Diffusion Language Models (Wen Wang, Bozhen Fang, et al., Zhejiang University / Ant Group / Stanford University), arXiv:2508.09138v3, cs.CL, Oct 2025
**Date**: 2025-10 (v3)
**Relevance**: Distinct from this wiki's existing caching/decoding-speed literature — this paper is about *accuracy*, not throughput, by exploiting information discarded by standard final-step-only decoding. Introduces Temporal Semantic Entropy (TSE), a reusable uncertainty metric with both test-time (decoding) and training-time (RL reward) applications.
**Status**: processed
**Last updated**: 2026-07-03

## Summary

The paper identifies **temporal oscillation**: dLLMs frequently produce a correct answer at an intermediate denoising step, only to overwrite it with an incorrect one by the final step — evidenced by a persistent, large gap between final-step Pass@1 and "ever-pass" rate (correctness at *any* intermediate step) across GSM8K, MATH500, SVAMP, and Countdown on LLaDA-8B-Instruct and LLaDA-1.5. To quantify this, the paper introduces **Temporal Semantic Entropy (TSE)**, which clusters intermediate answers by semantic equivalence and measures the entropy of that cluster distribution across the sampling trajectory — correctly-answered questions statistically exhibit lower TSE. Two methods exploit this: **Temporal Self-Consistency Voting**, a training-free test-time decoding strategy that aggregates intermediate predictions via weighted voting (no extra forward passes needed, unlike standard self-consistency); and **Temporal Consistency Reinforcement**, a GRPO-based post-training method using negative TSE as a self-supervised reward (no ground truth required), optionally combined with an accuracy reward via a spherical scoring rule.

## Key claims

- A significant, consistent gap exists between final Pass@1 and ever-pass rate across all four benchmarks and both models tested — e.g., GSM8K length 128, LLaDA-8B-Instruct: 68.5% final vs. 80.5% ever-pass, a 12.0-point gap (**sourced**).
- Token-level entropy decreases steadily during sampling; Intermediate-Correct questions (correct at some step, wrong at the end) begin with lower entropy than Always-Incorrect questions on GSM8K, suggesting initially confident but unstable predictions (**sourced**).
- Correctly-answered questions statistically exhibit lower Temporal Semantic Entropy than incorrectly-answered ones, across all four benchmarks (**sourced**).
- Temporal Self-Consistency Voting with exponential step-weighting (α=5) gives an average +1.5% accuracy improvement over the LLaDA-8B-Instruct baseline, with negligible overhead — no extra forward passes, since it reuses the existing sampling trajectory (**sourced**).
- Temporal Consistency Reinforcement using negative TSE reward alone gives an average +24.7% improvement on Countdown over the SFT baseline — matching or surpassing the ground-truth accuracy-reward baseline (d1) despite requiring no labels (e.g., Countdown: 24.7% vs. d1's 15.1%) (**sourced**).
- Combining TSE reward with accuracy reward yields further gains: +2.0% GSM8K, +4.3% MATH500, +6.6% SVAMP, +25.3% Countdown over the SFT baseline (**sourced**).
- The method's benefit is conditional on the model's baseline ability to sometimes produce correct intermediate answers: on the Sudoku dataset, where Temporal Accuracy (average correctness across all intermediate steps) stays below 5%, both temporal voting and negative-TSE-only RFT *underperform* the baseline — a documented failure mode, not merely an unstudied edge case (**sourced**).
- After RFT with the negative TSE reward, TSE decreases (more semantically consistent outputs), but ever-pass rate remains above final-pass rate — indicating temporal oscillation is reduced, not eliminated (**sourced**).
- Temporal Self-Consistency Voting still improves accuracy even after RFT — the two methods are complementary, not redundant (**sourced**).

## Mechanisms

- **Ever-pass rate**: EverPass@1|t = E_i[max_{k∈{1,...,t}} e_i,k] — the fraction of questions correctly answered at *any* step up to t, as an upper-bound reference against which final Pass@1 is compared.
- **Temporal Semantic Entropy (TSE)**: given T intermediate answers {x_0^t}, cluster them into semantic-equivalence groups C = {C_1,...,C_K}; TSE = −Σ_Ck (Σ p(x_0^t)) log(Σ p(x_0^t)) over each cluster's aggregated probability mass — higher TSE = more semantic fluctuation, lower TSE = convergence to a stable meaning.
- **Temporal Self-Consistency Voting**: a* = argmax_a Σ_t f(t)·1(meaning(x_0^t) = a), with f(t) a monotonically decreasing (later-step-favoring) weighting function; exponential weighting f(t) = exp(α(1 − t/T)) with α=5 empirically best, beating fixed (equal-weight) and linear weighting.
- **Temporal Consistency Reinforcement**: GRPO with reward r_i = −TSE(o_i) per response, using the diffu-GRPO method to estimate token-level probabilities via averaging over randomly masked prompt versions. When combined with accuracy reward: r_i = 1[o_i=o*] + spherical-scored normalized-confidence term, where confidence c(o_i) = (H_max − TSE(o_i))/H_max is derived from TSE and H_max = log T.
- **Training detail**: only answers from the second half of sampling steps are used for TSE computation (first-half answers are "rough and less reliable" per the paper); answer-parsing failures are discarded from the TSE calculation, not penalized.

## Hardware implications

- Temporal Self-Consistency Voting requires *no additional forward passes* — it reuses intermediate predictions already computed during standard sampling, making its runtime overhead essentially the cost of tracking/clustering intermediate answers, not additional model compute (**sourced**). This is a rare case in this wiki's literature of an accuracy improvement with near-zero inference-time compute cost, as opposed to a compute-for-quality trade.
- Storing/tracking intermediate answers across all T steps for voting requires retaining decoded text (or logits) per step rather than discarding them immediately — a modest memory-retention requirement, distinct from the KV-cache memory concerns dominant elsewhere in this wiki (**inferred**).
- Temporal Consistency Reinforcement is a training-time (not inference-time) cost — it doesn't affect deployed inference latency/memory at all, only training compute (**sourced**, GRPO/LoRA training details in the paper; **inferred** the deployment-cost framing).

## Relevance to Apple Silicon

- **Likely useful**: Temporal Self-Consistency Voting's near-zero overhead makes it a low-risk accuracy improvement to layer on top of any existing Apple Silicon dLLM inference pipeline, independent of whatever caching/kernel-fusion strategy is used elsewhere in this wiki, since it only requires retaining and comparing already-computed intermediate outputs (**inferred**).
- **Not directly a kernel-design concern**: unlike most sources in this wiki, this paper's contributions (voting, RL reward) are algorithm/training-level, not attention/caching/kernel-level — no direct Metal kernel implications beyond the modest intermediate-answer-retention memory noted above (**inferred**).
- **Unknown portability**: whether the modest memory overhead of retaining per-step intermediate answers meaningfully competes with KV-cache memory budget on unified-memory Apple Silicon systems is not evaluated in the source (**speculative**).

## Extracted concepts

- [[temporal-oscillation]]
- [[temporal-semantic-entropy]]
- [[temporal-self-consistency-voting]]
- [[temporal-consistency-reinforcement]]

## Proposal impact

- No `04-Proposals/` pages exist yet. This source is a candidate input for a future *decoding-strategy* proposal (as distinct from the wiki's dominant caching/kernel-fusion proposals) since Temporal Self-Consistency Voting's near-zero overhead makes it a low-cost addition to any Apple Silicon inference pipeline regardless of which caching/kernel strategy is chosen.

## Open questions

- Does Temporal Self-Consistency Voting compose with [[credit-decoding]] (which also accumulates confidence signal across steps, but to influence *when tokens commit* rather than to *select the final answer* post-hoc)? The two mechanisms operate at different points in the pipeline but share the "aggregate across steps" premise.
- The Sudoku failure case (Temporal Accuracy <5%) suggests this method's benefit is gated on a baseline level of intermediate correctness — is there a way to detect this failure mode automatically at inference time and fall back to standard final-step decoding, rather than requiring it to be known in advance per-task?
- How does semantic-equivalence clustering (needed for TSE) behave for open-ended generation tasks without a clean extractable answer (the paper only evaluates math/reasoning benchmarks with parseable numeric answers)?
- Would Temporal Self-Consistency Voting's near-zero overhead still hold if combined with aggressive parallel-decoding methods elsewhere in this wiki (e.g., [[hierarchical-caching]], [[complementary-attention-mask]]) that already reduce the number of steps — does fewer steps reduce the voting signal's reliability?
