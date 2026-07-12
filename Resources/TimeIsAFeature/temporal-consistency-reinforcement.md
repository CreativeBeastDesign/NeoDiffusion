# Temporal Consistency Reinforcement (TSE)

**Summary**: A GRPO-based post-training method using negative Temporal Semantic Entropy as a self-supervised reward — no ground-truth labels required — to encourage stable, consistent generations; optionally combined with a ground-truth accuracy reward via a spherical scoring rule for further gains.
**Aliases**: TSE-RFT, negative-TSE reward training
**Status**: draft
**Last updated**: 2026-07-03
**Sources**:
- [[time-is-a-feature]]

---

## Definition

Temporal Consistency Reinforcement is a reinforcement fine-tuning (RFT) method that uses Group Relative Policy Optimization (GRPO) with reward r_i = −TSE(o_i) per sampled response — encouraging the model to produce generations whose intermediate predictions remain semantically consistent throughout decoding, without requiring ground-truth answers for reward computation (**sourced**, [[time-is-a-feature]]). This is explicitly contrasted with prior dLLM RL post-training methods named in the source (diffu-GRPO, UniGRPO, coupled-GRPO), all of which rely on ground-truth rewards — this method is described as fully unsupervised (**sourced**, [[time-is-a-feature]]).

## Why it matters

Negative-TSE-only training achieves a remarkable +24.7% average improvement on Countdown — matching or surpassing the ground-truth accuracy-reward baseline (d1: +15.1%) despite using no labels at all (**sourced**, [[time-is-a-feature]]). This is a significant result for any setting where labeled data is scarce or unavailable: it demonstrates that a purely self-supervised signal derived from the model's own trajectory dynamics can substitute for, and in this case exceed, supervised reward on at least one benchmark. Combining TSE reward with an accuracy reward yields further gains across all four benchmarks tested (+2.0% GSM8K, +4.3% MATH500, +6.6% SVAMP, +25.3% Countdown over the SFT baseline) (**sourced**, [[time-is-a-feature]]).

## Mechanism

Standard GRPO formulation: for each question q, sample a group of G responses from the old policy, compute advantage A_i^k(π) = r_i(π) − mean{r_j(π)}, and optimize the clipped-ratio GRPO objective with a KL penalty against a reference policy. Token-level probabilities for the importance-sampling ratio are estimated via the diffu-GRPO method (averaging outputs from multiple randomly masked versions of the prompt) (**sourced**, [[time-is-a-feature]]). When combining TSE with accuracy reward: r_i = 1[o_i=o*] + spherical-scored confidence term, where confidence c(o_i) = (H_max − TSE(o_i))/H_max is derived from TSE — an ablation over four scoring-rule variants (entropy, quadratic, logistic, spherical) found spherical scoring superior (**sourced**, [[time-is-a-feature]], Table S3). Training used LoRA (rank 128, scaling factor 64), sequences of 256 tokens, 8 H800 GPUs; LLaDA-8B-Instruct underwent SFT on s1K before RFT, while LLaDA-1.5 skipped SFT (found to cause performance degradation, likely because it had already undergone sophisticated post-training) (**sourced**, [[time-is-a-feature]]).

## Trade-offs

- **Gated on baseline intermediate-correctness**: per the paper's own limitations discussion (Sec. E.1/E.2), this method depends on the model already being able to generate correct or near-correct intermediate answers — on the Sudoku dataset (Temporal Accuracy below 5%), negative-TSE-only RFT *underperforms* both the baseline and the accuracy-reward alternative (e.g., length 512: baseline 5.5%, d1 accuracy-reward 12.7%, negative-TSE-only 3.3%) (**sourced**, [[time-is-a-feature]]).
- **Combining TSE with accuracy reward mitigates but doesn't fully resolve the gating issue**: on Sudoku, combining both rewards (27.5%/27.8%/16.6% across lengths 128/256/512) substantially outperforms accuracy-reward-alone (23.2%/17.8%/12.7%) — the paper hypothesizes this is because TSE provides more fine-grained reward signal than the binary correct/incorrect accuracy reward alone, even when TSE alone is insufficient (**sourced**, [[time-is-a-feature]]).
- After RFT, TSE decreases and ever-pass rate remains above final-pass rate — the method reduces but does not eliminate [[temporal-oscillation]] (**sourced**, [[time-is-a-feature]]); effective-token count also declines post-RFT (more concise outputs), which the authors suggest but do not confirm may itself reduce oscillation opportunity.

## Apple Silicon implications

- **Sourced**: this is purely a training-time cost (GRPO/LoRA fine-tuning on 8 H800 GPUs) — it does not affect deployed inference latency or memory at all; a model fine-tuned this way runs with identical inference-time compute/memory characteristics to an equivalent non-RFT'd model.
- **Inferred**: because the benefit is baked into model weights (not an inference-time algorithm), this method is orthogonal to every kernel/caching design consideration elsewhere in this wiki — a model trained with Temporal Consistency Reinforcement can be deployed through any of the inference-acceleration methods covered elsewhere without conflict, since it changes *what* the model outputs, not *how* inference is computed.

## Related concepts
- [[temporal-semantic-entropy]] (the reward signal this method optimizes against)
- [[temporal-oscillation]] (the phenomenon this method aims to reduce)
- [[temporal-self-consistency-voting]] (the complementary test-time method — shown to still help even after this training-time method is applied)

## Open questions
- Is there a way to predict in advance (without running full RFT) whether a given task will hit the Sudoku-style failure mode, to avoid wasted training compute on tasks where negative-TSE reward will underperform?
- How does the effective-token-count reduction observed post-RFT (more concise outputs) causally relate to reduced temporal oscillation — is conciseness a side effect of consistency training, or does it independently contribute to the observed accuracy gains?
- Would combining Temporal Consistency Reinforcement with any of this wiki's inference-acceleration methods (caching, speculative decoding) reveal any interaction effects, given the two operate at different stages (training vs. inference) and are argued to be orthogonal but not empirically tested together in the source?
