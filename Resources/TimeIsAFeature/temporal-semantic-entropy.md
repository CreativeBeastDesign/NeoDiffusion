# Temporal Semantic Entropy (TSE)

**Summary**: An uncertainty metric that clusters a trajectory's intermediate answers by semantic equivalence and measures the entropy of that cluster distribution — higher TSE means more semantic fluctuation across denoising steps, lower TSE means convergence to a stable meaning. Correctly-answered questions statistically exhibit lower TSE.
**Aliases**: TSE, semantic trajectory entropy
**Status**: draft
**Last updated**: 2026-07-03
**Sources**:
- [[time-is-a-feature]]

---

## Definition

Temporal Semantic Entropy quantifies semantic instability across a dLLM's sampling trajectory. Given T intermediate answers {x_0^t}_{t=1}^T extracted during decoding, they are clustered into semantic-equivalence groups C = {C_1,...,C_K} (answers meaning the same thing are grouped together regardless of surface-form differences); TSE is then the entropy of the resulting cluster-probability distribution: TSE({x_0^t}) = −Σ_Ck (Σ_{x_0^t∈Ck} p(x_0^t)) log(Σ_{x_0^t∈Ck} p(x_0^t)) (**sourced**, [[time-is-a-feature]]). This adapts semantic entropy (previously used for hallucination detection in AR LLMs, Farquhar et al. 2024) to the specific setting of a single dLLM's *own* denoising trajectory rather than multiple independent samples.

## Why it matters

TSE is the metric that operationalizes [[temporal-oscillation]] into something usable as both a diagnostic and, critically, a training signal: because it requires no ground-truth labels (it only needs the trajectory's own intermediate outputs, clustered by semantic equivalence), negative TSE can serve as a fully self-supervised reward for [[temporal-consistency-reinforcement]] — a genuinely unsupervised alternative to ground-truth accuracy rewards used by prior dLLM RL post-training methods (diffu-GRPO, UniGRPO, coupled-GRPO, all named in the source as relying on ground truth) (**sourced**, [[time-is-a-feature]]).

## Mechanism

Practical computation details from the source: only answers from the *second half* of sampling steps are used, since first-half answers are "rough and less reliable"; answer-parsing failures (where the extracted text doesn't parse to a clean answer) are discarded from the TSE calculation rather than penalized (**sourced**, [[time-is-a-feature]]). TSE is normalized in the combined-reward variant via c(o_i) = (H_max − TSE(o_i))/H_max, with H_max = log T, yielding a confidence score in [0,1] used alongside a binary accuracy reward and a spherical scoring rule (Gneiting & Raftery, 2007) — chosen over quadratic/logistic/entropy scoring alternatives after an ablation (Table S3 in the source) (**sourced**, [[time-is-a-feature]]).

## Trade-offs

- **Requires semantic clustering, which requires some notion of answer equivalence**: the paper's evaluated benchmarks (GSM8K, MATH500, SVAMP, Countdown) all have clean, parseable numeric answers, making clustering straightforward — how TSE would be computed for open-ended generation without an obvious equivalence relation is not addressed.
- **Discards first-half-step information by design choice**: this is a pragmatic engineering decision (early answers are noisy) that trades some information loss for a cleaner signal — the paper doesn't ablate how sensitive TSE's usefulness is to this specific half-and-half split.
- Correctly-answered questions exhibit lower TSE only *statistically* — this is a population-level correlation, not a per-instance guarantee, meaning TSE alone cannot certify any individual answer as correct.

## Apple Silicon implications

- **Inferred**: computing TSE requires retaining and clustering intermediate answers across (at least the second half of) the sampling trajectory — a modest memory-retention cost distinct from KV-cache memory pressure, more comparable to retaining decoded text/logits per step than to attention-state memory.
- No direct kernel/hardware implications — this is a training-and-evaluation-time metric, not an inference-path computation in deployed models (the reward is used during RL fine-tuning, and [[temporal-self-consistency-voting]]'s use of similar trajectory information at inference time is covered separately).

## Related concepts
- [[temporal-oscillation]] (the phenomenon TSE quantifies)
- [[temporal-self-consistency-voting]] (uses trajectory information similarly, for a different purpose — inference-time answer selection rather than a training reward)
- [[temporal-consistency-reinforcement]] (uses negative TSE directly as an RL reward)

## Open questions
- How would semantic clustering for TSE be defined for tasks without clean, parseable answers (open-ended generation, free-form QA)?
- Is the second-half-only step-filtering heuristic robust across different total step counts T, or would it need re-tuning (e.g., a different fraction) for much shorter or longer diffusion schedules?
- Does TSE correlate with [[premature-overconfidence]] (D2Cache's related concept about early confident-but-wrong predictions) — both concern confidence/certainty dynamics over the trajectory, but from different papers and for different purposes, not yet reconciled.
