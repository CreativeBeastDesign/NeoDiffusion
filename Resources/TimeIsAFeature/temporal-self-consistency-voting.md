# Temporal Self-Consistency Voting (TSE)

**Summary**: A training-free, test-time decoding strategy that aggregates a dLLM's intermediate predictions across denoising steps via weighted voting to select the final answer — unlike standard self-consistency, it requires no additional forward passes, since it reuses a single sampling trajectory already computed.
**Aliases**: temporal voting, weighted trajectory voting
**Status**: draft
**Last updated**: 2026-07-03
**Sources**:
- [[time-is-a-feature]]

---

## Definition

Temporal Self-Consistency Voting selects the final answer via weighted majority vote over a dLLM's own sampling trajectory: a* = argmax_a Σ_t f(t)·1(meaning(x_0^t) = a), where f(t) is a monotonically decreasing (later-step-favoring) weighting function (**sourced**, [[time-is-a-feature]]). It directly exploits [[temporal-oscillation]]: since correct answers sometimes appear at intermediate steps only to be overwritten, voting across the whole trajectory (rather than trusting only the final step) recovers some of that discarded correctness.

## Why it matters

The paper explicitly contrasts this with standard self-consistency decoding (Wang et al. 2022), which improves AR-LLM reasoning by sampling multiple *independent* full generations and majority-voting — an approach requiring several complete, expensive forward passes. Temporal Self-Consistency Voting achieves an analogous effect using a *single* dLLM sampling trajectory, because dLLM decoding already naturally produces a sequence of intermediate predictions as a byproduct of its iterative denoising process — a structural advantage unique to diffusion-style generation that AR self-consistency cannot exploit (**sourced**, [[time-is-a-feature]]). This gives an average +1.5% accuracy improvement over the LLaDA-8B-Instruct baseline with negligible overhead (**sourced**, [[time-is-a-feature]]).

## Mechanism

Three weighting schemes for f(t) are compared: fixed (f(t)=1, equal weight to all steps), linear (f(t) = 1 − t/T), and exponential (f(t) = exp(α(1−t/T)), α=5). Exponential weighting gives the largest gains in ablation — fixed weighting performs slightly *worse* than baseline in some cases, attributed to equal weights amplifying inaccurate early predictions (**sourced**, [[time-is-a-feature]], Table 1). The α hyperparameter itself is ablated (Fig. 5a): values from 1 to 11 consistently improve accuracy, peaking at α=5 with an average gain of 1.5% (**sourced**, [[time-is-a-feature]]). The method remains effective even after [[temporal-consistency-reinforcement]] fine-tuning — Table S4 in the source shows continued (smaller) gains from voting on top of an already-RFT'd model, indicating the two methods are complementary rather than redundant (**sourced**, [[time-is-a-feature]]).

## Trade-offs

- **Ceiling is the ever-pass rate, not perfect accuracy**: the paper reports EverPass@1|t as an oracle upper bound for comparison — Temporal Self-Consistency Voting captures only part of the gap between final Pass@1 and this oracle (e.g., GSM8K-128: baseline 68.5%, voting 70.1%, oracle ever-pass 80.5%), meaning substantial recoverable accuracy remains unexploited by this specific voting mechanism.
- **Fails when intermediate answers are rarely correct at all**: on the Sudoku dataset (a documented failure case in the source, Sec. E.2), Temporal Accuracy across all intermediate steps stays below 5%, and voting actually *underperforms* the baseline (12.2%→12.5% at length 128 but 6.7%→6.1% and 5.5%→2.8% at lengths 256/512) — voting has no reliable correct signal to aggregate when the model rarely produces correct intermediate answers (**sourced**, [[time-is-a-feature]]).
- Requires retaining intermediate predictions across the full trajectory (or at least enough of it to vote over), a modest but non-zero memory-retention cost not present in standard final-step-only decoding.

## Apple Silicon implications

- **Sourced**: negligible computational overhead — no additional forward passes required, since it reuses predictions already computed during standard sampling.
- **Inferred**: this is a rare case in this wiki of an accuracy improvement with near-zero inference-time compute cost, making it a low-risk addition to layer on top of any existing Apple Silicon dLLM inference pipeline regardless of which caching/kernel-fusion strategy is used elsewhere — it operates at the answer-selection layer, orthogonal to attention/caching mechanics.
- **Speculative**: whether the modest memory-retention requirement (storing intermediate answers across the trajectory) meaningfully competes with KV-cache memory budget on unified-memory systems is not evaluated in the source.

## Related concepts
- [[temporal-oscillation]] (the phenomenon this method exploits)
- [[temporal-semantic-entropy]] (a related but distinct trajectory-aggregation metric, used for RL reward rather than voting)
- [[temporal-consistency-reinforcement]] (the complementary training-time method — shown to combine additively with this one)
- [[credit-decoding]] (a pre-existing wiki concept that also aggregates confidence signal across steps, but to decide *when tokens commit during generation* rather than to select the *final answer post-hoc* — different point in the pipeline, open question on whether they compose)

## Open questions
- Is there a way to detect the Sudoku-style failure mode (low Temporal Accuracy) automatically at inference time, to fall back to standard final-step decoding rather than requiring foreknowledge of which tasks this method helps or hurts on?
- Does this method's near-zero overhead still hold when combined with aggressive parallel-decoding methods elsewhere in this wiki that already reduce the number of denoising steps — does fewer steps reduce the voting signal's reliability, since there is less trajectory to vote over?
- How does semantic-equivalence clustering (needed to determine which intermediate answers "agree") scale or degrade for tasks without a clean parseable answer format?
