# Offline Draft Calibration (Spiffy)

**Summary**: A one-time, pre-inference calibration procedure (<50 samples, <30 minutes on one GPU) that determines a fixed draft graph structure by selecting draft formulas via "degree-1-accumulation" — shown empirically superior to frequency-only or full-ancestor-chain selection.
**Aliases**: draft graph calibration, degree-1-accumulation selection
**Status**: draft
**Last updated**: 2026-07-03
**Sources**:
- [[spiffy]]

---

## Definition

Offline draft calibration is the procedure (Algorithm 2 in the source) that produces the specific [[directed-draft-graph]] used at inference time: it collects (i,j) token-position/vocabulary-rank sequences from rewound generation traces over a small calibration dataset (fewer than 50 samples), then selects D draft formulas via a "degree-1-accumulation" optimization: Q* = argmax Σ(count(q) + Σ_parents count(p)) (**sourced**, [[spiffy]]). This is a training-free, one-time procedure — it runs once (in under 30 minutes on a single GPU) and produces a fixed structure reused across all subsequent inference, not a per-inference or continuously-updated process.

## Why it matters

This is what turns [[directed-draft-graph]] from an abstract structural possibility into a concrete, usable inference-time artifact. Without a way to decide *which* of the enormous space of possible draft candidates to actually try, the graph structure alone doesn't yield a runnable algorithm. The calibration procedure's cheapness (under 50 samples, under 30 minutes) is itself notable — it's a small fraction of typical model training or even typical fine-tuning cost, making it practical to run once per deployment target rather than being a barrier to adoption.

## Mechanism

1. Run rewound generation traces over a small calibration dataset, collecting the sequence of (token-position-rank i, vocabulary-rank j) pairs that occur.
2. Evaluate candidate draft formulas Q using degree-1-accumulation: Q* = argmax Σ(count(q) + Σ_parents count(p)) — i.e., score each candidate by its own occurrence count *plus* the occurrence counts of its immediate parents in the draft graph (one level of ancestry, not the full ancestor chain).
3. Select the top D draft formulas by this score, forming the fixed draft graph used at inference.

Ablations (Table 4 / Appendix E) show degree-1-accumulation beats two alternatives: "degree-0-accumulation" (frequency only, no parent information) and "total-accumulation" (full ancestor chain, not just immediate parents) (**sourced**, [[spiffy]]). The paper also reports the resulting calibration graphs generalize well across datasets/models with only minor structural variation, and remain effective when reused rather than needing per-deployment recalibration (**sourced**, [[spiffy]]).

## Trade-offs

- **Degree-1 is an empirical sweet spot, not a theoretically derived optimum**: the paper shows degree-1-accumulation beats degree-0 and total-accumulation in ablation, but doesn't provide a first-principles explanation for why exactly one level of parent-context is optimal rather than two or more — flagged as an empirical finding, not a proven-optimal design.
- **Fixed graph trades adaptivity for cheapness**: because calibration is offline and produces one fixed structure, the draft graph cannot adapt to per-input variation the way a dynamically-constructed candidate set could — the paper's finding that graphs "generalize well... with only minor structural variation" is the evidence offered that this trade-off is acceptable in practice, not a guarantee.

## Apple Silicon implications

- **Inferred**: because calibration is a one-time, offline, small-cost procedure, it is naturally compatible with a "calibrate once per deployment target, ship a fixed artifact" workflow — a calibrated draft graph could be calibrated once (potentially even off-device, on more powerful hardware) and shipped as a fixed asset alongside an Apple-Silicon-deployed model, rather than requiring on-device calibration.
- No further Apple-Silicon-specific hardware claims are made in the source.

## Related concepts
- [[directed-draft-graph]] (the structure this calibration produces)
- [[auto-speculative-decoding]] (the drafting approach this calibration supports)

## Open questions
- Does calibration need to be re-run per target hardware (e.g., separately for a discrete-GPU deployment vs. an Apple-Silicon deployment), or is the calibrated graph purely a function of the model/data and hardware-agnostic, as the "generalizes well" finding might suggest?
- Is there a principled reason degree-1-accumulation (one level of parent context) outperforms both degree-0 and full-ancestor-chain accumulation, or is this purely an empirical finding specific to the evaluated setups?
