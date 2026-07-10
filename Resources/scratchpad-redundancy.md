# Scratchpad Redundancy in dLLMs

**Summary**: The empirical observation that suffix tokens in dLLMs function as an information-collecting "scratchpad" for the prefix, but are themselves low-entropy and largely redundant — the underlying problem [[suffix-dropout]] is designed to exploit.
**Aliases**: scratchpad pruning target, suffix redundancy
**Status**: draft
**Last updated**: 2026-07-03
**Sources**:
- [[dpad]]

---

## Definition

Scratchpad redundancy is DPad's characterization of *why* suffix tokens in dLLMs are safe to drop: suffix positions (not-yet-finalized future tokens) act as an "information reservoir" — collecting signal from already-decoded prefix tokens to help guide generation — but they do not themselves carry direct semantic information, and most of them are low-entropy, often forming long runs of similar or predictable content (**sourced**, [[dpad]]). This is the *problem statement* DPad addresses; [[suffix-dropout]] and the underlying [[sliding-window-attention]] / [[distance-decay]] mechanisms are the *technique* that exploits it. The two should be kept distinct: scratchpad redundancy is an empirical claim about suffix-token informativeness, not a method.

## Why it matters

Without this observation, dropping suffix tokens from attention would just look like an accuracy-risking shortcut. DPad's contribution is the empirical claim that most suffix-token computation is wasted in the first place — the suffix "collects" prefix signal but doesn't independently encode meaningful new content — which is what licenses removing distant suffix tokens from attention with (per the source) minimal quality loss, up to 61.4× speedup (**sourced**, [[dpad]]).

## Mechanism

The source note frames this as a two-part claim rather than a single mechanism:
1. **Reservoir role**: suffix tokens exist, functionally, to receive and hold signal propagated from the prefix during bidirectional attention — not to originate new semantic content themselves.
2. **Low-entropy redundancy**: empirically, suffix tokens tend toward long runs of similar/predictable content, meaning many of them carry near-duplicate information relative to their neighbors.

Together these justify treating distant suffix tokens as safe to exclude from attention computation, which is operationalized by [[sliding-window-attention]] (hard cutoff) and [[distance-decay]] (soft, position-weighted cutoff) inside [[suffix-dropout]] (**sourced**, [[dpad]]).

## Trade-offs

- **Claim, not proof**: the source note asserts redundancy empirically (via the paper's reported results) rather than deriving it from a formal argument — the wiki has no independent verification of *how* low-entropy suffix runs are, only the downstream speedup/accuracy numbers as indirect evidence.
- **Task dependence**: whether suffix content is reliably low-entropy plausibly varies by task (e.g. code generation with long, structurally-necessary suffixes vs. open-ended prose) — not addressed directly in the ingested source, flagged here as a reasonable extrapolation. **Inferred**.
- Distinguishing this from [[suffix-dropout]] matters for the wiki's evidence/design separation: this page states the *premise*; the technique page states the *exploitation* — conflating them would blur what's an empirical claim about the model versus what's an engineering choice built on top of it.

## Apple Silicon implications

No independent hardware implication beyond what [[suffix-dropout]] and [[local-attention-dllm]] already state — this page is the empirical premise underlying those mechanisms' Apple Silicon fusion/masking opportunities, not a separate implementation target.

## Related concepts
- [[suffix-dropout]] (the technique built on this observation)
- [[sliding-window-attention]] (one of the two mechanisms exploiting this redundancy)
- [[distance-decay]] (the other mechanism, folded into [[suffix-dropout]])
- [[local-attention-dllm]] (the broader category both belong to)

## Open questions
- Does suffix-token low-entropy redundancy hold uniformly across task types (code vs. long-form reasoning vs. open-ended generation), or does it vary enough to require task-adaptive window sizing?
- Is there a way to directly measure/quantify scratchpad redundancy (e.g. an entropy metric over suffix positions) rather than relying on downstream benchmark speedup as indirect evidence?
