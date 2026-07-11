# Auto-Speculative Decoding (Spiffy)

**Summary**: A form of speculative decoding for dLLMs where draft candidates are sampled from the model's own output distribution rather than from a separate drafter model — eliminating the two-model memory/compute burden of drafter-verifier approaches.
**Aliases**: auto-speculation, self-distributed drafting, drafterless speculative decoding
**Status**: draft
**Last updated**: 2026-07-03
**Sources**:
- [[spiffy]]

---

## Definition

Auto-speculative decoding is Spiffy's core departure from the drafter-verifier paradigm covered by [[speculative-decoding-dllm]] (DualDiffusion): instead of running a separate, cheaper "drafter" model to propose candidate tokens for a slower "verifier" model to check, Spiffy draws draft states directly from the same dLLM's own output distribution, using token-position rank and vocabulary rank to select candidates (**sourced**, [[spiffy]]). No second model is trained, loaded, or run — the "speculation" is generated and verified by one model.

## Why it matters

[[speculative-decoding-dllm]] (DualDiffusion) explicitly names memory footprint as an open risk: "if both models are large (e.g., 7B each), on-device deployment may be impossible." Auto-speculative decoding sidesteps this concern entirely by construction — there is no second model to keep resident in memory. This makes it a structurally different, and for memory-constrained (including Apple Silicon unified-memory) deployment, potentially more attractive category of speculative decoding than the two-model drafter-verifier approach already documented in this wiki.

## Mechanism

Draft candidates are selected via two rankings computed from the dLLM's own single forward pass: token position rank i (via ArgSort of max marginal probabilities across positions) and vocabulary rank j (via ArgSort of per-position token probabilities), yielding candidate tokens c_ij (**sourced**, [[spiffy]]). Which specific (i,j) combinations to actually try as drafts is not decided per-inference but fixed in advance by [[offline-draft-calibration]], and the space of draft possibilities is structured as a [[directed-draft-graph]]. Verification is proven exactly lossless (Appendix A.1 of the source), preserving the same output distribution as fully sequential decoding (**sourced**, [[spiffy]]).

## Trade-offs

- **No separate drafter means no drafter-quality risk**: [[speculative-decoding-dllm]]'s trade-offs section flags "a poorly performing drafter will cause many remasks, reducing speedup" as a central risk of the two-model approach — auto-speculation has no analogous drafter-quality failure mode, since drafts come from the verifying model's own distribution by construction.
- **Trades that risk for a different one**: because drafts come from the same model, auto-speculation cannot benefit from a drafter that is cheaper *per forward pass* (e.g., a genuinely smaller model or heavily quantized variant) — its speedup comes entirely from batching multiple draft candidates into fewer verification passes, not from running a cheaper model more often.
- Near-losslessness (not exactly lossless) on LLaDA specifically, due to bfloat16 precision effects under batched attention masks — an implementation/precision caveat, not a fundamental limitation of the auto-speculation idea itself (**sourced**, [[spiffy]]).

## Apple Silicon implications

- **Inferred**: eliminating the second model is a direct, first-order memory-footprint win for unified-memory deployment — this is the single clearest Apple Silicon advantage of auto-speculative decoding over [[speculative-decoding-dllm]]'s two-model approach, and one explicitly named as an open risk in that page's own Apple Silicon section.
- **Speculative**: the bfloat16-specific near-losslessness caveat may behave differently under Metal's fp16/bf16 numerics — not evaluated in the source.

## Related concepts
- [[speculative-decoding-dllm]] (the two-model drafter-verifier paradigm this approach avoids; update candidate for cross-reference)
- [[directed-draft-graph]] (the structure organizing which draft candidates to try)
- [[offline-draft-calibration]] (how the draft graph is determined in advance)
- [[self-speculative-decoding-dlm]] (S2D2 — another single-model self-speculation approach, but via block-size-1 AR verification rather than distribution-based drafting; direct comparison not made in either source)
- [[lossless-parallel-decoding]] (the broader losslessness-guarantee category this belongs to)

## Open questions
- How does auto-speculative decoding compare directly, same benchmarks, to [[self-speculative-decoding-dlm]] (S2D2), the wiki's other single-model self-speculation approach?
- Does auto-speculation's reliance on the model's own distribution quality mean it degrades more gracefully or less gracefully than a separate-drafter approach when the base model itself is weak on a given task?
- Could auto-speculative decoding be combined with a genuinely cheaper (quantized/distilled) version of the same model as an additional drafter layer, hybridizing with the two-model approach rather than treating them as mutually exclusive?
