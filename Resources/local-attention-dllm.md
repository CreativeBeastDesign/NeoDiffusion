# Local Attention in dLLMs

**Summary**: The broad category of techniques that restrict a dLLM's attention computation to a spatially local neighborhood around each position, of which sliding-window attention and distance-decay dropout (DPad) are specific instances.
**Aliases**: locality-restricted attention, neighborhood attention
**Status**: draft
**Last updated**: 2026-07-03
**Sources**:
- [[dpad]]

---

## Definition

Local attention in dLLMs is the general principle that a position's attention computation can be restricted to nearby positions without significant quality loss, because distant tokens (particularly distant suffix/MASK tokens) contribute diminishing, low-entropy signal. DPad's source note lists this as a distinct, broader concept from its two concrete mechanisms — [[sliding-window-attention]] (hard cutoff) and [[distance-decay]] dropout, folded into [[suffix-dropout]] (soft, distance-weighted cutoff) — treating "local attention" as the umbrella category both instantiate (**sourced**, [[dpad]]).

## Why it matters

Framing local attention as its own concept (rather than folding it entirely into suffix-dropout) matters because the *locality principle* generalizes beyond DPad's specific suffix-tokens application. The same reasoning — that distant tokens in a dLLM's bidirectional field contribute less signal per unit of compute than nearby ones — motivates Elastic-Cache's windowing of active MASK prediction ([[sliding-window-attention]]) and could plausibly motivate other locality-restricted designs not yet covered in this wiki (e.g. local attention applied to the *prefix* side, which no ingested source currently addresses).

## Mechanism

DPad's instantiation combines two locality mechanisms applied to suffix tokens specifically:
- A hard sliding window (tokens outside window size are excluded).
- A soft distance-decay function within/at the window boundary (keep probability decreases with distance — linear, exponential, or step function).

Both are deterministic (same tokens treated the same way given the same relative positions) rather than learned or content-adaptive, which the source note calls out as a deliberate simplicity choice enabling "a few lines of code" implementation (**sourced**, [[dpad]]).

## Trade-offs

- **Determinism vs. adaptivity**: DPad's locality restriction is purely positional, not content-aware — unlike, say, [[mask-query-attention]]'s content-driven importance scoring. This trades potential accuracy (a distant-but-important token would still be dropped) for implementation simplicity and zero runtime scoring overhead.
- **Generality vs. verification**: as a category, "local attention" is broader than what any single ingested source has actually validated — DPad validates it for suffix tokens specifically; whether the same locality assumption holds for prefix tokens or for prompt tokens is untested in this wiki's source material.

## Apple Silicon implications

- Purely positional locality restrictions (unlike content-adaptive ones) can be expressed as a static, precomputed attention mask — cheap to generate once and reuse, and naturally maps to fixed-size Metal threadgroups sized to the locality window.
- **Speculative**: combining DPad's suffix-side local attention with Elastic-Cache's active-window local attention (a form of prefix/MASK-side locality) could in principle restrict the *entire* attention computation to a bounded local region rather than just one side — not evaluated together in any ingested source, but a natural fusion candidate to flag for a proposal page.

## Related concepts
- [[suffix-dropout]] (DPad's concrete distance-decay + window mechanism)
- [[sliding-window-attention]] (the hard-cutoff instance of this category)
- [[elastic-cache]] (a differently-motivated but structurally similar local-window mechanism)

## Open questions
- Does the locality principle hold for prefix tokens, not just suffix tokens? No ingested source addresses this directly.
- Could local attention be made content-adaptive (e.g. combined with mask-query attention scoring) without losing DPad's implementation simplicity?
- What is the actual measured speedup on Apple Silicon hardware specifically, versus the CUDA numbers reported in the source paper?
