# WP-1b — Training-free MultiBD on LLaDA2.1-mini (provisional accept)

**Summary**: Second NeoDiffusion Phase-3 work package: the [[mbd-lms]] Block-Buffer MultiBD decoding (N_buf=2, training-free) implemented in the Phase-2 loop and swept on chat/reasoning — the domains the source paper leaves uncharted. TPF-logical +23% (chat, τ_add=0.5) / +22% (reasoning, τ_add=0.1–0.3) with per-domain optima that differ from the paper's math/code settings; M1 wall-clock does not convert (compute-bound, 1.75× dual-step multiplier), so the net-TPS gate defers to the Studio backfill.
**Status**: implemented + M1-swept (2026-07-11); provisional algorithmic accept, served default stays nBuf=1
**Type**: experiment (05-Experiments)
**Engine**: NeoDiffusion branch `wp-1b-multibd`; record `Plans/wp1b-logbook.md`
**Host**: dev M1 under the roadmap §0.1 dev-only protocol (counters decide; wall-clock host-scoped)
**Sources**: [[mbd-lms]] (= arXiv:2606.29215, "Multi-Block Diffusion Language Models" — Algorithm 5 + Appendix C.3/C.4 + Table 4 deep-read this session), `Plans/phase-3-optimisation-roadmap.md` §1

---

## What was built

Algorithm 5 ("Optimized MultiBD with a fixed Block Buffer") in the existing Block-Buffer-shaped loop:

- **Slot scheduler**: up to two concurrently active blocks in one 64-token cached forward against the ExactPrefixCache; front-block in-order commit (streaming semantics intact); EOS early-stop cancels the trailing slot.
- **Attention**: the cached path's `mask: nil` invariant holds only for one active block; dual phases pass a block-causal active-window mask (prefix columns free, trailing block sees front, never vice versa). Proven by cached==uncached token identity with 88 activation events exercised — the uncached full-window `.strict` mask is the semantics anchor.
- **τ_add (sourced)**: activate the next block when the newest active block's decoded fraction (generated positions) strictly exceeds τ_add. **τ_semi (sourced, and NOT what the roadmap draft guessed)**: it gates the *trailing block's top-1 fallback* — a block with zero above-threshold acceptances gets the forced acceptance only if its predecessor is semi-complete (progress > τ_semi) or committed. The paper's LLaDA2.1-Mini row (Table 4) uses τ_M2T=0.70/τ_T2T=0.50 — exactly the Q-mode thresholds, a strong reference-alignment cross-check.
- **Scheduling = batch-break events**: activation/settle/budget are per-step in-graph flags; one `[2K]` stacked readback per speculative batch keeps the M5(c) sync budget and makes results K-invariant (tested K=1 vs K=4 including activation timing).

## Results (M1, 4-bit, Q mode, gen-128; deterministic counters)

| τ_add | chat TPF-logical | reasoning TPF-logical |
|---|---|---|
| — (nBuf=1) | 1.79 | 4.77 |
| 0.1 / 0.3 | 2.16 (+21%) | **5.83 (+22%)** |
| **0.5** | **2.21 (+23%)** | 5.24 (+10%) |
| 0.7 / 0.9 | ~1.85 (+3%) | ~5.24 (+10%) |

- **gen-256 probes**: reasoning's gain grows with runway (+29.5% at τ_add=0.5, vs +22% at gen-128) and both suites converge on τ_add≈0.5 — the gen-128 numbers are runway-limited lower bounds, and a single long-form preset looks viable. The chat cliff localizes to (0.6, 0.7).
- Chat has a **cliff between 0.6 and 0.7**; reasoning prefers early activation. Both optima differ from the paper's math (0.10) and code (0.90) — τ_add must be tuned per serving mode, not ported.
- Front-block interference (the [[mbd-lms]] open question on editing with two live blocks): +2 steps/block median at τ_add=0.1, **zero at ≥0.5**; T2T editing keeps functioning (chat edits 4→8).
- **Wall-clock does not convert on the M1**: dual-phase per-step latency ≈1.75× single (H100: 1.24×) ⇒ ≈+11% net M1 wall-clock at the chat winner despite −20% steps. Compute-bound roofline attribution; net-TPS verdict → Studio backfill (M2 Ultra expected to behave like the paper's H100 conversion, 1.78×TPF/1.24× → 1.44× TPS).
- TPF-honest lags TPF-logical as event density rises (batch-split overshoot at K=4) — an event-aware-K follow-up is filed under the kernel/loop track.

## Results (Mac Studio M2 Ultra Backfill, 2026-07-14)

| Domain | Baseline TPS | MultiBD TPS | Δ (Wall-Clock TPS) | Δ (Logical TPF) |
|---|---|---|---|---|
| Chat | 48.19 | 39.82 | −17.4% | +13.3% |
| Code | 99.25 | 78.98 | −20.4% | +64.7% |
| Reasoning | 76.31 | 66.90 | −12.3% | +74.3% |

- **Step latency multiplier overrides logical savings**:
  On the target Studio hardware, evaluation of two active slots concurrently in Q-mode increases latency per forward significantly (due to sequence/active-window length doubling). This multiplier is larger than the logical-step savings, rendering the overall wall-clock serving throughput (TPS) negative across all suites.
- **Composition with Credit Decoding (`p4-combo`)**:
  Exposing Credit Decoding jointly with MultiBD + Dynamic-τ achieves **45.55 TPS on chat**, cutting the baseline deficit to just −5.5% and yielding a **+14.4% serving speedup** over `p3-combo` (39.82 TPS).
- **Serving Verdict (REJECT for default-on, ACCEPT as served preset)**:
  We reject MultiBD + Dynamic-τ as the default-on serving configuration. However, it is accepted and exposed as a custom served preset option for long-form, runway-unlimited generations where step count is the main constraint.

## Related

[[mbd-lms]] · [[block-buffer]] · [[multi-block-teacher-forcing]]


