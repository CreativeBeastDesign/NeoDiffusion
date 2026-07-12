# WP-4D — Credit Decoding: Decay-Discounted Credit Feedback Matrix (Implemented; Logical-Step Reduction Measured, Wall-Clock/Quality Gates Pending)

**Summary**: Fourth NeoDiffusion Phase-4 work package: [[credit-decoding]] implemented in the `DiffusionEngine` pipeline and verified for cached/uncached parity, speculation invariance, and synthetic liveness. Credit Decoding acts as a token-level momentum filter: it accumulates raw prediction confidence for each position's top candidate across denoising steps, decays it over time, and boosts that candidate's logit so high-consensus masked tokens clear the unmasking threshold sooner, reducing steps/block.
**Status**: implemented and unit-verified on the dev host (2026-07-12). **Not yet an accepted optimisation** — logical-step reduction is measured; wall-clock and blind-quality gates are outstanding (see "Open gates" below). Following the phase-3 §0.1 protocol, only hardware-independent metrics are treated as decided here; wall-clock is host-scoped and deferred to the Studio backfill.
**Type**: optimization (05-Experiments)
**Engine**: branch `optimisation/p4/WP4d`
**Sources**: [[credit-decoding]] ("dInfer: Accelerated Free-Form Selective Generation via Credit Feedback", arXiv:2510.01239)

---

## What was built

- **Credit Feedback State Management [sourced: DiffusionEngine.swift]**: Added `var credit: MLXArray? = nil` to the slot-state tracker `SlotRun`, with per-slot slice allocation for MultiBD (`credit[0..., (s*B)..<((s+1)*B), 0...]`) and Option-C early commits (`front.credit?[0..., n..<L, 0...]`).
- **In-Graph Credit Update [sourced: DiffusionEngine.swift]**: `windowStep` accepts and returns a `[1, A, V]` credit matrix, per step:
  1. Decay: \(C \leftarrow C \cdot \beta\)
  2. Confidence boost: \(b \leftarrow (p_{\text{raw}})^{\gamma}\) (from the **raw** top-candidate probability)
  3. One-hot accumulate on the raw argmax: \(C_{t, \hat{x}} \leftarrow C_{t, \hat{x}} + b_t\)
  4. Logit enhancement: \(\tilde{f}_{t} \leftarrow f_{t} + \alpha \cdot \log(1 + C_t)\)
- **Boost scoped to the Γ (unmasking) pathway [sourced: DiffusionEngine.swift — see "Deviations"]**: the enhanced logits set only a *masked* position's confidence (`unmaskConf`, tested against τ_mask) and the token written when it unmasks (`unmaskTok`). Δ-editing, JOT freezing, the ICE answer-confidence signal, and all diagnostics run on the **raw** predictions, so credit never re-biases an already-decoded position.
- **Token-One-Hot Memory Safety [sourced: DiffusionEngine.swift]**: avoids a \(V \times V\) identity (\(V = 157{,}184 \Rightarrow \approx 98.8\) GB FP32) by broadcasting `arange(V)` against the argmax in-graph, so per-step credit allocation is \(\approx 20\) MB at \(A = 32\). **Caveat**: the matrix is dense over the full vocabulary even though only the historical top candidate per position ever accrues credit — a sparse `[1, A]` (id + value) representation would remove the full-vocab `pow`/`log1p`/compare and is the obvious next optimisation if this ships default-on.
- **Parity, Invariance, and Liveness Tests [sourced: CreditDecodingTests.swift]**: four tests, all passing (2026-07-12):
  1. `testCreditDecodingCachedUncachedIdentity` — token-for-token cached==uncached on 16 fixture cases.
  2. `testCreditDecodingSpeculationInvariance` — K=1 vs K=4 identical output and step counts.
  3. `testCreditDecodingLivenessSynthetic` — a synthetic underconfident-but-stable forward settles in 2 steps with credit vs ≥B without.
  4. `testCreditDecodingWithIceAndVoting` — credit + ICE + TSCV compose and preserve cached/uncached parity.

---

## Results

> **Provenance warning.** The step-count figures below were measured on an **earlier build** in which the boost was applied to *all* logits, before it was scoped to the Γ pathway (see "Deviations"). They are retained as directional evidence only and **must be re-measured on the current code before any promotion decision**. All figures are single- or few-prompt anecdotes on the 16 GB dev host under heavy memory paging — they are **not** a statistical result and carry no wall-clock or quality verdict.

1. **Logical-step reduction under real workloads (measured, hardware-independent) [inferred pending re-run: diffusion-bench log task-295]**:
   With \(\alpha=1.0, \beta=0.9, \gamma=0.5\) on the `chat` suite vs `q-baseline`, denoising steps/block fell on high-consensus prompts (e.g. `chat-recipe` 9.5 → 6.5; `chat-capital` 12.5 → 9.0) and were roughly flat on lower-consensus ones (`chat-explain` 19.0 vs 19.5). This is consistent with the mechanism — momentum helps where early consensus exists — but the magnitudes are pre-scoping-fix and unweighted across a 4-prompt sample.

2. **Composes with ICE + TSCV [inferred pending re-run: diffusion-bench logs task-363/379/407/413]**:
   On GSM8K math prompts (gen-128), Credit+ICE+TSCV showed lower logical steps/block than ICE+TSCV alone on the sampled prompts (e.g. one prompt 62 → 33, another 66 → 20). These are illustrative single-prompt traces, not an accuracy study; in at least one case the greedy final answer was wrong and only TSCV's vote recovered a correct value — i.e. the step saving and the answer correctness are independent effects and neither is established at sample size.

3. **Parity and K-invariance hold (measured, on current code) [sourced: swift test, 2026-07-12]**:
   All four `CreditDecodingTests` pass on the scoped-boost build, confirming credit preserves exact cached/uncached parity and speculation invariance, alone and jointly with ICE + TSCV. The M5a/M5b baseline parity suites (16/16 token-for-token) are unaffected — credit-off output is byte-identical to the reference.

### Not established (open gates)

- **Wall-clock speedup — UNKNOWN on dev host.** Under isolated runs the suite's steady-state denoise seconds were essentially flat (\(\approx 20.2\) s baseline vs \(\approx 21.1\) s credit) despite the logical-step drop, because free memory was logged at \(\le 128\) MB and macOS paging dominated wall-clock. No net-latency claim can be made from the dev M1. Any "≈50% serving speedup" is a **speculative** extrapolation from the logical-step count, not a measurement, and is explicitly *not* claimed here — it is a Studio-backfill question.
- **Quality — NOT gated.** No blind paired scoring has been run (the Phase-4 acceptance discipline requires it). Credit changes the denoising trajectory, so a quality sheet is required before default-on; none exists yet.

---

## Deviations from source / scope limits

- **Boost scoped to unmasking (Γ), not all logits [DiffusionEngine.swift]**: dInfer's credit mechanism accelerates masked-position decoding. A naive implementation that lets the enhanced logits drive `x0`/`x0p` everywhere also biases Δ-editing decisions, JOT freezing, and the ICE answer-confidence signal on already-decoded positions — outside the mechanism the paper describes. The implementation therefore keeps raw predictions for those consumers and uses the enhanced values only for the masked position's τ_mask test and its write token. This is a deliberate deviation; it is also recorded in `Plans/phase-4-further-optimisations.md` §WP-4d.
- **Dense credit matrix**: see the memory caveat above — structurally sparse, stored dense; optimisation deferred.
- **Composition status**: credit is verified to *compose* (parity holds) with ICE and TSCV. It is **not** composed with S2D2 speculation (that verifier branch keeps its own correction token); credit × speculation is an untested arm.

---

## Key lessons

- **Diffusion search spaces benefit from momentum**: autoregressive models never revise a committed token, so they need no credit memory; block-diffusion predictions oscillate across steps, and decayed credit acts as a temporal low-pass filter that reinforces early consensus so high-confidence tokens commit in bursts. This is the *hypothesised* mechanism, supported by the liveness test and the directional step-count evidence — not yet by a wall-clock or quality gate.
- **Scope a logit boost to the pathway it targets**: applying credit to every logit silently changed editing and freezing behaviour. Restricting it to Γ keeps the intervention interpretable and keeps the other Phase-3/4 levers (JOT, ICE, dynamic τ) on their intended raw signal.
- **In-graph vocabulary-scale allocations need broadcast discipline**: building the one-hot via `arange(V)` broadcast (not a materialized identity) is what keeps the per-step cost at ~20 MB instead of OOM.
