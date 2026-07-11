# WP-2a Auto-speculation Logbook — LLaDA2.1-mini, M1 dev host

**Status**: in progress (2026-07-11, branch `wp-2a-speculation`)
**Scope**: single-model speculation behind one policy seam — S2D2 (arXiv:2603.25702) implemented first; Spiffy (arXiv:2509.18085) behind a calibration-trace ceiling gate (plan §Phase 3). nBuf=1, elastic off (one-variable; composability cells stay open). §0.1 protocol: counters decide, wall-clock host-scoped, arms are recorded command lines.

## 0. Method in one paragraph

S2D2 v1: every denoising step drafts the first contiguous masked span from the target forward's own argmax, then one extra 2B-wide "M_ver" forward re-scores the span under a block-size-1 AR view (2L trick at duplicated absolute positions, against committed prefix KV); the matching prefix is accepted, the first mismatch takes the verifier's token. Acceptance replaces Γ on the span (no top-1 fallback needed); Δ editing untouched; all in-graph (no new readbacks — sync budget formula unchanged, tested). Min-span routing at batch boundaries (τ_span, masks-remaining proxy) is the real-savings lever: an unverified batch never builds the verifier forward. Correctness anchors: verifier ≡ sequential blockLength-1 causal decoding (exact test); vectorized acceptance ≡ scalar reference; K-invariance at τ_span=1. **S2D2 output legitimately differs from vanilla decoding** — quality via blind sheet, no token-parity gate.

## 1. Starting point (sourced)

- S2D2 paper, LLaDA2.1-Mini "Conservative": 2.2× vs AR, 1.3× vs dynamic-threshold baseline, GSM8K 89.8 / MBPP 68.8 (accuracy sometimes *improves* — AR verification acts as energy correction).
- Roadmap accept gate: better-of-two lands at ≥20% TPS over the then-current stack; M1 decides via counters (width-corrected TPF), TPS defers to Studio.
- WP-1b measured facts reused: M1 is compute-bound (64-wide dual multiplier ~1.75×); the smoke below confirms width is the binding constraint here too.

## 2. Hypotheses (pre-registered 2026-07-11 ~11:00, BEFORE the τ_span sweep; smoke-informed)

The always-verify smoke (timeline 4) already showed: acceptance 3.2–5.3 tokens/verified step (mechanism works, paper-level), steps/block ×0.63–0.76, but width-corrected TPF 0.52–0.84 vs baseline ≈1.3–1.5 — **always-verify is compute-negative ~2× on the width metric**. The sweep therefore tests the routing operating point:

- **H1**: τ_span routing monotonically trades acceptance for width: τ_span=8/16 verify only long-span (early-block) steps where acceptance is highest; width-corrected TPF improves over τ_span=1 on all suites.
- **H2**: an operating point with width-corrected TPF ≥ baseline exists at τ_span ∈ {8, 16} on **code** (the paper's strength domain, longest spans); chat likely stays below baseline on the width metric — **speculative**.
- **H3**: even where the width metric stays <1×, the *steps* metric (TPF-logical) improves ≥40% at τ_span=1 — relevant because Studio wall-clock per forward is latency- not width-proportional (WP-1b: 64-wide costs 1.75×, not 2×); the M1 width metric is the conservative bound, the Studio TPS run decides the gate.
- **H4**: quality (blind sheet, τ_span=8 winner) is non-degrading — ties dominate; corrections are AR-guided and the paper saw accuracy gains.
- **H5**: acceptance/verified-step is higher on code ≥ reasoning ≥ chat (span coherence ordering).

## 3. Timeline

| # | When | What | Result / note |
|---|---|---|---|
| 1 | 10:20 | Step 0: merged `wp-1b-multibd` → main (d56f478); JOT Tier-3 deferral entry; branch `wp-2a-speculation` | |
| 2 | 10:25–10:40 | Implementation: `s2d2VerifierMask` (M_ver, full-B static shapes); `GenerationParams.speculation/tauSpan`; `VerifierForward` closure (memoized mask, duplicated positions); windowStep S2D2 branch (span via cumsum, cumprod-style acceptance, correction token, non-span threshold Γ, fallback subsumed); stats [4S]→[5S] with accepted counts; width-aware `tokensProcessedInForwards` | build clean |
| 3 | 10:43 | Tests: mask cell-by-cell; **verifier ≡ sequential blockLength-1 AR (anchor, exact argmax + logits <1e-3)**; acceptance ≡ scalar reference (5 patterns incl. gap-span/no-mask/immediate-fail); E2E fixtures; K-invariance; sync budget | 7/7 + full suite **87 tests, 0 failures**. Toolchain note: width-1 reference forwards hit a missing `gemv_float32` metallib specialization — reference windows padded by one causally-invisible column (engine never runs width-1) |
| 4 | 10:50 | Smoke, real 4-bit, chat gen-64, always-verify: `--speculation s2d2` | **acceptance 3.2–5.3/step; steps/blk 7–11.3 (vs 12.5–18 base); width-corrected TPF 0.52–0.84 — compute-negative ~2×**; peak 9.62 GB |
| 5 | 10:55 | Min-span batch-boundary routing (`tauSpan`, masks-remaining proxy from already-read stats; real graph savings; K-dependent for τ_span>1 by design — Proposal-B precedent, K-invariance tested at τ_span=1) | suite still green |
| 6 | 11:05 | **τ_span sweep launched** (arms below) | log `scratch/wp2a_sweep.log` |

Sweep arms (recorded verbatim; binary = post-commit `wp-2a-speculation` release build):
`diffusion-bench llada --runs 2 --cooldown 30 --suites chat,reasoning,code --gen-length 128 --arm {base-wp2a | s2d2-tspan1 | s2d2-tspan8 | s2d2-tspan16} --arm-mode q [--speculation s2d2 --tau-span τ] --json scratch/wp2a_sweep.jsonl`

## 4. Findings

**F1 — Acceptance is exceptional; the S2D2 mechanism works better here than in the paper's own setting.** (sourced: `scratch/wp2a_sweep.jsonl`, τ_span=1) Accepted tokens per verified step (median): **code 9.77, reasoning 6.78, chat 4.12** (H5 ordering confirmed). Steps/run drop: code 24→20, chat 50→40, with TPF-logical +21% (code) to +73% (chat). The verifier and the diffusion drafter agree on long runs — LLaDA2.1's block-causal AR view is a strong self-verifier.

**F2 — Always-verify is compute-negative ~2–3× on the width metric (H2 first half refuted at τ_span=1).** (sourced) Width-corrected TPF: chat 0.82 (−45%), reasoning 1.12 (−65%), code 1.53 (−57%) vs baselines 1.49/3.21/3.57. Cause: every step pays 3B of forward width (target B + verifier 2B) for a ~0.7× step reduction, plus K=4 overshoot at 3B per discarded forward.

**F3 — Batch-boundary routing is inert in this regime (H1 refuted at K=4).** (sourced) τ_span ∈ {8, 16} produced rows *identical* to τ_span=1 (same tokensProcessed to the token). Diagnosis: acceptance is so high that blocks settle in 1–2 batches of K=4; the first batch is always verified by initialization, code/reasoning blocks are single-batch (the routing update point — a batch with no event — never executes), and chat retains ≥τ_span masks after batch 1. **The routing granularity (4 steps × 3B) exceeds the total speculation opportunity per block.** The Proposal-B-style pattern that worked for elastic (long phases) cannot work when speculation itself shortens phases below one batch.

**F4 — K=1 (zero overshoot, per-step routing) improves but does not convert; masks-remaining routing cannot discriminate.** (sourced) K=1 + τ_span 8: chat −34.5%, reasoning −55.5%, code −46.2% width-corrected (vs −45/−65/−57 at K=4 — overshoot elimination is worth ~10–20pp). τ_span 8 vs 16 identical at K=1 too: masks stay above both thresholds throughout the productive phase (blocks start at 32).

**F5 — The break-even identity, and why Q-mode wins (the WP-1a structural echo).** (inferred from F1–F4 arithmetic, then confirmed causally in F6) A verification (2B extra width) pays iff it saves ≥2 target forwards, i.e. iff **acc/verified-step > 2 × baseline tokens-per-step**. Q-mode's aggressive τ=0.7 thresholding already yields 2.3 (chat) to 5.3 (code/reasoning) tokens/step *for free*, so the bar is 4.6–10.7 — and measured acceptance (4.1/6.8/9.8) sits just under it on every suite. **The engine's baseline again already banks most of what the paper sells** (WP-1a: ExactPrefixCache absorbed Elastic-Cache's tiers; here: threshold parallel decoding absorbs self-speculation's parallelism). The S2D2 paper's LLaDA2.1 baseline decodes conservatively (τ≈0.95-class), where the bar is ~2×1.2 ≈ 2.4.

**F6 — Causal confirmation: against a τ=0.95 baseline, S2D2 converts (+19.1% chat, +15.2% reasoning, +4.9% code, width-corrected; steps halve).** (sourced: `base-t95` / `s2d2-t95-k1` arms) The identity predicted conversion exactly where it happened. But in absolute terms `s2d2-t95` (chat 1.13) still loses to plain Q-mode (1.49): on compute-bound hosts, aggressive thresholding is the cheaper harvest of the same parallelism. **Surviving niche**: a quality-leaning "verified-conservative" preset (τ=0.95 + AR verification — the S2D2 paper reports accuracy *gains* in this regime) at ~25% compute premium over Q-mode; decision needs the blind sheet + Studio wall-clock.

## 5. Results summary

*(pending)*

## 6. Deviations from the source (recorded)

1. **Full-B verifier window** (paper verifies only the span): fixed 2B shape is required by the K-step lazy batching; non-span verifier rows are don't-care. Costs width, bought back by routing.
2. **Routing at batch boundaries** with a masks-remaining proxy (paper routes per step on |C_t|): per-step routing cannot change lazy graph structure; the proxy equals span length while the masked region is contiguous. τ_span>1 is K-dependent by design.
3. **Greedy acceptance only** (temp 0): argmax match, no rejection sampling — the min(1,q/p) rule degenerates to this at temperature 0.
4. **Non-span masked positions keep plain threshold Γ on verified steps** (paper's Alg 3 verifies the span within the same outer loop; combining both is the natural extension for LLaDA2.1's scattered late-block masks); span acceptance subsumes the top-1 fallback.
5. **Δ editing untouched** — accepted tokens remain editable on later steps (the paper calls S2D2 complementary to LLaDA2.1 self-correction; we keep both live).
6. S2D2 is **cached-path only** (verifier conditions on committed KV); `generate()` + speculation traps by precondition.
