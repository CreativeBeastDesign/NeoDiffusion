# WP-2b Streaming-dLLM Logbook — LLaDA2.1-mini, M1 dev host

**Status**: in progress (2026-07-11, branch `wp-2b-streaming`)
**Scope**: the Streaming-dLLM cluster (arXiv:2601.17917) per roadmap §2 WP-2b, under the §0.1 dev-only protocol: counters decide, wall-clock host-scoped, every arm a recorded command line. Plan approved by André 2026-07-11 (2b-1 closed without implementation; τ_edit stays static; τ_eos knob only if the text-change rate is nonzero).

## 0. Method in one paragraph

Two of the paper's three modules are implemented as default-off knobs, ablated separately then combined with MultiBD: **dynamic confidence threshold** τ(t) = τ0·(1 − α(1 − r_mask)) replacing the static τ_mask inside the per-slot Γ (r_mask = start-of-step masked fraction over the slot's generated positions; α=0 ⇒ byte-identical parity), and **EOS early exit** (once a Γ/Δ-settled non-prompt EOS exists in the front block, every still-masked window position after the first EOS is filled with EOS in-graph — the block and any trailing slot settle via the normal mask-free break; output is trim-invariant by construction). The third module (suffix window + final-token cue) is closed without implementation (§2). Gates: cached==uncached identity at α>0, K-invariance, synthetic-forward liveness (17→7 steps) and early-exit behavior (17→2 steps, trim-identical) — toy random weights produce no mid-range confidences and no EOS, hence the injected-`Forward` tests.

## 1. Starting point (sourced)

- Streaming-dLLM (source note `Resources/possibly new/streaming-dllm.md`): three modules each contribute in ablation (Table 3); hyperparameters τ0=0.9 fixed, α≈0.6 (Fig. 6), block 32; up to 68.2× combined (225.3× at gen-2048) on **full-bidirectional** models (Dream/LLaDA/LLaDA-1.5) where the suffix is forwarded every step and generation budgets are long. Our ≤4k chat profile at gen-128/256 sees a fraction by construction.
- **Formula-vs-prose discrepancy (recorded)**: the source note's prose says the threshold "tightens as fewer masked tokens remain", but the formula (quoted identically in the roadmap §2) *loosens* it — τ0 at r_mask=1 → τ0(1−α) at r_mask=0. The formula matches the paper's motivation (mean confidence rises within a block; late steps can accept more) and is what we implement.
- Prior baseline-relativity lessons (WP-1a: ExactPrefixCache; WP-2a: threshold decoding): checked per-module in §2/§3 before implementation this time.
- M6 frozen dev baseline + envValid discipline; effective-echo provenance rule (elastic F7, validated in WP-1b F7): JSONL rows record engine echoes `dynamicTauAlpha`/`eosEarlyExit`, never CLI values. Smoke-verified before the sweep (timeline 4).

## 2. Sub-item 2b-1 (suffix window + final-token cue): CLOSED — N/A, no implementation (the fourth baseline-relativity instance)

The paper's Attenuation-Guided Suffix Modeling prunes the *suffix*: distant future masked blocks that full-bidirectional dLLMs forward at every denoising step. On block-causal LLaDA2.x that object **does not exist anywhere in the pipeline**:
1. The reference itself forwards `cur_x = x[:, :current_window_end]` — prefix + active block only (`modeling_llada2_moe.py:1343`, verified this session). There is no suffix compute to prune even in the ground truth.
2. Our engine's window is `[committed prefix (or cache) ++ active slots]` (`DiffusionEngine.swift`); distant MASK blocks are never materialized (already noted in the WP-1a closure).
3. Under the `.strict` block-causal mask the active block *could not attend* to suffix positions even if they were computed — so the paper's accuracy-preserving final-token positional cue has no signal to preserve either: the model was trained block-causally and never sees a suffix.
Verdict: zero ceiling, architecturally; nothing to ablate. Sequence: WP-1a (prefix recompute pre-banked by ExactPrefixCache) → WP-2a (parallelism pre-banked by threshold decoding) → 2b-1 (suffix pruning pre-banked by block-causality). Approved for closure without implementation by André (plan review, 2026-07-11).

## 3. Hypotheses (pre-registered 2026-07-11 ~16:35 CEST — written while the sweep ran, before any row was read)

- **H1 (2b-2 main)**: some α at Q mode reduces steps/block ≥10% on chat+reasoning (the plan's accept line) with smoke-clean text. Mechanism: late-block *fallback steps* (no position clears static τ0 → top-1 only) convert into multi-accept steps as τ(t) drops.
- **H2**: steps saved grow monotonically with α, but α=0.9 (late-block floor τ0(1−α)=0.07) accepts near-anything and risks visible quality damage; expected winner 0.3–0.6.
- **H3 (the pre-registered baseline-relativity check)**: the overlap risk here is *partial*, not total — Q-mode's high-confidence bursts already accept multi-token steps early in the block, but the late-block tail is fallback-dominated, which is exactly the paper's target. Judged by the transfersPerStep shape: if late-block steps are already multi-accept at baseline, dynamic τ has nothing to sell (record negative, lesson #4b).
- **H4**: S mode (τ0=0.5) gains less than Q — its baseline is already loose; floor at α=0.6 is 0.2.
- **H5 (2b-3)**: step savings concentrate in the EOS block; text-change rate **zero** (trim-invariance argument); chat total-step saving ~5–15% at gen-128 (answers end early), reasoning ~0. Wall-clock additionally saves the never-started blocks only via eosEarlyStop, which the baseline arm already has — the marginal saving is within-final-block.
- **H6 (composability)**: dynamic τ gains stack roughly additively with nBuf=2 (per-slot τ(t)); eosexit does not interfere with dual-slot scheduling (activation is already EOS-gated under eosEarlyStop).

### τ_edit decision + follow-up estimate (André's plan-review ask)

τ_edit stays **static** in v1 (one-variable discipline; the paper's models have no editing mechanism, so there is no sourced dynamic-τ_edit form). Is a dynamic-τ_edit follow-up test worthwhile? **Estimate: low value; conditional entry only.** Reasoning: (a) the Δ set is tiny at the served modes (WP-1b measured 4–8 edits per 128-token chat run — there is almost no volume to modulate); (b) *loosening* τ_edit late-block would create more edits, and edits gate the settle break — more steps, not fewer, plus churn risk; (c) *tightening* it late could only save steps if edit-churn is what keeps blocks alive, which post-steps/block already measures. **Entry condition (recorded)**: if the 2b-2 winner still shows post-steps/block ≥ ~3 driven by late edits, run one dynamic-τ_edit probe arm (tightening direction only); otherwise drop without implementation.

## 4. Timeline

| # | When (CEST) | What | Result / note |
|---|---|---|---|
| 1 | ~13:30 | Research + plan (plan doc `WP-2b — Streaming-dLLM cluster`): reference forward-window check killed 2b-1 (§2); André approved plan + decisions; branch `wp-2b-streaming` from main (f007a2f) | |
| 2 | 13:40–14:20 | Implementation: `GenerationParams.dynamicTauAlpha/eosEarlyExit`; `windowStep` per-slot τ(t) + post-Γ/Δ EOS fill; preconditions (dynamic τ × speculation forbidden — matrix cell 2a×2b-2 open; eosEarlyExit requires eosEarlyStop); `Metrics` echoes; bench `--dyn-tau-alpha`/`--eos-early-exit` + JSONL fields | build clean |
| 3 | 14:33 | Tests `WP2bTests` (6): cached==uncached at α=0.6 (16 cases); synthetic liveness **17→7 steps** at α=0.5 (identical tokens); K-invariance both knobs; synthetic early exit **17→2 steps**, trim-identical; combined knobs + nBuf=2 identity. Full generation suite green (existing parity untouched — both knobs default-off). Note: fixture-weight liveness assertion failed first (toy confidences sit outside the τ0→τ0(1−α) window at α=0.6 — recorded; liveness moved to the synthetic forward) | 6/6 + suite green |
| 4 | 16:30 | Smoke, real 4-bit: custom prompt, α=0.6 + eos-early-exit; echoes verified (`dynamicTauAlpha 0.6, eosEarlyExit true`), peak 9.57 GB | F7 rule satisfied pre-sweep |
| 5 | 16:36 | **Sweep launched** (driver `scratch/wp2b_driver.sh`, log `scratch/wp2b_sweep.log`, rows `scratch/wp2b_sweep.jsonl`): 13 arms × chat+reasoning × 4 prompts × 1 run (counters deterministic), gen-128, K=4 — Q α∈{0,0.3,0.6,0.9}, S α∈{0,0.3,0.6}, τ0=0.9 static+dynamic probe, eosexit A/B with `--dump-text`, nBuf=2 (τ_add=0.5) × {base, α=0.6, eosexit} | results §5 |

## 5. Findings (sweep: 13 arms × chat+reasoning × 4 prompts × 1 run, gen-128, K=4; counters deterministic; envValid ~0/8 per arm — memory-pressured session, wall-clock indicative only, counters decide per §0.1)

**F1 — Dynamic τ clears the accept line on both suites; winner α=0.6 at Q (H1 confirmed, H2 mostly).** (sourced: `scratch/wp2b_sweep.jsonl`) Q-mode logical steps: chat 216→184 (**−14.8%**, TPF-logical 1.89→2.39 = +26%), reasoning 97→86 (**−11.3%**, 4.22→4.77 = +13%) at α=0.6. Not monotonic on chat: α=0.9 gives −11.6% but post-steps/block jump 1.00→1.62 — the near-zero late-block floor (τ0(1−α)=0.07) buys acceptances that Δ then has to repair (edit churn). Reasoning tolerates α=0.9 (−21.6%, floor still 0.07 but its blocks settle in 5–6 steps anyway). Winner: **α=0.6**, the paper's own optimum, transferring across τ0 0.9→0.7 and model families.

**F2 — The mechanism is the predicted one (H3: the overlap is partial, and the un-banked part is real).** Mean Γ-acceptances/step rise 1.85→2.21 (chat, base→α=0.6) while post-steps stay near 1: the converted steps are the late-block fallback-dominated tail, exactly the paper's target — the first Phase-3 candidate whose lever was *not* already banked by the engine's baseline (contrast WP-1a/WP-2a/2b-1).

**F3 — S mode gains MORE on chat, not less (H4 refuted), but with visible churn.** S chat 225→170 (−24.4% at α=0.6) — but post-steps/block 1.11→2.08. S's τ_edit=0 lets any changed prediction edit, so cheap late acceptances recycle through refinement. Usable, but Q+α=0.6 is the cleaner operating point.

**F4 — τ0=0.9 probe (threshold-calibration consolidation): static τ0=0.9 is dominated; dynamic helps everywhere; Q-mode τ0=0.7 stays the anchor.** t09-static: chat 343 / reasoning 167 steps (vs Q base 216/97); +α=0.6 recovers to 266/97 but never beats Q base. Verdict for `Optimisations.md`'s convergence item: per-step dynamic logic wins over any single recalibrated τ0; OSDT-style one-shot calibration is only needed to set (τ0, α) per domain — and α=0.6 transferred as-is.

**F5 — EOS early exit is a mechanistic NULL on real weights (H5 refuted — the saving is pre-banked, fifth instance).** eosexit rows are *byte-identical* to baseline (steps, per-block distributions, texts: 0/8 changed; same at nBuf=2). Diagnosis from the trajectories: post-EOS padding positions are high-confidence EOS predictions that clear τ_mask early; the boundary EOS itself settles late (often via fallback). By the time a settled EOS exists, nothing after it is still masked — threshold decoding already banks the paper's post-EOS saving, and commit-time `eosEarlyStop` already cancels the never-started blocks. Robust to generation length (the mechanism is per-block). Per André's plan condition (text-change rate is zero): **no τ_eos knob**; the capability stays landed default-off, tested, with this null recorded.

**F6 — Composability with MultiBD: additive, cell closed (H6 confirmed).** nBuf=2 (τ_add=0.5) + α=0.6: chat 190→162 steps (−14.7%, matching the single-slot −14.8%), reasoning 84→70 (−16.7%). Cumulative vs the WP-2b baseline: **chat TPF-logical 1.89→2.67 (+41%), reasoning 4.22→5.92 (+40%)** — the two levers stack. eosexit × nBuf=2: null × anything = null, no interference.

### Results summary (Q mode, gen-128, logical steps over 4 prompts; Δ vs base-wp2b)

| Arm | chat steps (Δ) | chat TPF-log | reasoning steps (Δ) | reasoning TPF-log | post/blk chat |
|---|---|---|---|---|---|
| base-wp2b | 216 (—) | 1.89 | 97 (—) | 4.22 | 1.00 |
| dyntau-a03 | 202 (−6.5%) | 2.11 | 88 (−9.3%) | 4.58 | 1.06 |
| **dyntau-a06** | **184 (−14.8%)** | **2.39** | **86 (−11.3%)** | **4.77** | 1.16 |
| dyntau-a09 | 191 (−11.6%) | 2.28 | 76 (−21.6%) | 5.46 | 1.62 |
| t09-static | 343 (+59%) | 1.18 | 167 (+72%) | 2.47 | 1.27 |
| t09-dyn-a06 | 266 (+23%) | 1.68 | 97 (±0%) | 4.25 | 1.10 |
| eosexit | 216 (±0%) | 1.89 | 97 (±0%) | 4.22 | 1.00 |
| nbuf2-base (τ_add .5) | 190 (−12%) | 2.26 | 84 (−13%) | 4.82 | 1.25 |
| **nbuf2-dyntau-a06** | **162 (−25%)** | **2.67** | **70 (−28%)** | **5.92** | 1.90 |
| nbuf2-eosexit | 190 | 2.26 | 84 | 4.82 | 1.25 |

S mode: 225→187 (α=0.3) → 170 (α=0.6) chat; 90→89→82 reasoning; post/blk up to 2.08 (F3).

**Verdict (dev host, §0.1): 2b-2 provisional algorithmic ACCEPT at α=0.6; 2b-3 NULL; 2b-1 closed N/A (§2).** Deciders are deterministic counters and clear the plan's ≥10% line on both suites; served default stays `dynamicTauAlpha 0` until (a) André's blind scores (`scratch/wp2b_blind/sheet.md`, 8 prompts static-vs-α=0.6, objective checks clean: no mask leaks, no 4-gram degeneration deltas, comparable lengths; score with `Tools/m8_blind_sheet.py --pairs scratch/wp2b_blind/pairs.json --out scratch/wp2b_blind --score scratch/wp2b_blind/sheet.md`) and (b) the Studio scored-set + wall-clock backfill (recorded arms, driver `scratch/wp2b_driver.sh`). τ_edit follow-up: entry condition NOT met (post-steps/block 1.16 at the winner < ~3) — dropped per §3. Recommended preset direction (post-Studio): Q + α=0.6 (+ nBuf=2 τ_add=0.5 once WP-1b's Studio gate lands — the stack is additive, F6).

> ⚠️ **AMENDED 2026-07-17 (§8): the α=0.6 accept does not survive its quality gate on chat.** Condition (a) has returned and is not clean — α=0.6 shows visible token corruption on 2/4 chat prompts, 0/4 reasoning. The α=0.6 accept **stands for reasoning** and is **withdrawn for chat** (candidate: α=0.3). The "objective checks clean" claim above is retracted as evidence of non-degradation — the checks passed on visibly corrupted text (F12). The recommended preset direction (Q + α=0.6 + nBuf=2) is **suspended pending a quality sheet on the stack itself** (F10).

## 6. Deviations from the source (recorded)

1. **No suffix module** — §2: the paper's third component is inapplicable to block-causal LLaDA2.x; closed, not skipped silently.
2. **r_mask denominator = generated positions only** (paper: ratio over the block; our prompt-tail block would skew it — same deviation as WP-1b's progress denominators).
3. **Early exit trigger = settled EOS, not "high-confidence EOS prediction"**: the fill requires the EOS to have been *written* by Γ/Δ (already past τ_mask/τ_edit), i.e. the confidence gate is the decoding threshold itself; a separate τ_eos is added only if the text-change rate is nonzero (André's condition).
4. **Early exit fills rather than terminates**: remaining masked positions after the first EOS are filled with EOS in-graph and the loop exits through the normal settle break next step (one extra forward vs a hard break — K-invariant and preserves the commit-cleanliness contract; a hard in-graph break would need new break plumbing for ~1 forward/generation).
5. **τ(t) applies to Γ only**; τ_edit static (§3 decision).

## 7. Mac Studio M2 Ultra Backfill (2026-07-14)

- **Telemetry Validity (`envValid`)**: Sourced from `scratch/llada_bench.jsonl`. Runs completed successfully with `envValid: true` (free memory > 98 GB, swap growth = 0 MB, totalMemoryMB output verified).
- **Wall-Clock Serving Throughput (dyntau/p3-combo vs baseline)**:
  - Dynamic τ (α=0.6) was evaluated as part of the `p3-combo` arm, achieving −14.8% (chat) and −11.3% (reasoning) step reduction. However, the MultiBD slot latency multiplier overrides this, yielding a net-negative wall-clock TPS (-12.3% to -20.4%).
  - Composition with Credit Decoding (`p4-combo`): adding Credit Decoding to MultiBD + Dynamic-τ recovers serving performance on chat, reaching **45.55 TPS** (only −5.5% behind baseline).
- **Suffix Pruning (2b-1) and EOS Early Exit (2b-3)**:
  - Suffix pruning is closed N/A.
  - EOS early exit is verified as a complete mechanistic null under clean environment testing.
- **Verdict (ACCEPTED for serving presets only, default off)** — ⚠️ **SUSPENDED 2026-07-17, see §8 (F10).** This accepted α=0.6 *through the `p3-combo` preset* (= nBuf=2 + α=0.6) on step/wall-clock evidence, while the quality gate named in §5 condition (a) was still open. That gate has now returned dirty on chat, and `p3-combo` is precisely the configuration with the highest measured edit-churn in the campaign (chat post-steps/blk 1.90) and **zero quality coverage** — neither this sheet (α=0.6 alone, 1.16) nor WP-1b's (nBuf=2 alone, 1.25) scored it.
  Dynamic τ (α=0.6) is accepted as a serving preset option (available within the `p3-combo` preset) for step-limited long-form generations, but is disabled by default for generic real-time serving.


## 8. Blind quality scores → α=0.6 fails its gate on chat (2026-07-17)

**Scores** (`blinds/wp2b_blind/`, 8 prompts, static vs α=0.6, scored against its own `key.json`): **static 2 wins, α=0.6 zero wins, 6 ties.**

**F8 — α=0.6 corrupts chat text; reasoning is clean. The split follows the step savings.** (sourced) Both decided prompts are chat, both go to static, and both show *local token corruption in the α=0.6 arm* — not a length or content preference:
- `chat-explain`: "Imagine the sky is like a big, **of,, and** Sunlight is like a flashlight…" (André: C:3 I:3 F:3, vs static 5/5).
- `chat-recipe`: broken header "Simple Pancakes for Two **2**", "20 g vegetable oil **( melted butter)**", and **1 tsp salt** where static gives a pinch (André: I:4 vs static 5/5/5).

Reasoning is **4/4 ties**. The asymmetry tracks the savings: chat is where α=0.6 cuts most (−14.8% vs reasoning's −11.3%). The count alone is *not* statistically resolvable (two non-ties both to baseline ⇒ p=0.25, sign test) — the weight here is that the defects are **mechanism-predicted and corroborated by an independent measured indicator** (F9), not that 2/8 is significant. Status per §0.1: smoke-level, directional, sufficient to block a default-on, insufficient to kill the lever.

**F9 — post-steps/block is a leading indicator of the defect, and F1 already described the mechanism without connecting it to quality.** (inferred from F1 + F8) F1 diagnosed α=0.9's non-monotonicity as "the near-zero late-block floor (τ0(1−α)) buys acceptances that Δ then has to repair (edit churn)". F8's corruption is that same mechanism where **Δ's repair came up short** — a cheap late acceptance that never got fixed. The arms rank in defect order by churn:

| arm | τ floor = τ0(1−α) | chat post-steps/blk | quality evidence |
|---|---|---|---|
| α=0.3 | 0.49 | 1.06 | never scored |
| **α=0.6** | 0.28 | 1.16 | **2/4 chat prompts corrupted** |
| α=0.9 | 0.07 | 1.62 | never scored (F1 flagged the churn) |
| **nbuf2 + α=0.6** | 0.28 | **1.90** | **never scored — and it is the recommended stack** |

Consequence for the τ_edit follow-up (§3, dropped on "post-steps/block 1.16 < ~3"): that entry condition was read as "churn is low, nothing to fix". F8 says churn at 1.16 is *already* leaving damage, so the threshold was calibrated against the wrong failure mode. Not reopened here, but the rationale is void.

**F10 — the churniest, least-scored configuration in the campaign is the one already accepted as a preset.** (inferred) F6's headline (nBuf=2 + α=0.6: chat TPF-logical +41%, "the two levers stack") is a *step-count* result. Its quality coverage is empty: WP-1b's sheet scored nBuf=2 alone (post/blk 1.25 — 7 ties/1 baseline win, and that loss was benign, "baseline gave more information"), this sheet scored α=0.6 alone (post/blk 1.16 — 2 corrupted). **Nobody has scored the combination at post/blk 1.90**, where both mechanisms compound. F6's "additive" claim is sourced for steps and **unevidenced for quality**.

The sharp edge: **that combination is `p3-combo`, and §7 already accepted α=0.6 as a serving preset *through* it** — on step/wall-clock evidence, with §5's quality condition (a) still open at the time. So the one configuration this campaign has shipped a preset for is the one with the most churn and the least quality evidence. §7's verdict is suspended accordingly; note also that §7 measured `p3-combo` as **net-negative wall-clock on the Studio anyway** (−12.3% to −20.4%, MultiBD's slot-latency multiplier overriding the step saving), so suspending it costs no measured throughput. This is also why final-plan Step 1 exists: α=0.6 has **never been measured standalone on the Studio** — only welded to MultiBD inside `p3-combo`/`p4-combo`.

**F11 — α was selected on speed alone; the sheet says the speed winner is past the quality knee on chat.** (inferred) §5's deciders were explicitly deterministic counters, with quality deferred to this sheet — so α=0.6 winning F1 was never a quality statement. α=0.3's numbers are already recorded and suggest the per-suite split writes itself: reasoning keeps α=0.6 (clean, −11.3%; α=0.3 would still bank −9.3% if a uniform α is ever wanted), chat drops to **α=0.3** (−6.5%, post/blk 1.06) or off. Per-suite α is the shape of the final plan's Step 2 deliverable anyway.

**F12 — methodology: the scripted checks are blind to this defect class.** (sourced) `checks.md` passed every row — `max4gramRepeat` 1, no mask leaks, comparable lengths (106 vs 91 words on `chat-explain`) — on text reading "a big, of,, and". 4-gram repetition and mask-leak heuristics detect *degeneration*, not *local incoherence*. The §5 verdict's "objective checks clean" is therefore retracted as evidence of non-degradation. An 8-prompt blind sheet is currently the only instrument that catches this class; treat quality sheets as on the critical path, not as a formality.

**Disposition**: `dynamicTauAlpha` stays landed, **default-off** (unchanged). The final plan's Step 1 sweeps `{static, α=0.3, α=0.6}` on the Studio so per-suite perf exists at the α the quality evidence supports; Step 2 must carry a quality sheet on the winning stack before any preset ships.

**Host caveat** `[Inferred]`: these are dev-M1 generations and trajectories are host-dependent (M1↔Studio step divergence is GPU floating-point). The specific corruptions will not reproduce verbatim on the Studio; the *mechanism* is a threshold-floor property, not a numerics one, so it should — and post-steps/block re-measures in Step 1 as the leading indicator.

**F13 — Studio standalone (2026-07-17 re-analysis of existing rows): α=0.6 is a REASONING-ONLY lever; chat's step saving reverses sign across hosts.** (sourced: `scratch/dyntau_factorial_0316.jsonl`, mlx-swift 0.31.6, commit `66494bd`; 3 runs × 12 prompts × gen-128, warmup + `envValid:false` rows excluded, 143/144 usable, thermal nominal, swap 0; engine echoes verified `dynamicTauAlpha 0.6, nBuf 1`)

| suite | Studio TPS Δ | Studio steps Δ | **M1 steps Δ (F1)** | Studio post/blk | sheet (§8) |
|---|---|---|---|---|---|
| chat | **+0.2%** | **+4.8%** | **−14.8%** | 1.35 | 2/4 corrupted |
| code | +3.5% | −8.0% | (not swept) | 1.65 | not scored |
| reasoning | **+13.8%** | **−13.6%** | −11.3% | 1.70 | 4/4 ties |

Three things follow, and they resolve §8's open questions rather than complicate them:

1. **Chat is dead on two independent axes.** F8's blind sheet (visible corruption) and this wall-clock/counter evidence (no gain: +0.2% TPS on +4.8% steps) were produced by different instruments on different hosts and agree. α=0.6 on chat buys nothing and costs text. The Studio churn is *higher* than the M1's (post/blk 1.35 vs 1.16) — same mechanism (F9), worse expression, on the host that also fails to save steps.
2. **F11's α=0.3-for-chat recommendation is undercut** (inferred): if the aggressive α makes Studio chat steps go *up*, the gentler one is unlikely to save much. The M1's α=0.3 chat −6.5% should not be expected to transfer any better than its α=0.6 −14.8% did (which reversed). Run the arm — it is free — but the honest prior is ~0.
3. **F1's headline was a dev-host artefact for chat, not for reasoning.** Reasoning transfers and slightly over-delivers (M1 −11.3% → Studio −13.6%). Chat does not transfer at all.

**F14 — counters are not host-independent, contra §0.1's working assumption.** (sourced, from F13) The §0.1 protocol has dev-host deterministic counters *decide* and treats wall-clock as host-scoped. A **sign reversal** on `logicalSteps` between M1 and Studio (−14.8% → +4.8%, same code, same α, same prompts) is a direct counterexample. It is consistent with the known M1↔Studio GPU-floating-point divergence, but the magnitude means **dev-host step counts cannot be treated as portable verdicts for threshold-sensitive levers** — a lever that decides acceptance by comparing a confidence against a moving threshold is precisely where small per-position FP differences compound into different trajectories. Counters gate what is *worth measuring* on the Studio; they do not decide it. Every dev-host "algorithmic ACCEPT" resting on counters alone inherits this caveat — including this logbook's own §5 verdict, which is hereby scoped to the dev host for chat.

**Revised disposition (supersedes §8's)**: `dynamicTauAlpha` stays landed, **default-off**. The surviving candidate is a **reasoning-only preset at α=0.6** (+13.8% TPS on the Studio, 4/4 quality ties on the dev sheet) — pending a *Studio* blind sheet at α=0.6 on reasoning, since the dev sheet does not transfer verbatim. Chat: closed negative. Code: +3.5%, never quality-scored — not worth a sheet at that size unless it moves.

## 9. Step-1 α sweep on the Studio, standalone + interleaved (2026-07-17)

**Run** (`scratch/refreeze_step1.jsonl`, one interleaved P2 run-major process, MLX core 0.31.1 / mlx-swift 0.31.6, host Mac14,14): `q-cached` × `dt-tau` (α=0.6) × `dt-tau-a03` (α=0.3), 3 runs × chat+reasoning+code × gen-128, warmup+`envValid:false` excluded (107/108 usable), variance gate PASS all arms (≤0.3%).

**P7 re-freeze cross-check**: q-cached TPS reproduces `dyntau_factorial_0316` within noise (chat +2.6%, reasoning +0.2%, code +0.1%) — the frozen 0.31.6 baseline holds; no re-anchoring needed.

| suite | α=0.3 ΔTPS (Δsteps, post/blk) | α=0.6 ΔTPS (Δsteps, post/blk) |
|---|---|---|
| chat | **+0.5%** (−1.9%, 1.18) | −0.1% (+4.8%, 1.35) |
| reasoning | +9.3% (−15.2%, 1.10) | **+14.1%** (−13.6%, 1.70) |
| code | **+11.2%** (−14.2%, 1.10) | +3.6% (−8.0%, 1.65) |

**F15 — the per-suite optimum is α-dependent, and it is NOT uniform: reasoning wants α=0.6, code wants α=0.3.** (sourced) α=0.6 reproduces F13 (reasoning +14.1% vs the earlier +13.8%; chat dead). But the new α=0.3 arm overturns two §1.5 assumptions: (i) **code jumps to +11.2% at α=0.3 from +3.6% at α=0.6** — code was called "not worth a sheet" in §1.4 on the strength of the α=0.6 number; at the α that actually suits it, it is the second-best cell in the whole sweep; (ii) reasoning keeps +9.3% at α=0.3 (65% of α=0.6's gain) at **less than half the churn** (post/blk 1.10 vs 1.70). Chat is dead at both α (α=0.3 +0.5%, α=0.6 −0.1%), closing that suite for good — F13's chat verdict survives the gentler α, as F-b predicted.

**F16 — α=0.3 is the systematically lower-churn point (post/blk 1.10–1.18 vs 1.35–1.70), i.e. the lower corruption risk per F9.** (sourced) This reframes the preset question. It is no longer "α=0.6 or off per suite"; it is a genuine quality/speed trade *within* the accepted suites: reasoning α=0.6 buys +4.8pp TPS (14.1 vs 9.3) at +0.6 post/blk of churn over α=0.3. Whether that edge is real quality-neutral throughput or churn that a sheet would catch is exactly what the Studio blind sheet must now decide.

**Revised blind-sheet scope (supersedes §8's "reasoning α=0.6 only")**: the Studio sheet should score **reasoning at α=0.3 and α=0.6** (to price the churn trade) **and code at α=0.3** (newly a live preset candidate, never scored). Chat needs no sheet — it is closed. This is the item André approved 2026-07-17.
