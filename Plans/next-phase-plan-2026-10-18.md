# Next-Phase Plan — Research Track & Engine Track (2026-10-18)

**Status**: PROPOSAL — awaiting André's decisions in §6. Nothing here has been run.
**Author**: Claude (Opus 5.5) session of 2026-10-18, after a review of the 2026-10-17/18 cache campaign.
**Supersedes nothing.** `Plans/final_optimisations_plans.md` remains the plan of record for optimisation
work; this document adds a research track (Goal A) and reorders the cache-campaign follow-ups (Goal B).
Every claim below carries a provenance tag (`[Sourced: …]` / `[Inferred]` / `[Speculative]`).

---

## 0. Where we stand (what the review found)

The cache campaign produced one solid engineering result and one interesting scientific observation.
It also had defects that this session fixed or recorded (errata live in each affected document, marked
`E-n`; the code changes are listed in §0.2).

### 0.1 Results that survive the review

| Result | Status | Evidence |
|---|---|---|
| Clean-commit skip (`skipRedundantCommitForward`) on the plain path: one forward saved per block, output token-identical | **Holds.** Aggregate gain **+9.0 % (K=4) / +9.7 % (K=1)**, not the +13.0 / +15.1 % quoted (those were mean-of-per-prompt ratios) | `[Sourced: Tools/analysis/cache_campaign_aggregate.py over scratch/cache_campaign{1_k4,2_k1}.jsonl]` |
| int4 per-token Keys inflate denoising steps; int8 Keys do not (2×2 factorial, equal bytes) | **Holds as an observation** at n = 12 prompts, gen 64, 4-bit weights, one host | `[Sourced: scratch/track1_kv_starvation_k1.jsonl, 96/96 envValid]` |
| KIVI-style per-channel int4 Keys remove the step inflation | **Holds as an observation**, same caveats; parity 58.7 % vs 73.0 % for k8v4, so it is a trade, not a cure | same file |
| KV quantization makes forwards faster | **False.** ms/forward is 189–191 for every KV arm at K=1; TPS differences come from changed trajectories | `[Sourced: same script, ms/fwd column]` |
| Studio-class K=1 default; M1 keeps K=4 | **Contradicted on M1**: at K=4 the M1 evaluates 374 forwards vs 337 at K=1 for the same 291 logical steps, at the same ms/forward | `[Sourced: cache_campaign1_k4 q-cached vs cache_campaign2_k1 spk-1; cross-process → needs a same-process confirmation]` |

Further errata found by the doc pass (full list: `Plans/cache-optimisation-logbook.md` § Errata, E-1…E-17):
`scratch/cache_k8v4_clean_k1.jsonl` is **0/12 envValid** (the logbook claimed 12/12); the 14.3 % attention /
56.3 % MoE forward budget was measured on the **Studio** (WP-6c) yet is used to explain M1 results; the
"shared expert ≈ 11.7 %" and "dense layer 0 + norms ≈ 26.4 %" shares are subtraction remainders (WP-6c
measured the shared expert at 4.2 %); and the presets described as "shipped in `main`" were never committed.
With aggregate TPS, k8v4 and kivi4 at K=1 come out *faster* than FP16 (+2.0 / +3.2 %) — trajectory effects
of the same order as the noise, opposite in sign to the mean-of-ratios figures.

### 0.2 Code fixes made 2026-10-18 (uncommitted, on branch `kernel`)

1. **Clean-commit gate is now an allowlist** (`GenerationParams.cleanCommitSkipEligible`). Before, it did not
   exclude CSA2, dual caching, JOT, SureLock, FlashBlock, Elastic-Cache, router reuse, MoE capacity, or ICE.
   On a CSA2 reuse step the pending K/V is the *previous* step's (`LLaDA2Attention` CSA2 branch), so a
   settle on that step committed stale K/V — possibly computed over `[MASK]` tokens — into the
   **exact** prefix cache. Arms `cache-csa2-periodic-k4`, `cache-dual-opt`, `cache-dual-k8v4` are
   therefore **confounded**. The bench/engine echo `effectiveSkipRedundantCommitForward` now reports
   eligibility, not the request. Two regression tests added (`DenoisingLoopParityTests`).
2. **`dual-k8v4` preset no longer sets `dualCachingEnabled`** (which silently enabled FlashBlock, interval 2,
   unlike the measured arm). The preset is still **unmeasured as defined** — see §3, B-0.
3. **Presets carry `qualityGateOpen`**; the server refuses trajectory-changing presets and any KV
   quantization unless started with `--allow-experimental-cache` (standing rule 1). Unknown preset /
   kv-quant strings are now errors in server and bench (previously silently ignored → silently FP16).
4. **`kivi4` + misaligned append** now fails loudly (precondition) instead of concatenating two layouts;
   `kivi4` + sub-block commit is rejected up front. Dead, wrong `compress(_:mode:)` (applied key bits to
   values) removed.
5. **`int8`/`int4`/`int4g32` aliases** (`KVCacheQuantization.parse`); `fp8`/`fp4` documented as affine int.
6. **Bench `envValid` laundering removed**: the anonymous-memory page-touch before baseline sampling is now
   opt-in (`--reclaim-pages`), recorded per row (`reclaimPerformed`, pre-reclaim swap/free), and swap
   growth is measured from the earliest baseline.
7. **Bench rows now record source provenance**: `gitCommit`, `gitDirtyFileCount`, `gitDiffSHA256`
   (the 2026-10-17 campaign ran on an uncommitted tree — its rows cannot be tied to exact code).
8. **`.gitignore`**: `models/` matched `App/Diffusion/Models/` on case-insensitive APFS, so the macOS app's
   five model-layer source files (`Settings.swift`, `ConnectionTarget.swift`, …) **were never in git**.
   Anchored to `/models/`. Those files now show as untracked — André should review and commit them.
9. App flag descriptors: KV quantization relabelled `.experimental` with the gate explained.

Still open from the review, deliberately not done here: the bench runs arms **blocked** (all prompts of
arm 1, then arm 2 …), which exposes arm comparisons to thermal/memory drift; a prompt-major interleave
mode is item P-3 in §4.

### 0.3 Baseline test run after the clean-up (2026-10-18, 15:06–15:16 CEST)

`[Sourced: scratch/2026-10-18/baseline-full-suite.log; gate log /tmp/neodiffusion-compute-gate.log]`
Host `MacBookPro17,1`, tree = HEAD `1ff0053` + 22 modified files (diff sha256 prefix `f830cbfb2b92`),
`swift test` under `Tools/compute-gate.sh` (waited 0 s, ran 627 s, rc 0). Per bundle:
`DiffusionKernelsTests` 1 test / 0 failures; `DiffusionGenerationTests` **97 tests, 11 skipped, 0 failures**
(602 s); `DiffusionCoreTests` **58 tests, 13 skipped, 0 failures**. Total 156 executed, 24 skipped (all
opt-in benches / real-weight / dumpers — list in the log), 0 failures. The three clean-commit tests pass:
`testSkipRedundantCommitForwardParity` (37.9 s), `testCleanCommitSkipIsIneligibleUnderCSA2` (18.4 s),
`testCleanCommitSkipEligibilityAllowlist`. (CLAUDE.md's "120 tests" count predates later additions.)
This is the reference for regression checks in Exp-5B.

---

## 1. Two goals, one discipline

- **Goal A — research contribution**: characterise how KV-cache quantization interacts with
  threshold-based parallel decoding in block-diffusion LMs. Output: a public write-up (blog post or
  arXiv note), built from a `paper/` knowledge base with strict fact/interpretation separation.
- **Goal B — engine**: an optimised block-diffusion inference engine on Apple Silicon, developed in a
  loop *find → implement → experiment → evaluate*.

They share infrastructure (the bench, its provenance, the prompt suites) but have **different success
criteria**: Goal A succeeds when a claim is *well-supported or cleanly refuted*, whichever way it falls;
Goal B succeeds when a change *clears its gates*. A negative result is a full success for A and a recorded
negative for B. Do not let a Goal-A finding become a Goal-B preset without passing Goal-B gates.

---

## 2. Goal A — the phenomenon and how to study it

### 2.1 Naming

"Confidence starvation" is evocative but has two problems: (1) the metric it was attached to — steps
with exactly one transfer — is **49.6 % of steps in the unquantized baseline** (it is the normal top-1
fallback, not a pathology) `[Sourced: track1 table, cache-clean-commit-k1 row]`; (2) it names a mechanism
(confidence goes down) that is not yet shown — what is shown is the *consequence* (more steps).

Recommended terms (name the observable, keep the mechanism as a hypothesis):

| Term | Refers to | Status |
|---|---|---|
| **Parallelism erosion** (recommended headline term) | KV perturbation lowers the number of tokens committed per denoising step → more steps per block | Observed (steps 291 → 333 for int4 KV) |
| **Confidence deflation** | Proposed mechanism: per-position max-probability drops below the commit threshold τ | Hypothesis (H2) |
| **Silent divergence** | Argmax flips at unchanged or higher confidence → different text, no step penalty | Hypothesis (H3); suggested by int8's 70 % parity with no step inflation |

Alternatives considered: "threshold-crossing deficit" (precise but clumsy), "commit-rate collapse"
("collapse" overstates a +14 % effect).

### 2.2 Novelty check first (A-0, no GPU)

Before claiming a gap, search for prior work on KV quantization *in diffusion LMs with confidence-threshold
decoding*. Known adjacent work: MaskAhead / Q-MaskAhead (arXiv:2610.06996, eviction + low-bit KV in block
diffusion), KIVI (arXiv:2402.02750, AR models), "Quantized Keys Steal Attention" (arXiv:2605.26266, **video**
diffusion). Also post-training-quantization studies of dLLMs (weights/activations). Deliverable:
`paper/sources/` entries + a one-page "what is known" note. **If someone already reports step inflation
from KV quantization under threshold decoding, the contribution narrows to the mechanism and the fix.**

**A-0 result (2026-10-18, literature agent; record: `paper/sources/S-001…S-018`, synthesis
`paper/drafts/A-0-related-work.md`, lint clean)** `[Sourced: those files; full text read for S-001/2/4/5, abstracts only for the rest]`:
(a) no verified paper reports step count / tokens-per-step / NFE under KV quantization with threshold
decoding — every dLLM KV-quant paper found reports task accuracy only; SoloQ (arXiv:2610.07121) notices the
trajectory changes but fixes the forward count for its latency comparison; "Quantization Meets dLLMs" lists
step-count interaction as future work → **the measured effect looks novel**. (b) K-vs-V asymmetry is already
reported *for accuracy* (Q-MaskAhead K/V-bit ablation incl. LLaDA-2.0-mini) → new only as a step-count effect
with a mechanism. (c) Per-channel (KIVI-style) keys are standard in dLLM KV-quant work → new only as evidence
that it removes step inflation; no paper found quantizing keys before the qk-norm gain. Closest prior work:
Q-MaskAhead, SoloQ, FAIR-Calib (margins/decision flips, but for W4A4, not KV).
Caveat: a step table could exist in bodies only skimmed via abstracts; re-check before publishing.

### 2.3 Hypotheses (pre-registered before any new run)

- **H1 (effect)**: At fixed bytes, int4 per-token Keys increase logical steps per block relative to FP16
  KV; int4 Values do not. *Falsified if* on ≥ 100 prompts the K4V8 step increase is not larger than
  the K8V4 one, with a paired per-prompt test.
- **H2 (mechanism: deflation)**: The step increase is caused by fewer positions crossing τ_mask per step,
  i.e. the distribution of per-position max-probability shifts down. *Test*: log per-step max-prob and
  margin at masked positions; compare distributions.
- **H3 (second channel: silent divergence)**: Token disagreement with FP16 arises mostly from argmax flips
  at positions whose confidence is *not* reduced. *Test*: same logging, classify each first-divergence
  position.
- **H4 (structure)**: Key outlier channels are largely the learned per-channel gain of `key_layernorm`
  (qk-norm is active), so quantizing keys **before** that gain (gain re-applied after dequant) — or
  per-channel as KIVI does — removes most of the effect. *Test*: real-key statistics (per-channel range
  before/after the gain), then an arm. Note RoPE here is partial (dims 0–63), so a pre-RoPE argument
  covers only half the channels.
- **H5 (weights confound)**: The effect is not an artefact of the 4-bit expert weights. *Test*: repeat the
  core arms with BF16 weights on the Studio.

### 2.4 Measurement rules specific to Goal A

- **Quality, not parity**: token match vs FP16 measures trajectory difference, not quality. Primary quality
  metric: task accuracy on `gsm8k_100` (exists, bench `--baseline-check`) plus blind-scored chat/code.
- **n**: ≥ 100 prompts for any quantitative claim; report per-prompt paired deltas with a bootstrap CI.
  At temperature 0 with deterministic kernels, "seeds" do not add samples — prompts do. (The revisit
  note's "3 seeds" only applies if temperature > 0, which changes the question.)
- **Aggregate TPS** (Σtokens/Σseconds), never mean of per-prompt ratios. Steps and forwards are the
  primary metrics anyway — they are hardware-independent; wall-clock is secondary for Goal A.
- **Operator benchmarks on real tensors only** — the synthetic Regime B disagreed with end-to-end
  rankings, so synthetic results are illustrations, not evidence.

### 2.5 Experiment ladder (each a separate pre-registration in `paper/experiments/`)

| ID | What | Host / cost | Needs |
|---|---|---|---|
| A-0 | Literature / novelty check | none | — |
| A-1 | Re-run the 2×2 + kivi4 on `gsm8k_100` + chat/reasoning/code, K=1, clean-commit, prompt-major interleave | M1, GPU ≈ 6 arms × ~35 min ≈ 3.5 h (one process, overnight) `[Inferred from ~0.19 s/forward]` | P-3 interleave |
| A-2 | Per-step confidence/margin logging (H2, H3) on a 30-prompt subset | M1, GPU ≈ 1 h | small engine instrumentation (opt-in, off the hot path) |
| A-3 | Real-key statistics: per-channel ranges, effect of `key_layernorm` gain, RoPE halves (H4) | M1, light GPU (one forward per prompt, dump K) | key dump hook |
| A-4 | Pre-gain int4 Keys arm (H4) | M1, ≈ 35 min | A-3 result |
| A-5 | BF16-weight control of A-1 core arms (H5) | Studio | Studio access |
| A-6 | MoE routing drift under KV noise (revisit Exp 4) | M1, ≈ 1 h | router-index dump |

### 2.6 What the write-up would claim (only if the data allow)

"In confidence-threshold block-diffusion decoding, KV-cache quantization error in Keys reduces decoding
parallelism (parallelism erosion), which can cost more wall-clock than the memory saving buys; the effect
is Key-specific, mechanistically explained by X, and removed by Y." Every clause maps to an
interpretation that cites facts. If H2 fails, the paper says so.

---

## 3. Goal B — engine backlog, re-ordered

Loop protocol per item: **(1) find** — write the hypothesis and expected effect with provenance;
**(2) implement** default-off behind a flag with an effective-echo field; **(3) experiment** —
pre-registered arms, same process, interleaved, valid rows only; **(4) evaluate** against gates —
toy FP32 parity → (where output-invariant) token identity on 4-bit → blind quality sheet if the
trajectory changes → Studio wall-clock. Record negatives in the logbook. Never promote a default while
any gate is open.

| Order | Item | Why this position | GPU on M1? |
|---|---|---|---|
| **B-NT** (new 2026-10-18, correctness — ahead of everything) | **Latent non-termination of the denoising loop**: nothing prevents x0 = `mask_id`, so a block can stay masked forever; `max_post_steps` only counts mask-free iterations and there is no global cap. Reproduced on the base engine with the toy model in S mode (`DenoisingLoopParityTests.testReproSModeNonTermination`, opt-in, hangs today; evidence `Plans/exp5b-opus-vs-fable-logbook.md` §9.4). The reference `generate()` shares the flaw, so any fix is a **deviation from the reference** (phase-2 §5 list) — André's call: (a) parity-neutral hard cap on steps/block (only fires on a would-be hang; log it), (b) suppress `mask_id` (and arguably pad) logits in x0, or both. Recommendation: (a) now, (b) evaluated separately because it changes outputs | none to implement; repro is toy-only |
| B-0 | Re-run the CSA2/dual arms with the fixed gate (or retire `dual-k8v4`) | Three published numbers are confounded; cheapest is to retire the preset, which has no use case at ≤ 4k context | ≈ 20 min if re-run |
| B-1 | Same-process `spk-1` vs `spk-4` on M1 at gen 128, 3 suites | Output-invariant (no quality gate); data suggest K=4 wastes ~11 % forwards on M1 → possible free default change | ≈ 30 min |
| B-2 | **Session prefix cache** (cross-turn reuse of committed 32-token blocks) | Largest product value for chat; developable on FP32 toy config. Corrections to the handoff: ~~reuse is exact *mathematically*, not bit-exact on 4-bit (prefill tile shapes differ)~~ **[self-correction 2026-10-18: prefill commits block by block with the same 32-token capture forward, so reuse should be bit-exact on 4-bit too — the tile-shape caveat applies to the uncached path only; checked by the Exp-5B hidden real-model test]**; re-tokenized history and end-of-turn tokens usually end reuse at the previous assistant reply; "80–95 % TTFT" is untested | dev: tests only |
| B-3 | Prefix-length sweep (gen 512/1024) for FP16 vs k8v4 vs kivi4 | Decides whether KV compression has *any* engine value at the in-scope ≤ 4k context (FP16 KV at 4k ≈ 160 MiB vs 9.5 GB weights) — shared with Goal A | ≈ 1–2 h |
| B-4 | Step-adaptive expert top-k (REFLEX-style probe) | Cheapest sync-free MoE lever; measure ms/forward on a fixed trajectory first. REFLEX itself reports ~15 % expert-compute saving, not the 50 % in the handoff. Needs a blind sheet | ≈ 30 min probe |
| — | Deprioritised: DES coreset (Step 4c / WP-6f measured coalescing *slower*), FFN-output reuse (every active-block reuse so far inflated steps), long-context prefix selection (no long-context eval suite; small ceiling at ≤ 4k), Sumi Elastic-Cache (Studio, separate model) | | |

---

## 4. Paranoia standards (apply to both goals)

Already in place: `envValid`, warmup exclusion, effective echoes, toolchain versions; added today: git
commit + diff hash per row, opt-in recorded reclaim.

To add (P-items, small):
- **P-1 Pre-registration**: hypotheses, arms, metrics and gates written and committed *before* the run;
  results appended, never edited in. A changed analysis after seeing data is labelled post-hoc.
- **P-2 Raw data is evidence**: `scratch/` is git-ignored, so the raw JSONL behind every published number
  is currently unversioned. Copy each cited file into `paper/facts/data/` (or a tracked `evidence/`)
  with its SHA-256; facts cite the hash.
- **P-3 Prompt-major interleaving** in the bench (`--interleave`): run prompt p on every arm before p+1,
  rotating arm order, so drift spreads evenly across arms.
- **P-4 One analysis script per table**, committed; documents quote its output, never hand-copied numbers.
- **P-5 Commit before measuring**: a run on a dirty tree is allowed for exploration, but numbers that go
  into a logbook verdict must come from a committed tree (the row now proves which).
- **P-6 Errata are append-only**: correct with strike-through + errata entry, never silent rewrite.

---

## 5. Proposed `paper/` layout (details in `paper/README.md`)

A small knowledge base with one-way dependencies, so a reader can trace every sentence of the write-up
down to raw data or an external source:

```
paper/
  README.md            rules + dependency graph + glossary
  sources/      S-*    external literature only; cites nothing internal
  facts/        F-*    measured numbers; cite only facts/data + facts/runs + scripts
    data/              immutable raw files (sha256 in MANIFEST.sha256)
    runs/       R-*    one manifest per run: command, git commit/diff hash, host, toolchain, row counts
  observations/ O-*    qualitative readings (blind sheets, text inspection); cite only raw data
  experiments/  E-*    pre-registrations; may cite I-* (what they test); results link forward to F-*
  interpretations/ I-* claims; cite F/O/S/I only; carry status + falsifier
  scripts/             deterministic analysis producing facts
  drafts/              the post / paper; may cite anything
  tools/lint_links.py  enforces the dependency rules
```

---

## 6. Decisions for André

1. **Commit the clean-up?** Recommendation: yes, as one commit after the test run, then commit the 5
   previously-ignored App files separately after you review them.
2. **Name**: "parallelism erosion" (observable) with "confidence deflation" / "silent divergence" as
   hypotheses? Recommendation: yes; keep "confidence starvation" only as a historical alias.
3. **`paper/` layout**: accept as proposed, or prefer a flatter structure?
4. **Order of work**: recommendation — A-0 and P-1…P-4 first (no GPU), then B-1 (30 min GPU), then A-1
   overnight. B-2 (session cache) can be developed in parallel because it needs no GPU until its final check.
5. **`dual-k8v4`**: retire (recommended) or re-measure?
