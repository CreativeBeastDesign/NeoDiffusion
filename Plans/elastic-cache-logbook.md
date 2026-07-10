# WP-1a Elastic-Cache Logbook — LLaDA2.1-mini, M1 dev host

**Status**: closed — **negative result** (Phase A rejected; Phases B/C not run, inherit the verdict)
**Scope**: first Elastic-Cache experiments (2026-07-10 afternoon, André) + same-evening forensic evaluation and salvage experiments (Claude). All runs on the dev M1 (MacBookPro17,1), 4-bit artefact, Q mode, chat suite, gen-128, K=4 CLI.
**Provenance note**: unlike the M6/M8 logbooks, this one was written **after** the experiments, reconstructed from git history, `scratch/llada_bench.jsonl` timestamps, and code archaeology. Commands marked *(reconstructed)* are inferred from JSONL fields, not recorded invocations. This gap is itself Finding F7.

## 0. Method in one paragraph

WP-1a Phase A (roadmap §1) was implemented as an `ActiveBlockCache` that stores the **active block's** per-layer K/V between denoising steps, plus the paper's most-attended-token drift test (σ_t^ℓ, threshold γ) deciding a depth boundary ℓ★: layers ≥ ℓ★ recompute active K/V, layers < ℓ★ reuse the previous step's. Three refresh policies were swept: dynamic per-step boundary ("Option 1", live readback), static boundary S ("Proposal A"), and dynamic boundary delayed to speculation-batch boundaries ("Proposal B"). Decision metrics per roadmap §0.1: hardware-independent counters (steps/block, forwards, syncs) decide; wall-clock is host-scoped and, this session, mostly env-invalid.

## 1. Starting point (sourced)

- M6 frozen dev baseline: chat ≈ 5–12 tok/s gen-128, steady forward ≈ 0.34 s, sync budget ≤1 blocking readback per K steps + 1 per commit (`Plans/m6-logbook.md`).
- Roadmap §1 scope note (pre-registered, 2026-07-04): "our engine already has ExactPrefixCache, which the CUDA baselines lacked — Elastic-Cache's headline numbers partly include savings we already banked in Phase 2. The honest measurement is marginal gain over M6."
- Source concept pages: `Resources/elastic-cache.md` (+ v2, atomic breakouts). Paper: "Attention Is All You Need for KV Cache in Diffusion LLMs", arXiv:2510.14973.

## 2. What was built (sourced: git 2026-07-10 14:56–16:11)

| Commit | Time | Content |
|---|---|---|
| 80dc6b1 | 14:56 | pre-Phase-3 baseline tag |
| 6aaf806…633cd69 | 14:57–14:58 | `ActiveBlockCache`/`LayerActiveCache`; `LLaDA2Attention` elastic overload (drift test); routing through layer/model |
| 9291113 | 14:58 | engine integration; **speculation force-scaled to K=1 when elastic enabled** (`effectiveSpecK = params.elasticCacheEnabled ? 1 : speculationK`) |
| 2e6e25c | 15:00 | bench CLI flags + JSONL fields |
| e39cac7 | 15:04 | `ActiveBlockCache` → DiffusionCore; parity tests (`testActiveBlockCacheParity` γ=2.0 token-identity gate, `testActiveBlockCacheRealDrift`) |
| 9be24b4 | 15:42 | "Option 1" stacked drift-similarity readback (batch the 20 per-layer σ evals) |
| 73fc6d8 | 15:51 | Proposal A (static boundary S) + Proposal B (delayed boundary at batch end, restores K=4); **removes the forced K=1** — Option 1 vs B now selected by engine `speculationK` |
| 6f2b9b0 | 16:06 | finalised sweep plumbing (`BoundaryHolder` public) |
| d2f2c47 | 16:11 | "future experiment directions" added to roadmap |
| 660a9a7 | 22:59 | **serving-path fix** (post-evaluation): `generateCached` bypasses the elastic overload when `elasticCacheEnabled == false` — before this, the served path paid the drift instrumentation unconditionally (see F5) |

## 3. Timeline (reconstructed from `scratch/llada_bench.jsonl`; local times CEST = JSONL UTC+2)

All arms *(reconstructed)*: `swift run -c release diffusion-bench llada --runs 1 --cooldown 0 --suites chat --gen-length 128 --arms q-cached [elastic flags]`.

| # | Local | Arm (elastic flags) | Binary | Effective K | Result / note |
|---|---|---|---|---|---|
| 1 | 15:07 | off (baseline for the sweep) | post-e39cac7 | 4 | steps/blk med 12.6, fwd 62, sync 18-ish, tps med 6.35 (3/4 rows env-invalid) |
| 2 | 15:09 | `--elastic-cache --elastic-gamma 0.9` (Option 1, dynamic) | same | **1** (forced; JSONL says 4 — see F6) | steps/blk med **29.8** (+136%), sync≈fwd (per-step readback) |
| 3 | 15:14 | `--elastic-gamma 0.98` run 1 | pre-9be24b4 | 1 | steps/blk med 19.2 (+52%); deterministic counters |
| 4 | 15:41, 15:46 | `--elastic-gamma 0.98` runs 2–3 | post-9be24b4 (stacked readback) | 1 | counters identical to #3; tps within noise of #3 → **stacked-readback opt: no measurable wall-clock gain** (all rows env-invalid) |
| 5 | 15:54–16:03 | `--elastic-static-boundary {4,8,12,16}` (Proposal A) | post-73fc6d8 | 4 | steps/blk med: S4 13.4, S8 22.0, S12 28.8, S16 30.6 — monotone in reuse depth; edits → 0 at S≥12 |
| 6 | 16:04 | `--elastic-gamma 0.98` delayed (Proposal B) | post-73fc6d8 | 4 | steps/blk med 23.6; sync back to ≈fwd/3.5 (batched) but more forwards than Option 1 at same γ |
| 7 | 22:30 | forensic evaluation (Claude): code + data analysis | — | — | Findings F1–F8 below; serving-path fix 660a9a7; parity suite re-run green (6/6) |

## 4. Findings

**F1 — Active-KV reuse inflates steps/block monotonically in reuse depth; the edit mechanism starves.** (sourced: timeline 2/5) Baseline 12.6 steps/blk → S4 13.4, S8 22.0, S12 28.8, S16 30.6, dynamic γ=0.9 29.8. `editsTotal` drops to 0 at S≥12 (stale K/V suppresses the confidence Δ depends on). Every elastic arm changes the output text on every prompt vs the off arm. Hardware-independent, deterministic across repeats → transfers to the Studio without re-running.

**F2 — The reuse branch saves ~nothing by construction.** (sourced: code, `LLaDA2Attention.swift` reuse branch) The fused `query_key_value` matmul (gotcha #3) runs in **both** branches — the reuse branch computes K,V and discards them; the MoE FFN runs for all active tokens at every layer regardless (next layer needs hidden states). Skipped work = key RMSNorm + RoPE-on-keys + two assignments. Meanwhile the drift test **adds** a second, unfused full attention computation per layer (GQA keys explicitly repeated 4×, fp32 softmax, full score matrix materialized) on top of the fused SDPA that still runs.

**F3 — Theoretical ceiling ≈ 4–5% even for a perfect implementation; ~1% at the only step-neutral operating point.** (inferred: FLOP arithmetic, LLaDA2.1-mini config) Per MoE layer per token: QKV ≈ 12.6 MFLOPs (K,V share 1/3 ≈ 4.2), o_proj ≈ 8.4, MoE 8×512-int experts + shared ≈ 57, router ≈ 1 → K,V projection ≈ 5% of layer compute; over the full stack incl. lm_head (≈ 644 MFLOPs/token, ~29% of total) the all-layer ceiling is ≈ 3.8%. The sweep's only near-step-neutral arm (S=4, +7% steps) could at best skip 4/20 layers → ≈ 1% — and the as-built implementation skips ~0% (F2). No γ/S/β tuning changes this arithmetic.

**F4 — The structural cause: block-causal LLaDA2.x + ExactPrefixCache already absorb both of the paper's savings tiers.** (inferred, load-bearing) The paper's speedup comes from (a) not recomputing *decoded* tokens' KV each step (refreshed on drift) and (b) block-caching *distant MASK* tokens (worth 30–40% of its throughput per its own ablation). In this engine: (a) the committed prefix is **exact by block-causality** — never stale, no refresh question exists; (b) the Block-Buffer loop never computes distant MASKs at all. The only cacheable object left is the active window itself — exactly what the paper always recomputes. Elastic-Cache's premise has no object here; the experiments tested the one thing the paper deliberately never caches, and confirmed why.

**F5 — Serving-path regression (fixed 660a9a7).** (sourced: code) `generateCached` routed through the elastic overload unconditionally; the drift test (second attention matrix + per-layer blocking `argMax().item()` readback ≈ 20 hidden syncs/forward) ran even with elastic off, invisible to the bench's `syncPoints` counter — the M7 server sat on this path. The sweep's "off" arm is therefore not the M6 baseline path wall-clock-wise (its counters match M6 exactly; the algorithm was unaffected).

**F6 — Fidelity deviations from the paper.** (sourced: code vs `Resources/elastic-cache.md` §Mechanism) (i) The drift test compares σ = cos(S_t[T_t], S_{t−1}[T_{t−1}]) — attention columns of **different tokens** whenever the argmax moves; the paper compares the **same** token T_{t−1} across steps. `previousMostAttendedIndex` is stored and never read. (ii) Proposal B reads the boundary from the **last speculative forward of the batch**, including discarded overshoot forwards; overshoot K/V also stays in `ActiveBlockCache` and is attended by the next batch's reuse layers (the speculation-vs-cache-commit gotcha, unhandled for the active tier). (iii) `elasticBeta` (sliding window, paper component 1) is parsed and recorded but **never implemented**.

**F7 — Provenance defects.** (sourced: JSONL vs git) (i) Option-1 rows record `speculationK: 4` (the CLI value) while the engine forced K=1 — recorded parameter contradicts effective behavior; detectable only because sync≈fwd. (ii) The Proposal-B arm is distinguishable from Option 1 only by timestamp + commit archaeology — no JSONL field identifies the policy. (iii) 34 of 40 elastic rows are `envValid: false` (free mem ~80–120 MB, GBs of swap) — per m6 rules no TPS conclusion survives; the verdict rests entirely on the deterministic counters. Rule going forward (adopted in WP-1b): **JSONL records effective values echoed from engine Metrics, never CLI inputs**, and every arm's command line goes in the logbook before the run.

**F8 — What held up.** (sourced) `testActiveBlockCacheParity` (γ=2.0 forced-recompute ⇒ token-identical to disabled) is the right correctness anchor and passes; the sweep design was clean one-variable work; the deterministic counters repeated exactly across the three γ=0.98 repeats (within-process determinism confirmed at K_eff=1).

## 5. Results summary

| Arm | steps/blk (med) | forwards (med) | edits (med) | verdict |
|---|---|---|---|---|
| off | 12.6 | 62 | 4 | reference |
| S=4 | 13.4 | 70 | 2 | +7% steps for ≤1% ceiling — reject |
| S=8 | 22.0 | 110 | 2 | reject |
| S=12 | 28.8 | 131 | 0 | reject (edits starved) |
| S=16 | 30.6 | 156 | 0 | reject |
| dyn γ=0.9 (Option 1) | 29.8 | 125 | 4 | reject |
| dyn γ=0.98 (Option 1) | 19.2 | 102 | 5 | reject |
| dyn γ=0.98 (Proposal B) | 23.6 | 120 | — | reject |

**Verdict: WP-1a Phase A REJECTED on the dev host under the §0.1 protocol — and the rejection is architectural, not a tuning failure.** Reuse of active-window KV has a ~4–5% theoretical ceiling (≈0% as implemented) against a measured step-inflation cost monotone in reuse depth. Phases B (token-stability signal) and C (combined signals) select *what to reuse in the active window* and inherit the same empty ceiling — **closed without running**. The roadmap's "Future Experiment Directions" (finer S sweeps, lighter similarity metrics, adaptive block scaling) are superseded: the first two cannot change the sign of a zero-ceiling optimisation; the third is a different lever (not Elastic-Cache).

**Salvage session (same night)**: see §6 for the salvage-experiment record (paper-faithful drift test) — run to distinguish "unfaithful signal" from "staleness is inherently costly".

**Surviving direction (design-only, not prototyped — per André's scope decision)**: per-*position* selective recompute. Only Γ/Δ-touched positions (~1–2/step) change tokens between steps; caching per-position layer outputs and recomputing only changed positions through attention+MoE has a real ceiling (MoE is per-token). That is d²Cache / vicinity-refresh territory (roadmap Tier 3) and needs its own proposal + drift-vs-quality analysis; the drift-test machinery built here (attention-column similarity) could be repurposed as its refresh trigger.

**Studio debts from this WP**: none — no wall-clock claim survives that would need backfilling; the counters transfer as-is.

## 6. Salvage experiments (Claude, overnight 2026-07-10/11)

*(filled in by the wp1a-salvage branch run — see git branch `wp1a-salvage`)*
