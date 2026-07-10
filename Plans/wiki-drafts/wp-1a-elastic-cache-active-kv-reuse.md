# WP-1a — Elastic-Cache active-KV reuse on LLaDA2.1-mini (negative result)

**Summary**: First NeoDiffusion Phase-3 work package: Elastic-Cache-style depth-selective reuse of the *active block's* KV between denoising steps. Rejected — the savings tiers that give [[elastic-cache]] its speedup are structurally absorbed by NeoDiffusion's Phase-2 architecture, leaving a ~4–5% theoretical ceiling against a measured step-inflation cost of up to 2.4×.
**Status**: closed (negative result, 2026-07-10)
**Type**: experiment (05-Experiments)
**Engine**: NeoDiffusion @ `wp1a` commits 6aaf806…6f2b9b0; record: `Plans/elastic-cache-logbook.md` in the repo
**Host**: dev M1 (hardware-independent counters decide; wall-clock rows mostly env-invalid)
**Sources**: [[elastic-cache]], [[elastic-cache-v2]], [[elastic-cache-metal-kernel]] (proposal), [[depth-aware-refresh]], [[most-attended-drift]], [[attention-aware-drift-test]]

---

## Hypothesis

[[elastic-cache-metal-kernel]] proposed porting Elastic-Cache's pipeline (sliding window β, most-attended-token drift test γ, depth boundary ℓ★ recompute-deep/reuse-shallow) against NeoDiffusion's `ActiveBlockCache`. Expected: "far less than the paper's 45.1×" since ExactPrefixCache already eliminates prefix recompute; the honest measurement is marginal gain over the frozen M6 baseline.

## What was actually testable

The port surfaced a scope collapse that the proposal's caveat only gestured at:

1. **"When to refresh decoded tokens' KV" has no object here.** LLaDA2.x is block-causal; committed blocks' KV is *exact* forever ([[exact-prefix-cache]] semantics). The paper's drift-triggered refresh exists because its models (LLaDA-1.5/V, fully bidirectional) have decoded-token KV that goes stale. Ours cannot.
2. **"Block-caching distant MASKs" has no object either.** The Block-Buffer loop never computes tokens beyond the active block(s); the paper's own ablation credits 30–40% of its throughput to skipping distant MASKs — work NeoDiffusion never does in the first place.
3. What remains cacheable is only the **active window itself** — the tokens that change every step, which the paper *always recomputes*. The experiment therefore tested a reuse the source method deliberately avoids.

## Results (chat suite, gen-128, Q mode, 4-bit; deterministic counters)

| Arm | steps/block (median, baseline 12.6) | edits |
|---|---|---|
| static boundary S=4 (reuse 4 shallow layers) | 13.4 (+7%) | halved |
| S=8 | 22.0 | halved |
| S=12 / S=16 | 28.8 / 30.6 | **0** (edit signal starved) |
| dynamic γ=0.9 / 0.98 | 29.8 / 19.2 | ≈baseline |

Step inflation is monotone in reuse depth; every arm changed output text on every prompt. Compute-side, the implementation skips ~0% by construction (fused QKV runs in both branches; MoE runs for all active tokens at every layer), and the perfect-implementation ceiling is ≈ 4–5% of total FLOPs (K,V projection share) — see logbook F2/F3 for the arithmetic.

## Verdict

**Rejected — architecturally, not for lack of tuning.** No γ/S/β setting can make a ≤5%-ceiling optimisation pay for a monotone step-inflation cost. Phases B/C (token-stability signal, combined signals) select *what to reuse within the active window* and inherit the same empty ceiling: closed without running. A salvage experiment with the paper-faithful drift signal (same-token σ, see logbook §6) tested whether signal infidelity caused the inflation.

## What transfers / follow-ups

- **Negative-result generalisation** (inferred): adaptive-refresh KV methods designed for fully-bidirectional dLLMs ([[elastic-cache]], [[dkv-cache]], [[fast-dllm]] Dual Cache) lose their object on block-causal engines with an exact prefix tier. Check the attention regime before porting any of this family.
- **Surviving direction**: per-*position* selective recompute (only Γ/Δ-touched positions change per step; MoE is per-token, so skipping unchanged positions has a real ceiling) — [[d2cache]] / [[vicinity-kv-cache-refresh]] territory, Tier 3 in the roadmap; the attention-column-similarity machinery built here could serve as its refresh trigger. Needs its own proposal.
- **Engine hygiene fixed in passing**: the drift instrumentation had leaked onto the serving path (double attention + ~20 hidden blocking readbacks per forward, invisible to the sync audit) — fixed on `main` (660a9a7).
- **Provenance rule adopted**: bench JSONL must record *effective* parameters echoed from the engine, never CLI inputs (an Option-1 sweep recorded K=4 while the engine forced K=1).
