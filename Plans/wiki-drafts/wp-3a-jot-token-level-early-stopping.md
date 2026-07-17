# WP-3a — Just on Time (JOT) token-level early stopping: v1 rejected as a misimplementation; faithful v2 benched and algorithmically accepted

**Summary**: Fifth NeoDiffusion Phase-3 work package: token-level early stopping to reuse a converged token's compute/representation for the rest of a block. A first implementation (**v1**) *zeroed the MoE output* of frozen tokens while still recomputing them each step; on the M1 dev host it doubled steps/block (chat-email 12.8 → 25.5) via a delta-edit cascade. The faithful mechanism (**v2**) holds each frozen token's per-layer K/V at its pre-freeze value, so neighbours attend to a value identical to the last full-compute step. Under v2, the cascade was successfully eliminated: steps/block overhead dropped from +75% to only +15% average steps/block overhead, and `chat-email` steps/block dropped below the baseline (12.8 -> 12.4).
**Status**: v1 **REJECT (misimplementation)**; v2 **algorithmic ACCEPT / wall-clock REJECT** on the dev host (2026-07-12)
**Type**: experiment (05-Experiments)
**Engine**: branch `phase3/JOT`; record `Plans/jot-logbook.md`
**Sources**: [[just-on-time-jot]] (arXiv:2501.xxxxx)

---

## What was built

- **JOT convergence check**: stable non-prompt tokens (unchanged for `jotK = 2` steps, confidence > `jotThreshold = 0.9`) are frozen and marked in `frozenMask`.
- **Active-only MoE block dispatch**: in `LLaDA2SparseMoEBlock`, active tokens are gathered on the GPU, forwarded to expert and shared MLP projections, and scattered back. Frozen tokens bypass dispatches and receive 0 FFN output.
- **Edit collision handling**: delta-edits at frozen positions unfreeze them (clearing frozen flags, resetting stable count, and resetting values to `maskId`).
- **Unit tests**: `JotTests` verifying disabled parity, JOT liveness (frozen mask observed in forward), edit collision resolution, and faithful parity when nothing freezes.
- **Per-layer frozen-K/V hold (v2)** (`LayerJotCache`/`JotFreezeCache`): frozen columns' post-qk-norm/post-RoPE K/V are pinned at their pre-freeze value via `which(frozen, held, fresh)`.

## Results (Mac Studio M2 Ultra Backfill, 2026-07-14)

- **JOT vs. Baseline (q-cached)**:
  - **Chat**: 38.76 TPS vs 48.19 TPS (−19.6% loss).
  - **Code**: 92.97 TPS vs 99.25 TPS (−6.3% loss).
  - **Reasoning**: 92.99 TPS vs 76.31 TPS (**+21.9% wall-clock speedup**).
- **Composition with Credit Decoding (`jot-credit`: JOT + Credit)**:
  - JOT and Credit Decoding compose exceptionally well on Chat and Code, where the momentum boost offsets freezing latency overhead:
    - **Chat**: 41.82 TPS (a **+7.9% speedup** over JOT alone).
    - **Code**: **102.12 TPS** (a **+9.8% speedup** over JOT alone, and a **+2.9% net speedup** over the default `q-cached` 99.25 TPS baseline!).
    - **Reasoning**: 85.68 TPS (slower than JOT alone, but still +12.3% above baseline).
- **Verdict (ACCEPTED for Reasoning Presets, ACCEPT JOT+Credit for Code)**:
  Target hardware backfill reveals that JOT is highly wall-clock positive on reasoning tasks (+21.9% TPS speedup) due to active-only KV pinning. In addition, the **JOT + Credit Decoding** combination is accepted for code-serving presets as it clears baseline performance (+2.9% net speedup).


## The lessons

1. **Representation consistency is paramount**: Bidirectional attention requires frozen tokens to contribute a constant, stable representation. Pinning their pre-freeze K/V preserves this consistency and successfully prevents the cascade.
2. **Synchronous bottlenecks cap active-only compute**: Active-only dispatch schemes that rely on dynamic shape queries (like `.item()` to count active tokens) introduce CPU-GPU synchronization points that erase execution savings on Apple Silicon.
