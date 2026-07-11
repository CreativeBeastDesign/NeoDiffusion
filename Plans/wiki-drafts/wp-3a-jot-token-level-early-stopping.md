# WP-3a — Just on Time (JOT) token-level early stopping: REJECTED due to negative interaction with delta-editing

**Summary**: Fifth NeoDiffusion Phase-3 work package: token-level early stopping to bypass MoE layer compute for stabilized tokens. Measured on the M1 dev host, JOT has a strong negative interaction with the delta-editing mechanism, doubling the steps per block (chat-email steps/blk 12.8 → 25.5, chat-explain steps/blk 18.0 → 28.8). Bypassing MoE calculations by setting expert outputs to 0 perturbs hidden states, which propagates via bidirectional attention to neighboring non-frozen tokens. This shifts their predictions, triggering cascades of delta-edits that reset tokens to masks and stall block settlement. Disabling delta editing on frozen tokens hides the corruption but fails to stop neighboring cascades or representation decay. 
**Status**: closed on the dev host (2026-07-11) as an **algorithmic REJECT**
**Type**: experiment (05-Experiments)
**Engine**: branch `phase3/JOT`; record `Plans/jot-logbook.md`
**Sources**: [[just-on-time-jot]] (arXiv:2501.xxxxx)

---

## What was built

- **JOT convergence check**: stable non-prompt tokens (unchanged for `jotK = 2` steps, confidence > `jotThreshold = 0.9`) are frozen and marked in `frozenMask`.
- **Active-only MoE block dispatch**: in `LLaDA2SparseMoEBlock`, active tokens are gathered on the GPU, forwarded to expert and shared MLP projections, and scattered back. Frozen tokens bypass dispatches and receive 0 FFN output.
- **Edit collision handling**: delta-edits at frozen positions unfreeze them (clearing frozen flags, resetting stable count, and resetting values to `maskId`).
- **Unit tests**: `JotTests` verifying disabled parity, synthetic JOT liveness (frozen mask observed in forward), and edit collision resolution.

## The findings

1. **Steps per block doubled**: logical step count for `q-cached` on the chat suite doubled (e.g. chat-email: 51 → 102 steps, steps/blk 12.8 → 25.5).
2. **Representation perturbation cascades**: setting MoE outputs to 0 for frozen tokens perturbs their representations. This perturbation propagates through bidirectional attention, causing neighboring non-frozen tokens to shift predictions and trigger delta-edits (token resets to masks), preventing the block from settling.
3. **Restricting delta edits fails**: preventing frozen tokens themselves from being edited hides the prediction shifts on frozen tokens but does not prevent neighboring tokens from getting corrupted and triggering edits.

## The JOT lesson

Bidirectional attention in diffusion language models requires strict representation consistency. Unlike autoregressive generation where committed tokens are cached once and for all, active window tokens in diffusion are continuously evaluated. Training-free token-level MoE skipping breaks this representation consistency, rendering JOT incompatible with bidirectional delta-editing.
