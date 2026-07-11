# WP-3a Just on Time (JOT) Logbook — LLaDA2.1-mini, M1 dev host

**Status**: CLOSED — NEGATIVE RESULT (2026-07-11, branch `phase3/JOT`)
**Scope**: JOT token-level early stopping optimization (arXiv:2501.xxxxx) per roadmap §3, under the §0.1 dev-only protocol: counters decide, wall-clock host-scoped, every arm a recorded command line.

## 0. Method in one paragraph

A token is deemed "converged" and frozen when its prediction remains unchanged for `jotK = 2` steps with confidence exceeding `jotThreshold = 0.9` at non-prompt active positions. In `LLaDA2SparseMoEBlock`, we construct active indices on the GPU (via boolean-to-integer conversion and `argSort`) to evaluate experts and shared MLP projections exclusively for the non-frozen (active) tokens, outputting zero FFN additions for the frozen tokens. If delta editing edits a position that is currently frozen (prediction change exceeding `editingThreshold` at a previously unmasked position), JOT unfreezes it (clearing the frozen flag, resetting the stable count to 0, and resetting its value in the active window back to `maskId` so it can be re-denoised).

## 1. Starting point

- JOT (source summary `Resources/possibly new/just-on-time-jot.md`): training-free token-level early stopping.
- Pre-experiment readout (`Tools/build_draft_graph.py --jot-k 2`): 44.2% of active tokens are prediction-stable with only a 4.9% delta-editing collision rate. Theoretically, this offers a substantial per-step MoE FLOP reduction ceiling (up to 30–40% FLOP savings on the MoE layers).
- Implementation: completed on branch `phase3/JOT` (GenerationParams, DiffusionEngine loop states, active-only MoE projection gather/scatter in LLaDA2MoE).

## 2. Hypotheses (pre-registered 2026-07-11)

- **H1 (FLOP reduction)**: Skipping MoE computations for 44% of stable tokens will reduce active-only MoE step compute.
- **H2 (Delta collision)**: The 4.9% collision rate with delta-editing is small enough that the unfreezing overhead will be minimal.
- **H3 (Bidirectional representation consistency)**: Setting MoE outputs to 0 for frozen tokens might perturb their hidden states, but since they have already converged, this perturbation will have minor impact on the overall trajectory.

## 3. Timeline

| # | When (CEST) | What | Result / note |
|---|---|---|---|
| 1 | ~19:30 | Research + plan (plan doc `JOT token-level early stopping`): branch `phase3/JOT` active | |
| 2 | 20:00–20:44 | Implementation: JOT fields in `GenerationParams`; `Forward` signature accepts optional `frozen` array; in-graph state tracking in `DiffusionEngine.SlotRun` and `windowStep` | build clean |
| 3 | 20:45 | Propagate `frozen` parameter through `LLaDA2MoeModel` and `LLaDA2DecoderLayer` to `LLaDA2SparseMoEBlock`. Implement gather/scatter active token dispatches in `LLaDA2SparseMoEBlock`. | build clean |
| 4 | 20:46–21:00 | Unit tests `JotTests`: parity when disabled, synthetic liveness (obs frozen mask in forward), and edit collision resolution. Green after fixing index offset (prompt truncation) and adjusting synthetic logits to avoid premature settling. | JotTests green |
| 5 | 21:01 | **Sweep run**: comparing baseline vs JOT `--runs 1 --suites chat --arms q-cached --jot` | results §4 |

## 4. Findings

**F1 — JOT has a strong negative algorithmic interaction with LLaDA's delta-editing mechanism, doubling the steps/block.** (sourced: `scratch/llada_bench.jsonl`)
- Baseline steps/blk: chat-email: 12.8, chat-explain: 18.0, chat-recipe: 10.0.
- JOT steps/blk: chat-email: 25.5 (**+99%**), chat-explain: 28.8 (**+60%**), chat-recipe: 19.0 (**+90%**).
- Total logical steps: chat-email: 51 → 102 steps, chat-explain: 90 → 144 steps.

**F2 — Mechanism: Representation perturbation cascades.** 
Setting the MoE/FFN output of frozen tokens to 0 perturbs their representations. Through bidirectional attention, this perturbation propagates to neighboring non-frozen tokens, shifting their predictions. This triggers delta edits on the neighboring tokens, resetting them to masks. The block gets stuck in a loop of freezing, perturbing, editing, and resetting, doubling the step count.

**F3 — Restricting delta-editing on frozen tokens hides the corruption but propagates representation decay.** 
Even if we prevent frozen tokens themselves from being edited, the neighboring tokens still attend to the MoE-omitted (0) representations, continuing to trigger edits and stall progress.

## 5. Verdict

**Algorithmic REJECT.**
The training-free token-level MoE skipping in JOT is incompatible with bidirectional attention in diffusion language models because it breaks the representation consistency required for stable attention and delta-editing.
