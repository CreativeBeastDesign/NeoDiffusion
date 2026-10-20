# Batched-prefill blind sheet — what it decides

Generated 2026-10-20. Studio outputs (Mac14,14, mlx core 0.31.1 / mlx-swift 0.31.6), git 7e0952a (dirty 0),
served 4-bit artefact, uncompressed KV cache (`cache-clean-commit-k1`), gen 512, K=1, temperature 0,
`.build/release/diffusion-bench --suites longctx_blind --batched-prefill {1,128}`. All 24 rows envValid v2;
effective `batchedPrefillBlocks` echoed as set (`rows-off.jsonl`, `rows-c128.jsonl`).

**One A/B comparison**: batched prefill at 128 blocks per forward vs per-block prefill (the current default).
Each prompt's two outputs are shuffled per prompt (seeded, deterministic) and withheld in `key.json` —
do not open it while scoring.

| what a verdict means | |
|---|---|
| ties / no consistent loser | batched prefill can become the default (E-012 already found no step or GSM8K-accuracy shift, F-034/F-035; 2.4× faster generations at ~2.85k-token prompts) |
| batched outputs consistently worse | it stays opt-in; the trajectory change is not benign on open-ended text |

**Why these prompts**: the standard chat/reasoning/code prompts are 15–44 tokens (≤ 2 prompt blocks), where
batched prefill does almost nothing. These 12 (`Tools/make_longctx_blind_suite.py`) carry 1.3–3k tokens of
real context — this repo's own docs, fact files and source — so you can check factual consistency yourself.
The task is shown first; the context is collapsed under each prompt.

**Known before scoring** (not a hint at the A/B assignment):
- byte-identical outputs on 1 of 12 prompts: lc-reason-gsm1300 — a genuine tie.
- gold answers for the two GSM8K prompts: `lc-reason-gsm1300` = 120, `lc-reason-gsm1301` = 23.
- several outputs stop at the 512-token limit (see `checks.md`, eosBlock `none`) — judge what is there, a cut-off is not a defect of either arm unless only one is cut.
- `checks.md` heuristics (4-gram repetition, mask leaks) catch degeneration only, **not local incoherence** (F12) — the manual read is the gate.

Scoring, once `sheet.md` is filled:
```
python3 Tools/m8_blind_sheet.py --pairs blinds/bp_blind/pairs.json --out blinds/bp_blind --score blinds/bp_blind/sheet.md
```
Per the standing rule, the `batchedPrefillBlocks` default does not flip until this is scored.
