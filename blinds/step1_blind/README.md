# Step-1 dyn-τ blind sheet — what it decides

Generated 2026-07-17 (`final_optimisations_plans.md` §1.5 F-f, `wp2b-logbook` §9). Studio
outputs (Mac14,14, MLX core 0.31.1), gen-128, temp 0, `.build/release/diffusion-bench`.

Two A/B comparisons, one sheet. Each prompt's two outputs are the two arms below, shuffled
and withheld in `key.json` (do not open while scoring):

| prompts | comparison | what a verdict means |
|---|---|---|
| `reason-*` (4) | dyn-τ **α=0.3 vs α=0.6** | α=0.6 is +14.1% TPS but higher churn (post/blk 1.70); α=0.3 is +9.3% at 1.10. If α=0.6 is quality-clean here, it wins reasoning; if it shows corruption vs α=0.3, reasoning drops to α=0.3. |
| `code-*` (4) | **static (α=0) vs α=0.3** | code α=0.3 is +11.2% TPS and never quality-scored. This confirms the winner doesn't corrupt vs the trusted baseline. |

**Known before scoring** (not a hint at the A/B assignment): on `code`, α=0.3 produced
*byte-identical* text to static on 2 of 4 prompts (`code-fizzbuzz`, `code-regex`) — those
are genuine ties (a free speedup, no text change). Only `code-sql`/`code-swift-struct` and
all four `reason-*` differ enough to judge.

Scoring, once `sheet.md` is filled:
```
python3 Tools/m8_blind_sheet.py --pairs blinds/step1_blind/pairs.json \
  --out blinds/step1_blind --score blinds/step1_blind/sheet.md
```
The pooled summary reports wins by label (`rea-a0.3` / `rea-a0.6` / `cod-static` / `cod-a0.3`),
so the two comparisons stay separated. **Per the standing rule, no dyn-τ default or preset
ships until this is scored** — and remember (F12) the scripted `checks.md` catches degeneration,
not local incoherence, so the manual read is the gate.
