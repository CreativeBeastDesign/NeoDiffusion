# paper/ — Parallelism Erosion under KV-Cache Quantization in Block-Diffusion LMs

**Status**: PROPOSED LAYOUT (2026-10-18). Empty apart from this file; nothing is migrated until André
accepts the structure (see `Plans/next-phase-plan-2026-10-18.md` §5–6).

Working title: *Parallelism erosion: how KV-cache quantization slows confidence-threshold decoding in
block-diffusion language models.* "Confidence starvation" is the historical name used in the
2026-10-18 logbooks; it survives here only as an alias (see Glossary).

---

## 1. Why this structure

The goal is that **every sentence of the final write-up can be traced to raw data or to an external
source, and the trace only ever goes in one direction**. Interpretation never leaks into the record of
what was measured, and literature never stands in for a measurement.

This mirrors the hexagonal architecture of the engine: the record layers (`facts`, `observations`,
`sources`) are leaves that depend on nothing else in this folder, the way the engine's ports describe
the world without importing the application core; `interpretations` is the core that combines them; the
write-up in `drafts/` is the outermost layer.

```
                drafts/                     may cite anything
                   │
            interpretations/   I-*         cite F, O, S, other I
             │      │      │
   experiments/ E-* │      │               cite the I-* they test (before the run);
       │ produces   │      │               link to the R-*/F-* they produced (after)
       ▼            ▼      ▼
   facts/ F-*   observations/ O-*   sources/ S-*      ◄── leaves: cite NOTHING above them
     │   │            │
     │   runs/ R-*    │                    run manifests
     │   │            │
     └── data/ ◄──────┘                    immutable raw files (+ scripts/ that read them)
```

Dependency rules (enforced by `tools/lint_links.py`, to be written; CI-style, run before any commit
touching `paper/`):

| Layer | May link to | Must never contain |
|---|---|---|
| `sources/S-*` | external URLs only | anything from this repo; our numbers; our opinions of the paper beyond what it states |
| `facts/data/` | — | edits after ingestion (append-only; sha256 in `MANIFEST.sha256`) |
| `facts/runs/R-*` | `facts/data/`, git commits | interpretation |
| `facts/F-*` | `facts/data/`, `facts/runs/`, `scripts/` | words like *because, suggests, shows that, cures*; references to S/O/I |
| `observations/O-*` | `facts/data/` (raw text outputs, scored sheets) | numbers not in the linked raw file; causes |
| `experiments/E-*` | `interpretations/` (what it tests), then `facts/runs/` (results) | edits to the pre-registered part after the run started |
| `interpretations/I-*` | F, O, S, I | numbers not quoted from an F-* |
| `drafts/` | anything | numbers not traceable to an F-* |

## 2. Layout

```
paper/
  README.md                this file: rules, glossary, index
  sources/
    S-001-kivi.md          one external work per file
  facts/
    data/                  raw JSONL / text dumps / tensors, copied from scratch/, never edited
      MANIFEST.sha256      sha256  path  — one line per file
    runs/
      R-001.md             one per bench invocation
    F-001.md               one measured statement per file
  observations/
    O-001.md               one qualitative reading per file (blind sheet result, text inspection)
  experiments/
    E-001.md               pre-registration + result links
  interpretations/
    I-001.md               one claim per file
  scripts/                 deterministic analysis; stdlib-only python where possible
  drafts/
    post.md                the write-up
  tools/
    lint_links.py          enforces §1
```

## 3. File templates

### `sources/S-NNN-<slug>.md`
```
id: S-NNN
title: <exact title>
authors: <…>
venue/id: arXiv:XXXX.XXXXX (vN), date
url: <…>
verified: <date> by <who> — abstract page fetched, ID ↔ title match
---
What it states (paraphrase; ≤ 1 short quote with page/section):
Scope (models, setting — e.g. "video diffusion, not dLLMs"):
```

### `facts/runs/R-NNN.md`
```
id: R-NNN
command: <exact command line>
git: <commit> dirty=<n> diffSHA256=<…>          (from row toolchain fields)
host: <hw.model>, RAM, OS build
toolchain: mlxSwiftPackage / mlxCoreVersion / swiftCompiler
model artefact: <path> + sha256 of weights index
data: facts/data/<file> sha256=<…>
rows: total / warmup / envValid=false (listed with reason)
arm order: interleaved | blocked
```

### `facts/F-NNN.md`
```
id: F-NNN
statement: <one sentence, numbers only, no causal words>
value(s): <table>
derived by: scripts/<x>.py <args>  → output excerpt
inputs: R-NNN (rows used: valid only / all)
denominators: <n prompts, n tokens>
```

### `observations/O-NNN.md`
```
id: O-NNN
what was looked at: facts/data/<file>, rows <…>
procedure: blind (who scored, key withheld until <date>) | unblinded inspection
observation: <what was seen, verbatim excerpts ≤ 2 lines each>
```

### `experiments/E-NNN.md`
```
id: E-NNN
tests: I-NNN (hypothesis H…)
registered: <date, git commit of this file>
prediction: <which F would refute it, with thresholds>
arms / prompts / n / metrics / exclusion rules / analysis script
--- (below this line only after the run)
runs: R-…   facts: F-…   outcome: supported | refuted | inconclusive
deviations from registration: <none | list>
```

### `interpretations/I-NNN.md`
```
id: I-NNN
claim: <one sentence>
status: conjecture | supported | refuted | superseded-by I-…
rests on: F-…, O-…, S-…
falsifier: <what observation would kill it>
alternatives considered: <…>
```

## 4. Integrity rules ("paranoid")

1. **Append-only record.** F/O/R/S files are never edited after creation except to add a
   `superseded-by:` line. A corrected number is a new F with a link from the old one.
2. **Hashes, not paths.** Every F cites data by sha256; `MANIFEST.sha256` is checked by the linter.
3. **Pre-registration is committed before the run** (git timestamp is the proof).
4. **No hand-copied numbers.** Numbers in F come from a script's printed output; numbers in I and drafts
   are quoted from F. The linter flags numbers in I/drafts that appear in no F.
5. **Valid rows only**, and every exclusion is listed by row in R.
6. **Blind before reading.** Quality sheets are scored before the arm key is opened; O records who scored and when.
7. **Negative and inconclusive results are first-class.** A refuted I stays, marked refuted.
8. **Provenance tags carry over**: F = sourced, I = inferred, conjecture = speculative.

## 5. Glossary

| Term | Meaning | Status |
|---|---|---|
| Parallelism erosion | fewer tokens committed per denoising step (more steps/block) when the KV cache is perturbed | observed (needs n ≥ 100 replication, E-planned) |
| Confidence deflation | proposed mechanism: per-position max-probability shifts below τ_mask | hypothesis |
| Silent divergence | argmax flips at unchanged confidence: different text, no step cost | hypothesis |
| Confidence starvation | historical alias (2026-10-18 logbooks); its metric "1-transfer steps" is 49.6 % even in the FP16 baseline, so do not use it as a pathology measure | deprecated |
| int8 / int4 KV | affine integer group quantization (scale + bias per group); logged as `fp8` / `fp4` in older rows | — |

## 6. Migration plan (after the layout is accepted)

1. Copy `scratch/track1_kv_starvation_k1.jsonl` (+ text dump) and the campaign files into `facts/data/`,
   write `MANIFEST.sha256`, write R-001…R-005 from their rows.
2. Derive F-* from `Tools/analysis/cache_campaign_aggregate.py` (moved/duplicated into `scripts/`).
3. Write S-* for the verified literature (Elastic-Cache, KIVI, MaskAhead, MAGE, REFLEX, DES, TEAM,
   DyLLM, DARE, FlashBlock, "Quantized Keys Steal Attention" — scoped as video diffusion).
4. Turn the Track-1 mechanism claims into I-* with status `conjecture`, linked to the A-1…A-6
   pre-registrations in `experiments/`.
