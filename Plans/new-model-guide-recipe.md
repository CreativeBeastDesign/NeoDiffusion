# Recipe: Implementation & Optimisation Guide for a New Diffusion LLM

**Purpose of this file.** This is a **recipe**, not a guide. It tells an agent (Claude Code, Perplexity Computer, a future you) *how to produce* an Implementation & Optimisation Guide for a new masked-diffusion or block-diffusion LLM that André wants to bring up inside the **NeoDiffusion** engine on Apple Silicon (MLX + Metal).

The output of this recipe is a document with the same shape and rigour as `Plans/lmoe-implementation-guide.md`, but for a different model.

**Assumed baseline.** LLaDA2.1-mini in NeoDiffusion has been unblocked and works end-to-end (M4/M5 BF16 parity gate passed, M6+ landed). The new model is added *on top of* a working multi-model engine.

**Style rule (non-negotiable).** André's user preference: do **not** support ideas unconditionally. The guide must critically reflect the model choice, flag speculative claims, and give honest feedback on which optimisations actually apply. Every load-bearing statement must be labelled **sourced** / **inferred** / **speculative**. This is not a stylistic choice, it is a correctness discipline.

---

## 0. Goal, target, non-goals

### 0.1 Goal
Produce a single markdown file, `Plans/<model-slug>-implementation-guide.md`, that lets André (or Claude Code operating on his behalf) implement and optimise the new model inside NeoDiffusion **without re-reading the source papers**. Everything load-bearing must be in the guide.

### 0.2 Target reader
André, and Claude Code running inside `./`. Assume both know Swift/MLX, Metal, and diffusion LLMs at a working level. Do not re-explain what masked diffusion is. Do explain what is different about *this* model.

### 0.3 Non-goals
- Not a research paper summary. If the paper's contribution isn't observable in the released weights or reference code, omit it.
- Not a tutorial on MLX or Metal. Sketches, not tutorials.
- Not a promise. If something is uncertain, say so and mark it speculative.

### 0.4 Definition of done
The guide is done when:
1. All five "hard facts" (§4 below) are answered from primary sources with links.
2. The refactor plan lands with all existing NeoDiffusion tests still green at each step.
3. Every Resources/*.md optimisation is classified as *applies / applies with rework / doesn't apply*, with a one-line reason each.
4. Both an MLX sketch and a Metal sketch exist for the model's forward pass.
5. Memory footprint on M1 16GB is computed for BF16 and 4-bit. Memory footprint on M2 Ultra 192GB is computed for full precision.
6. A copy-pasteable kick-off objective for Claude Code exists at the end.
7. An "uncertainty flags" section lists every speculative claim and the source that would resolve it.

---

## 1. Inputs the agent needs from André

Before starting, confirm:
1. **Model identity**: Hugging Face repo URL (e.g. `https://huggingface.co/<org>/<model>`). No repo → stop and ask.
2. **Slug for filenames**: e.g. `lmoe`, `dream`, `mmada`. Lowercase, hyphen-safe.
3. **Current NeoDiffusion state**: which milestone is done? Which sampler(s) already exist? Assume LLaDA2.1-mini works unless told otherwise.
4. **Any paper URL** (arXiv) — optional but useful for the followup section.
5. **Any known constraints** — e.g. "I want to skip Metal for now", "I care about latency not throughput", "budget one week".

If any of these are missing and the request is not a followup on this same recipe, ask before writing. Otherwise use sensible defaults and flag them.

---

## 2. Sources to read, in order

The agent should read these files in the sandbox before writing a single line. If they are not in the sandbox, pull them from the Mac with `pc pull ./<path>`.

### 2.1 NeoDiffusion state (read all)
- `Plans/CLAUDE.md` — engine invariants, style rules, milestone status.
- `Plans/handoff-post-M5.md` (or the latest handoff) — what actually shipped.
- `Plans/phase-1-conceptual-design.md` — the mental model of the engine.
- `Plans/phase-2-implementation-guide.md` — component contracts.
- `Plans/phase-3-optimisation-roadmap.md` — what optimisations are already landed vs planned.
- `Plans/lmoe-implementation-guide.md` — **the reference output shape**. The new guide should look like this one.

### 2.2 Existing Swift sources (read at least these)
Under `Sources/NeoDiffusion/` (adjust to the current layout):
- The current top-level model file (post-M5, likely `LLaDA2MoeModel.swift` or a generic `DiffusionModel.swift`).
- `LLaDA2Attention.swift` — attention module.
- `LLaDA2DecoderLayer.swift` — decoder block.
- `LLaDA2MoE.swift` — routed MoE block.
- `LLaDA2MoeConfig.swift` — config type.
- The sampler(s) currently registered under `SamplingStrategy`.
- The `ExactPrefixCache` implementation.
- The tokenizer / prompt-formatter / mask-scheduler layer if present.

### 2.3 Resources notes (read every one)
Under `Resources/`, read all `res_*.md`. As of this recipe, that includes:
alpha-moe-megakernel (+ source), block-causal-attn, block-diffusion, adaptive-kv, layer-kv, sel-layer, elastic-cache (v1, v2), mask-caching, fp8, m2t, t2t, llada2-tech, step-imp, depth, hierarchical, sliding.

Each note becomes one row in §7's applicability matrix. Do **not** skip a note because it "obviously" doesn't apply — say so explicitly with a one-line reason.

### 2.4 The new model's own artefacts (primary sources)
From the model's Hugging Face repo, fetch and read:
1. `config.json` — architecture ground truth.
2. `configuration_*.py` — reveals default flags and any non-standard config fields.
3. `modeling_*.py` — **the single most important file**. This is where attention masking, sampling, routing, and RoPE variant are actually defined. Read it end-to-end.
4. `generation_config.json` — sampling defaults.
5. `README.md` / model card — sampling example code, prompt format, mask id, eos id.
6. `tokenizer_config.json` — special tokens, chat template.
7. Any `*.py` inference helper the repo ships (some repos ship a `dinfer`-style script).

Store the raw URLs; the guide cites them inline.

### 2.5 The paper — read *last*, and selectively
Only skim for: (a) what problem the loss/router/attention solves that isn't visible from `modeling_*.py`, (b) any training-time behaviour that changes inference (e.g. logit-scaling factors, temperature norms). If the paper contradicts the released code, **trust the code** and note the discrepancy.

### 2.6 Web search — this is where this recipe differs from the LMOE guide
The LMOE guide was written with a fixed Resources/ set. For a *new* model, also search for optimisations that emerged after the Resources/ notes were written.

Search queries to run (adapt model name):
- `<model-name> inference optimization`
- `<model-name> KV cache diffusion`
- `<model-name> quantization Apple Silicon`
- `masked diffusion LLM inference speedup 2026` (adjust year)
- `block diffusion sampling <model-name>`
- `<model-name> MLX` and `<model-name> Metal` (someone may already have a port)
- `<model-name> dInfer` if it's an inclusionAI model
- Sibling models from the same lab (they often share optimisations).

**Hugging Face URLs** — `fetch_url` on the `raw/main/` variants of config/modeling files works reliably.

For every optimisation surfaced by web search:
- Add a row to §7 with source URL.
- Classify honestly (applies / rework / no). "Someone tweeted it works" is not evidence.
- If it depends on a paper without released code, mark **speculative** and don't build the guide around it.

---

## 3. Output file location and naming

- Write to `/home/user/workspace/<slug>-implementation-guide.md` first.
- After it validates (§8), push to `./Plans/<slug>-implementation-guide.md` with `pc push`.
- Share the sandbox copy with `share_file` (name argument: `<slug>-implementation-guide`).

---

## 4. The five hard facts every guide must answer

These are the questions that make or break a masked-diffusion port. Answer each in §1 of the output guide, from primary sources, with URLs.

1. **Attention masking** — Is attention causal, bidirectional, block-causal, or block-diffusion? Look at `is_causal`, `attention_mask` construction, and any custom mask builder in `modeling_*.py`. **This determines whether ExactPrefixCache is correct as-is.**
2. **Sampling algorithm** — Draft-and-edit (LLaDA2.x), scheduled remask (LLaDA-1.x / LMOE), block-by-block (block diffusion), or something new? Extract the exact `generate()` function into the guide as a pinned reference.
3. **Router shape (if MoE)** — Softmax or sigmoid? Groups? Expert bias? `norm_topk_prob`? Scaling factor? Shared expert? These directly determine whether the Alpha-MoE megakernel applies.
4. **Attention shape** — Fused vs split QKV? Full vs partial RoPE (`partial_rotary_factor`)? MHA vs GQA (`num_attention_heads` vs `num_key_value_heads`)? `qk_layernorm`? These determine which Metal kernels port.
5. **Tokenizer & special tokens** — vocab size, mask id, eos id, pad id, chat template. If it matches an existing NeoDiffusion tokenizer, the whole tokenizer/formatter layer is free.

Every answer gets a **sourced** tag with the URL and line reference.

---

## 5. Required section structure of the output guide

The output guide must follow this outline. It mirrors `lmoe-implementation-guide.md`. Sections can be short but cannot be missing.

```
§0  Recommended reading order (5–10 min path through the guide)
§1  Quick facts (the five hard facts, tabulated, with URLs)
§2  Pinned reference sampling algorithm
    - The exact `generate()` verbatim from modeling_*.py
    - Annotate each line: which NeoDiffusion component owns it
§3  Multi-model integration plan
    §3.1  What already exists in NeoDiffusion after LLaDA2.1-mini
    §3.2  Refactor steps (numbered, each keeps tests green)
    §3.3  New types to add (Config, Model, Attention, Block, Sampler)
    §3.4  Correctness decisions André must make before code lands
          (e.g. cache invalidation under a new mask shape)
§4  MLX sketches
    §4.1  Attention module (with the model's specific RoPE / QKV / GQA shape)
    §4.2  MoE block if applicable (router + experts, no shared unless present)
    §4.3  Decoder layer wiring
    §4.4  Forward pass and sampler entry point
§5  Metal sketches
    §5.1  Which existing Metal kernels port unchanged
    §5.2  Which need re-derivation, with reasons
    §5.3  What NOT to write yet (deferred to a later milestone)
§6  Web-search additions
    §6.1  Optimisations not in Resources/ that apply to this model
    §6.2  Sibling-model tricks worth stealing
    §6.3  Known-bad advice found online (call it out)
§7  Optimisation applicability matrix
    - One row per Resources/*.md note
    - Plus one row per §6 addition
    - Columns: Optimisation | Verdict | Reason | Cost | Priority
    - Verdicts: Applies / Applies with rework / Doesn't apply
§8  Memory & quantisation logistics on M1 16GB
    - BF16 weights + activations
    - 4-bit and 8-bit variants with math shown
    - Peak footprint under the model's sampler
§9  Milestones (name them <slug>-M1, <slug>-M2, …)
    - Ordered so each milestone is independently shippable
    - Each has an acceptance criterion tied to a test
§10 Risks & open questions
§11 Kick-off objective for Claude Code
    - Copy-pasteable
    - Names the exact first file to touch
    - Names the exact first test to keep green
§12 Uncertainty flags
    - Every "inferred" or "speculative" claim in the guide
    - What source would resolve it
```

---

## 6. Style discipline

Copy these rules verbatim from the LMOE guide:

1. **Label every load-bearing claim**: **sourced** (URL provided), **inferred** (derived from a source but not stated), **speculative** (no direct source, argue it).
2. **Deviate explicitly**. If the guide recommends something different from the model's reference implementation, say so and justify.
3. **"Gotchas that will bite" section** — at least three concrete traps specific to *this* model, not generic ones.
4. **Honest opinion required** — if an optimisation is fashionable but doesn't buy anything on M1 16GB, say so. If the model is a bad fit for the current engine, say so.
5. **No cheerleading**. "This is elegant" is not evidence. Either it makes tests faster or smaller, or cut it.
6. **URLs, not "the paper says"**. Every claim traces to a file the reader can open.

---

## 7. Web-search discipline

Because this recipe adds web search to the process, the agent must:

1. **Prefer primary sources** — Hugging Face repo, GitHub repo, arXiv PDF. Blog posts are secondary. Tweets are tertiary and must be corroborated.
2. **Prefer `fetch_url`** for Hugging Face `raw/main/*` files and for arXiv abstracts. `browser_task` is a last resort for auth-gated pages, not for public docs.
3. **Timebox** — cap web search at ~20 queries total. If nothing new turns up after that, the model likely has no post-Resources/ optimisations worth adding.
4. **Log negative results** — if a search finds nothing useful, note it in §6. Absence of evidence is useful evidence.

---

## 8. Self-check before delivering

Before pushing to the Mac and calling `share_file`, the agent must verify:

- [ ] All five hard facts (§4) answered with URLs.
- [ ] §3.2 refactor sequence names the exact Swift files touched at each step.
- [ ] §3.4 lists at least one correctness decision (or explicitly states "none — the model fits the existing invariants").
- [ ] MLX sketches compile in the reader's head: no undefined variables, right tensor shapes.
- [ ] Metal §5.3 explicitly lists what is *deferred*, not just what is done.
- [ ] Every Resources/*.md note appears in §7.
- [ ] §8 memory numbers are computed, not guessed. Show the arithmetic.
- [ ] §11 kick-off objective is under 300 words and names one file + one test.
- [ ] §12 uncertainty flags is non-empty. If it's empty, the guide is either dishonest or the model is trivially similar to an existing one (unlikely).

If any check fails, iterate before sharing.

---

## 9. What to hand back to André after delivery

A short message with:
1. The three most surprising findings from primary sources (bidirectional attention, no shared expert, whatever the specific model reveals).
2. Any §3.4 correctness decision that blocks Claude Code from starting.
3. One-line memory verdict for M1 16GB.
4. Pointer to §11.

Do not summarise the whole guide — the guide is the deliverable, not the message.

---

## 10. Meta: when to update this recipe

Update this file if:
- A future guide surfaces a section the outline in §5 missed.
- A new Resources/*.md note lands and should always be in the matrix.
- The engine's invariants change in a way that changes the refactor pattern in §3.
- The self-check list in §8 catches recurring omissions — add them.

The recipe is a living document. The LMOE guide was v1 of this shape; each new model is a chance to tighten it.
