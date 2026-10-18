id: S-018
title: MAGE: All-[MASK] Block Already Knows Where to Look in Block Diffusion LLM
authors: Omin Kwon et al.
venue/id: arXiv:2602.14209 (v2), 2026-06-05
url: https://arxiv.org/abs/2602.14209
verified: 2026-10-18 by A-0 literature agent — abstract page fetched, ID ↔ title match
---
What it states (paraphrase; at most one short quote < 15 words): The first-step all-MASK block query already identifies the KV subset a block needs; one exact attention pass selects top-k indices reused for the block's remaining denoising steps.
Scope (models, setting): Block diffusion LLMs; LongBench, long context.
Relevance to parallelism erosion (what it does / does not measure — factual, no opinions about our work): Sparse KV selection, not quantization; no step-count analysis in the abstract.
