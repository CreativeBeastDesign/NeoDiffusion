id: S-017
title: DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention
authors: Younjoo Lee et al.
venue/id: arXiv:2603.08026 (v2), 2026-05-30; ICML 2026
url: https://arxiv.org/abs/2603.08026
verified: 2026-10-18 by A-0 literature agent — abstract page fetched, ID ↔ title match
---
What it states (paraphrase; at most one short quote < 15 words): Recomputes only salient tokens (detected by attention-context change between adjacent steps) and reuses cached activations for the rest.
Scope (models, setting): LLaDA and Dream; reasoning and code.
Relevance to parallelism erosion (what it does / does not measure — factual, no opinions about our work): Approximate reuse, not quantization; abstract does not mention thresholds or step counts.
