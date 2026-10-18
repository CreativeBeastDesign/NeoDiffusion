id: S-010
title: Attention Is All You Need for KV Cache in Diffusion LLMs
authors: Quan Nguyen-Tri et al.
venue/id: arXiv:2510.14973 (v2), 2025-12-28 (v1 2025-10-16)
url: https://arxiv.org/abs/2510.14973
verified: 2026-10-18 by A-0 literature agent — abstract page fetched, ID ↔ title match
---
What it states (paraphrase; at most one short quote < 15 words): Elastic-Cache decides when (attention drift test on the most-attended token) and where (from a chosen layer onward) to recompute KV in dLLMs, reusing shallow-layer and off-window MASK caches.
Scope (models, setting): LLaDA-Instruct, LLaDA-1.5, LLaDA-V; math and code.
Relevance to parallelism erosion (what it does / does not measure — factual, no opinions about our work): KV staleness/reuse, not quantization; no KV precision study.
