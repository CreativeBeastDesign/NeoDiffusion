id: S-005
title: Quantization Meets dLLMs: A Systematic Study of Post-training Quantization for Diffusion LLMs
authors: Haokun Lin et al.
venue/id: arXiv:2508.14896 (v3), 2026-03-13 (v1 2025-08-20)
url: https://arxiv.org/abs/2508.14896
verified: 2026-10-18 by A-0 literature agent — abstract page fetched, ID ↔ title match
---
What it states (paraphrase; at most one short quote < 15 words): First systematic PTQ study of dLLMs; finds activation outliers in LLaDA and Dream and compares weight-only and weight-activation PTQ methods across bit-width, method, task and model; rotation-based methods outperform SmoothQuant. Full text: KV cache not quantized (caching described as orthogonal); decoding uses low-confidence remasking with steps equal to generation length. Lists as future work how the number of generation steps interacts with quantization level.
Scope (models, setting): LLaDA-Base, LLaDA-Instruct, Dream; weights and activations.
Relevance to parallelism erosion (what it does / does not measure — factual, no opinions about our work): Does not quantize KV, does not use threshold decoding, and does not measure steps or confidence; explicitly names steps-vs-quantization as an open question.
