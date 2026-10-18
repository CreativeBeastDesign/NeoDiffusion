id: S-013
title: KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache
authors: Zirui Liu et al.
venue/id: arXiv:2402.02750 (v2), 2024-07-25; ICML 2024
url: https://arxiv.org/abs/2402.02750
verified: 2026-10-18 by A-0 literature agent — abstract page fetched, ID ↔ title match
---
What it states (paraphrase; at most one short quote < 15 words): From a study of KV element distributions, keys should be quantized per-channel and values per-token; builds a tuning-free asymmetric 2-bit KV quantizer on that basis.
Scope (models, setting): Autoregressive LLMs (Llama, Falcon, Mistral).
Relevance to parallelism erosion (what it does / does not measure — factual, no opinions about our work): Origin of the per-channel-key / per-token-value scheme; AR only, no diffusion or threshold decoding.
