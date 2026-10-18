id: S-007
title: Quant-dLLM: Post-Training Extreme Low-Bit Quantization for Diffusion Large Language Models
authors: Tianao Zhang et al.
venue/id: arXiv:2510.03274 (v1), 2025-09-27; ICLR 2026
url: https://arxiv.org/abs/2510.03274
verified: 2026-10-18 by A-0 literature agent — abstract page fetched, ID ↔ title match
---
What it states (paraphrase; at most one short quote < 15 words): 2-bit weight PTQ for dLLMs using masked calibration simulation, a data-aware any-order quantizer, and blockwise mixed precision; reports better 2-bit accuracy than AR-transferred PTQ methods.
Scope (models, setting): dLLMs (LLaDA/Dream family), weight-only 2-bit.
Relevance to parallelism erosion (what it does / does not measure — factual, no opinions about our work): Weight-only; abstract does not mention KV cache, decoding steps, or confidence.
