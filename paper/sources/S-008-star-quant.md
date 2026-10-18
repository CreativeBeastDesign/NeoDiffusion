id: S-008
title: STaR-Quant: State-Time Consistent Post-Training Quantization for Diffusion Large Language Models
authors: Xin Yan et al.
venue/id: arXiv:2606.04945 (v2), 2026-06-08
url: https://arxiv.org/abs/2606.04945
verified: 2026-10-18 by A-0 literature agent — abstract page fetched, ID ↔ title match
---
What it states (paraphrase; at most one short quote < 15 words): Identifies masked/unmasked activation-distribution differences and cross-step error accumulation as obstacles for low-bit dLLM PTQ; proposes state-guided activation transformation and a temporal attention compensation (block-diagonal affine correction of quantized attention).
Scope (models, setting): dLLMs, low-bit weight-activation quantization.
Relevance to parallelism erosion (what it does / does not measure — factual, no opinions about our work): Abstract does not mention KV-cache quantization, threshold decoding, step counts, or confidence shifts.
