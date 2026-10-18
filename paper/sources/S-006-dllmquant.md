id: S-006
title: DLLMQuant: Quantizing Diffusion-based Large Language Models
authors: Chen Xu et al.
venue/id: arXiv:2508.14090 (v2), 2025-08-26
url: https://arxiv.org/abs/2508.14090
verified: 2026-10-18 by A-0 literature agent — abstract page fetched, ID ↔ title match
---
What it states (paraphrase; at most one short quote < 15 words): Attributes PTQ degradation in dLLMs to step-varying token distributions, error accumulation across iterations, and masked/unmasked distribution differences; proposes timestep- and mask-aware calibration, interaction-guided activation quantization, and confidence-score-weighted weight error compensation.
Scope (models, setting): LLaDA and related dLLMs, W4A4 and similar; block-wise confidence-based unmasking per the official LLaDA setup.
Relevance to parallelism erosion (what it does / does not measure — factual, no opinions about our work): Uses token confidence scores as a calibration weight but does not report how confidence, step counts, or tokens per step change under quantization; KV cache storage precision is not discussed.
