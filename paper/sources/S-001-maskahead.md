id: S-001
title: Mask-Guided KV Cache Eviction in Block Diffusion Language Models
authors: Gleb Molodtsov et al.
venue/id: arXiv:2610.06996 (v1), 2026-10-04
url: https://arxiv.org/abs/2610.06996
verified: 2026-10-18 by A-0 literature agent — abstract page fetched, ID ↔ title match
---
What it states (paraphrase; at most one short quote < 15 words): Proposes MaskAhead, a training-free KV selection/eviction method for block diffusion LMs driven by mask-query rankings; a quantized variant (Q-MaskAhead) computes selection and attention directly from a 4-bit KV cache with no bf16 shadow copy. Full text (Sec. 4, App. D): keys are quantized asymmetric per-channel over groups of 32 positions, values per-token in groups of 32 channels. App. D.1 (MATH-500) compares K4V4, K4V2, K2V4, K2V2 and states "Reducing key precision to two bits costs more accuracy than reducing value precision" at similar cache size. A rotated-and-centered key transform is also tested on HotpotQA.
Scope (models, setting): Block diffusion LMs — Fast-dLLM-v2-7B, DreamReasoner-8B, LLaDA-2.0-mini; long-prompt QA and MATH-500; eviction combined with 4-bit (and 2-bit ablation) KV.
Relevance to parallelism erosion (what it does / does not measure — factual, no opinions about our work): Reports a key-vs-value precision asymmetry, but measured as task accuracy only. Does not report denoising step counts, tokens per step, NFE, or the confidence-threshold settings used; does not examine whether KV quantization changes the number of steps. Reports generated lengths differ slightly between quantized and dense runs. Keys are already quantized per-channel (KIVI-style) in all arms, so per-token vs per-channel keys is not compared.
