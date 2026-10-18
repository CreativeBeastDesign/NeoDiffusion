id: S-002
title: SoloQ: Calibration-Free Quantization for Diffusion Language Models
authors: Donghyun Lee et al.
venue/id: arXiv:2610.07121 (v1), 2026-10-05
url: https://arxiv.org/abs/2610.07121
verified: 2026-10-18 by A-0 literature agent — abstract page fetched, ID ↔ title match
---
What it states (paraphrase; at most one short quote < 15 words): Calibration-free weight/activation quantization for dLLMs via structured rotation plus rescaling, with distribution-matched codebooks or NVFP4. For block-diffusion models it quantizes the KV cache only when a block is committed; the active block stays full precision. Sec. 4.5: "Keys use KIVI-style per-channel asymmetric quantization"; values use the method's activation path. Sec. 5.4 notes that for Nemotron-Labs-Diffusion the EOS trajectory changes after quantization, so latency is compared by replaying a BF16-recorded sequence of forward passes. App. D.4: keeping the attention sink in full precision matters mainly at 2-bit KV.
Scope (models, setting): LLaDA, LLaDA-1.5, Dream (full-sequence) and Fast-dLLM v2, Nemotron-Labs-Diffusion (block diffusion); W4A4KV4 and KV2 settings; Nemotron decoded with threshold 0.9, Fast-dLLM v2 with threshold 1.
Relevance to parallelism erosion (what it does / does not measure — factual, no opinions about our work): Applies per-channel key quantization in a dLLM with threshold decoding, and observes that quantization changes the decoding trajectory (EOS). It does not report NFE, step counts, or tokens per forward for quantized vs full-precision runs (the latency comparison deliberately fixes the forward count), does not ablate keys vs values separately, and contains no confidence or margin analysis (its divergence analysis is logit KL across masking ratios).
