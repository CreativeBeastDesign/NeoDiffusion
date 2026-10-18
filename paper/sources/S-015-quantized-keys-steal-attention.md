id: S-015
title: Quantized Keys Steal Attention: Bias Correction for KV-Cache Compression in Video Diffusion
authors: Tuna Tuncer et al.
venue/id: arXiv:2605.26266 (v1), 2026-05-25
url: https://arxiv.org/abs/2605.26266
verified: 2026-10-18 by A-0 literature agent — abstract page fetched, ID ↔ title match
---
What it states (paraphrase; at most one short quote < 15 words): Attributes quality loss from low-bit KV caches to a softmax convexity ('Jensen') bias by which quantized cached keys draw extra attention away from the unquantized current chunk; proposes a per-score correction from key step sizes and query norm.
Scope (models, setting): Chunk-wise autoregressive video diffusion (MAGI-1, SkyReels-V2, HY-WorldPlay); not language models.
Relevance to parallelism erosion (what it does / does not measure — factual, no opinions about our work): Gives a key-side mechanism for KV-quantization damage to attention; no token-commit thresholds, no step counts, no explicit key-vs-value comparison in the abstract.
