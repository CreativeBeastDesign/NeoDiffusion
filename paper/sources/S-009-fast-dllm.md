id: S-009
title: Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding
authors: Chengyue Wu et al.
venue/id: arXiv:2505.22618 (v3), 2025-07-03
url: https://arxiv.org/abs/2505.22618
verified: 2026-10-18 by A-0 literature agent — abstract page fetched, ID ↔ title match
---
What it states (paraphrase; at most one short quote < 15 words): Introduces a block-wise approximate KV cache for bidirectional dLLMs and confidence-aware parallel decoding that commits only tokens whose confidence exceeds a threshold; reports large throughput gains with small accuracy loss.
Scope (models, setting): LLaDA and Dream.
Relevance to parallelism erosion (what it does / does not measure — factual, no opinions about our work): Defines the threshold-decoding regime in which tokens committed per step depend on confidence; does not study quantization.
