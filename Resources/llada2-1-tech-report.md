# LLaDA2.1: Speeding Up Text Diffusion via Token Editing

**Type**: paper  
**Canonical source**: 01-Inbox/llada2_1_tech_report.pdf  
**Date**: 2025 (inferred from references)  
**Relevance**: Introduces token editing for dLLMs, configurable dual-threshold decoding, and RL alignment via EBPO; foundational for LLaDA2.1 models.  
**Status**: processed  
**Last updated**: 2026-04-16

## Summary
LLaDA2.1 extends discrete diffusion language models (dLLMs) with a novel "Draft-and-Edit" paradigm that combines Mask-to-Token (M2T) and Token-to-Token (T2T) operations under dual configurable confidence thresholds (τ_mask, τ_edit). This enables two operating modes: Speedy Mode (S Mode) with aggressive M2T thresholds for high throughput, and Quality Mode (Q Mode) with conservative thresholds. The model uses a Mixture of M2T and T2T objectives during CPT and SFT, and introduces ELBO-based Block-level Policy Optimization (EBPO) for reinforcement learning scaling. Achieves up to 892 TPS on HumanEval+ (100B model) with minimal quality drop versus Q Mode.

## Key claims
- Editing (T2T) operations enable retroactive error correction, breaking the speed-accuracy tradeoff.
- Lowering τ_mask aggressively in S Mode yields dramatic throughput gains with manageable quality loss.
- EBPO scales RL for dLLMs to long contexts by vectorizing ELBO computation.
- Multi-Block Editing (MBE) further improves quality through cross-block refinement.
- dLLMs' parallel decoding error accumulation (exposure bias) is mitigated by timely editing.

## Mechanisms
- Dual threshold decoding: Γ_t (unmasking set) for M2T, Δ_t (editing set) for T2T based on confidence > τ_mask or τ_edit.
- Mixture of M2T and T2T during training: exposes model to both masked prediction and noise-based token recovery.
- Multi-turn Forward (MTF) augmentation increases editing scenario diversity.
- EBPO: Uses ELBO as likelihood proxy; aggregates block-level contributions via vectorized computation.
- MBE: Revisits previously generated blocks based on later content for global consistency.

## Hardware implications
- High parallelism of dLLMs requires efficient attention over long contexts → block-wise causal masked attention reduces KV cache compute.
- MBE introduces additional forward passes but improves quality per token; throughput vs quality tradeoff is domain-dependent (best for code/math).
- FP8 quantization and MoE megakernel (Alpha-MoE) integration accelerate inference.
- Radix caching and batching support essential for production deployment in SGLang.

## Relevance to Apple Silicon
- Unified memory architecture benefits from block-wise attention (single pass KV cache).
- MoE fusion and per-block quantization align with Metal's threadgroup and cache hierarchy.
- Throughput gains in code/math suggest potential for high-TPS on Apple Silicon if kernels are optimized for GPU occupancy and memory bandwidth.
- Need to map block diffusion patterns to Metal threadgroup sizes and evaluate memory traffic for T2T edits.

## Extracted concepts
- [[mask-to-token-m2t]]
- [[token-to-token-t2t]]
- [[editable-state-evolution]]
- [[configurable-threshold-decoding]]
- [[speedy-mode-s-mode]]
- [[quality-mode-q-mode]]
- [[exposure-bias-in-dllms]]
- [[multi-block-editing-mbe]]
- [[elbo-based-block-level-policy-optimization-ebpo]]
- [[multi-turn-forward-mtf]]
- [[vectorized-likelihood-estimation]]
- [[block-wise-causal-attention]]
- [[alpha-moe-megakernel]]
- [[per-block-fp8-quantization]]
- [[radix-caching]]
- [[sglang-rollout-engine]]

## Proposal impact
- Grounds [[mega-kernel-v1-fused-remask-sample]] with concrete thresholds and modes.
- Suggests new proposal: [[fusion-architecture-for-editable-dllms]] combining M2T/T2T in a single kernel launch.
- Highlights need for Apple-Silicon-specific proposal: [[metal-implementation-of-mbe]].

## Open questions
- How do τ_mask and τ_edit interact with Metal threadgroup occupancy limits?
- Can EBPO's vectorized likelihood estimation be expressed efficiently in Metal compute shaders?
- What is the memory traffic cost of MBE when blocks are edited after initial generation?
- How does per-block FP8 quantization affect numerical stability on Apple Silicon GPUs?
- Are there domain-specific optimal threshold configurations for code vs math vs general instruction?