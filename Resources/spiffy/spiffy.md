# Spiffy

**Type**: paper
**Canonical source**: Spiffy: Multiplying Diffusion LLM Acceleration via Lossless Speculative Decoding (Sudhanshu Agrawal et al., Qualcomm AI Research), arXiv:2509.18085v3, cs.LG, Jan 2026
**Date**: 2026-01 (v3)
**Relevance**: Lossless speculative decoding for dLLMs achieving 2.8-3.1x alone and up to 7.9x combined with parallel decoding — directly extends this wiki's existing speculative-decoding-dLLM concept with a training-free, no-separate-drafter-model approach.
**Status**: processed
**Last updated**: 2026-07-03

## Summary

Spiffy introduces **auto-speculation**: instead of pairing a fast drafter model with an accurate verifier model (as in [[speculative-decoding-dllm]] / DualDiffusion), Spiffy draws draft states directly from the dLLM's own output distribution, requiring no separate drafter. To structure which draft candidates to try, it introduces **directed draft graphs** — a generalization of AR speculative decoding's draft trees that allows nodes to have multiple parent blocks, a structure unique to bidirectional dLLMs. The specific graph used at inference is fixed in advance via an **offline calibration algorithm** run on fewer than 50 calibration samples, using a "degree-1-accumulation" selection rule shown empirically superior to alternative accumulation strategies. Verification is proven exactly lossless (formal proof in the paper's appendix), and overhead is negligible (<5% of inference time even at the largest draft budget tested).

## Key claims

- Auto-speculation requires no separate drafter model — draft candidates come from the dLLM's own distribution (**sourced**).
- Verification is formally proven lossless — output distribution matches sequential decoding exactly (proof in Appendix A.1) (**sourced**). Near-lossless (not exactly) on LLaDA specifically due to minor bfloat16 precision effects under batched attention masks; would be exactly lossless with dual KV caches or blockwise-causal dLLMs (**sourced**).
- Directed draft graphs generalize AR draft trees by allowing multiple parent blocks per node — unique to bidirectional dLLM structure, giving multiple pathways to acceptance (**sourced**).
- Offline calibration (<50 samples, <30 min on a single GPU) determines a fixed draft graph structure that generalizes well across datasets/models with only minor structural variation (**sourced**).
- Degree-1-accumulation (Q* = argmax Σ(count(q) + Σ_parents count(p))) empirically outperforms degree-0-accumulation (frequency only) and total-accumulation (full ancestor chain) for graph selection (Table 4 / Appendix E) (**sourced**).
- 2.24x-3.07x speedup alone (draft budgets D=3,5,8,10); up to 7.9x when combined with Fast-dLLM-style threshold/hard-code parallel decoding (**sourced**).
- Overhead is negligible: drafting <0.5%, verification <3% of model inference time even at D=10 (Table 3) (**sourced**).
- Per-block acceptance rates improve for later blocks, attributed to more available context to condition on (**sourced**).

## Mechanisms

- **Draft content selection**: for each draft position, token position rank i is determined via ArgSort of max marginal probabilities, and vocabulary rank j via ArgSort of per-position token probabilities, giving candidate tokens c_ij.
- **Directed draft graph structure**: a draft block B is a child of block A if tb = ta − 1, |unmasked(A)| + S_tb = |unmasked(B)|, and unmasked(A) ⊂ unmasked(B) — i.e., B extends A by unmasking additional positions consistent with A's own unmasking.
- **Offline calibration (Algorithm 2)**: collect (i,j) sequences from rewound generation traces over a small calibration dataset; select D draft formulas by degree-1-accumulation optimization over those traces.
- **Verification (Algorithm 1)**: proven lossless — preserves the exact output distribution of sequential (non-speculative) decoding.

## Hardware implications

- Because the draft graph is fixed offline (not recomputed per-inference), the runtime cost is essentially the extra forward-pass batching for draft candidates plus verification — a static, predictable compute shape rather than a dynamic search, which is favorable for kernel specialization (**inferred**).
- No separate drafter model means no additional model weights resident in memory — a direct memory-footprint advantage over DualDiffusion-style two-model speculative decoding on memory-constrained unified-memory systems (**sourced** claim of no separate drafter; **inferred** the Apple Silicon memory implication).
- The near-losslessness caveat (bfloat16 batched-attention-mask precision effects) suggests the exact numerical behavior is somewhat implementation/precision-sensitive — worth flagging before assuming exact losslessness ports directly to a different numerical backend (e.g., Metal's fp16/bf16 handling) (**inferred**).

## Relevance to Apple Silicon

- **Likely useful**: single-model auto-speculation avoids the two-model memory burden that DualDiffusion-style approaches carry, which is a direct advantage for on-device deployment (**inferred**).
- **Likely useful**: offline-calibrated, fixed draft graphs are compute-predictable and could be specialized into a dedicated Metal kernel path rather than requiring dynamic graph construction at inference time (**inferred**).
- **Unknown portability**: the exact-losslessness proof's dependency on numerical precision behavior under batched attention masks (bfloat16-specific effects noted by the authors) may manifest differently under Metal's floating-point semantics — not evaluated in the source (**speculative**).

## Extracted concepts

- [[auto-speculative-decoding]]
- [[directed-draft-graph]]
- [[offline-draft-calibration]]

## Proposal impact

- No `04-Proposals/` pages exist yet; this source is a strong candidate input for a future proposal combining auto-speculation with existing caching proposals ([[elastic-cache]], [[cache-eviction]]) since Spiffy's overhead is reported as negligible and could stack with cache-based acceleration rather than compete with it.

## Open questions

- How does auto-speculative decoding compare directly (same model, same benchmarks) against [[self-speculative-decoding-dlm]] (S2D2), which also avoids a separate drafter model but uses a different self-speculation mechanism (block-size-1 AR verifier)?
- Does the offline-calibrated draft graph need to be recalibrated per deployment hardware, or is calibration purely data/model-dependent and hardware-agnostic?
- What is the actual numerical losslessness behavior under Metal's bf16/fp16 handling, given the paper's own caveat about batched-attention-mask precision effects on LLaDA?
- Can directed draft graphs be combined with [[dynamic-bidirectional-cache-eviction]] (Sparse-dLLM) without the eviction disrupting the draft graph's assumed KV availability for multi-parent nodes?
