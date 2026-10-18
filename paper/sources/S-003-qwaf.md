id: S-003
title: QWaF: Ultra-Low-Bit KV-Cache Quantization for Block-Diffusion Language Models
authors: Junwon Lee et al.
venue/id: NeurIPS 2026 DiffuLM Workshop (no arXiv ID found), 2026-09
url: https://iris.skku.edu/publication/c111_neurips_diffulm_workshop_2026/
verified: 2026-10-18 by A-0 literature agent — abstract page fetched (lab publication page; no arXiv listing found), title match
---
What it states (paraphrase; at most one short quote < 15 words): Applies three AR KV-cache quantizers (uniform, outlier-protected, SVD mixed precision) to a block-diffusion model and its AR parent; the AR model degrades more and collapses below about 2 bits while the block-diffusion model retains most accuracy, which the authors attribute to weaker attention-sink concentration in the diffusion model. Proposes QWaF, a query-aware mixed-precision quantizer preserving key directions future queries use; claims to be the first KV-cache quantization method for dLLMs.
Scope (models, setting): SDAR-8B block diffusion vs AR parent; LongBench; A100 fused CUDA path.
Relevance to parallelism erosion (what it does / does not measure — factual, no opinions about our work): Measures task-score robustness to KV quantization in block diffusion. The abstract does not mention decoding steps, tokens per step, threshold decoding, a key-vs-value comparison, or per-channel key quantization. Full paper not located; only the abstract was checked.
