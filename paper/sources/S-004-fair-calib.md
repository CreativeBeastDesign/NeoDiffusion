id: S-004
title: FAIR-Calib: Frontier-Aware Instability-Reweighted Calibration for Post-Training Quantization of Diffusion Large Language Models
authors: Haoyu Huang et al.
venue/id: arXiv:2606.06547 (v2), 2026-07-19 (v1 2026-06-04); ICML 2026 poster
url: https://arxiv.org/abs/2606.06547
verified: 2026-10-18 by A-0 literature agent — abstract page fetched, ID ↔ title match
---
What it states (paraphrase; at most one short quote < 15 words): Reports that PTQ error flips borderline token decisions at the dLLM write frontier and that these flips become permanent after commit and propagate. Proposes a two-stage calibration that reweights layer-wise hidden-state MSE toward fragile frontier positions. App. D.3 measures a commit-step logit margin (teacher-committed token vs strongest competitor) for the quantized model and reports it is larger under the proposed calibration than under unweighted calibration.
Scope (models, setting): LLaDA and Dream at W4A4 (weights + activations); LLaDA with a confidence-driven policy, Dream entropy-driven; no threshold value stated.
Relevance to parallelism erosion (what it does / does not measure — factual, no opinions about our work): Closest work on decision/margin effects of quantization in dLLMs, but for weights/activations, not the KV cache. Measures teacher-forced decision flips and mean logit margins; does not report step counts, tokens per step, or threshold-crossing rates under quantization, and does not separate keys from values.
