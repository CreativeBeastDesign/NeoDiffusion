While reading through the logs and the conclusions, some things I noticed:

# Findings

## Token‑match vs FP16 is a parity metric, not a quality metric

At temperature 0 a single early argmax flip cascades through the rest of the block, so 73.0 % vs 41.5 % is not a linear quality scale. Also the denominators differ across arms (716 / 719 / 732 / 718), so percentages aren't strictly comparable. The only real correctness signal in the whole report is the `reason-logic` syllogism pass/fail — and that's n = 1 per arm.

## The 4/12 vs 0/12 exact‑match contrast is underpowered

Fisher exact gives p ≈ 0.047 one‑sided, ≈ 0.09 two‑sided [recomputed] — not significant at α = 0.05 two‑tailed with 12 prompts. The step counts, starvation rates and TPS deltas are the robust signals; the exact‑match dichotomy should be demoted to supporting evidence.

## Confidence metrics are not monotone with parity

`kivi4` is faster, needs fewer steps, is less starved, and is more confident than the exact FP16 baseline — yet matches only 58.7 % of its tokens. A compressed cache beating lossless on every "health" metric is a red flag that those metrics are partly luck, and it shows two distinct failure channels that §2.2 conflates:

- Confidence collapse → step inflation (the real "starvation" story; clearly a 4‑bit‑Key effect).
- Argmax flips at unchanged or higher confidence (explains `fp8`'s 30 % token loss with no step penalty at all).

Please separate these explicitly. As written, the mechanism section attributes nearly everything to distribution flattening, which the 8‑bit data contradict.

## Operator‑level and end‑to‑end rankings disagree for `kivi4` vs `fp4g32`

In Regime B, `kivi4` has lower Key L2 (0.0666 vs 0.1130) but higher attention KL (0.04702 vs 0.03575) and higher output error (0.2310 vs 0.1953) — yet end‑to‑end it wins decisively (280 vs 311 steps, 15.46 vs 14.31 TPS, 4/12 vs 1/12 exact). Either Regime B's synthetic outlier structure (a constant ±6.0 offset) doesn't represent real post‑RoPE Key statistics, or L2/KL on a synthetic op aren't good predictors. Since Regime B is the load‑bearing evidence for the outlier mechanism, this needs reconciling — ideally by repeating the benchmark on real Keys extracted from the model rather than synthesized ones.

## `kivi4`'s 58.7 % parity is worse than `k8v4`'s 73.0 %

The report presents kivi4 as curing starvation (true) but the shipped preset framing slightly oversells it: you trade 14 pp of token parity for 11 pp more compression. That's a legitimate trade, but it should be stated as a trade, not a strict improvement.

## Statistical power and repetition

`--runs 1`, single host, single seed, 12 prompts. Step deltas like 284 vs 291 (−2.4 %) are within plausible run‑to‑run noise for a thresholded sampler. The `envValid` gate is excellent hygiene for timing validity but does nothing for sampling variance. Three seeds × 30+ prompts, with per‑prompt step deltas and CIs, would move this from "suggestive" to "publishable."

## Scope: this is a short‑context result

`genLength=64` with `blockLength=32` means two blocks per generation and a prefix of ≤ 128 tokens (max KV 5.0 MiB). Everything — including the "bandwidth savings are negligible" conclusion — is conditioned on that regime. The generalization claim needs explicit scoping until Track 3/5C land.

## A few loose ends in the text

- §2.2 claims forwards are "97.8 % MoE expert + attention compute," but the logbook's own profiling says attention ≈ 14.3 % and MoE ≈ 56.3 % (and elsewhere "42.9 %+"). 97.8 % is unsupported and inconsistent with the rest of the document.
- §2.3 numbers the mitigations 1, 3, 2 (ordering typo).
- The hybrid4 ablation is coarse: only "protect layers 0–3" was tested. The reverse arrangement (FP4 shallow, K8V4 deep) and a per‑layer sensitivity sweep would properly support "middle and deep layers remain sensitive."
- Track‑1 Table 1.3A omits ms/fwd and forward counts, which is why the mechanism had to be inferred from suite‑time/step ratios. Add them.
- kivi4's group size is hard‑coupled to blockLength = 32. If blockLength changes, or if you combine it with within‑block reuse (CSA2, JOT v2 — Track 2), the per‑channel groups can't be finalized until the block settles. That interaction is untested and is the obvious next risk.

# Revisions

(but perhaps not on this device)

- Separate confidence‑collapse from argmax‑flip as two failure channels (the fp8/kivi4 data force this)
- replace the unweighted "Mean TPS" with aggregate tokens/time or report both
- add seeds and error bars
- reconcile the kivi4 operator‑level vs end‑to‑end ranking
- validate Regime B against real extracted Keys
- scope the claims to ≤ 128‑token prefixes pending the long‑context tracks

# Further experiments

## Phase 1 - Disentangling the Dual Failure Modes

Previous results showed a paradox: kivi4 has better step counts and confidence than FP16, but worse token parity. The literature on "Quantized Keys Steal Attention" explains this: 4-bit noise systematically inflates attention logits, artificially boosting confidence while routing it to the wrong tokens.

### Experiment 1 - The Softmax Bias & Margin Audit

- **Hypothesis**: `kivi4` suffers from "inflation bias" rather than true confidence, and that `k4v8` suffers from "margin collapse."
- **Method**: For the active block at step t, extract the vocabulary logits z_v for all masked positions. Compute:
  - Softmax Bias: β=∑exp(s^)∑exp(s)\beta = \frac{\sum \exp(\hat{s})}{\sum \exp(s)}
  - Commitment Margin: mi=max(v)Pθ(xi=v)−second_max(v)Pθ(xi=v)m_i = \max_v P_\theta(x_i=v) - \text{second\_max}_v P_\theta(x_i=v)
  - Logit Entropy: H(Pθ(xi))H(P_\theta(x_i))
- **Expected Result**: k4v8 will show collapsed margins (high entropy, causing starvation). kivi4 will show inflated bias (β>1) but tight margins on the wrong tokens (causing parity loss without starvation). FP16 and k8v4 will have β≈1.

### Experiment 2 - Real-Key Operator Benchmark (Replacing Synthetic Regime B)

`kivi4` beat `fp4g32` end-to-end but lost in the synthetic operator benchmark.

- **Hypothesis**: `kivi4` will dominate `fp4g32` on real Keys, aligning the micro and macro benchmarks.
- **Method**: Extract actual Key tensors from LLaDA2.1-mini at Step 0 (all [MASK]) and Step 5 (mixed) for 100 prompts. Run the exact same operator harness (Table 1.2A/B) on these real tensors.
- **Expected Results**: Real Keys have structured, persistent channel outliers that synthetic Gaussian+Offset models imperfectly. `kivi4` (per-channel over the 32-token block) will likely dominate `fp4g32` on real Keys, aligning the micro and macro benchmarks.

### Experiment 3 - Metric Predictiveness Ablation

Goal would be to identify the exact layer-wise metric that predicts end-to-end failure.

- **Hypothesis**: MSE is useless as a metric for predicting Step Delta or Token Parity.
- **Method**: Run 50 diverse prompts. For every layer and every step, record: Key MSE, Value MSE, Centered Score Error, Attention KL, Softmax Bias, and Commitment Margin. Correlate these with the end-to-end Step Delta (vs FP16) and Token Parity (vs FP16).
- **Expected Results**:
  - Key MSE will have near-zero correlation with Step Delta.
  - Attention KL will strongly predict Step Delta (Confidence Collapse).
  - Softmax Bias / Commitment Margin will strongly predict Token Parity (Argmax Flips).

## Phase 2 - The MoE Interaction

This is where the MoE research (DES, REFLEX, MoECa; see [moe_research.md](../Notes/moe_research.md)) becomes critical. In an MoE model, a small perturbation in the hidden state h (caused by KV noise) alters the router logits. If the token crosses a routing boundary, it activates a completely different set of experts. This Expert Misrouting is a discrete, non-linear amplifier of KV noise that autoregressive models don't suffer from in the same way (because AR models process 1 token at a time, allowing the router to adapt sequentially, whereas diffusion processes 32 tokens in parallel).

### Experiment 4 - MoE Routing Drift & "Expert Misrouting" Rate

- **Hypothesis**: KV quantization causes tokens to be routed to the wrong experts.
- **Method**: For the `FP16` baseline and the `kivi4`/`k4v8` arms, extract the top-k expert selections for every token in the active block. Compute the Jaccard Similarity of the expert sets.
- **Expected Results**: `k8v4` will have >95% routing overlap with `FP16`. `k4v8` and `kivi4` will show significant "Expert Misrouting" (e.g., 15-20% of tokens route to a different expert set). This proves that KV noise doesn't just blur the FFN output; it completely changes the FFN computation path.

### Experiment 5 - Block-Level Expert Coreset (DES) as a Noise Filter

- **Hypothesis**: Restricting routing to a shared coreset (DES) stabilizes the model against KV noise
- **Method**: Implement DES: aggregate router scores over the 32-token block, select a coreset of the top 16 experts, and force all tokens to route within this coreset. Run this combined with `kivi4`.
- **Expected Results**: By preventing tokens from routing to rare, outlier experts due to KV noise, DES might act as a regularizer, improving the token parity of `kivi4` without changing the KV cache size.

### Experiment 6 - Token-Selective FFN under KV Noise (REFLEX / DyLLM)

- **Hypothesis**: Reusing FFN outputs for "stable" tokens (DyLLM) is safe when the KV cache is quantized.
- **Method**: Implement DyLLM's heuristic: reuse the FFN output from step t−1 if the hidden-state drift is below ϵ. Compare the drift threshold required for `k8v4` vs `kivi4`.
- **Expected Results**: Because KV noise introduces high-frequency variance in the hidden states, `kivi4` will trigger the drift threshold much more often than `FP16`, nullifying the FLOPs savings of DyLLM unless the threshold is aggressively relaxed (which hurts quality).

## Phase 3 - Advanced Mitigations (Beyond kivi4)

### Experiment 7 - Pre-RoPE vs. Post-RoPE `kivi4`

"Outlier smearing" hypothesis: RoPE rotates channel pairs (d,d+D/2) by token position. If you quantize after RoPE, a persistent channel outlier gets smeared across the 32-token block.

- **Hypothesis**: Quantizing before RoPE reduces the severity of outlier smearing.
- **Method**: Quantize Keys before RoPE (post-LayerNorm). Apply RoPE dynamically during the attention dot-product using FP16/FP32 math.
- **Expected Results**: Pre-RoPE `kivi4` will retain the exact same cache size but show a measurable drop in Attention KL and an increase in Token Parity, as the outlier channels remain perfectly stationary across the block.

### Experiment 8 - FP32 QK^T Accumulation

- **Hypothesis**: The bottleneck is the storage precision or the dot-product precision.
- **Method**: Store Keys in 4-bit (`kivi4`), but dequantize them into an FP32 accumulator buffer during the QK^T matrix multiplication, executing the softmax in FP32.
- **Expected Results**: This eliminates the E[exp(s+ϵ)] bias and the argmax flips. If parity jumps to >90% while maintaining the 70% cache reduction, it proves that 4-bit storage is viable if and only if the compute kernel supports FP32 accumulation.

### Experiment 9 - The "Starvation Rescue" Fallback

Goal would be a pure systems-level mitigation.

- **Hypothesis**:
- **Method**: If the engine detects N consecutive starvation steps (`transfers == 1`), it temporarily dequantizes the prefix cache to FP8 (or FP16) for one forward pass to break the loop, then drops back to 4-bit.
- **Expected Results**: This bounds the worst-case latency of `k4v8`/`fp4` without requiring the heavy engineering of `kivi4`.

## Phase 4 - Scaling to Long Context

Currently impossible, as access to MacStudio is only possible on Monday.

### Experiment 10 - Prefix Length Sweep & The Bandwidth Cliff

Goal would be to find the break-even point where KV cache compression actually yields wall-clock speedups.

- **Hypothesis**:
- **Method**: Run `exact-fast`, `k8v4`, and `kivi4` on a fixed prompt, extending the generation length to 512, 1024, and 2048 tokens (or using a long-context retrieval task). Measure ms/fwd at each length.
- **Expected Results**: At L=128, ms/fwd is identical across arms (~197ms). At L=2048, the memory bandwidth cost of dequantizing kivi4 (or the HBM traffic of FP16) will cause ms/fwd to diverge. This establishes the Memory-Compute Crossover Point, proving that kivi4 is not just a quality-preserving trick, but a necessary optimization for long-context diffusion.
