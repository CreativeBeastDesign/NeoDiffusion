# Just on Time (JOT): Token-Level Early Stopping for Diffusion Language Models

**Type**: paper  
**Canonical source**: 01-Inbox/Just on Time.pdf  
**Date**: 2025 (arXiv)  
**Relevance**: Introduces token-level early stopping based on convergence signals. Tokens that stabilize early can be "finalized" without waiting for all denoising steps. Complementary to caching methods.  
**Status**: processed  
**Last updated**: 2026-04-16

## Summary

Just on Time (JOT) addresses the inefficiency of running all denoising steps for all tokens. The key insight is that many tokens reach stability (final prediction) long before the last denoising step.

The method:
1. Monitors convergence signals per token position.
2. When a token's prediction stabilizes (converges), finalize it.
3. Skip further computation for that position.

This is training-free and adaptive per-token.

## Key claims

- Different tokens converge at different rates; running uniform steps for all is wasteful.
- Lightweight convergence signals can identify stable tokens without task-specific tuning.
- JOT achieves state-of-the-art efficiency gains while preserving quality.
- Per-token freezing is adaptive; doesn't require knowing task in advance.
- Works across math reasoning, QA, and scientific understanding benchmarks.

## Mechanisms

### Convergence Signals

A token is "converged" when:
1. Its prediction has been consistent across multiple steps.
2. The model's confidence in the prediction is high.
3. Local context supports the prediction.

Signals used:
- **Prediction stability**: Token ID unchanged for last k steps.
- **Confidence**: Model's max probability above threshold.
- **Local context consistency**: Prediction aligns with neighboring tokens.

### Per-Token Freezing

Once a token is deemed converged:
1. Mark token as "finalized".
2. Exclude from subsequent denoising steps.
3. Token is treated as clean; no more mask/noise operations.

### Dynamic Step Reduction

Total steps reduced because:
- Some tokens finalized early.
- Remaining steps focus on unconverged tokens.
- Overall compute proportional to number of unconverged tokens per step.

### Lightweight Overhead

Monitoring convergence signals adds minimal overhead:
- Just tracking predictions and confidence.
- Can be fused into existing denoising kernels.

## Trade-offs

- **Early stopping vs quality**: Risk of premature stopping; mitigated by using multiple signals.
- **Signal thresholds**: Require tuning; too aggressive loses quality, too conservative loses speed.
- **Task dependence**: Some tasks may have tokens that naturally converge later.

## Hardware implications

- Reduced compute per step as tokens are finalized.
- Simpler computation for remaining tokens.
- Memory: Finalized tokens can be cached/stored differently.
- Bandwidth: Less data movement for finalized tokens.

## Relevance to Apple Silicon

- Per-token early stopping could be implemented in Metal.
- Convergence monitoring can be fused into attention/output kernels.
- Complementary to caching: finalized tokens have stable KV (easy to cache).
- Could combine with Elastic-Cache: once token is converged, cache aggressively.
- Works well with block-based decoding where blocks can be independently finalized.

## Extracted concepts

- [[token-level-early-stopping]]
- [[convergence-signals]]
- [[prediction-stability]]
- [[per-token-freezing]]

## Open questions

- What are the optimal convergence signal weights?
- Can convergence be predicted earlier using other signals?
- How does JOT interact with confidence-based decoding methods?
- What is the tradeoff between stopping precision and recall?
- How does JOT compare to block-level early stopping (ICE)?
- Metal kernel design for efficient convergence monitoring?

## Related pages

- [[confidence-based-decoding]] (confidence as a signal)
- [[ice]] (block-level early stopping)
- [[osdt]] (dynamic thresholding for speedup)
- [[llada2-1-tech-report]] (base model)