# OSDT: One-Shot Dynamic Thresholding for Diffusion Language Models

**Type**: paper  
**Canonical source**: 01-Inbox/OSDT.pdf  
**Date**: 2025 (arXiv)  
**Relevance**: Proposes calibration of confidence thresholds on a single sequence, then applying to future inputs. Addresses block/step-wise confidence fluctuations in masked diffusion. Achieves +24-50% tokens/s improvement.  
**Status**: processed  
**Last updated**: 2026-04-16

## Summary

OSDT addresses the inefficiency of fixed confidence thresholds in masked diffusion decoding. The key insight is that:

1. Confidence trajectories vary significantly across steps within a sequence.
2. Similar inputs have near-identical confidence trajectories (measured by cosine similarity).

OSDT calibrates thresholds on a single sequence and applies them to future inputs with similar characteristics.

## Key claims

- Fixed global thresholds ignore step-wise confidence fluctuations within a sequence.
- Confidence trajectories are reusable across inputs with similar characteristics.
- One-shot calibration (on single sequence) is sufficient for future sequences.
- +24% tokens/s on GSM8K at best accuracy.
- +45% tokens/s on GPQA with comparable accuracy.
- +50% tokens/s on HumanEval with modest accuracy gap.
- Broader implications: task-level confidence signatures could enable systems innovations.

## Mechanisms

### Problem with Fixed Thresholds

Standard approach:
- Use a fixed threshold θ to decide when to unmask tokens.
- If confidence > θ, unmask token.
- Problem: Some steps need higher θ, others lower. Fixed θ is suboptimal.

### Dynamic Thresholding

OSDT approach:
1. Run a "calibration" sequence through the model.
2. Observe confidence at each step.
3. Adjust thresholds per step based on observed confidence patterns.
4. Apply adjusted thresholds to future sequences.

### Confidence Trajectory Calibration

Key observation: Similar inputs have similar confidence trajectories (cosine similarity of confidence vectors across steps is high).

Calibration process:
1. Run initial sequence through model.
2. Record confidence at each step.
3. Compute per-step thresholds based on confidence distribution.
4. Store thresholds as a "signature" for that task/dataset.

### One-Shot Learning

The "one-shot" aspect:
- Only need one calibration sequence per task/dataset.
- Thresholds generalize to other sequences in the same distribution.
- No fine-tuning required; just inference-time calibration.

### Block-Wise vs Step-Wise

OSDT addresses both:
- **Block-wise**: Different blocks may need different thresholds.
- **Step-wise**: Different denoising steps have different confidence patterns.

## Trade-offs

- **Calibration cost**: One additional forward pass per dataset.
- **Generalization**: May not transfer across very different inputs.
- **Accuracy vs speed**: Higher thresholds = more tokens/s but potential quality loss.

## Hardware implications

- Calibration adds one forward pass but amortized across many future sequences.
- Per-step thresholding requires storing and applying threshold arrays.
- Memory: Small overhead for threshold storage.
- Compute: No additional compute beyond calibration.

## Relevance to Apple Silicon

- One-shot calibration is efficient for batch inference scenarios.
- Per-step thresholds could be baked into Metal kernels.
- Works well with block-based decoding (LLaDA).
- Complementary to other optimization methods (caching, early stopping).
- Could be combined with ICE: ICE uses task-specific signals, OSDT uses confidence patterns.

## Extracted concepts

- [[confidence-threshold-calibration]]
- [[one-shot-learning]]
- [[confidence-trajectory]]
- [[dynamic-thresholding]]

## Open questions

- How many calibration samples are needed for stable thresholds?
- Does OSDT work for online/live inference where calibration isn't feasible?
- What is the interaction between OSDT and other threshold-based methods?
- How does trajectory similarity generalize across domains?
- Metal kernel design: efficiently applying per-step thresholds?
- Can OSDT be combined with adaptive block sizes?

## Related pages

- [[confidence-based-decoding]] (confidence signals for decoding)
- [[ice]] (early exit strategies)
- [[just-on-time-jot]] (token-level early stopping)
- [[fast-dllm]] (threshold-based parallel decoding)