# Per-Block FP8 Quantization

**Summary**: Quantization technique that applies FP8 precision to model weights and/or activations at the block level, balancing speed and accuracy.  
**Aliases**: block-wise quantization, FP8 inference  
**Status**: stable  
**Last updated**: 2026-04-16  
**Sources**:
- [[llada2-1-tech-report]]

---

## Definition

Per-Block FP8 Quantization reduces the numerical precision of model weights and/or activations to 8-bit floating point (FP8) but does so at the granularity of blocks (groups of tokens or layers). This contrasts with uniform quantization across the whole model. The block-wise approach allows for better handling of outliers and maintains accuracy while still gaining the memory bandwidth and compute benefits of lower precision.

## Why it matters

LLaDA2.1-Flash achieves high throughput partly due to per-block FP8 quantization. Lower precision reduces memory traffic (smaller data moved) and can accelerate matrix multiplies on hardware that supports FP8 natively (e.g., NVIDIA Hopper, potentially future Apple Silicon). The block-wise strategy helps preserve model quality compared to naive full-model quantization.

## Mechanism

- Model is partitioned into blocks (could be token blocks, layer blocks, or expert blocks).
- Each block's weights/activations are scaled and quantized to FP8 independently.
- Dequantization to higher precision may happen during computation or computation may be performed directly in FP8.
- Scaling factors stored per block.

## Trade-offs

- **Memory**: 4× reduction vs FP16 (or 2× vs BF16).
- **Compute**: FP8 operations faster if hardware supports; otherwise may require dequantization.
- **Accuracy**: Block-wise helps maintain quality; paper notes "balance inference speed and model accuracy".
- **Complexity**: Requires careful calibration of per-block scaling factors.

## Apple Silicon implications

- Current Apple Silicon GPUs (as of 2025) do not have native FP8 matrix units; would likely require dequantization to FP16/BF16 for compute.
- Memory savings still valuable: more model fits in unified memory, less bandwidth used.
- Could be combined with other optimizations like selective quantization (only some blocks to FP8).
- Implementation in Metal: use `short` or `half` to store FP8 and convert on load.

## Related concepts
- [[alpha-moe-megakernel]] (co-deployed)
- [[radix-caching]] (both reduce memory traffic)

## Open questions
- Does Apple Silicon's memory bandwidth become the bottleneck even with FP8 quantization?
- What is the quality delta when using per-block FP8 vs higher precision on Apple Silicon hardware benchmarks?