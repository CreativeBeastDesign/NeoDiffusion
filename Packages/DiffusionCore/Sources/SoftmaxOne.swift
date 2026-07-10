import Foundation
import MLX

/// Evan Miller's "off-by-one" softmax (attention sink): `exp(x_i) / (1 + Σ_j exp(x_j))`.
/// Sumi applies it at every attention layer (`modeling_sumi.py: softmax_one`); stock fused
/// SDPA/flash kernels cannot express it, which is why `OffByOneAttention` is a manual path.
///
/// Two formulations, mathematically identical (metal-shader-guide §2.4):
/// - ``softmaxOneSink(_:axis:)`` — verbatim reference form: append a zero-logit sink column,
///   softmax, drop the sink. Kept as the validation oracle (analogous to `attendReference`).
/// - ``softmaxOne(_:axis:)`` — multiplicative/normaliser form used on the hot path: standard
///   stable softmax with the denominator replaced by `Σ exp(x_j − m) + exp(−m)` (the sink term
///   shifts with the row max: `1·exp(0−m)`). No sink allocation, no concat, and the same
///   single-line change fuses into a flash-style Metal kernel later (S4.1).
///
/// Callers must pass FP32 logits for parity-critical paths — the reference upcasts
/// (`softmax_one(attn_weights, dtype=torch.float32)`), and the sink term underflows earlier
/// at FP16 (`exp(-m)` for m ≳ 11). Underflow of the sink at very confident rows is correct
/// behaviour (reduces to standard softmax), not something to clamp away.
public func softmaxOne(_ logits: MLXArray, axis: Int = -1) -> MLXArray {
    let m = logits.max(axis: axis, keepDims: true)
    let e = exp(logits - m)
    let denom = e.sum(axis: axis, keepDims: true) + exp(-m)
    return e / denom
}

/// Reference formulation A: append a zero-logit sink along `axis`, softmax, drop the sink.
/// Oracle for ``softmaxOne(_:axis:)`` — not used on the hot path.
public func softmaxOneSink(_ logits: MLXArray, axis: Int = -1) -> MLXArray {
    let normalizedAxis = axis >= 0 ? axis : logits.ndim + axis
    var sinkShape = logits.shape
    sinkShape[normalizedAxis] = 1
    let sink = MLXArray.zeros(sinkShape, dtype: logits.dtype)
    let extended = concatenated([logits, sink], axis: normalizedAxis)
    let probs = softmax(extended, axis: normalizedAxis)
    return split(probs, indices: [logits.dim(normalizedAxis)], axis: normalizedAxis)[0]
}
