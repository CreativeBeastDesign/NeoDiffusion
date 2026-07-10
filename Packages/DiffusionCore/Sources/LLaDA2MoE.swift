import Foundation
import MLX
import MLXNN

// MARK: - Expert weight stacking

/// Collapses per-expert checkpoint tensors into the gathered `[numExperts, ...]` layout the
/// ``SwitchLinear`` blocks expect (phase-2 §2.5).
public enum ExpertWeightStacking {
    /// `<prefix>.experts.<e>.<proj>.<suffix>` for `e in 0..<E` becomes
    /// `<prefix>.experts.<proj>.<suffix>` with a new leading expert axis. Applies to any
    /// tensor kind (`weight`, `scales`, `biases`). All other keys pass through untouched.
    public static func stack(_ arrays: [String: MLXArray]) -> [String: MLXArray] {
        var groups: [String: [(index: Int, value: MLXArray)]] = [:]
        var result: [String: MLXArray] = [:]

        for (key, value) in arrays {
            // Matches "experts.<e>." whether or not preceded by a dot (full-model vs
            // module-relative keys).
            guard let range = key.range(of: #"experts\.\d+\."#, options: .regularExpression) else {
                result[key] = value
                continue
            }
            let indexString = key[range].dropFirst("experts.".count).dropLast()
            guard let index = Int(indexString) else {
                result[key] = value
                continue
            }
            let stackedKey = key.replacingCharacters(in: range, with: "experts.")
            groups[stackedKey, default: []].append((index, value))
        }

        for (stackedKey, entries) in groups {
            let ordered = entries.sorted { $0.index < $1.index }.map { $0.value }
            result[stackedKey] = stacked(ordered, axis: 0)
        }
        return result
    }
}

// MARK: - Switch (per-expert gathered) linear layers

/// A linear layer applied per-token through a stack of expert weight matrices
/// `[numExperts, outputDims, inputDims]`, dispatched with `gatherMM` — the engine's
/// replacement for the reference's sort-by-expert CPU loop (phase-2 §5 deviation 4;
/// identical math, no per-step CPU sync).
///
/// Checkpoints store per-expert tensors (`experts.{e}.gate_proj.weight`); the loader
/// stacks them along axis 0 before `update(parameters:)`.
open class SwitchLinear: Module, Quantizable {
    public let weight: MLXArray

    public init(inputDims: Int, outputDims: Int, numExperts: Int) {
        self.weight = MLXArray.zeros([numExperts, outputDims, inputDims])
        super.init()
    }

    public init(weight: MLXArray) {
        self.weight = weight
        super.init()
    }

    /// - Parameters:
    ///   - x: `[..., 1, 1, inputDims]` (token batch dims, then two singleton dims)
    ///   - indices: expert indices `[..., k]` (flat over the expert axis)
    /// - Returns: `[..., k, 1, outputDims]`
    open func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        gatherMM(x, weight.swappedAxes(-1, -2), rhsIndices: indices)
    }

    public func toQuantized(groupSize: Int, bits: Int, mode: QuantizationMode) -> Module {
        QuantizedSwitchLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

/// Quantized variant of ``SwitchLinear`` dispatching through `gatherQuantizedMM`
/// (`gather_qmm`, phase-2 §2.5).
public final class QuantizedSwitchLinear: SwitchLinear, Quantized {
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode
    public let scales: MLXArray
    public let biases: MLXArray?

    public init(_ other: SwitchLinear, groupSize: Int = 64, bits: Int = 4,
                mode: QuantizationMode = .affine) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            other.weight, groupSize: groupSize, bits: bits, mode: mode)
        self.scales = scales
        self.biases = biases

        super.init(weight: quantizedWeight)
        self.freeze()
    }

    override public func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        gatherQuantizedMM(
            x, weight, scales: scales, biases: biases,
            rhsIndices: indices, transpose: true,
            groupSize: groupSize, bits: bits, mode: mode)
    }
}

/// The routed experts of one MoE layer: three ``SwitchLinear`` stacks forming a SwiGLU
/// evaluated only at the top-k experts of each token.
public final class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") public var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") public var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") public var downProj: SwitchLinear

    public init(hiddenSize: Int, intermediateSize: Int, numExperts: Int) {
        self._gateProj = ModuleInfo(
            wrappedValue: SwitchLinear(
                inputDims: hiddenSize, outputDims: intermediateSize, numExperts: numExperts),
            key: "gate_proj")
        self._upProj = ModuleInfo(
            wrappedValue: SwitchLinear(
                inputDims: hiddenSize, outputDims: intermediateSize, numExperts: numExperts),
            key: "up_proj")
        self._downProj = ModuleInfo(
            wrappedValue: SwitchLinear(
                inputDims: intermediateSize, outputDims: hiddenSize, numExperts: numExperts),
            key: "down_proj")
        super.init()
    }

    /// - Parameters:
    ///   - x: token hidden states `[T, hiddenSize]`
    ///   - indices: top-k expert indices `[T, k]`
    /// - Returns: per-expert outputs `[T, k, hiddenSize]`
    public func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let expanded = x.expandedDimensions(axes: [-2, -3])  // [T, 1, 1, H]
        let gate = gateProj(expanded, indices: indices)
        let up = upProj(expanded, indices: indices)
        let down = downProj(silu(gate) * up, indices: indices)  // [T, k, 1, H]
        return down.squeezed(axis: -2)
    }
}

// MARK: - Router

/// Sigmoid router with expert bias and group-limited top-k (phase-2 §2.5).
///
/// Dtype rules are load-bearing (§2.7): the routing matmul, sigmoid, and selection all
/// run in FP32 regardless of activation or stored-weight dtype; the expert bias affects
/// *selection only* — combination weights are gathered from the unbiased scores.
public final class LLaDA2MoEGate: Module {
    public let topK: Int
    public let numExperts: Int
    public let nGroup: Int
    public let topkGroup: Int
    public let routedScalingFactor: Float

    public let weight: MLXArray
    @ParameterInfo(key: "expert_bias") public var expertBias: MLXArray

    public init(
        hiddenSize: Int,
        numExperts: Int,
        numExpertsPerTok: Int,
        nGroup: Int,
        topkGroup: Int,
        routedScalingFactor: Float
    ) {
        self.topK = numExpertsPerTok
        self.numExperts = numExperts
        self.nGroup = nGroup
        self.topkGroup = topkGroup
        self.routedScalingFactor = routedScalingFactor
        self.weight = MLXArray.zeros([numExperts, hiddenSize])
        self._expertBias = ParameterInfo(
            wrappedValue: MLXArray.zeros([numExperts]), key: "expert_bias")
        super.init()
    }

    /// - Parameter x: token hidden states `[T, hiddenSize]`
    /// - Returns: `indices` `[T, topK]` (descending by biased masked score, matching
    ///   `torch.topk` order), `weights` `[T, topK]` FP32, `logits` `[T, numExperts]` FP32.
    public func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray, logits: MLXArray) {
        let logits = matmul(x.asType(.float32), weight.asType(.float32).transposed())
        let scores = sigmoid(logits)
        let scoresForRouting = scores + expertBias.asType(.float32)

        let tokens = scoresForRouting.dim(0)

        // Group score = sum of the top-2 expert scores in each group (top-2 is fixed in
        // the reference, independent of topK).
        let grouped = scoresForRouting.reshaped(tokens, nGroup, numExperts / nGroup)
        let groupScores = top(grouped, k: 2, axis: -1).sum(axis: -1)  // [T, nGroup]

        // Keep the top `topkGroup` groups; mask experts of dropped groups to -inf.
        let topGroups = argSort(-groupScores, axis: -1)[0..., ..<topkGroup]  // [T, kg]
        let groupIota = MLXArray(0 ..< Int32(nGroup))
        let groupMask = (topGroups.expandedDimensions(axis: -1) .== groupIota)
            .any(axis: 1)  // [T, nGroup]
        let scoreMask = broadcast(
            groupMask.expandedDimensions(axis: -1),
            to: [tokens, nGroup, numExperts / nGroup]
        ).reshaped(tokens, numExperts)

        let maskedScores = which(scoreMask, scoresForRouting, MLXArray(-Float.infinity))
        let indices = argSort(-maskedScores, axis: -1)[0..., ..<topK]  // [T, topK] descending

        // Combination weights from the *unbiased* scores, normalized, then scaled.
        var weights = takeAlong(scores, indices, axis: -1)
        if topK > 1 {
            weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        }
        weights = weights * routedScalingFactor

        return (indices, weights, logits)
    }
}

// MARK: - Sparse MoE block

/// One MoE layer (phase-2 §2.5): router + routed SwiGLU experts + always-on shared
/// expert. Expert outputs are combined in FP32 (matching the reference's
/// `.type(topk_weight.dtype)` upcast) and cast back to the activation dtype.
public final class LLaDA2SparseMoEBlock: Module {
    @ModuleInfo(key: "gate") public var gate: LLaDA2MoEGate
    @ModuleInfo(key: "experts") public var experts: SwitchGLU
    @ModuleInfo(key: "shared_experts") public var sharedExperts: LLaDA2MLP?

    public init(
        hiddenSize: Int,
        moeIntermediateSize: Int,
        numExperts: Int,
        numSharedExperts: Int,
        numExpertsPerTok: Int,
        nGroup: Int,
        topkGroup: Int,
        routedScalingFactor: Float
    ) {
        self._gate = ModuleInfo(
            wrappedValue: LLaDA2MoEGate(
                hiddenSize: hiddenSize, numExperts: numExperts,
                numExpertsPerTok: numExpertsPerTok, nGroup: nGroup, topkGroup: topkGroup,
                routedScalingFactor: routedScalingFactor),
            key: "gate")
        self._experts = ModuleInfo(
            wrappedValue: SwitchGLU(
                hiddenSize: hiddenSize, intermediateSize: moeIntermediateSize,
                numExperts: numExperts),
            key: "experts")
        self._sharedExperts = ModuleInfo(
            wrappedValue: numSharedExperts > 0
                ? LLaDA2MLP(
                    hiddenSize: hiddenSize,
                    intermediateSize: moeIntermediateSize * numSharedExperts)
                : nil,
            key: "shared_experts")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let flat = x.reshaped(-1, shape.last!)  // [T, H]

        let (indices, weights, _) = gate(flat)
        let expertOut = experts(flat, indices: indices)  // [T, k, H]

        var combined = (expertOut.asType(.float32) * weights.expandedDimensions(axis: -1))
            .sum(axis: 1)
            .asType(x.dtype)

        if let sharedExperts {
            combined = combined + sharedExperts(flat)
        }
        return combined.reshaped(shape)
    }
}
