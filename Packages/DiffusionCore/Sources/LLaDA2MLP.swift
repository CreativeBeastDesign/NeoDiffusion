import Foundation
import MLX
import MLXNN

/// SwiGLU feed-forward matching the reference `LLaDA2MoeMLP`:
/// `down(silu(gate(x)) * up(x))`. The reference hardcodes `bias=False` for all three
/// projections (it ignores `use_bias`), so no bias option is exposed here.
public final class LLaDA2MLP: Module {
    @ModuleInfo(key: "gate_proj") public var gateProj: Linear
    @ModuleInfo(key: "up_proj") public var upProj: Linear
    @ModuleInfo(key: "down_proj") public var downProj: Linear

    public init(hiddenSize: Int, intermediateSize: Int) {
        self._gateProj = ModuleInfo(
            wrappedValue: Linear(hiddenSize, intermediateSize, bias: false), key: "gate_proj")
        self._upProj = ModuleInfo(
            wrappedValue: Linear(hiddenSize, intermediateSize, bias: false), key: "up_proj")
        self._downProj = ModuleInfo(
            wrappedValue: Linear(intermediateSize, hiddenSize, bias: false), key: "down_proj")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}
