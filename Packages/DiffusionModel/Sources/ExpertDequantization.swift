import Foundation
import MLX
import DiffusionCore
import MLXNN

/// **Diagnostic only.** Replaces every routed expert's quantized projections with pre-dequantized
/// FP16 ones, so the MoE dispatches `gatherMM` instead of `gatherQuantizedMM`.
///
/// ## Why: this is the direct causal test of Case B
///
/// Case B says the dequantization chain inside `gather_qmm` (unpack 4-bit, apply per-group
/// scale/bias in the accumulator loop) caps throughput via ALU work and register pressure. Every
/// attempt to size that from bandwidth models has failed (`Plans/gather_qmm_handoff.md` §5–§10).
/// This tests the premise itself by *removing dequantization entirely*:
///
/// | | 4-bit `gather_qmm` | FP16 `gatherMM` |
/// |---|---|---|
/// | expert bytes/forward | ~1.9 GB | **~7.7 GB (4×)** |
/// | dequant work | yes | **none** |
/// | FLOPs | identical | identical |
///
/// So the comparison is causally clean in the direction that matters:
/// - FP16 **faster** despite 4× the bytes ⇒ dequant costs more than the bandwidth quantization
///   saves ⇒ **Case B confirmed**, and the kernel is the right target.
/// - FP16 **slower** ⇒ the bandwidth saving dominates ⇒ dequant is not the bottleneck ⇒ Case B's
///   premise is wrong however the profiler reads.
///
/// Costs ~30 GB of RAM for `llada2.1-mini` — Studio only; do not attempt on the 16 GB dev M1.
public enum ExpertDequantization {
    /// Dequantize all routed experts in place. Router, shared expert, lm_head and embeddings are
    /// untouched — only the `gather_qmm` dispatch changes.
    ///
    /// Uses `leafModules()` + `update(modules:)`, mirroring `MLXNN.quantize` — MLX traps on direct
    /// assignment to a `@ModuleInfo` property ("please use Model.update(modules:) rather than
    /// mutating the Module property directly").
    ///
    /// - Returns: number of projections converted (expect 3 × MoE layers = 57 for llada2.1-mini).
    @discardableResult
    public static func dequantizeRoutedExperts(_ model: LLaDA2MoeModel) -> Int {
        var converted = 0
        // Update each SwitchGLU directly rather than the whole model. Model-wide paths look like
        // `model.layers.1.mlp.experts.gate_proj`, and layer 0 is a *dense* FFN with no experts —
        // so a model-level update yields `layers.1…19` with index 0 missing, MLX cannot rebuild
        // the layer *array* from a gappy index set, and `update(modules:)` traps with
        // `unexpectedStructure(key: "layers")`. (MLXNN.quantize avoids this only because it
        // touches every layer.) A SwitchGLU's children are three named projections — no array.
        for layer in model.model.layers {
            guard let moe = layer.mlp as? LLaDA2SparseMoEBlock else { continue }
            let glu = moe.experts
            var updates: [(String, Module)] = []
            if let d = dequantize(glu.gateProj) { updates.append(("gate_proj", d)) }
            if let d = dequantize(glu.upProj) { updates.append(("up_proj", d)) }
            if let d = dequantize(glu.downProj) { updates.append(("down_proj", d)) }
            guard !updates.isEmpty else { continue }
            glu.update(modules: ModuleChildren.unflattened(updates))
            converted += updates.count
        }
        eval(model)  // realise now, so conversion cost cannot leak into the first timed forward
        return converted
    }

    private static func dequantize(_ module: Module) -> SwitchLinear? {
        guard let q = module as? QuantizedSwitchLinear else { return nil }
        let w = dequantized(
            q.weight, scales: q.scales, biases: q.biases,
            groupSize: q.groupSize, bits: q.bits, mode: q.mode, dtype: .float16)
        return SwitchLinear(weight: w)
    }
}
