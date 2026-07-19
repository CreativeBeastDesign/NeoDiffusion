// MoEGatherQMVRunner.swift
//
// Step 2 of the kernel plan (`Plans/step5-kernel-handoff.md`, logbook
// `Plans/step5-kernel-logbook.md`): a walking-skeleton inline-MSL port of MLX's
// `affine_gather_qmv_fast` (4-bit, group 64) running the MoE routed-expert SwiGLU as two
// `MLXFast.metalKernel` dispatches instead of three `gatherQuantizedMM` calls. The port is
// deliberately a *naive structural copy* of the stock kernel (`quantized.h` `qmv_fast_impl`
// l.750: 2 simdgroups × 32 threads, 4 output rows per simdgroup, 16 values/thread, 512-value
// K-blocks, pre-scaled activations + masked-nibble dot, FP32 accumulate) so the equivalence
// gate is near-exact; register-pressure restructuring is Step 5's optimization loop, not this.
//
// Dtype contract: the kernels always run in float32 buffers (inputs cast on the way in —
// exact for f16 sources); the result is returned in the promoted stock output dtype
// (`outDType`), with the SwiGLU intermediate rounded through it to mirror stock's rounding
// points. CORRECTED 2026-07-19 (logbook post-close addendum): TRUE production is f16 x +
// f16 scales → stock dispatches the `_half_` gather (accumulating in float internally, like
// this kernel) and outDType is f16. The earlier claim that production runs `_float_` came
// from the Step 5a capture — which is the SYNTHETIC capture test (f32 scales), not serving.
//
// Default-off behind `MoEFusedQMVConfig.enabled`; the stock `gatherQuantizedMM` path is
// untouched when the flag is off or the fast-path preconditions in `SwitchGLU` aren't met
// (4-bit affine group-64 with biases, `hiddenSize` and `intermediateSize` multiples of 512 —
// the same alignment family that gates stock `qmv_fast` vs `qmv`).

import Foundation
import MLX
import MLXNN

/// Default-off switch for the fused gather-QMV kernel path.
/// Settable (not just env-derived) so tests can toggle it without env-var games; still seeded
/// from `NEODIFFUSION_FUSED_QMV=1` for shell opt-in.
public enum MoEFusedQMVConfig {
    // `nonisolated(unsafe)`: same pattern as `moeRoutingTrace` in `RoutingTrace.swift` — a
    // process-wide test/opt-in toggle, not per-request state; callers are responsible for not
    // flipping it across concurrent generations.
    nonisolated(unsafe) public static var enabled: Bool =
        ProcessInfo.processInfo.environment["NEODIFFUSION_FUSED_QMV"] == "1"

    /// Incremented once per `MoEGatherQMVRunner.forward` call so tests/benches can prove the
    /// fused path actually ran (an ineligible shape silently falls back to stock).
    nonisolated(unsafe) public private(set) static var dispatchCount: Int = 0

    nonisolated(unsafe) private static var didLogActivation = false

    static func recordDispatch() {
        dispatchCount += 1
        if !didLogActivation {
            didLogActivation = true
            // Effective-echo (AGENTS.md): serving/bench logs must show the fused path actually
            // engaged, not just that the env var was set.
            print("[fused-qmv] active: first fused gather-QMV dispatch in this process")
        }
    }
}

/// Runs one MoE layer's routed-expert SwiGLU (`gate_proj` / `up_proj` / `down_proj`) through two
/// hand-written `MLXFast.metalKernel` dispatches instead of three `gatherQuantizedMM` calls.
/// Modeled directly on ``FlashBlockRunner``: kernels are compiled once in `init` and cached as
/// instance properties, with dimensions baked in as header `#define`s (compile-time
/// specialization) rather than passed as per-call arguments.
public final class MoEGatherQMVRunner {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numExperts: Int
    public let topK: Int
    public let groupSize: Int
    public let bits: Int

    private let gateUpKernel: MLXFast.MLXFastKernel
    private let downKernel: MLXFast.MLXFastKernel

    /// Preconditions the MSL is specialized for; `SwitchGLU` checks these before constructing.
    public static func isEligible(hiddenSize: Int, intermediateSize: Int, groupSize: Int, bits: Int)
        -> Bool
    {
        // 512 = block_size of the qmv_fast K-loop (16 values/thread × 32 lanes); both matmul
        // input dims must be whole blocks, and 512-multiples also satisfy the 8-row tiling.
        groupSize == 64 && bits == 4 && hiddenSize % 512 == 0 && intermediateSize % 512 == 0
    }

    public init(
        hiddenSize: Int, intermediateSize: Int, numExperts: Int, topK: Int,
        groupSize: Int, bits: Int
    ) {
        precondition(
            Self.isEligible(
                hiddenSize: hiddenSize, intermediateSize: intermediateSize,
                groupSize: groupSize, bits: bits),
            "MoEGatherQMVRunner: caller must check isEligible before constructing")
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numExperts = numExperts
        self.topK = topK
        self.groupSize = groupSize
        self.bits = bits

        // Baked constants + shared helpers as a header (FlashBlockRunner pattern) — no
        // `#include`/`using namespace` needed, the JIT wrapper already provides `metal_stdlib`.
        //
        // The helpers are line-for-line ports of `quantized.h`'s `load_vector` (bits==4, l.62)
        // and `qdot` (bits==4, l.235): activations are pre-divided by 16/256/4096 so the dot can
        // multiply masked-but-unshifted nibbles from uint16 reads, and the affine dequant folds
        // into `scale * accum + sum * bias` per 512-block (`qdot` return, l.289).
        let header = """
        #define H_DIM \(hiddenSize)
        #define I_DIM \(intermediateSize)
        #define N_EXPERTS \(numExperts)
        #define K_TOP \(topK)
        #define GROUP_SIZE \(groupSize)

        template <int VPT>
        inline float load_xn(const device float* x, thread float* x_thread) {
            float sum = 0.0f;
            for (int i = 0; i < VPT; i += 4) {
                float x0 = x[i], x1 = x[i + 1], x2 = x[i + 2], x3 = x[i + 3];
                sum += x0 + x1 + x2 + x3;
                x_thread[i] = x0;
                x_thread[i + 1] = x1 / 16.0f;
                x_thread[i + 2] = x2 / 256.0f;
                x_thread[i + 3] = x3 / 4096.0f;
            }
            return sum;
        }

        template <int VPT>
        inline float qdotn(
            const device uint8_t* w, const thread float* x_thread,
            float scale, float bias, float sum
        ) {
            float accum = 0.0f;
            const device uint16_t* ws = (const device uint16_t*)w;
            for (int i = 0; i < VPT / 4; i++) {
                accum += x_thread[4 * i] * (ws[i] & 0x000f)
                    + x_thread[4 * i + 1] * (ws[i] & 0x00f0)
                    + x_thread[4 * i + 2] * (ws[i] & 0x0f00)
                    + x_thread[4 * i + 3] * (ws[i] & 0xf000);
            }
            return scale * accum + sum * bias;
        }

        // One simdgroup's share of a single (token, expert) GEMV: ROWS consecutive output rows,
        // K-loop in 512-value blocks — stock `qmv_fast_impl` is ROWS=4; ROWS=8 halves x-load
        // traffic per output element (+4 accumulator registers). Pointers are expert-slab
        // bases; `xr` is the input row; `yr` is the pair's output row.
        template <int IN_DIM, int ROWS, int VPT>
        inline void qmv_rows(
            const device uint8_t* ws, const device float* sl, const device float* bl,
            const device float* xr, device float* yr,
            int out_row, uint simd_lid
        ) {
            const int w_row = IN_DIM / 2;          // packed 4-bit bytes per output row
            const int g_row = IN_DIM / GROUP_SIZE; // quant groups per output row
            const int blk = VPT * 32;              // K-loop block: VPT values × 32 lanes

            ws += out_row * w_row + simd_lid * (VPT / 2);      // VPT/8 packs × 4 bytes
            sl += out_row * g_row + simd_lid / (GROUP_SIZE / VPT);
            bl += out_row * g_row + simd_lid / (GROUP_SIZE / VPT);
            xr += simd_lid * VPT;

            thread float x_thread[VPT];
            thread float result[ROWS];
            for (int row = 0; row < ROWS; row++) { result[row] = 0.0f; }

            for (int k = 0; k < IN_DIM; k += blk) {
                float sum = load_xn<VPT>(xr, x_thread);
                for (int row = 0; row < ROWS; row++) {
                    result[row] += qdotn<VPT>(
                        ws + row * w_row, x_thread,
                        sl[row * g_row], bl[row * g_row], sum);
                }
                ws += blk / 2;
                sl += blk / GROUP_SIZE;
                bl += blk / GROUP_SIZE;
                xr += blk;
            }

            for (int row = 0; row < ROWS; row++) {
                float r = simd_sum(result[row]);
                if (simd_lid == 0) {
                    yr[out_row + row] = r;
                }
            }
        }

        """

        // Grid contract for both kernels (MLX grid = total threads):
        //   grid = (64, OUT_DIM / 8, T * K_TOP), threadGroup = (64, 1, 1)
        //   → one threadgroup = 2 simdgroups × 4 rows = 8 output rows of one (token, expert)
        //   pair; tid.y picks the row group, tid.z the pair. Same tiling as stock.
        let gateUpSource = """
            uint3 tid = threadgroup_position_in_grid;
            uint simd_gid = simdgroup_index_in_threadgroup;
            uint simd_lid = thread_index_in_simdgroup;

            const uint pair = tid.z;
            const uint token = pair / K_TOP;
            const uint expert = uint(expertIdx[pair]);

            const ulong wSlab = (ulong)expert * I_DIM * (H_DIM / 2);
            const ulong gSlab = (ulong)expert * I_DIM * (H_DIM / GROUP_SIZE);
            const device float* xr = x + (ulong)token * H_DIM;
            const int out_row = tid.y * 16 + simd_gid * 8;

            qmv_rows<H_DIM, 8, 8>(
                (const device uint8_t*)wG + wSlab, sG + gSlab, bG + gSlab,
                xr, outGate + (ulong)pair * I_DIM, out_row, simd_lid);
            qmv_rows<H_DIM, 8, 8>(
                (const device uint8_t*)wU + wSlab, sU + gSlab, bU + gSlab,
                xr, outUp + (ulong)pair * I_DIM, out_row, simd_lid);
        """

        let downSource = """
            uint3 tid = threadgroup_position_in_grid;
            uint simd_gid = simdgroup_index_in_threadgroup;
            uint simd_lid = thread_index_in_simdgroup;

            const uint pair = tid.z;
            const uint expert = uint(expertIdx[pair]);

            const ulong wSlab = (ulong)expert * H_DIM * (I_DIM / 2);
            const ulong gSlab = (ulong)expert * H_DIM * (I_DIM / GROUP_SIZE);
            const device float* xr = x + (ulong)pair * I_DIM;
            const int out_row = tid.y * 16 + simd_gid * 8;

            qmv_rows<I_DIM, 8, 8>(
                (const device uint8_t*)wD + wSlab, sD + gSlab, bD + gSlab,
                xr, outDown + (ulong)pair * H_DIM, out_row, simd_lid);
        """

        self.gateUpKernel = MLXFast.metalKernel(
            name: "moe_gather_qmv_gate_up",
            inputNames: ["x", "wG", "sG", "bG", "wU", "sU", "bU", "expertIdx"],
            outputNames: ["outGate", "outUp"],
            source: gateUpSource,
            header: header
        )

        self.downKernel = MLXFast.metalKernel(
            name: "moe_gather_qmv_down",
            inputNames: ["x", "wD", "sD", "bD", "expertIdx"],
            outputNames: ["outDown"],
            source: downSource,
            header: header
        )
    }

    /// - Parameters:
    ///   - x: token hidden states `[T, hiddenSize]`
    ///   - indices: top-k expert indices `[T, k]`, `k == topK`
    ///   - gate/up/down: the layer's quantized projections (4-bit affine group-64; caller
    ///     verifies eligibility before constructing/using the runner)
    /// - Returns: per-expert outputs `[T, k, hiddenSize]` in `x.dtype`, matching the stock
    ///   `SwitchGLU` contract.
    public func forward(
        x: MLXArray, indices: MLXArray,
        gate: QuantizedSwitchLinear, up: QuantizedSwitchLinear, down: QuantizedSwitchLinear
    ) -> MLXArray {
        guard let bG = gate.biases, let bU = up.biases, let bD = down.biases else {
            fatalError(
                "MoEGatherQMVRunner.forward requires affine-mode biases; caller must gate on this")
        }
        MoEFusedQMVConfig.recordDispatch()

        let T = x.dim(0)
        let k = indices.dim(-1)
        let expertIdx = indices.reshaped([-1]).asType(.int32)

        // Stock `gatherQuantizedMM` returns the promoted type of x and scales (fp16 x with fp32
        // scales → fp32), and its SwiGLU intermediate lives in that dtype too. Mirror it exactly
        // — rounding the intermediate through a narrower dtype than stock costs ~1e1 absolute
        // error through the 512-term down GEMV (measured, step-2 first run).
        let outDType: DType = x.dtype == gate.scales.dtype ? x.dtype : .float32

        let gateUpResults = gateUpKernel(
            [
                x.asType(.float32),
                gate.weight, gate.scales.asType(.float32), bG.asType(.float32),
                up.weight, up.scales.asType(.float32), bU.asType(.float32),
                expertIdx,
            ],
            grid: (64, intermediateSize / 16, T * k),
            threadGroup: (64, 1, 1),
            outputShapes: [[T * k, intermediateSize], [T * k, intermediateSize]],
            outputDTypes: [.float32, .float32]
        )

        // SwiGLU combine stays MLX-side, rounded through the stock output dtype so the down
        // kernel sees the same intermediate values the stock path's `silu(gate) * up` produces.
        let gateAct = gateUpResults[0].asType(outDType)
        let upAct = gateUpResults[1].asType(outDType)
        let glu = (silu(gateAct) * upAct).asType(.float32)

        let downResults = downKernel(
            [glu, down.weight, down.scales.asType(.float32), bD.asType(.float32), expertIdx],
            grid: (64, hiddenSize / 16, T * k),
            threadGroup: (64, 1, 1),
            outputShapes: [[T * k, hiddenSize]],
            outputDTypes: [.float32]
        )

        return downResults[0].reshaped([T, k, hiddenSize]).asType(outDType)
    }
}
