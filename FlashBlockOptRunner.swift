// FlashBlockOptRunner.swift
//
// Swift host for FlashBlockOpt.metal. Builds pipeline states for the two
// optimized kernels (external pass + compose-and-proj), wires up the extra
// buffers (RoPE tables, KV writeback slots, W_o), and exposes a single
// `stepOptimized` API that handles both the τ-gated refresh vs. reuse paths.
//
// Designed to sit next to FlashBlockRunner (which drives the simple kernels
// in FlashBlock.metal) so you can keep both in the tree and A/B them.

import Foundation
import Metal

// Matches FlashBlockParams in FlashBlockOpt.metal — layout must stay in sync.
struct FlashBlockOptParams {
    var num_seqs: Int32
    var block_len: Int32
    var num_q_heads: Int32
    var num_kv_heads: Int32
    var sm_scale: Float
    var compose_gamma: Float
    var use_dirty_gate: Int32
    var use_head_gamma: Int32
    var hidden_dim: Int32
    var rope_base_offset: Int32
    var _pad0: Int32 = 0
    var _pad1: Int32 = 0
}

public struct FlashBlockOptConfig {
    public var numSeqs: Int
    public var blockLen: Int
    public var numQHeads: Int
    public var numKVHeads: Int
    public var headDim: Int          // 64, 96, 128 (multiple of 8)
    public var hiddenDim: Int        // must be multiple of tnOut
    public var pageSize: Int
    public var maxPagesPerSeq: Int

    public var blockM: Int           // must be multiple of 8 and >= blockLen
    public var blockN: Int           // multiple of 8
    public var tnOut: Int            // output-dim tile per threadgroup (multiple of 8)
    public var hPerTg: Int           // 1, or 2 when headDim <= 64
    public var ropeOn: Bool

    public var tau: Int
    public var composeGamma: Float

    public init(numSeqs: Int, blockLen: Int,
                numQHeads: Int, numKVHeads: Int,
                headDim: Int, hiddenDim: Int,
                pageSize: Int = 256, maxPagesPerSeq: Int = 64,
                blockM: Int = 16, blockN: Int = 64, tnOut: Int = 64,
                hPerTg: Int = 1, ropeOn: Bool = true,
                tau: Int = 4, composeGamma: Float = 0) {
        precondition(headDim % 8 == 0)
        precondition(blockM % 8 == 0 && blockM >= blockLen)
        precondition(blockN % 8 == 0)
        precondition(tnOut % 8 == 0)
        precondition(hiddenDim % tnOut == 0, "hidden_dim must be divisible by tnOut")
        precondition(numQHeads % numKVHeads == 0)
        precondition(hPerTg == 1 || (hPerTg == 2 && headDim <= 64),
                     "hPerTg=2 requires headDim <= 64")
        precondition(numQHeads % hPerTg == 0)
        self.numSeqs = numSeqs
        self.blockLen = blockLen
        self.numQHeads = numQHeads
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.hiddenDim = hiddenDim
        self.pageSize = pageSize
        self.maxPagesPerSeq = maxPagesPerSeq
        self.blockM = blockM
        self.blockN = blockN
        self.tnOut = tnOut
        self.hPerTg = hPerTg
        self.ropeOn = ropeOn
        self.tau = tau
        self.composeGamma = composeGamma
    }

    var kvGroup: Int { numQHeads / numKVHeads }
    var nqTotal: Int { numSeqs * blockLen }
}

public final class FlashBlockOptRunner {

    public enum StepKind { case refreshCache, reuseCache }

    public let device: MTLDevice
    public let queue: MTLCommandQueue
    public let config: FlashBlockOptConfig

    let psoExternal: MTLComputePipelineState
    let psoComposeProj: MTLComputePipelineState

    // Host-owned buffers (bind before dispatch).
    public var kCache: MTLBuffer?
    public var vCache: MTLBuffer?
    public var blockTables: MTLBuffer?
    public var ctxLens: MTLBuffer?
    public var ropeCos: MTLBuffer?
    public var ropeSin: MTLBuffer?
    public var kvWriteSlots: MTLBuffer?
    public var Wo: MTLBuffer?

    // Owned caches.
    public let attnOutPast: MTLBuffer  // fp32 [Nq * H * D]
    public let logsumexp:  MTLBuffer   // fp32 [Nq * H]
    public let dirtyMask:  MTLBuffer   // u8   [Nq]
    public var headGamma: MTLBuffer?

    let paramsBuf: MTLBuffer
    let _gammaFallback: MTLBuffer

    public init(device: MTLDevice, libraryURL: URL, config: FlashBlockOptConfig) throws {
        self.device = device
        guard let q = device.makeCommandQueue() else {
            throw NSError(domain: "FlashBlockOpt", code: 1)
        }
        self.queue = q
        self.config = config

        let lib = try device.makeLibrary(URL: libraryURL)
        let fc = MTLFunctionConstantValues()
        var v: Int32 = 0
        v = Int32(config.blockM);         fc.setConstantValue(&v, type: .int, index: 0)
        v = Int32(config.blockN);         fc.setConstantValue(&v, type: .int, index: 1)
        v = Int32(config.headDim);        fc.setConstantValue(&v, type: .int, index: 2)
        v = Int32(config.kvGroup);        fc.setConstantValue(&v, type: .int, index: 3)
        v = Int32(config.pageSize);       fc.setConstantValue(&v, type: .int, index: 4)
        v = Int32(config.maxPagesPerSeq); fc.setConstantValue(&v, type: .int, index: 5)
        v = Int32(config.hPerTg);         fc.setConstantValue(&v, type: .int, index: 6)
        v = Int32(config.ropeOn ? 1 : 0); fc.setConstantValue(&v, type: .int, index: 7)
        v = Int32(config.tnOut);          fc.setConstantValue(&v, type: .int, index: 8)

        let fnExt  = try lib.makeFunction(name: "flashblock_external_pass_opt", constantValues: fc)
        let fnProj = try lib.makeFunction(name: "flashblock_compose_and_proj",  constantValues: fc)
        self.psoExternal    = try device.makeComputePipelineState(function: fnExt)
        self.psoComposeProj = try device.makeComputePipelineState(function: fnProj)

        let nqTotal = config.nqTotal
        let rows = nqTotal * config.numQHeads
        guard
            let a = device.makeBuffer(length: rows * config.headDim * MemoryLayout<Float>.size,
                                      options: .storageModePrivate),
            let l = device.makeBuffer(length: rows * MemoryLayout<Float>.size,
                                      options: .storageModePrivate),
            let d = device.makeBuffer(length: max(1, nqTotal), options: .storageModeShared),
            let p = device.makeBuffer(length: MemoryLayout<FlashBlockOptParams>.stride,
                                      options: .storageModeShared),
            let g = device.makeBuffer(length: max(1, config.numQHeads) * MemoryLayout<Float>.size,
                                      options: .storageModeShared)
        else { throw NSError(domain: "FlashBlockOpt", code: 2) }
        self.attnOutPast = a
        self.logsumexp = l
        self.dirtyMask = d
        self.paramsBuf = p
        self._gammaFallback = g
    }

    // Decide refresh vs. reuse based on the per-block dirty count.
    public func chooseStepKind(dirtyPerSeq: [Int], isFirstStepOfBlock: Bool) -> StepKind {
        if isFirstStepOfBlock { return .refreshCache }
        return dirtyPerSeq.contains { $0 >= config.tau } ? .refreshCache : .reuseCache
    }

    /// End-to-end optimized attention step:
    ///   refreshCache → runs the fused external pass (with RoPE + KV writeback),
    ///                  writes A_out/L_out and full O.
    ///   reuseCache   → runs compose-and-proj directly on Q/K/V and cached
    ///                  (A_out, L_out), producing Y = A_full · W_o.
    /// In the refresh path, the caller is responsible for applying W_o via a
    /// separate matmul on O — that keeps the refresh path simple and lets you
    /// share the W_o matmul path across all diffusion steps of the first-in-block.
    public func encode(commandBuffer cb: MTLCommandBuffer,
                       kind: StepKind,
                       Q: MTLBuffer, Kcur: MTLBuffer, Vcur: MTLBuffer,
                       O: MTLBuffer,           // used by refresh
                       Y: MTLBuffer? = nil,    // used by reuse
                       ropeBaseOffset: Int,
                       useDirtyGate: Bool = false) {

        var params = FlashBlockOptParams(
            num_seqs:         Int32(config.numSeqs),
            block_len:        Int32(config.blockLen),
            num_q_heads:      Int32(config.numQHeads),
            num_kv_heads:     Int32(config.numKVHeads),
            sm_scale:         (1.0 / Float(config.headDim).squareRoot()) * Float(log2(M_E)),
            compose_gamma:    config.composeGamma,
            use_dirty_gate:   useDirtyGate ? 1 : 0,
            use_head_gamma:   headGamma != nil ? 1 : 0,
            hidden_dim:       Int32(config.hiddenDim),
            rope_base_offset: Int32(ropeBaseOffset)
        )
        memcpy(paramsBuf.contents(), &params, MemoryLayout<FlashBlockOptParams>.size)

        guard let enc = cb.makeComputeCommandEncoder() else { return }

        switch kind {
        case .refreshCache:
            enc.setComputePipelineState(psoExternal)
            enc.setBuffer(Q,           offset: 0, index: 0)
            enc.setBuffer(Kcur,        offset: 0, index: 1)
            enc.setBuffer(Vcur,        offset: 0, index: 2)
            enc.setBuffer(kCache,      offset: 0, index: 3)
            enc.setBuffer(vCache,      offset: 0, index: 4)
            enc.setBuffer(blockTables, offset: 0, index: 5)
            enc.setBuffer(ctxLens,     offset: 0, index: 6)
            enc.setBuffer(O,           offset: 0, index: 7)
            enc.setBuffer(attnOutPast, offset: 0, index: 8)
            enc.setBuffer(logsumexp,   offset: 0, index: 9)
            enc.setBuffer(ropeCos ?? _gammaFallback, offset: 0, index: 10)
            enc.setBuffer(ropeSin ?? _gammaFallback, offset: 0, index: 11)
            enc.setBuffer(kvWriteSlots, offset: 0, index: 12)
            enc.setBuffer(paramsBuf,    offset: 0, index: 13)

            // Threadgroup memory: Q_tg (H_PER_TG * BLOCK_M * D) + K_tg + V_tg.
            let tgHalfs = config.hPerTg * config.blockM * config.headDim
                        + 2 * config.blockN * config.headDim
            enc.setThreadgroupMemoryLength(tgHalfs * MemoryLayout<Float16>.size, index: 0)

            let groups = MTLSize(width: config.numSeqs,
                                 height: config.numQHeads / config.hPerTg,
                                 depth: 1)
            // Threads/threadgroup = QM_TILES simdgroups × 32 = (BLOCK_M/8) * 32.
            let threads = MTLSize(width: (config.blockM / 8) * 32, height: 1, depth: 1)
            enc.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)

        case .reuseCache:
            precondition(Y != nil, "Y buffer required for reuseCache path (fused compose+proj)")
            enc.setComputePipelineState(psoComposeProj)
            enc.setBuffer(Q,           offset: 0, index: 0)
            enc.setBuffer(Kcur,        offset: 0, index: 1)
            enc.setBuffer(Vcur,        offset: 0, index: 2)
            enc.setBuffer(attnOutPast, offset: 0, index: 3)
            enc.setBuffer(logsumexp,   offset: 0, index: 4)
            enc.setBuffer(Wo,          offset: 0, index: 5)
            enc.setBuffer(Y!,          offset: 0, index: 6)
            enc.setBuffer(dirtyMask,   offset: 0, index: 7)
            enc.setBuffer(headGamma ?? _gammaFallback, offset: 0, index: 8)
            enc.setBuffer(ropeCos ?? _gammaFallback,   offset: 0, index: 9)
            enc.setBuffer(ropeSin ?? _gammaFallback,   offset: 0, index: 10)
            enc.setBuffer(paramsBuf,   offset: 0, index: 11)

            // Threadgroup memory: Q_tg + K_tg + V_tg + W_tg + Af_tg
            let tgHalfs = 3 * config.blockM * config.headDim
                        + config.headDim * config.tnOut
                        + config.blockM * config.headDim
            enc.setThreadgroupMemoryLength(tgHalfs * MemoryLayout<Float16>.size, index: 0)

            let groups = MTLSize(width: config.numSeqs,
                                 height: config.hiddenDim / config.tnOut,
                                 depth: 1)
            let threads = MTLSize(width: (config.blockM / 8) * 32, height: 1, depth: 1)
            enc.dispatchThreadgroups(groups, threadsPerThreadgroup: threads)
        }

        enc.endEncoding()
    }
}

// MARK: - Helper: precompute RoPE cos/sin tables in fp16.

public enum RoPETables {
    /// Standard rotary embedding (LLaMA/Qwen/SDAR half-rotation variant).
    /// Returns `(cos, sin)` buffers of shape `[maxPos, headDim/2]` in fp16.
    public static func makeBuffers(device: MTLDevice,
                                   maxPos: Int, headDim: Int,
                                   base: Double = 10_000.0) -> (MTLBuffer, MTLBuffer)? {
        precondition(headDim % 2 == 0)
        let halfD = headDim / 2
        let count = maxPos * halfD
        guard
            let cos = device.makeBuffer(length: count * MemoryLayout<Float16>.size,
                                        options: .storageModeShared),
            let sin = device.makeBuffer(length: count * MemoryLayout<Float16>.size,
                                        options: .storageModeShared)
        else { return nil }
        let cp = cos.contents().bindMemory(to: Float16.self, capacity: count)
        let sp = sin.contents().bindMemory(to: Float16.self, capacity: count)
        for pos in 0..<maxPos {
            for d in 0..<halfD {
                let theta = Double(pos) / pow(base, Double(2 * d) / Double(headDim))
                cp[pos * halfD + d] = Float16(Foundation.cos(theta))
                sp[pos * halfD + d] = Float16(Foundation.sin(theta))
            }
        }
        return (cos, sin)
    }
}
