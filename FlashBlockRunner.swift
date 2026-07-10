// FlashBlockRunner.swift
//
// Swift host for FlashBlock.metal. Loads the library, builds pipeline states
// with function constants, owns the paged KV cache and the (A_out, L_out)
// caches, and dispatches the two kernels with the τ / γ reuse gates.
//
// Designed to drop into NeoDiffusion. Assumes Q, K_current, V_current are
// produced by an upstream QKV-projection step (SDPA fusion notes are in the
// README).

import Foundation
import Metal

// MARK: - Types

public struct FlashBlockConfig {
    public var numSeqs: Int
    public var blockLen: Int            // B
    public var numQHeads: Int
    public var numKVHeads: Int
    public var headDim: Int
    public var pageSize: Int
    public var maxPagesPerSeq: Int

    // Tile sizes (must satisfy blockLen <= blockM).
    public var blockM: Int              // 16 or 32
    public var blockN: Int              // 64 or 128

    // Reuse gates.
    public var tau: Int                 // per-block dirty-token threshold
    public var composeGamma: Float      // per-head similarity threshold (video only)

    public init(numSeqs: Int, blockLen: Int,
                numQHeads: Int, numKVHeads: Int, headDim: Int,
                pageSize: Int = 256, maxPagesPerSeq: Int = 64,
                blockM: Int = 32, blockN: Int = 128,
                tau: Int = 4, composeGamma: Float = 0.0) {
        precondition(numQHeads % numKVHeads == 0, "GQA: numQHeads must be a multiple of numKVHeads")
        precondition(blockLen <= blockM, "blockLen must be <= blockM")
        precondition([64, 96, 128].contains(headDim), "headDim must be 64/96/128 for the constant-unroll path")
        self.numSeqs = numSeqs
        self.blockLen = blockLen
        self.numQHeads = numQHeads
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.pageSize = pageSize
        self.maxPagesPerSeq = maxPagesPerSeq
        self.blockM = blockM
        self.blockN = blockN
        self.tau = tau
        self.composeGamma = composeGamma
    }

    var kvGroup: Int { numQHeads / numKVHeads }
    var nqTotal: Int { numSeqs * blockLen }
}

// Matches FlashBlockParams in the .metal file. Layout must stay in sync.
struct FlashBlockParams {
    var num_seqs: Int32
    var block_len: Int32
    var num_q_heads: Int32
    var num_kv_heads: Int32
    var sm_scale: Float
    var compose_gamma: Float
    var use_dirty_gate: Int32
    var use_head_gamma: Int32
}

// MARK: - Runner

public final class FlashBlockRunner {

    public enum StepKind {
        case refreshCache  // first step of a block, or M^{s+1} >= tau  → Kernel A
        case reuseCache    // subsequent step, M^{s+1} <  tau           → Kernel B
    }

    let device: MTLDevice
    let queue: MTLCommandQueue
    let config: FlashBlockConfig

    let psoExternal: MTLComputePipelineState
    let psoInternal: MTLComputePipelineState

    // KV cache (paged) — owned externally; runner just holds references.
    public var kCache: MTLBuffer?
    public var vCache: MTLBuffer?
    public var blockTables: MTLBuffer?
    public var ctxLens: MTLBuffer?

    // FlashBlock caches (A_out, L_out).
    public let attnOutPast: MTLBuffer  // fp32, [numSeqs * blockLen * numQHeads * headDim]
    public let logsumexp:   MTLBuffer  // fp32, [numSeqs * blockLen * numQHeads]

    // Reuse control.
    public let dirtyMask: MTLBuffer    // uchar, [numSeqs * blockLen]
    public var headGamma: MTLBuffer?   // optional fp32 [numQHeads]

    let paramsBuf: MTLBuffer

    // MARK: init

    public init(device: MTLDevice,
                libraryURL: URL,
                config: FlashBlockConfig) throws {
        self.device = device
        guard let q = device.makeCommandQueue() else {
            throw NSError(domain: "FlashBlock", code: 1, userInfo: [NSLocalizedDescriptionKey: "no command queue"])
        }
        self.queue = q
        self.config = config

        let lib = try device.makeLibrary(URL: libraryURL)

        // ---- Build function constants ----
        let fc = MTLFunctionConstantValues()
        var blockM   = Int32(config.blockM);   fc.setConstantValue(&blockM,   type: .int, index: 0)
        var blockN   = Int32(config.blockN);   fc.setConstantValue(&blockN,   type: .int, index: 1)
        var headDim  = Int32(config.headDim);  fc.setConstantValue(&headDim,  type: .int, index: 2)
        var kvGroup  = Int32(config.kvGroup);  fc.setConstantValue(&kvGroup,  type: .int, index: 3)
        var pageSize = Int32(config.pageSize); fc.setConstantValue(&pageSize, type: .int, index: 4)
        var maxPages = Int32(config.maxPagesPerSeq); fc.setConstantValue(&maxPages, type: .int, index: 5)

        let fnExternal = try lib.makeFunction(name: "flashblock_external_pass", constantValues: fc)
        let fnInternal = try lib.makeFunction(name: "flashblock_internal_and_compose", constantValues: fc)

        self.psoExternal = try device.makeComputePipelineState(function: fnExternal)
        self.psoInternal = try device.makeComputePipelineState(function: fnInternal)

        // ---- Allocate caches ----
        let numRows = config.nqTotal * config.numQHeads
        let aoutBytes = numRows * config.headDim * MemoryLayout<Float>.size
        let lseBytes  = numRows * MemoryLayout<Float>.size

        guard
            let aout = device.makeBuffer(length: aoutBytes, options: .storageModePrivate),
            let lse  = device.makeBuffer(length: lseBytes,  options: .storageModePrivate),
            let dmask = device.makeBuffer(length: max(1, config.nqTotal), options: .storageModeShared),
            let params = device.makeBuffer(length: MemoryLayout<FlashBlockParams>.stride,
                                           options: .storageModeShared),
            let gammaDummy = device.makeBuffer(length: max(1, config.numQHeads) * MemoryLayout<Float>.size,
                                               options: .storageModeShared)
        else {
            throw NSError(domain: "FlashBlock", code: 2, userInfo: [NSLocalizedDescriptionKey: "buffer alloc failed"])
        }
        self.attnOutPast = aout
        self.logsumexp = lse
        self.dirtyMask = dmask
        self.paramsBuf = params
        self._gammaFallback = gammaDummy
    }

    // Always-bound fallback so head_gamma slot is never null.
    let _gammaFallback: MTLBuffer

    // MARK: dispatch

    /// Decide which kernel to run this step, given how many tokens in each
    /// block are dirty. If any block exceeds tau, we refresh the cache for
    /// all blocks — that matches the paper's per-block gate applied per
    /// forward pass. If you want per-sequence gating, split the batch.
    public func chooseStepKind(dirtyPerSeq: [Int], isFirstStepOfBlock: Bool) -> StepKind {
        if isFirstStepOfBlock { return .refreshCache }
        let anyOverTau = dirtyPerSeq.contains { $0 >= config.tau }
        return anyOverTau ? .refreshCache : .reuseCache
    }

    /// Encode the appropriate kernel into an existing command buffer.
    /// Q, K_cur, V_cur are the current block's projected tensors (half).
    /// output receives the full attention output (half).
    public func encode(commandBuffer: MTLCommandBuffer,
                       kind: StepKind,
                       Q: MTLBuffer, Kcur: MTLBuffer, Vcur: MTLBuffer,
                       output: MTLBuffer,
                       useDirtyGate: Bool = false) {

        // Fill uniform params.
        var params = FlashBlockParams(
            num_seqs:       Int32(config.numSeqs),
            block_len:      Int32(config.blockLen),
            num_q_heads:    Int32(config.numQHeads),
            num_kv_heads:   Int32(config.numKVHeads),
            sm_scale:       (1.0 / Float(config.headDim).squareRoot()) * Float(log2(M_E)),
            compose_gamma:  config.composeGamma,
            use_dirty_gate: useDirtyGate ? 1 : 0,
            use_head_gamma: headGamma != nil ? 1 : 0
        )
        memcpy(paramsBuf.contents(), &params, MemoryLayout<FlashBlockParams>.size)

        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }

        switch kind {
        case .refreshCache:
            enc.setComputePipelineState(psoExternal)
            enc.setBuffer(Q,          offset: 0, index: 0)
            enc.setBuffer(Kcur,       offset: 0, index: 1)
            enc.setBuffer(Vcur,       offset: 0, index: 2)
            enc.setBuffer(kCache,     offset: 0, index: 3)
            enc.setBuffer(vCache,     offset: 0, index: 4)
            enc.setBuffer(blockTables, offset: 0, index: 5)
            enc.setBuffer(ctxLens,    offset: 0, index: 6)
            enc.setBuffer(output,     offset: 0, index: 7)
            enc.setBuffer(attnOutPast, offset: 0, index: 8)
            enc.setBuffer(logsumexp,  offset: 0, index: 9)
            enc.setBuffer(paramsBuf,  offset: 0, index: 10)

            // Threadgroup memory: 2 * BLOCK_N * HEAD_DIM halfs.
            let tgBytes = 2 * config.blockN * config.headDim * MemoryLayout<Float16>.size
            enc.setThreadgroupMemoryLength(tgBytes, index: 0)

            enc.dispatchThreadgroups(
                MTLSize(width: config.numSeqs, height: config.numQHeads, depth: 1),
                threadsPerThreadgroup: MTLSize(width: config.blockM, height: 1, depth: 1)
            )

        case .reuseCache:
            enc.setComputePipelineState(psoInternal)
            enc.setBuffer(Q,           offset: 0, index: 0)
            enc.setBuffer(Kcur,        offset: 0, index: 1)
            enc.setBuffer(Vcur,        offset: 0, index: 2)
            enc.setBuffer(attnOutPast, offset: 0, index: 3)
            enc.setBuffer(logsumexp,   offset: 0, index: 4)
            enc.setBuffer(output,      offset: 0, index: 5)
            enc.setBuffer(dirtyMask,                     offset: 0, index: 6)
            enc.setBuffer(headGamma ?? _gammaFallback,   offset: 0, index: 7)
            enc.setBuffer(paramsBuf,                     offset: 0, index: 8)

            // Threadgroup memory: 2 * BLOCK_M * HEAD_DIM halfs.
            let tgBytes = 2 * config.blockM * config.headDim * MemoryLayout<Float16>.size
            enc.setThreadgroupMemoryLength(tgBytes, index: 0)

            enc.dispatchThreadgroups(
                MTLSize(width: config.numSeqs, height: config.numQHeads, depth: 1),
                threadsPerThreadgroup: MTLSize(width: config.blockM, height: 1, depth: 1)
            )
        }

        enc.endEncoding()
    }
}
