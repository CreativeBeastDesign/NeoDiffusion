import Foundation
import MLX
import MLXNN
import DiffusionCore
import DiffusionModel

/// Public generation entry points (cache-disabled M5(a) parity anchor and the
/// ExactPrefixCache M5(b) path) plus the Elastic-Cache boundary read. Split out of the
/// engine core for readability (WP-4 refactor); same module, no behavioural change.
extension DiffusionEngine {
    // MARK: - Public entry point (cache-disabled)

    /// Cache-disabled generation — the M5(a) parity anchor: every step recomputes the full
    /// window under a fresh block mask, exactly like the reference recomputes its prefix
    /// (deviation 1 not yet applied). `streamBlock` receives each committed block's ids.
    ///
    /// `maskSemantics` defaults to `.strict` (NeoDiffusion's real semantics, §6);
    /// `.referenceBias` reproduces the stock-`generate()` 0/1 soft-bias mask and exists only
    /// for the M6 quality diagnostic (§6 item 4) — never serve it, and never combine it with
    /// the cached path (ExactPrefixCache's exactness argument holds only under `.strict`).
    public func generate(
        prompt: [Int],
        params: GenerationParams,
        maskSemantics: BlockDiffusionMask.Semantics = .strict,
        iceTemplate: [Int]? = nil,
        streamBlock: (([Int]) -> Void)? = nil
    ) -> Output {
        let forward: Forward = { [model] windowIds, activeLen, frozen in
            let W = windowIds.dim(windowIds.ndim - 1)
            let logits = model.logits(
                forTokens: windowIds, blockLength: params.blockLength,
                maskSemantics: maskSemantics)
            return logits[0..., (W - activeLen)..., 0...]
        }
        return run(prompt: prompt, params: params, iceTemplate: iceTemplate, forward: forward, streamBlock: streamBlock)
    }

    // MARK: - Public entry point (ExactPrefixCache enabled)

    /// Cache-enabled generation (phase-2 §5 M5(b), deviation 1). Each denoising step forwards
    /// **only the active block** against the committed-block K/V in ``ExactPrefixCache`` (no mask
    /// needed — every committed key is in an allowed ≤-current block), instead of recomputing the
    /// whole prefix. Proven identical to ``generate`` because a committed block's K/V equal what
    /// the full-window forward computes for those immutable, causal positions.
    ///
    /// Commit-cleanliness (gotcha 6): at each block commit a dedicated **capture forward** runs
    /// over the *final* committed tokens and its K/V is what gets appended — so speculative
    /// overshoot during denoising (which leaves stale `pending` K/V) never corrupts the cache.
    public func generateCached(
        prompt: [Int],
        params: GenerationParams,
        iceTemplate: [Int]? = nil,
        streamBlock: (([Int]) -> Void)? = nil
    ) -> Output {
        let B = params.blockLength
        let cache = ExactPrefixCache(layerCount: model.layerCount)
        let activeCache = ActiveBlockCache(layerCount: model.layerCount)
        // Faithful JOT (WP-3a v2): per-layer frozen-K/V hold. Only allocated/used when the
        // faithful mechanism is on; otherwise it stays inert (the v1 path ignores it).
        let jotCache = JotFreezeCache(layerCount: model.layerCount)
        let stats = RunStats()

        // Faithful JOT preconditions (see LayerJotCache / GenerationParams.jotFaithful):
        // the K/V hold mutates in-graph once per step and is not rolled back across a K>1
        // speculative batch, and it manages the active window's K/V (mutually exclusive with
        // Elastic-Cache and with a two-block window).
        if params.jotEnabled && params.jotFaithful {
            precondition(self.speculationK == 1,
                "faithful JOT requires speculationK == 1 — the frozen-K/V hold is not "
                + "snapshot/rolled-back across a speculative batch.")
            precondition(!params.elasticCacheEnabled,
                "faithful JOT and Elastic-Cache both manage the active-window K/V — enable at most one.")
            precondition(params.nBuf == 1,
                "faithful JOT v1 supports a single active block (nBuf == 1).")
            precondition(params.speculation == .none,
                "faithful JOT is not composed with S2D2 speculation (WP-3a v1 scope).")
        }
        if params.flashBlockEnabled {
            precondition(self.speculationK == 1,
                "FlashBlock requires speculationK == 1 — execution is synchronous on the Metal queue.")
            precondition(params.speculation == .none,
                "FlashBlock is not composed with speculation.")
            precondition(params.nBuf == 1,
                "FlashBlock integration currently supports a single active block (nBuf == 1).")
            precondition(!params.elasticCacheEnabled,
                "FlashBlock and Elastic-Cache cannot be enabled together.")
        }
        precondition(!params.subBlockCommit || (params.jotEnabled && params.jotFaithful),
            "sub-block prefix commit (Option C) requires faithful JOT (it keys off the frozen "
            + "mask and commits into the ExactPrefixCache); enable jotEnabled + jotFaithful.")

        // Capture forward over a full block of `ids` at absolute `startPos`, then commit its K/V.
        func captureAndCommit(_ ids: MLXArray, startPos: Int) {
            let positionIds = MLXArray(Int32(startPos) ..< Int32(startPos + ids.dim(1)))
                .expandedDimensions(axis: 0)
            _ = model(ids, positionIds: positionIds, caches: cache.layers)
            cache.commitBlock()
            stats.extraForwards += 1
            // Instrumented runs pin the capture forward's cost to the phase it belongs to
            // (prefill / commit) instead of letting it evaluate lazily inside the next
            // block's first denoising batch.
            if instrument {
                for layer in cache.layers {
                    if let k = layer.keys, let v = layer.values { eval(k, v) }
                }
            }
        }

        // Prefill: commit the pure-prompt blocks [0, prefillBlocks) into the cache so the first
        // generated block denoises against them. (Uniform with generated blocks — a prompt block
        // is just a pre-settled block; within-block bidirectional, cross-block causal via cache.)
        let prefillStart = Date()
        let prefillBlocks = prompt.count / B
        for b in 0 ..< prefillBlocks {
            let ids = MLXArray(prompt[b * B ..< (b + 1) * B].map { Int32($0) }).reshaped(1, B)
            captureAndCommit(ids, startPos: b * B)
        }
        stats.prefillSeconds = Date().timeIntervalSince(prefillStart)

        var stepIndex = 0

        let boundaryHolder = BoundaryHolder()

        // Memoized active-window masks for dual-active phases (WP-1b): constant across every
        // step of a phase — keyed by committed length since activeLen is 2B whenever used.
        var maskMemo: [Int: MLXArray] = [:]

        // Active-only forward: slice the active window out of the full window; positions are
        // absolute from the committed length (== window length − active length).
        let forward: Forward = { [self, model, cache, activeCache, jotCache, boundaryHolder] windowIds, activeLen, frozen in
            let W = windowIds.dim(windowIds.ndim - 1)
            let activeIds = windowIds[0..., (W - activeLen)...]
            let positionIds = MLXArray(Int32(W - activeLen) ..< Int32(W)).expandedDimensions(axis: 0)

            // Faithful JOT (WP-3a v2) / FlashBlock (WP-3b)
            if params.flashBlockEnabled || (params.jotEnabled && params.jotFaithful) {
                let frozenMaskVal = frozen ?? MLXArray.zeros([1, activeLen], dtype: .bool)
                let capacity: Int? = (params.moeCapacityRatio > 0 && params.jotEnabled)
                    ? Int((params.moeCapacityRatio * Float(activeLen)).rounded(.up))
                    : nil
                
                let isFirstStep = (stepIndex == 0)
                
                let dirtyCount: Int
                if let frozen {
                    let dirtyMask = (MLXArray.ones(like: frozen) - frozen.asType(.int32))
                    dirtyCount = Int(dirtyMask.sum().item(Int32.self))
                } else {
                    dirtyCount = activeLen
                }
                
                return model(activeIds, positionIds: positionIds, caches: cache.layers,
                             jotCaches: jotCache.layers, frozen: frozenMaskVal, mask: nil, capacity: capacity,
                             flashBlockEnabled: params.flashBlockEnabled, flashBlockTau: params.flashBlockTau,
                             isFirstStepOfBlock: isFirstStep, dirtyPerSeq: [dirtyCount])
            }

            // Elastic off (the served default): plain cached forward. The elastic overload
            // materializes full attention weights per layer for the drift test — that
            // instrumentation must never run on the serving path.
            guard params.elasticCacheEnabled else {
                // Single active block: mask nil (every committed key is allowed — the
                // ExactPrefixCache argument). Two active blocks: block-causal active mask
                // (the trailing block sees the front, never vice versa — WP-1b).
                guard activeLen > B else {
                    return model(activeIds, positionIds: positionIds, caches: cache.layers, mask: nil, frozen: frozen)
                }
                let prefixLen = W - activeLen
                let mask = maskMemo[prefixLen] ?? {
                    let m = BlockDiffusionMask.activeWindowMask(
                        prefixLen: prefixLen, activeLen: activeLen, blockLength: B)
                    maskMemo[prefixLen] = m
                    return m
                }()
                return model(activeIds, positionIds: positionIds, caches: cache.layers, mask: mask, frozen: frozen)
            }

            let prefixLen = W - activeLen

            var recomputeFlags = Array(repeating: true, count: model.layerCount)
            if stepIndex > 0 {
                if let staticBoundary = params.elasticStaticBoundary {
                    for l in 0 ..< model.layerCount {
                        recomputeFlags[l] = (l >= staticBoundary)
                    }
                } else if self.speculationK == 1 {
                    // Option 1: Live readback at each step
                    let boundary = self.readBoundary(activeCache: activeCache, layerCount: model.layerCount, gamma: params.elasticGamma)
                    boundaryHolder.value = boundary
                    for l in 0 ..< model.layerCount {
                        recomputeFlags[l] = (l >= boundary)
                    }
                } else {
                    // Proposal B: Use the delayed boundary from the previous batch
                    let boundary = boundaryHolder.value
                    for l in 0 ..< model.layerCount {
                        recomputeFlags[l] = (l >= boundary)
                    }
                }
            }

            let logits = model(activeIds, positionIds: positionIds, caches: cache.layers,
                               activeCache: activeCache, prefixLen: prefixLen,
                               recomputeActiveFlags: recomputeFlags)

            if params.elasticStaticBoundary == nil {
                var similarities: [MLXArray] = []
                for layer in activeCache.layers {
                    if let sim = layer.lastDriftSimilarity {
                        similarities.append(sim)
                    }
                }
                eval(similarities)
            }

            stepIndex += 1
            return logits
        }

        // S2D2 verifier forward (WP-2a): one 2B-wide forward under the M_ver mask at duplicated
        // absolute positions, against the committed prefix KV. Mask memoized per committed
        // length (constant across a block's denoising; commits happen only between phases).
        var verifierMaskMemo: [Int: MLXArray] = [:]
        let verifierForward: VerifierForward? = params.speculation == .s2d2
            ? { [model, cache] pairIds in
                let blockLen = pairIds.dim(1) / 2
                let prefixLen = cache.committedLength
                let blockPositions = MLXArray(Int32(prefixLen) ..< Int32(prefixLen + blockLen))
                let positionIds = concatenated([blockPositions, blockPositions], axis: 0)
                    .expandedDimensions(axis: 0)
                let mask = verifierMaskMemo[prefixLen] ?? {
                    let m = BlockDiffusionMask.s2d2VerifierMask(
                        prefixLen: prefixLen, blockLength: blockLen)
                    verifierMaskMemo[prefixLen] = m
                    return m
                }()
                return model(pairIds, positionIds: positionIds, caches: cache.layers, mask: mask)
            }
            : nil

        // On settle, run the capture forward over the committed tokens and commit clean K/V.
        let onSettled: (Int, MLXArray) -> Void = { _, committedActive in
            // Commit at the current committed length, not blockIndex*B — a block that already
            // sub-committed a prefix (Option C) starts its suffix past the block-aligned offset.
            captureAndCommit(committedActive, startPos: cache.committedLength)
            activeCache.clear()
            // Faithful JOT: the held K/V belong to the settled block's active window; the next
            // block starts with no frozen columns, so drop them (also avoids carrying a stale
            // window across a block boundary).
            jotCache.clear()
            stepIndex = 0
            boundaryHolder.value = 0
        }

        // Option C (WP-3a §11): commit a settled frozen prefix mid-block. Captures the prefix's KV
        // into the ExactPrefixCache at its absolute start and drops the held JOT columns for it.
        let onPrefixSettled: (Int, MLXArray) -> Void = { startPos, prefixIds in
            captureAndCommit(prefixIds, startPos: startPos)
            jotCache.dropPrefix(prefixIds.dim(1))
        }

        return run(prompt: prompt, params: params, iceTemplate: iceTemplate, activeCache: activeCache, boundaryHolder: boundaryHolder,
                   forward: forward, verifierForward: verifierForward,
                   streamBlock: streamBlock, onBlockSettled: onSettled,
                   onPrefixSettled: onPrefixSettled, stats: stats)
    }

    // Internal (not private): called from both the cached entry point and denoisePhase, which
    // live in separate files of this module after the WP-4 refactor.
    func readBoundary(activeCache: ActiveBlockCache, layerCount: Int, gamma: Float) -> Int {
        var simsList: [MLXArray] = []
        for l in 0 ..< layerCount {
            if let sim = activeCache.layers[l].lastDriftSimilarity {
                simsList.append(sim.reshaped([1]))
            } else {
                simsList.append(MLXArray(Float(-1.0)).reshaped([1]))
            }
        }
        let sims = concatenated(simsList, axis: 0)
        let stale = sims .< MLXArray(gamma)
        let indices = MLXArray(0 ..< Int32(layerCount))
        let staleIndices = which(stale, indices, MLXArray(Int32(layerCount)))
        let boundaryLayerTensor = staleIndices.min()
        return Int(boundaryLayerTensor.item(Int32.self))
    }
}
