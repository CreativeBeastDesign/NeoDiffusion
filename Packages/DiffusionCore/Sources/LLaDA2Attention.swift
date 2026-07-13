import Foundation
import MLX
import MLXNN
import Metal

/// Self-attention block matching the reference `LLaDA2MoeAttention` (phase-2 §2.3):
/// fused QKV (split order 16 Q / 4 K / 4 V on the head axis), optional per-head-dim
/// qk-norm *before* RoPE, partial RoPE, SDPA with FP32 softmax, GQA via native
/// MLX SDPA (no pre-tiling of K/V), then the `dense` output projection.
public final class LLaDA2Attention: Module {
    public let numHeads: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let scale: Float

    @ModuleInfo(key: "query_key_value") public var queryKeyValue: Linear
    @ModuleInfo(key: "query_layernorm") public var queryLayernorm: LLaDA2RMSNorm?
    @ModuleInfo(key: "key_layernorm") public var keyLayernorm: LLaDA2RMSNorm?
    @ModuleInfo(key: "dense") public var dense: Linear

    public init(
        hiddenSize: Int,
        numHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        rmsNormEps: Float,
        useQkNorm: Bool = true,
        useQkvBias: Bool = false,
        useDenseBias: Bool = false
    ) {
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.scale = pow(Float(headDim), -0.5)

        let qkvDims = (numHeads + 2 * numKVHeads) * headDim
        self._queryKeyValue = ModuleInfo(
            wrappedValue: Linear(hiddenSize, qkvDims, bias: useQkvBias), key: "query_key_value")
        // qk-norm is per-head-dim (reference: RMSNorm(head_dim)), applied on [B, H, L, D]
        self._queryLayernorm = ModuleInfo(
            wrappedValue: useQkNorm ? LLaDA2RMSNorm(dimensions: headDim, eps: rmsNormEps) : nil,
            key: "query_layernorm")
        self._keyLayernorm = ModuleInfo(
            wrappedValue: useQkNorm ? LLaDA2RMSNorm(dimensions: headDim, eps: rmsNormEps) : nil,
            key: "key_layernorm")
        self._dense = ModuleInfo(
            wrappedValue: Linear(numHeads * headDim, hiddenSize, bias: useDenseBias), key: "dense")

        super.init()
    }

    /// - Parameters:
    ///   - x: hidden states `[B, L, hiddenSize]`
    ///   - mask: additive attention mask `[1, 1, L, L]` (see ``BlockDiffusionMask``)
    ///   - cos/sin: FP32 rotary tables `[B, L, rotaryDim]` from ``PartialRotaryEmbedding``
    public func callAsFunction(
        _ x: MLXArray, mask: MLXArray?, cos: MLXArray, sin: MLXArray
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let qkv = queryKeyValue(x).reshaped(B, L, numHeads + 2 * numKVHeads, headDim)
        let parts = split(qkv, indices: [numHeads, numHeads + numKVHeads], axis: 2)
        var queries = parts[0].transposed(0, 2, 1, 3)  // [B, nH, L, D]
        var keys = parts[1].transposed(0, 2, 1, 3)     // [B, nKV, L, D]
        let values = parts[2].transposed(0, 2, 1, 3)   // [B, nKV, L, D]

        if let queryLayernorm { queries = queryLayernorm(queries) }
        if let keyLayernorm { keys = keyLayernorm(keys) }

        queries = PartialRotaryEmbedding.apply(queries, cos: cos, sin: sin)
        keys = PartialRotaryEmbedding.apply(keys, cos: cos, sin: sin)

        let attended = Self.attend(queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        let output = attended
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, numHeads * headDim)

        return dense(output)
    }

    /// Cache-aware attention (phase-2 §5 M5): compute Q/K/V for the active window only, attend
    /// against `[cache.keys ++ activeK]` with **no mask** (every committed key is in an allowed
    /// ≤-current block), and stash the active-block K/V into `cache.pending*` for a later commit.
    ///
    /// Produces bitwise-identical active-position outputs to the uncached ``callAsFunction`` full
    /// forward under `.strict`, because the committed K/V equal what the full forward computes for
    /// those (immutable, causal) positions — the ExactPrefixCache exactness argument (deviation 1).
    ///
    /// - Parameters:
    ///   - x: active-window hidden states `[1, B, hiddenSize]`
    ///   - cos/sin: rotary tables `[1, B, rotaryDim]` for the active window's **absolute** positions
    ///   - cache: this layer's committed K/V store; updated in place (`pending*` set)
    ///   - mask: optional additive mask `[1, 1, activeLen, committedLen + activeLen]` — nil for
    ///     a single active block (all keys allowed); WP-1b passes
    ///     ``BlockDiffusionMask/activeWindowMask(prefixLen:activeLen:blockLength:dtype:)`` when
    ///     the window holds two concurrently active blocks (block-causal between them).
    public func callAsFunction(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray, cache: LayerKVCache,
        mask: MLXArray? = nil
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let qkv = queryKeyValue(x).reshaped(B, L, numHeads + 2 * numKVHeads, headDim)
        let parts = split(qkv, indices: [numHeads, numHeads + numKVHeads], axis: 2)
        var queries = parts[0].transposed(0, 2, 1, 3)
        var activeKeys = parts[1].transposed(0, 2, 1, 3)
        let activeValues = parts[2].transposed(0, 2, 1, 3)

        if let queryLayernorm { queries = queryLayernorm(queries) }
        if let keyLayernorm { activeKeys = keyLayernorm(activeKeys) }

        queries = PartialRotaryEmbedding.apply(queries, cos: cos, sin: sin)
        activeKeys = PartialRotaryEmbedding.apply(activeKeys, cos: cos, sin: sin)

        // Attend against committed prefix ++ active. `mask` is nil for a single active block
        // (all committed keys allowed); a two-block window passes the block-causal active mask.
        let keys = cache.keys.map { concatenated([$0, activeKeys], axis: 2) } ?? activeKeys
        let values = cache.values.map { concatenated([$0, activeValues], axis: 2) } ?? activeValues

        cache.pendingKeys = activeKeys
        cache.pendingValues = activeValues

        let attended = Self.attend(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        let output = attended
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, numHeads * headDim)
        return dense(output)
    }

    /// Faithful-JOT cache-aware attention (WP-3a v2) + FlashBlock (WP-3b).
    public func callAsFunction(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray, cache: LayerKVCache,
        jot: LayerJotCache, frozen: MLXArray, mask: MLXArray? = nil,
        flashBlockEnabled: Bool = false, flashBlockTau: Int = 4,
        isFirstStepOfBlock: Bool = false, dirtyPerSeq: [Int] = [0]
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let qkv = queryKeyValue(x).reshaped(B, L, numHeads + 2 * numKVHeads, headDim)
        let parts = split(qkv, indices: [numHeads, numHeads + numKVHeads], axis: 2)
        var queries = parts[0].transposed(0, 2, 1, 3)
        var freshKeys = parts[1].transposed(0, 2, 1, 3)   // [1, nKV, L, D]
        let freshValues = parts[2].transposed(0, 2, 1, 3)

        if let queryLayernorm { queries = queryLayernorm(queries) }
        if let keyLayernorm { freshKeys = keyLayernorm(freshKeys) }

        queries = PartialRotaryEmbedding.apply(queries, cos: cos, sin: sin)
        freshKeys = PartialRotaryEmbedding.apply(freshKeys, cos: cos, sin: sin)

        // Hold frozen columns at their captured value; refresh the rest. Broadcast `frozen`
        // `[1, L]` over the head (nKV) and head-dim (D) axes → `[1, 1, L, 1]`.
        let frozenKV = frozen.reshaped(1, 1, L, 1)
        let activeKeys = jot.keys.map { which(frozenKV, $0, freshKeys) } ?? freshKeys
        let activeValues = jot.values.map { which(frozenKV, $0, freshValues) } ?? freshValues
        jot.keys = activeKeys
        jot.values = activeValues

        let keys = cache.keys.map { concatenated([$0, activeKeys], axis: 2) } ?? activeKeys
        let values = cache.values.map { concatenated([$0, activeValues], axis: 2) } ?? activeValues

        cache.pendingKeys = activeKeys
        cache.pendingValues = activeValues

        if flashBlockEnabled {
            let device = MTLCreateSystemDefaultDevice()!
            
            // Lazy init FlashBlockRunner on the cache
            if cache.flashBlockRunner == nil {
                let repoRoot = FileManager.default.currentDirectoryPath
                let libraryURL = URL(fileURLWithPath: "\(repoRoot)/FlashBlock.metallib")
                let config = FlashBlockConfig(
                    numSeqs: B,
                    blockLen: L,
                    numQHeads: numHeads,
                    numKVHeads: numKVHeads,
                    headDim: headDim,
                    pageSize: 256,
                    maxPagesPerSeq: 64,
                    tau: flashBlockTau
                )
                do {
                    cache.flashBlockRunner = try FlashBlockRunner(
                        device: device,
                        libraryURL: libraryURL,
                        config: config
                    )
                    
                    let maxPages = 64
                    cache.blockTablesBuffer = device.makeBuffer(length: maxPages * MemoryLayout<UInt32>.size, options: .storageModeShared)
                    if let table = cache.blockTablesBuffer?.contents().bindMemory(to: UInt32.self, capacity: maxPages) {
                        for i in 0..<maxPages {
                            table[i] = UInt32(i)
                        }
                    }
                    cache.ctxLensBuffer = device.makeBuffer(length: MemoryLayout<Int32>.size, options: .storageModeShared)
                } catch {
                    fatalError("Failed to initialize FlashBlockRunner: \(error)")
                }
            }
            
            guard let runner = cache.flashBlockRunner else {
                fatalError("FlashBlockRunner is nil")
            }
            
            // Transpose inputs to FlashBlock layout: [Nq, H, D] where Nq = B * L
            let qContiguous = queries.transposed(0, 2, 1, 3).reshaped(B * L, numHeads, headDim).contiguous()
            let kContiguous = activeKeys.transposed(0, 2, 1, 3).reshaped(B * L, numKVHeads, headDim).contiguous()
            let vContiguous = activeValues.transposed(0, 2, 1, 3).reshaped(B * L, numKVHeads, headDim).contiguous()
            
            // Get MTLBuffers
            guard let qBuf = qContiguous.asMTLBuffer(device: device, noCopy: true),
                  let kBuf = kContiguous.asMTLBuffer(device: device, noCopy: true),
                  let vBuf = vContiguous.asMTLBuffer(device: device, noCopy: true) else {
                fatalError("Failed to get MTLBuffer for Q/K/V")
            }
            
            // Get caches
            let committedLen = cache.committedLength
            guard committedLen > 0, let cacheKeys = cache.keys, let cacheValues = cache.values else {
                fatalError("ExactPrefixCache keys/values are empty but we are in decoding loop")
            }
            
            let cacheKeysContiguous = cacheKeys.transposed(0, 2, 1, 3).reshaped(committedLen, numKVHeads, headDim).contiguous()
            let cacheValuesContiguous = cacheValues.transposed(0, 2, 1, 3).reshaped(committedLen, numKVHeads, headDim).contiguous()
            
            guard let cacheKBuf = cacheKeysContiguous.asMTLBuffer(device: device, noCopy: true),
                  let cacheVBuf = cacheValuesContiguous.asMTLBuffer(device: device, noCopy: true) else {
                fatalError("Failed to get MTLBuffer for cache K/V")
            }
            
            // Bind references
            runner.kCache = cacheKBuf
            runner.vCache = cacheVBuf
            runner.blockTables = cache.blockTablesBuffer
            
            // Write ctx lens
            if let ctxLensPtr = cache.ctxLensBuffer?.contents().bindMemory(to: Int32.self, capacity: 1) {
                ctxLensPtr[0] = Int32(committedLen)
            }
            runner.ctxLens = cache.ctxLensBuffer
            
            // Populate dirty mask: dirty_mask is uchar [numSeqs * blockLen]
            // We set it to 1 - frozen Mask. frozen is shape [1, L] Bool
            let dirtyMaskArray = (MLXArray.ones(like: frozen) - frozen.asType(.int32)).asType(.uint8)
            let dirtyVals = dirtyMaskArray.asArray(UInt8.self)
            let destPtr = runner.dirtyMask.contents().bindMemory(to: UInt8.self, capacity: L)
            for i in 0..<L {
                destPtr[i] = dirtyVals[i]
            }
            
            // Allocate output array and get its buffer
            let outputArray = MLXArray.zeros([B * L, numHeads, headDim], dtype: queries.dtype)
            guard let outputBuf = outputArray.asMTLBuffer(device: device, noCopy: true) else {
                fatalError("Failed to allocate output MTLBuffer")
            }
            
            // Decide step kind
            let stepKind = runner.chooseStepKind(dirtyPerSeq: dirtyPerSeq, isFirstStepOfBlock: isFirstStepOfBlock)
            
            // Create command buffer, encode, commit and wait
            guard let cb = runner.queue.makeCommandBuffer() else {
                fatalError("Failed to create MTLCommandBuffer")
            }
            
            runner.encode(
                commandBuffer: cb,
                kind: stepKind,
                Q: qBuf,
                Kcur: kBuf,
                Vcur: vBuf,
                output: outputBuf,
                useDirtyGate: true
            )
            
            // Wait for GPU stream synchronization before we launch our command buffer
            Stream.gpu.synchronize()
            
            cb.commit()
            cb.waitUntilCompleted()
            
            // Wrap the output back into the expected MLX array shape [B, L, H * D]
            let output = outputArray.reshaped(B, L, numHeads * headDim)
            return dense(output)
        }

        let attended = Self.attend(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        let output = attended
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, numHeads * headDim)
        return dense(output)
    }

    /// Elastic-Cache aware attention forward pass (WP-1a).
    ///
    /// If `recomputeActive` is true, recomputes keys/values for the active window and updates `activeCache`.
    /// If `recomputeActive` is false, reuses keys/values from `activeCache`.
    /// Always computes queries, calculates attention weights, and evaluates the drift similarity (σ_t^l).
    public func callAsFunction(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray,
        cache: LayerKVCache, activeCache: LayerActiveCache,
        prefixLen: Int, recomputeActive: Bool
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var activeKeys: MLXArray
        var activeValues: MLXArray
        var queries: MLXArray

        if recomputeActive {
            let qkv = queryKeyValue(x).reshaped(B, L, numHeads + 2 * numKVHeads, headDim)
            let parts = split(qkv, indices: [numHeads, numHeads + numKVHeads], axis: 2)
            queries = parts[0].transposed(0, 2, 1, 3)
            activeKeys = parts[1].transposed(0, 2, 1, 3)
            activeValues = parts[2].transposed(0, 2, 1, 3)

            if let queryLayernorm { queries = queryLayernorm(queries) }
            if let keyLayernorm { activeKeys = keyLayernorm(activeKeys) }

            queries = PartialRotaryEmbedding.apply(queries, cos: cos, sin: sin)
            activeKeys = PartialRotaryEmbedding.apply(activeKeys, cos: cos, sin: sin)

            activeCache.keys = activeKeys
            activeCache.values = activeValues
        } else {
            let qkv = queryKeyValue(x).reshaped(B, L, numHeads + 2 * numKVHeads, headDim)
            let parts = split(qkv, indices: [numHeads, numHeads + numKVHeads], axis: 2)
            queries = parts[0].transposed(0, 2, 1, 3)

            if let queryLayernorm { queries = queryLayernorm(queries) }
            queries = PartialRotaryEmbedding.apply(queries, cos: cos, sin: sin)

            guard let ak = activeCache.keys, let av = activeCache.values else {
                fatalError("activeCache keys/values are nil but recomputeActive is false")
            }
            activeKeys = ak
            activeValues = av
        }

        // Attend against committed prefix ++ active. No mask: all committed keys are allowed.
        let keys = cache.keys.map { concatenated([$0, activeKeys], axis: 2) } ?? activeKeys
        let values = cache.values.map { concatenated([$0, activeValues], axis: 2) } ?? activeValues

        // Update pending keys/values in the prefix cache (for block commit later)
        cache.pendingKeys = activeKeys
        cache.pendingValues = activeValues

        // Compute attention weights for drift test (GQA repeated keys if necessary)
        if prefixLen > 0 {
            var kForWeights = keys
            if numHeads != numKVHeads {
                let nRep = numHeads / numKVHeads
                kForWeights = Self.repeatKV(keys, nRep: nRep)
            }
            let scores = matmul(queries * scale, kForWeights.swappedAxes(-1, -2))
            let weights = softmax(scores.asType(.float32), axis: -1).asType(queries.dtype)
            
            // Average weights over heads [1, numHeads, L_queries, L_total] -> [1, L_queries, L_total]
            let meanWeights = weights.mean(axis: 1)
            let sumQueryWeights = meanWeights.sum(axis: 1) // [1, L_total]
            
            // Find argmax over prefix indices
            let prefixScore = sumQueryWeights[0..., 0..<prefixLen]
            let mostAttendedIndex = Int(prefixScore.argMax().item(Int32.self))
            
            // Extract vector: [numHeads, L_queries]
            let vector = weights[0, 0..., 0..., mostAttendedIndex]
            
            // Calculate similarity if previous vector exists
            if let prevVector = activeCache.previousAttentionVector {
                let dot = (vector * prevVector).sum()
                let normA = sqrt((vector * vector).sum())
                let normB = sqrt((prevVector * prevVector).sum())
                let sim = dot / (normA * normB + MLXArray(Float(1e-8)))
                activeCache.lastDriftSimilarity = sim
            } else {
                activeCache.lastDriftSimilarity = MLXArray(Float(1.0))
            }
            
            activeCache.previousAttentionVector = vector
            activeCache.previousMostAttendedIndex = mostAttendedIndex
        } else {
            activeCache.lastDriftSimilarity = MLXArray(Float(1.0))
            activeCache.previousAttentionVector = nil
            activeCache.previousMostAttendedIndex = nil
        }

        let attended = Self.attend(
            queries: queries, keys: keys, values: values, scale: scale, mask: nil)
        let output = attended
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, numHeads * headDim)
        return dense(output)
    }

    /// Scaled dot-product attention via MLX's fused kernel. FP32 softmax is internal (§2.7)
    /// and GQA is native (K/V are *not* pre-tiled). Verified against ``attendReference`` to
    /// bitwise agreement on the block-diffusion 0/-inf mask (see `testSDPAvsManual`).
    static func attend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, mask: MLXArray?
    ) -> MLXArray {
        MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: scale, mask: mask?.asType(queries.dtype))
    }

    /// Explicit reference attention (manual GQA expansion + FP32 softmax) kept to validate the
    /// fused ``attend`` path. Not used in the forward pass.
    static func attendReference(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, mask: MLXArray?
    ) -> MLXArray {
        let nQ = queries.dim(1)
        let nKV = keys.dim(1)
        var k = keys
        var v = values
        if nQ != nKV {
            let nRep = nQ / nKV
            k = repeatKV(keys, nRep: nRep)
            v = repeatKV(values, nRep: nRep)
        }
        var scores = matmul(queries * scale, k.swappedAxes(-1, -2))
        if let mask { scores = scores + mask.asType(scores.dtype) }
        let weights = softmax(scores.asType(.float32), axis: -1).asType(queries.dtype)
        return matmul(weights, v)
    }

    /// `[B, nKV, L, D]` -> `[B, nKV * nRep, L, D]` with each KV head repeated `nRep` times
    /// consecutively (matches the reference `repeat_kv`).
    static func repeatKV(_ x: MLXArray, nRep: Int) -> MLXArray {
        let (b, nKV, l, d) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let expanded = broadcast(x.expandedDimensions(axis: 2), to: [b, nKV, nRep, l, d])
        return expanded.reshaped(b, nKV * nRep, l, d)
    }
}
