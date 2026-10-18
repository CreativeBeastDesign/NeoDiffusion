import Foundation
import MLX
import DiffusionCore
import DiffusionModel

/// Cross-call (multi-turn) reuse of committed prefix blocks (exp-5B, engine track B-2).
///
/// Holds the committed K/V of the **last** `generateCached(…, session:)` call together with the
/// token ids they were computed for. The next call reuses the longest leading run of whole
/// `blockLength` blocks whose ids equal the held ids (capped at the new prompt's full blocks) and
/// prefills only the rest — in multi-turn chat that is the previous prompt + answer.
///
/// **Exactness** `[Inferred: code reading, checked by SessionPrefixCacheTests in FP32]`: a block's
/// committed K/V is a deterministic function of (model, block length, prefix storage form, the ids
/// of that block and every block before it). Prefill computes it with `captureAndCommit` — one
/// B-wide plain cached forward at absolute positions against the already-committed prefix. Every
/// commit the session accepts was produced by that same computation: generated blocks commit via
/// the same capture forward, or via the clean-commit skip, which `cleanCommitSkipEligible` allows
/// only when the settle step was that same plain fresh forward over exactly the committed ids. So
/// a held block equals what this call's prefill would compute — by induction over blocks, since
/// block b's K/V depend on blocks < b only through the same held storage. The one feature that
/// commits anything else is sub-block commit (Option C: n < B-token capture forwards, different
/// within-block attention); calls using it bypass the session (see ``bypassReason(_:)``).
///
/// **Compatibility key** — what changes the K/V function or its storage, so it forces a fresh start:
/// model instance, `blockLength`, `kvQuantization`, `kvPoolEnabled`, and mask semantics (the
/// cached path is `.strict`-only today; the field is in the key so adding a semantics parameter to
/// the cached path cannot silently reuse across semantics). Deliberately **not** in the key —
/// they change *which* tokens get generated, never the K/V of given tokens: thresholds, nBuf
/// (MultiBD commits still go through the B-wide capture forward), speculation policy and
/// `speculationK`, dynamic τ, JOT / FlashBlock / CSA2 / Elastic / router reuse / ICE (all only
/// shape the denoising forwards; the commit stays a plain capture or an allowlisted skip).
///
/// **Memory**: the session keeps the last call's `ExactPrefixCache` storage alive between calls —
/// the same bytes that call already allocated, `2 · layers · nKV · headDim · bytesPerElement` per
/// token for FP16 (40,960 B/tok on LLaDA2.1-mini, `[Sourced: KVCacheQuantization doc]`), less
/// under KV quantization (e.g. 16,640 B/tok k8v4, 12,160 B/tok kivi4). With the pool on, the
/// allocation is the pool capacity (geometric growth), not the committed length. The natural
/// bound is one call's committed length (≤ prompt + genLength, rounded up to B), further capped by
/// ``maxCachedTokens`` (default 8,192; keeps the leading whole blocks; `nil` = uncapped).
///
/// **Threading**: one call at a time. State hand-over is lock-protected, and a call *takes* the
/// storage out of the session for its duration, so a concurrent second call sees an empty session
/// (correct, just no reuse) instead of racing on shared buffers.
public final class SessionPrefixCache: @unchecked Sendable {
    /// Configuration a held store was built under. Any difference ⇒ the store is dropped.
    struct CompatibilityKey: Equatable {
        let model: ObjectIdentifier
        let layerCount: Int
        let blockLength: Int
        let kvQuantization: KVCacheQuantization
        let kvPoolEnabled: Bool
        let maskSemantics: BlockDiffusionMask.Semantics
    }

    /// Optional hard cap on held tokens (rounded down to whole blocks after each call). `nil` = the
    /// last call's full committed length.
    public let maxCachedTokens: Int?

    private let lock = NSLock()
    private var store: ExactPrefixCache?
    private var tokens: [Int] = []
    private var key: CompatibilityKey?
    /// Weak so a session never keeps a ~9.5 GB model alive; a deallocated model ⇒ incompatible.
    private weak var model: LLaDA2MoeModel?

    /// Number of calls that reused ≥ 1 block / started fresh because of an incompatible key /
    /// bypassed the session entirely — diagnostics for logs and tests.
    public private(set) var reuseCount = 0
    public private(set) var incompatibleResetCount = 0
    public private(set) var bypassCount = 0

    /// Default cap 8,192 tokens (≈ 320 MiB FP16 on LLaDA2.1-mini; from impl-y). On overflow the
    /// leading whole blocks are kept (impl-x behaviour), not the whole session dropped.
    public init(maxCachedTokens: Int? = 8192) {
        precondition(maxCachedTokens.map { $0 >= 0 } ?? true, "maxCachedTokens must be >= 0")
        self.maxCachedTokens = maxCachedTokens
    }

    /// Tokens whose K/V are currently held.
    public var cachedTokenCount: Int {
        lock.lock(); defer { lock.unlock() }
        return tokens.count
    }

    /// The ids the held K/V belong to (diagnostic).
    public var cachedTokens: [Int] {
        lock.lock(); defer { lock.unlock() }
        return tokens
    }

    /// Drop everything.
    public func reset() {
        lock.lock(); defer { lock.unlock() }
        store = nil
        tokens = []
        key = nil
        model = nil
    }

    /// Why a call with these params cannot use (read or write) a session, or `nil` if it can.
    public static func bypassReason(_ params: GenerationParams, iceTemplate: [Int]? = nil) -> String? {
        if iceTemplate != nil { return "iceTemplate (ICE canvas; non-served path)" }
        guard !params.sessionReuseEligible else { return nil }
        if params.subBlockCommit {
            return "subBlockCommit (Option C commits n<B-token segments whose K/V differ from a "
                + "B-wide prefill capture)"
        }
        var r: [String] = []
        if params.nBuf != 1 { r.append("nBuf=\(params.nBuf)") }
        if params.flashBlockEnabled { r.append("flashBlock") }
        if params.dualCachingEnabled { r.append("dualCaching") }
        if params.elasticCacheEnabled { r.append("elastic") }
        if params.moduleAblation != .none { r.append("moduleAblation") }
        if params.iceEnabled { r.append("ice") }
        return "not sessionReuseEligible: " + r.joined(separator: ", ")
    }

    // MARK: - Engine hand-over (internal)

    /// Outcome of ``checkout``: storage to seed the call's cache with (already truncated to
    /// `reusedTokens`), or nothing.
    struct Checkout {
        let reusedTokens: Int
        let store: ExactPrefixCache?
        let key: CompatibilityKey
    }

    /// Takes the reusable part of the held store out of the session (the session is left empty
    /// until ``checkin``). Pure host-side work: an `[Int]` prefix compare plus lazy slices —
    /// no GPU readback.
    func checkout(prompt: [Int], key newKey: CompatibilityKey, model newModel: LLaDA2MoeModel)
        -> Checkout
    {
        lock.lock(); defer { lock.unlock() }
        let B = newKey.blockLength
        guard let held = store, key == newKey, model === newModel else {
            if store != nil { incompatibleResetCount += 1 }
            store = nil; tokens = []; key = nil; model = nil
            return Checkout(reusedTokens: 0, store: nil, key: newKey)
        }
        // Longest leading run of whole blocks equal to the held ids, never past the prompt's full
        // blocks (the prompt tail shares its block with generation and is not prefilled).
        let maxBlocks = min(prompt.count / B, tokens.count / B, held.committedLength / B)
        var reuseBlocks = 0
        while reuseBlocks < maxBlocks {
            let lo = reuseBlocks * B, hi = lo + B
            if prompt[lo ..< hi].elementsEqual(tokens[lo ..< hi]) { reuseBlocks += 1 } else { break }
        }
        let reused = reuseBlocks * B
        store = nil; tokens = []; key = nil; model = nil
        guard reused > 0 else {
            return Checkout(reusedTokens: 0, store: nil, key: newKey)
        }
        for layer in held.layers { layer.truncateCommitted(to: reused) }
        reuseCount += 1
        return Checkout(reusedTokens: reused, store: held, key: newKey)
    }

    /// Hands a finished call's cache to the session. `committedTokens` must be exactly the ids
    /// whose K/V `cache` holds (`Output.finalSequence`, all of which was committed).
    func checkin(cache: ExactPrefixCache, committedTokens: [Int], key newKey: CompatibilityKey,
                 model newModel: LLaDA2MoeModel)
    {
        lock.lock(); defer { lock.unlock() }
        // Defensive: a store whose length disagrees with its ids is never kept.
        guard cache.committedLength == committedTokens.count,
              cache.layers.allSatisfy({ $0.committedLength == committedTokens.count })
        else {
            store = nil; tokens = []; key = nil; model = nil
            return
        }
        var keep = committedTokens.count
        if let cap = maxCachedTokens {
            keep = min(keep, (cap / newKey.blockLength) * newKey.blockLength)
        }
        guard keep > 0 else {
            store = nil; tokens = []; key = nil; model = nil
            return
        }
        // Drop per-call transient state so the session holds only committed K/V (the next call
        // seeds a *fresh* per-call cache from this storage; FlashBlock/pending state never crosses).
        for layer in cache.layers {
            layer.pendingKeys = nil
            layer.pendingValues = nil
            layer.attnOutPast = nil
            layer.logsumexp = nil
            layer.flashBlockRunner = nil
            layer.blockTables = nil
            layer.ctxLens = nil
        }
        if keep < committedTokens.count {
            for layer in cache.layers { layer.truncateCommitted(to: keep) }
        }
        store = cache
        tokens = Array(committedTokens.prefix(keep))
        key = newKey
        model = newModel
    }

    func noteBypass() {
        lock.lock(); defer { lock.unlock() }
        bypassCount += 1
    }
}
