import Foundation
import MLX
import Metal

/// Per-layer key/value store for the cached attention path (phase-2 §5 M5, deviation 1).
///
/// Holds one layer's **committed-block** K/V — the keys already have qk-norm + partial RoPE
/// applied at their own absolute positions, exactly as an uncached forward would produce them,
/// so a cached forward's active-block queries attend to `[committed ++ active]` with *no mask*
/// (every committed key is in an allowed, ≤-current block). This is what makes
/// `ExactPrefixCache` mathematically exact under the `.strict` mask.
///
/// `pending*` holds the most recent forward's active-block K/V; `commitPending()` appends it to
/// the committed store. The commit is driven by the loop's commit-cleanliness rule (a dedicated
/// capture forward over the final committed tokens), never by a raw denoising step — see
/// `DiffusionEngine`.
public final class LayerKVCache {
    /// Committed keys `[1, nKV, committedLen, D]` (post qk-norm + RoPE), or `nil` when empty.
    public private(set) var keys: MLXArray?
    /// Committed values `[1, nKV, committedLen, D]`, or `nil` when empty.
    public private(set) var values: MLXArray?
    /// Last forward's active-block keys `[1, nKV, B, D]` (candidate to commit).
    public var pendingKeys: MLXArray?
    /// Last forward's active-block values `[1, nKV, B, D]`.
    public var pendingValues: MLXArray?

    // FlashBlock runner and auxiliary buffers (WP-3b)
    public var flashBlockRunner: FlashBlockRunner?
    public var blockTables: MLXArray?
    public var ctxLens: MLXArray?
    
    // Persistent FlashBlock cache tensors
    public var attnOutPast: MLXArray?
    public var logsumexp: MLXArray?

    public init() {}

    /// Number of committed key/value positions.
    public var committedLength: Int { keys?.dim(2) ?? 0 }

    /// Append `pendingKeys`/`pendingValues` to the committed store (commit-cleanliness point).
    public func commitPending() {
        guard let pk = pendingKeys, let pv = pendingValues else {
            preconditionFailure("commitPending called with no pending K/V — a capture forward "
                + "must run over the final committed tokens before commit")
        }
        keys = keys.map { concatenated([$0, pk], axis: 2) } ?? pk
        values = values.map { concatenated([$0, pv], axis: 2) } ?? pv
        pendingKeys = nil
        pendingValues = nil
    }
}
