import Foundation
import DiffusionCore

/// The **approximate** KV tier: within-block K/V for the active block (phase-1 §5 "two caches,
/// two contracts"). Kept a distinct type from ``ExactPrefixCache`` on purpose.
///
/// **Phase 2 contract: recompute every step.** The active block's tokens change across denoising
/// steps, so its K/V are recomputed each forward and never reused — exact, zero staleness. This
/// type therefore holds no state yet; it exists so that Phase 3's staleness policies
/// (`[[elastic-cache-metal-kernel]]`) have *this tier and only this tier* to touch, without
/// disturbing the exact prefix. Making the seam explicit now is the phase-1 §6 requirement that
/// "DecodingPolicy and CacheManager as pure array-op functions" survive review.
public struct ActiveBlockCache {
    /// Phase 2 recomputes the active block every step (no reuse). Phase 3 flips this.
    public let recomputeEveryStep: Bool

    public init() {
        self.recomputeEveryStep = true
    }
}
