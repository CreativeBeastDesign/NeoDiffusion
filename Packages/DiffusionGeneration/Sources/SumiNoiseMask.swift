import Foundation
import MLX

/// A token pinned at a fixed canvas position for the whole denoising loop. Sumi's default
/// anchor is the `[EOS, BOS]` document delimiter at `promptLength + budget` — a training-
/// distribution artefact the README flags as required for coherent output.
public struct FrozenAnchor: Equatable, Sendable {
    public let position: Int
    public let tokenId: Int32

    public init(position: Int, tokenId: Int32) {
        self.position = position
        self.tokenId = tokenId
    }
}

/// The Sumi noise mask (`_build_noise_mask` in generation_sumi.py): `true` = denoise here,
/// `false` = frozen. Prompt positions are frozen; anchors freeze their positions (the engine
/// writes their token ids into the canvas); `denoiseEnd` freezes the tail at its
/// prior-random init so the step budget concentrates on the content window.
///
/// Deviation from the reference API (decision, sumi-plan.md §2 S0.1-2): anchors inside the
/// prompt are **rejected** (`precondition`) instead of silently skipped — an in-prompt anchor
/// is a caller bug, and allowing it would fragment the prompt slice for any future
/// prompt-KV caching.
public enum SumiNoiseMask {
    /// Builds the `[1, totalLength]` boolean denoise mask (single-request engine, B = 1).
    public static func build(
        totalLength: Int,
        promptLength: Int,
        anchors: [FrozenAnchor] = [],
        denoiseEnd: Int? = nil
    ) -> MLXArray {
        precondition(promptLength <= totalLength, "prompt exceeds canvas")
        var mask = [Bool](repeating: true, count: totalLength)
        for i in 0 ..< promptLength { mask[i] = false }

        for anchor in anchors {
            precondition(
                anchor.position >= promptLength && anchor.position < totalLength,
                "anchor at \(anchor.position) is inside the prompt or out of bounds — rejected"
            )
            mask[anchor.position] = false
        }

        if let end = denoiseEnd, end >= 0, end < totalLength {
            for i in end ..< totalLength { mask[i] = false }
        }

        return MLXArray(mask).reshaped(1, totalLength)
    }
}
