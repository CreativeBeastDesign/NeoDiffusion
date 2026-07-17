import Foundation
import MLX
import MLXNN
import DiffusionCore
import DiffusionModel

/// The block-diffusion denoising loop (phase-2 §1 / M5). Reproduces the reference
/// `generate()` algorithm **token-for-token at temperature 0** under NeoDiffusion's `.strict`
/// block mask (the corrected baseline — never stock `generate()`'s 0/1 soft-bias mask; §6,
/// deviation 8).
///
/// Shape and state:
/// - Built **Block-Buffer-shaped from day one** (``BlockBuffer``, `nBuf == 1` — phase-1 §5).
///   At `nBuf == 1` the slot machine reduces exactly to the reference SingleBD schedule.
/// - Both selection sets come from **one forward per step**: Γ (M2T — masked positions whose
///   confidence clears τ_mask, with a top-`numToTransfer` fallback guaranteeing progress) and
///   Δ (T2T — unmasked, non-prompt positions whose confidence clears τ_edit *and* whose
///   prediction changed). Committed blocks are immutable (phase-1 §5).
/// - Selection, thresholding, argmax and the state update are **MLX array ops** — no Swift-side
///   loops over logits (phase-1 §6). Loop control uses K-step speculative execution: up to `K`
///   steps are built into one graph and a single stacked "break" flag is read back per batch
///   (≤1 blocking readback per K steps + 1 per block commit — the M5(c) sync budget). Overshoot
///   past the break point is discarded via per-step snapshots, so the result stays exact.
///
/// The forward is injected (``Forward``) so the *same* loop drives both the cache-disabled path
/// (M5(a) parity anchor) and the ExactPrefixCache path (M5(b)); see ``generate`` vs the cached
/// entry point.
public final class DiffusionEngine {
    public let model: LLaDA2MoeModel
    /// Speculative batch size for loop control (phase-1 §6, K≈2–4). `K == 1` degenerates to a
    /// blocking readback every step; larger K amortizes the sync at the cost of bounded wasted
    /// forwards on overshoot. Output is identical for any K ≥ 1 (proven by test).
    public let speculationK: Int
    /// When `true`, per-phase wall-clock in ``Output/Metrics`` is made *real* by forcing
    /// `eval` at phase boundaries (prefill end, each commit's capture forward) — the M6 bench
    /// setting (handoff §3). Adds ≤1 extra synchronization per block commit and per prefill
    /// block, so leave `false` (the default) when only tokens or the sync audit matter.
    public let instrument: Bool

    public final class BoundaryHolder {
        public var value: Int = 0
        public init() {}
    }

    /// One decoded step's per-position trace (WP-2a calibration / JOT pre-experiment; emitted
    /// only when `onTrace` is set — an offline instrumentation path, never the serving path).
    /// Arrays are over the front block's `B` positions at that logical step.
    public struct StepTrace {
        public let blockIndex: Int
        public let stepInBlock: Int
        /// Top-1 confidence per position (x0_p).
        public let confidence: [Float]
        /// Positions unmasked this step (Γ, including speculation acceptances).
        public let transferred: [Bool]
        /// Positions edited this step (Δ).
        public let edited: [Bool]
        /// Argmax token id per position at this step.
        public let argmaxToken: [Int]
        /// Masked positions at the START of this step.
        public let masked: [Bool]
    }

    /// Offline trace hook (WP-2a): called once per applied logical step with the front slot's
    /// per-position trace. Adds a per-step readback of small arrays — set it only for
    /// calibration/analysis runs (<50 prompts), never in serving.
    public var onTrace: ((StepTrace) -> Void)?

    public init(model: LLaDA2MoeModel, speculationK: Int = 4, instrument: Bool = false) {
        precondition(speculationK >= 1, "speculationK must be >= 1")
        self.model = model
        self.speculationK = speculationK
        self.instrument = instrument
    }

    /// Result of one `generate` call.
    public struct Output {
        /// Trimmed generated ids (reference tail: up to and including the first `eosId`, else the
        /// full `genLength`), matching `generate()`'s return slice.
        public let tokens: [Int]
        /// The full padded sequence `x` `[total_length]` after every block committed — the
        /// token-for-token parity target against the reference trace's `final_x`.
        public let finalSequence: [Int]
        /// Per-committed-block full-prefix snapshots `x[:(b+1)*B]` (parity vs `block_commits`).
        public let blockCommits: [[Int]]
        /// Number of denoising steps spent per generated block (diagnostic; not a gate).
        /// Caveat (handoff §4): on a budget-break exit this counts the break iteration the
        /// reference does not (reads +1 vs the reference trace). Tokens are unaffected.
        public let stepsPerBlock: [Int]
        /// Count of blocking readbacks performed (M5(c) sync audit).
        public let syncPoints: Int
        /// M6 bench diagnostics (phase-2 §4 M6). Step/forward counts are always exact;
        /// the per-phase wall-clocks are only *real* when the engine was built with
        /// `instrument: true` (otherwise lazy evaluation blurs phase attribution).
        public let metrics: Metrics
        /// The sequence of tokens at each applied logical step (optional, collected if enabled).
        public let trajectorySequences: [[Int]]?

        public init(
            tokens: [Int],
            finalSequence: [Int],
            blockCommits: [[Int]],
            stepsPerBlock: [Int],
            syncPoints: Int,
            metrics: Metrics,
            trajectorySequences: [[Int]]? = nil
        ) {
            self.tokens = tokens
            self.finalSequence = finalSequence
            self.blockCommits = blockCommits
            self.stepsPerBlock = stepsPerBlock
            self.syncPoints = syncPoints
            self.metrics = metrics
            self.trajectorySequences = trajectorySequences
        }
    }

    /// Per-run diagnostics for the M6 metric set.
    public struct Metrics {
        /// Wall-clock spent prefilling pure-prompt blocks into the cache (cached path only).
        public let prefillSeconds: Double
        /// Wall-clock spent inside per-block denoising loops.
        public let denoiseSeconds: Double
        /// Wall-clock spent committing blocks (capture forward + block readback + streaming).
        public let commitSeconds: Double
        /// Per generated block: the reference's `post_steps` counter at loop exit — the number
        /// of mask-free (refinement) iterations, including the iteration that broke the loop.
        public let postStepsPerBlock: [Int]
        /// Forwards actually *evaluated*, honestly counting speculative overshoot (each K-batch
        /// evaluates K forwards even when the break lands early) and, on the cached path, the
        /// per-block commit-cleanliness capture forwards + prompt prefill forwards. TPF against
        /// this counter is the honest figure; against `stepsPerBlock.sum()` it is the logical
        /// (reference-comparable) figure (handoff §3 item 4).
        public let forwardsEvaluated: Int
        /// Per block, per logical step: |Γ| — masked positions transferred that step (M8 E1
        /// trajectory diagnostic; distinguishes "quantization changes final text" from
        /// "quantization changes denoising dynamics").
        public let transfersPerStep: [[Int]]
        /// Per block, per logical step: |Δ| — unmasked non-prompt positions edited that step.
        public let editsPerStep: [[Int]]
        /// Per block, per logical step: mean confidence (`x0_p`) over the positions written
        /// that step (Γ ∪ Δ); 0 where nothing was written (budget-break step).
        public let meanTransferConfidencePerStep: [[Float]]
        /// Index of the committed block whose generated region first contained `eosId`
        /// (the block `eos_early_stop` fires on), or nil if eos never appeared.
        public let eosBlockIndex: Int?

        // MARK: WP-1b MultiBD diagnostics + effective-parameter echoes
        /// Total logical steps taken by the loop (the TPF denominator). At `nBuf == 1` this
        /// equals `stepsPerBlock.sum()`; with two concurrently active blocks a single step
        /// advances both, so `stepsPerBlock.sum() - logicalStepsTotal == dualActiveSteps`.
        public let logicalStepsTotal: Int
        /// Logical steps during which two blocks were active simultaneously.
        public let dualActiveSteps: Int
        /// Global step index (0-based, counted in logical steps) at which each activation
        /// event fired (a trailing block entered the buffer). Empty at `nBuf == 1`.
        public let activationSteps: [Int]
        /// Per generated block: steps this block spent as the *trailing* slot with masks
        /// remaining and zero Γ acceptances — i.e. steps where the τ_semi gate (or the
        /// threshold) starved it. The τ_semi diagnostic.
        public let trailingStarvedStepsPerBlock: [Int]
        /// `denoiseSeconds` split by phase width — the WP-1b step-latency-multiplier inputs:
        /// single-active per-step latency = single/(logicalStepsTotal − dualActiveSteps),
        /// dual = dual/dualActiveSteps. Host-scoped (wall-clock); real per batch because every
        /// speculative batch ends in a blocking readback.
        public let singleActiveDenoiseSeconds: Double
        public let dualActiveDenoiseSeconds: Double
        // MARK: WP-2a speculation diagnostics
        /// Per block, per logical step: tokens written by the speculation policy that step
        /// (accepted draft prefix + the correction token; 0 on non-speculated steps or when
        /// `speculation == .none`). The accepted-tokens histogram source.
        public let acceptedPerStep: [[Int]]
        /// Total active-window tokens processed across all evaluated forwards (width-aware
        /// accounting: a 2B-wide verifier forward counts 2B, a B-wide target forward counts B;
        /// includes speculative overshoot and capture/prefill forwards at their widths).
        /// Width-corrected TPF = tokens / (tokensProcessedInForwards / blockLength).
        public let tokensProcessedInForwards: Int

        /// Effective parameters as the engine actually ran them (provenance rule: benches must
        /// record these echoes, never the CLI inputs — elastic-cache logbook F7).
        public let effectiveNBuf: Int
        public let effectiveTauAdd: Float
        public let effectiveTauSemi: Float
        public let effectiveSpeculationK: Int
        public let effectiveSpeculation: String
        public let effectiveTauSpan: Int
        // MARK: WP-2b effective echoes
        public let effectiveDynamicTauAlpha: Float
        public let effectiveEosEarlyExit: Bool
        // MARK: JOT effective echoes
        public let effectiveJotEnabled: Bool
        public let effectiveJotK: Int
        public let effectiveJotThreshold: Float
        /// WP-3a v2: whether the faithful (per-layer frozen-K/V hold) mechanism ran, vs the v1
        /// MoE-zeroing behaviour. Only meaningful when `effectiveJotEnabled`.
        public let effectiveJotFaithful: Bool
        /// WP-3a §10: the static MoE capacity ratio actually applied (0 = Option-A mask path).
        public let effectiveMoeCapacityRatio: Float
        // MARK: ICE effective echoes
        public let effectiveIceEnabled: Bool
        public let effectiveIceTau: Float
        public let effectiveIceNt: Int
        public let effectiveIceThinkingLength: Int
        // MARK: Credit Decoding effective echoes
        public let effectiveCreditDecodingEnabled: Bool
        public let effectiveCreditAlpha: Float
        public let effectiveCreditBeta: Float
        public let effectiveCreditGamma: Float

        // MARK: In-situ attribution effective echo
        /// What the engine **actually ran with**, not what was requested. Load-bearing: an
        /// ablation that silently fails to reach the forward would otherwise look like a clean
        /// "no effect" result. The default path (`DiffusionEngine+Entry`) has several overloads
        /// and only the ones the served config reaches carry the switch — this echo is what makes
        /// that verifiable from the JSONL instead of by inspection.
        public let effectiveModuleAblation: ModuleAblation
    }

    /// Mutable stats shared between the cached entry point's closures and `run` (the capture
    /// forwards fire inside `run`'s commit section, after `run` has begun).
    final class RunStats {
        var prefillSeconds: Double = 0
        var extraForwards: Int = 0
    }

    /// Signature of an injected forward: given the current window ids `[1, W]` and the count of
    /// active-block positions `B`, return **FP32 logits over the last `B` positions** `[1, B, V]`.
    /// The cache-disabled path builds a full `.strict` mask over the window; a cached path can
    /// forward only the active block against committed KV (no mask needed).
    public typealias Forward = (_ windowIds: MLXArray, _ activeLen: Int, _ frozen: MLXArray?) -> MLXArray

    /// Signature of the S2D2 verifier forward (WP-2a): given the `[1, 2B]` pair window
    /// `[draft copy | mask copy]`, return FP32 logits `[1, 2B, V]`. The closure owns the
    /// duplicated absolute position ids and the memoized ``BlockDiffusionMask/s2d2VerifierMask``;
    /// cached path only (conditions on committed prefix KV).
    public typealias VerifierForward = (_ pairWindowIds: MLXArray) -> MLXArray
}
