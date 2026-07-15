import XCTest
import MLX
import DiffusionModel
@testable import DiffusionGeneration

/// LM-head precision eval (requested by André 2026-07-09): quantizing the F16 output head
/// to 4-bit saves ~620 MB and some step time, but every sampler's confidences flow through
/// these logits — the M8-sweep caveat inherited from the LLaDA plan says head precision is
/// quality-sensitive. This measures the damage directly on real weights: same canvas
/// through both heads → top-1 flip rate, max softmax-prob deviation, and the confidence
/// margins the adaptive top-k ranking uses.
///
/// Diagnostic (prints measurements + a loose flip-rate bound); the speed side is the
/// bench's `--quantize-lm-head` arm. Opt-in:
///   NEODIFFUSION_SUMI_REAL=1 swift test --filter SumiLMHeadPrecisionTests
final class SumiLMHeadPrecisionTests: XCTestCase {

    static let modelDir = URL(fileURLWithPath:
        "./models/sumi-7b-4bit")
    static let tokenizerDir = URL(fileURLWithPath:
        "./Tools/reference/sumi")

    func testHeadQuantizationImpact() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["NEODIFFUSION_SUMI_REAL"] == "1",
            "opt-in: set NEODIFFUSION_SUMI_REAL=1 (loads 5.5 GB)")
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: Self.modelDir.appendingPathComponent("model.safetensors").path))

        let container = try SumiDiffusionModel.load(from: Self.modelDir)
        let tokenizer = try await SumiTokenizer.from(modelFolder: Self.tokenizerDir)

        // A realistic mid-generation canvas: prompt + partly-clean text + random tail.
        let text = "Question: What is the capital of Japan?\nAnswer: The capital of Japan is"
        var ids = tokenizer.encode(text: text).map(Int32.init)
        var state: UInt64 = 0x9E3779B97F4A7C15
        while ids.count < 1024 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            ids.append(Int32(state % UInt64(container.config.vocabSize)))
        }
        let input = MLXArray(ids).reshaped(1, ids.count)

        let logitsF16Head = container.model.logits(forTokens: input)
        eval(logitsF16Head)

        container.quantizeLMHead()
        let logitsQ4Head = container.model.logits(forTokens: input)
        eval(logitsQ4Head)

        // Top-1 flips overall AND among confident positions. The overall rate on this
        // canvas is dominated by the random tail, where distributions are near-flat and
        // argmax flips between ~equal candidates are harmless (first run measured 93%
        // overall flips yet clean end-to-end generations — the metric, not the head, was
        // broken). What the samplers actually rely on is stability where the F16 head is
        // confident.
        let top1A = logitsF16Head.argMax(axis: -1)
        let top1B = logitsQ4Head.argMax(axis: -1)
        let flipped = (top1A .!= top1B)
        let flips = flipped.sum().item(Int.self)
        let total = top1A.size

        let probsA = softmax(logitsF16Head, axis: -1)
        let probsB = softmax(logitsQ4Head, axis: -1)
        let maxProbDiff = abs(probsA - probsB).max().item(Float.self)
        let pMaxA = probsA.max(axis: -1)
        let pMaxB = probsB.max(axis: -1)
        let maxConfShift = abs(pMaxA - pMaxB).max().item(Float.self)

        var confidentSummary: [String] = []
        var worstConfidentFlipRate: Float = 0
        for threshold: Float in [0.3, 0.5, 0.7] {
            let confident = pMaxA .>= threshold
            let n = confident.sum().item(Int.self)
            let flipsAt = (flipped .&& confident).sum().item(Int.self)
            let rate = n > 0 ? Float(flipsAt) / Float(n) : 0
            confidentSummary.append("p_max≥\(threshold): \(flipsAt)/\(n) (\(rate))")
            if threshold >= 0.5 { worstConfidentFlipRate = max(worstConfidentFlipRate, rate) }
        }

        print("""
            [sumi-LMHEAD] 4-bit head vs F16 head (canvas 1024):
              top-1 flips overall: \(flips)/\(total) (\(Float(flips) / Float(total))) — tail-dominated, diagnostic only
              confident-position flips: \(confidentSummary.joined(separator: " | "))
              max |Δprob|: \(maxProbDiff), max |Δp_max| (confidence shift): \(maxConfShift)
            """)

        // The gate that matters: where the F16 head is confident (p_max ≥ 0.5), the 4-bit
        // head must agree on ≥95% of decisions.
        XCTAssertLessThanOrEqual(
            worstConfidentFlipRate, 0.05,
            "4-bit lm_head flips confident decisions — reject without a task-level run")
    }

    /// Margin-conditioned analysis (André's second metric catch, 2026-07-09): the
    /// p_max-thresholded gate above pools razor-margin positions (p1/p2 ≈ 0.40/0.35 —
    /// flips expected and benign under any perturbation) with wide-margin ones
    /// (0.40/0.02 — a flip means damage), and misses ambiguity below its thresholds
    /// entirely. This test measures, on **real mid-generation canvases** (captured from an
    /// actual F16-head adaptive trajectory, the sampler's true input distribution):
    ///
    ///   1. flip rate binned by the F16 top-1/top-2 prob margin — healthy quantization
    ///      concentrates flips in the near-zero-margin bins and produces NONE at wide
    ///      margins; the "largest margin that flipped" is the headline number;
    ///   2. the margin-perturbation scale itself (|gap_f16 − gap_q4| on the F16 pair) —
    ///      flips at margins ≫ this scale would be anomalous;
    ///   3. the downstream quantity directly: the adaptive k=4 selection + committed
    ///      tokens under both heads on identical inputs.
    ///
    /// Gates: zero flips at margin ≥ 0.10, and zero committed-token disagreements at
    /// jointly-selected positions.
    func testHeadQuantizationMarginAnalysis() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["NEODIFFUSION_SUMI_REAL"] == "1",
            "opt-in: set NEODIFFUSION_SUMI_REAL=1 (loads 5.5 GB, ~5 min)")
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: Self.modelDir.appendingPathComponent("model.safetensors").path))

        let container = try SumiDiffusionModel.load(from: Self.modelDir)
        let tokenizer = try await SumiTokenizer.from(modelFolder: Self.tokenizerDir)
        let engine = SumiEngine(model: container.model)
        let config = container.config

        // 1) Capture real mid-generation canvases from an F16-head adaptive trajectory.
        let prompt = "Question: What is the capital of Japan?\nAnswer:"
        let promptIds = tokenizer.encode(text: prompt).map(Int32.init)
        let P = promptIds.count
        let budget = 64
        let canvasLength = 1024
        let sampleSteps: Set<Int> = [0, 5, 10, 15]
        var canvases: [MLXArray] = []
        _ = engine.generate(
            SumiGenerationRequest(
                promptIds: promptIds, maxNewTokens: budget, canvasLength: canvasLength,
                numDenoisingSteps: 16, sampler: .adaptive, temperature: 0,
                tokensPerStep: 4, denoiseEnd: P + budget + 2, trimAtEOS: false, seed: 5),
            onStep: { step, z in
                if sampleSteps.contains(step) { canvases.append(z) }
            })
        let noiseMask = SumiNoiseMask.build(
            totalLength: canvasLength, promptLength: P,
            anchors: [
                FrozenAnchor(position: P + budget, tokenId: Int32(config.eosTokenId)),
                FrozenAnchor(position: P + budget + 1, tokenId: Int32(config.bosTokenId)),
            ],
            denoiseEnd: P + budget + 2)

        // 2) F16-head logits for every canvas FIRST (quantizeLMHead mutates in place).
        let logitsF16 = canvases.map { (z: MLXArray) -> MLXArray in
            let l = container.model.logits(forTokens: z)
            eval(l)
            return l
        }
        container.quantizeLMHead()
        let logitsQ4 = canvases.map { (z: MLXArray) -> MLXArray in
            let l = container.model.logits(forTokens: z)
            eval(l)
            return l
        }

        // 3) Pool per-position records: F16 margin, flip flag, margin perturbation.
        struct Record { let margin: Float; let flipped: Bool; let perturbation: Float }
        var records: [Record] = []
        var selectionMismatch = 0
        var commitDisagree = 0
        var selectionsCompared = 0

        for (lF, lQ) in zip(logitsF16, logitsQ4) {
            let probsF = softmax(lF, axis: -1)
            let probsQ = softmax(lQ, axis: -1)
            let top1F = lF.argMax(axis: -1)                                   // [1, S]
            let top1Q = lQ.argMax(axis: -1)
            let p1F = probsF.max(axis: -1)
            // Top-2 under F16: zero out the top-1 prob, take the max again.
            let zeroed = putAlong(
                probsF, top1F.expandedDimensions(axis: -1),
                values: MLXArray(Float(0)), axis: -1)
            let p2F = zeroed.max(axis: -1)
            let top2F = zeroed.argMax(axis: -1)
            // Same token pair's gap under the Q4 head.
            let p1FunderQ = takeAlong(probsQ, top1F.expandedDimensions(axis: -1), axis: -1)
                .squeezed(axis: -1)
            let p2FunderQ = takeAlong(probsQ, top2F.expandedDimensions(axis: -1), axis: -1)
                .squeezed(axis: -1)

            let margins = (p1F - p2F).reshaped(-1).asArray(Float.self)
            let perturb = abs((p1F - p2F) - (p1FunderQ - p2FunderQ))
                .reshaped(-1).asArray(Float.self)
            let flips = (top1F .!= top1Q).reshaped(-1).asArray(Bool.self)
            for i in 0 ..< margins.count {
                records.append(Record(
                    margin: margins[i], flipped: flips[i], perturbation: perturb[i]))
            }

            // 4) Downstream gate: adaptive k=4 selection under both heads.
            let z = canvases[selectionsCompared]
            let (zF, selF) = UniformStateSampler.adaptiveStep(
                z: z, logits: lF, noiseMask: noiseMask, tokensPerStep: 4, temperature: 0)
            let (zQ, selQ) = UniformStateSampler.adaptiveStep(
                z: z, logits: lQ, noiseMask: noiseMask, tokensPerStep: 4, temperature: 0)
            let setF = Set(selF.reshaped(-1).asArray(Int32.self))
            let setQ = Set(selQ.reshaped(-1).asArray(Int32.self))
            if setF != setQ { selectionMismatch += 1 }
            let zFa = zF.reshaped(-1).asArray(Int32.self)
            let zQa = zQ.reshaped(-1).asArray(Int32.self)
            for pos in setF.intersection(setQ) where zFa[Int(pos)] != zQa[Int(pos)] {
                commitDisagree += 1
            }
            selectionsCompared += 1
        }

        // Margin-binned flip table.
        let bins: [(Float, Float, String)] = [
            (0.00, 0.01, "[0.00,0.01)"), (0.01, 0.05, "[0.01,0.05)"),
            (0.05, 0.10, "[0.05,0.10)"), (0.10, 0.25, "[0.10,0.25)"),
            (0.25, 0.50, "[0.25,0.50)"), (0.50, 1.01, "[0.50,1.00]"),
        ]
        var tableLines: [String] = []
        for (lo, hi, label) in bins {
            let inBin = records.filter { $0.margin >= lo && $0.margin < hi }
            let flips = inBin.filter(\.flipped).count
            tableLines.append("\(label): \(flips)/\(inBin.count)")
        }
        let largestFlippedMargin = records.filter(\.flipped).map(\.margin).max() ?? 0
        let sortedPerturb = records.map(\.perturbation).sorted()
        let p99Perturb = sortedPerturb[Int(Double(sortedPerturb.count - 1) * 0.99)]
        let flipsWideMargin = records.filter { $0.flipped && $0.margin >= 0.10 }.count

        print("""
            [sumi-LMHEAD-MARGIN] \(records.count) positions over \(canvases.count) real mid-generation canvases:
              flips by F16 top1-top2 prob margin: \(tableLines.joined(separator: " | "))
              largest margin that flipped: \(largestFlippedMargin)
              margin perturbation: p99 \(p99Perturb), max \(sortedPerturb.last ?? 0)
              adaptive k=4 selection: set mismatches \(selectionMismatch)/\(selectionsCompared) canvases, \
            committed-token disagreements at joint positions: \(commitDisagree)
            """)

        XCTAssertEqual(flipsWideMargin, 0, "flips at margin ≥ 0.10 — genuine quantization damage")
        XCTAssertEqual(commitDisagree, 0, "4-bit head changes committed tokens at jointly-selected positions")
    }
}
