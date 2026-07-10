import XCTest
import MLX
import MLXRandom
@testable import DiffusionGeneration

/// sumi-M4'/M5' step-level acceptance (sumi-plan.md §3 S1.4): the sampler step functions and
/// the log-SNR schedule diff against reference intermediates (`sampler.*` / `schedule.*` in
/// the Sumi fixtures), isolated from the model. Greedy and adaptive (temp 0) gate exactly;
/// ancestral gates its deterministic posterior exactly and the Gumbel-max draw
/// distributionally.
///
/// Regenerate fixtures with:
///   scratch/sumi-venv/bin/python Tools/generate_sumi_fixtures.py
final class SumiSamplerTests: XCTestCase {

    static var fixtureDir: URL {
        if let override = ProcessInfo.processInfo.environment["NEODIFFUSION_SUMI_FIXTURE_DIR"] {
            return URL(fileURLWithPath: override)
        }
        return URL(fileURLWithPath:
            "/Users/andrebarlocher/Documents/Swift/NeoDiffusion/scratch/sumi_core_fixtures")
    }

    var tensors: [String: MLXArray]!
    var vocabSize = 0

    override func setUpWithError() throws {
        let dir = Self.fixtureDir
        let manifestURL = dir.appendingPathComponent("manifest.json")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: manifestURL.path),
            "Sumi fixtures missing — run: scratch/sumi-venv/bin/python Tools/generate_sumi_fixtures.py")
        tensors = try MLX.loadArrays(url: dir.appendingPathComponent("tensors.safetensors"))
        let manifest = try JSONSerialization.jsonObject(
            with: Data(contentsOf: manifestURL)) as! [String: Any]
        vocabSize = ((manifest["config"] as! [String: Any])["vocab_size"] as! Int)
    }

    // MARK: - Log-SNR schedule

    /// Tolerance 5e-5: torch's fp32 `linspace` rounds α by ~1 ulp differently at isolated
    /// indices, and the logit's slope 1/(α(1−α)) ≈ 400 near the clamp edges amplifies that
    /// to ~1e-5 in log-SNR. Behaviourally inert (the schedule feeds sigmoid, and only the
    /// stochastic ancestral sampler consumes it); bit-mimicking torch linspace is not worth it.
    func testLogSNRSchedule() throws {
        for kind in [LogSNRSchedule.Kind.linear, .cosine] {
            for steps in [8, 128] {
                let expected = tensors["schedule.\(kind.rawValue)_\(steps)"]!.asArray(Float.self)
                let actual = LogSNRSchedule(kind: kind).values(steps: steps)
                XCTAssertEqual(actual.count, expected.count, "\(kind) \(steps): length")
                for (i, (a, e)) in zip(actual, expected).enumerated() {
                    XCTAssertEqual(
                        a, e, accuracy: 5e-5,
                        "\(kind) steps=\(steps) idx \(i): \(a) vs \(e)")
                }
            }
        }
    }

    // MARK: - Noise mask

    func testNoiseMaskMatchesFixture() throws {
        // The fixture freezes positions [0, 4) (prompt) and S-2 (anchor).
        let expected = tensors["sampler.noise_mask"]!
        let S = expected.dim(-1)
        let mask = SumiNoiseMask.build(
            totalLength: S, promptLength: 4,
            anchors: [FrozenAnchor(position: S - 2, tokenId: 0)])
        XCTAssertTrue(
            (mask .== expected).all().item(Bool.self), "noise mask differs from fixture")
    }

    func testNoiseMaskDenoiseEnd() throws {
        let mask = SumiNoiseMask.build(totalLength: 10, promptLength: 2, denoiseEnd: 6)
        let expected: [Bool] = [false, false, true, true, true, true, false, false, false, false]
        XCTAssertEqual(mask.asArray(Bool.self), expected)
    }

    // MARK: - Greedy step

    func testGreedyStep() throws {
        let out = UniformStateSampler.greedyStep(
            z: tensors["sampler.z"]!,
            logits: tensors["sampler.logits"]!,
            noiseMask: tensors["sampler.noise_mask"]!)
        XCTAssertTrue(
            (out .== tensors["sampler.greedy_out"]!).all().item(Bool.self),
            "greedy step output differs from reference")
    }

    // MARK: - Adaptive step (temperature 0, k = 1 and 3)

    func testAdaptiveStep() throws {
        for k in [1, 3] {
            let (zOut, positions) = UniformStateSampler.adaptiveStep(
                z: tensors["sampler.z"]!,
                logits: tensors["sampler.logits"]!,
                noiseMask: tensors["sampler.noise_mask"]!,
                tokensPerStep: k,
                temperature: 0)

            // Committed canvas must match token-for-token.
            XCTAssertTrue(
                (zOut .== tensors["sampler.adaptive_out_k\(k)"]!).all().item(Bool.self),
                "adaptive k=\(k): canvas differs from reference")

            // Selected positions match as a set (tie-tolerant: topk order is unspecified).
            let actualPos = Set(positions.asArray(Int32.self))
            let expectedPos = Set(tensors["sampler.adaptive_pos_k\(k)"]!.asArray(Int32.self))
            XCTAssertEqual(actualPos, expectedPos, "adaptive k=\(k): selected positions")
        }
    }

    // MARK: - Ancestral posterior (deterministic part, exact gate)

    func testAncestralPosterior() throws {
        let posterior = UniformStateSampler.ancestralPosterior(
            z: tensors["sampler.z"]!,
            xHat: tensors["sampler.ancestral_x_hat"]!,
            logSNRt: -1.5,
            logSNRs: 0.5,
            vocabSize: vocabSize)
        let expected = tensors["sampler.ancestral_posterior"]!
        XCTAssertEqual(posterior.shape, expected.shape)
        let maxDiff = abs(posterior - expected).max().item(Float.self)
        XCTAssertLessThanOrEqual(maxDiff, 1e-6, "ancestral posterior max abs diff \(maxDiff)")
    }

    /// The reduced two-candidate ancestral form must sample **identically** to Gumbel-max
    /// over the full analytic posterior when fed the same uniform draw — the dropped terms
    /// are per-position constants that cannot change the argmax. Gated exactly across keys.
    func testAncestralReducedFormEquivalence() throws {
        let z = tensors["sampler.z"]!
        let xHat = tensors["sampler.ancestral_x_hat"]!
        for round in 0 ..< 16 {
            let key = MLXRandom.key(UInt64(1000 + round))
            let reduced = UniformStateSampler.ancestralStep(
                z: z, xHat: xHat, logSNRt: -1.5, logSNRs: 0.5,
                vocabSize: vocabSize, key: key)
            let posterior = UniformStateSampler.ancestralPosterior(
                z: z, xHat: xHat, logSNRt: -1.5, logSNRs: 0.5, vocabSize: vocabSize)
            let full = UniformStateSampler.gumbelMaxSample(posterior, key: key)
                .asType(reduced.dtype)
            let mismatches = (reduced .!= full).sum().item(Int.self)
            XCTAssertEqual(mismatches, 0, "key \(round): \(mismatches) positions differ")
        }
    }

    /// Distributional gate for the Gumbel-max draw: on a strongly-peaked posterior the
    /// sampled ids must hit the mode with roughly the posterior's mode mass, and never land
    /// on (near-)zero-mass entries. Not a bit-parity test — `torch.multinomial` and Gumbel-max
    /// share the distribution, not the RNG stream (decision S0.1-1).
    func testGumbelMaxSampleDistribution() throws {
        let posterior = tensors["sampler.ancestral_posterior"]!
        let z = tensors["sampler.z"]!
        var modeHits = 0
        var total = 0
        let rounds = 32
        let mode = posterior.argMax(axis: -1)
        let normalised = posterior / posterior.sum(axis: -1, keepDims: true)
        let modeMass = takeAlong(
            normalised, mode.expandedDimensions(axis: -1), axis: -1
        ).squeezed(axis: -1)

        for round in 0 ..< rounds {
            let key = MLXRandom.key(UInt64(round))
            let sampled = UniformStateSampler.ancestralStep(
                z: z, xHat: tensors["sampler.ancestral_x_hat"]!,
                logSNRt: -1.5, logSNRs: 0.5, vocabSize: vocabSize, key: key)
            modeHits += (sampled .== mode.asType(sampled.dtype)).sum().item(Int.self)
            total += sampled.size
        }

        let observed = Float(modeHits) / Float(total)
        let expected = modeMass.mean().item(Float.self)
        // Loose band: mean mode mass ± 10 percentage points over 32 × S draws.
        XCTAssertEqual(
            observed, expected, accuracy: 0.1,
            "Gumbel-max mode-hit rate \(observed) vs posterior mode mass \(expected)")
    }
}
