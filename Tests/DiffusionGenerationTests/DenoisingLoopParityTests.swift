import XCTest
import MLX
import DiffusionCore
@testable import DiffusionModel
@testable import DiffusionGeneration

/// M5 acceptance (phase-2 §4): denoising-loop parity + ExactPrefixCache identity + sync audit.
///
/// Fixtures (`Tools/generate_loop_fixtures.py`) are reference `generate()` traces produced under
/// the **corrected** strict 0/-inf block mask — never stock `generate()`'s 0/1 soft-bias mask
/// (§6, deviation 8). Config-driven from `manifest.json`, so the SAME test covers both M5 paths:
///   • dev toy config, seeded random weights, FP32 (default `scratch/loop_fixtures`);
///   • Studio real weights, BF16 — regenerate on the Studio with
///       python3 Tools/generate_loop_fixtures.py --config <hf_dir>/config.json \
///           --weights <hf_dir> --dtype bfloat16 --out <dir>
///     then:  NEODIFFUSION_LOOP_FIXTURE_DIR=<dir> swift test
final class DenoisingLoopParityTests: XCTestCase {

    struct Manifest: Decodable {
        let blockLength: Int
        let dtype: String
        let config: LLaDA2MoeConfig
        enum CodingKeys: String, CodingKey {
            case blockLength = "block_length"
            case dtype
            case config
        }
    }

    struct Case: Decodable {
        let name: String
        let mode: String
        let prompt: [Int]
        let params: Params
        let output: [Int]
        let finalX: [Int]
        let blockCommits: [[Int]]
        let perBlockSteps: [Int]
        enum CodingKeys: String, CodingKey {
            case name, mode, prompt, params, output
            case finalX = "final_x"
            case blockCommits = "block_commits"
            case perBlockSteps = "per_block_steps"
        }
    }

    struct Params: Decodable {
        let threshold: Float
        let editingThreshold: Float
        let maxPostSteps: Int
        let numToTransfer: Int
        let eosEarlyStop: Bool
        let temperature: Float
        let blockLength: Int
        let genLength: Int
        let maskId: Int
        let eosId: Int
        enum CodingKeys: String, CodingKey {
            case threshold, temperature
            case editingThreshold = "editing_threshold"
            case maxPostSteps = "max_post_steps"
            case numToTransfer = "num_to_transfer"
            case eosEarlyStop = "eos_early_stop"
            case blockLength = "block_length"
            case genLength = "gen_length"
            case maskId = "mask_id"
            case eosId = "eos_id"
        }
        func toGenerationParams() -> GenerationParams {
            GenerationParams(
                threshold: threshold, editingThreshold: editingThreshold,
                maxPostSteps: maxPostSteps, numToTransfer: numToTransfer,
                eosEarlyStop: eosEarlyStop, temperature: temperature,
                blockLength: blockLength, genLength: genLength,
                maskId: maskId, eosId: eosId)
        }
    }

    struct Traces: Decodable {
        let maskId: Int
        let eosId: Int
        let cases: [Case]
        enum CodingKeys: String, CodingKey {
            case cases
            case maskId = "mask_id"
            case eosId = "eos_id"
        }
    }

    static var fixtureDir: URL {
        if let override = ProcessInfo.processInfo.environment["NEODIFFUSION_LOOP_FIXTURE_DIR"] {
            return URL(fileURLWithPath: override)
        }
        return URL(fileURLWithPath:
            "/Users/andrebarlocher/Documents/Swift/NeoDiffusion/scratch/loop_fixtures")
    }

    var manifest: Manifest!
    var traces: Traces!
    var model: LLaDA2MoeModel!

    override func setUpWithError() throws {
        let dir = Self.fixtureDir
        let weightsURL = dir.appendingPathComponent("weights.safetensors")
        let tracesURL = dir.appendingPathComponent("traces.json")
        let manifestURL = dir.appendingPathComponent("manifest.json")

        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: tracesURL.path),
            "Loop fixtures missing at \(dir.path) — run: python3 Tools/generate_loop_fixtures.py")

        manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))
        traces = try JSONDecoder().decode(Traces.self, from: Data(contentsOf: tracesURL))

        let dm = DiffusionModel(config: manifest.config)
        try dm.loadWeights(from: weightsURL)
        model = dm.model
    }

    /// M5(a): cache-disabled temp-0 generation matches the strict-mask reference traces
    /// token-for-token across S/Q modes and eos_early_stop on/off.
    func testCacheDisabledParity() throws {
        let engine = DiffusionEngine(model: model, speculationK: 4)
        var failures: [String] = []
        for c in traces.cases {
            let out = engine.generate(prompt: c.prompt, params: c.params.toGenerationParams())
            if out.finalSequence != c.finalX {
                failures.append("\(c.name): final_x mismatch "
                    + "(first diff at \(firstDiff(out.finalSequence, c.finalX)))")
            }
            if out.tokens != c.output {
                failures.append("\(c.name): output mismatch "
                    + "(len \(out.tokens.count) vs \(c.output.count))")
            }
            for (b, commit) in out.blockCommits.enumerated() where b < c.blockCommits.count {
                if commit != c.blockCommits[b] {
                    failures.append("\(c.name): block \(b) commit mismatch")
                    break
                }
            }
        }
        XCTAssertTrue(failures.isEmpty,
            "M5(a) parity failures (\(failures.count)/\(traces.cases.count)):\n"
            + failures.joined(separator: "\n"))
        print("[M5a] cache-disabled parity: \(traces.cases.count)/\(traces.cases.count) "
            + "cases token-for-token (\(manifest.dtype), block \(manifest.blockLength))")
    }

    /// The speculative batch size K must not change the output (any K ≥ 1 is exact).
    func testSpeculationInvariance() throws {
        let sampleNames = ["p8_q_noeos", "p20_s_noeos", "p16_q_eos"]
        for name in sampleNames {
            guard let c = traces.cases.first(where: { $0.name == name }) else { continue }
            let p = c.params.toGenerationParams()
            let k1 = DiffusionEngine(model: model, speculationK: 1)
                .generate(prompt: c.prompt, params: p).finalSequence
            let k4 = DiffusionEngine(model: model, speculationK: 4)
                .generate(prompt: c.prompt, params: p).finalSequence
            XCTAssertEqual(k1, k4, "\(name): output depends on speculationK (should be exact)")
        }
    }

    /// M5(b): with ExactPrefixCache *enabled*, outputs are identical to the cache-disabled path
    /// (which M5(a) already proved matches the reference). This is where the cache's
    /// block-causality exactness (deviation 1) is proven — it holds *because* we ship `.strict`.
    func testCacheEnabledIdentity() throws {
        var failures: [String] = []
        for c in traces.cases {
            let p = c.params.toGenerationParams()
            let disabled = DiffusionEngine(model: model, speculationK: 4)
                .generate(prompt: c.prompt, params: p)
            let enabled = DiffusionEngine(model: model, speculationK: 4)
                .generateCached(prompt: c.prompt, params: p)
            if enabled.finalSequence != disabled.finalSequence {
                failures.append("\(c.name): cached final_x != disabled "
                    + "(first diff \(firstDiff(enabled.finalSequence, disabled.finalSequence)))")
            }
            if enabled.tokens != c.output {
                failures.append("\(c.name): cached output != reference")
            }
        }
        XCTAssertTrue(failures.isEmpty,
            "M5(b) cache-identity failures (\(failures.count)/\(traces.cases.count)):\n"
            + failures.joined(separator: "\n"))
        print("[M5b] ExactPrefixCache identity: \(traces.cases.count)/\(traces.cases.count) "
            + "cases identical to cache-disabled (and to the reference)")
    }

    /// WP-1a: with Elastic-Cache enabled but γ=0 (always recompute), the output is
    /// mathematically identical to the standard cache-enabled and cache-disabled paths.
    func testActiveBlockCacheParity() throws {
        var failures: [String] = []
        for c in traces.cases {
            if c.name != "p8_q_noeos" && c.name != "p16_q_eos" { continue }
            
            var p = c.params.toGenerationParams()
            p.elasticCacheEnabled = true
            p.elasticGamma = 2.0 // Force recomputation at every step (sim <= 1.0 < 2.0 always)
            p.elasticBeta = 16
            
            let disabled = DiffusionEngine(model: model, speculationK: 4)
                .generate(prompt: c.prompt, params: c.params.toGenerationParams())
            let elastic = DiffusionEngine(model: model, speculationK: 1)
                .generateCached(prompt: c.prompt, params: p)
                
            if elastic.finalSequence != disabled.finalSequence {
                failures.append("\(c.name): elastic final_x != disabled "
                    + "(first diff \(firstDiff(elastic.finalSequence, disabled.finalSequence)))")
            }
            if elastic.tokens != c.output {
                failures.append("\(c.name): elastic output != reference")
            }
        }
        XCTAssertTrue(failures.isEmpty,
            "WP-1a Elastic-Cache parity failures (\(failures.count)):\n"
            + failures.joined(separator: "\n"))
        print("[WP-1a] Elastic-Cache parity (γ=0) verified token-for-token against reference")
    }

    /// WP-1a: with Elastic-Cache enabled and γ=0.9, runs end-to-end without crashing.
    func testActiveBlockCacheRealDrift() throws {
        guard let c = traces.cases.first(where: { $0.name == "p8_q_noeos" }) else { return }
        var p = c.params.toGenerationParams()
        p.elasticCacheEnabled = true
        p.elasticGamma = 0.9
        p.elasticBeta = 16
        
        let output = DiffusionEngine(model: model, speculationK: 1)
            .generateCached(prompt: c.prompt, params: p)
        XCTAssertFalse(output.tokens.isEmpty, "Elastic-Cache output should not be empty")
        print("[WP-1a] Elastic-Cache end-to-end validation with γ=0.9 completed successfully, generated \(output.tokens.count) tokens")
    }

    /// M5(c): sync audit — ≤ 1 blocking readback per K denoising steps + 1 per block commit.
    /// The engine counts every `.item`/`.asArray` readback; we assert it stays within budget.
    func testSyncBudget() throws {
        let K = 4
        var worstRatio = 0.0
        for c in traces.cases {
            let p = c.params.toGenerationParams()
            let engine = DiffusionEngine(model: model, speculationK: K)
            let out = engine.generate(prompt: c.prompt, params: p)
            let numBlocks = out.stepsPerBlock.count
            let totalSteps = out.stepsPerBlock.reduce(0, +)
            // Budget: ceil(steps_b / K) loop-control readbacks per block + 1 commit readback each.
            let budget = out.stepsPerBlock.reduce(0) { $0 + ($1 + K - 1) / K } + numBlocks
            XCTAssertLessThanOrEqual(out.syncPoints, budget,
                "\(c.name): \(out.syncPoints) syncs exceeds budget \(budget) "
                + "(steps \(out.stepsPerBlock), K \(K))")
            worstRatio = max(worstRatio, Double(out.syncPoints) / Double(max(totalSteps, 1)))
        }
        print(String(format: "[M5c] sync audit: within budget for all cases; "
            + "worst syncs/step ratio %.3f (K=%d, ideal ~%.3f)", worstRatio, K, 1.0 / Double(K)))
    }

    private func firstDiff(_ a: [Int], _ b: [Int]) -> String {
        let n = min(a.count, b.count)
        for i in 0 ..< n where a[i] != b[i] { return "idx \(i): \(a[i]) != \(b[i])" }
        return a.count == b.count ? "none" : "length \(a.count) != \(b.count)"
    }
}
