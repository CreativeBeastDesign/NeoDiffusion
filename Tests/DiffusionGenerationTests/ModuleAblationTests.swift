import XCTest
import MLX
import DiffusionCore
@testable import DiffusionModel
@testable import DiffusionGeneration

/// Guards for the in-situ module-ablation diagnostic (`Plans/gather_qmm_handoff.md` §5.5).
///
/// Two things must hold, and they pull in opposite directions:
///  1. **Inert by default** — ablation must not exist on the served path.
///  2. **Actually reachable** — it must bite on the path the *served default config* takes.
///
/// (2) is not paranoia. `DiffusionEngine+Entry` has several forward overloads, and the served
/// default (elastic/JOT/FlashBlock all off) goes through the plain M5 cached overload — **not**
/// the JOT/capacity one. Wiring the switch into the wrong overload yields an ablation that
/// silently does nothing, which is indistinguishable from "the module is free" and would have
/// burned a Studio session on a fake null result. `testAblationActuallyBitesOnDefaultPath` is
/// the check that catches that class of mistake.
final class ModuleAblationTests: XCTestCase {

    var manifest: DenoisingLoopParityTests.Manifest!
    var traces: DenoisingLoopParityTests.Traces!
    var model: LLaDA2MoeModel!

    override func setUpWithError() throws {
        let dir = DenoisingLoopParityTests.fixtureDir
        let tracesURL = dir.appendingPathComponent("traces.json")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: tracesURL.path),
            "Loop fixtures missing at \(dir.path) — run: python3 Tools/generate_loop_fixtures.py")
        manifest = try JSONDecoder().decode(
            DenoisingLoopParityTests.Manifest.self,
            from: Data(contentsOf: dir.appendingPathComponent("manifest.json")))
        traces = try JSONDecoder().decode(
            DenoisingLoopParityTests.Traces.self, from: Data(contentsOf: tracesURL))
        let dm = DiffusionModel(config: manifest.config)
        try dm.loadWeights(from: dir.appendingPathComponent("weights.safetensors"))
        model = dm.model
    }

    /// Default is `.none`, and the served presets never turn it on.
    func testAblationIsOffByDefault() throws {
        for mode: GenerationParams.Mode in [.q, .s] {
            let p = GenerationParams.mode(mode, blockLength: 32, genLength: 32, maskId: 0, eosId: 1)
            XCTAssertEqual(p.moduleAblation, ModuleAblation.none,
                           "served preset \(mode) must not carry an ablation")
        }
        for c in traces.cases {
            XCTAssertEqual(c.params.toGenerationParams().moduleAblation, ModuleAblation.none,
                           "fixture params must not carry an ablation")
        }
    }

    /// `.none` must reproduce the baseline token-for-token — the ablation switch must not perturb
    /// the served path merely by existing (the analogue of `testJotParityWhenDisabled`).
    func testNoneIsBitIdenticalToBaseline() throws {
        for c in traces.cases {
            let p = c.params.toGenerationParams()
            var pNone = p
            pNone.moduleAblation = .none

            let base = DiffusionEngine(model: model, speculationK: 4)
                .generateCached(prompt: c.prompt, params: p)
            let none = DiffusionEngine(model: model, speculationK: 4)
                .generateCached(prompt: c.prompt, params: pNone)

            XCTAssertEqual(none.finalSequence, base.finalSequence,
                           "\(c.name): .none must not change the served trajectory")
            XCTAssertEqual(none.metrics.effectiveModuleAblation, ModuleAblation.none)
        }
    }

    /// **The load-bearing test.** Every ablation must actually reach the forward on the path the
    /// *served default* uses, and must therefore change the output. If an arm's output matches the
    /// baseline, the switch never fired — the experiment would silently measure nothing.
    ///
    /// Asserting on `finalSequence` (not timing) is deliberate: a unit test cannot measure the ms
    /// deltas we are after, but it *can* prove the code path is live, which is the part that
    /// silently breaks.
    func testAblationActuallyBitesOnDefaultPath() throws {
        for c in traces.cases {
            let p = c.params.toGenerationParams()
            // Confirm the fixture exercises the served default path (no JOT/FlashBlock/elastic),
            // otherwise this test would prove nothing about production.
            XCTAssertFalse(p.jotEnabled || p.flashBlockEnabled || p.elasticCacheEnabled,
                           "\(c.name): fixture must run the served default path for this guard")

            let base = DiffusionEngine(model: model, speculationK: 4)
                .generateCached(prompt: c.prompt, params: p)

            for ablation: ModuleAblation in [.moeRoutedExperts, .moeExpertGEMMs, .moeFixedExperts, .moeAll, .attention] {
                var pa = p
                pa.moduleAblation = ablation
                let out = DiffusionEngine(model: model, speculationK: 4)
                    .generateCached(prompt: c.prompt, params: pa)

                XCTAssertEqual(out.metrics.effectiveModuleAblation, ablation,
                               "\(c.name)/\(ablation): effective echo must report what ran")
                XCTAssertNotEqual(
                    out.finalSequence, base.finalSequence,
                    """
                    \(c.name)/\(ablation): ablated output is identical to baseline — the switch \
                    did not reach the forward. Check that DiffusionEngine+Entry passes \
                    params.moduleAblation on the overload this config actually takes.
                    """)
            }
        }
    }

    /// **The DCE guard.** `.moeExpertGEMMs` is the one arm that keeps a module running *without*
    /// the thing that normally consumes it, so it is the one arm MLX's lazy graph could silently
    /// optimise away — leaving us measuring "no router" while believing we measured "router only".
    /// Timing cannot detect that (an eliminated router and a free router look identical), so it is
    /// asserted behaviourally instead: this arm's output depends on `weights`, and `weights =
    /// takeAlong(scores, indices)` depends on the whole top-k chain. If the router were elided,
    /// the output would collapse to `.moeRoutedExperts`'s (shared expert alone). It must not.
    func testExpertGEMMsAblationStillRunsTheRouter() throws {
        for c in traces.cases {
            var pGEMMs = c.params.toGenerationParams()
            pGEMMs.moduleAblation = .moeExpertGEMMs
            var pRouted = c.params.toGenerationParams()
            pRouted.moduleAblation = .moeRoutedExperts

            let gemms = DiffusionEngine(model: model, speculationK: 4)
                .generateCached(prompt: c.prompt, params: pGEMMs)
            let routed = DiffusionEngine(model: model, speculationK: 4)
                .generateCached(prompt: c.prompt, params: pRouted)

            XCTAssertNotEqual(
                gemms.finalSequence, routed.finalSequence,
                """
                \(c.name): .moeExpertGEMMs produced the same output as .moeRoutedExperts, which \
                skips the router entirely. The router's `weights` is therefore NOT reaching the \
                output — MLX has elided it, and any timing from this arm measures 'no router', \
                not 'router only'. Do not trust the router/gather_qmm split until this passes.
                """)
        }
    }

    /// The ablations must be ordered by how much work they remove: skipping the whole MoE removes
    /// strictly more than skipping only its routed half. Cheap structural check that
    /// `.moeRoutedExperts` really does retain the shared expert rather than collapsing to
    /// `.moeAll` (which would silently mis-attribute the shared-expert cost to the routed path).
    func testRoutedAndAllAblationsDiffer() throws {
        for c in traces.cases {
            var pRouted = c.params.toGenerationParams()
            pRouted.moduleAblation = .moeRoutedExperts
            var pAll = c.params.toGenerationParams()
            pAll.moduleAblation = .moeAll

            let routed = DiffusionEngine(model: model, speculationK: 4)
                .generateCached(prompt: c.prompt, params: pRouted)
            let all = DiffusionEngine(model: model, speculationK: 4)
                .generateCached(prompt: c.prompt, params: pAll)

            XCTAssertNotEqual(routed.finalSequence, all.finalSequence,
                              "\(c.name): .moeRoutedExperts must retain the shared expert, so it "
                              + "cannot be identical to .moeAll")
        }
    }
}

/// Guards for router reuse (`GenerationParams.routerReuseSteps`) — experimental, default off.
final class RouterReuseTests: XCTestCase {
    var manifest: DenoisingLoopParityTests.Manifest!
    var traces: DenoisingLoopParityTests.Traces!
    var model: LLaDA2MoeModel!

    override func setUpWithError() throws {
        let dir = DenoisingLoopParityTests.fixtureDir
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: dir.appendingPathComponent("traces.json").path),
            "Loop fixtures missing at \(dir.path)")
        manifest = try JSONDecoder().decode(
            DenoisingLoopParityTests.Manifest.self,
            from: Data(contentsOf: dir.appendingPathComponent("manifest.json")))
        traces = try JSONDecoder().decode(
            DenoisingLoopParityTests.Traces.self,
            from: Data(contentsOf: dir.appendingPathComponent("traces.json")))
        let dm = DiffusionModel(config: manifest.config)
        try dm.loadWeights(from: dir.appendingPathComponent("weights.safetensors"))
        model = dm.model
    }

    /// Off by default, and `0`/`1` must be bit-identical to the baseline: the reuse plumbing must
    /// not perturb the served path merely by existing.
    func testReuseOffIsBitIdentical() throws {
        for c in traces.cases {
            let p = c.params.toGenerationParams()
            XCTAssertEqual(p.routerReuseSteps, 0, "reuse must be off by default")
            for n in [0, 1] {
                var pr = p; pr.routerReuseSteps = n
                let base = DiffusionEngine(model: model, speculationK: 4)
                    .generateCached(prompt: c.prompt, params: p)
                let out = DiffusionEngine(model: model, speculationK: 4)
                    .generateCached(prompt: c.prompt, params: pr)
                XCTAssertEqual(out.finalSequence, base.finalSequence,
                               "\(c.name): routerReuseSteps=\(n) must be inert")
            }
        }
    }

    /// Reuse must actually *bite* — if the output is unchanged, the reuse path never fired and any
    /// speed reading from it would be measuring nothing (the `attr-*` lesson).
    func testReuseChangesBehaviour() throws {
        var changed = false
        for c in traces.cases {
            let p = c.params.toGenerationParams()
            var pr = p; pr.routerReuseSteps = 2
            let base = DiffusionEngine(model: model, speculationK: 4)
                .generateCached(prompt: c.prompt, params: p)
            let out = DiffusionEngine(model: model, speculationK: 4)
                .generateCached(prompt: c.prompt, params: pr)
            if out.finalSequence != base.finalSequence { changed = true }
        }
        XCTAssertTrue(changed,
            "routerReuseSteps=2 changed no output on any fixture — the reuse path is not firing, "
            + "so its timings would be meaningless. (A toy fixture settling in <2 steps could also "
            + "explain this; check stepsPerBlock before trusting a green result here.)")
    }

    /// Reuse state must not leak between generations — a decision cached from another prompt is
    /// nonsense. Running B after A must equal running B alone.
    func testReuseStateDoesNotLeakAcrossGenerations() throws {
        guard traces.cases.count >= 2 else { throw XCTSkip("need 2 fixture cases") }
        let a = traces.cases[0], b = traces.cases[1]
        var pr = b.params.toGenerationParams(); pr.routerReuseSteps = 2
        var pa = a.params.toGenerationParams(); pa.routerReuseSteps = 2

        let bAlone = DiffusionEngine(model: model, speculationK: 4)
            .generateCached(prompt: b.prompt, params: pr)
        let engine = DiffusionEngine(model: model, speculationK: 4)
        _ = engine.generateCached(prompt: a.prompt, params: pa)
        let bAfterA = engine.generateCached(prompt: b.prompt, params: pr)

        XCTAssertEqual(bAfterA.finalSequence, bAlone.finalSequence,
            "router-reuse state leaked from the previous generation — resetRouterCache() is not "
            + "being called, and results depend on what ran before.")
    }
}
