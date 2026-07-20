import XCTest
import MLX
import MLXNN
@testable import DiffusionCore

/// Task T2a (gate G0) of the control-kernel campaign: prove — BEFORE the real control kernel is
/// built — that an `MLXFast.metalKernel` with the planned I/O signature works inside the engine's
/// K-step chained lazy-graph pattern (mirrors `DiffusionEngine+Step.swift`'s per-step loop, which
/// dispatches K kernels back-to-back with no `eval` in between, then does ONE blocking readback of
/// the concatenated `[K]` break-flag array — see `DiffusionEngine+Step.swift:139-219`).
///
/// This file declares a minimal pass-through "control kernel" with the exact planned signature
/// (inputs: windowActive/x0/x0p/promptMask/frozenMask/posts/ctrlState/rtScalars; outputs:
/// nextWindow/resultWindow/breakFlag/nextPosts/stats/nextCtrlState) and drives it through four
/// sub-probes:
///   1. Do declared dtypes — in particular `.bool` — compile and dispatch as metalKernel I/O?
///   2. Do outputs compose as lazy graph nodes when chained K=4 times with no intermediate eval?
///   3. Does a `#pragma STDC FP_CONTRACT OFF` kernel source still compile/dispatch correctly?
///   4. Does `concatenated` on `[1]`-shaped Bool kernel outputs read back correctly (the exact
///      seam `denoisePhase` uses for its `[2K]` flags array)?
///
/// Every sub-probe wraps its GPU-touching work in `withError { ... }` (mlx-swift's scoped error
/// handler, `ErrorHandler.swift`) rather than calling the kernel bare: the *default* MLX error
/// handler calls `fatalError` on any C++-side exception (Metal compile failure, bad grid config,
/// etc.), which would abort the whole `swift test` process and destroy the diagnostic value of
/// this file. `withError` converts that into a catchable Swift `MLXError`, so a real compile/
/// dispatch failure becomes a clear, recorded `XCTFail` instead of a process crash.
final class ControlKernelFeasibilityTests: XCTestCase {

    // MARK: - Shared control-kernel declaration (the planned real-kernel I/O signature)

    /// Window width baked into the toy kernel — arbitrary, matches the task's [1,32] shapes.
    static let B = 32

    /// Builds the pass-through control kernel with the exact planned I/O signature. Body is a
    /// trivial copy/increment per the task spec, but every declared tensor is read from or written
    /// to at least once so this is a genuine dtype-interop probe, not a dead-code no-op:
    ///   - `nextWindow`      = `windowActive` (straight copy)
    ///   - `resultWindow`    = `promptMask ? windowActive : x0` (reads the Bool INPUT to steer an
    ///                          Int32 output — the input-side Bool probe)
    ///   - `breakFlag`       = `posts[0] > rtScalars[0]` (writes a Bool OUTPUT from an Int32
    ///                          comparison — the output-side Bool probe)
    ///   - `nextPosts`       = `posts[0] + 1`
    ///   - `stats`           = `[x0p[0], rtScalars[0], rtScalars[1], rtScalars[2], frozenMask[0]]`
    ///                          (touches the FP32 and second Bool input)
    ///   - `nextCtrlState`   = `ctrlState + 1` (elementwise, UInt32)
    static func makeControlKernel() -> MLXFast.MLXFastKernel {
        let source = """
            uint elem = thread_position_in_grid.x;
            if (elem < \(B)u) {
                nextWindow[elem] = windowActive[elem];
                resultWindow[elem] = promptMask[elem] ? windowActive[elem] : x0[elem];
                nextCtrlState[elem] = ctrlState[elem] + 1u;
            }
            if (elem == 0u) {
                nextPosts[0] = posts[0] + 1;
                breakFlag[0] = (posts[0] > rtScalars[0]);
                stats[0] = x0p[0];
                stats[1] = float(rtScalars[0]);
                stats[2] = float(rtScalars[1]);
                stats[3] = float(rtScalars[2]);
                stats[4] = frozenMask[0] ? 1.0f : 0.0f;
            }
            """
        return MLXFast.metalKernel(
            name: "control_kernel_feasibility_probe",
            inputNames: [
                "windowActive", "x0", "x0p", "promptMask", "frozenMask", "posts", "ctrlState",
                "rtScalars",
            ],
            outputNames: [
                "nextWindow", "resultWindow", "breakFlag", "nextPosts", "stats", "nextCtrlState",
            ],
            source: source
        )
    }

    /// Standard grid for the [1,32]-shaped-window kernel: one thread per window element.
    static let grid: (Int, Int, Int) = (B, 1, 1)
    static let threadGroup: (Int, Int, Int) = (B, 1, 1)
    static let outputShapes: [[Int]] = [[1, B], [1, B], [1], [1], [5], [1, B]]
    static let outputDTypes: [DType] = [.int32, .int32, .bool, .int32, .float32, .uint32]

    /// Builds one call's worth of inputs in the kernel's declared `inputNames` order.
    static func makeInputs(
        windowActive: [Int32], x0: [Int32], x0p: [Float], promptMask: [Bool], frozenMask: [Bool],
        posts: Int32, ctrlState: [UInt32], rtScalars: [Int32]
    ) -> [MLXArray] {
        [
            MLXArray(windowActive, [1, B]),
            MLXArray(x0, [1, B]),
            MLXArray(x0p, [1, B]),
            MLXArray(promptMask, [1, B]),
            MLXArray(frozenMask, [1, B]),
            MLXArray([posts], [1]),
            MLXArray(ctrlState, [1, B]),
            MLXArray(rtScalars, [3]),
        ]
    }

    // MARK: - 1. I/O dtypes: does `.bool` work as metalKernel input AND output?

    /// KEY QUESTION for the real kernel's interface: do `.bool` MLXArrays work as metalKernel
    /// inputs (`promptMask`, `frozenMask`) AND outputs (`breakFlag`) simultaneously?
    ///
    /// This is structured to make the finding explicit either way: if dispatch throws (compile
    /// error) or produces wrong values, this XCTFails with the exact MLX error message — the
    /// primary deliverable for the real kernel's interface decision. If it succeeds, the assertions
    /// below are the positive finding (bool I/O works, no fallback needed).
    func testBoolIODispatchesAndReturnsCorrectValues() throws {
        let kernel = Self.makeControlKernel()

        let windowActive: [Int32] = (0 ..< Int32(Self.B)).map { $0 }
        let x0: [Int32] = (0 ..< Int32(Self.B)).map { $0 + 1000 }
        let x0p: [Float] = (0 ..< Self.B).map { Float($0) * 0.5 }
        // Alternating promptMask so resultWindow must genuinely select per-element, not just
        // copy one side wholesale (would silently pass a broken bool read).
        let promptMask: [Bool] = (0 ..< Self.B).map { $0 % 2 == 0 }
        let frozenMask: [Bool] = Array(repeating: true, count: Self.B)
        let ctrlState: [UInt32] = Array(repeating: 0, count: Self.B)
        let rtScalars: [Int32] = [2, 10, 20]

        var outputs: [MLXArray] = []
        do {
            outputs = try withError {
                let out = kernel(
                    Self.makeInputs(
                        windowActive: windowActive, x0: x0, x0p: x0p, promptMask: promptMask,
                        frozenMask: frozenMask, posts: 0, ctrlState: ctrlState, rtScalars: rtScalars
                    ),
                    grid: Self.grid, threadGroup: Self.threadGroup,
                    outputShapes: Self.outputShapes, outputDTypes: Self.outputDTypes
                )
                eval(out)
                return out
            }
        } catch {
            XCTFail(
                "G0 FINDING: bool I/O FAILED to compile/dispatch on the planned control-kernel "
                    + "signature (promptMask/frozenMask inputs, breakFlag output all .bool). "
                    + "MLX error: \(error). The real kernel must fall back to uint8 or int32 for "
                    + "these tensors — record this verbatim in the T2 handoff.")
            return
        }

        XCTAssertEqual(outputs.count, 6, "kernel must produce all 6 declared outputs")

        let nextWindow = outputs[0].asArray(Int32.self)
        let resultWindow = outputs[1].asArray(Int32.self)
        let breakFlag = outputs[2].asArray(Bool.self)
        let nextPosts = outputs[3].asArray(Int32.self)
        let stats = outputs[4].asArray(Float.self)
        let nextCtrlState = outputs[5].asArray(UInt32.self)

        XCTAssertEqual(outputs[2].dtype, .bool, "breakFlag output must actually be .bool, not silently promoted")
        XCTAssertEqual(nextWindow, windowActive, "nextWindow must be a straight copy of windowActive")

        let expectedResultWindow = (0 ..< Self.B).map { promptMask[$0] ? windowActive[$0] : x0[$0] }
        XCTAssertEqual(
            resultWindow, expectedResultWindow,
            "resultWindow must select per-element via the Bool promptMask input — a wrong value "
                + "here (not a crash) is exactly the 'bool input silently misreads' failure mode")

        XCTAssertEqual(breakFlag, [false], "posts[0]=0 is not > rtScalars[0]=2 — breakFlag must be false")
        XCTAssertEqual(nextPosts, [1])
        XCTAssertEqual(stats, [x0p[0], 2.0, 10.0, 20.0, 1.0], "stats must reflect x0p[0], rtScalars, and frozenMask[0]")
        XCTAssertEqual(nextCtrlState, Array(repeating: UInt32(1), count: Self.B), "nextCtrlState must be ctrlState+1 elementwise")

        // Positive finding, printed for the T2 handoff (AGENTS.md effective-echo discipline: show
        // the mechanism actually engaged, don't just assert silently).
        print("[G0 finding] bool I/O: PASS — .bool works as both metalKernel input (promptMask, "
            + "frozenMask) and output (breakFlag); no fallback dtype needed for the real kernel.")
    }

    // MARK: - 2. Purity / K-chaining: outputs compose as lazy graph nodes, no eval between calls

    /// Chains the control kernel K=4 times with NO eval between calls — step i's
    /// nextCtrlState/nextWindow/nextPosts outputs feed step i+1's ctrlState/windowActive/posts
    /// inputs, exactly mirroring `DiffusionEngine+Step.swift`'s K-step speculative loop (l.139-219:
    /// `breakFlags.append(s.breakFlag)` across K iterations, then ONE
    /// `concatenated(breakFlags + activationFlags, axis: 0)` + `eval` + `asArray(Bool.self)`).
    ///
    /// rtScalars[0] (the break threshold) is fixed at 1 across the chain, and posts starts at 0,
    /// so the predicted per-step sequence is:
    ///   step 1: posts_in=0 -> breakFlag = (0>1) = false, nextPosts=1
    ///   step 2: posts_in=1 -> breakFlag = (1>1) = false, nextPosts=2
    ///   step 3: posts_in=2 -> breakFlag = (2>1) = true,  nextPosts=3
    ///   step 4: posts_in=3 -> breakFlag = (3>1) = true,  nextPosts=4
    /// i.e. breakFlags == [false, false, true, true] (flips at step 3), and ctrlState goes 0->4
    /// elementwise. If any intermediate `eval` were accidentally required, this would either crash
    /// (using an un-evaluated array across kernel calls is exactly what lazy graphs must support)
    /// or the values would come out wrong; catching that is the point of this test.
    func testKStepChainedLazyGraphComposesCorrectly() throws {
        let kernel = Self.makeControlKernel()
        let K = 4

        // Fixed across the chain (not looped outputs per the task spec).
        let x0: [Int32] = (0 ..< Int32(Self.B)).map { $0 + 1000 }
        let x0p: [Float] = Array(repeating: 0.25, count: Self.B)
        let promptMask: [Bool] = Array(repeating: false, count: Self.B)
        let frozenMask: [Bool] = Array(repeating: false, count: Self.B)
        let rtScalars: [Int32] = [1, 0, 0]

        // Looped state — reassigned each iteration from the PREVIOUS step's lazy outputs, with no
        // eval() call anywhere in this loop.
        var windowActive: MLXArray = MLXArray((0 ..< Int32(Self.B)).map { $0 }, [1, Self.B])
        var ctrlState: MLXArray = MLXArray(Array(repeating: UInt32(0), count: Self.B), [1, Self.B])
        var posts: MLXArray = MLXArray([Int32(0)], [1])

        var breakFlags: [MLXArray] = []

        for _ in 0 ..< K {
            let out = kernel(
                [
                    windowActive,
                    MLXArray(x0, [1, Self.B]),
                    MLXArray(x0p, [1, Self.B]),
                    MLXArray(promptMask, [1, Self.B]),
                    MLXArray(frozenMask, [1, Self.B]),
                    posts,
                    ctrlState,
                    MLXArray(rtScalars, [3]),
                ],
                grid: Self.grid, threadGroup: Self.threadGroup,
                outputShapes: Self.outputShapes, outputDTypes: Self.outputDTypes
            )
            // out[0]=nextWindow, out[1]=resultWindow, out[2]=breakFlag, out[3]=nextPosts,
            // out[4]=stats, out[5]=nextCtrlState — feed the three looped tensors forward per the
            // task spec; deliberately NOT calling eval() here.
            windowActive = out[0]
            posts = out[3]
            ctrlState = out[5]
            breakFlags.append(out[2])
        }

        // Mirrors DiffusionEngine+Step.swift:201 exactly: concat the K [1]-Bool flags into one
        // [K] array, ONE eval, ONE readback.
        let flags = concatenated(breakFlags, axis: 0)
        XCTAssertEqual(flags.shape, [K], "concatenated K [1]-Bool outputs must yield [K]")

        do {
            try withError {
                eval(flags, ctrlState, posts, windowActive)
            }
        } catch {
            XCTFail(
                "G0 FINDING: the K=4 chained lazy graph (no intermediate eval) FAILED at the single "
                    + "terminal eval. MLX error: \(error). This means metalKernel outputs cannot be "
                    + "fed back as later metalKernel inputs purely lazily — a load-bearing assumption "
                    + "for the real control kernel's K-step loop.")
            return
        }

        XCTAssertEqual(
            flags.asArray(Bool.self), [false, false, true, true],
            "breakFlags must flip exactly at step 3 given rtScalars[0]=1 and posts starting at 0")
        XCTAssertEqual(
            ctrlState.asArray(UInt32.self), Array(repeating: UInt32(K), count: Self.B),
            "ctrlState must have accumulated exactly K increments across the unevaluated chain")
        XCTAssertEqual(posts.asArray(Int32.self), [Int32(K)], "posts must have accumulated K increments")

        print("[G0 finding] K-step chaining: PASS — \(K) metalKernel dispatches composed as pure "
            + "lazy graph nodes with zero intermediate eval; one terminal eval + readback matches "
            + "DiffusionEngine+Step.swift's K-step budget.")
    }

    // MARK: - 3. FP_CONTRACT pragma: does a kernel source starting with the pragma still compile?

    /// A minimal second kernel whose `source` string begins with `#pragma STDC FP_CONTRACT OFF`,
    /// computing `a*b+c` on float32 inputs. Inputs are small exactly-representable integers (3, 4,
    /// 5 -> 3*4+5=17) so fused vs. non-fused evaluation cannot differ — the 0-tolerance comparison
    /// against the CPU value is a compile/dispatch probe, not a numerics probe.
    ///
    /// If the pragma fails to compile in MLX's JIT wrapper (the generated source splices `source`
    /// directly into the function body per `custom_kernel.cpp`'s `write_signature`, so a stray
    /// preprocessor directive there is exactly the kind of thing that could confront an unexpected
    /// parse context), this records that finding and falls back to verifying the same kernel body
    /// WITHOUT the pragma still works — isolating "pragma specifically broke it" from "kernels are
    /// broken in general".
    func testFPContractPragmaCompilesAndDispatches() throws {
        let sourceWithPragma = """
            #pragma STDC FP_CONTRACT OFF
            uint elem = thread_position_in_grid.x;
            if (elem == 0u) {
                fmaOut[0] = a[0] * b[0] + c[0];
            }
            """
        let pragmaKernel = MLXFast.metalKernel(
            name: "control_kernel_fp_contract_probe",
            inputNames: ["a", "b", "c"],
            outputNames: ["fmaOut"],
            source: sourceWithPragma
        )

        let a = MLXArray([Float(3)], [1])
        let b = MLXArray([Float(4)], [1])
        let c = MLXArray([Float(5)], [1])
        let expected: Float = 3 * 4 + 5  // 17, exact in float32 fused or not

        do {
            let out = try withError {
                let out = pragmaKernel(
                    [a, b, c], grid: (1, 1, 1), threadGroup: (1, 1, 1),
                    outputShapes: [[1]], outputDTypes: [.float32]
                )
                eval(out)
                return out
            }
            XCTAssertEqual(
                out[0].asArray(Float.self), [expected],
                "pragma kernel compiled but produced the wrong value — 0 tolerance since inputs "
                    + "are exactly representable")
            print("[G0 finding] #pragma STDC FP_CONTRACT OFF: PASS — compiles and dispatches "
                + "correctly inside MLX's JIT kernel-body wrapper.")
        } catch {
            // Record the pragma failure, then prove the SAME body without the pragma still works
            // — isolates the pragma as the specific cause.
            print("[G0 finding] #pragma STDC FP_CONTRACT OFF: FAILED to compile. MLX error: "
                + "\(error). The real kernel must omit this pragma — record this verbatim in the "
                + "T2 handoff.")

            let sourceWithoutPragma = """
                uint elem = thread_position_in_grid.x;
                if (elem == 0u) {
                    fmaOut[0] = a[0] * b[0] + c[0];
                }
                """
            let plainKernel = MLXFast.metalKernel(
                name: "control_kernel_fp_contract_probe_no_pragma",
                inputNames: ["a", "b", "c"],
                outputNames: ["fmaOut"],
                source: sourceWithoutPragma
            )
            let out = try withError {
                let out = plainKernel(
                    [a, b, c], grid: (1, 1, 1), threadGroup: (1, 1, 1),
                    outputShapes: [[1]], outputDTypes: [.float32]
                )
                eval(out)
                return out
            }
            XCTAssertEqual(
                out[0].asArray(Float.self), [expected],
                "sanity check: the identical kernel body WITHOUT the pragma must still work, "
                    + "isolating the pragma itself as the failure cause")
            XCTFail(
                "G0 FINDING: #pragma STDC FP_CONTRACT OFF failed to compile inside MLXFast.metalKernel "
                    + "(same body without the pragma compiles fine). MLX error: \(error)")
        }
    }

    // MARK: - 4. [1]-Bool flag concat seam

    /// Asserts the exact seam `denoisePhase` uses: `concatenated([flag1, flag2], axis: 0)` on two
    /// independent `[1]`-shaped Bool kernel outputs yields a `[2]` array correctly readable via
    /// `asArray(Bool.self)`. Uses two independent (non-chained) dispatches with different `posts`
    /// values so the two flags are provably distinct (false, then true).
    func testOneShapeBoolFlagConcatSeam() throws {
        let kernel = Self.makeControlKernel()

        let windowActive: [Int32] = Array(repeating: 0, count: Self.B)
        let x0: [Int32] = Array(repeating: 0, count: Self.B)
        let x0p: [Float] = Array(repeating: 0, count: Self.B)
        let promptMask: [Bool] = Array(repeating: false, count: Self.B)
        let frozenMask: [Bool] = Array(repeating: false, count: Self.B)
        let ctrlState: [UInt32] = Array(repeating: 0, count: Self.B)
        let rtScalars: [Int32] = [1, 0, 0]  // threshold = 1

        func dispatch(posts: Int32) -> MLXArray {
            let out = kernel(
                Self.makeInputs(
                    windowActive: windowActive, x0: x0, x0p: x0p, promptMask: promptMask,
                    frozenMask: frozenMask, posts: posts, ctrlState: ctrlState, rtScalars: rtScalars
                ),
                grid: Self.grid, threadGroup: Self.threadGroup,
                outputShapes: Self.outputShapes, outputDTypes: Self.outputDTypes
            )
            return out[2]  // breakFlag
        }

        let flag1 = dispatch(posts: 0)  // 0 > 1 -> false
        let flag2 = dispatch(posts: 5)  // 5 > 1 -> true
        XCTAssertEqual(flag1.shape, [1])
        XCTAssertEqual(flag2.shape, [1])

        let flags = concatenated([flag1, flag2], axis: 0)
        XCTAssertEqual(flags.shape, [2], "concatenated two [1]-Bool outputs must yield [2]")

        try withError {
            eval(flags)
        }

        XCTAssertEqual(
            flags.asArray(Bool.self), [false, true],
            "the [1]-Bool concat seam must read back the two independently-dispatched flags correctly")
    }
}
