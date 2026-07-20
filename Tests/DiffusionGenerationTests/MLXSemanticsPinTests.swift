import XCTest
import MLX
import MLXNN

/// Pin MLX primitive behaviors by test, converting inferred claims into sourced-by-test facts.
/// These tests validate critical control-path assumptions used in threshold decoding and
/// masked-position confidence handling.
///
/// House provenance rule (CLAUDE.md working rules): sourced claims replace inferred ones.
final class MLXSemanticsPinTests: XCTestCase {

    // MARK: - argMax tie-break behavior

    /// pins: MLX argMax tie-break = first index — the Metal control kernel must reproduce this
    func testArgMaxFirstIndexTieBreak() throws {
        // Case 1: maxima at positions 1 and 2, argMax should return 1 (first)
        do {
            let arr = MLXArray([0.5, 0.9, 0.9, 0.1] as [Float])
            let idx = argMax(arr, axis: -1)
            let result = idx.item(Int32.self)
            XCTAssertEqual(result, 1, "argMax([0.5, 0.9, 0.9, 0.1]) should return index 1 (first max)")
        }

        // Case 2: all values equal
        do {
            let arr = MLXArray([0.7, 0.7, 0.7, 0.7] as [Float])
            let idx = argMax(arr, axis: -1)
            let result = idx.item(Int32.self)
            XCTAssertEqual(result, 0, "argMax with all equal values should return index 0 (first)")
        }

        // Case 3: tie at positions 0 and 3 (first and last)
        do {
            let arr = MLXArray([1.0, 0.5, 0.5, 1.0] as [Float])
            let idx = argMax(arr, axis: -1)
            let result = idx.item(Int32.self)
            XCTAssertEqual(result, 0, "argMax with tie at first and last should return index 0 (first)")
        }

        // Case 4: max at position 0
        do {
            let arr = MLXArray([2.0, 1.5, 1.5, 1.0] as [Float])
            let idx = argMax(arr, axis: -1)
            let result = idx.item(Int32.self)
            XCTAssertEqual(result, 0, "argMax with max at position 0 should return 0")
        }
    }

    // MARK: - Strict greater-than with -inf

    /// pins: MLX (a .> τ) is strict inequality; -inf .> -inf is false; -inf .> finite is false
    /// Critical for masked-position confidence thresholding: -inf padded positions must never pass τ
    func testStrictGreaterWithNegInf() throws {
        let negInf = MLXArray(Float(-Float.infinity))

        // Array containing all test values
        let arr = MLXArray([Float(-Float.infinity), Float(-0.5), 0.0, 0.7, 0.8] as [Float])
        let threshold = MLXArray(Float(0.7))

        // Compute strict greater-than
        let result = (arr .> threshold)
        let resultVals = result.asArray(Bool.self)

        // Position 0: -inf > 0.7? NO
        XCTAssertFalse(resultVals[0], "-inf .> 0.7 should be false")

        // Position 1: -0.5 > 0.7? NO
        XCTAssertFalse(resultVals[1], "-0.5 .> 0.7 should be false")

        // Position 2: 0.0 > 0.7? NO
        XCTAssertFalse(resultVals[2], "0.0 .> 0.7 should be false")

        // Position 3: 0.7 > 0.7? NO (strict, not >=)
        XCTAssertFalse(resultVals[3], "0.7 .> 0.7 should be false (strict inequality)")

        // Position 4: 0.8 > 0.7? YES
        XCTAssertTrue(resultVals[4], "0.8 .> 0.7 should be true")

        // Isolated: -inf .> -inf is false
        do {
            let cond = (negInf .> negInf)
            let condVal = cond.item(Bool.self)
            XCTAssertFalse(condVal, "-inf .> -inf should be false")
        }
    }

    // MARK: - argMax with -inf padding

    /// pins: argMax on -inf-padded array returns the non-inf position; argMax of all-inf returns 0
    /// Control kernel top-1 fallback: masked positions are -inf, argMax selects first unmasked
    func testArgMaxWithNegInfPadding() throws {
        // Case 1: one position is not -inf, rest are -inf; argMax should select the non-inf
        do {
            let arr = MLXArray([Float(-Float.infinity), 0.5, Float(-Float.infinity), Float(-Float.infinity)] as [Float])
            let idx = argMax(arr, axis: -1)
            let result = idx.item(Int32.self)
            XCTAssertEqual(result, 1, "argMax with one non-inf should return its index (1)")
        }

        // Case 2: multiple non-inf, argMax selects the max
        do {
            let arr = MLXArray([Float(-Float.infinity), 0.3, 0.9, Float(-Float.infinity)] as [Float])
            let idx = argMax(arr, axis: -1)
            let result = idx.item(Int32.self)
            XCTAssertEqual(result, 2, "argMax should select the maximum non-inf value at index 2")
        }

        // Case 3: all -inf; argMax returns 0 (first index, consistent with all-equal case)
        do {
            let arr = MLXArray([Float(-Float.infinity), Float(-Float.infinity), Float(-Float.infinity), Float(-Float.infinity)] as [Float])
            let idx = argMax(arr, axis: -1)
            let result = idx.item(Int32.self)
            XCTAssertEqual(result, 0, "argMax of all-inf should return 0 (first index)")
        }

        // Case 4: tie between -inf and -inf (all -inf) across a larger span
        do {
            let arr = MLXArray([Float(-Float.infinity), Float(-Float.infinity), Float(-Float.infinity)] as [Float])
            let idx = argMax(arr, axis: -1)
            let result = idx.item(Int32.self)
            XCTAssertEqual(result, 0, "argMax([−∞, −∞, −∞]) should return 0")
        }
    }

    // MARK: - which (ternary select)

    /// pins: MLX which(cond, ifTrue, ifFalse) picks ifTrue where cond is true, ifFalse where false
    /// Used in masked selection: where conf > τ, select edited token; else select cached token
    func testWhichSelectsElementwise() throws {
        // Simple case: bool condition array, two value arrays
        let cond = MLXArray([true, false, true, false] as [Bool])
        let ifTrue = MLXArray([1.0, 1.0, 1.0, 1.0] as [Float])
        let ifFalse = MLXArray([2.0, 2.0, 2.0, 2.0] as [Float])

        let result = which(cond, ifTrue, ifFalse)
        let resultVals = result.asArray(Float.self)

        // Position 0: cond=true, should pick ifTrue (1.0)
        XCTAssertEqual(resultVals[0], 1.0, "which(true, 1.0, 2.0) should return 1.0")

        // Position 1: cond=false, should pick ifFalse (2.0)
        XCTAssertEqual(resultVals[1], 2.0, "which(false, 1.0, 2.0) should return 2.0")

        // Position 2: cond=true, should pick ifTrue (1.0)
        XCTAssertEqual(resultVals[2], 1.0, "which(true, 1.0, 2.0) should return 1.0")

        // Position 3: cond=false, should pick ifFalse (2.0)
        XCTAssertEqual(resultVals[3], 2.0, "which(false, 1.0, 2.0) should return 2.0")

        // Verify entire result array is correct
        let expected: [Float] = [1.0, 2.0, 1.0, 2.0]
        for i in 0..<expected.count {
            XCTAssertEqual(resultVals[i], expected[i],
                "which result[\(i)] mismatch")
        }
    }
}
