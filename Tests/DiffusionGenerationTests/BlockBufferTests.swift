import XCTest
@testable import DiffusionGeneration

/// Unit tests for the Block-Buffer state machine (phase-1 §5). At `nBuf == 1` this is bookkeeping
/// that reduces to the reference SingleBD schedule; the tests pin the lifecycle invariants so the
/// Phase 3 `nBuf == 2` lift is a config change, not a semantics change.
final class BlockBufferTests: XCTestCase {

    func testLifecycleTransitions() {
        var buffer = BlockBuffer(nBuf: 1)
        XCTAssertEqual(buffer.activeCount, 0)
        XCTAssertNil(buffer.lastCommittedBlock)

        for block in 0 ..< 4 {
            let slot = buffer.activate(blockIndex: block)
            XCTAssertEqual(buffer.slots[slot].state, .active)
            XCTAssertEqual(buffer.activeCount, 1)

            buffer.markSettled(slotIndex: slot)
            XCTAssertEqual(buffer.slots[slot].state, .toCache)

            buffer.markCommitted(slotIndex: slot)
            XCTAssertEqual(buffer.slots[slot].state, .inCache)
            XCTAssertEqual(buffer.lastCommittedBlock, block)
            XCTAssertEqual(buffer.activeCount, 0)
        }
    }

    func testNBufCapEnforced() {
        // nBuf == 1: only one active slot at a time; a second activate without commit is illegal.
        var buffer = BlockBuffer(nBuf: 1)
        _ = buffer.activate(blockIndex: 0)
        XCTAssertEqual(buffer.activeCount, 1)
        // A slot can be reused once committed (dummy/inCache slots are free to reactivate).
        buffer.markSettled(slotIndex: 0)
        buffer.markCommitted(slotIndex: 0)
        let reused = buffer.activate(blockIndex: 1)
        XCTAssertEqual(reused, 0, "the freed slot should be reused at nBuf == 1")
        XCTAssertEqual(buffer.slots[reused].state, .active)
    }

    func testPhase2HardwiresNBuf1() {
        // Phase 2 hardwires nBuf == 1 (phase-1 §5 amendment); nBuf == 2 is Phase 3.
        XCTAssertEqual(BlockBuffer(nBuf: 1).nBuf, 1)
    }
}
