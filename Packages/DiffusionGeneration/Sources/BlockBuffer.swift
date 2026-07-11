import Foundation

/// Lifecycle of one block slot in the denoising buffer (phase-1 §5 amendment, 2026-07-04,
/// from `[[block-buffer]]` / `[[mbd-lms]]`).
///
/// ```
/// dummy ──activate──▶ active ──settle──▶ toCache ──commitKV──▶ inCache
/// ```
///
/// - `dummy`: unallocated — a future block not yet under refinement.
/// - `active`: currently being denoised (masked positions resolving, edits applying).
/// - `toCache`: settled (loop exited) but its committed-block KV not yet appended to
///   ``ExactPrefixCache`` — the window in which the commit-cleanliness rule (phase-2 §5 M5,
///   gotcha 6) is applied.
/// - `inCache`: committed and immutable; its KV lives in the prefix cache; never revisited.
public enum BlockSlotState: Sendable, Equatable {
    case dummy
    case active
    case toCache
    case inCache
}

/// One slot in the ``BlockBuffer``: the block it currently holds and that block's lifecycle state.
public struct BlockSlot: Sendable, Equatable {
    /// Global block index this slot holds, or `nil` when `dummy`.
    public var blockIndex: Int?
    public var state: BlockSlotState

    public static let empty = BlockSlot(blockIndex: nil, state: .dummy)
}

/// Fixed-size block buffer with a per-slot state machine (phase-1 §5).
///
/// Phase 2 ran with `nBuf == 1` (SingleBD — every M5 parity gate holds there). WP-1b
/// (training-free MultiBD, arXiv:2606.29215 Algorithm 5) raises the cap to 2: up to two
/// blocks are refined concurrently, activation gated by τ_add/τ_semi in the engine.
/// Front-block in-order commit (a slot only reaches `inCache` after all lower-indexed
/// blocks have) keeps §7 streaming semantics intact at any `nBuf`.
public struct BlockBuffer {
    public let nBuf: Int
    public private(set) var slots: [BlockSlot]
    /// Highest block index that has reached `inCache`, or `nil` if none committed yet.
    public private(set) var lastCommittedBlock: Int?

    public init(nBuf: Int = 1) {
        precondition(nBuf >= 1, "nBuf must be >= 1")
        precondition(nBuf <= 2, "WP-1b implements nBuf <= 2; larger buffers are untested "
            + "(the paper uses up to 4 — extend deliberately, with parity gates)")
        self.nBuf = nBuf
        self.slots = Array(repeating: .empty, count: nBuf)
        self.lastCommittedBlock = nil
    }

    /// Number of slots currently in the `active` state.
    public var activeCount: Int { slots.lazy.filter { $0.state == .active }.count }

    /// The index into `slots` of the single active slot, if any (nBuf=1 convenience).
    public var activeSlotIndex: Int? { slots.firstIndex { $0.state == .active } }

    /// Indices of all `active` slots, ordered by ascending block index (front first).
    public var activeSlotIndices: [Int] {
        slots.indices.filter { slots[$0].state == .active }
            .sorted { (slots[$0].blockIndex ?? .max) < (slots[$1].blockIndex ?? .max) }
    }

    /// Allocate a `dummy` slot to a new block and mark it `active`. Enforces the `nBuf` cap.
    /// Returns the slot index.
    @discardableResult
    public mutating func activate(blockIndex: Int) -> Int {
        precondition(activeCount < nBuf, "cannot exceed nBuf=\(nBuf) active slots")
        guard let idx = slots.firstIndex(where: { $0.state == .dummy || $0.state == .inCache })
        else { preconditionFailure("no free slot to activate (buffer full)") }
        slots[idx] = BlockSlot(blockIndex: blockIndex, state: .active)
        return idx
    }

    /// `active → toCache`: the block has settled; its KV is not yet in the prefix cache.
    /// Only the **front** active block may settle (in-order commit; a trailing block that
    /// finishes early simply waits in `active` until promoted).
    public mutating func markSettled(slotIndex: Int) {
        precondition(slots[slotIndex].state == .active, "markSettled requires an active slot")
        let blockIndex = slots[slotIndex].blockIndex!
        for other in slots where other.state == .active {
            precondition((other.blockIndex ?? .max) >= blockIndex,
                "only the front active block may settle (got \(blockIndex), "
                + "active \(other.blockIndex ?? -1))")
        }
        slots[slotIndex].state = .toCache
    }

    /// `active → dummy`: discard an in-flight block without committing it (EOS early-stop
    /// cancels the trailing slot). Never touches the prefix cache — a cancelled block was
    /// never captured.
    public mutating func cancel(slotIndex: Int) {
        precondition(slots[slotIndex].state == .active, "cancel requires an active slot")
        slots[slotIndex] = .empty
    }

    /// `toCache → inCache`: the block's committed KV has been appended to ``ExactPrefixCache``.
    /// Enforces front-block in-order commit for streaming correctness (phase-1 §7).
    public mutating func markCommitted(slotIndex: Int) {
        precondition(slots[slotIndex].state == .toCache, "markCommitted requires a toCache slot")
        let blockIndex = slots[slotIndex].blockIndex!
        if let last = lastCommittedBlock {
            precondition(blockIndex == last + 1, "commits must be in block order "
                + "(last \(last), got \(blockIndex))")
        }
        slots[slotIndex].state = .inCache
        lastCommittedBlock = blockIndex
    }
}
