import XCTest
import MLX
@testable import DiffusionCore

final class DiffusionCoreTests: XCTestCase {
    func testAttentionRoutingBlock() {
        let core = DiffusionCore()
        
        let q = MLX.ones([1, 8, 64])
        let k = MLX.ones([1, 8, 64])
        let v = MLX.ones([1, 8, 64]) * 2.0
        
        let output = core.attentionBlock(query: q, key: k, value: v)
        
        XCTAssertEqual(output.shape, [1, 8, 64])
    }
}
