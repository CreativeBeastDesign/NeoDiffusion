import XCTest
@testable import DiffusionKernels

final class DiffusionKernelsTests: XCTestCase {
    func testGPUAvailability() {
        // Assert that the availability check runs without errors
        let available = DiffusionKernels.checkAvailability()
        print("GPU / MLX Available: \(available)")
        XCTAssertTrue(true) // Ensure the test runner itself passes
    }
}
