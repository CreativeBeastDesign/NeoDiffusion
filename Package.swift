// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "NeoDiffusion",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .library(name: "DiffusionKernels", targets: ["DiffusionKernels"]),
        .library(name: "DiffusionCore", targets: ["DiffusionCore"]),
        .library(name: "DiffusionModel", targets: ["DiffusionModel"]),
        .library(name: "DiffusionGeneration", targets: ["DiffusionGeneration"]),
    ],
    dependencies: [
        // .exact, not from:, so both hosts cannot re-resolve to different MLX versions —
        // cross-host step counts and kernel timings are only comparable within one version.
        .package(url: "https://github.com/ml-explore/mlx-swift.git", exact: "0.31.6"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.1.1"),
        .package(url: "https://github.com/apple/swift-numerics.git", from: "1.1.0"),
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.0.0")
    ],
    targets: [
        .target(
            name: "DiffusionKernels",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift")
            ],
            path: "Packages/DiffusionKernels/Sources"
        ),
        .target(
            name: "DiffusionCore",
            dependencies: [
                "DiffusionKernels",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Numerics", package: "swift-numerics")
            ],
            path: "Packages/DiffusionCore/Sources"
        ),
        .target(
            name: "DiffusionModel",
            dependencies: [
                "DiffusionCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers")
            ],
            path: "Packages/DiffusionModel/Sources"
        ),
        .target(
            name: "DiffusionGeneration",
            dependencies: [
                "DiffusionModel",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift")
            ],
            path: "Packages/DiffusionGeneration/Sources"
        ),
        .executableTarget(
            name: "diffusion-bench",
            dependencies: [
                "DiffusionCore",
                "DiffusionModel",
                "DiffusionGeneration",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift")
            ],
            path: "Tools/diffusion-bench/Sources"
        ),
        .executableTarget(
            name: "diffusion-server",
            dependencies: [
                "DiffusionGeneration",
                .product(name: "Hummingbird", package: "hummingbird")
            ],
            path: "Tools/diffusion-server/Sources"
        ),
        .testTarget(
            name: "DiffusionKernelsTests",
            dependencies: ["DiffusionKernels"],
            path: "Tests/DiffusionKernelsTests"
        ),
        .testTarget(
            name: "DiffusionCoreTests",
            dependencies: ["DiffusionCore"],
            path: "Tests/DiffusionCoreTests"
        ),
        .testTarget(
            name: "DiffusionGenerationTests",
            dependencies: ["DiffusionGeneration"],
            path: "Tests/DiffusionGenerationTests"
        )
    ]
)
