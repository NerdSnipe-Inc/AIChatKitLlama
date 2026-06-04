// swift-tools-version: 5.10

import PackageDescription

// AIChatKitLlama adds on-device GGUF inference to any app that already uses AIChatKit.
// The LlamaProvider (~500 MB binary XCFramework via llama.swift) is isolated here so
// apps that only need cloud or MLX providers don't pull the large binary.
//
// Usage:
//   .package(url: "https://github.com/NerdSnipe-Inc/AIChatKit",      from: "0.1.0"),
//   .package(url: "https://github.com/NerdSnipe-Inc/AIChatKitLlama", from: "0.1.0"),

let package = Package(
    name: "AIChatKitLlama",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "AIChatLlama", targets: ["AIChatLlama"]),
    ],
    dependencies: [
        .package(url: "https://github.com/NerdSnipe-Inc/AIChatKit.git", from: "0.1.0"),
        .package(url: "https://github.com/mattt/llama.swift", .upToNextMajor(from: "2.9469.0")),
    ],
    targets: [
        .target(
            name: "AIChatLlama",
            dependencies: [
                .product(name: "AIChatCore", package: "AIChatKit"),
                .product(name: "LlamaSwift", package: "llama.swift"),
            ],
            path: "Sources/AIChatLlama"
        ),
        .testTarget(name: "AIChatLlamaTests", dependencies: ["AIChatLlama"], path: "Tests/AIChatLlamaTests"),
    ]
)
