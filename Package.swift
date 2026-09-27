// swift-tools-version: 5.10

import Foundation
import PackageDescription

private let packageDirectory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()

private func siblingOrRemote(
    siblingRelativePath: String,
    url: String,
    range: Range<Version>
) -> Package.Dependency {
    let siblingManifest = packageDirectory
        .appendingPathComponent(siblingRelativePath)
        .standardized
        .appendingPathComponent("Package.swift")

    let forceRemote = ProcessInfo.processInfo.environment["SPI_PROCESSING"] != nil
        || ProcessInfo.processInfo.environment["FORCE_REMOTE_PACKAGES"] != nil

    if !forceRemote, FileManager.default.fileExists(atPath: siblingManifest.path) {
        return .package(path: siblingRelativePath)
    }
    return .package(url: url, range)
}

let package = Package(
    name: "AIChatKitLlama",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "AIChatLlama", targets: ["AIChatLlama"]),
    ],
    dependencies: [
        siblingOrRemote(
            siblingRelativePath: "../AIChatKit",
            url: "https://github.com/NerdSnipe-Inc/AIChatKit.git",
            // Only AIChatCore is used; it is source-compatible across 1.x and 2.x.
            range: "1.0.0"..<"3.0.0"
        ),
        // llama.swift versions are 2.<llama.cpp build>.<patch>. The sampler code below targets the
        // llama.cpp build of 2.9469.x: 2.10549.0 changed `llama_sampler_init_penalties` (extra `n_vocab`
        // argument), so a wider range fails to compile. Move to a newer build deliberately, with a live test.
        .package(url: "https://github.com/mattt/llama.swift", .upToNextMinor(from: "2.9469.0")),
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
