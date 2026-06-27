// swift-tools-version: 5.10

import Foundation
import PackageDescription

private let packageDirectory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()

private func siblingOrRemote(
    siblingRelativePath: String,
    url: String,
    from version: Version
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
    return .package(url: url, from: version)
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
            from: "1.0.0"
        ),
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
