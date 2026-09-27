# Changelog

All notable changes to AIChatKitLlama are documented in this file. Each version has a matching
[GitHub release](https://github.com/NerdSnipe-Inc/AIChatKitLlama/releases) with the same notes.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.3] - 2026-09-27

### Changed
- Accepts `AIChatKit` 1.0.0 up to (not including) 3.0.0, so it can be used with AIChatKit 2.x. Only
  `AIChatCore` is used, and it is unchanged in 2.0.0. Verified against AIChatKit 2.0.0 (14 tests).

### Fixed
- **Does not build against llama.swift 2.10549.0 or later.** The manifest allowed any 2.x from 2.9469.0, so
  a fresh resolve picked 2.10549.0, whose `llama_sampler_init_penalties` takes an extra `n_vocab` argument,
  and `LlamaSampler.swift` failed to compile. The dependency is now `.upToNextMinor(from: "2.9469.0")`
  (llama.swift versions are `2.<llama.cpp build>.<patch>`, so this stays on the build the sampler code
  was written for). Moving to a newer llama.cpp build needs a code change and a live test.

### Upgrading from 1.0.2
- No code changes. If your app or another package pins `llama.swift` to a newer 2.x than 2.9469.x, resolution
  now fails; pin it to 2.9469.x, which is what this package builds against.

## [1.0.2] - 2026-08-07

### Fixed
- SPI manifest `authors` field uses the valid string format.
- Duplicate AIChatKit identity in monorepo builds.

## [1.0.1] - 2026-06-27

### Changed
- DocC documentation expansion for the public `AIChatLlama` APIs; SPI manifest optimised for docs publishing.

## [1.0.0] - 2026-06-27

First stable release.

- **AIChatLlama**: in-process GGUF inference via llama.cpp (Metal GPU).
- Integrates with the [AIChatKit](https://github.com/NerdSnipe-Inc/AIChatKit) `ChatProvider` protocol.
- Model download, progress reporting, and generation options (`topK`, `minP`, repeat penalty).
- MIT license and Swift Package Index manifest (`.spi.yml`); SPI compatibility badges in README.

## 0.1.0

Pre-release (2026-06-04).

[Unreleased]: https://github.com/NerdSnipe-Inc/AIChatKitLlama/compare/1.0.3...HEAD
[1.0.3]: https://github.com/NerdSnipe-Inc/AIChatKitLlama/compare/1.0.2...1.0.3
[1.0.2]: https://github.com/NerdSnipe-Inc/AIChatKitLlama/compare/1.0.1...1.0.2
[1.0.1]: https://github.com/NerdSnipe-Inc/AIChatKitLlama/compare/1.0.0...1.0.1
[1.0.0]: https://github.com/NerdSnipe-Inc/AIChatKitLlama/releases/tag/1.0.0
