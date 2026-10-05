# AIChatKitLlama

![Swift 5.10](https://img.shields.io/badge/Swift-5.10-orange?logo=swift)
![iOS 17+](https://img.shields.io/badge/iOS-17%2B-blue?logo=apple)
![macOS 14+](https://img.shields.io/badge/macOS-14%2B-blue?logo=apple)
![MIT License](https://img.shields.io/badge/license-MIT-green)
![SPM](https://img.shields.io/badge/SPM-compatible-brightgreen)
[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FNerdSnipe-Inc%2FAIChatKitLlama%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/NerdSnipe-Inc/AIChatKitLlama)
[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FNerdSnipe-Inc%2FAIChatKitLlama%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/NerdSnipe-Inc/AIChatKitLlama)

Adds on-device GGUF inference via [llama.cpp](https://github.com/ggerganov/llama.cpp) to any app already using [AIChatKit](https://github.com/NerdSnipe-Inc/AIChatKit). Models run entirely in-process using Metal GPU acceleration — no network calls after the initial download.

**Platforms:** macOS 14+ · iOS 17+  
**Language:** Swift 5.10+  
**Binary size:** ~500 MB (llama.cpp XCFramework via [llama.swift](https://github.com/mattt/llama.swift))

> **Requires AIChatKit.** Add both packages to your target.

Prefer Apple MLX models? [AIChatKitMLX](https://github.com/NerdSnipe-Inc/AIChatKitMLX) is the sibling package: text and vision models on Metal and the Neural Engine, behind the same AIChatKit protocol.

---

[![Sponsor NerdSnipe-Inc](https://img.shields.io/badge/Sponsor-NerdSnipe--Inc-ea4aaa?logo=githubsponsors&logoColor=white)](https://github.com/sponsors/NerdSnipe-Inc)

> AIChatKitLlama is free and open source. If it saved you time, [sponsoring NerdSnipe Inc](https://github.com/sponsors/NerdSnipe-Inc) pays for the maintenance, bug fixes and new releases that keep it working.

## Installation

```swift
// Package.swift
.package(url: "https://github.com/NerdSnipe-Inc/AIChatKit",      from: "1.0.0"),
.package(url: "https://github.com/NerdSnipe-Inc/AIChatKitLlama", from: "1.0.3"),

// Target dependencies
.product(name: "AIChatCore",  package: "AIChatKit"),
.product(name: "AIChatUI",    package: "AIChatKit"),    // if using ChatSession / ChatView
.product(name: "AIChatLlama", package: "AIChatKitLlama"),
```

> **Note:** `AIChatLlama` pulls a ~500 MB binary XCFramework. Add it only to targets that actually need local inference. Do not commit the resolved XCFramework to git — add `AIChatKitLlama` to your `.gitignore`.

> **`llama.swift` is pinned to 2.9469.x** (`.upToNextMinor`), not any 2.x. `llama.swift` versions are
> `2.<llama.cpp build>.<patch>`; `LlamaSampler.swift`'s calls into `llama_sampler_init_penalties`
> target the 2.9469.x build's signature. 2.10549.0 changed that signature (added an `n_vocab`
> argument) and fails to compile against this package. Moving to a newer build needs a code change
> and a live test — see the 1.0.3 entry in [CHANGELOG.md](CHANGELOG.md). Accepts `AIChatKit` 1.x or
> 2.x as of 1.0.3 (only `AIChatCore` is used).

---

## Quick start

```swift
import AIChatLlama
import AIChatUI

let provider = LlamaProvider(
    modelPath: "/path/to/model.gguf",
    contextSize: 4096,
    nGpuLayers: 99   // 99 = all layers on Metal GPU; -1 = CPU only
)

@StateObject private var session = ChatSession(
    provider: provider,
    model: "local",  // LlamaProvider ignores the model string; pass anything
    options: ChatRequestOptions(
        maxTokens: 512,
        temperature: 0.7,
        systemPrompt: "You are a helpful assistant."
    )
)
```

`LlamaProvider` is an **actor**. The model loads from disk on the first `stream()` or `complete()` call and stays resident in memory.

---

## Supported models

Any GGUF model compatible with llama.cpp. Tested with:

- **Gemma 4** (`bartowski/google_gemma-4-E2B-it-GGUF`) — Gemma 4 chat template applied automatically
- **Llama 3.x** — standard chat template
- **Mistral / Mixtral** — standard chat template
- **Phi-3 / Phi-4** — standard chat template

Download models from [Hugging Face](https://huggingface.co/models?library=gguf). Q4_K_M quantisation is a good balance of quality and size for most use cases.

---

## Options

```swift
LlamaProvider(
    modelPath:   "/path/to/model.gguf",
    contextSize: 8192,   // KV cache size in tokens (default 8192)
    nBatch:      512,    // prompt-evaluation batch size (default 512)
    nGpuLayers:  99,     // 99 = all on GPU, 0 = CPU only, -1 = CPU only (default -1)
    maxTurns:    20      // older turns truncated beyond this
)
```

Sampling parameters (set via `ChatRequestOptions`):

```swift
ChatRequestOptions(
    maxTokens:      512,
    temperature:    0.7,
    topP:           0.95,
    topK:           40,
    minP:           0.05,
    penaltyRepeat:  1.1
)
```

Cloud providers silently ignore `topK`, `minP`, and `penaltyRepeat` — safe to use the same options struct across providers.

---

## License

MIT

## Support this project

AIChatKitLlama is built and maintained by [NerdSnipe Inc](https://nerdsnipe.cc), a small independent studio in Ottawa. Sponsorship funds keeping up with llama.cpp releases.

- [Sponsor on GitHub](https://github.com/sponsors/NerdSnipe-Inc), from $5/month or a one-time amount
- [More about what we fund](https://nerdsnipe.cc/sponsor)
