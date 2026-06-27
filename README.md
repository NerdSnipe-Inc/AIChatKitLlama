# AIChatKitLlama

[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FNerdSnipe-Inc%2FAIChatKitLlama%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/NerdSnipe-Inc/AIChatKitLlama)
[![](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2FNerdSnipe-Inc%2FAIChatKitLlama%2Fbadge%3Ftype%3Dplatforms)](https://swiftpackageindex.com/NerdSnipe-Inc/AIChatKitLlama)

Adds on-device GGUF inference via [llama.cpp](https://github.com/ggerganov/llama.cpp) to any app already using [AIChatKit](https://github.com/NerdSnipe-Inc/AIChatKit). Models run entirely in-process using Metal GPU acceleration — no network calls after the initial download.

**Platforms:** macOS 14+ · iOS 17+  
**Language:** Swift 5.10+  
**Binary size:** ~500 MB (llama.cpp XCFramework via [llama.swift](https://github.com/mattt/llama.swift))

> **Requires AIChatKit.** Add both packages to your target.

---

## Installation

```swift
// Package.swift
.package(url: "https://github.com/NerdSnipe-Inc/AIChatKit",      from: "0.1.0"),
.package(url: "https://github.com/NerdSnipe-Inc/AIChatKitLlama", from: "0.1.0"),

// Target dependencies
.product(name: "AIChatCore",  package: "AIChatKit"),
.product(name: "AIChatUI",    package: "AIChatKit"),    // if using ChatSession / ChatView
.product(name: "AIChatLlama", package: "AIChatKitLlama"),
```

> **Note:** `AIChatLlama` pulls a ~500 MB binary XCFramework. Add it only to targets that actually need local inference. Do not commit the resolved XCFramework to git — add `AIChatKitLlama` to your `.gitignore`.

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
    contextSize: 8192,   // KV cache size in tokens
    nGpuLayers:  99,     // 99 = all on GPU, 0 = CPU only, -1 = CPU only
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
