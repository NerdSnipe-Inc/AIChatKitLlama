import Foundation
import LlamaSwift
import AIChatCore

/// A `ChatProvider` that runs inference directly in-process via `llama.cpp`.
///
/// `LlamaProvider` maintains a long-lived model and context so chat turns can reuse
/// KV cache state instead of re-evaluating the entire prompt every request.
/// It also handles template selection, sampling strategy, and tool-call extraction.
///
/// ## Key Behaviors
/// - KV-cache prefix reuse for efficient multi-turn conversations.
/// - Sampler-chain configuration for top-k, min-p, top-p, temperature, and penalties.
/// - Tool definition injection into the system prompt.
/// - `<tool_call>...</tool_call>` detection and emission as `.toolCallComplete` events.
/// - Gemma 4 prompt fallback when built-in template parsing fails.
///
/// ## Threading Model
/// This type is an `actor`, so mutable model/context state is serialized by actor
/// isolation. `stream(messages:model:options:)` is marked `nonisolated` and forwards
/// work back to actor-isolated methods.
///
/// ## Example
/// ```swift
/// let provider = LlamaProvider(
///     modelPath: "/path/to/model.gguf",
///     contextSize: 8192,
///     nBatch: 512,
///     nGpuLayers: 99,
///     maxTurns: 20
/// )
/// ```
///
/// - Important: Ensure `modelPath` points to a valid, local `.gguf` file.
public actor LlamaProvider: ChatProvider {

    /// Stable provider identifier used by `AIChatCore` routing.
    ///
    /// Use this value when you need to recognize responses emitted by this provider.
    public nonisolated let id   = "llama"
    /// Human-readable provider display name.
    ///
    /// This value is intended for logs and UI surfaces.
    public nonisolated let name = "llama.cpp"

    // MARK: - Configuration

    private let modelPath:   String
    private let contextSize: UInt32
    private let nBatch:      UInt32
    private let nGpuLayers:  Int32
    private let maxTurns:    Int

    // MARK: - Persistent resources (context must be freed before model)

    private var loadedModel: OpaquePointer?
    private var liveContext: LlamaContext?

    // MARK: - Init

    /// Creates a provider instance backed by a local llama.cpp-compatible model.
    ///
    /// - Parameters:
    ///   - modelPath: Filesystem path to a `.gguf` model file.
    ///   - contextSize: Target context window (`n_ctx`) for the runtime context.
    ///   - nBatch: Decode batch size (`n_batch`) used during prompt evaluation.
    ///   - nGpuLayers: Number of transformer layers to offload to GPU.
    ///     Use `-1` for CPU-only mode; high positive values request broad offload.
    ///   - maxTurns: Maximum number of non-system messages retained when truncating
    ///     history before prompt construction.
    ///
    /// - Note: On simulator builds, GPU offload is disabled internally.
    public init(
        modelPath:   String,
        contextSize: UInt32 = 8192,
        nBatch:      UInt32 = 512,
        nGpuLayers:  Int32  = -1,
        maxTurns:    Int    = 20
    ) {
        self.modelPath   = modelPath
        self.contextSize = contextSize
        self.nBatch      = nBatch
        self.nGpuLayers  = nGpuLayers
        self.maxTurns    = maxTurns
        llama_backend_init()
    }

    deinit {
        liveContext  = nil
        if let m = loadedModel { llama_model_free(m) }
        llama_backend_free()
    }

    // MARK: - ChatProvider

    /// Streams assistant output events for a chat request.
    ///
    /// The stream may emit `.text` chunks and `.toolCallComplete` events as model
    /// output is decoded. The stream finishes when generation ends or throws.
    ///
    /// - Parameters:
    ///   - messages: Ordered conversation history to include in prompt construction.
    ///   - model: Logical model hint provided by `AIChatCore`.
    ///     This provider currently uses its configured local model and ignores
    ///     this argument for runtime model switching.
    ///   - options: Sampling, system prompt, tool definitions, and stop conditions.
    /// - Returns: An `AsyncThrowingStream` of `ChatStreamEvent` values.
    public nonisolated func stream(
        messages: [ChatMessage],
        model: String,
        options: ChatRequestOptions
    ) -> AsyncThrowingStream<ChatStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            Task {
                do {
                    try await self.generate(messages: messages, options: options, into: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// Produces a single non-streaming completion by collecting streamed text output.
    ///
    /// This helper consumes `stream(messages:model:options:)`, concatenates `.text`
    /// events, and returns a final `ChatCompletionResult`.
    ///
    /// - Parameters:
    ///   - messages: Ordered conversation history to include in prompt construction.
    ///   - model: Logical model hint provided by the caller.
    ///   - options: Sampling, system prompt, and tool configuration for generation.
    /// - Returns: A completion result containing the assembled assistant message.
    /// - Throws: Any error propagated by streaming generation.
    public func complete(
        messages: [ChatMessage],
        model: String,
        options: ChatRequestOptions
    ) async throws -> ChatCompletionResult {
        var text = ""
        for try await event in stream(messages: messages, model: model, options: options) {
            if case .text(let t) = event { text += t }
        }
        return ChatCompletionResult(
            id: nil, model: id,
            message: ChatMessage(role: .assistant, content: text),
            usage: nil, finishReason: .stop
        )
    }

    // MARK: - Core inference

    private func generate(
        messages: [ChatMessage],
        options:  ChatRequestOptions,
        into continuation: AsyncThrowingStream<ChatStreamEvent, Error>.Continuation
    ) async throws {
        let model = try ensureModel()

        // ── Build prompt entries ────────────────────────────────────────────────
        var entries = LlamaChatTemplate.convert(
            LlamaChatTemplate.truncate(messages, maxTurns: maxTurns)
        )
        if let sys = options.systemPrompt {
            entries.insert(LlamaChatTemplate.Entry(role: "system", content: sys), at: 0)
        }

        // Inject tool schemas into the system message if tools are provided.
        if let tools = options.tools, !tools.isEmpty {
            let toolSection = buildToolsSection(tools)
            if let idx = entries.firstIndex(where: { $0.role == "system" }) {
                let existing = entries[idx]
                entries[idx] = LlamaChatTemplate.Entry(
                    role: "system",
                    content: existing.content.isEmpty
                        ? toolSection
                        : existing.content + "\n\n" + toolSection
                )
            } else {
                entries.insert(LlamaChatTemplate.Entry(role: "system", content: toolSection), at: 0)
            }
        }

        let prompt       = try buildPrompt(model: model, entries: entries)
        let promptTokens = try tokenize(prompt: prompt, model: model)

        // ── Ensure the context can fit this prompt + at least one generation ───
        let ctx        = try ensureContext(model: model)
        let minReserve = Int32(options.maxTokens ?? 512)

        if ctx.remainingCapacity < minReserve {
            ctx.reset()
        }
        guard Int32(promptTokens.count) < ctx.maxContextSize else {
            throw ChatError.streamError(
                "Prompt (\(promptTokens.count) tokens) exceeds context size (\(ctx.maxContextSize))."
            )
        }

        // ── Evaluate prompt with KV cache prefix reuse ──────────────────────────
        try ctx.prepareForTokens(promptTokens)

        // ── Sampler ─────────────────────────────────────────────────────────────
        let sampler = LlamaSampler(
            temperature:    Float(options.temperature ?? 0.7),
            topK:           Int32(options.topK ?? 40),
            topP:           Float(options.topP ?? 0.95),
            minP:           Float(options.minP ?? 0.05),
            penaltyRepeat:  Float(options.penaltyRepeat ?? 1.1),
            penaltyFreq:    Float(options.penaltyFreq ?? 0.0),
            penaltyPresent: Float(options.penaltyPresent ?? 0.0)
        )

        // ── EOT markers ─────────────────────────────────────────────────────────
        let eotMarkers: [String] = [
            "<end_of_turn>",    // Gemma 4
            "<|im_end|>",       // ChatML
            "<|eot_id|>",       // Llama 3
            "</s>",             // Llama 2 / Mistral
            "<|endoftext|>",    // GPT-style
        ]
        // Hold back enough characters to catch the longest EOT or tool-call marker.
        let maxMarkerLen = max(
            eotMarkers.map(\.count).max() ?? 0,
            "<tool_call>".count
        )

        // ── Generation loop ─────────────────────────────────────────────────────
        let maxNew    = options.maxTokens ?? 2048
        let stops     = options.stop ?? []
        let vocab     = llama_model_get_vocab(model)
        var generated = ""       // full output so far — used for EOT / tool-trigger detection
        var pending   = ""       // lookahead buffer — prevents partial-marker leaks to UI
        var toolBuf   = ""       // accumulates JSON inside <tool_call>…</tool_call>
        var inToolCall = false   // whether we are inside a tool call span

        for _ in 0..<maxNew {
            try Task.checkCancellation()

            let next = sampler.sample(ctx.ctx)
            if llama_vocab_is_eog(vocab, next) {
                sampler.accept(next)
                break
            }

            var buf = [CChar](repeating: 0, count: 64)
            let len = llama_token_to_piece(vocab, next, &buf, Int32(buf.count), 0, false)

            sampler.accept(next)
            try ctx.decodeSingleToken(next)

            guard len > 0 else { continue }

            let piece = String(cString: buf)

            // ── Tool call capture mode ──────────────────────────────────────────
            if inToolCall {
                toolBuf += piece
                if let closeRange = toolBuf.range(of: "</tool_call>") {
                    let json = String(toolBuf[..<closeRange.lowerBound])
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    emitToolCall(json, into: continuation)
                    // Reset — the model may emit another tool call or an EOT.
                    toolBuf    = ""
                    inToolCall = false
                    generated  = ""
                }
                continue
            }

            // ── Normal text streaming ───────────────────────────────────────────
            generated += piece
            pending   += piece

            // Tool call trigger: model wants to invoke a tool.
            if generated.hasSuffix("<tool_call>") {
                let tagLen  = "<tool_call>".count
                let safeLen = max(0, pending.count - tagLen)
                if safeLen > 0 {
                    continuation.yield(.text(String(pending.prefix(safeLen))))
                }
                pending    = ""
                inToolCall = true
                toolBuf    = ""
                continue
            }

            // EOT marker — model finished its turn.
            if let marker = eotMarkers.first(where: { generated.hasSuffix($0) }) {
                let safeLen = max(0, pending.count - marker.count)
                if safeLen > 0 {
                    continuation.yield(.text(String(pending.prefix(safeLen))))
                }
                pending = ""
                break
            }

            // Stop sequences.
            if stops.contains(where: { LlamaChatTemplate.containsStop($0, in: generated) }) {
                if !pending.isEmpty { continuation.yield(.text(pending)) }
                pending = ""
                break
            }

            // Flush safe prefix of the lookahead buffer.
            if pending.count > maxMarkerLen {
                let safeCount = pending.count - maxMarkerLen
                continuation.yield(.text(String(pending.prefix(safeCount))))
                pending = String(pending.suffix(maxMarkerLen))
            }
        }

        // Final flush — strip any trailing EOT that arrived immediately before EOS.
        if !pending.isEmpty {
            var toFlush = pending
            for marker in eotMarkers where toFlush.hasSuffix(marker) {
                toFlush = String(toFlush.dropLast(marker.count))
                break
            }
            if !toFlush.isEmpty { continuation.yield(.text(toFlush)) }
        }
    }

    // MARK: - Tool call helpers

    /// Serialise tool definitions into a system-message section the model can follow.
    private func buildToolsSection(_ tools: [ChatRequestOptions.ToolDefinition]) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(tools),
              let json = String(data: data, encoding: .utf8) else { return "" }

        return """
        ## Available Tools

        When you want to call a tool, output ONLY this on its own line — no prose before or after on the same turn:
        <tool_call>{"name": "tool_name", "arguments": {"key": "value"}}</tool_call>

        You may call multiple tools, one per line, before providing a final answer.
        Do not make up tool names — only use tools listed below.

        \(json)
        """
    }

    /// Parse a raw JSON string extracted from between `<tool_call>` tags and emit
    /// a `.toolCallComplete` event. Accepts both `"arguments"` and `"args"` keys
    /// since different models use different field names.
    private func emitToolCall(
        _ json: String,
        into continuation: AsyncThrowingStream<ChatStreamEvent, Error>.Continuation
    ) {
        guard
            let data = json.data(using: .utf8),
            let obj  = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let name = obj["name"] as? String
        else { return }

        let argsObj  = obj["arguments"] ?? obj["args"] ?? [String: Any]()
        let argsData = (try? JSONSerialization.data(withJSONObject: argsObj)) ?? Data("{}".utf8)
        let argsJSON = String(data: argsData, encoding: .utf8) ?? "{}"

        continuation.yield(.toolCallComplete(id: UUID().uuidString, name: name, arguments: argsJSON))
    }

    // MARK: - Tokenisation

    private func tokenize(prompt: String, model: OpaquePointer) throws -> [llama_token] {
        let bytes    = Array(prompt.utf8)
        let maxCount = Int32(bytes.count + 1)
        var tokens   = [llama_token](repeating: 0, count: Int(maxCount))
        let count    = llama_tokenize(
            llama_model_get_vocab(model),
            prompt, Int32(bytes.count),
            &tokens, maxCount,
            true, true
        )
        guard count > 0 else {
            throw ChatError.streamError("Tokenisation produced no tokens — is the model valid?")
        }
        return Array(tokens.prefix(Int(count)))
    }

    // MARK: - Chat template

    /// Build the formatted prompt string from message entries.
    ///
    /// Strategy:
    /// 1. Model's own built-in Jinja template via `llama_chat_apply_template`.
    /// 2. Gemma 4 manual fallback — llama.cpp's Jinja parser rejects Gemma templates.
    /// 3. ChatML as a last resort.
    private func buildPrompt(
        model:   OpaquePointer,
        entries: [LlamaChatTemplate.Entry]
    ) throws -> String {
        // NSString instances must stay alive for the duration of llama_chat_apply_template
        // so their C string pointers remain valid.
        let nsRoles    = entries.map { $0.role    as NSString }
        let nsContents = entries.map { $0.content as NSString }

        var cMsgs = zip(nsRoles, nsContents).map { r, c in
            llama_chat_message(role: r.utf8String, content: c.utf8String)
        }

        // 1. Model's built-in template.
        let modelTmpl = llama_model_chat_template(model, nil)
        let needed = withExtendedLifetime((nsRoles, nsContents)) {
            llama_chat_apply_template(modelTmpl, &cMsgs, cMsgs.count, true, nil, 0)
        }
        if needed > 0 {
            var buf = [CChar](repeating: 0, count: Int(needed) + 1)
            withExtendedLifetime((nsRoles, nsContents)) {
                _ = llama_chat_apply_template(modelTmpl, &cMsgs, cMsgs.count, true, &buf, needed + 1)
            }
            return String(cString: buf)
        }

        // 2. Gemma 4 fallback.
        var tmplBuf = [CChar](repeating: 0, count: 16384)
        let tmplLen = llama_model_meta_val_str(model, "tokenizer.chat_template", &tmplBuf, 16384)
        if tmplLen > 0, String(cString: tmplBuf).contains("start_of_turn") {
            return LlamaChatTemplate.buildGemma4Prompt(entries: entries)
        }

        // 3. ChatML default.
        let neededDefault = withExtendedLifetime((nsRoles, nsContents)) {
            llama_chat_apply_template(nil, &cMsgs, cMsgs.count, true, nil, 0)
        }
        guard neededDefault > 0 else {
            throw ChatError.streamError("llama_chat_apply_template failed — no usable template found")
        }
        var buf = [CChar](repeating: 0, count: Int(neededDefault) + 1)
        withExtendedLifetime((nsRoles, nsContents)) {
            _ = llama_chat_apply_template(nil, &cMsgs, cMsgs.count, true, &buf, neededDefault + 1)
        }
        return String(cString: buf)
    }

    // MARK: - Resource lifecycle

    private func ensureModel() throws -> OpaquePointer {
        if let m = loadedModel { return m }
        var p          = llama_model_default_params()
        #if targetEnvironment(simulator)
        p.n_gpu_layers = 0
        #else
        p.n_gpu_layers = nGpuLayers
        #endif
        guard let m = llama_model_load_from_file(modelPath, p) else {
            throw ChatError.invalidConfiguration(
                "Failed to load model at '\(modelPath)'. " +
                "Verify the path exists and points to a valid .gguf file."
            )
        }
        loadedModel = m
        return m
    }

    private func ensureContext(model: OpaquePointer) throws -> LlamaContext {
        if let c = liveContext { return c }
        let c = try LlamaContext(model: model, contextSize: contextSize, nBatch: nBatch)
        liveContext = c
        return c
    }
}
