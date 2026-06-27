import Foundation
import AIChatCore

/// Utilities for mapping `ChatMessage` history into llama.cpp chat-template inputs.
///
/// This namespace contains conversion, truncation, stop-detection, and Gemma-specific
/// prompt building helpers used by `LlamaProvider`.
public struct LlamaChatTemplate {

    /// A role/content pair ready for `llama_chat_apply_template`.
    ///
    /// Instances of `Entry` are intentionally lightweight and represent the
    /// minimal payload required by llama.cpp chat template rendering.
    public struct Entry {
        /// Chat role string expected by the selected template (for example, `user`).
        public let role: String
        /// Message content already normalized for template application.
        public let content: String
    }

    // MARK: - Message conversion

    /// Converts high-level chat history into template-ready entries.
    ///
    /// Conversion rules:
    /// - Thinking blocks are dropped and never included in prompt text.
    /// - Image blocks are converted to a placeholder marker (`[image]`).
    /// - Tool calls are serialized as `<tool_call>{...}</tool_call>` snippets.
    /// - Tool message content is mapped to role `"tool"`.
    ///
    /// - Parameter messages: Source conversation history in chronological order.
    /// - Returns: Template entries suitable for prompt rendering.
    public static func convert(_ messages: [ChatMessage]) -> [Entry] {
        messages.compactMap { msg in
            let role: String
            switch msg.role {
            case .system:    role = "system"
            case .user:      role = "user"
            case .assistant: role = "assistant"
            case .tool:      role = "tool"
            }

            var parts: [String] = []

            for block in msg.content {
                switch block {
                case .text(let t):
                    parts.append(t)
                case .thinking, .redactedThinking:
                    break
                case .image:
                    parts.append("[image]")
                case .toolCall(let tc):
                    parts.append("<tool_call>{\"name\":\"\(tc.name)\",\"arguments\":\(tc.arguments)}</tool_call>")
                case .toolResult(let tr):
                    parts.append(tr.content)
                }
            }

            for tc in msg.toolCalls ?? [] {
                parts.append("<tool_call>{\"name\":\"\(tc.name)\",\"arguments\":\(tc.arguments)}</tool_call>")
            }

            if msg.role == .tool, parts.isEmpty,
               let first = msg.content.first, case .text(let t) = first {
                parts.append(t)
            }

            let content = parts.joined()
            guard !content.isEmpty || msg.role == .system else { return nil }
            return Entry(role: role, content: content)
        }
    }

    // MARK: - Gemma 4 manual template

    /// Builds a Gemma 4 prompt manually when built-in template rendering is unavailable.
    ///
    /// llama.cpp's Jinja parser may reject some Gemma templates. This helper emits
    /// the explicit `<start_of_turn>...<end_of_turn>` format expected by Gemma 4.
    ///
    /// Format used by Gemma 4 (and Gemma 4 E2B IT):
    /// ```
    /// <start_of_turn>user
    /// content<end_of_turn>
    /// <start_of_turn>model
    /// content<end_of_turn>
    /// <start_of_turn>model
    /// ```
    /// BOS is intentionally not included because tokenization can prepend specials.
    ///
    /// - Parameter entries: Template entries in conversation order.
    /// - Returns: A prompt string ending with an assistant/model turn prefix.
    public static func buildGemma4Prompt(entries: [Entry]) -> String {
        var prompt = ""
        for entry in entries {
            let role = entry.role == "assistant" ? "model" : entry.role
            prompt += "<start_of_turn>\(role)\n\(entry.content)<end_of_turn>\n"
        }
        prompt += "<start_of_turn>model\n"
        return prompt
    }

    // MARK: - Stop sequence detection

    /// Returns whether generated text contains a stop sequence.
    ///
    /// - Parameters:
    ///   - stop: Stop string to detect.
    ///   - generated: Generated output buffer to scan.
    /// - Returns: `true` if `generated` includes `stop`; otherwise `false`.
    public static func containsStop(_ stop: String, in generated: String) -> Bool {
        guard !stop.isEmpty else { return false }
        return generated.contains(stop)
    }

    // MARK: - Context window management

    /// Trims history to system messages plus the latest non-system turns.
    ///
    /// - Parameters:
    ///   - messages: Full chat history in chronological order.
    ///   - maxTurns: Maximum number of non-system messages to retain.
    /// - Returns: A reduced message list preserving all system messages and the
    ///   newest `maxTurns` non-system messages.
    public static func truncate(_ messages: [ChatMessage], maxTurns: Int) -> [ChatMessage] {
        let systemMessages = messages.filter { $0.role == .system }
        let nonSystem      = messages.filter { $0.role != .system }

        guard nonSystem.count > maxTurns else { return messages }

        return systemMessages + Array(nonSystem.suffix(maxTurns))
    }
}
