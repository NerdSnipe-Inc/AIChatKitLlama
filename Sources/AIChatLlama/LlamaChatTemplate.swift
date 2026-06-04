import Foundation
import AIChatCore

/// Utilities for converting `ChatMessage` history into the format llama.cpp expects.
public struct LlamaChatTemplate {

    /// A role/content pair ready to pass to `llama_chat_apply_template`.
    public struct Entry {
        public let role: String
        public let content: String
    }

    // MARK: - Message conversion

    /// Convert `[ChatMessage]` into `[Entry]` for `llama_chat_apply_template`.
    /// - Thinking blocks are stripped (internal reasoning, never part of the chat surface).
    /// - Tool calls in assistant messages are serialised inline.
    /// - Tool result messages use role `"tool"`.
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

    /// Builds a Gemma 4 prompt manually when llama.cpp's Jinja parser can't handle
    /// the model's built-in template (which is typical for all Gemma variants).
    ///
    /// Format used by Gemma 4 (and Gemma 4 E2B IT):
    /// ```
    /// <start_of_turn>user
    /// content<end_of_turn>
    /// <start_of_turn>model
    /// content<end_of_turn>
    /// <start_of_turn>model
    /// ```
    /// BOS is not included here — `llama_tokenize(add_special: true)` prepends it.
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

    /// Returns `true` if `generated` contains any occurrence of `stop`.
    public static func containsStop(_ stop: String, in generated: String) -> Bool {
        guard !stop.isEmpty else { return false }
        return generated.contains(stop)
    }

    // MARK: - Context window management

    /// Trim history to keep the system message + the most recent `maxTurns` non-system messages.
    public static func truncate(_ messages: [ChatMessage], maxTurns: Int) -> [ChatMessage] {
        let systemMessages = messages.filter { $0.role == .system }
        let nonSystem      = messages.filter { $0.role != .system }

        guard nonSystem.count > maxTurns else { return messages }

        return systemMessages + Array(nonSystem.suffix(maxTurns))
    }
}
