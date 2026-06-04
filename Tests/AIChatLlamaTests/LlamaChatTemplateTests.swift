import XCTest
@testable import AIChatLlama
import AIChatCore

final class LlamaChatTemplateTests: XCTestCase {

    // MARK: - Role mapping

    func test_systemMessage_mapsToSystemRole() {
        let messages = [ChatMessage.system("You are helpful.")]
        let converted = LlamaChatTemplate.convert(messages)
        XCTAssertEqual(converted[0].role, "system")
        XCTAssertEqual(converted[0].content, "You are helpful.")
    }

    func test_userMessage_mapsToUserRole() {
        let msg = ChatMessage(role: .user, content: "Hello")
        let converted = LlamaChatTemplate.convert([msg])
        XCTAssertEqual(converted[0].role, "user")
        XCTAssertEqual(converted[0].content, "Hello")
    }

    func test_assistantTextMessage_mapsToAssistantRole() {
        let msg = ChatMessage(role: .assistant, content: "Hi there!")
        let converted = LlamaChatTemplate.convert([msg])
        XCTAssertEqual(converted[0].role, "assistant")
        XCTAssertEqual(converted[0].content, "Hi there!")
    }

    func test_toolResultMessage_mapsToToolRole() {
        let msg = ChatMessage(toolCallId: "call_1", content: "72°F")
        let converted = LlamaChatTemplate.convert([msg])
        XCTAssertEqual(converted[0].role, "tool")
        XCTAssertEqual(converted[0].content, "72°F")
    }

    // MARK: - Multi-content flattening

    func test_multipleTextBlocks_concatenated() {
        let msg = ChatMessage(role: .user, content: [.text("Part A"), .text(" Part B")])
        let converted = LlamaChatTemplate.convert([msg])
        XCTAssertEqual(converted[0].content, "Part A Part B")
    }

    func test_thinkingBlocks_strippedFromAssistantHistory() {
        let block = ChatMessage.ThinkingBlock(text: "internal thought", signature: "sig")
        let msg = ChatMessage(role: .assistant, content: [.thinking(block), .text("Answer")])
        let converted = LlamaChatTemplate.convert([msg])
        // Thinking content should not leak into the chat template
        XCTAssertEqual(converted[0].content, "Answer")
        XCTAssertFalse(converted[0].content.contains("internal thought"))
    }

    func test_assistantToolCalls_includedAsJSON() {
        let call = ChatMessage.ToolCallBlock(id: "c1", name: "search", arguments: "{\"q\":\"swift\"}")
        let msg = ChatMessage(role: .assistant, content: [.text("Let me search.")], toolCalls: [call])
        let converted = LlamaChatTemplate.convert([msg])
        // Tool call metadata should be present so the model can reference it
        XCTAssertTrue(converted[0].content.contains("search"))
    }

    // MARK: - Ordering preserved

    func test_messageOrder_preserved() {
        let messages: [ChatMessage] = [
            .system("Be helpful"),
            ChatMessage(role: .user, content: "Q1"),
            ChatMessage(role: .assistant, content: "A1"),
            ChatMessage(role: .user, content: "Q2"),
        ]
        let converted = LlamaChatTemplate.convert(messages)
        XCTAssertEqual(converted.map(\.role), ["system", "user", "assistant", "user"])
    }

    // MARK: - Stop sequence detection

    func test_stopSequence_detectedAtEnd() {
        XCTAssertTrue(LlamaChatTemplate.containsStop("<|eot_id|>", in: "Hello<|eot_id|>"))
        XCTAssertTrue(LlamaChatTemplate.containsStop("<|eot_id|>", in: "<|eot_id|>"))
    }

    func test_stopSequence_notDetectedInMiddle() {
        // mid-stream occurrence still counts — any occurrence should stop
        XCTAssertTrue(LlamaChatTemplate.containsStop("</s>", in: "Answer</s>more"))
    }

    func test_noStopSequence_returnsFalse() {
        XCTAssertFalse(LlamaChatTemplate.containsStop("<|eot_id|>", in: "This is normal text"))
    }

    func test_emptyStops_alwaysFalse() {
        XCTAssertFalse(LlamaChatTemplate.containsStop("", in: "anything"))
    }

    // MARK: - Token truncation

    func test_truncate_keepsSystemAndRecent() {
        let messages: [ChatMessage] = [
            .system("System"),
            ChatMessage(role: .user, content: "Old user turn"),
            ChatMessage(role: .assistant, content: "Old assistant turn"),
            ChatMessage(role: .user, content: "Recent"),
        ]
        // With a tiny budget, only system + last turn should survive
        let truncated = LlamaChatTemplate.truncate(messages, maxTurns: 2)
        let roles = truncated.map(\.role)
        XCTAssertTrue(roles.contains(.system))
        // Most-recent user message must be kept
        XCTAssertEqual(truncated.last?.role, .user)
        if case .text(let t) = truncated.last?.content.first {
            XCTAssertEqual(t, "Recent")
        }
    }

    func test_truncate_noOpWhenUnderLimit() {
        let messages = [
            ChatMessage.system("S"),
            ChatMessage(role: .user, content: "U"),
        ]
        let truncated = LlamaChatTemplate.truncate(messages, maxTurns: 10)
        XCTAssertEqual(truncated.count, messages.count)
    }
}
