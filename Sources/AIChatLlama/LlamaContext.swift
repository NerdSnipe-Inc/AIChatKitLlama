import LlamaSwift
import AIChatCore

/// Manages a single `llama_context` and its KV cache across multiple inference requests.
///
/// Rather than creating a new context for every request (O(n²) prompt re-evaluation
/// for multi-turn chat), this type tracks every token that has been evaluated so that
/// subsequent requests can reuse the KV cache for any common prefix.
///
/// Typical multi-turn flow:
///  1. Turn 1 prompt tokenises to [A, B, C]. All three are evaluated; position = 3.
///  2. Turn 2 prompt tokenises to [A, B, C, D, E]. The common prefix is 3 tokens,
///     so only [D, E] need to be evaluated; position = 5.
///  3. If the conversation is reset or a different branch is taken, the KV cache is
///     trimmed to the common prefix via `llama_memory_seq_rm` and the new suffix
///     is evaluated.
final class LlamaContext {

    let ctx: OpaquePointer
    private var batch: llama_batch
    let maxContextSize: UInt32          // UInt32 — matches llama_n_ctx return type
    private let batchSize: Int

    /// All tokens currently reflected in the KV cache: prompt tokens + generated tokens.
    private(set) var cachedTokens: [llama_token] = []

    /// Number of tokens in the KV cache — used as the next token's KV position.
    var position: Int32 { Int32(cachedTokens.count) }

    /// How many more tokens can be evaluated before the context window is exhausted.
    var remainingCapacity: Int32 { Int32(maxContextSize) - position }

    // MARK: - Lifecycle

    init(model: OpaquePointer, contextSize: UInt32, nBatch: UInt32) throws {
        var params     = llama_context_default_params()
        params.n_ctx   = contextSize
        params.n_batch = nBatch
        guard let c = llama_init_from_model(model, params) else {
            throw ChatError.invalidConfiguration("llama_init_from_model returned nil")
        }
        ctx           = c
        batchSize     = Int(nBatch)
        maxContextSize = llama_n_ctx(c)     // actual allocated size (may differ from requested)
        batch         = llama_batch_init(Int32(nBatch), 0, 1)
    }

    deinit {
        llama_batch_free(batch)
        llama_free(ctx)
    }

    // MARK: - KV cache management

    /// Evaluate `tokens` into the KV cache, reusing any common prefix already there.
    ///
    /// - If `tokens` starts with everything in `cachedTokens`, only the new suffix is
    ///   evaluated (cache hit on the common prefix).
    /// - If `tokens` diverges at some point, the KV cache is trimmed to the common
    ///   prefix via `llama_memory_seq_rm`, then the diverging suffix is evaluated.
    ///
    /// Logits are enabled only for the very last token so that `LlamaSampler.sample`
    /// can draw the first generated token immediately afterwards.
    func prepareForTokens(_ tokens: [llama_token]) throws {
        // Find longest common prefix between what is cached and the new token list.
        var commonLen = 0
        let limit = min(cachedTokens.count, tokens.count)
        while commonLen < limit, cachedTokens[commonLen] == tokens[commonLen] {
            commonLen += 1
        }

        if commonLen < cachedTokens.count {
            // Cached tokens diverge from the new prompt — trim KV cache to common prefix.
            // p1 = -1 means "to the end of the sequence" in llama.cpp.
            llama_memory_seq_rm(llama_get_memory(ctx), 0, Int32(commonLen), -1)
            cachedTokens = Array(cachedTokens.prefix(commonLen))
        }

        let suffix = Array(tokens.dropFirst(commonLen))
        guard !suffix.isEmpty else { return }
        try evaluateChunked(suffix)
    }

    /// Evaluate and KV-cache a single just-generated token, then decode for next logits.
    func decodeSingleToken(_ token: llama_token) throws {
        batch.n_tokens      = 1
        batch.token[0]      = token
        batch.pos[0]        = position          // position before this token is appended
        batch.n_seq_id[0]   = 1
        batch.seq_id[0]?[0] = 0
        batch.logits[0]     = 1                 // enable logits for next-token sampling

        guard llama_decode(ctx, batch) == 0 else {
            throw ChatError.streamError("llama_decode failed at position \(position)")
        }
        cachedTokens.append(token)
    }

    /// Clear the KV cache entirely and reset position to zero.
    func reset() {
        llama_memory_clear(llama_get_memory(ctx), true)
        cachedTokens = []
    }

    // MARK: - Private helpers

    /// Evaluate `tokens` in batchSize-sized chunks. Logits enabled for the final token only.
    private func evaluateChunked(_ tokens: [llama_token]) throws {
        let startPos = position     // capture before any append; position is a computed property
        var offset   = 0

        while offset < tokens.count {
            let chunkSize   = min(batchSize, tokens.count - offset)
            let isLastChunk = (offset + chunkSize) >= tokens.count

            batch.n_tokens = Int32(chunkSize)
            for i in 0..<chunkSize {
                batch.token[i]      = tokens[offset + i]
                batch.pos[i]        = startPos + Int32(offset) + Int32(i)
                batch.n_seq_id[i]   = 1
                batch.seq_id[i]?[0] = 0
                // Enable logits only for the very last token of the very last chunk.
                batch.logits[i]     = (isLastChunk && i == chunkSize - 1) ? 1 : 0
            }

            guard llama_decode(ctx, batch) == 0 else {
                throw ChatError.streamError(
                    "llama_decode failed at position \(startPos + Int32(offset))"
                )
            }
            offset += chunkSize
        }

        cachedTokens.append(contentsOf: tokens)   // update after all chunks succeed
    }
}
