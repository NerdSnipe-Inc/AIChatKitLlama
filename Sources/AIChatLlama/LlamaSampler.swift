import LlamaSwift

/// Wraps the llama_sampler chain API for composable, stateful token sampling.
///
/// Replaces the manual Swift softmax in the old LlamaProvider — the chain handles
/// top-k, top-p, min-p, temperature, and repetition penalties in the correct order.
/// Call `accept(_:)` after each generated token so repetition penalties track history.
final class LlamaSampler {

    // llama_sampler_chain_init returns UnsafeMutablePointer<llama_sampler>?
    private let chain: UnsafeMutablePointer<llama_sampler>

    init(
        temperature:    Float,
        topK:           Int32,
        topP:           Float,
        minP:           Float,
        penaltyRepeat:  Float,
        penaltyFreq:    Float,
        penaltyPresent: Float
    ) {
        // llama_sampler_chain_init is non-failing in practice; crash on nil is appropriate.
        chain = llama_sampler_chain_init(llama_sampler_chain_default_params())!

        if temperature <= 0 {
            // Greedy: always pick the highest-probability token.
            llama_sampler_chain_add(chain, llama_sampler_init_greedy())
        } else {
            // Standard pipeline:
            //  1. Penalise recently-seen tokens (frequency / presence / repetition).
            //     penalty_last_n = -1 → look at the entire context (same as LocalLLMClient).
            //  2. Top-k: keep only the K highest-logit candidates.
            //  3. Min-p: drop tokens whose probability < minP × top-token probability.
            //  4. Top-p: nucleus sampling — keep the smallest set summing to ≥ topP.
            //  5. Temperature: scale the remaining logits.
            //  6. Dist: draw the final token from the resulting distribution.
            llama_sampler_chain_add(
                chain,
                llama_sampler_init_penalties(-1, penaltyRepeat, penaltyFreq, penaltyPresent)
            )
            if topK > 0 {
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(topK))
            }
            if minP > 0 {
                llama_sampler_chain_add(chain, llama_sampler_init_min_p(minP, 1))
            }
            if topP < 1.0 {
                llama_sampler_chain_add(chain, llama_sampler_init_top_p(topP, 1))
            }
            llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature))
            llama_sampler_chain_add(chain, llama_sampler_init_dist(UInt32.random(in: 0 ... UInt32.max)))
        }
    }

    deinit { llama_sampler_free(chain) }

    /// Sample a token from the last-evaluated position's logits in `ctx`.
    func sample(_ ctx: OpaquePointer) -> llama_token {
        llama_sampler_sample(chain, ctx, -1)
    }

    /// Notify the sampler that `token` was accepted so penalties can track it.
    func accept(_ token: llama_token) {
        llama_sampler_accept(chain, token)
    }

    /// Reset the sampler's internal state (e.g. penalty token history).
    func reset() {
        llama_sampler_reset(chain)
    }
}
