# Guided Diffusion for LLMs

**Summary**: A training-free acceleration technique for diffusion language models that uses a lightweight autoregressive model to guide token unmasking decisions, ensuring semantic coherence without speculative decoding's correction overhead.  
**Aliases**: AR-guided diffusion, FlashDLM Guided Diffusion, DLM-AR collaboration  
**Status**: emerging  
**Last updated**: 2026-04-16  
**Sources**:
- [[flashdlm-tech-report]]

---

## Definition

Guided Diffusion is a method to accelerate masked diffusion language models (dLLMs) by leveraging a small pretrained autoregressive (AR) model to guide the selection of tokens to unmask at each denoising step. The dLM proposes tokens for all masked positions in parallel, but only those tokens where the AR guider agrees (top-k match) are actually unmasked. This cross-model agreement acts as a coherence prior, reducing the risk of semantically incoherent parallel unmasking. The approach is training-free, adds minimal memory overhead, and combined with FreeCache yields up to 17.8× speedup on LLaDA-8B-Instruct with negligible accuracy loss.

## Why it matters

Parallel token generation in dLLMs can lead to incoherent outputs because each token is chosen independently from its marginal distribution. Heuristic confidence-based selection (e.g., top-k, entropy) degrades accuracy when denoising steps are reduced. Guided Diffusion introduces an external coherence signal from an AR model, which naturally models token dependencies in a left-to-right manner. The AR model's predictions provide a cheap consistency check: if the dLM's proposed token matches what the AR model would generate in that position, it's likely to fit the context. This allows aggressive parallel unmasking without sacrificing quality, and it avoids the repeated correction overhead of speculative decoding.

## Mechanism

### Core algorithm

At each diffusion step for a sequence containing both unmasked and masked tokens:

1. **DLM forward pass**: The diffusion model fθ produces logits for all masked positions. Compute top-1 predictions: `t_DLM = argmax(softmax(fθ(x)))` for each masked position. Order masked positions by some criterion (e.g., confidence).
2. **AR guider forward pass**: Construct a sequence where the currently unmasked tokens remain as is, and the DLM's proposals fill the masked positions. Feed this sequence to a pretrained AR model gϕ. The AR model processes the sequence left-to-right and produces logits for each position. Take its top-1: `t_AR = argmax(softmax(gϕ(t_DLM)))`.
3. **Agreement check**: Let M = [i₁, ..., iₘ] be the ordered masked positions. Find the largest k such that `t_DLM[iⱼ] == t_AR[iⱼ]` for all j ≤ k. This means the DLM and AR agree on a prefix of the proposed tokens.
4. **Unmask**: If k > 0, unmask positions i₁..iₖ with the DLM's proposed tokens. If k = 0 (no agreement), conservatively unmask only i₁ (the highest-confidence token according to DLM's ordering).
5. Repeat until all tokens unmasked.

Optionally, a stochastic variant: a token is unmasked if DLM's max logit > τ × (AR's max logit), with τ = 0.5 by default.

### Key properties

- **No correction loop**: Unlike speculative decoding where the AR drafter proposes a sequence and the target model verifies each token (with possible rejection and back-off), here the AR model only provides a binary accept/reject for the DLM's proposals. The AR model's own token predictions are not used; only its agreement matters.
- **DLM remains primary generator**: The final output quality is determined by the DLM's reasoning power. The AR guider merely ensures coherence.
- **Lightweight overhead**: The AR model runs a single forward pass per diffusion step; it does not need to generate tokens autoregressively over the entire sequence each time because the DLM's proposals already fill the masked positions. Actually, the AR model still processes the sequence left-to-right, but since many tokens are already unmasked, it can take those as given. The AR forward pass is relatively cheap because the model is small (1.5B vs 7B DLM).
- **Training-free**: Both models are used off-the-shelf; no fine-tuning required.

### Comparison with speculative decoding

| Aspect | Speculative Decoding (e.g., Medusa, DualDiffusion) | Guided Diffusion (FlashDLM) |
|--------|---------------------------------------------------|----------------------------|
| Drafter | Small/fast model (AR or DLM) | Not applicable |
| Verifier | Large accurate model (DLM) | The DLM itself is the verifier? Actually DLM is the main generator; AR is just a guider |
| Process | Drafter proposes multi-token draft; verifier checks each token; rejected tokens cause back-off | DLM proposes all masked tokens; AR filters; no back-off, no repeated verification |
| Overhead | Verification may need to re-run if many rejections | Single AR forward pass per step |
| Goal | Reduce number of expensive verifier steps | Reduce number of diffusion steps by enabling more parallel unmasking per step |
| Model roles | Two models collaborate; drafter's quality matters | DLM is primary; AR guide only influences which tokens get unmasked |

Guided Diffusion is more like a "coherence filter" than a speculative system.

## Trade-offs

- **Guider model size**: Larger AR models (7B) provide marginal accuracy improvements over smaller ones (1.5B), but increase memory and latency. A 1.5B model often suffices.
- **Matching criterion**: Top-1 agreement (exact match) is strict; using top-k (e.g., top-2 or top-5) increases acceptance rate and can improve accuracy (e.g., GSM8K: 79.91% with Top-1, 80.06% with Top-5 using Qwen2.5-3B). However, too permissive may admit incoherent tokens.
- **Guidance confidence threshold τ**: The stochastic variant allows tuning; τ=0.5 is default. Lower values accept more tokens (more aggressive) but risk lower quality.
- **Dependency on AR model quality**: If the AR guider is weak, its predictions may not provide a good coherence signal. Using a domain-specific AR model (e.g., Qwen2.5-Math) can boost performance on math tasks.
- **Vocabulary mismatch**: The agreement check uses token IDs; if DLM and AR have different vocabularies, direct matching may be problematic. Could use embedding similarity instead, but that adds cost.
- **Latency overhead**: Running both models each step adds overhead; but because the AR model is small and the DLM's forward pass dominates, the combined cost is still much lower than running the DLM for many more steps.
- **Accuracy**: In some cases, Guided Diffusion improves accuracy over baseline DLM (e.g., GSM8K: 79.68% → 80.33%). This suggests AR guidance corrects some incoherence errors. However, accuracy can also drop slightly (e.g., Dream-7B: 79.68% → 77.18% with FreeCache only, but Guided Diffusion recovers).

## Hardware implications

- **Memory**: Must keep both DLM and AR model weights in memory simultaneously. For LLADA2.1 (8B) plus a 1.5B guider, total ~9.5B parameters. At FP16, that's ~19 GB. On Apple Silicon, high-end memory may be 32GB or 64GB; feasible but tight. Could quantize the AR guider to INT8 or use a smaller model (e.g., 500M).
- **Compute**: The AR model runs a full forward pass per diffusion step. However, the AR model processes a sequence of length L (same as DLM output length) but its per-token cost is lower due to smaller size. The DLM's forward pass is still the dominant cost. Still, the AR forward pass adds overhead; need to ensure it's negligible compared to DLM.
- **Parallelism**: Could the AR model run in parallel on a separate stream? The DLM and AR need to run sequentially within a step (DLM first, then AR). But maybe the AR's forward pass for the next step could be overlapped with the DLM's computation? Possibly not because the AR needs the DLM's proposals first. Could offload AR to Neural Engine while DLM runs on GPU.
- **Kernel implementation**: The DLM produces logits; then we need to compute top-1 (or top-k) for each masked position. That's an argmax reduction per position. Then we need to construct the proposal sequence and run AR forward pass. The AR forward pass is standard transformer. Then compare tokens to find the longest prefix match. These are control-heavy operations; may not fuse well. Could implement as separate kernel launches.

## Apple Silicon implications

- Directly applicable to LLADA2.1; the paper validated on LLaDA-8B-Instruct.
- FreeCache + Guided Diffusion combination yields massive speedups (12-18×) that make on-device dLLM inference practical.
- **Implementation strategy**:
  - Run DLM on GPU (Metal) for parallel computation.
  - Run AR guider on Neural Engine (Core ML) to hide latency; the guider is small enough to fit on Neural Engine (ANE can handle ~1B parameters?). Need to check Apple's limits.
  - Transfer activations between GPU and ANE via shared memory (Unified Memory) might incur overhead; could pipeline.
- **Block size tuning**: Use 256 as starting point, but adjust to Apple Silicon's L2 cache size per core (e.g., 1MB). For d=4096, 256 tokens KV = 4MB, which may exceed L2; could use 128 tokens (2MB) or 64 (1MB). Experiment.
- **Combine with other optimizations**:
  - Guided Diffusion could be used with ICE: after early exit triggers, maybe skip AR guidance for the final answer? Or use a heavier guider?
  - Could use Model Scheduling: switch between DLM alone and DLM+Guided Diffusion based on task demands.
  - Could combine with Elastic-Cache: the AR guider might help decide when a token is stable enough to freeze.
- **Open questions**:
  - What is the minimal AR guider size that preserves accuracy? Possibly a 500M or even 100M model distilled for coherence checking.
  - Could the AR guider be replaced by a tiny classifier or a heuristic? The agreement metric might be approximated more cheaply.
  - How does Guided Diffusion perform on very long contexts (>2k tokens)? The AR model may have limited context window; need to ensure it can handle the full sequence.
  - Does Guided Diffusion introduce bias toward tokens typical of the AR model's distribution? Might affect diversity or cause mode dropping for domain-specific tasks.
  - What is the energy consumption impact on battery? Running two models, albeit one small, might increase power; but the reduced steps may compensate.
  - Could we implement the AR guider as a "read-only" copy of part of the DLM's own weights (self-guidance)? Possibly via confidence self-consistency.
  - How does Guided Diffusion interact with exposure bias? The AR model's own autoregressive nature might introduce a different bias.
  - Could we use the AR guider's probabilities to make soft decisions (e.g., require agreement with confidence > threshold) instead of binary top-1 match?
  - What happens if the DLM and AR model disagree on many tokens? The method falls back to slow single-token unmasking; still correct but slower. The frequency of this affects speed.
  - Could we precompute the AR model's predictions for the entire sequence in one go? No, because the AR model depends on the DLM's proposals which change each step.

## Related concepts

- [[speculative-decoding-dllm]] (DualDiffusion uses drafter-verifier; Guided Diffusion uses AR guidance without correction)
- [[in-place-chain-of-thought]] (ICE reduces steps via early exit; Guided Diffusion increases parallelism per step; could combine)
- [[freecache]] (FlashDLM's KV caching component; often used together)
- [[fast-dllm]] (caching approximations; different approach)
- [[dkv-cache]] (KV caching for dLLMs)
- [[llada2-1-tech-report]] (base model)

## Extracted concepts from this source

- AR-guided token unmasking
- Cross-model agreement as coherence signal
- Top-k matching between DLM and AR
- DLM-AR collaboration without speculative correction
- Training-free coherence enhancement
- Stochastic guided unmasking (confidence threshold variant)
- Lightweight AR guider

## Open questions

- What is the optimal AR guider model size for on-device use? Can we get away with <500M parameters? Could we train a tiny specialized guider?
- Could the AR guider be replaced by a simpler heuristic (e.g., n-gram consistency, embedding similarity) to reduce overhead further?
- How does Guided Diffusion scale with sequence length? The AR model's self-attention is O(L²); for long outputs (>2k), AR cost might dominate.
- Could we run the AR guider on the Neural Engine in parallel with the DLM on GPU? Requires careful synchronization; but maybe the AR can process previous step's proposals while DLM computes next step's proposals? Not possible because AR needs DLM's proposals first.
- Does Guided Diffusion work with encoder-decoder AR models? Possibly, but the decoder is the relevant part.
- How does the method handle tokenization differences between DLM and AR? If they use different tokenizers, need subword mapping or use character-level? Likely both use same tokenizer in practice.
- Could we use the AR guider's top-k probabilities to weight the DLM's confidence, creating a hybrid score?
- What is the impact on sample diversity? The AR model might favor more probable (less diverse) tokens, potentially reducing creativity.
- Could Guided Diffusion be applied to multimodal dLLMs where the AR guider is a vision-language model? Interesting but complex.
- How does the method behave when the DLM is already highly confident? The AR might not add much.
- Could the AR guider be distilled from the DLM itself to reduce memory footprint? Possibly train a small AR to mimic the DLM's conditional distributions.
- Implementation complexity: The top-k matching and longest prefix computation need to be efficient; can be done with parallel prefix scan.
- Does Guided Diffusion work with continuous diffusion? The concept of token matching wouldn't apply; but maybe latent-space alignment?
- What is the latency breakdown on Apple Silicon? GPU vs Neural Engine; memory transfer overhead.
- Could we use multiple AR guiders with different strengths and combine their votes?
- How does Guided Diffusion compare to using the DLM's own self-consistency (e.g., multiple diffusion steps, pick the most consistent)? Possibly slower.
- Could the AR guider be used not only for unmasking decisions but also to influence which positions to unmask (the ordering)? Currently ordering is by DLM confidence; could incorporate AR confidence.
- Is there a risk of error amplification: if the AR model consistently makes a certain type of mistake, it might reject correct DLM proposals? Possibly.
- Could we update the AR guider's weights during inference? Probably not; but maybe a Bayesian approach.
- How does the method interact with model quantization? Both DLM and AR might be quantized; need to ensure agreement metric still valid.
- Could we use a more flexible matching metric than exact token ID equality? For example, embedding similarity above a threshold. This would allow different vocabularies but adds cost.
- What is the effect of the speculation block size (how many tokens the DLM proposes at once)? Paper uses 32; larger blocks might increase parallelism but require more AR processing.
- Could we skip AR guidance in later steps when the DLM is already stable? Adaptive guidance.
- How does Guided Diffusion perform on tasks with very low token diversity (e.g., code generation)? The coherence benefit might be smaller.

## Related pages

- [[freecache]]
- [[flashdlm-tech-report]]
- [[speculative-decoding-dllm]]
- [[in-place-chain-of-thought]]
- [[llada2-1-tech-report]]
