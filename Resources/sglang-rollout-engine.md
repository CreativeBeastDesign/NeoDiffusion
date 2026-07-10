# SGLang Rollout Engine

**Summary**: Customized version of SGLang used as the dedicated rollout engine for LLaDA2.1's reinforcement learning and inference.  
**Aliases**: SGLang for dLLMs, rollout engine  
**Status**: stable  
**Last updated**: 2026-04-16  
**Sources**:
- [[llada2-1-tech-report]]

---

## Definition

SGLang is an open-source system for efficient execution of large language models. LLaDA2.1 uses a customized version of SGLang as its rollout engine, which handles the generation of trajectories during reinforcement learning training and also powers inference. The customization likely includes support for block diffusion, editable state evolution, and efficient batching.

## Why it matters

The rollout engine is responsible for sampling from the policy during RL and for serving the model at inference. SGLang's optimizations (radix caching, batching, block-wise attention) are essential for achieving LLaDA2.1's high throughput. Understanding SGLang's role helps in mapping components to potential Metal implementations.

## Mechanism

- SGLang provides a high-level programming abstraction for defining LLM execution graphs.
- Customizations for dLLMs: support for M2T/T2T decoding, block-wise attention, radix caching.
- During RL: generates rollouts (sequences or blocks) according to current policy for further evaluation.
- During inference: interprets the decoding algorithm (thresholds, MBE) and executes kernels accordingly.

## Trade-offs

- **Flexibility**: SGLang allows rapid experimentation with decoding strategies.
- **Portability**: SGLang is primarily CUDA-focused; porting to Metal requires significant engineering.
- **Performance**: Customizations must be carefully tuned to avoid overhead; abstraction may limit low-level optimizations.
- **Ecosystem**: Leverages existing SGLang features (e.g., radix caching, batching) which are proven at scale.

## Apple Silicon implications

- SGLang would need a Metal backend; this is non-trivial but could be built using Metal Performance Shaders or custom kernels.
- The rollout engine's batching and caching logic could be retained; only the low-level kernel execution changes.
- Might consider using SGLang's model specification but replace runtime with Metal.

## Related concepts
- [[radix-caching]]
- [[block-wise-causal-attention]]
- [[areal-framework]] (RL integration)

## Open questions
- How much of SGLang's codebase is reusable vs needs rewriting for Metal?
- Does SGLang's abstraction impose performance penalties that are unacceptable for high-throughput Apple Silicon deployment?