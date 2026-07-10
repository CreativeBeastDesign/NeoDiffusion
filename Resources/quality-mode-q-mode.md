# Quality Mode (Q Mode)

**Summary**: Operating mode that uses conservative τ_mask to maximize output quality at the cost of throughput.  
**Aliases**: quality-focused mode, conservative mode  
**Status**: stable  
**Last updated**: 2026-04-16  
**Sources**:
- [[llada2-1-tech-report]]

---

## Definition

Quality Mode (Q Mode) configures LLaDA2.1's decoding with a higher mask-to-token confidence threshold τ_mask. Only high-confidence masked positions are filled at each step, resulting in fewer tokens generated per forward pass and slower overall generation, but with cleaner initial drafts that require less editing correction.

## Why it matters

Q Mode serves as the quality anchor, comparable to traditional dLLM decoding. It demonstrates that LLaDA2.1 can match or exceed LLaDA2.0's benchmark scores while still supporting the editing infrastructure. It's the recommended mode for tasks where quality is paramount and speed is secondary.

## Mechanism

- τ_mask set high (exact values not given)
- Fewer unmasking events per step → longer generation time
- T2T editing still active but may correct fewer errors due to cleaner drafts
- MBE can still be applied for additional polish

## Trade-offs

- **Quality**: Highest benchmark scores, often surpassing LLaDA2.0
- **Speed**: Lower TPS (e.g., ~3–5 TPF vs 5–10+ in S Mode)
- **Reliability**: Less susceptible to stuttering artifacts
- **Domain**: Works well across all domains; especially recommended for general chat

## Apple Silicon implications

- Lower parallelism means GPU may be underutilized; could adjust batch size to improve occupancy.
- Still benefits from block-wise attention and FP8 quantization.
- Might be preferred for on-device scenarios where quality is more important than raw speed.
- Could implement dynamic switching between Q and S modes based on user preference.

## Related concepts
- [[speedy-mode-s-mode]]
- [[configurable-threshold-decoding]]

## Open questions
- Can Q Mode benefit from selective MBE without sacrificing too much speed?
- How does Q Mode's throughput scale with Apple Silicon memory bandwidth?