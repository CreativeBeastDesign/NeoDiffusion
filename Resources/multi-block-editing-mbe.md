# Multi-Block Editing (MBE)

**Summary**: Mechanism that allows the model to revise previously generated blocks based on content from newly decoded blocks, improving global consistency.  
**Aliases**: cross-block refinement, multi-pass editing  
**Status**: stable  
**Last updated**: 2026-04-16  
**Sources**:
- [[llada2-1-tech-report]]

---

## Definition

Multi-Block Editing (MBE) extends the basic T2T editing (which operates within a single block) to operate across block boundaries. After generating a new block, the model can revisit and revise earlier blocks, using the new context to improve coherence and correctness.

## Why it matters

MBE addresses the limitation that editing confined to the current block cannot fix global inconsistencies that involve multiple blocks. The paper shows MBE yields consistent performance improvements across benchmarks, particularly on reasoning and coding tasks, at a modest throughput cost.

## Mechanism

- Generation proceeds block by block.
- After completing block `b+1`, the model re-evaluates block `b` (or earlier blocks) with the new context available.
- Editing criteria same as T2T: if alternative token confidence exceeds τ_edit and differs from current, replace.
- Can be applied iteratively across multiple block ranges.

## Trade-offs

- **Quality**: Improves global consistency, especially for long-form or structured output.
- **Throughput**: Additional forward passes per edited block; paper shows ~5–15% TPF reduction.
- **Latency**: Increases total generation time proportionally to number of editing passes.
- **Diminishing returns**: More than 1–2 passes may yield minimal gains.

## Apple Silicon implications

- Block size should be tuned to Apple Silicon's cache hierarchy and threadgroup sizes.
- MBE increases kernel launch frequency; consider fusing multiple edit passes into one kernel with divergent control flow.
- Memory bandwidth: re-reading earlier block embeddings for editing evaluation.
- Could schedule MBE opportunistically when GPU is idle to hide latency.

## Related concepts
- [[token-to-token-t2t]]
- [[editable-state-evolution]]
- [[block-wise-causal-attention]]

## Open questions
- What is the optimal number of MBE passes for typical task lengths on Apple Silicon?
- Can MBE be parallelized across multiple blocks simultaneously?