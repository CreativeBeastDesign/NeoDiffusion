Core neural blocks (M3, phase-2 §2), each fixture-diffed against the reference:

- `LLaDA2RMSNorm` — FP32-internal RMSNorm (§2.2)
- `PartialRotaryEmbedding` — partial RoPE, FP32 freqs (§2.3)
- `BlockDiffusionMask` — analytic block mask, `.strict` (0/-inf) vs `.referenceBias` (0/1); see §6 mask-semantics finding
- `LLaDA2Attention` — fused QKV split, per-head-dim qk-norm, partial RoPE, GQA via MLX fused SDPA (§2.3)
- `LLaDA2MLP` — SwiGLU dense FFN (§2.4)
- `LLaDA2MoE` — sigmoid router w/ expert bias + group-limited top-k, gathered experts (`gather_qmm` via `SwitchLinear`), shared expert (§2.5), plus `ExpertWeightStacking`
- `LLaDA2DecoderLayer` — pre-norm attention + FFN residual composition

Legacy `DiffusionCore` struct (`attentionBlock`) is an early stub, no longer used by the model.
