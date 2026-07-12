# dInfer: Efficient Inference Framework for Diffusion Language Models

**Source**: dInfer technical report (Ant Group et al., October 2025)  
**Title**: "dInfer: An Efficient Inference Framework for Diffusion Language Models"  
**Authors**: Yuxin Ma, Lun Du, Lanning Wei, Kun Chen, Qian Xu, Kangyu Wang, Guofeng Feng, Guoshan Lu, Lin Liu, Xiaojing Qi, Xinyuan Zhang, Zhen Tao, Haibo Feng, Zhiyun Jiang, Ying Xu, Zenan Huang, Yihong Zhuang, Haokai Xu, Jiaqi Hu, Zhenzhong Lan, Junbo Zhao, Jianguo Li, Da Zheng  
**Institutions**: Ant Group, Zhejiang University, Westlake University, Renmin University of China, University of Chinese Academy of Sciences, Shanghai Jiao Tong University  
**Web**: https://github.com/inclusionAI/dInfer  
**PDF**: `01-Inbox/dInfer.pdf`  
**Status**: stable  
**Ingested**: 2026-04-16

---

## Summary

dInfer is the first modularized inference framework for diffusion-based large language models (dLLMs). It decomposes the inference pipeline into four components—model, diffusion iteration manager, decoding strategy, and KV-cache manager—and integrates algorithmic innovations (hierarchical decoding, credit decoding, iteration smoothing, vicinity KV-cache refresh) with system-level optimizations (tensor parallelism, expert parallelism, PyTorch compilation, CUDA Graphs, loop unrolling). dInfer achieves over 1,100 tokens per second on HumanEval at batch size 1 on 8× H800 GPUs, representing a 10× speedup over Fast-dLLM and 2–3× speedup over vLLM on Qwen2.5-3B while maintaining comparable accuracy.

## Key claims

1. **First modular dLLM framework**: dInfer introduces a four-component architecture enabling flexible algorithm combinations.
2. **Parallel decoding breakthroughs**: Hierarchical decoding and credit decoding increase tokens decoded per forward pass (TPF) by 30–40% and up to 99.8% in Trajectory Distillation models.
3. **KV-cache for dLLMs**: Vicinity KV-cache refresh selectively updates cache near decoding blocks, balancing reuse and accuracy.
4. **System efficiency**: Combines TP+EP, torch.compile, CUDA Graphs, and loop unrolling to achieve >100% improvement from parallelism alone, plus 200%+ from compilation, and 5–10% from loop unrolling.
5. **Trajectory Distillation**: Post-training method that fine-tunes models on compressed transitions from high-quality generation trajectories, boosting TPF by 45–99%.
6. **Batch size 1 viability**: dInfer demonstrates that dLLMs can exceed AR models in throughput even at batch size 1, challenging the notion that dLLMs require large batches.

## Mechanisms

### Four modular components

1. **Model**: Supports LLaDA-MoE, LLaDA-1.5, LLaDA-Instruct; uses vLLM backend.
2. **Diffusion iteration manager**: Controls iteration flow:
   - Blockwise: fixed-size spans
   - Iteration Smoothing (IterSmooth): fuses logit-weighted embeddings from previous iteration into mask embeddings to enrich context.
3. **Decoding strategy**: Three training-free methods:
   - Threshold decoding: commit tokens with confidence > threshold.
   - Hierarchical decoding: recursively partition masked spans to ensure at least one token decoded per region, reducing local dependencies.
   - Credit decoding: accumulate credit scores across steps; tokens with stable predictions get boosted.
4. **KV-cache manager**:
   - Vicinity refresh: recompute K/V for masked tokens and immediate neighbors within a small window; full cache update after block completion.

### System-level optimizations

- **Tensor parallelism (TP)**: Applied to linear layers before attention.
- **Expert parallelism (EP)**: For MoE layers; effective even at batch size 1.
- **torch.compile**: Fuses CUDA kernels.
- **CUDA Graphs**: Eliminates PyTorch execution overhead.
- **Loop unrolling**: Eliminates CUDA stream bubbles between diffusion iterations, keeping GPU pipelines occupied.
- **Early termination**: Upon EOS generation, fills remaining blocks with EOS to skip unnecessary computation.

### Trajectory Distillation (LLaDA-MoE-TD)

Two-stage fine-tuning:
1. **Trajectory distillation**: Generate high-quality trajectories using pretrained model; filter by correctness (e.g., math verifier).
2. **Compressed transition learning**: Train model to predict multi-step transitions (from state `si` to `sj` where i > j), reducing required denoising steps.

Loss: `Lcompress(θ) = -E[∑_{k∈Δi→j} log pθ(xk = sj[k] | si)]`.

## Hardware assumptions

- NVIDIA H800 GPUs (Hopper)
- CUDA, PyTorch 2.9+, vLLM backend
- FP16/BF16 likely (not specified)
- High memory bandwidth for KV-cache
- Fast inter-GPU communication for TP/EP AllReduce

## Extracted concepts

- Diffusion language model inference
- Modular framework design
- Blockwise decoding
- Iteration smoothing (IterSmooth)
- Hierarchical decoding
- Credit decoding
- Vicinity KV-cache refresh
- Trajectory distillation
- Tokens per forward (TPF)
- Tokens per second (TPS)
- Early termination (EOS detection)
- Loop unrolling
- CUDA Graphs
- torch.compile
- Expert parallelism (EP) in dLLMs
- Tensor parallelism (TP) in dLLMs
- Masked token embedding enrichment
- Multi-step transition training

## Relevance to dLLMs

- dInfer provides a reference architecture for building dLLM inference systems, including potential implementations on Apple Silicon.
- Iteration Smoothing's idea of enriching masked positions with distributional information could be adapted to improve dLLM convergence without retraining.
- Credit decoding's temporal accumulation could be implemented within dLLM sampling loops on Metal.
- Vicinity KV-cache refresh addresses the fundamental challenge of bidirectional attention in dLLMs; similar strategies needed for Metal-based caching.
- Trajectory Distillation shows that models can be trained to require fewer denoising steps, directly benefiting latency on resource-constrained devices.
- The demonstrated >1,100 TPS at batch size 1 validates that dLLMs can be efficient in low-batch scenarios, contrary to prior belief that large batches are necessary.

## Open questions (raised by this source)

- How does dInfer's vicinity KV-cache refresh perform on long sequences (e.g., >4k tokens) where vicinities may overlap insufficiently?
- What is the memory overhead of storing credits per token per vocabulary in CreditDecoding? Could it become a bottleneck for long contexts?
- Can Iteration Smoothing's mixing weight αt be scheduled optimally per dataset or model size?
- Does Trajectory Distillation generalize across domains (e.g., math→code) or is it domain-specific?
- How does dInfer compare to SGLang's rollout engine for LLaDA2.1? Both claim high throughput; what are the algorithmic differences?
- Could the combination of TP+EP be adapted to Apple Silicon's unified memory architecture? What would be the optimal partitioning?
- Is loop unrolling necessary on Metal, or does the stream model differ sufficiently that bubble overhead is lower?
- What is the accuracy impact of vicinity refresh vs full recompute? The paper gives accuracy numbers but not detailed ablations of cache staleness.

## Suggested next probes

- Deep-dive into IterSmooth: exact formula, αt schedule sensitivity, and effect on convergence.
- Implement a simplified version of credit decoding in a dLLM sampler to measure credit storage and speed impact.
- Compare dInfer's blockwise vs Hierarchical decoding token budgets per forward pass.
- Extract detailed performance numbers: TPF and TPS per dataset, per configuration (with/without KV cache, with/without Trajectory Distillation).
- Understand the interaction between iteration smoothing and vicinity refresh: does smoothing help when KV cache is stale?
- Evaluate memory footprint of credit decoding: per-token per-vocab credit is O(V) per position; can it be compressed or approximated?
- Investigate whether dInfer's optimizations (especially TP+EP) are applicable to Apple Silicon's multi-core GPU; how would one implement expert parallelism on Metal?

## Related pages

- [[multi-block-editing-mbe]] (dInfer uses blockwise operations)
- [[editable-state-evolution]] (dVicinity refresh manages editable states)
- [[vectorized-likelihood-estimation]] (credit decoding uses per-token scoring)
- [[mega-kernel-v1-fused-remask-sample]] (dInfer's modularity contrasts with megakernel approach; explore synergies)
- [[fusion-architecture-for-editable-dllms]] (dInfer's design could inform a fused Metal implementation)
- [[sglang-rollout-engine]] (compare architectures)

## Technical details

### Algorithm 1 (Blockwise dLLM Inference)

```
Require: Input tokens X ∈ ℤ^(B×L) (undecided positions marked as mask id); block size S; model M; decoder D; KV-cache manager K; block iteration manager I
Ensure: Completed tokens X̂ ∈ ℤ^(B×L)
1: K.CREATE(B, L)
2: while I.HASNEXT() do
3:   [start:end] ← I.NEXTBLOCK()
4:   undecided ← (X[start:end] = mask id)
5:   while any(undecided) do
6:     if K.SHOULDUPDATE(loop context, start:end) then
7:       K.UPDATE(X, start:end)
8:     end if
9:     logits ← M.FORWARD(X, K, start:end)
10:    (X, undecided) ← D.DECODE(logits, X, undecided, start:end)
11:  end while
12: end while
13: return X
```

### Iteration Smoothing (IterSmooth)

For masked positions at step t with logits `z_t[i]`:

```
p_t[i] = softmax(z_t[i])
Δe_t[i] = p_t[i] · W_emb
e_{t+1}[i] = e_mask + α_t · Δe_t[i]
α_t = min(α_init + α_growth·t, α_preset)
```

Typically α_init=0.1, α_preset∈[0.2,0.4]. Also decode-threshold schedule decays from 1.0 toward target.

### Credit Decoding

Credit update for position i, token v:

```
C_{i,v}^t = {
    β·C_{i,v}^{t-1} + (p_θ(v|x_t))^γ,   if v = v* (top candidate)
    β·C_{i,v}^{t-1},                    otherwise
}
```

Enhanced logits: `f̃_θ(x_t)_i^v = f_θ(x_t)_i^v + α·log(1 + C_{i,v}^t)`

Parameters: β∈(0,1), γ∈(0,1), α>0.

### Vicinity KV-Cache Refresh

- Prefix look: recompute K/V for `tokens before block - prefix_look`
- After look: recompute for `tokens after block + after_look`
- Warmup: number of initial iterations before enabling refresh
- Default: prefix_look=16, after_look=16, warmup=4

### Trajectory Distillation

- Generate trajectories τ = (s_N, s_{N-1}, ..., s_0) using pretrained model.
- Filter correct trajectories: T_gold = { τ | V(s_0) = True }.
- Sample i > j from τ, train on Δ_{i→j} = M_i \ M_j (tokens revealed between states).
- Loss: `L_compress(θ) = -E[∑_{k∈Δ_{i→j}} log p_θ(x_k = s_j[k] | s_i)]`.

Improves TPF by 45–99%.

## Performance summary

| Configuration | Avg Perf | Avg TPF | Avg TPS |
|---------------|----------|---------|---------|
| LLaDA-MoE (baseline) | 54.83 | 1 | 277.45 |
| Fast-dLLM (w/o KV) | 53.52 | 2.82 | 63.61 |
| dInfer (w/o KV) | 54.33 | 4.29 | 407.36 |
| Fast-dLLM (w/ KV) | 52.15 | 2.46 | 110.98 |
| dInfer (w/ KV) | 53.96 | 3.87 | 680.71 |
| dInfer + TrajDist (w/ KV) | 52.72 | 5.67 | 847.22 |

Benchmarks: CRUX-O, GSM8K, HumanEval, IFEval, MBPP, LCB V6. Hardware: 8× H800, batch size 1, seq len 1024, block size 64.

Speedups:
- vs Fast-dLLM (KV): ~10× (TPS 680 vs 63)
- vs QWen2.5-3B vLLM: 2.5× (TPS 680 vs 277)
- vs Fast-dLLM (KV) w/ TrajDist: >13× (TPS 847 vs 63)
