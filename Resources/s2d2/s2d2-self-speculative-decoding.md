---
Created: 2025-01-XX
Type: source
Subtype: paper
Tags: [self-speculation, block-diffusion, training-free, speedup]
related_to:
  - "[[just-on-time-jot]]"
  - "[[locally-coherent-codila]]"
---

# S2D2: Fast Decoding for Diffusion LLMs via Training-Free Self-Speculation

**Authors:** Ligong Han, Hao Wang, Han Gao, Kai Xu, Akash Srivastava
**Institution:** Red Hat AI Innovation, MIT-IBM Watson AI Lab, Iowa State University, Core AI, IBM
**arXiv:** [2603.25702](https://arxiv.org/abs/2603.25702)
**Code:** [github.com/phymhan/S2D2](https://github.com/phymhan/S2D2)

## Core Contribution

S2D2 presents the first **training-free self-speculative decoding framework** for block-diffusion language models. The key insight: when block size is reduced to 1, a block-diffusion model becomes autoregressive, allowing the same pretrained model to serve as both **drafter** (standard block-diffusion) and **verifier** (block-size-1 AR mode).

## Key Results

| Model | Config | GSM8K | MBPP | Speedup vs AR | Speedup vs Dynamic |
|-------|--------|-------|------|---------------|-------------------|
| SDAR-1.7B | Config-B | 73.8 | 44.4 | 4.7× | 1.57× |
| SDAR-4B | Config-B | 87.4 | 57.0 | 4.3× | 1.45× |
| SDAR-8B | Config-A | 89.6 | 62.0 | 2.0× | 1.29× |
| LLaDA2.1-Mini | Conservative | 89.8 | 68.8 | 2.2× | 1.3× |

**Highlights:**
- Up to 4.7× speedup over AR decoding on SDAR-1.7B
- Up to 1.57× speedup over dynamic confidence-threshold baseline with +4.5 accuracy improvement
- Complementary to LLaDA's built-in self-correction mechanism

## Method

### Self-Speculative Decoding Architecture

```
Standard Block-Diffusion Decoding (Drafter):
  Block size B → parallel token proposal with confidence scores

Verification Step (Verifier):
  Block size 1 → AR mode, rejection sampling on proposed tokens

Verification Routing:
  Lightweight policies decide when verification is worth the extra cost
```

### Verification Mode

For position-aligned models (LLaDA, SDAR), uses "2L trick" for parallel verification:

```
M_ver = [[A_L,     0_L],
         [A_<L,    I_L]]
```

Where A_L is the causal mask and A_<L is strict lower-triangular.

For right-shifted models (Dream, Fast-dLLM v2), standard causal mask provides verifier view.

### Routing Policies

1. **Minimum-span:** Verify when |Ct| ≥ τ_span
2. **Score-threshold:** Verify when s ≥ τ_score using margin/entropy-based scores
3. **Hysteresis:** Avoid oscillation between speculative and diffusion modes
4. **UCB bandit:** Contextual bandit router for adaptive routing

### Expected Accepted Prefix Length

$$\hat{K} = \sum_{k=1}^{L} k \prod_{i=1}^{k} \alpha_i$$

Where α_i is approximated by:
- Margin-based: α_i = 1[m_i ≥ τ_margin]
- Entropy-based: α_i = exp(-β * H̃_i)

## Connection to Residual Energy Correction

The local residual energy for a drafted token:

$$E_i(\hat{x}_i) = -\log q_i + \log p_i$$

Acceptance probability: min(1, q_i/p_i) = min(1, exp(-E_i))

This provides a **stochastic, greedy local preference for lower residual energy**, interpreting speculative verification as AR-guided energy correction.

## Models Evaluated

1. **SDAR** (1.7B/4B/8B) - Adapted from AR models
2. **Fast-dLLM v2** - Right-shifted architecture  
3. **LLaDA2.1** - Trained from scratch, supports token editing

## Benchmarks

- **GSM8K** - Mathematical reasoning
- **MBPP** - Python code generation
- **HumanEval** - Code generation
- **IFEval** - Instruction following

## Comparison with Related Work

| Method | Training Required | Extra Compute | Plug-and-Play |
|--------|-------------------|---------------|---------------|
| EDLM | Yes (AR energy model) | Multi-sample inference | No |
| ASSD | Yes (any-subset AR) | Architecture-specific | No |
| SSD (Gao et al.) | No | Hierarchical batching | Partial |
| **S2D2** | **No** | **One verifier pass** | **Yes** |

## Limitations

- Verification only on first contiguous masked span Ct
- Trajectory is hybrid, not globally causal
- Optional partially causal drafting variant available

## Open Questions

1. How does S2D2 interact with larger block sizes where diffusion decoding is unstable?
2. Can the routing policies be learned adaptively during generation?
3. How does the local energy interpretation connect to global quality improvements?