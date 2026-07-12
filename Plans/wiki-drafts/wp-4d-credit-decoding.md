# WP-4D — Credit Decoding: Decay-Discounted Credit Feedback Matrix Implemented and Verified (Settle Steps Reduced up to 31.5%)

**Summary**: Fourth NeoDiffusion Phase-4 work package: [[credit-decoding]] implemented in the `DiffusionEngine` pipeline and verified across cached/uncached parity, speculation invariance, and synthetic/real benchmarks. Credit Decoding acts as a token-level momentum filter, accumulating raw prediction confidence for top candidates across step trajectories, decaying them over time, and boosting their logits to trigger early block convergence.
**Status**: completed on the dev host (2026-07-12); logic validated and benchmarked
**Type**: optimization (05-Experiments)
**Engine**: branch `optimisation/p4/WP4d`
**Sources**: [[credit-decoding]] ("dInfer: Accelerated Free-Form Selective Generation via Credit Feedback", arXiv:2510.01239)

---

## What was built

- **Credit Feedback State Management [sourced: DiffusionEngine.swift]**: Added `var credit: MLXArray? = nil` to the slot-state tracker `SlotRun`. Implemented appropriate multi-block slice allocation (`reshaped.credit = front.credit?[0..., n ..< L, 0...]` for Option-C early commits and `slots[s].credit = S == 1 ? credit : credit[0..., (s * B) ..< ((s + 1) * B), 0...]` for MultiBD split evaluations).
- **In-Graph Credit Decoding Loop [sourced: DiffusionEngine.swift]**: Updated `windowStep` to accept and return credit matrices of shape `[1, A, V]`, implementing the following update rule:
  1. Decay current credit: \(C \leftarrow C \cdot \beta\)
  2. Compute confidence boost: \(b \leftarrow (p_{\text{raw}})^{\gamma}\)
  3. One-hot mask and broadcast updates: \(C_{t, x} \leftarrow C_{t, x} + b_t \cdot [x = \hat{x}_{0, t}]\)
  4. Logit boost: \(\tilde{f}_{t} \leftarrow f_{t} + \alpha \cdot \log(1 + C_t)\)
  5. Compute new probabilities and selections from \(\tilde{f}\)
- **Token-One-Hot Memory Safety [sourced: DiffusionEngine.swift]**: Avoided allocating a full identity matrix of vocabulary size \(V = 157,184\) (\(\approx 98.8\) GB at FP32) by building localized broadcasting masks of shape `[1, A, V]` in-graph:
  ```swift
  let vocabIndices = arange(model.config.vocabSize, dtype: .int32).reshaped([1, 1, model.config.vocabSize])
  let mask = (rawX0.expandedDimensions(axis: -1) .== vocabIndices).asType(.float32)
  let updatedCredit = decayed + mask * boost.expandedDimensions(axis: -1)
  ```
  Reducing memory consumption per step to \(\approx 20\) MB.
- **Parity, Invariance, and Liveness Tests [sourced: CreditDecodingTests.swift]**: Added a test suite with 3 test cases:
  1. `testCreditDecodingCachedUncachedIdentity` verification of token-for-token parity.
  2. `testCreditDecodingSpeculationInvariance` checking that speculation \(K=1\) vs \(K=4\) outputs match.
  3. `testCreditDecodingLivenessSynthetic` demonstrating threshold acceleration.

---

## Three Key Results

1. **Successful step reduction under real workloads (Up to 31.5% decrease) [sourced: diffusion-bench log task-295]**:
   Using hyperparameters \(\alpha=1.0, \beta=0.9, \gamma=0.5\) on the `chat` prompt suite compared to baseline `q-baseline`:
   - `chat-recipe`: steps/block fell from **9.5 to 6.5** (\(\mathbf{31.5\%}\) reduction)
   - `chat-capital`: steps/block fell from **12.5 to 9.0** (\(\mathbf{28.0\%}\) reduction)
   - `chat-email`: steps/block fell from **10.5 to 10.0** (\(\mathbf{4.8\%}\) reduction)
   - `chat-explain`: steps/block stayed similar (**19.0 vs 19.5**)
   This verifies the algorithm's capability to accelerate settle times on prompts with high-confidence semantic stability.

2. **Parity and K-invariance hold perfectly [sourced: swift test task-203]**:
   All tests in `CreditDecodingTests.swift` passed. Credit Decoding preserves exact cached/uncached mathematical identity (token-for-token matches across 16 different evaluation prompts) and speculation K-invariance.

3. **High memory paging overhead on dev host masks wall-clock speedup [sourced: diffusion-bench logs]**:
   While the logical step count fell considerably, total steady-state denoise seconds for the suite stayed comparable (\(\approx 20.2\) s baseline vs \(\approx 21.1\) s credit decoding). This is *inferred* to be due to severe system memory pressure and thermal throttling on the 16GB dev host (free memory was logged at \(\le 128\) MB, inducing macOS disk paging swap activity during generation). On target hardware (M2 Studio Ultra), the wall-clock speedup is *speculative* to align directly with the logical step count reductions.

---

## Key Lessons Learned

- **Diffusion search spaces benefit from momentum**: Autoregressive models do not need credit matrix memory because they decode sequentially without revision. Non-autoregressive diffusion models, however, fluctuate dynamically during the denoising steps. Credit feedback functions as a "temporal low-pass filter", reinforcing early consensus predictions and suppressing late decoding noise, allowing high-confidence tokens to commit in bursts.
- **In-graph memory allocation requires strict scaling protection**: Broadcasting shapes with huge vocabulary dimensions (\(V \approx 150k\)) requires broadcasting small indices inside lazy evaluation sub-graphs to prevent out-of-memory or excessive allocation crashes.
