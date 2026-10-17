# WP-2b Dynamic Threshold Optimization: Forensic Diagnosis, Deadband Elimination, and Non-Linear Confidence Dynamics

**Author**: Gemini 4 Argon (Pro)  
**Date**: October 2026  
**Status**: Implemented & Verified in Unit Tests (Awaiting Hardware Benchmarking Slot)  
**Target Codebase**: NeoDiffusion (`DiffusionGeneration`, `DiffusionEngine`)  
**Methodology / Discipline**: Strictly adheres to `.agents/AGENTS.md` (Claims labeled `[Sourced]`, `[Inferred]`, `[Speculative]`).

---

## 1. Executive Summary & Forensic Root Cause

### 1.1 The Original Implementation
Work Package 2b (WP-2b) implements dynamic confidence thresholding $\tau(t)$ based on Streaming-dLLM (arXiv:2601.17917).
The original formulation in `DiffusionEngine+Step.swift` dropped the unmasking threshold $\tau_{\text{mask}}$ linearly with the remaining masked fraction $r_{\text{mask}}$:
$$\tau(t) = \tau_0 \cdot \big(1 - \alpha (1 - r_{\text{mask}})\big)$$
where:
- $\tau_0 = 0.70$ (Q-mode threshold `params.threshold`) `[Sourced: GenerationParams.swift:275]`
- $r_{\text{mask}} \in [0, 1]$ is the fraction of masked positions remaining in the active block `[Sourced: DiffusionEngine+Step.swift:550]`
- $\alpha \in [0, 1]$ is `params.dynamicTauAlpha` (optimum in literature $\alpha \approx 0.6$) `[Sourced: wp2b-logbook.md:12]`

### 1.2 The Deadband Pathology `[Sourced & Inferred]`
In NeoDiffusion's block denoising loop, token transitions are governed by two distinct operators:
1. **$\Gamma$ (Unmasking / M2T)**: Accepts predicted tokens at currently masked positions if confidence $> \tau(t)$.
2. **$\Delta$ (Editing / T2T)**: Modifies already unmasked tokens if the model predicts a changed token with confidence $> \tau_{\text{edit}} = 0.50$ (`params.editingThreshold` in Q-mode).

Under the original linear schedule with $\tau_0 = 0.70$ and $\alpha = 0.60$:
- At block start ($r_{\text{mask}} = 1.0$): $\tau(t) = 0.70$.
- At midpoint ($r_{\text{mask}} = 0.524$): $\tau(t) = 0.50 = \tau_{\text{edit}}$.
- At block end ($r_{\text{mask}} \to 0$): $\tau(t) \to 0.70 \cdot (1 - 0.60) = 0.28$.

**The Fatal Mechanism**:
Whenever $r_{\text{mask}} < 0.524$, $\tau(t) < \tau_{\text{edit}}$. Any token unmasked with confidence $c \in [\tau(t), 0.50)$ enters an **un-editable deadband**:
1. $\Gamma$ writes the low-confidence token into the active window.
2. In subsequent denoising steps (and during mask-free post-refinement steps), $\Delta$ requires confidence $> 0.50$ to edit or correct any token.
3. If the model predicts an alternative correct token with confidence $0.35$–$0.49$, $\Delta$ strictly rejects the edit.
4. The low-confidence token is permanently locked in. Because discrete diffusion employs bidirectional self-attention across the active block, this corrupted token corrupts the contextual embeddings of all neighbouring positions.
5. In blind human evaluation `[Sourced: wp2b-logbook.md:108-111]`, this produced severe text degradation on chat prompts (e.g. `chat-explain`: *"Imagine the sky is like a big, of,, and Sunlight is like a flashlight…"*; `chat-recipe`: repeated headers and salt quantities).

### 1.3 Mismatch with Diffusion Confidence Dynamics `[Inferred]`
Linear threshold decay assumes token confidence rises uniformly with unmasked progress $u = 1 - r_{\text{mask}}$. In reality, masked diffusion models exhibit concave/S-curve confidence dynamics:
- **Early/Mid Phase ($u < 0.5$)**: Semantic skeleton is being formed; uncertainty is high; dropping threshold early permits catastrophic premature commitments.
- **Late Phase ($u > 0.7$)**: Rich bidirectional context is established; remaining tokens are heavily constrained; confidence accelerates. This is the only phase where threshold relaxation is mathematically sound.

---

## 2. Mathematical Formulation of the Enhanced Schedule

To eliminate the deadband while respecting diffusion model confidence dynamics, we formulate the **Floor-Clamped Non-Linear Dynamic Threshold Schedule**:

### 2.1 Progress Transform $g(u)$
Let $u \in [0, 1]$ be the decoding progress over generated positions:
$$u = \text{clip}\left(1 - \frac{\sum \text{slotMask}}{\text{genCount}}, 0, 1\right)$$

We define three non-linear schedule functions $g(u) \in [0, 1]$:
1. **Half-Cosine Schedule (`.cosine`, Default)**:
   $$g_{\text{cos}}(u) = \frac{1 - \cos(\pi \cdot u)}{2}$$
   - *Boundary derivatives*: $\left.\frac{dg}{du}\right|_{u=0} = 0$ and $\left.\frac{dg}{du}\right|_{u=1} = 0$.
   - *Dynamics*: Starts flat to protect early-block semantic anchor tokens, accelerates decay during mid-to-late steps, and smoothly lands at the target floor without derivative discontinuities.
2. **Convex Power Schedule (`.power`, $p \ge 1$, default $p=2$)**:
   $$g_{\text{pow}}(u) = u^p$$
   - For $p=2$ (quadratic): at $u = 0.5$, $g(u) = 0.25$ (only 25% of decay occurs in the first half of the block). Threshold drops primarily in the final $30\%$ of the block where context is dense.
3. **Linear Schedule (`.linear`)**:
   $$g_{\text{lin}}(u) = u$$
   - Legacy Streaming-dLLM trajectory, preserved for backwards compatibility and controlled ablation.

### 2.2 Raw Threshold & Invariant Floor Clamping
The raw adaptive threshold is computed as:
$$\tau_{\text{raw}}(u) = \tau_0 \cdot \big(1 - \alpha \cdot g(u)\big)$$

The effective threshold is strictly clamped by the floor invariant:
$$\tau(t) = \max\big(\tau_{\text{raw}}(u), \, \tau_{\text{floor}}\big)$$
where:
$$\tau_{\text{floor}} = \text{dynamicTauFloor} \;??\; \text{editingThreshold}$$

### 2.3 The Deadband Elimination Invariant `[Inferred - Mathematical Proof]`
Since $\tau_{\text{floor}} \ge \tau_{\text{edit}}$ by default ($\tau_{\text{edit}} = 0.50$ in Q-mode):
$$\tau(t) \ge \tau_{\text{edit}} \quad \forall t$$
Therefore:
$$\{c \in \mathbb{R} \mid \tau(t) \le c < \tau_{\text{edit}}\} = \emptyset$$
The set of deadband tokens is strictly empty. Every token unmasked via threshold satisfies $c > \tau_{\text{edit}}$, guaranteeing that any token accepted by $\Gamma$ can subsequently be edited by $\Delta$ if the model predicts an alternative with $c' > \tau_{\text{edit}}$.

---

## 3. Architecture & Code Modifications

### 3.1 `GenerationParams.swift`
1. Added `DynamicTauSchedule` enum (`.cosine`, `.power`, `.linear`).
2. Added fields:
   - `dynamicTauSchedule: DynamicTauSchedule` (default: `.cosine`)
   - `dynamicTauPower: Float` (default: `2.0`)
   - `dynamicTauFloor: Float?` (default: `nil`, resolving to `editingThreshold`)
3. Updated `GenerationParams.init` with default arguments to ensure 100% backward compatibility for all existing call sites.

### 3.2 `DiffusionEngine+Step.swift`
Replaced linear `tauMask` calculation with vectorized MLX operations:
```swift
let u = MLX.maximum(MLXArray(Float(0)), MLX.minimum(MLXArray(Float(1)), MLXArray(Float(1)) - rMask))
let progressDecay: MLXArray
switch params.dynamicTauSchedule {
case .linear:
    progressDecay = u
case .cosine:
    let pi_u = u * MLXArray(Float.pi)
    progressDecay = (MLXArray(Float(1)) - cos(pi_u)) * MLXArray(Float(0.5))
case .power:
    progressDecay = pow(u, MLXArray(params.dynamicTauPower))
}
let tauRaw = MLXArray(params.threshold)
    * (MLXArray(Float(1)) - MLXArray(params.dynamicTauAlpha) * progressDecay)
let floorVal = MLXArray(params.dynamicTauFloor ?? params.editingThreshold)
tauMask = MLX.maximum(tauRaw, floorVal)
```

### 3.3 Provenance & Effective Echoes (Rule F7)
- Added `effectiveDynamicTauSchedule: String` and `effectiveDynamicTauFloor: Float` to `DiffusionEngine.Metrics`.
- Instrumented `DiffusionEngine+Scheduler.swift` to record actual runtime values.
- Updated `LLaDABench.swift` (`LLaDARunResult`) to serialize these effective echoes to every JSONL benchmark row.
- Added CLI arguments `--dyn-tau-schedule`, `--dyn-tau-power`, and `--dyn-tau-floor` to `LLaDABench`.

---

## 4. Acceptance Gates & Verification Status

| Gate | Requirement | Status | Sourced Evidence |
|---|---|---|---|
| **Gate 1: Toy-Config Parity** | Pass all unit tests in FP32 on seeded random weights | **PASS** | `swift test --filter WP2bTests` (9/9 passed, 57.2s) |
| **Gate 2: BF16 Token Parity** | Match reference PyTorch outputs token-for-token | **PASS** | `DenoisingLoopParityTests` (6/6 passed, 16/16 cases identical) |
| **Deadband Elimination** | Deadband tokens ($c=0.40$) strictly rejected by threshold | **PASS** | `testDynamicTauDeadbandElimination` (legacy=7 steps burst vs protected=10 steps fallback) |
| **Non-Linear Dynamics** | `.cosine`, `.power`, `.linear` execute cleanly, preserve tokens | **PASS** | `testDynamicTauNonlinearSchedules` (all 3 schedules green) |
| **Effective Echoes (Rule F7)** | Engine emits true executed schedule and floor | **PASS** | `testDynamicTauEffectiveEchoes` |
| **Hardware Benchmarking** | Validate TPS, steps/block, and text coherence on LLaDA2.1 | **QUEUED** | Ready to launch when hardware slot opens |

---

## 5. Next Steps
Once the peer benchmark process terminates and physical RAM recovers to $\ge 10$ GB, execute the benchmark driver:
```bash
.build/release/diffusion-bench llada \
  --runs 1 \
  --arms q-cached,dt-legacy,dt-linear-clamped,dt-tau,dt-power,dt-tau-a03 \
  --suites chat,reasoning,code \
  --dump-text scratch/wp2b_evaluation_texts.jsonl \
  --json scratch/wp2b_evaluation_metrics.jsonl
```
This will provide definitive, drift-free, single-process empirical evidence comparing the un-clamped legacy failure against our optimized deadband-free schedules across all standard prompts.
