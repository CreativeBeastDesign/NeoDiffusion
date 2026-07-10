# NeoDiffusion — Metal shader guide & operation-level sketches

**Status**: draft, for André's review
**Written**: 2026-07-07, alongside the LMoE / Sumi / Nemotron guides
**Companion documents**: `phase-1-conceptual-design.md` §4.2 & §8, `phase-3-optimisation-roadmap.md` §4 (kernel-fusion track), `lmoe-implementation-guide.md` §5, `sumi-implementation-guide.md` §5, `nemotron-labs-diffusion-implementation-guide.md` §5, `diffusiongemma-implementation-guide.md` §5. This document is the **cross-model kernel reference** those §5 sections point into.

**Provenance discipline**: same as the model guides — every load-bearing claim is **sourced** (URL/file:line), **inferred** (derived from sourced facts), or **speculative** (argued without a direct source).

---

## TL;DR — the honest take before the sketches

Four things you should internalise before writing any `.metal` file:

1. **Do not write Metal until Instruments says the corresponding MLX region is a hot path.** The `phase-3-optimisation-roadmap.md` §4 discipline is not there to slow you down — it is there because every one of these kernels has a "looks obviously fused, actually slower than MLX baseline" mode. `fused_residual_rmsnorm` written naively is *slower* than MLX's `add + rms_norm` because MLX's kernels are already threadgroup-memory-tuned. The measured baseline is what tells you whether your fusion is a win or a wash.

2. **M1/M2 GPUs lack native BF16 compute.** Confirmed [`philipturner/metal-benchmarks`](https://github.com/philipturner/metal-benchmarks), corroborated by the practitioner note [DEV Community — bf16 vs fp16 M1](https://dev.to/sleepyquant/why-apple-silicon-quietly-won-the-local-ai-race-april-2026-34g7) and by the Apple developer thread on [MPSMatrixMultiplication bf16](https://developer.apple.com/forums/thread/707757). **Consequence**: MSL kernels must declare working precision as `float` or `half`; `bfloat` as an MSL type exists in the shading language spec but on Apple7/Apple8 GPUs it degrades to software-emulated conversion at load/store. **Use `bfloat` only for tape-in/tape-out with global memory**; the arithmetic in every kernel below is `float` for reductions and `half` for the dense mac path. On the M2 Ultra Studio (Apple8) the same rule applies — Apple9 (M3) added `bfloat` to `simdgroup_matrix` but the M2 Ultra does not have that path. Verify at kernel compile with `-fmetal-enable-logging` on the target hardware if uncertain.

3. **`simdgroup_matrix` is your best friend on M1 and non-negotiable on M2 Ultra.** [MSL specification §6.7 (`Metal-Shading-Language-Specification.pdf`)](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) and MLX's own [`conv.metal`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/conv.metal) show the 8×8×8 FP16/FP32 tile that gives 80 % ALU utilisation vs 25 % without. Every GEMM-shaped kernel in §3–§5 below assumes `simdgroup_matrix<float, 8, 8>` accumulation and `simdgroup_matrix<half, 8, 8>` operands unless the shape forces otherwise.

4. **Use the MLX custom-metal-kernel path (`MLXFastKernel`), not raw command buffers.** [`mlx-swift/Source/MLXFast/MLXFastKernel.swift`](https://github.com/ml-explore/mlx-swift/blob/main/Source/MLXFast/MLXFastKernel.swift) exposes just-in-time compilation of shader source strings against MLX arrays; you keep MLX's allocator, autograd, and lazy scheduling. Raw MTLCommandBuffer / MTLComputePipelineState is the Phase-1 §13.1 escape hatch — reserved for the whole-step megakernel that only lands if the profiling gate opens. None of the kernels below need to leave `MLXFastKernel`.

The pattern: **MLX kernel first, then MSL sketch, then integration test against the MLX reference at 1e-4**. Where I break that pattern below I flag it.

---

## 0. Recommended reading order

1. This §0 + §1 (naming, precision policy, calling conventions).
2. §2 (operations shared across every model — RMSNorm, RoPE, KV-scatter, top-k mask, off-by-one softmax).
3. Model-specific §3–§6 (skip the ones you're not porting).
4. §7 (integration into `Packages/DiffusionKernels`, `MLXFastKernel` wiring, tests).
5. §8 (what NOT to do — the "training wheels" list of common Metal traps).
6. §9 (uncertainty flags — where I'm speculating and what would resolve it).

If you only have 20 minutes: read §0, §2.1 (RMSNorm), §2.4 (off-by-one softmax), §8. That's the "don't-bugger-it-up" core.

---

## 1. Cross-model conventions

### 1.1 File & type layout

Every shader below lives in `Packages/DiffusionKernels/Sources/Shaders/*.metal`, one operation per file. Swift wrappers live alongside as `<Op>Kernel.swift`. The current `Packages/DiffusionKernels/Sources/DiffusionKernels.swift` stub gets replaced by the wrappers; the `checkAvailability` remains as a diagnostic.

```
Packages/DiffusionKernels/
├── Sources/
│   ├── DiffusionKernels.swift              # keep, extend with dispatch helpers
│   ├── KernelPrecision.swift               # KernelPrecision enum, Metal dtype mapping
│   ├── Shaders/
│   │   ├── rms_norm.metal
│   │   ├── rope_apply.metal
│   │   ├── rope_apply_partial.metal
│   │   ├── rope_apply_yarn.metal           # Nemotron only (data-only delta vs default)
│   │   ├── off_by_one_softmax.metal        # Sumi (and any future Miller-softmax model)
│   │   ├── off_by_one_attention.metal      # Sumi fused, once §2.4 lands
│   │   ├── top_k_mask.metal                # LMOE / Nemotron sampling policy
│   │   ├── low_confidence_remask.metal     # LLaDA-family sampling policy
│   │   ├── fused_residual_rmsnorm.metal    # every model, Phase-3 fusion
│   │   ├── swiglu_gate_up.metal            # every model (dense MLP path)
│   │   ├── routed_moe_dispatch.metal       # LMOE / LLaDA2.1-mini
│   │   └── parallel_dense_moe.metal        # DiffusionGemma
│   └── OpWrappers/
│       ├── RMSNormKernel.swift             # calls MLXFastKernel
│       ├── ...
│       └── FusedAttentionEpilogueKernel.swift
└── Tests/DiffusionKernelsTests/
    ├── RMSNormParityTests.swift            # kernel vs MLX ref at 1e-4
    ├── OffByOneSoftmaxParityTests.swift    # kernel vs formulation-A ref
    └── ...
```

`KernelPrecision` is a small enum:

```swift
public enum KernelPrecision {
    case fp32
    case fp16
    case bf16Storage  // stored bf16, computed in fp32 or fp16 — never fp bf16 on Apple7/8
    var mslType: String { switch self { case .fp32: return "float"; case .fp16: return "half"; case .bf16Storage: return "bfloat" } }
    var accumType: String { .fp32.mslType }  // reductions always fp32
}
```

### 1.2 Calling convention for `MLXFastKernel`

Every kernel below is invoked through the same shape:

```swift
public struct KernelInvocation {
    let name: String                  // matches `kernel void <name>` in the .metal file
    let inputs: [MLXArray]
    let outputShape: [Int]
    let outputDtype: DType
    let grid: (x: Int, y: Int, z: Int)
    let threadgroup: (x: Int, y: Int, z: Int)
    let templateArgs: [String: Any] = [:]   // e.g. ["HEAD_DIM": 128, "PRECISION": "half"]
    let constants: [String: Any] = [:]      // Metal function constants for specialisation
}
```

**Rule**: dispatch grid = `(numOutputTokens, numHeads, batchSize)` or the natural per-op equivalent. Threadgroup size = `(hiddenPerThread, 1, 1)` where `hiddenPerThread` divides `hiddenSize` and is ≤ `maxTotalThreadsPerThreadgroup` (query the pipeline state; on M1 = 1024, on M2 Ultra = 1024, verify per-kernel via `computePipelineState.maxTotalThreadsPerThreadgroup`).

**Never hard-code 1024.** [Apple's Metal shader-optimization guide](https://developer.apple.com/videos/play/tech-talks/111373/) is explicit: register-pressure-dependent occupancy means the effective max is often 256 or 512. The dispatch helper below reads it at runtime.

```swift
func chooseThreadgroup(pipeline: MTLComputePipelineState, preferred: Int) -> Int {
    min(preferred, pipeline.maxTotalThreadsPerThreadgroup)
}
```

### 1.3 Numerical policy (the guardrail you keep coming back to)

Follow the same rules everywhere. This is the single biggest source of "kernel matches MLX at unit test, diverges by 3 % at fixture test" bugs.

| Op class | Load dtype | Compute dtype | Reduction / accumulator | Store dtype |
|---|---|---|---|---|
| RMSNorm | model dtype (bf16/fp16/fp32) | fp32 | fp32 (Kahan not needed at H≤4096) | model dtype |
| Softmax (any) | model dtype | fp32 for `max`, `sum`, `1/denom` | fp32 | model dtype |
| Off-by-one softmax | model dtype | fp32 for all sink math | fp32 | model dtype |
| Attention scores (Q·K^T) | fp16/fp32 (per §1.1) | fp16 muladd, fp32 accum | fp32 in `simdgroup_matrix<float, 8, 8>` | fp16/fp32 (matches Q) |
| Attention output (P·V) | fp16 | fp16 muladd, fp32 accum | fp32 | model dtype |
| SwiGLU MLP | model dtype | fp16 muladd, fp32 accum | fp32 | model dtype |
| MoE router logits | model dtype | **fp32 mandatory** (Phase 1 §4.2, sourced from LLaDA2 config) | fp32 | fp32 |
| Softmax on router logits | fp32 | fp32 | fp32 | fp32 |
| Top-k / argmax over vocab | fp32 preferred (fp16 safe if V ≤ 128k and no `+∞` values) | — | fp32 comparator | int32 |
| Position ids, mask bookkeeping | int32 | int32 | int32 | int32 |
| RoPE cos/sin table | fp32 (precomputed on CPU) | fp32 | — | fp32 |

**Non-negotiable**: the softmax denominator is always fp32. On a 100 k vocab in fp16, `sum(exp(scores))` overflows past ~65 504 within a handful of tokens once `max(scores) > 10`. This is *the* bug that will bite you if you copy MLX's fp16 SDPA and add an off-by-one term.

---

## 2. Shared operations — the building blocks

Every model in `Plans/` reduces to a stack of these ops. Get the six below right and 80 % of the model-specific Metal work is done.

### 2.1 RMSNorm (all models)

**MLX baseline**: `Packages/DiffusionCore/Sources/LLaDA2RMSNorm.swift` — already correct. Only fuse into Metal if profiling shows the RMSNorm + residual-add pair is >5 % of step time.

**Kernel signature**:

```
kernel void rms_norm(
    device const T* x            [[buffer(0)]],   // [B*L, H]
    device const T* gamma        [[buffer(1)]],   // [H]
    device       T* out          [[buffer(2)]],   // [B*L, H]
    constant  float& eps         [[buffer(3)]],
    constant  uint& hidden       [[buffer(4)]],
    threadgroup float* scratch   [[threadgroup(0)]],  // size = threads_per_tg * sizeof(float)
    uint tid                     [[thread_position_in_threadgroup]],
    uint tg_id                   [[threadgroup_position_in_grid]],
    uint tg_size                 [[threads_per_threadgroup]])
```

**Sketch (template on `T`)**:

```metal
// One threadgroup per token. threads_per_tg = 128 for H ≤ 4096, 256 for H ≤ 8192.
kernel void rms_norm(...) {
    const uint row_offset = tg_id * hidden;

    // Phase 1: per-thread partial sum of squares.
    float sq = 0.0f;
    for (uint h = tid; h < hidden; h += tg_size) {
        float xv = float(x[row_offset + h]);
        sq = fma(xv, xv, sq);
    }
    scratch[tid] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: threadgroup reduction. Simdgroup shuffle first (32 lanes), then final tree.
    for (uint stride = tg_size / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Simdgroup reduce for the last 32.
    float partial = tid < 32 ? scratch[tid] : 0.0f;
    partial = simd_sum(partial);
    if (tid == 0) scratch[0] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float rms = rsqrt(scratch[0] / float(hidden) + eps);

    // Phase 3: normalise + scale + store.
    for (uint h = tid; h < hidden; h += tg_size) {
        float xv = float(x[row_offset + h]);
        float gv = float(gamma[h]);
        out[row_offset + h] = T(xv * rms * gv);
    }
}
```

**Common traps** (things that will bite you — training wheels):
- **Do not** reduce in `T` when `T = half`. `H ≥ 4096` overflows the half mantissa on the sum. Cast to `float` at load time and keep `sq` in `float`.
- **Do not** skip the `mem_threadgroup` barriers between phases. On Apple7 the SIMD group implicitly reorders within 32 lanes but not across; without the barrier you get non-deterministic reads of `scratch[tid + stride]`.
- **Do not** use `sqrt` then divide — `rsqrt` is a single hardware instruction on Apple7+; dividing by `sqrt(x)` is two ops plus a special-case for zero.
- **eps position matters**: it goes *inside* the `rsqrt` argument (`rsqrt(mean_sq + eps)`), not outside. Otherwise you diverge from PyTorch's canonical form at very small activations.

**Parity gate**: 1e-5 vs MLX `RMSNorm` on random `[8, 128, 2560]` (Sumi hidden) and `[1, 32, 2048]` (LMOE hidden). If parity fails at the 5th decimal, the `float` cast is missing somewhere.

### 2.2 RoPE apply (default + partial + YaRN as data delta)

**MLX baseline**: `Packages/DiffusionCore/Sources/PartialRotaryEmbedding.swift` — already handles default and partial. YaRN is a table-swap.

The math is identical across models — only the frequency table changes:

- **LMOE**: full rotary, `theta = 500 000` (from LMOE `config.json`).
- **LLaDA2.1-mini**: partial rotary, first 64 dims of 128, `theta = 500 000` (from Phase 2 handoff `use_qk_norm=True`).
- **Sumi**: full rotary, `theta = 500 000` (per Sumi guide §1).
- **Nemotron**: YaRN-modified `inv_freq` table with `factor=16, alpha=1, beta=32, orig_ctx=16384` (Nemotron guide §4.1). CPU precomputes the modified table; the Metal kernel is unchanged.

**Kernel signature**:

```
kernel void rope_apply(
    device       T*   qk        [[buffer(0)]],   // in-place, [B, H, L, D]
    device const float* cos_tab [[buffer(1)]],   // [L, D/2]  (or D for full rotary)
    device const float* sin_tab [[buffer(2)]],   // [L, D/2]
    constant uint&  rotary_dim  [[buffer(3)]],   // = D for full RoPE, D/2 for partial
    constant uint&  head_dim    [[buffer(4)]],
    constant uint&  seq_len     [[buffer(5)]],
    uint3 tid                   [[thread_position_in_grid]])
```

**Sketch** (in-place rotation, standard formulation — first half of the rotary segment is `x0`, second half is `x1`; output is `x0·cos - x1·sin`, `x0·sin + x1·cos`):

```metal
kernel void rope_apply(...) {
    const uint b_h = tid.z;         // batch * numHeads
    const uint l   = tid.y;         // position along sequence
    const uint d   = tid.x;         // dim pair index (0 ≤ d < rotary_dim/2)
    if (d >= rotary_dim / 2) return;

    const uint base = (b_h * seq_len + l) * head_dim + d;
    const uint pair = base + rotary_dim / 2;

    const float c = cos_tab[l * (rotary_dim / 2) + d];
    const float s = sin_tab[l * (rotary_dim / 2) + d];

    const float x0 = float(qk[base]);
    const float x1 = float(qk[pair]);

    qk[base] = T(x0 * c - x1 * s);
    qk[pair] = T(x0 * s + x1 * c);
    // Dims [rotary_dim, head_dim) are untouched (partial-rotary case).
}
```

**Training wheels**:
- **Partial-rotary layout confusion**: LLaDA2.1-mini stores `[x0..x63 rotated, x64..x127 identity]` per head. Sumi/LMOE with full rotary store `[x0..x63 rotated, x64..x127 rotated]`. If you copy a full-rotary kernel into a partial model you silently overwrite the identity half. Assert `rotary_dim ≤ head_dim` and range-check the write.
- **cos/sin table dtype**: precompute on CPU at fp32, upload as fp32 buffer. Do not cast to fp16 — the compression at head_dim 128 costs a full digit of positional fidelity.
- **YaRN mscale**: the Nemotron mscale = 1 + 0.1·log(factor) ≈ 1.277 is applied to attention *logits*, not to Q or K directly (per Nemotron guide §4.1 verbatim). Fuse it into the SDPA scale, not into RoPE. Doing it in RoPE breaks parity with the reference.
- **Positions across a block-diffusion generation**: LMOE / LLaDA2.1 use absolute positions across `[0, total_length)`, not local positions per block. Precompute the `cos_tab`/`sin_tab` once per `generate` call over the full padded length; index into it per step. Don't rebuild per step.

**Parity gate**: 1e-5 vs MLX `PartialRotaryEmbedding` on `[1, 32, 32, 128]` (LLaDA2.1-mini head shape) with `rotary_dim=64`.

### 2.3 Top-K mask (LMOE sampling, LLaDA-family low-confidence remask)

**Purpose**: given `confidence` `[B, L]` and per-row `k[B]` (values in `[0, L]`), produce a boolean mask `[B, L]` where each row has exactly `k[b]` `true`s at the top-`k[b]` positions of `confidence[b]`.

The LMOE guide §4.4 pointed out that this is the piece of `ScheduledLowConfidencePolicy` that must *not* hit CPU. MLX's argsort-then-scatter works but issues 3 kernel launches (argsort, arange broadcast, scatter). If profiling shows this as the hot spot, one fused Metal kernel replaces those 3 dispatches.

**Kernel signature**:

```
kernel void top_k_mask(
    device const float* confidence  [[buffer(0)]],   // [B, L]
    device const int*   k_per_row   [[buffer(1)]],   // [B]
    device       bool*  mask_out    [[buffer(2)]],   // [B, L]
    constant uint& L                [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]])
```

The clean implementation is a per-row bitonic sort in threadgroup memory when `L ≤ 4096` (typical: block length 32, generation length up to 2048). One threadgroup per batch row.

**Sketch** (bitonic sort, produces `sorted_indices`, then compare rank vs `k`):

```metal
constant int MAX_L = 4096;   // static upper bound; assert at compile

kernel void top_k_mask(
    device const float* confidence  [[buffer(0)]],
    device const int*   k_per_row   [[buffer(1)]],
    device       bool*  mask_out    [[buffer(2)]],
    constant uint& L                [[buffer(3)]],
    threadgroup float* conf_scratch [[threadgroup(0)]],
    threadgroup uint*  idx_scratch  [[threadgroup(1)]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tg   [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    const uint row_base = tg * L;
    const int  k        = k_per_row[tg];

    // Load into threadgroup memory.
    for (uint i = tid; i < L; i += tg_size) {
        conf_scratch[i] = confidence[row_base + i];
        idx_scratch[i]  = i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Bitonic sort descending, standard log^2 rounds.
    for (uint size = 2; size <= L; size <<= 1) {
        for (uint stride = size >> 1; stride > 0; stride >>= 1) {
            for (uint i = tid; i < L; i += tg_size) {
                const uint partner = i ^ stride;
                if (partner > i) {
                    const bool asc = ((i & size) == 0);
                    const bool swap = asc
                        ? (conf_scratch[i] < conf_scratch[partner])
                        : (conf_scratch[i] > conf_scratch[partner]);
                    if (swap) {
                        float tf = conf_scratch[i]; conf_scratch[i] = conf_scratch[partner]; conf_scratch[partner] = tf;
                        uint  ti = idx_scratch[i];  idx_scratch[i]  = idx_scratch[partner];  idx_scratch[partner]  = ti;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // idx_scratch[r] is now the index of the r-th largest element.
    // Write mask: mask_out[b, idx_scratch[r]] = (r < k)
    for (uint r = tid; r < L; r += tg_size) {
        const bool in_top_k = int(r) < k;
        mask_out[row_base + idx_scratch[r]] = in_top_k;
    }
}
```

**Training wheels**:
- **Threadgroup memory ceiling on M1 is 32 KB.** For `MAX_L = 4096`, `float + uint` = 8 bytes per slot = 32 KB. Fits *exactly*; go to `L = 8192` and you must tile. Verify `computePipelineState.threadgroupMemoryLength` at pipeline creation; if the query returns anything less than `L * 8`, fall back to the MLX two-pass argsort path. The Sumi guide §4.4 already documented that `[1, 2048, 100278]` posterior is peak-memory concern; keep this kernel to the block-length shape, not the full-canvas shape.
- **Ties**: bitonic sort is not stable. If ties matter (they do for LMOE's `num_transfer_tokens=1` argmax parity with the reference), post-process with a small stable pass over the tied region. In practice, floating confidences almost never tie at the mantissa level; document the deviation.
- **Do not read `k_per_row[tg]` in the inner loop.** Load once at the top.
- **Do not use `atomic_bool` on the output.** `bool` writes are non-conflicting across threads because each `r` maps to a unique `idx_scratch[r]`; there's no race.

**Parity gate**: exact match vs MLX `argSort(-conf) [.stride(), ..<k]` on random `[1, 256]` and `[1, 2048]` inputs; ties handled by the ref's stable-sort semantics — accept 0 % row-level mismatch on tie-free inputs, log the tie-mismatch rate on random-integer confidences.

### 2.4 Off-by-one softmax (Sumi, and any future Miller-softmax model)

**Purpose**: implement Evan Miller's [`softmax_1(x) = exp(x_i) / (1 + Σ exp(x_j))`](https://www.evanmiller.org/attention-is-off-by-one.html) — the "attention sink" variant. Sumi's `modeling_sumi.py` uses formulation A (prepend a zero-logit sink); the multiplicative form B (Sumi guide §4.1) is `softmax(x)_i × Z/(Z+1)`.

The two forms are mathematically identical. Empirically, at fp32 the discrepancy is ε ≲ 1e-6 (Sumi guide §4.1 — verified there). **We ship formulation B in Metal.** Reasons:
- No extra allocation for the sink column.
- Fits the standard flash-attention online-softmax structure — one extra scalar in the running normaliser, nothing else.

**Structural insight**: in a numerically stable softmax we already track `m = max(x)` and `Z = Σ exp(x_j - m)`. The off-by-one variant just replaces the denominator `Z` with `Z + exp(-m)`. That's it. Everything else in the kernel is identical to standard softmax.

**Standalone kernel** (for correctness testing, not the fused path):

```metal
kernel void off_by_one_softmax_rowwise(
    device const T* logits   [[buffer(0)]],   // [B*rows, cols]
    device       T* probs    [[buffer(1)]],   // [B*rows, cols]
    constant uint& cols      [[buffer(2)]],
    threadgroup float* scratch [[threadgroup(0)]],   // size = tg_size floats
    uint tid    [[thread_position_in_threadgroup]],
    uint tg_id  [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    const uint row = tg_id * cols;

    // Phase 1: row max.
    float local_max = -INFINITY;
    for (uint c = tid; c < cols; c += tg_size) {
        float lv = float(logits[row + c]);
        local_max = fmax(local_max, lv);
    }
    scratch[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) scratch[tid] = fmax(scratch[tid], scratch[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float m = tid < 32 ? scratch[tid] : -INFINITY;
    m = simd_max(m);
    if (tid == 0) scratch[0] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = scratch[0];

    // Phase 2: row sum of exp(x_j - m).
    float local_sum = 0.0f;
    for (uint c = tid; c < cols; c += tg_size) {
        local_sum += exp(float(logits[row + c]) - m);
    }
    scratch[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tg_size / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float z = tid < 32 ? scratch[tid] : 0.0f;
    z = simd_sum(z);

    // ============ THE SINGLE LINE THAT DIFFERS FROM STANDARD SOFTMAX ============
    // Add the sink probability mass: exp(0 - m) = exp(-m). Softmax_1 denominator.
    if (tid == 0) scratch[0] = z + exp(-m);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float denom = scratch[0];
    // ===========================================================================

    // Phase 3: write probs.
    const float inv_denom = 1.0f / denom;
    for (uint c = tid; c < cols; c += tg_size) {
        probs[row + c] = T(exp(float(logits[row + c]) - m) * inv_denom);
    }
}
```

**Training wheels — the ones specific to off-by-one**:
- **The sink term is `exp(-m)`, not `exp(0)`.** The `1` in `1 + Σ exp(x_j)` is the *unshifted* zero-logit sink. Once you shift by `m` for numerical stability, the sink shifts too: `1·exp(-m) = exp(-m)`. Getting this wrong gives you standard softmax and a silent divergence from the reference at ~30 % relative error whenever `m ≫ 0` (i.e. always, in an attention layer).
- **Underflow of `exp(-m)` when `m > 88` at fp32 (`m > ~11` at fp16).** For very confident heads the sink term underflows to zero and off-by-one becomes standard softmax. That is *correct behaviour* — Evan Miller's argument is that the sink matters when the head wants to output near-zero, i.e. `m` is small or negative. But do not "fix" the underflow by clamping `m`; that breaks numerical stability for other tokens.
- **Fp16 forbidden for the reductions.** Sumi's vocab is 100 278 (Sumi guide §1); a row-sum of exp values easily exceeds `half_max = 65 504`. This kernel *must* keep `m`, `z`, `denom` in fp32.

**Fused off-by-one attention** (§5 of the Sumi guide, verbatim structural note): take any flash-attention forward kernel and change the online-softmax update from

```
new_max = max(m_prev, m_new)
z_new = z_prev · exp(m_prev - new_max) + Σ exp(x_j - new_max)
```

to

```
new_max = max(m_prev, m_new)
z_new = z_prev · exp(m_prev - new_max) + Σ exp(x_j - new_max)
z_with_sink = z_new + exp(-new_max)     // <-- the only difference
p_j = exp(x_j - new_max) / z_with_sink  // epilogue: divide by z_with_sink, not z_new
```

**Do not** compute `z_with_sink` inside the tile loop and update it incrementally — the sink `exp(-new_max)` depends only on the final `new_max`, so it goes in the epilogue after the last tile. Adding it per-tile double-counts.

**Parity gate**:
- vs formulation A (`softmaxOne` in Sumi guide §4.1) on random `[1, 16, 32, 32]` (small): 1e-6 at fp32.
- vs standard `softmax(x) · Z/(Z+1)` reduction (formulation B) on same shape: bit-exact.
- vs the reference PyTorch `softmax_one` on Sumi's inference fixture: 1e-4 (the 1e-4 is bounded by the fp16 storage of Q/K, not by the softmax itself).

### 2.5 SwiGLU MLP (dense path, every model)

**MLX baseline**: SwiGLU is `down(silu(gate(x)) * up(x))`. MLX's fused `silu` op with the pointwise multiply is already efficient. **Only fuse in Metal if profiling shows the intermediate `[B*L, intermediate]` buffer is a memory-bandwidth pressure point.** That happens on Sumi (intermediate 12288 at Sumi hidden 2560) and Nemotron (intermediate 9216).

**Kernel signature** — fused `gate + up + silu + mul`, does NOT include the `down_proj`:

```
kernel void swiglu_gate_up(
    device const T* x            [[buffer(0)]],  // [B*L, H]
    device const T* w_gate       [[buffer(1)]],  // [I, H]
    device const T* w_up         [[buffer(2)]],  // [I, H]
    device       T* out          [[buffer(3)]],  // [B*L, I]
    constant uint& H             [[buffer(4)]],
    constant uint& I             [[buffer(5)]],
    ...
)
```

The GEMM lives in MLX (or its `quantized_matmul` for the 4-bit path); the fused kernel here is the epilogue: read `gate_out[i]` and `up_out[i]`, compute `silu(gate_out[i]) * up_out[i]`, write. This is a pointwise op, so it's memory-bound; the fusion win is that you avoid materialising `gate_out` and `up_out` as separate tensors.

**Better structure**: don't split gate and up projections at all — concatenate the weights on load into a single `[2I, H]` matrix, do one GEMM to `[B*L, 2I]`, then a single pointwise kernel splits and combines:

```metal
kernel void swiglu_epilogue(
    device const T* gate_up_concat  [[buffer(0)]],  // [B*L, 2I]  first I = gate, second I = up
    device       T* out             [[buffer(1)]],  // [B*L, I]
    constant uint& I                [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    const uint row = tid / I;
    const uint col = tid % I;
    const uint base = row * 2 * I;
    const float g = float(gate_up_concat[base + col]);
    const float u = float(gate_up_concat[base + I + col]);
    const float silu_g = g / (1.0f + exp(-g));   // sigmoid form of silu is more stable at large |g|
    out[row * I + col] = T(silu_g * u);
}
```

**Training wheels**:
- **`silu(x) = x · sigmoid(x)` is the correct definition; do not use `swish(x) = x · sigmoid(β·x)` with `β ≠ 1`.** No model in `Plans/` uses β ≠ 1; hardcoding it prevents future confusion.
- **`silu` at large negative `x` should return `≈ 0`.** The naive form `x / (1 + exp(-x))` is fine for negatives; the fp16 overflow risk is at very large positives (`exp(88+) = ∞` at fp32, `exp(11+) = ∞` at fp16). Since `silu(x) → x` as `x → ∞`, guard: `if (x > 20) return T(x * u);` — this avoids the fp32 pipeline hazard and matches PyTorch's `F.silu` behaviour at scale.
- **`intermediate_size` per model**: LMOE 1024 per expert (routed) / LLaDA2.1-mini 512 per expert / Sumi 12288 dense / Nemotron 9216 dense / DiffusionGemma 2112 dense + 704 per expert. Never hardcode; pass as a function constant.
- **Weight-concatenation on load** requires a matching change in the safetensors loader (see `Tools/convert_weights.py`). Do this once at load, not per step.

### 2.6 KV cache write (all block-diffusion models)

**Purpose**: after the commit-forward at the end of each block (per `handoff-post-M5.md` §4 speculation-vs-cache-commit note), write the committed block's K and V into the layer's `LayerKVCache` at the correct positions. This is a scatter along the sequence axis.

Under LLaDA2.1-mini (block-causal + ExactPrefixCache = exact), this is the only KV mutation per block. Under LMOE default (bidirectional, ExactPrefixCache = approximate per LMOE guide §3.4), a full recomputation happens; the "cache write" is really "cache replace".

**MLX baseline**: `MLXArray.scatterAlong` or a plain slice assignment. Fast enough — this is not a hot path. **Only sketch here for completeness; do not write Metal for this unless the LLaDA-2 kernel-fusion track (`phase-3.md` §4 item 3) proves the selection-set + KV-scatter combined dispatch beats MLX.**

Skipped for brevity — MLX baseline stays.

### 2.7 Fused residual + RMSNorm (post-attention, post-MLP)

**Purpose**: the pattern `y = norm(x + h)` where `x` is the residual and `h` is the layer output. On every layer, every step, this fires twice (post-attention, post-MLP). Reading `x` once for the add and once for the norm is wasteful; a fused kernel reads it once.

**MLX baseline**: separate `add` + `rms_norm`, well-optimised. **Only fuse if profiling shows the extra global memory read of `x` is >3 % of step time.** On Apple Silicon UMA this is *usually not the case* — the read comes from cache, not from DRAM. Do the measurement first.

**Sketch** — combine §2.1's RMSNorm with the residual read:

```metal
kernel void fused_residual_rmsnorm(
    device const T* residual     [[buffer(0)]],   // [B*L, H]
    device const T* h_layer      [[buffer(1)]],   // [B*L, H]
    device       T* residual_out [[buffer(2)]],   // [B*L, H] — write x + h back for the next layer
    device       T* normed_out   [[buffer(3)]],   // [B*L, H] — normalised feed to MLP or attention
    device const T* gamma        [[buffer(4)]],   // [H]
    constant float& eps          [[buffer(5)]],
    constant uint& hidden        [[buffer(6)]],
    ...)
{
    // Phase 1: compute sum-of-squares of (residual + h_layer) in fp32.
    // Simultaneously write the sum to residual_out so the next layer reads it once.
    ...
    // Phase 2: reduce, compute rms.
    ...
    // Phase 3: write normed = (residual + h) * rms * gamma to normed_out.
    ...
}
```

The structure is exactly §2.1's RMSNorm with three storage stores per element instead of one. **Cost**: 1 extra buffer write per element (write `x + h` back). **Benefit**: skip a full 4-byte-per-hidden pass over global memory.

**Training wheels**:
- **Do NOT skip the `residual_out` write** thinking "MLX will hold the fused sum lazily". The next layer's attention or MLP reads it as a first-class MLX array; if you don't materialise it, MLX will insert a duplicate `add` op elsewhere and you lose the fusion.
- **eps and gamma** are per-model. LLaDA2.1-mini and LMOE use `rms_norm_eps = 1e-5`; Sumi uses `1e-6` (Sumi guide §1); Nemotron `1e-5` (Nemotron guide §1). Always pass eps as a runtime constant.

---

## 3. LLaDA2.1-mini & LMOE — routed MoE + partial/full RoPE

### 3.1 What's specific to write

**Nothing beyond §2** during Phase 2 M4/M5. Phase 3 kernel-fusion track (per `phase-3.md` §4) opens up:

1. **Fused routed-MoE dispatch** — LMOE guide §5.1 sketched the shape. Rewriting here with concrete Metal:

```
// Shape recap:
//   x            [T, H]      T = tokens in the flat batch, H = 2048 (LMOE) or 2048 (LLaDA2.1)
//   expert_idx   [T, K]      K = 8 (LMOE)
//   expert_wgt   [T, K]      router weights (softmax-normalised or raw per model)
//   W_gate[e]    [I, H]      one per expert; I = 1024 (LMOE) or 512 (LLaDA2.1)
//   W_up[e]      [I, H]
//   W_down[e]    [H, I]
//   out          [T, H]      accumulator

kernel void routed_moe_dispatch(
    device const T*   x             [[buffer(0)]],
    device const int* expert_idx    [[buffer(1)]],
    device const float* expert_wgt  [[buffer(2)]],
    device const T*   w_gate_all    [[buffer(3)]],   // [E, I, H]
    device const T*   w_up_all      [[buffer(4)]],
    device const T*   w_down_all    [[buffer(5)]],   // [E, H, I]
    device       T*   out           [[buffer(6)]],
    constant uint& T_tokens         [[buffer(7)]],
    constant uint& H                [[buffer(8)]],
    constant uint& I                [[buffer(9)]],
    constant uint& K                [[buffer(10)]],
    threadgroup float* scratch      [[threadgroup(0)]])
```

The kernel structure the LMOE guide sketched was pseudo-Metal; here is the *concrete* structure. One threadgroup per token; the threadgroup iterates over the K experts serially:

```metal
kernel void routed_moe_dispatch(...) {
    const uint t = threadgroup_position_in_grid;   // token
    const uint tid = thread_position_in_threadgroup;
    const uint tg_size = threads_per_threadgroup;

    // Threadgroup memory for the K-expert accumulation.
    // gate_out [I], up_out [I], acc [H]
    threadgroup float gate_out[MAX_INTER];   // MAX_INTER = 1024 for LMOE, template param
    threadgroup float up_out[MAX_INTER];
    threadgroup float acc[MAX_HIDDEN];       // MAX_HIDDEN = 2048, template param

    // Zero acc
    for (uint h = tid; h < H; h += tg_size) acc[h] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 0; k < K; ++k) {
        const int   e     = expert_idx[t * K + k];
        const float w_e   = expert_wgt[t * K + k];

        // W_gate[e] · x[t]   →   gate_out
        // W_up[e]   · x[t]   →   up_out
        // Both matmuls: [I, H] · [H] → [I]
        for (uint i = tid; i < I; i += tg_size) {
            float g = 0.0f, u = 0.0f;
            for (uint h = 0; h < H; ++h) {
                const float xv = float(x[t * H + h]);
                g = fma(float(w_gate_all[e * I * H + i * H + h]), xv, g);
                u = fma(float(w_up_all  [e * I * H + i * H + h]), xv, u);
            }
            const float silu_g = g / (1.0f + exp(-g));
            gate_out[i] = silu_g * u;                 // SwiGLU in-place
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // W_down[e] · gate_out  →   acc (weighted by w_e)
        // Matmul: [H, I] · [I] → [H]
        for (uint h = tid; h < H; h += tg_size) {
            float a = 0.0f;
            for (uint i = 0; i < I; ++i) {
                a = fma(float(w_down_all[e * H * I + h * I + i]), gate_out[i], a);
            }
            acc[h] = fma(w_e, a, acc[h]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store back.
    for (uint h = tid; h < H; h += tg_size) {
        out[t * H + h] = T(acc[h]);
    }
}
```

**Training wheels — routed MoE**:
- **Threadgroup memory ceiling**: gate_out (I = 1024) × 4 = 4 KB + up_out 4 KB (only needed if not fusing SwiGLU into gate_out) + acc (H = 2048) × 4 = 8 KB. If you fuse `up_out` into `gate_out` (compute `silu_g * u` immediately and store to a single buffer), you save 4 KB. **Do it**; 12 KB total leaves headroom under the M1's 32 KB threadgroup memory. For LLaDA2.1-mini with I = 512 the numbers halve.
- **The inner GEMM loop is compute-bound at this scale**, not memory-bound — `w_gate[e]` is `[1024, 2048]` = 8 MB at fp16, does not fit L2 (M1 = 8 MB L2 on the GPU, but shared with rest of the workload). The load pattern is critical: **read `x[t]` once, iterate `i` in the outer loop, `h` in the inner loop** so that each thread's `x[t, h]` sits in registers across `I` iterations. The sketch above does this.
- **4-bit weights**: LMOE guide §7 specifies routed experts as 4-bit MLX-affine, group_size=64. The kernel above assumes dequantised weights in `w_gate_all[e * I * H + ...]`. Rewriting for 4-bit-in-place is a substantial change — **do not attempt in the first landing**. Land the fp16 fused kernel first, benchmark, then decide whether to rewrite for `q4` operands. MLX's `gather_qmm` already handles the 4-bit routed dispatch efficiently; the fused kernel's win is *dispatch reduction*, not *quant elimination*.
- **`E`, `I`, `K` as function constants**, not runtime constants. Metal specialises the pipeline per (E, I, K) tuple, unrolling the inner loops.

2. **Attention epilogue fusion** — §2.7 covers the shape; nothing model-specific.

### 3.2 What NOT to write in Metal for these models (verbatim from LMOE §5.3, restated for clarity)

- Block-diffusion mask kernel — `BlockDiffusionMask.strict` is already efficient in MLX.
- Fused softmax+top-K router — router argsort over 64 (LMOE) or 256 (LLaDA2.1-mini) experts is fast enough in MLX; the Phase-3 selection-set-fusion (§3.1 above) is a bigger win.
- Anything FP8-shaped — Apple Silicon does not have FP8.

---

## 4. DiffusionGemma — parallel dense+MoE

### 4.1 Structural note (from `diffusiongemma-implementation-guide.md` §4.3, §5.1)

Each layer runs a dense MLP *and* a routed MoE in parallel, then sums. The MoE routes to 8-of-128 experts at intermediate 704; the dense MLP has intermediate 2112. **Do not** fuse dense + MoE into a single megakernel — dispatch them on separate MLX streams (Phase 1's stream separation) and let the M2 Ultra's dual-command-processor pipeline them. Custom Metal here is a Phase-3 speculative win.

### 4.2 What's specific to write

**One kernel worth writing** if profiling justifies: `parallel_dense_moe_reduce` — the final `+` step that sums dense_out and moe_out. Trivial pointwise op; only useful because it fuses with the following residual add. **Skip until profiling.**

**One kernel that ports unchanged from §3**: routed MoE dispatch. Only shape param changes (I = 704, K = 8, E = 128).

**Self-conditioning matmul** (DiffusionGemma guide §5.3): `[B, 256, 262144] @ [262144, 2560]` per denoising step. This is a *very wide* matmul against the embedding matrix. MLX will dispatch this as its optimised `matmul`; a custom kernel here has to beat the MLX GEMM at a shape MLX has tuned for. **Do not write; measure first.** The DiffusionGemma guide's own §5.4 concurs.

### 4.3 What NOT to write in Metal for DiffusionGemma

- Fused encoder-decoder switch — it's a Swift branch, not a kernel.
- Router — same as §3.
- Sampler — entropy calculation over the canvas is fine in MLX.

---

## 5. Sumi — off-by-one attention, dense-only

### 5.1 What's specific to write

**One kernel is worth writing by hand for Sumi.** Everything else ports from §2.

**Fused off-by-one attention.** The Sumi guide §5.2 sketched it; here's the concrete structure. This is a flash-attention-style tile-based kernel with the single line difference from §2.4 (`z_with_sink = z + exp(-m)` at the epilogue).

```metal
// One threadgroup per (batch, head, query_tile). Query tile size Bq = 16 or 32.
// Iterate over K/V tiles Bkv = 32 or 64.
kernel void off_by_one_attention(
    device const T* Q            [[buffer(0)]],   // [B, H, S, D]
    device const T* K            [[buffer(1)]],
    device const T* V            [[buffer(2)]],
    device       T* O            [[buffer(3)]],   // [B, H, S, D]
    device const T* mask         [[buffer(4)]],   // [S, S] or null
    constant float& inv_sqrt_d   [[buffer(5)]],
    constant uint& S             [[buffer(6)]],
    constant uint& D             [[buffer(7)]],
    threadgroup float* q_tile    [[threadgroup(0)]],   // [Bq, D]
    threadgroup float* k_tile    [[threadgroup(1)]],   // [Bkv, D]
    threadgroup float* v_tile    [[threadgroup(2)]],   // [Bkv, D]
    threadgroup float* scores    [[threadgroup(3)]],   // [Bq, Bkv]
    threadgroup float* m_running [[threadgroup(4)]],   // [Bq]
    threadgroup float* z_running [[threadgroup(5)]],   // [Bq]
    threadgroup float* o_running [[threadgroup(6)]],   // [Bq, D]
    uint3 tid                    [[thread_position_in_threadgroup]])
{
    // Load Q tile for this threadgroup.
    // Initialise m_running = -inf, z_running = 0, o_running = 0.

    for (uint kv_tile = 0; kv_tile < S; kv_tile += Bkv) {
        // Load K tile, V tile.
        // Compute scores = q_tile · k_tile^T * inv_sqrt_d, add mask if any.
        // Online softmax update:
        //   m_new = max(m_running, rowmax(scores))
        //   scale = exp(m_running - m_new)
        //   z_new = z_running * scale + rowsum(exp(scores - m_new))
        //   o_running = o_running * scale + exp(scores - m_new) @ v_tile
        //   m_running = m_new; z_running = z_new;
    }

    // ============ EPILOGUE — the single off-by-one line ============
    // z_with_sink = z_running + exp(-m_running)   (elementwise per query row)
    // O_row = o_running / z_with_sink
    // ================================================================
}
```

**Training wheels — flash-attention on Metal**:
- **`simdgroup_matrix<float, 8, 8>` for the `q · k^T` and `p · v` matmuls.** MSL's `simdgroup_load` + `simdgroup_multiply_accumulate` is the FA idiom on Apple GPUs. Cite: [`Metal-Shading-Language-Specification.pdf` §7.7](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf), and [MLX's `steel_attention` kernel](https://github.com/ml-explore/mlx/tree/main/mlx/backend/metal/kernels) as the closest reference implementation to crib from.
- **Bq · D ≤ 4 KB and Bkv · D ≤ 4 KB in fp32** to keep threadgroup memory manageable. For Sumi head_dim=128: Bq=16, Bkv=32 fits well.
- **Never store the full `[Bq, Bkv]` scores matrix in fp16.** Softmax stability requires fp32 for `scores`.
- **The sink term is applied ONCE at the epilogue**, not per K/V tile. If you re-add it per tile, you get `z + K_tiles · exp(-m)` which is wrong.

Everything else in Sumi (RMSNorm, RoPE with `theta=500000`, dense SwiGLU MLP) is §2.

### 5.2 What NOT to write in Metal for Sumi

Per Sumi guide §5.3, restated:
- No routed-MoE kernel — dense model.
- No block-diffusion mask kernel — uniform-state attention has no block structure.
- No mask-position dispatch — no mask token in Sumi's uniform-state vocab.
- No self-conditioning kernel — Sumi does not use self-conditioning.
- No sliding-window kernel — Sumi is full-attention throughout.
- Fused Gumbel-max sampler — defer until attention no longer dominates (per Sumi guide §5.3, unlikely).

---

## 6. Nemotron — YaRN RoPE + Q-scaling + tri-mode attention

### 6.1 What's specific to write

**Two data-only deltas** (per Nemotron guide §5.2):

1. **YaRN-modified inv_freq buffer.** Same §2.2 RoPE kernel; different `cos_tab`/`sin_tab` computed on CPU at load time per Nemotron guide §4.1.

2. **Llama-4 Q-scaling.** Multiply Q by a position-dependent scalar. Nemotron guide §4.2 recommends a *separate* 1-line kernel after Q-projection; I concur — folding into RoPE is a false economy on M1 (adds a per-token constant to a kernel that already reads per-position tables).

```metal
kernel void q_scale_apply(
    device       T* q            [[buffer(0)]],   // [B, H, S, D]
    device const float* mult     [[buffer(1)]],   // [S]  precomputed on CPU
    constant uint& S             [[buffer(2)]],
    constant uint& D             [[buffer(3)]],
    uint3 tid                    [[thread_position_in_grid]])
{
    const uint s = tid.y;
    const uint d = tid.x;
    if (d >= D || s >= S) return;
    const uint idx = tid.z * S * D + s * D + d;    // z = batch * numHeads
    q[idx] = T(float(q[idx]) * mult[s]);
}
```

**Training wheels — Nemotron-specific**:
- **`mult[s]` must be computed once per generation**, uploaded, not recomputed per step. Position 0 = 1.0 exactly (Nemotron guide §4.2 verified). If your `mult[0] != 1.0`, the CPU-side formula is wrong (check the `floor(position / orig_max)` — at position 0, `floor(0/16384) = 0`, `log(1+0) = 0`, `mult = 1`).
- **Do not apply Q-scaling in AR verify mode of `linear_spec`** unless the reference does. The Nemotron guide §4.5 sampler wraps this in a `qScale: Llama4QScale?` — a nullable — for exactly this reason.

**Attention mode dispatch**: same kernel for causal and bidirectional (Nemotron guide §5.2 explicitly notes this). Only the mask tensor differs. The mask builder (Nemotron guide §4.3) is a Swift decision, not a kernel.

### 6.2 What NOT to write in Metal for Nemotron

Per Nemotron guide §5.3, restated:
- No fused block-diffusion sampler.
- No LoRA-adapter mergemat kernel (rank-16/32; two small kernels suffice).
- No custom `_get_transfer_index`. MLX default is fine.

**Nemotron guide §5.4 also warns off FlashAttention MPS variants on M1** — the M1 GPU family does not have the shared-memory shape those exploit. On M2 Ultra (the Studio), MPS FA is worth *measuring* but not writing to; Apple's MPS team will have tuned it better than a from-scratch port.

---

## 7. Integration: `Packages/DiffusionKernels` and `MLXFastKernel`

### 7.1 The current state (as of this doc)

`Packages/DiffusionKernels/Sources/DiffusionKernels.swift` is a 13-line stub with only `checkAvailability()`. That is fine — **the package should stay empty until §7.2 lands its first real kernel**. Do not scaffold code that has no tests behind it.

### 7.2 First landing: `RMSNormKernel` (the smallest useful kernel)

The order of operations to land the first real kernel:

1. Add `Packages/DiffusionKernels/Sources/Shaders/rms_norm.metal` (contents from §2.1).
2. Add `Packages/DiffusionKernels/Sources/OpWrappers/RMSNormKernel.swift`:

```swift
import Foundation
import MLX
import MLXFast

public enum RMSNormKernel {
    private static let source: String = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void rms_norm_half(...) { /* §2.1 contents, T = half */ }
    kernel void rms_norm_float(...) { /* §2.1 contents, T = float */ }
    """

    public static func apply(_ x: MLXArray, gamma: MLXArray, eps: Float) -> MLXArray {
        precondition(x.dim(-1) == gamma.dim(0), "hidden size mismatch")
        let hidden = x.dim(-1)
        let rows = x.size / hidden

        let kernelName: String
        switch x.dtype {
        case .float16: kernelName = "rms_norm_half"
        case .float32: kernelName = "rms_norm_float"
        case .bfloat16:
            // Metal has no bf16 compute on M1/M2. Upcast, compute, downcast.
            let promoted = x.asType(.float16)
            let normed = apply(promoted, gamma: gamma.asType(.float16), eps: eps)
            return normed.asType(.bfloat16)
        default:
            fatalError("Unsupported dtype: \\(x.dtype)")
        }

        let kernel = MLXFast.metalKernel(
            name: kernelName,
            source: source,
            inputNames: ["x", "gamma", "eps", "hidden"],
            outputNames: ["out"],
            outputShapes: [x.shape],
            outputDTypes: [x.dtype],
            grid: (Int32(rows), 1, 1),
            threadGroup: (128, 1, 1),           // adjust per pipeline max
            templateArguments: [:],
            initValue: 0
        )
        return kernel([x, gamma], scalars: [eps, Float(hidden)]).first!
    }
}
```

(The exact `MLXFast.metalKernel` signature is what mlx-swift 0.31+ exposes; consult [`mlx-swift/Source/MLXFast/MLXFastKernel.swift`](https://github.com/ml-explore/mlx-swift/blob/main/Source/MLXFast/MLXFastKernel.swift) at the pinned version. If the API drifts, this wrapper is the only file that needs to change.)

3. Add `Tests/DiffusionKernelsTests/RMSNormParityTests.swift`:

```swift
import XCTest
import MLX
@testable import DiffusionKernels
import DiffusionCore   // for LLaDA2RMSNorm reference

final class RMSNormParityTests: XCTestCase {
    func testMatchesMLXReference_smallShapes() {
        let x = MLXRandom.normal([8, 128, 2048])
        let gamma = MLXRandom.normal([2048])
        let refNorm = LLaDA2RMSNorm(hidden: 2048, eps: 1e-5)
        refNorm.gamma = gamma

        let ref = refNorm(x)
        let kernel = RMSNormKernel.apply(x, gamma: gamma, eps: 1e-5)

        let diff = (ref - kernel).abs().max().item(Float.self)
        XCTAssertLessThan(diff, 1e-4, "kernel vs MLX ref diff = \\(diff)")
    }

    func testMatchesMLXReference_shapesPerModel() {
        // LLaDA2.1-mini hidden 2048, LMOE 2048, Sumi 2560, Nemotron 3072, DiffusionGemma 2560.
        for hidden in [2048, 2560, 3072] {
            let x = MLXRandom.normal([1, 32, hidden])
            let gamma = MLXRandom.normal([hidden])
            let refNorm = LLaDA2RMSNorm(hidden: hidden, eps: 1e-5)
            refNorm.gamma = gamma
            let diff = ((refNorm(x)) - RMSNormKernel.apply(x, gamma: gamma, eps: 1e-5)).abs().max().item(Float.self)
            XCTAssertLessThan(diff, 1e-4, "hidden=\\(hidden) diff=\\(diff)")
        }
    }

    func testBF16Storage_promotesAndReturnsBF16() {
        let x = MLXRandom.normal([1, 32, 2048]).asType(.bfloat16)
        let gamma = MLXRandom.normal([2048]).asType(.bfloat16)
        let out = RMSNormKernel.apply(x, gamma: gamma, eps: 1e-5)
        XCTAssertEqual(out.dtype, .bfloat16)
        // Compare against fp32 reference on the promoted values.
        let refFP32 = LLaDA2RMSNorm(hidden: 2048, eps: 1e-5)
        refFP32.gamma = gamma.asType(.float32)
        let ref = refFP32(x.asType(.float32))
        let diff = (ref - out.asType(.float32)).abs().max().item(Float.self)
        XCTAssertLessThan(diff, 3e-3, "bf16 storage path diff = \\(diff)")
    }
}
```

Everything after `RMSNormKernel` follows the same pattern: one `.metal` file, one Swift wrapper, one parity test. **Do not** land a kernel without the test. **Do not** land a test that only checks the kernel against itself — always compare to the MLX reference on the same inputs.

### 7.3 Wiring into DiffusionCore

`LLaDA2RMSNorm.callAsFunction` stays as-is (fine-grained MLX ops). Add a feature flag:

```swift
public class LLaDA2RMSNorm: Module {
    public static var useKernel: Bool = false   // toggled by bench harness, not tests

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        if Self.useKernel {
            return RMSNormKernel.apply(x, gamma: weight, eps: eps)
        } else {
            // existing MLX implementation
        }
    }
}
```

The bench harness (`Tools/diffusion-bench`) flips the flag per run and reports the delta.

### 7.4 What lands in what order (revised Phase-3 kernel track)

Given the model-guide-specific §5 sections and this doc's §2–§6, the actionable landing order for Phase 3's kernel track (`phase-3.md` §4) is:

1. **RMSNorm** (§2.1) — measure on M1 first. If <5 % of step time, close as "no fusion needed".
2. **Off-by-one softmax** (§2.4) — required for Sumi, no MLX equivalent. Land as a *correctness* item, not an *optimisation* item.
3. **Fused residual + RMSNorm** (§2.7) — measure on M2 Ultra first. This is the cross-cutting attention-epilogue fusion (`phase-3.md` §4 item 2).
4. **Top-K mask** (§2.3) — only if bench shows `low_confidence_remask` selection dispatch is a hot spot on M2 Ultra at long generation lengths.
5. **Off-by-one fused attention** (§5.1) — only after (2) proves the standalone kernel matches parity, and only if Sumi's attention becomes the dominant step cost.
6. **Routed MoE dispatch** (§3.1) — only after (3) lands and the M2 Ultra Instruments trace confirms MoE dispatch >40 % of step (per LMOE guide §5.1).
7. **Q-scaling for Nemotron** (§6.1) — 30-line kernel; land if and when Nemotron is being served.

Everything else (routed MoE for 4-bit, self-conditioning fused matmul, whole-step megakernel) stays in the "raw-Metal escape hatch" bucket (`phase-1.md` §13.1). Do not open that hatch until Phase 3's measured baseline says the escape gate is triggered.

---

## 8. Training wheels — the cross-cutting "don't-bugger-it-up" list

The traps that will silently bite you on Apple GPUs, distilled from every §5 in the model guides plus what I found verifying MSL specs for this doc. In descending order of "this has cost me a day":

1. **BF16 is storage-only on M1 & M2.** Anywhere you write `bfloat` in an MSL kernel and *actually rely* on arithmetic in that type, you get FP16-emulated math with silent precision loss. See §0 point 2. Upcast to `float` or `half` before compute; downcast at store.

2. **Softmax denominators in fp16 overflow above `max(scores) ≈ 10`.** Non-negotiable rule: `m`, `z`, `denom` are `float` regardless of input dtype. Applies to standard softmax, off-by-one softmax, and every online-softmax variant. See §1.3.

3. **The off-by-one sink is `exp(-m)`, not `1`, once you shift for numerical stability.** See §2.4. This is a 30 % relative-error bug, not a rounding-noise bug.

4. **Threadgroup memory ceiling on M1 GPU is 32 KB.** If a kernel needs more, tile — do not silently cap at 16 KB "just in case", and do not assume 64 KB (that's M3+). Query via `computePipelineState.threadgroupMemoryLength` at pipeline creation.

5. **Metal `simdgroup_matrix` is 8×8×8 only, and only fp16/fp32 operands on Apple7/8.** No 16×16, no bf16, no int8. Anything else is a software loop. See [`MSL specification`](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) §6.7.

6. **`maxTotalThreadsPerThreadgroup` varies with register pressure.** Never hardcode 1024. Query per pipeline. A `float scratch[1024]` in threadgroup memory + 32 registers per thread frequently drops the max to 512. See §1.2.

7. **Grid dispatch shape matters.** MPP guide (`Metal-Performance-Primitives-Programming-Guide.pdf`) recommends 1-D grid with a Morton-ordered map inside the kernel for cache locality. For our matmul-heavy kernels, a 3-D `(tokens, heads, batch)` grid is fine — MLX's own kernels use exactly that shape — but do not use a 2-D grid where a 3-D one naturally exists. The threadgroup scheduler is smarter with the higher-dim grid.

8. **Barriers: `mem_flags::mem_threadgroup` between threadgroup-memory phases, `mem_flags::mem_device` before writing back to global.** Skipping the barrier gives you race-condition bugs that only trigger under load; not a compile error, not a validator error.

9. **`simd_sum` / `simd_max` reduce within a 32-lane simdgroup only.** For threadgroup-wide reductions, do simdgroup reduce first (fast), then a small threadgroup-memory tree over the simdgroup leaders. Every reduction sketch above follows this pattern.

10. **On M1, `divide` and `%` by non-literal denominators are unusually expensive.** Prefer bit-shifts (`>> 1`, `& (n-1)`) for power-of-two arithmetic. Prefer precomputing reciprocals on CPU (`inv_denom`) for anything divided repeatedly. Apple's WWDC16 [Advanced Metal Shader Optimisation](https://developer.apple.com/la/videos/play/wwdc2016/606/) is old but still exactly correct on this point for Apple7/8/9.

11. **Do NOT reuse the same `MLXFastKernel` handle across dtype specialisations.** MLX compiles per-source-string; if you template on `T` and stringify differently per dtype, you need a distinct handle per dtype. Cache the compiled pipeline per `(kernelName, dtype)` tuple.

12. **`MLXFast.metalKernel` requires `initValue` for the output buffer.** Even if the kernel writes every output element, the initValue must be provided (default 0). If the kernel has a conditional early exit, the initValue is what un-written elements will contain — make sure that's what you want.

13. **Do not use `atomic_float` on Apple7.** Atomic ops on floats are Apple8+ (M2+). If you need cross-simdgroup accumulation into a float, either promote to `atomic_int` bitcast trick (fragile) or restructure to per-simdgroup accumulators + a final threadgroup reduce (correct).

14. **`bfloat` cast round-trips through `float`.** Reading a `bfloat` in MSL is `float(bfloat_val)` which triggers the 16→32 exchange. Do this once at load, keep the value in `float`. Do NOT convert to `half` in between; that costs mantissa bits.

15. **The MLX allocator recycles buffers.** If two kernels back-to-back write and read the same buffer, MLX handles synchronisation via its stream. But if you take a raw MTL pointer via `MLXArray.buffer(as:)` and pass it to raw Metal, you own synchronisation — insert your own `waitUntilCompleted` or use an event.

16. **`function constants` vs `buffer constants`.** Function constants specialise the pipeline (recompile per value); buffer constants are runtime uniforms. For `hidden_dim`, `num_experts`, `head_dim` — use function constants (Metal inlines the loops). For `eps`, `seq_len` — use buffer constants (avoid pipeline recompile per generation).

17. **Never trust a kernel's own unit test with a random seed.** Every parity test in this doc compares against MLX or against a formulation-A verbatim implementation. "Kernel matches itself" is not evidence.

18. **`mlx.metal.set_wired_limit` matters on M1 16 GB.** Not a kernel-writing gotcha *per se*, but any Phase-3 kernel that increases peak memory (e.g. materialising `[Bq, Bkv]` scores tiles) can push the M1 into swap. See [DEV Community — 40 GB MLX on M1](https://dev.to/sleepyquant/i-run-a-40gb-ai-model-on-a-macbook-three-months-of-mlx-on-m1-max-has-changed-how-i-think-about-h6j) for the practical setup.

19. **Instruments Metal capture is your ground truth.** The trace tells you kernel launch overhead vs execution time vs the memory-stall breakdown. Do not tune off `NSDate.timeIntervalSince` alone — it aliases everything into "wall clock".

20. **`fma(a, b, c)` in MSL** compiles to a single hardware FMA and is faster than `a * b + c`. Use it in every inner loop.

---

## 9. Uncertainty flags — where I'm speculating and what would resolve it

Every claim below is either flagged **inferred** (derived from a Metal-doc / MLX-source fact but not directly measured) or **speculative** (I'm arguing rather than citing). Numbers next to each flag are the resolvable-by-measurement priority.

**Inferred**:
- §0.2, §8.1: "bf16 arithmetic degrades to fp16 on Apple7/8" — sourced to [`philipturner/metal-benchmarks`](https://github.com/philipturner/metal-benchmarks) and the MPS matmul error message, but I have not personally run a `float x = float(bfloat_val); x = x + 1.0f;` benchmark on M1 that isolates the FP32-vs-emulated-FP16 pipeline. **Resolve** with a microbenchmark on `brawler_yukon`. Priority: high — this changes the storage/compute policy for every kernel.
- §2.3, §4.2: "MLX's argsort-then-scatter is fast enough for Top-K sampling policy" — I'm accepting the LMOE guide's assessment. **Resolve** with Instruments on the M6 baseline once low-confidence-remask is exercised at generation length ≥ 512.
- §3.1: "the fused routed-MoE kernel beats MLX `gather_qmm` on M2 Ultra". The LMOE guide §5.1 already flagged this as speculative; I have not narrowed it. **Resolve** with the Phase-3 §4 item 1 measurement.
- §5.1: "Bq=16, Bkv=32 is the right tile size for Sumi off-by-one attention on M1". Directly cribbed from MLX's own `steel_attention` tile sizes for similar head_dim. **Resolve** with a Bq × Bkv sweep in the parity test.
- §7.4 landing order: "off-by-one softmax is a correctness item, not an optimisation item" — correct for Sumi's parity gate; but ranks it above (3) fused-residual-rms-norm which might be a bigger *step-time* win on M2 Ultra. If Sumi lands after LMOE and DiffusionGemma, the order flips. **Resolve** by fixing the model priority.

**Speculative**:
- §8.14: "`bfloat → half` costs mantissa bits". Technically the fp16 mantissa is 10 bits, bf16 is 7 bits; going `bf16 → fp16` is *lossless* on the mantissa but *lossy* on the exponent range (bf16 has fp32 range, fp16 doesn't). Rewriting §8.14: "converting `bfloat → half` truncates the exponent range from fp32-scale to fp16-scale; large activations saturate." Correct, but not a "mantissa bits" claim. **Fix in v2 of this doc.**
- §2.3: "bitonic sort in threadgroup memory beats MLX two-pass argsort for L ≤ 4096". True in launch count (1 vs 3), likely true in wall-clock, unverified in absolute latency. **Resolve** with a `L ∈ {32, 128, 512, 2048}` microbench.
- §5.1: "the sink term `exp(-m)` is applied ONCE at the epilogue, not per K/V tile". This is a mathematical certainty from the softmax algebra; but I want to hold the "speculative" tag because I have not yet seen a fused flash-attention off-by-one implementation in the wild to cross-check against. **Resolve** when the parity test in §7.2's pattern passes on the Sumi fixture.

If any of these get resolved with a measurement, update this doc's §9 to `sourced` with the measurement URL.

---

## 10. What to say to Claude Code when you kick this off

Copy-pasteable:

> Land the smallest useful kernel from `Plans/metal-shader-guide.md`: `RMSNormKernel` per §2.1 + §7.2. Add `Packages/DiffusionKernels/Sources/Shaders/rms_norm.metal`, `Packages/DiffusionKernels/Sources/OpWrappers/RMSNormKernel.swift`, and `Tests/DiffusionKernelsTests/RMSNormParityTests.swift`. The parity test must compare against `LLaDA2RMSNorm` from `Packages/DiffusionCore` on the exact shapes listed. Do not touch any other file. Do not add the feature flag to `LLaDA2RMSNorm.callAsFunction` until the parity test is green. When the test passes, share the diff for review before adding the feature flag or landing any second kernel.

If the first kernel lands cleanly and parity holds at 1e-4, the pattern is validated — the rest of §7.4's landing order can follow the same recipe with the same review cadence.

---

## 11. Companion documents to update after this doc lands

- `phase-3-optimisation-roadmap.md` §4: replace the four cross-cutting kernel-fusion bullets with references to §7.4 of this doc. The ordering here is more precise.
- `lmoe-implementation-guide.md` §5: keep the model-specific notes; append a pointer at the top of §5 to this doc's §3.
- `sumi-implementation-guide.md` §5: same; point §5.2's fused off-by-one attention sketch at §5.1 of this doc for the concrete Metal.
- `nemotron-labs-diffusion-implementation-guide.md` §5: same; point §5.2's YaRN + Q-scale notes at §6.1 of this doc.
- `diffusiongemma-implementation-guide.md` §5: same; point §5.1's parallel-dense-MoE note at §4 of this doc.
- `phase-1-conceptual-design.md` §13.1: no change — the raw-Metal escape hatch is orthogonal to `MLXFastKernel` fusion. This doc explicitly uses `MLXFastKernel`; the escape hatch is for the whole-step megakernel.

If any of these companion updates diverge in intent (e.g. Sumi's guide wants the fused off-by-one attention *before* the standalone off-by-one softmax lands), reconcile in the model guide's §9 (open questions), not by rewriting this doc.
