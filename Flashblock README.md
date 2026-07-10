# FlashBlock — Metal port

A Metal implementation of [FlashBlock (Chen, Cai, Zhuang, arXiv:2602.05305v1)](https://arxiv.org/abs/2602.05305v1)
for Apple Silicon. Two kernels:

- `flashblock_external_pass` — FlashAttention-style streaming softmax over the
  paged external KV cache, capturing `A_out` and `L_out` (paper Eq. 6), then
  folding in the current block for a complete full-attention output. Run this
  at the first diffusion step of a block, or whenever the number of dirty
  tokens `M^{s+1} ≥ τ`.
- `flashblock_internal_and_compose` — dense attention over just the current
  block's `B × B` internal keys, then log-space composition with the cached
  `(A_out, L_out)` per Eq. (8)–(9). Run this on subsequent steps when
  `M^{s+1} < τ`.

## Build

```bash
xcrun -sdk macosx metal   -c Sources/FlashBlockMetal/FlashBlock.metal -o FlashBlock.air
xcrun -sdk macosx metallib  FlashBlock.air -o FlashBlock.metallib

python3 Scripts/gen_reference.py   # writes Scripts/fixtures/*.npy

swiftc -O -parse-as-library \
  Sources/FlashBlockMetal/FlashBlockRunner.swift \
  Tests/FlashBlockTest.swift \
  -o flashblock_test -framework Metal -framework Foundation
./flashblock_test FlashBlock.metallib Scripts/fixtures
```

Tolerance: `atol = rtol = 1e-2`, matching the reference repo's Triton kernel
test.

## Layout

```
Q            [Nq, H_q, D]           half
K_cur, V_cur [Nq, H_kv, D]          half
K_page,V_page [P, pageSize, H_kv, D] half
block_tables [numSeqs, maxPages]    uint32
ctx_lens     [numSeqs]              int32
A_out        [Nq, H_q, D]           float32   (paper's cached attention output)
L_out        [Nq, H_q]              float32   (paper's log-normalizer, natural log)
dirty_mask   [numSeqs, blockLen]    uint8     (1 = token resampled this step)
head_gamma   [H_q]                  float32   (video path, optional)
O            [Nq, H_q, D]           half
```

Nq = numSeqs · blockLen. GQA: H_q = kv_group · H_kv.

`sm_scale = (1/√D) · log₂ e` — precomputed on the host so the softmax stays in
log₂ domain with `fast::exp2` (mirrors the CUDA reference).

## Grid

Both kernels dispatch `MTLSize(numSeqs, num_q_heads, 1)` threadgroups with
`(BLOCK_M, 1, 1)` threads. One thread == one query row of the block. Function
constants (`BLOCK_M`, `BLOCK_N`, `HEAD_DIM`, `KV_GROUP`, `PAGE_SIZE`,
`MAX_PAGES`) are baked at pipeline creation, so the inner loops can be fully
unrolled.

# Fusion opportunities

Short answer: **yes for the ops that live on the same tensors and don't
inflate register pressure past ~256 fp32 slots per thread; no for anything
that would force us to spill or change the grid.**

The current kernels already fuse the two things that matter most in the
reference implementation:

- Online softmax (FA-2 style) fused with the K/V streaming loop — no separate
  softmax pass.
- The `A_out` / `L_out` capture is fused into the same streaming pass, so we
  don't reread the paged KV to build the cache. This is exactly what the
  reference's modified `flash_attn_with_kvcache` does.

## What's worth fusing next

### 1. Q · K rotary application into the attention kernel  →  ✅ do it

Cost: one extra sincos per (q_row, half of D) plus one for (k_row, half of D)
inside the tile load. Register cost: ~2×D floats for the rotation table if you
precompute it, or 0 if you evaluate `sincos` inline (Apple's `fast::sincos`
is cheap). No extra threadgroup memory.

Payoff: eliminates a full read/write pass over Q and K_cur (roughly `Nq · H ·
D · 2 bytes · 2` traffic — meaningful when blockLen and Nq are small and the
external KV loop is short).

Recommendation: fuse rotary into the tile load. Add `rope_cos`, `rope_sin`
buffers as kernel arguments; apply them right after each half is loaded into
threadgroup memory.

### 2. QKV projection into the attention kernel  →  ❌ don't

This is the tempting one, but it's a bad idea on M-series:

- QKV projection is a big matmul over `[Nq, hidden_dim] × [hidden_dim, 3·H·D]`.
  It wants `simdgroup_matrix` tiles in a totally different layout from
  attention's `qk` product.
- Doing it inside the attention kernel forces you to keep the full input
  hidden state resident in threadgroup memory across the entire KV streaming
  loop, or reread it — either destroys occupancy or destroys bandwidth.
- On CUDA/H100 people fuse this because of async copies + huge shared memory.
  Metal has neither in the same form; threadgroup memory on M3 is 32 KB per
  threadgroup, and there's no equivalent of `cp.async`.

Recommendation: keep QKV projection as an MPS `MPSMatrixMultiplication` call
(or a dedicated matmul kernel), write outputs to K_cur/V_cur, then attention.

### 3. Output projection + residual add + RMSNorm  →  ⚠️ partially

Fusing the output projection with attention has the same layout mismatch
problem as QKV. **Do not** fuse output projection into the attention kernel.

However, fusing **residual add + RMSNorm** onto the output projection is a
standard, well-known win. That's a separate kernel outside FlashBlock's scope
but worth doing. Metal Performance Shaders Graph will do this for you if you
express the block as a graph and let it schedule.

### 4. Multiple attention heads per threadgroup  →  ⚠️ only if D is small

If `HEAD_DIM = 64`, you can pack 2 heads per threadgroup for the same query
row and share the K/V tile load between them. That halves the paged-KV traffic
per query row, which is the FlashBlock dominant cost at long context. But it
doubles register pressure (`acc[D]` becomes `acc[2·D]`) and doubles the
per-thread `qk` array. On M3+ this is fine at D=64; on M1 it will spill.

Recommendation: add a `HEADS_PER_TG` function constant, default 1, allow 2 for
D ≤ 64 as a compile-time flavor.

### 5. Composition + output-projection matmul  →  ✅ do it

The compose kernel writes `[Nq, H, D]` in fp16, and the *very next* op is a
matmul against `W_o` of shape `[H·D, hidden_dim]`. Reading `[Nq, H, D]` back
from VRAM only to matmul it is pure bandwidth waste. Two options:

- Fuse the log-space compose *into* the output-projection matmul kernel: for
  each row `q`, compute `A_full` in registers, then dot into `W_o` columns.
  This works because compose is already row-local (no cross-row dependence).
- Or, simpler: emit `A_full` into threadgroup memory of the output-proj
  kernel via a shared-buffer handoff.

The first is strictly better and only ~40 extra lines of MSL. Worth the
complexity if output projection is a measurable cost in your model. On
Trado-8B / SDAR profiles the reference repo linked to, output projection is
~8–12% of layer time.

### 6. K/V cache write of newly generated tokens  →  ✅ already fusable

The reference calls `store_kvcache(...)` before attention. On Metal you can
fuse that into the current-block tile-load phase of Kernel A: as each thread
touches its K_cur/V_cur slice, write it to the paged KV at the appropriate
slot. Zero extra bandwidth, one less kernel launch.

## What NOT to fuse

- **Softmax normalization of the final output into the compose step.**
  The compose step already produces normalized output (the division by
  `denom` in Eq. 9). No separate normalization exists.
- **Anything cross-block.** The whole design relies on independence between
  blocks and between (seq, head) pairs. Anything that couples them wrecks the
  grid.
- **fp16 accumulation.** Keep `acc`, `l_i`, `m_i`, `A_out`, `L_out` in fp32.
  32 diffusion steps of fp16 accumulation on 8k-context KV drifts far past
  1e-2 in my experience; the reference uses fp32 for `attn_output_past` for
  the same reason.

## Practical fusion roadmap

Order of expected impact on M3/M4 with a Trado-style 8B diffusion LM at 8k
context, `blockLen = 32`:

1. **Fuse rotary into tile loads** (~5–8% end-to-end).
2. **Fuse KV writeback for current block into Kernel A** (~2–4%; more if you
   were previously doing it as a separate kernel).
3. **Fuse compose + output projection** (~5–10%).
4. **Add simdgroup_matrix path for qk and pv** (~30–50% on M3+; this dwarfs
   the fusion wins and is the single most impactful change you can make).

If I were prioritizing, I'd land the simdgroup_matrix rewrite (#4) before any
fusion. Fusion changes save memory-traffic constants; the matmul rewrite
changes the FLOP roofline.

# Files

```
Sources/FlashBlockMetal/
    FlashBlock.metal           — the two kernels
    FlashBlockRunner.swift     — pipeline states, dispatch, reuse gate
Scripts/
    gen_reference.py           — fp32 reference tensors as .npy
    fixtures/                  — produced by the script
Tests/
    FlashBlockTest.swift       — loads fixtures, dispatches, checks tolerance
```
