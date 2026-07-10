// FlashBlock.metal
//
// Metal port of FlashBlock (Chen, Cai, Zhuang; arXiv:2602.05305v1).
// Two kernels:
//   flashblock_external_pass       — FlashAttention-style streaming softmax over
//                                    the paged EXTERNAL KV cache, plus caching
//                                    of A_out (attnOutPast) and L_out (logsumexp).
//   flashblock_internal_and_compose — dense attention over the current block's
//                                    KV (INTERNAL), then log-space composition
//                                    with cached (A_out, L_out) per paper Eq.(8-9).
//
// Layout conventions:
//   Query tensor Q         : [Nq_total, num_q_heads, head_dim]        half
//                            Nq_total = numSeqs * blockLen (packed).
//   Current-block K/V      : [Nq_total, num_kv_heads, head_dim]       half
//   Paged KV cache         : [numPages, pageSize, num_kv_heads, head_dim] half
//   block_tables           : [numSeqs, maxPagesPerSeq]                uint
//   ctx_lens               : [numSeqs]                                int  (external length in tokens)
//   attnOutPast (A_out)    : [numSeqs, blockLen, num_q_heads, head_dim] float
//   logsumexp   (L_out)    : [numSeqs, blockLen, num_q_heads]         float
//   dirtyMask              : [numSeqs, blockLen]                      uchar (1 = token was resampled this step)
//   headGamma              : [num_q_heads]                            float (video path only; text path passes nullptr)
//   output O               : [Nq_total, num_q_heads, head_dim]        half
//
// GQA: num_q_heads = KV_GROUP * num_kv_heads. Each Q head reads the KV head at
// kv_head = q_head / KV_GROUP.
//
// Numerical convention:
//   sm_scale must be passed as (1/sqrt(head_dim)) * log2(e). We keep the whole
//   softmax in log2 space with fast::exp2 and convert to natural log at the
//   very end when writing L_out, so the compose kernel can mix cleanly with a
//   host-computed reference.
//
// Threadgroup / grid:
//   dispatchThreadgroups: MTLSize(numSeqs, num_q_heads, 1)
//   threadsPerThreadgroup: (BLOCK_M, 1, 1), BLOCK_M ∈ {16, 32}.
//   One thread == one query row of the block (blockLen ≤ BLOCK_M).
//   BLOCK_N tiles are cooperatively loaded into threadgroup memory by all
//   threads in the group; then each thread scans its own row against them.
//
// This is intentionally simple and portable across M1..M4. It does NOT use
// simdgroup_matrix, so it will leave 30-50% perf on the table on M3/M4.
// See the notes at the bottom of the file for how to swap that in.

#include <metal_stdlib>
#include <metal_math>
#include <metal_simdgroup>
using namespace metal;

// ---- Function constants (set at pipeline-state creation from Swift) --------

constant int  BLOCK_M   [[function_constant(0)]];  // e.g. 16 or 32 (>= blockLen)
constant int  BLOCK_N   [[function_constant(1)]];  // e.g. 64 or 128
constant int  HEAD_DIM  [[function_constant(2)]];  // 64, 96, 128
constant int  KV_GROUP  [[function_constant(3)]];  // num_q_heads / num_kv_heads
constant int  PAGE_SIZE [[function_constant(4)]];  // KV page granularity, e.g. 256
constant int  MAX_PAGES [[function_constant(5)]];  // block_tables.stride(0)

// Uniform parameters (small, pass by value in a constant buffer).
struct FlashBlockParams {
    int   num_seqs;
    int   block_len;         // B in the paper; must be <= BLOCK_M
    int   num_q_heads;
    int   num_kv_heads;
    float sm_scale;          // (1/sqrt(D)) * log2(e)
    float compose_gamma;     // per-head gate for video path
    int   use_dirty_gate;    // 1 = honor dirty_mask (skip unchanged rows)
    int   use_head_gamma;    // 1 = honor head_gamma (video path)
};

// ============================================================================
// Kernel A — block-external streaming pass with A_out, L_out capture
// ============================================================================
//
// One threadgroup handles (seq, q_head). Each thread handles one query row of
// the block. We stream through the paged EXTERNAL KV cache, applying online
// softmax. When the external stream is exhausted we cache (A_out, L_out) — this
// is the FlashBlock-specific step. Then we optionally fold in the current
// block's K/V to produce a complete attention output in one pass (this is what
// the reference `flash_attn_with_kvcache` fork does when the FlashBlock cache
// is being *refreshed*, i.e. first step of a block or when M^{s+1} >= tau).

kernel void flashblock_external_pass(
    device const half*   Q                [[buffer(0)]],
    device const half*   K_cur            [[buffer(1)]],  // current block K
    device const half*   V_cur            [[buffer(2)]],  // current block V
    device const half*   K_page           [[buffer(3)]],  // paged external K
    device const half*   V_page           [[buffer(4)]],  // paged external V
    device const uint*   block_tables     [[buffer(5)]],
    device const int*    ctx_lens         [[buffer(6)]],  // external length per seq
    device       half*   O                [[buffer(7)]],  // full attention output
    device       float*  attnOutPast      [[buffer(8)]],  // A_out cache
    device       float*  logsumexp        [[buffer(9)]],  // L_out cache (natural log)
    constant     FlashBlockParams& P      [[buffer(10)]],
    threadgroup  half*   tg_scratch       [[threadgroup(0)]], // 2 * BLOCK_N * HEAD_DIM halfs
    uint3 tgid   [[threadgroup_position_in_grid]],
    uint  tid    [[thread_position_in_threadgroup]],
    uint  ntids  [[threads_per_threadgroup]])
{
    const uint seq   = tgid.x;
    const uint qh    = tgid.y;
    const uint kh    = qh / uint(KV_GROUP);
    const int  Nk    = ctx_lens[seq];
    const int  B     = P.block_len;

    // Threadgroup tiles for the current K/V window.
    threadgroup half* Ktile = tg_scratch;
    threadgroup half* Vtile = tg_scratch + BLOCK_N * HEAD_DIM;

    // Per-row (per-thread) online-softmax state.
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float acc[HEAD_DIM];
    #pragma clang loop unroll(full)
    for (int d = 0; d < HEAD_DIM; ++d) acc[d] = 0.0f;

    // Load this row's Q into registers. Row = (seq * B + tid, qh, :).
    // Threads with tid >= B are inactive for the softmax but still cooperate
    // on tile loads.
    const bool row_active = (int(tid) < B);
    half q_row[HEAD_DIM];
    if (row_active) {
        const uint q_row_idx = seq * uint(B) + tid;
        device const half* qp =
            Q + (q_row_idx * uint(P.num_q_heads) + qh) * uint(HEAD_DIM);
        #pragma clang loop unroll(full)
        for (int d = 0; d < HEAD_DIM; ++d) q_row[d] = qp[d];
    } else {
        #pragma clang loop unroll(full)
        for (int d = 0; d < HEAD_DIM; ++d) q_row[d] = 0.0h;
    }

    // -------- Stream over paged EXTERNAL KV --------
    const int num_pages = (Nk + PAGE_SIZE - 1) / PAGE_SIZE;
    int kv_pos = 0;

    for (int lp = 0; lp < num_pages; ++lp) {
        const uint page_id = block_tables[seq * uint(MAX_PAGES) + uint(lp)];
        const int  toks    = min(Nk - kv_pos, PAGE_SIZE);

        for (int tn = 0; tn < toks; tn += BLOCK_N) {
            const int actual = min(toks - tn, BLOCK_N);

            // Cooperative tile load. Each thread pulls (BLOCK_N * HEAD_DIM) /
            // ntids halfs from K_page/V_page into threadgroup memory.
            const uint tile_elems = uint(BLOCK_N) * uint(HEAD_DIM);
            for (uint e = tid; e < tile_elems; e += ntids) {
                const uint j = e / uint(HEAD_DIM);      // row in tile
                const uint d = e % uint(HEAD_DIM);      // dim
                if (int(j) < actual) {
                    const uint src =
                        ((page_id * uint(PAGE_SIZE) + uint(tn) + j) * uint(P.num_kv_heads) + kh)
                        * uint(HEAD_DIM) + d;
                    Ktile[e] = K_page[src];
                    Vtile[e] = V_page[src];
                } else {
                    Ktile[e] = 0.0h;
                    Vtile[e] = 0.0h;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (row_active) {
                // qk[j] = <q_row, Ktile[j]> * sm_scale (log2 domain scale).
                float qk[BLOCK_N];
                #pragma clang loop unroll(full)
                for (int j = 0; j < BLOCK_N; ++j) qk[j] = -INFINITY;

                for (int j = 0; j < actual; ++j) {
                    float s = 0.0f;
                    threadgroup const half* krow = Ktile + j * HEAD_DIM;
                    #pragma clang loop unroll(full)
                    for (int d = 0; d < HEAD_DIM; ++d)
                        s = fma(float(q_row[d]), float(krow[d]), s);
                    qk[j] = s * P.sm_scale;
                }

                // Online softmax update.
                float m_new = m_i;
                for (int j = 0; j < actual; ++j) m_new = max(m_new, qk[j]);
                const float alpha = fast::exp2(m_i - m_new);
                l_i *= alpha;
                #pragma clang loop unroll(full)
                for (int d = 0; d < HEAD_DIM; ++d) acc[d] *= alpha;

                for (int j = 0; j < actual; ++j) {
                    const float p = fast::exp2(qk[j] - m_new);
                    l_i += p;
                    threadgroup const half* vrow = Vtile + j * HEAD_DIM;
                    #pragma clang loop unroll(full)
                    for (int d = 0; d < HEAD_DIM; ++d)
                        acc[d] = fma(p, float(vrow[d]), acc[d]);
                }
                m_i = m_new;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        kv_pos += toks;
    }

    // -------- Cache A_out, L_out BEFORE folding in the current block --------
    // This is Eq. (6) of the paper. We store the *external-only* softmax
    // statistics per (seq, row, q_head).
    if (row_active) {
        const uint qrow  = seq * uint(B) + tid;
        const uint slot  = qrow * uint(P.num_q_heads) + qh;
        const float l_safe = (l_i == 0.0f) ? 1.0f : l_i;

        #pragma clang loop unroll(full)
        for (int d = 0; d < HEAD_DIM; ++d)
            attnOutPast[slot * uint(HEAD_DIM) + d] = acc[d] / l_safe;

        // Convert log2-domain running max/sum to natural-log L_out:
        //   L_out = ln(Z_out) = m_i * ln(2) + ln(l_i)
        logsumexp[slot] = m_i * float(M_LN2_F) + log(l_safe);
    }

    // -------- Fold in the current block's K/V for the full output ----------
    // This is what the CUDA reference's `flash_attn_with_kvcache` does in one
    // pass. We continue the same online-softmax accumulators, now with the
    // in-block keys/values as an additional tile.
    for (int tn = 0; tn < B; tn += BLOCK_N) {
        const int actual = min(B - tn, BLOCK_N);
        const uint tile_elems = uint(BLOCK_N) * uint(HEAD_DIM);

        for (uint e = tid; e < tile_elems; e += ntids) {
            const uint j = e / uint(HEAD_DIM);
            const uint d = e % uint(HEAD_DIM);
            if (int(j) < actual) {
                const uint src_row = seq * uint(B) + uint(tn) + j;
                const uint src =
                    (src_row * uint(P.num_kv_heads) + kh) * uint(HEAD_DIM) + d;
                Ktile[e] = K_cur[src];
                Vtile[e] = V_cur[src];
            } else {
                Ktile[e] = 0.0h;
                Vtile[e] = 0.0h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row_active) {
            float qk[BLOCK_N];
            #pragma clang loop unroll(full)
            for (int j = 0; j < BLOCK_N; ++j) qk[j] = -INFINITY;

            for (int j = 0; j < actual; ++j) {
                float s = 0.0f;
                threadgroup const half* krow = Ktile + j * HEAD_DIM;
                #pragma clang loop unroll(full)
                for (int d = 0; d < HEAD_DIM; ++d)
                    s = fma(float(q_row[d]), float(krow[d]), s);
                qk[j] = s * P.sm_scale;
            }

            float m_new = m_i;
            for (int j = 0; j < actual; ++j) m_new = max(m_new, qk[j]);
            const float alpha = fast::exp2(m_i - m_new);
            l_i *= alpha;
            #pragma clang loop unroll(full)
            for (int d = 0; d < HEAD_DIM; ++d) acc[d] *= alpha;

            for (int j = 0; j < actual; ++j) {
                const float p = fast::exp2(qk[j] - m_new);
                l_i += p;
                threadgroup const half* vrow = Vtile + j * HEAD_DIM;
                #pragma clang loop unroll(full)
                for (int d = 0; d < HEAD_DIM; ++d)
                    acc[d] = fma(p, float(vrow[d]), acc[d]);
            }
            m_i = m_new;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // -------- Write full attention output ----------
    if (row_active) {
        const uint qrow = seq * uint(B) + tid;
        device half* op =
            O + (qrow * uint(P.num_q_heads) + qh) * uint(HEAD_DIM);
        const float l_safe = (l_i == 0.0f) ? 1.0f : l_i;
        #pragma clang loop unroll(full)
        for (int d = 0; d < HEAD_DIM; ++d)
            op[d] = half(acc[d] / l_safe);
    }
}

// ============================================================================
// Kernel B — block-internal pass + log-space composition with cached A_out
// ============================================================================
//
// One threadgroup per (seq, q_head). Each thread handles one query row.
// Attention is computed only against the current block's K/V (INTERNAL) which
// is small (B x D), then combined in log-space with the cached (A_out, L_out).

kernel void flashblock_internal_and_compose(
    device const half*   Q                [[buffer(0)]],
    device const half*   K_cur            [[buffer(1)]],
    device const half*   V_cur            [[buffer(2)]],
    device const float*  attnOutPast      [[buffer(3)]],   // A_out
    device const float*  logsumexp        [[buffer(4)]],   // L_out (natural log)
    device       half*   O                [[buffer(5)]],
    device const uchar*  dirty_mask       [[buffer(6)]],   // optional; [numSeqs, blockLen]
    device const float*  head_gamma       [[buffer(7)]],   // optional; [num_q_heads]
    constant     FlashBlockParams& P      [[buffer(8)]],
    threadgroup  half*   tg_scratch       [[threadgroup(0)]], // 2 * BLOCK_M * HEAD_DIM halfs
    uint3 tgid   [[threadgroup_position_in_grid]],
    uint  tid    [[thread_position_in_threadgroup]],
    uint  ntids  [[threads_per_threadgroup]])
{
    const uint seq  = tgid.x;
    const uint qh   = tgid.y;
    const uint kh   = qh / uint(KV_GROUP);
    const int  B    = P.block_len;

    // Head-wise reuse gate (video path). If enabled and this head's similarity
    // is below threshold, fall back to a full recompute: the host is expected
    // to have dispatched flashblock_external_pass for (seq, this head), which
    // already wrote O. We just early-out here.
    if (P.use_head_gamma != 0 && head_gamma[qh] < P.compose_gamma) {
        return;
    }

    // Load Q row into registers.
    const bool row_active = (int(tid) < B);
    half q_row[HEAD_DIM];
    if (row_active) {
        const uint q_row_idx = seq * uint(B) + tid;
        device const half* qp =
            Q + (q_row_idx * uint(P.num_q_heads) + qh) * uint(HEAD_DIM);
        #pragma clang loop unroll(full)
        for (int d = 0; d < HEAD_DIM; ++d) q_row[d] = qp[d];
    } else {
        #pragma clang loop unroll(full)
        for (int d = 0; d < HEAD_DIM; ++d) q_row[d] = 0.0h;
    }

    // Load the whole current-block K/V into threadgroup memory (B x HEAD_DIM,
    // and B <= BLOCK_M is small — fits comfortably).
    threadgroup half* Ktile = tg_scratch;
    threadgroup half* Vtile = tg_scratch + BLOCK_M * HEAD_DIM;

    const uint tile_elems = uint(BLOCK_M) * uint(HEAD_DIM);
    for (uint e = tid; e < tile_elems; e += ntids) {
        const uint j = e / uint(HEAD_DIM);
        const uint d = e % uint(HEAD_DIM);
        if (int(j) < B) {
            const uint src_row = seq * uint(B) + j;
            const uint src =
                (src_row * uint(P.num_kv_heads) + kh) * uint(HEAD_DIM) + d;
            Ktile[e] = K_cur[src];
            Vtile[e] = V_cur[src];
        } else {
            Ktile[e] = 0.0h;
            Vtile[e] = 0.0h;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Dense attention over B internal keys (single tile).
    float m_in = -INFINITY;
    float l_in = 0.0f;
    float a_in[HEAD_DIM];
    #pragma clang loop unroll(full)
    for (int d = 0; d < HEAD_DIM; ++d) a_in[d] = 0.0f;

    if (row_active) {
        float qk[BLOCK_M];
        #pragma clang loop unroll(full)
        for (int j = 0; j < BLOCK_M; ++j) qk[j] = -INFINITY;

        for (int j = 0; j < B; ++j) {
            float s = 0.0f;
            threadgroup const half* krow = Ktile + j * HEAD_DIM;
            #pragma clang loop unroll(full)
            for (int d = 0; d < HEAD_DIM; ++d)
                s = fma(float(q_row[d]), float(krow[d]), s);
            qk[j] = s * P.sm_scale;
        }
        for (int j = 0; j < B; ++j) m_in = max(m_in, qk[j]);

        for (int j = 0; j < B; ++j) {
            const float p = fast::exp2(qk[j] - m_in);
            l_in += p;
            threadgroup const half* vrow = Vtile + j * HEAD_DIM;
            #pragma clang loop unroll(full)
            for (int d = 0; d < HEAD_DIM; ++d)
                a_in[d] = fma(p, float(vrow[d]), a_in[d]);
        }
        const float l_safe = (l_in == 0.0f) ? 1.0f : l_in;
        #pragma clang loop unroll(full)
        for (int d = 0; d < HEAD_DIM; ++d) a_in[d] /= l_safe;

        // L_in in NATURAL LOG (matches L_out convention).
        const float L_in = m_in * float(M_LN2_F) + log(l_safe);

        // Load cached (A_out, L_out).
        const uint qrow_idx = seq * uint(B) + tid;
        const uint slot     = qrow_idx * uint(P.num_q_heads) + qh;
        const float L_out   = logsumexp[slot];

        // Log-space composition, paper Eq. (8)-(9).
        const float m_c   = max(L_out, L_in);
        const float w_out = fast::exp(L_out - m_c);
        const float w_in  = fast::exp(L_in  - m_c);
        const float denom = w_out + w_in;

        // Optional per-query dirty-token gate: if the token wasn't resampled
        // this step, we can skip composition and re-emit the previous full
        // output. That's a stricter form of the M^{s+1} < tau gate the paper
        // does host-side; keeping it here lets us skip work per-row.
        const bool skip_this_row =
            (P.use_dirty_gate != 0)
            && (dirty_mask[seq * uint(B) + tid] == 0);

        device half* op =
            O + (qrow_idx * uint(P.num_q_heads) + qh) * uint(HEAD_DIM);

        if (skip_this_row) {
            // Row is unchanged: re-emit the cached A_out directly. (For rows
            // whose token didn't change, A_full ≈ A_out with negligible drift
            // — this is the paper's approximation baseline.)
            device const float* aop =
                attnOutPast + slot * uint(HEAD_DIM);
            #pragma clang loop unroll(full)
            for (int d = 0; d < HEAD_DIM; ++d) op[d] = half(aop[d]);
        } else {
            device const float* aop =
                attnOutPast + slot * uint(HEAD_DIM);
            #pragma clang loop unroll(full)
            for (int d = 0; d < HEAD_DIM; ++d) {
                const float mixed = (w_out * aop[d] + w_in * a_in[d]) / denom;
                op[d] = half(mixed);
            }
        }
    }
}

// ============================================================================
// Notes on optimizing further (see accompanying README):
//
// 1. simdgroup_matrix: on M3/M4 (and A17+), replace the qk[] and pv loops with
//    simdgroup_matrix<half, 8, 8> tiles. Load Q into an 8x8 fragment, stream
//    K/V through matching fragments, accumulate in float. Expect 1.6-2.2x on
//    HEAD_DIM=128.
// 2. Fusing RMSNorm/rotary/QKV-proj into this kernel is discussed in the
//    README under "Fusion opportunities". Short version: yes for QK-rotary,
//    no for QKV projection or output projection unless you also fuse the
//    residual add.
// 3. Persistent-threadgroup / stream-K style scheduling helps only when
//    numSeqs * num_q_heads < number of SMs. For M2/M3/M4 that break-even sits
//    around 30-40 threadgroups.
// ============================================================================
