// FlashBlockRunner.swift
//
// Pure-Swift asynchronous FlashBlock runner using MLXFast.metalKernel.
// Eliminates CPU-GPU roundtrips by running directly on MLX's internal command buffer stream.

import Foundation
import MLX

public struct FlashBlockConfig {
    public var numSeqs: Int
    public var blockLen: Int            // B
    public var numQHeads: Int
    public var numKVHeads: Int
    public var headDim: Int
    public var pageSize: Int
    public var maxPagesPerSeq: Int

    // Tile sizes
    public var blockM: Int              // 16 or 32
    public var blockN: Int              // 64 or 128

    // Reuse gates
    public var tau: Int                 // per-block dirty-token threshold
    public var composeGamma: Float      // per-head similarity threshold

    public init(numSeqs: Int, blockLen: Int,
                numQHeads: Int, numKVHeads: Int, headDim: Int,
                pageSize: Int = 256, maxPagesPerSeq: Int = 64,
                blockM: Int = 32, blockN: Int = 128,
                tau: Int = 4, composeGamma: Float = 0.0) {
        precondition(numQHeads % numKVHeads == 0, "GQA: numQHeads must be a multiple of numKVHeads")
        precondition(blockLen <= blockM, "blockLen must be <= blockM")
        precondition([64, 96, 128].contains(headDim), "headDim must be 64/96/128")
        self.numSeqs = numSeqs
        self.blockLen = blockLen
        self.numQHeads = numQHeads
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.pageSize = pageSize
        self.maxPagesPerSeq = maxPagesPerSeq
        self.blockM = blockM
        self.blockN = blockN
        self.tau = tau
        self.composeGamma = composeGamma
    }

    var kvGroup: Int { numQHeads / numKVHeads }
    var nqTotal: Int { numSeqs * blockLen }
}

public final class FlashBlockRunner {

    public enum StepKind {
        case refreshCache
        case reuseCache
    }

    public let config: FlashBlockConfig
    private let externalKernel: MLXFast.MLXFastKernel
    private let internalKernel: MLXFast.MLXFastKernel

    public init(config: FlashBlockConfig) throws {
        self.config = config

        // Bake constants as preprocessor defines
        let header = """
        #define BLOCK_M \(config.blockM)
        #define BLOCK_N \(config.blockN)
        #define HEAD_DIM \(config.headDim)
        #define KV_GROUP \(config.kvGroup)
        #define PAGE_SIZE \(config.pageSize)
        #define MAX_PAGES \(config.maxPagesPerSeq)

        """

        let externalSource = """
            const uint3 tgid = threadgroup_position_in_grid;
            const uint3 tid = thread_position_in_threadgroup;
            const uint3 ntids = threads_per_threadgroup;

            const uint seq   = tgid.x;
            const uint qh    = tgid.y;
            const uint kh    = qh / uint(KV_GROUP);
            const int  Nk    = ctx_lens[seq];
            const int  B     = \(config.blockLen);
            const uint tid_x = tid.x;
            const uint ntids_x = ntids.x;

            struct FlashBlockParams {
                int num_seqs;
                int block_len;
                int num_q_heads;
                int num_kv_heads;
                float sm_scale;
                float compose_gamma;
                int use_dirty_gate;
                int use_head_gamma;
            };
            FlashBlockParams P;
            P.num_seqs = \(config.numSeqs);
            P.block_len = \(config.blockLen);
            P.num_q_heads = \(config.numQHeads);
            P.num_kv_heads = \(config.numKVHeads);
            P.sm_scale = \((1.0 / Float(config.headDim).squareRoot()) * Float(log2(M_E)));
            P.compose_gamma = \(config.composeGamma);
            P.use_dirty_gate = 1;
            P.use_head_gamma = 0;

            threadgroup half Ktile[BLOCK_N * HEAD_DIM];
            threadgroup half Vtile[BLOCK_N * HEAD_DIM];

            float m_i = -INFINITY;
            float l_i = 0.0f;
            float acc[128];
            #pragma clang loop unroll(full)
            for (int d = 0; d < 128; ++d) acc[d] = 0.0f;

            const bool row_active = (int(tid_x) < B);
            half q_row[128];
            if (row_active) {
                const uint q_row_idx = seq * uint(B) + tid_x;
                device const half* qp =
                    Q + (q_row_idx * uint(P.num_q_heads) + qh) * uint(HEAD_DIM);
                #pragma clang loop unroll(full)
                for (int d = 0; d < 128; ++d) {
                    if (d < HEAD_DIM) {
                        q_row[d] = qp[d];
                    } else {
                        q_row[d] = 0.0h;
                    }
                }
            } else {
                #pragma clang loop unroll(full)
                for (int d = 0; d < 128; ++d) q_row[d] = 0.0h;
            }

            const int num_pages = (Nk + PAGE_SIZE - 1) / PAGE_SIZE;
            int kv_pos = 0;

            for (int lp = 0; lp < num_pages; ++lp) {
                const uint page_id = block_tables[seq * uint(MAX_PAGES) + uint(lp)];
                const int  toks    = min(Nk - kv_pos, PAGE_SIZE);

                for (int tn = 0; tn < toks; tn += BLOCK_N) {
                    const int actual = min(toks - tn, BLOCK_N);

                    const uint tile_elems = uint(BLOCK_N) * uint(HEAD_DIM);
                    for (uint e = tid_x; e < tile_elems; e += ntids_x) {
                        const uint j = e / uint(HEAD_DIM);
                        const uint d = e % uint(HEAD_DIM);
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
                        float qk[128];
                        #pragma clang loop unroll(full)
                        for (int j = 0; j < 128; ++j) qk[j] = -INFINITY;

                        for (int j = 0; j < actual; ++j) {
                            float s = 0.0f;
                            threadgroup const half* krow = Ktile + j * HEAD_DIM;
                            #pragma clang loop unroll(full)
                            for (int d = 0; d < 128; ++d) {
                                if (d < HEAD_DIM) {
                                    s = fma(float(q_row[d]), float(krow[d]), s);
                                }
                            }
                            qk[j] = s * P.sm_scale;
                        }

                        float m_new = m_i;
                        for (int j = 0; j < actual; ++j) m_new = max(m_new, qk[j]);
                        const float alpha = fast::exp2(m_i - m_new);
                        l_i *= alpha;
                        #pragma clang loop unroll(full)
                        for (int d = 0; d < 128; ++d) acc[d] *= alpha;

                        for (int j = 0; j < actual; ++j) {
                            const float p = fast::exp2(qk[j] - m_new);
                            l_i += p;
                            threadgroup const half* vrow = Vtile + j * HEAD_DIM;
                            #pragma clang loop unroll(full)
                            for (int d = 0; d < 128; ++d) {
                                if (d < HEAD_DIM) {
                                    acc[d] = fma(p, float(vrow[d]), acc[d]);
                                }
                            }
                        }
                        m_i = m_new;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                kv_pos += toks;
            }

            if (row_active) {
                const uint qrow  = seq * uint(B) + tid_x;
                const uint slot  = qrow * uint(P.num_q_heads) + qh;
                const float l_safe = (l_i == 0.0f) ? 1.0f : l_i;

                #pragma clang loop unroll(full)
                for (int d = 0; d < 128; ++d) {
                    if (d < HEAD_DIM) {
                        attnOutPast[slot * uint(HEAD_DIM) + d] = acc[d] / l_safe;
                    }
                }
                logsumexp[slot] = m_i * float(M_LN2_F) + log(l_safe);
            }

            for (int tn = 0; tn < B; tn += BLOCK_N) {
                const int actual = min(B - tn, BLOCK_N);
                const uint tile_elems = uint(BLOCK_N) * uint(HEAD_DIM);

                for (uint e = tid_x; e < tile_elems; e += ntids_x) {
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
                    float qk[128];
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < 128; ++j) qk[j] = -INFINITY;

                    for (int j = 0; j < actual; ++j) {
                        float s = 0.0f;
                        threadgroup const half* krow = Ktile + j * HEAD_DIM;
                        #pragma clang loop unroll(full)
                        for (int d = 0; d < 128; ++d) {
                            if (d < HEAD_DIM) {
                                s = fma(float(q_row[d]), float(krow[d]), s);
                            }
                        }
                        qk[j] = s * P.sm_scale;
                    }

                    float m_new = m_i;
                    for (int j = 0; j < actual; ++j) m_new = max(m_new, qk[j]);
                    const float alpha = fast::exp2(m_i - m_new);
                    l_i *= alpha;
                    #pragma clang loop unroll(full)
                    for (int d = 0; d < 128; ++d) acc[d] *= alpha;

                    for (int j = 0; j < actual; ++j) {
                        const float p = fast::exp2(qk[j] - m_new);
                        l_i += p;
                        threadgroup const half* vrow = Vtile + j * HEAD_DIM;
                        #pragma clang loop unroll(full)
                        for (int d = 0; d < 128; ++d) {
                            if (d < HEAD_DIM) {
                                acc[d] = fma(p, float(vrow[d]), acc[d]);
                            }
                        }
                    }
                    m_i = m_new;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (row_active) {
                const uint qrow = seq * uint(B) + tid_x;
                device half* op =
                    O + (qrow * uint(P.num_q_heads) + qh) * uint(HEAD_DIM);
                const float l_safe = (l_i == 0.0f) ? 1.0f : l_i;
                #pragma clang loop unroll(full)
                for (int d = 0; d < 128; ++d) {
                    if (d < HEAD_DIM) {
                        op[d] = half(acc[d] / l_safe);
                    }
                }
            }
        """

        let internalSource = """
            const uint3 tgid = threadgroup_position_in_grid;
            const uint3 tid = thread_position_in_threadgroup;
            const uint3 ntids = threads_per_threadgroup;

            const uint seq  = tgid.x;
            const uint qh   = tgid.y;
            const uint kh   = qh / uint(KV_GROUP);
            const int  B    = \(config.blockLen);
            const uint tid_x = tid.x;
            const uint ntids_x = ntids.x;

            struct FlashBlockParams {
                int num_seqs;
                int block_len;
                int num_q_heads;
                int num_kv_heads;
                float sm_scale;
                float compose_gamma;
                int use_dirty_gate;
                int use_head_gamma;
            };
            FlashBlockParams P;
            P.num_seqs = \(config.numSeqs);
            P.block_len = \(config.blockLen);
            P.num_q_heads = \(config.numQHeads);
            P.num_kv_heads = \(config.numKVHeads);
            P.sm_scale = \((1.0 / Float(config.headDim).squareRoot()) * Float(log2(M_E)));
            P.compose_gamma = \(config.composeGamma);
            P.use_dirty_gate = 1;
            P.use_head_gamma = 0;

            const bool row_active = (int(tid_x) < B);
            half q_row[128];
            if (row_active) {
                const uint q_row_idx = seq * uint(B) + tid_x;
                device const half* qp =
                    Q + (q_row_idx * uint(P.num_q_heads) + qh) * uint(HEAD_DIM);
                #pragma clang loop unroll(full)
                for (int d = 0; d < 128; ++d) {
                    if (d < HEAD_DIM) {
                        q_row[d] = qp[d];
                    } else {
                        q_row[d] = 0.0h;
                    }
                }
            } else {
                #pragma clang loop unroll(full)
                for (int d = 0; d < 128; ++d) q_row[d] = 0.0h;
            }

            threadgroup half Ktile[BLOCK_M * HEAD_DIM];
            threadgroup half Vtile[BLOCK_M * HEAD_DIM];

            const uint tile_elems = uint(BLOCK_M) * uint(HEAD_DIM);
            for (uint e = tid_x; e < tile_elems; e += ntids_x) {
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

            float m_in = -INFINITY;
            float l_in = 0.0f;
            float a_in[128];
            #pragma clang loop unroll(full)
            for (int d = 0; d < 128; ++d) a_in[d] = 0.0f;

            if (row_active) {
                float qk[32];
                #pragma clang loop unroll(full)
                for (int j = 0; j < 32; ++j) qk[j] = -INFINITY;

                for (int j = 0; j < B; ++j) {
                    float s = 0.0f;
                    threadgroup const half* krow = Ktile + j * HEAD_DIM;
                    #pragma clang loop unroll(full)
                    for (int d = 0; d < 128; ++d) {
                        if (d < HEAD_DIM) {
                            s = fma(float(q_row[d]), float(krow[d]), s);
                        }
                    }
                    qk[j] = s * P.sm_scale;
                }
                for (int j = 0; j < B; ++j) m_in = max(m_in, qk[j]);

                for (int j = 0; j < B; ++j) {
                    const float p = fast::exp2(qk[j] - m_in);
                    l_in += p;
                    threadgroup const half* vrow = Vtile + j * HEAD_DIM;
                    #pragma clang loop unroll(full)
                    for (int d = 0; d < 128; ++d) {
                        if (d < HEAD_DIM) {
                            a_in[d] = fma(p, float(vrow[d]), a_in[d]);
                        }
                    }
                }
                const float l_safe = (l_in == 0.0f) ? 1.0f : l_in;
                #pragma clang loop unroll(full)
                for (int d = 0; d < 128; ++d) a_in[d] /= l_safe;

                const float L_in = m_in * float(M_LN2_F) + log(l_safe);

                const uint qrow_idx = seq * uint(B) + tid_x;
                const uint slot     = qrow_idx * uint(P.num_q_heads) + qh;
                const float L_out   = logsumexp[slot];

                const float m_c   = max(L_out, L_in);
                const float w_out = fast::exp(L_out - m_c);
                const float w_in  = fast::exp(L_in  - m_c);
                const float denom = w_out + w_in;

                const bool skip_this_row =
                    (P.use_dirty_gate != 0)
                    && (dirty_mask[seq * uint(B) + tid_x] == 0);

                device half* op =
                    O + (qrow_idx * uint(P.num_q_heads) + qh) * uint(HEAD_DIM);

                if (skip_this_row) {
                    device const float* aop =
                        attnOutPast + slot * uint(HEAD_DIM);
                    #pragma clang loop unroll(full)
                    for (int d = 0; d < 128; ++d) {
                        if (d < HEAD_DIM) {
                            op[d] = half(aop[d]);
                        }
                    }
                } else {
                    device const float* aop =
                        attnOutPast + slot * uint(HEAD_DIM);
                    #pragma clang loop unroll(full)
                    for (int d = 0; d < 128; ++d) {
                        if (d < HEAD_DIM) {
                            const float mixed = (w_out * aop[d] + w_in * a_in[d]) / denom;
                            op[d] = half(mixed);
                        }
                    }
                }
            }
        """

        self.externalKernel = MLXFast.metalKernel(
            name: "flashblock_external_pass",
            inputNames: ["Q", "K_cur", "V_cur", "K_page", "V_page", "block_tables", "ctx_lens"],
            outputNames: ["O", "attnOutPast", "logsumexp"],
            source: externalSource,
            header: header
        )

        self.internalKernel = MLXFast.metalKernel(
            name: "flashblock_internal_and_compose",
            inputNames: ["Q", "K_cur", "V_cur", "attnOutPast", "logsumexp", "dirty_mask"],
            outputNames: ["O"],
            source: internalSource,
            header: header
        )
    }

    public func chooseStepKind(dirtyPerSeq: [Int], isFirstStepOfBlock: Bool) -> StepKind {
        if isFirstStepOfBlock { return .refreshCache }
        let anyOverTau = dirtyPerSeq.contains { $0 >= config.tau }
        return anyOverTau ? .refreshCache : .reuseCache
    }

    public func forward(
        kind: StepKind,
        Q: MLXArray,
        Kcur: MLXArray,
        Vcur: MLXArray,
        kCache: MLXArray,
        vCache: MLXArray,
        blockTables: MLXArray,
        ctxLens: MLXArray,
        attnOutPast: MLXArray,
        logsumexp: MLXArray,
        dirtyMask: MLXArray
    ) -> (O: MLXArray, attnOutPast: MLXArray, logsumexp: MLXArray) {
        
        switch kind {
        case .refreshCache:
            let results = externalKernel(
                [Q, Kcur, Vcur, kCache, vCache, blockTables, ctxLens],
                grid: (config.numSeqs, config.numQHeads, 1),
                threadGroup: (config.blockM, 1, 1),
                outputShapes: [
                    [config.nqTotal, config.numQHeads, config.headDim], // O
                    [config.nqTotal, config.numQHeads, config.headDim], // attnOutPast
                    [config.nqTotal, config.numQHeads]                  // logsumexp
                ],
                outputDTypes: [.float16, .float32, .float32]
            )
            return (results[0], results[1], results[2])

        case .reuseCache:
            let results = internalKernel(
                [Q, Kcur, Vcur, attnOutPast, logsumexp, dirtyMask],
                grid: (config.numSeqs, config.numQHeads, 1),
                threadGroup: (config.blockM, 1, 1),
                outputShapes: [
                    [config.nqTotal, config.numQHeads, config.headDim]  // O
                ],
                outputDTypes: [.float16]
            )
            return (results[0], attnOutPast, logsumexp)
        }
    }
}
