// FlashBlockRunner.swift
//
// Pure-Swift asynchronous FlashBlock runner using MLXFast.metalKernel.
// Zero-copy native tensor layout, 4-wide vector loads/FMAs, and multi-head threadgroup tiling.

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
    public var blockN: Int              // 32, 64, or 128

    // Reuse gates
    public var tau: Int                 // per-block dirty-token threshold
    public var composeGamma: Float      // per-head similarity threshold

    public var dtype: DType

    public init(numSeqs: Int, blockLen: Int,
                numQHeads: Int, numKVHeads: Int, headDim: Int,
                pageSize: Int = 256, maxPagesPerSeq: Int = 64,
                blockM: Int = 32, blockN: Int = 32,
                tau: Int = 4, composeGamma: Float = 0.0,
                dtype: DType = .float16) {
        precondition(numQHeads % numKVHeads == 0, "GQA: numQHeads must be a multiple of numKVHeads")
        precondition(blockLen <= blockM, "blockLen must be <= blockM")
        precondition([32, 64, 96, 128].contains(headDim), "headDim must be 32/64/96/128")
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
        self.dtype = dtype
    }

    public var kvGroup: Int { numQHeads / numKVHeads }
    public var nqTotal: Int { numSeqs * blockLen }

    /// Multi-head threadgroup tiling: when heads are even and share a KV group, pack 2 heads per threadgroup
    /// to share KV tile loads in threadgroup memory and boost occupancy to 64 threads (2 simdgroups).
    public var headsPerThreadGroup: Int {
        (numQHeads % 2 == 0 && kvGroup >= 2) ? 2 : 1
    }
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

        let dTypeName = (config.dtype == .float32) ? "float" : "half"
        let dType4Name = (config.dtype == .float32) ? "float4" : "half4"
        let smScale = (1.0 / Float(config.headDim).squareRoot()) * Float(log2(M_E))

        let header = """
        #include <metal_stdlib>
        using namespace metal;

        #define DTYPE \(dTypeName)
        #define DTYPE4 \(dType4Name)
        #define BLOCK_M \(config.blockM)
        #define BLOCK_N \(config.blockN)
        #define HEAD_DIM \(config.headDim)
        #define KV_GROUP \(config.kvGroup)
        #define H_PER_TG \(config.headsPerThreadGroup)
        #define SM_SCALE \(smScale)f
        #define VEC_D (HEAD_DIM / 4)

        """

        let externalSource = """
            const uint3 tgid = threadgroup_position_in_grid;
            const uint3 tid = thread_position_in_threadgroup;
            const uint3 ntids = threads_per_threadgroup;

            const uint seq = tgid.x;
            const uint qh  = tgid.y * uint(H_PER_TG) + tid.y;
            const uint kh  = qh / uint(KV_GROUP);
            const int  Nk  = ctx_lens[seq];
            const int  B   = \(config.blockLen);
            const uint tid_x = tid.x;

            const uint thread_id = tid.y * ntids.x + tid.x;
            const uint total_threads = ntids.y * ntids.x;

            threadgroup DTYPE4 Ktile4[(BLOCK_N * HEAD_DIM) / 4];
            threadgroup DTYPE4 Vtile4[(BLOCK_N * HEAD_DIM) / 4];
            threadgroup DTYPE* Ktile = reinterpret_cast<threadgroup DTYPE*>(Ktile4);
            threadgroup DTYPE* Vtile = reinterpret_cast<threadgroup DTYPE*>(Vtile4);

            float m_i = -INFINITY;
            float l_i = 0.0f;
            float4 acc4[VEC_D];
            #pragma clang loop unroll(full)
            for (int d = 0; d < VEC_D; ++d) acc4[d] = float4(0.0f);

            const bool row_active = (int(tid_x) < B) && (qh < uint(\(config.numQHeads)));
            DTYPE4 q_row4[VEC_D];
            if (row_active) {
                device const DTYPE4* qp4 = reinterpret_cast<device const DTYPE4*>(
                    Q + ((seq * uint(\(config.numQHeads)) + qh) * uint(B) + tid_x) * uint(HEAD_DIM)
                );
                #pragma clang loop unroll(full)
                for (int d = 0; d < VEC_D; ++d) {
                    q_row4[d] = qp4[d];
                }
            } else {
                #pragma clang loop unroll(full)
                for (int d = 0; d < VEC_D; ++d) q_row4[d] = DTYPE4(0.0);
            }

            const uint total_vec = (uint(BLOCK_N) * uint(HEAD_DIM)) / 4;
            device const DTYPE4* k_cache4 = reinterpret_cast<device const DTYPE4*>(
                K_cache + (seq * uint(\(config.numKVHeads)) + kh) * uint(Nk) * uint(HEAD_DIM)
            );
            device const DTYPE4* v_cache4 = reinterpret_cast<device const DTYPE4*>(
                V_cache + (seq * uint(\(config.numKVHeads)) + kh) * uint(Nk) * uint(HEAD_DIM)
            );

            // 1. Prefix KV cache streaming loop
            for (int tn = 0; tn < Nk; tn += BLOCK_N) {
                const int actual = min(Nk - tn, BLOCK_N);
                const uint valid_vec = (uint(actual) * uint(HEAD_DIM)) / 4;
                const uint tn_vec    = (uint(tn) * uint(HEAD_DIM)) / 4;

                for (uint v = thread_id; v < total_vec; v += total_threads) {
                    if (v < valid_vec) {
                        Ktile4[v] = k_cache4[tn_vec + v];
                        Vtile4[v] = v_cache4[tn_vec + v];
                    } else {
                        Ktile4[v] = DTYPE4(0.0);
                        Vtile4[v] = DTYPE4(0.0);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (row_active) {
                    float qk[BLOCK_N];
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < BLOCK_N; ++j) qk[j] = -INFINITY;

                    for (int j = 0; j < actual; ++j) {
                        float s = 0.0f;
                        threadgroup const DTYPE4* krow4 = reinterpret_cast<threadgroup const DTYPE4*>(Ktile + j * HEAD_DIM);
                        #pragma clang loop unroll(full)
                        for (int d = 0; d < VEC_D; ++d) {
                            s += dot(float4(q_row4[d]), float4(krow4[d]));
                        }
                        qk[j] = s * SM_SCALE;
                    }

                    float m_new = m_i;
                    for (int j = 0; j < actual; ++j) m_new = max(m_new, qk[j]);
                    const float alpha = fast::exp2(m_i - m_new);
                    l_i *= alpha;
                    #pragma clang loop unroll(full)
                    for (int d = 0; d < VEC_D; ++d) acc4[d] *= alpha;

                    for (int j = 0; j < actual; ++j) {
                        const float p = fast::exp2(qk[j] - m_new);
                        l_i += p;
                        threadgroup const DTYPE4* vrow4 = reinterpret_cast<threadgroup const DTYPE4*>(Vtile + j * HEAD_DIM);
                        #pragma clang loop unroll(full)
                        for (int d = 0; d < VEC_D; ++d) {
                            acc4[d] = fma(float4(p), float4(vrow4[d]), acc4[d]);
                        }
                    }
                    m_i = m_new;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Stash cached prefix attention accumulators
            if (row_active) {
                const uint slot = ((seq * uint(\(config.numQHeads)) + qh) * uint(B) + tid_x);
                const float l_safe = (l_i == 0.0f) ? 1.0f : l_i;
                device float4* aop4 = reinterpret_cast<device float4*>(attnOutPast + slot * uint(HEAD_DIM));
                #pragma clang loop unroll(full)
                for (int d = 0; d < VEC_D; ++d) {
                    aop4[d] = acc4[d] / l_safe;
                }
                logsumexp[slot] = m_i * float(M_LN2_F) + log(l_safe);
            }

            // 2. Active block KV streaming loop
            device const DTYPE4* k_cur4 = reinterpret_cast<device const DTYPE4*>(
                K_cur + (seq * uint(\(config.numKVHeads)) + kh) * uint(B) * uint(HEAD_DIM)
            );
            device const DTYPE4* v_cur4 = reinterpret_cast<device const DTYPE4*>(
                V_cur + (seq * uint(\(config.numKVHeads)) + kh) * uint(B) * uint(HEAD_DIM)
            );

            for (int tn = 0; tn < B; tn += BLOCK_N) {
                const int actual = min(B - tn, BLOCK_N);
                const uint valid_vec = (uint(actual) * uint(HEAD_DIM)) / 4;
                const uint tn_vec    = (uint(tn) * uint(HEAD_DIM)) / 4;

                for (uint v = thread_id; v < total_vec; v += total_threads) {
                    if (v < valid_vec) {
                        Ktile4[v] = k_cur4[tn_vec + v];
                        Vtile4[v] = v_cur4[tn_vec + v];
                    } else {
                        Ktile4[v] = DTYPE4(0.0);
                        Vtile4[v] = DTYPE4(0.0);
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (row_active) {
                    float qk[BLOCK_N];
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < BLOCK_N; ++j) qk[j] = -INFINITY;

                    for (int j = 0; j < actual; ++j) {
                        float s = 0.0f;
                        threadgroup const DTYPE4* krow4 = reinterpret_cast<threadgroup const DTYPE4*>(Ktile + j * HEAD_DIM);
                        #pragma clang loop unroll(full)
                        for (int d = 0; d < VEC_D; ++d) {
                            s += dot(float4(q_row4[d]), float4(krow4[d]));
                        }
                        qk[j] = s * SM_SCALE;
                    }

                    float m_new = m_i;
                    for (int j = 0; j < actual; ++j) m_new = max(m_new, qk[j]);
                    const float alpha = fast::exp2(m_i - m_new);
                    l_i *= alpha;
                    #pragma clang loop unroll(full)
                    for (int d = 0; d < VEC_D; ++d) acc4[d] *= alpha;

                    for (int j = 0; j < actual; ++j) {
                        const float p = fast::exp2(qk[j] - m_new);
                        l_i += p;
                        threadgroup const DTYPE4* vrow4 = reinterpret_cast<threadgroup const DTYPE4*>(Vtile + j * HEAD_DIM);
                        #pragma clang loop unroll(full)
                        for (int d = 0; d < VEC_D; ++d) {
                            acc4[d] = fma(float4(p), float4(vrow4[d]), acc4[d]);
                        }
                    }
                    m_i = m_new;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Write final normalized output O in native layout [B, L, H * D]
            if (row_active) {
                device DTYPE4* op4 = reinterpret_cast<device DTYPE4*>(
                    O + ((seq * uint(B) + tid_x) * uint(\(config.numQHeads)) + qh) * uint(HEAD_DIM)
                );
                const float l_safe = (l_i == 0.0f) ? 1.0f : l_i;
                #pragma clang loop unroll(full)
                for (int d = 0; d < VEC_D; ++d) {
                    op4[d] = DTYPE4(acc4[d] / l_safe);
                }
            }
        """

        let internalSource = """
            const uint3 tgid = threadgroup_position_in_grid;
            const uint3 tid = thread_position_in_threadgroup;
            const uint3 ntids = threads_per_threadgroup;

            const uint seq = tgid.x;
            const uint qh  = tgid.y * uint(H_PER_TG) + tid.y;
            const uint kh  = qh / uint(KV_GROUP);
            const int  B   = \(config.blockLen);
            const uint tid_x = tid.x;

            const uint thread_id = tid.y * ntids.x + tid.x;
            const uint total_threads = ntids.y * ntids.x;

            const bool row_active = (int(tid_x) < B) && (qh < uint(\(config.numQHeads)));
            DTYPE4 q_row4[VEC_D];
            if (row_active) {
                device const DTYPE4* qp4 = reinterpret_cast<device const DTYPE4*>(
                    Q + ((seq * uint(\(config.numQHeads)) + qh) * uint(B) + tid_x) * uint(HEAD_DIM)
                );
                #pragma clang loop unroll(full)
                for (int d = 0; d < VEC_D; ++d) {
                    q_row4[d] = qp4[d];
                }
            } else {
                #pragma clang loop unroll(full)
                for (int d = 0; d < VEC_D; ++d) q_row4[d] = DTYPE4(0.0);
            }

            threadgroup DTYPE4 Ktile4[(BLOCK_M * HEAD_DIM) / 4];
            threadgroup DTYPE4 Vtile4[(BLOCK_M * HEAD_DIM) / 4];
            threadgroup DTYPE* Ktile = reinterpret_cast<threadgroup DTYPE*>(Ktile4);
            threadgroup DTYPE* Vtile = reinterpret_cast<threadgroup DTYPE*>(Vtile4);

            const uint total_vec = (uint(BLOCK_M) * uint(HEAD_DIM)) / 4;
            const uint valid_vec = (uint(B) * uint(HEAD_DIM)) / 4;

            device const DTYPE4* k_cur4 = reinterpret_cast<device const DTYPE4*>(
                K_cur + (seq * uint(\(config.numKVHeads)) + kh) * uint(B) * uint(HEAD_DIM)
            );
            device const DTYPE4* v_cur4 = reinterpret_cast<device const DTYPE4*>(
                V_cur + (seq * uint(\(config.numKVHeads)) + kh) * uint(B) * uint(HEAD_DIM)
            );

            for (uint v = thread_id; v < total_vec; v += total_threads) {
                if (v < valid_vec) {
                    Ktile4[v] = k_cur4[v];
                    Vtile4[v] = v_cur4[v];
                } else {
                    Ktile4[v] = DTYPE4(0.0);
                    Vtile4[v] = DTYPE4(0.0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float m_in = -INFINITY;
            float l_in = 0.0f;
            float4 a_in4[VEC_D];
            #pragma clang loop unroll(full)
            for (int d = 0; d < VEC_D; ++d) a_in4[d] = float4(0.0f);

            if (row_active) {
                float qk[BLOCK_M];
                #pragma clang loop unroll(full)
                for (int j = 0; j < BLOCK_M; ++j) qk[j] = -INFINITY;

                for (int j = 0; j < B; ++j) {
                    float s = 0.0f;
                    threadgroup const DTYPE4* krow4 = reinterpret_cast<threadgroup const DTYPE4*>(Ktile + j * HEAD_DIM);
                    #pragma clang loop unroll(full)
                    for (int d = 0; d < VEC_D; ++d) {
                        s += dot(float4(q_row4[d]), float4(krow4[d]));
                    }
                    qk[j] = s * SM_SCALE;
                }
                for (int j = 0; j < B; ++j) m_in = max(m_in, qk[j]);

                for (int j = 0; j < B; ++j) {
                    const float p = fast::exp2(qk[j] - m_in);
                    l_in += p;
                    threadgroup const DTYPE4* vrow4 = reinterpret_cast<threadgroup const DTYPE4*>(Vtile + j * HEAD_DIM);
                    #pragma clang loop unroll(full)
                    for (int d = 0; d < VEC_D; ++d) {
                        a_in4[d] = fma(float4(p), float4(vrow4[d]), a_in4[d]);
                    }
                }

                const float l_safe = (l_in == 0.0f) ? 1.0f : l_in;
                #pragma clang loop unroll(full)
                for (int d = 0; d < VEC_D; ++d) a_in4[d] /= l_safe;

                const float L_in = m_in * float(M_LN2_F) + log(l_safe);

                const uint slot   = (seq * uint(\(config.numQHeads)) + qh) * uint(B) + tid_x;
                const float L_out = logsumexp[slot];

                const float m_c   = max(L_out, L_in);
                const float w_out = fast::exp(L_out - m_c);
                const float w_in  = fast::exp(L_in  - m_c);
                const float denom = w_out + w_in;

                device DTYPE4* op4 = reinterpret_cast<device DTYPE4*>(
                    O + ((seq * uint(B) + tid_x) * uint(\(config.numQHeads)) + qh) * uint(HEAD_DIM)
                );
                device const float4* aop4 = reinterpret_cast<device const float4*>(attnOutPast + slot * uint(HEAD_DIM));
                #pragma clang loop unroll(full)
                for (int d = 0; d < VEC_D; ++d) {
                    float4 mixed = (float4(w_out) * aop4[d] + float4(w_in) * a_in4[d]) / float4(denom);
                    op4[d] = DTYPE4(mixed);
                }
            }
        """

        self.externalKernel = MLXFast.metalKernel(
            name: "flashblock_external_pass",
            inputNames: ["Q", "K_cur", "V_cur", "K_cache", "V_cache", "ctx_lens"],
            outputNames: ["O", "attnOutPast", "logsumexp"],
            source: externalSource,
            header: header
        )

        self.internalKernel = MLXFast.metalKernel(
            name: "flashblock_internal_and_compose",
            inputNames: ["Q", "K_cur", "V_cur", "attnOutPast", "logsumexp"],
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
        Q: MLXArray,           // Native [B, numQHeads, L, headDim]
        Kcur: MLXArray,        // Native [B, numKVHeads, L, headDim]
        Vcur: MLXArray,        // Native [B, numKVHeads, L, headDim]
        kCache: MLXArray?,     // Native [B, numKVHeads, committedLen, headDim]
        vCache: MLXArray?,     // Native [B, numKVHeads, committedLen, headDim]
        ctxLens: MLXArray,     // [B] Int32
        attnOutPast: MLXArray, // [B, numQHeads, L, headDim] (FP32)
        logsumexp: MLXArray    // [B, numQHeads, L] (FP32)
    ) -> (O: MLXArray, attnOutPast: MLXArray, logsumexp: MLXArray) {
        
        let tgSize = (config.blockM, config.headsPerThreadGroup, 1)
        let gridDim = (config.numSeqs * config.blockM, config.numQHeads, 1)

        switch kind {
        case .refreshCache:
            guard let kCache = kCache, let vCache = vCache else {
                fatalError("refreshCache requires non-nil kCache and vCache")
            }
            let results = externalKernel(
                [Q, Kcur, Vcur, kCache, vCache, ctxLens],
                grid: gridDim,
                threadGroup: tgSize,
                outputShapes: [
                    [config.numSeqs, config.blockLen, config.numQHeads * config.headDim], // O [B, L, H * D]
                    [config.numSeqs, config.numQHeads, config.blockLen, config.headDim],  // attnOutPast
                    [config.numSeqs, config.numQHeads, config.blockLen]                   // logsumexp
                ],
                outputDTypes: [config.dtype, .float32, .float32]
            )
            return (results[0], results[1], results[2])

        case .reuseCache:
            let results = internalKernel(
                [Q, Kcur, Vcur, attnOutPast, logsumexp],
                grid: gridDim,
                threadGroup: tgSize,
                outputShapes: [
                    [config.numSeqs, config.blockLen, config.numQHeads * config.headDim]  // O [B, L, H * D]
                ],
                outputDTypes: [config.dtype]
            )
            return (results[0], attnOutPast, logsumexp)
        }
    }
}
