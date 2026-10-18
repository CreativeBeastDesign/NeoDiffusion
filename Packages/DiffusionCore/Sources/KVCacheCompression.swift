import Foundation
import MLX

/// KV Cache Quantization mode for Inter-Block Prefix Storage (2026 Dual Caching Tier 1 & Track 1 Ablations).
///
/// **Naming caveat (errata 2026-10-18)**: every mode here is MLX *affine integer* group quantization
/// (`MLX.quantized` default mode: per-group scale + bias). `fp8`/`fp4`/`fp4g32` are historical names
/// for **int8 / int4 / int4-g32** — there is no floating-point FP8/FP4 format anywhere in this file.
/// The raw values are kept because they are recorded in bench JSONL rows and CLI arm names; use
/// ``KVCacheQuantization/parse(_:)`` to also accept the accurate aliases `int8`, `int4`, `int4g32`.
///
/// - `none`: Full-precision storage (FP16 / BF16, 40,960 B/tok across 20 layers).
/// - `fp8` (`K8V8`): 8-bit quantized K and V with per-group scaling (group 64, 21,760 B/tok, -46.88% size).
/// - `k8v4` (`K8V4`): Asymmetric 8-bit Keys + 4-bit Values (group 64, 16,640 B/tok, -59.38% size; preserves QK^T softmax precision).
/// - `k4v8` (`K4V8`): Asymmetric 4-bit Keys + 8-bit Values (group 64, 16,640 B/tok, -59.38% size; 2x2 factorial counterpart to K8V4).
/// - `fp4` (`K4V4`): 4-bit packed K and V with per-group scaling (group 64, 11,520 B/tok, -71.88% size).
/// - `fp4g32` (`K4V4-g32`): 4-bit packed K and V with finer group-32 scaling (12,800 B/tok, -68.75% size).
/// - `kivi4` (`K4-chan-V4-tok`): KIVI dual-axis 4-bit quantization — Keys quantized per-channel along the 32-token block axis (group 32), Values quantized per-token along headDim (group 64) (12,160 B/tok, -70.31% size).
/// - `k8v4s4` (`K8V4-shallow4 + K4V4-deep16`): KVTuner-style depth-hybrid allocation — K8V4 in shallow layers 0..<4 and K4V4 in deeper layers 4..<20 (12,544 B/tok, -69.38% size).
public enum KVCacheQuantization: String, Sendable, Equatable, CaseIterable {
    case none
    case fp8
    case k8v4
    case k4v8
    case fp4
    case fp4g32
    case kivi4
    case k8v4s4
    
    /// Parses a mode from its raw value or an accurate integer alias (`int8`→`fp8`, `int4`→`fp4`,
    /// `int4g32`→`fp4g32`). Returns `nil` for unknown strings — callers must surface that, not ignore it.
    public static func parse(_ s: String) -> KVCacheQuantization? {
        switch s.lowercased() {
        case "int8": return .fp8
        case "int4": return .fp4
        case "int4g32": return .fp4g32
        default: return KVCacheQuantization(rawValue: s)
        }
    }

    /// Key bit-width. (Historically named `bits`; values may use a different width — see `valueBits`.)
    public var bits: Int { keyBits }
    
    public var keyBits: Int {
        switch self {
        case .none: return 16
        case .fp8, .k8v4, .k8v4s4: return 8
        case .k4v8, .fp4, .fp4g32, .kivi4: return 4
        }
    }
    
    public var valueBits: Int {
        switch self {
        case .none: return 16
        case .fp8, .k4v8: return 8
        case .k8v4, .fp4, .fp4g32, .kivi4, .k8v4s4: return 4
        }
    }
    
    public var keyGroupSize: Int {
        switch self {
        case .fp4g32, .kivi4: return 32
        default: return 64
        }
    }
    
    public var valueGroupSize: Int {
        switch self {
        case .fp4g32: return 32
        default: return 64
        }
    }
    
    public var keyPerChannel: Bool {
        self == .kivi4
    }
}

/// Container for compressed (quantized) prefix key or value tensors.
public struct CompressedKVTensor {
    public let packed: MLXArray       // uint32 packed tensor
    public let scales: MLXArray       // per-group float scales
    public let biases: MLXArray?      // per-group float biases
    public let groupSize: Int
    public let bits: Int
    public let originalShape: [Int]
    public let perChannel: Bool
    
    public init(
        packed: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int,
        originalShape: [Int],
        perChannel: Bool = false
    ) {
        self.packed = packed
        self.scales = scales
        self.biases = biases
        self.groupSize = groupSize
        self.bits = bits
        self.originalShape = originalShape
        self.perChannel = perChannel
    }
    
    /// Total bytes occupied by the quantized representation (packed + scales + biases).
    public var byteCount: Int {
        packed.nbytes + scales.nbytes + (biases?.nbytes ?? 0)
    }
    
    /// Compresses a full-precision tensor using fine-grained group quantization with explicit bit-width
    /// and optional per-channel quantization (transposing sequence axis 2 and head-dim axis 3).
    public static func compress(
        _ x: MLXArray,
        bits: Int,
        groupSize: Int = 64,
        perChannel: Bool = false
    ) -> CompressedKVTensor {
        let usePerChannel = perChannel && x.ndim == 4 && x.dim(2) % groupSize == 0
        let target = usePerChannel ? x.transposed(0, 1, 3, 2) : x
        let (q, scales, biases) = MLX.quantized(target, groupSize: groupSize, bits: bits)
        return CompressedKVTensor(
            packed: q,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits,
            originalShape: x.shape,
            perChannel: usePerChannel
        )
    }
    
    /// Incrementally quantizes a newly committed block `[B, nKV, blockLen, D]` and appends it along
    /// sequence axis 2 (or axis 3 in transposed per-channel layout) without re-quantizing or
    /// materializing the previously committed prefix in FP16.
    public func append(_ block: MLXArray) -> CompressedKVTensor {
        let usePerChannel = perChannel && block.ndim == 4 && block.dim(2) % groupSize == 0
        // A per-channel (transposed) tensor can only grow by whole groups along the sequence axis.
        // Before 2026-10-18 a misaligned append (e.g. a sub-block commit of n < groupSize tokens
        // onto a kivi4 prefix) silently took the non-transposed branch and concatenated two
        // different layouts — a shape crash at best, garbage K/V at worst. Fail loudly instead.
        precondition(usePerChannel == perChannel,
                     "CompressedKVTensor.append: per-channel tensor needs appends that are multiples of groupSize=\(groupSize) tokens (got \(block.dim(2))). kivi4 is incompatible with sub-block commit.")
        let target = usePerChannel ? block.transposed(0, 1, 3, 2) : block
        let concatAxis = usePerChannel ? 3 : 2
        let (qBlock, sBlock, bBlock) = MLX.quantized(target, groupSize: groupSize, bits: bits)
        let newPacked = concatenated([packed, qBlock], axis: concatAxis)
        let newScales = concatenated([scales, sBlock], axis: concatAxis)
        let newBiases: MLXArray?
        if let biases, let bBlock {
            newBiases = concatenated([biases, bBlock], axis: concatAxis)
        } else {
            newBiases = nil
        }
        var newShape = originalShape
        newShape[2] += block.dim(2)
        return CompressedKVTensor(
            packed: newPacked,
            scales: newScales,
            biases: newBiases,
            groupSize: groupSize,
            bits: bits,
            originalShape: newShape,
            perChannel: usePerChannel
        )
    }
    
    /// Keeps only the first `length` sequence positions (session prefix-cache truncation, exp-5B).
    ///
    /// Every group is quantized independently (`MLX.quantized` per group of `groupSize` along the
    /// quantized axis), and appends quantize each committed block on its own, so slicing whole
    /// groups yields exactly the tensor an `append` chain over the kept blocks would have built:
    /// - token-major layout (`perChannel == false`, packed `[.., L, D·bits/32]`, scales
    ///   `[.., L, D/g]`): slice axis 2 to `length` — any `length` is valid.
    /// - per-channel layout (kivi4 Keys, packed `[.., D, L·bits/32]`, scales `[.., D, L/g]`): slice
    ///   axis 3 to `length·bits/32` / `length/g`; `length` must be a multiple of `groupSize` (a
    ///   quantization group spans `groupSize` tokens) and of `32/bits` (packing). Block-aligned
    ///   truncation always satisfies this because kivi4 appends are themselves group-aligned.
    public func truncated(toLength length: Int) -> CompressedKVTensor {
        let current = originalShape[2]
        precondition(length > 0 && length <= current,
            "CompressedKVTensor.truncated: length \(length) outside (0, \(current)]")
        if length == current { return self }
        let packedLen: Int
        let groupsLen: Int
        let axis: Int
        if perChannel {
            precondition(length % groupSize == 0 && (length * bits) % 32 == 0,
                "CompressedKVTensor.truncated: per-channel tensor can only be cut at multiples of "
                + "groupSize=\(groupSize) tokens (got \(length))")
            axis = 3
            packedLen = length * bits / 32
            groupsLen = length / groupSize
        } else {
            axis = 2
            packedLen = length
            groupsLen = length
        }
        func cut(_ a: MLXArray, _ n: Int) -> MLXArray {
            axis == 3 ? a[0..., 0..., 0..., 0 ..< n] : a[0..., 0..., 0 ..< n, 0...]
        }
        var newShape = originalShape
        newShape[2] = length
        return CompressedKVTensor(
            packed: cut(packed, packedLen),
            scales: cut(scales, groupsLen),
            biases: biases.map { cut($0, groupsLen) },
            groupSize: groupSize,
            bits: bits,
            originalShape: newShape,
            perChannel: perChannel
        )
    }

    /// Dequantizes the compressed tensor back to full-precision FP16.
    public func dequantize(dtype: DType = .float16) -> MLXArray {
        let deq = dequantized(
            packed,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits,
            dtype: dtype
        )
        return perChannel ? deq.transposed(0, 1, 3, 2) : deq
    }
}
