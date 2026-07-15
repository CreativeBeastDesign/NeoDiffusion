import Foundation

/// **Diagnostic only — never a serving feature.** Skips a module inside a *real* forward so that
/// module's in-situ cost can be measured by subtraction.
///
/// ## Why this exists
///
/// `LLaDAMoEDispatchBench` times ops as `for _ in 0..<reps { eval(body()) }`, paying a full
/// graph-eval/sync per rep that production never pays — a real forward fuses its ops into one lazy
/// graph. The resulting inflation is **host-dependent**: negligible against the M1's slow compute,
/// dominant against the Studio's ~5–10× faster compute. Extrapolating those numbers to a forward
/// exceeds the forward itself (19 MoE layers × 2.33 ms = 44.3 ms vs a measured 27.5 ms), so the
/// microbench cannot size any module's share on the Studio. See `Plans/gather_qmm_handoff.md` §5.
///
/// Ablation sidesteps that: timing a **whole forward** with and without a module makes any fixed
/// harness cost common-mode, so it cancels in the delta.
///
/// ## How to read the results
///
/// The metric is **ms per forward** (`denoiseSeconds / forwardsEvaluated`), *not* TPS. Ablated arms
/// emit garbage and their denoising trajectory diverges (different steps/block, different token
/// counts) — that is expected and harmless, because every forward is the same graph at T=32
/// regardless of what the tokens say. Do not compare TPS across ablation arms; it is meaningless.
///
/// Known bias (*inferred*, stated rather than corrected): a skipped MoE leaves ~9.5 GB of expert
/// weights untouched, so the *rest* of the forward may run marginally faster in ablated arms. That
/// makes `none − moeRoutedExperts` a slight **over**-estimate — conservative in the right direction
/// for a ceiling.
public enum ModuleAblation: String, Sendable, Codable, CaseIterable {
    /// Production path. The only value reachable from the server.
    case none

    /// Skip the router **and** the routed-expert GEMMs; keep the shared expert. Delta from
    /// ``none`` bounds *router + GEMMs together* — measured at **56.2%** of the served forward
    /// (2026-07-14). Use ``moeExpertGEMMs`` to split that.
    case moeRoutedExperts

    /// Skip **only** the expert GEMMs (the 57 `gatherQuantizedMM` calls per forward); the router
    /// still runs in full. `none − moeExpertGEMMs` isolates `gather_qmm` itself — the actual
    /// subject of the dequant-overhead question — where ``moeRoutedExperts`` only bounds it
    /// together with the router.
    ///
    /// **Not dead-code-eliminable, by construction.** The stand-in keeps the real combine
    /// arithmetic and consumes `weights`, and `weights = takeAlong(scores, indices)` depends on
    /// the entire top-k/group-limited selection chain — so the whole router is forced. Verified
    /// behaviourally (not by reasoning about MLX's optimiser) in `ModuleAblationTests`: this arm's
    /// output must differ from ``moeRoutedExperts``, which is only possible if the router ran.
    case moeExpertGEMMs

    /// Skip the whole MoE block (identity). Delta from ``moeRoutedExperts`` = shared-expert cost.
    case moeAll

    /// As ``moeExpertGEMMs`` (no expert GEMMs), **and** replace the group-limited top-k selection
    /// with a fixed index set — the router's matmul + sigmoid + expert-bias still run, only the
    /// two `argSort`s and the group masking are skipped.
    ///
    /// `moeExpertGEMMs − moeRouterNoTopK` = the **selection** cost; `moeRouterNoTopK −
    /// moeRoutedExperts` = the **matmul/sigmoid** cost. This matters because the router currently
    /// does `argSort(-maskedScores)[..<topK]` — a *full sort of all 256 experts per token* to take
    /// 8. If selection dominates the router's 13.8%, `argPartition` (O(E), available in MLX-Swift
    /// `Ops.swift:257`) is a few-line change with no kernel and no mlx-swift fork.
    ///
    /// Not eliminable: `weights = takeAlong(scores, fixedIndices)` still consumes `scores`, which
    /// forces the matmul chain.
    case moeRouterNoTopK

    /// Replace the LM head's output with a broadcast constant of the same shape. Delta from
    /// ``none`` = the lm_head projection's cost, splitting the 25.2% "remainder" (lm_head + norms
    /// + loop overhead) that §5.7 could only attribute by subtraction.
    ///
    /// The broadcast is not materialised, so this measures the [H → 157184] projection itself.
    /// Sampling then reads a constant and picks garbage — irrelevant, since the metric is
    /// ms/forward.
    case lmHead

    /// Skip the attention residual add. Delta from ``none`` is attention's **total** cost including
    /// KV-cache growth (the cache never fills in this arm) — not "attention math only".
    case attention
}
