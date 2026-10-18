In MoE diffusion LMs, the FFN can be targeted at **three different levels**: cache/reuse expert outputs, reduce the number of unique experts loaded, or avoid FFN execution for stable tokens. The strongest diffusion-specific evidence currently favors **block-level expert-set reuse plus token-selective FFN activation**, rather than naïvely caching the complete MoE output.

## Available approaches

| Target           | Representative work              | Mechanism                                                                          | Trade-off                                                          |
| ---------------- | -------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| FFN outputs      | DyLLM, DARE                      | Reuse the previous FFN output for non-salient or stable tokens                     | Approximate; stale residuals propagate into subsequent layers      |
| Expert branches  | MoECa, conceptually transferable | Cache individual \((token, expert)\) outputs rather than the aggregated MoE output | Finer reuse, but cache indexing and storage become substantial     |
| Unique experts   | DES, dMoE, TEAM                  | Restrict routing to a shared block/sequence-level expert coreset                   | Reduces weight traffic and kernel fragmentation; may alter routing |
| Expert budget    | REFLEX                           | Assign fewer experts to already decoded, cold, or low-priority tokens              | Saves actual FLOPs; approximate                                    |
| Expert residency | TIDE, OLED-MoE, WiSP             | Cache frequently reused expert weights in GPU memory                               | Lossless, but primarily removes CPU–GPU I/O rather than GEMM work  |
| FFN precision    | LaCache, ITC-MoE                 | FP8 or low-rank expert weights/activations                                         | Usually easier than output caching, but requires suitable kernels  |

## FFN-output caching

DyLLM directly skips the FFN for non-salient tokens and reuses their cached previous-step FFN output, while recomputing only salient tokens. DARE similarly extends diffusion caching beyond KV states to output activations, using query variation as a predictor of activation redundancy. [arxiv](https://arxiv.org/html/2603.08026v2)

For an MoE layer, this could be refined to branch granularity:

\[
y_{i,l}^{(t)}
=

\sum_{e\in R_{i,l}^{(t)}}g_{i,l,e}^{(t)}
E_{l,e}\!\left(h_{i,l}^{(t)}\right)
\]

Cache each expert result

\[
C_{i,l,e}=E_{l,e}\!\left(h_{i,l}^{(t_{\mathrm{last}})}\right)
\]

and recompute it only when:

- expert \(e\) newly enters the route;
- the token hidden-state drift exceeds a threshold;
- the router weight changes substantially;
- the token was just unmasked;
- a maximum cache age is reached.

Branch-level caching is preferable to caching the aggregated MoE output because routing weights and selected experts may change independently. MoECa demonstrates this exact granularity for MoE diffusion transformers: cross-timestep redundancy is better represented at the expert-branch level than at whole-token level, reaching up to 2.83× acceleration in its visual-DiT setting. It is not yet direct proof for language diffusion, but the mechanism transfers naturally. [arxiv](https://arxiv.org/pdf/2606.15615v1.pdf)

The downside is important: if top-\(k\) routing activates many different experts across tokens, the activation cache can become large and irregular. Caching every branch can exchange FFN compute for HBM traffic, potentially making performance worse.

## Expert-set consolidation

The most directly applicable method is **Dynamic Expert Sharing**. DES first chooses a sequence- or block-level expert coreset and then restricts each token’s local top-\(k\) routing to that set; it reports more than 55% fewer unique expert activations, up to 38% lower latency, and 99% retained accuracy. [arxiv](https://arxiv.org/abs/2602.00879)

This matters because parallel diffusion decoding causes an “expert explosion”: each token may activate only a few experts, but the union across the whole block can approach most of the expert pool. A shared coreset reduces:

- expert-weight reads from HBM;
- all-to-all or expert-parallel communication;
- number of grouped-GEMM partitions;
- CPU–GPU transfers under expert offloading;
- kernel-launch and small-batch fragmentation.

TEAM combines temporal and spatial routing consistency with token state. It applies delayed computation to decoded tokens, treats masked tokens as hot or cold, and reroutes cold tokens toward experts already activated by useful tokens; it reports 35–39% fewer activated experts and up to 2.2× speedup. [arxiv](https://arxiv.org/html/2602.08404v1)

dMoE pushes this further through learnable block-level routing. It aggregates token routing distributions into a shared block distribution and then performs token-level selection within the resulting coreset; reported unique experts fall from 69.5 to 16.5, with approximately 74–77% lower memory and 1.36–2.04× lower latency. Unlike DES and TEAM, this approach introduces training or adaptation rather than remaining purely runtime-level. [arxiv](https://arxiv.org/html/2605.30876)

## Variable expert budgets

Not every token needs the same top-\(k\). A useful diffusion-specific allocation would be:

- **Stable prefix/decoded token:** zero experts if an FFN output is reusable, otherwise top-1.
- **Cold masked token:** top-1 from the current resident coreset.
- **Hot masked token near commitment:** normal top-\(k\).
- **Newly changed or high-drift token:** potentially larger top-\(k\).
- **Refresh iteration:** restore normal routing for cache correction.

REFLEX implements this general direction without replacing the model router: it retains router ordering but changes how many experts each token receives according to block-relative refinement role and progress toward commitment. It reports 15% fewer expert-token computations on average while generally preserving quality. [semanticscholar](https://www.semanticscholar.org/paper/6f09da09efb90bcae5e4dd5943c3b3128580a59b)

This is attractive because it targets actual FFN FLOPs, unlike expert-weight caching, which targets only memory movement.

## Expert-weight caching

If experts do not fit in VRAM, the major cost may be **loading their weights**, not executing their matrix multiplications. In that regime:

- **TIDE** observes cross-step routing stability, keeps a GPU-resident expert set for several denoising iterations, and refreshes placement at optimised intervals. Routing and outputs remain unchanged, making it lossless; reported throughput improvements are up to 1.4–1.5× on LLaDA2.0 MoE models. [arxiv](https://arxiv.org/html/2605.20179)
- **OLED-MoE** retains experts predicted to be useful in the next denoising iteration and handles misses through cooperative CPU–GPU execution. It reports operation near full-residency performance with only 40% of expert GPU memory. [arxiv](https://arxiv.org/abs/2609.33385)
- **WiSP** treats expert residency as working-set paging and reports token-identical inference with up to 4× lower resident expert memory on diffusion MoEs. [arxiv](https://arxiv.org/html/2606.21868v2)

These methods are the safest starting point because they do not perturb numerical semantics. But they offer little benefit when all experts already reside in VRAM and expert GEMMs—not transfer—dominate runtime.

## Recommended design

For a new inference optimisation, I would combine:

1. **Block-level expert coreset:** aggregate router scores over the active diffusion block and retain the smallest set covering a target router mass.
2. **Per-token constrained routing:** preserve each token’s original top-\(k\) ordering inside that coreset.
3. **Branch cache:** cache outputs by \((layer, position, expert)\).
4. **Drift-triggered refresh:** recompute when hidden-state cosine/KL drift, router divergence, cache age, or token-state changes exceed thresholds.
5. **Dynamic expert count:** use top-1 or cached output for stable tokens and full top-\(k\) for the refinement frontier.
6. **Resident-set coupling:** keep coreset experts in GPU memory across neighbouring denoising iterations.
7. **Periodic exact step:** execute unrestricted routing and refresh cached branches to bound accumulated error.

The critical point is to optimise against the real bottleneck. If profiling shows high expert-weight HBM/PCIe traffic, target **expert-set consolidation and residency**. If grouped GEMMs dominate, target **variable top-\(k\) and branch-output reuse**. If the FFN is already bandwidth-bound and cached outputs are as large as its activations, whole-output caching may simply replace weight traffic with activation traffic and provide little gain. LaCache’s use of per-group FP8 FFN execution is a lower-risk alternative in that situation. [arxiv](https://arxiv.org/abs/2607.16339)
