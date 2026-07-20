# GPU Control Kernel V1 logbook — bit-exact fused selection/control

Campaign record for fusing the per-step selection/control chain (Γ/Δ, thresholding, top-1
fallback, commit/break logic) into a single `MLXFast.metalKernel`. Plan of record: the approved
plan at `~/.claude/plans/ok-next-experiment-i-m-logical-haven.md` (2026-07-20). House rules
apply: provenance labels (sourced / inferred / speculative), envValid discipline on every bench
row, repeated-median protocol, never a piped exit code.

## §0. Status

**Campaign AT CHEAP-EXIT CHECKPOINT (F4, 2026-07-20) — awaiting André's verdict.** T0/T1/T1.5/T2a/T4 complete; T2 not started. Started 2026-07-20, branch `GPU-contol`. **Host correction (F1): this
campaign is running on the Mac Studio M2 Ultra** (`hw.model=Mac14,14`, 192 GB, verified via
sysctl 2026-07-20 — the plan's "dev-M1-scoped" assumption was wrong for this session). Per §0.1
wall-clock verdicts are host-scoped: G2 numbers from this campaign are **Studio/serving-host
verdicts** (higher trust tier), and an M1 backfill, if ever wanted, is a separate session's
decision. Weights (mini + flash, BF16 + 4-bit) are all local on this host.

**Goal (one paragraph):** V1 moves the per-step loop control from "GPU forward → CPU decides →
GPU forward" to a single fused, GPU-resident control kernel that is **bit-exact** with the
current Q-mode plain-branch selection chain — output-invariant, trajectory-checkable, default-off
(`NEODIFFUSION_CONTROL_KERNEL` / `GenerationParams.controlKernelEnabled`). It replicates the
existing ~35–40 small MLX ops (activeMask → nextPosts → budgetBreak → maskConf/highConf →
top-1 fallback → Δ → finalTransfer → nextWindow → settleBreak/resultWindow → stats) as one
kernel dispatch, threading state through `SlotRun` the way `frozenMask` already is. **V2** (a new
margin+consecutive-agreement commit policy, packed `ctrlState` bitset) is explicitly **out of
scope** for this campaign — sketched only, its own gates, its own logbook section if it ever
starts.

## §1. Settled decisions (with André, 2026-07-20)

- **V1 is bit-exact, not a new policy.** The kernel replicates current Q-mode plain-branch
  semantics exactly. Output must be invariant to flag on/off; every parity/E2E test is a
  trajectory-identity check, not a quality comparison. A new commit policy is V2 and is not
  attempted here.
- **Acceptance gate is infrastructural, not a speed mandate.** The shipping bar is parity + ms/step
  not worse (≤ stock + 2%) + zero added host syncs (§2, G1/G2/G4). Op/graph-node-count reduction
  and the synthetic microbench are **diagnostics** (G3a/G3b) — they inform confidence, they do not
  gate. TPS/steady-forward deltas are reported, never gated.
- **Delegation is tiered by task risk** (André's explicit ask): **Haiku** for mechanical/pinning
  work, **Sonnet** for moderate engineering, **Fable** for complex or trajectory-risky work (the
  seam refactor T1 and the full kernel T2 are Fable-tier because a mistake there silently
  corrupts the denoising trajectory).
- **Floating-point parity policy** (André's amendment — do not trust hex-float identity alone;
  compiler transforms, MLX intermediate precision, and algebraic reordering can all break it):
  - **Safe class** (allowed in-kernel): compares where the LHS is a bit-identical raw input
    (`x0p`) against a bit-identical static constant (τ_mask, τ_edit) — no float arithmetic
    precedes the compare, so the decision is exact by construction. All integer/boolean logic is
    exact everywhere.
  - **Hazard class** (derived-float compares — anything that computes before comparing, e.g.
    dynamic-τ(t) which multiplies first, or a decoded-fraction-vs-τ_add compare): not trusted to
    hexfloats.
    - **Dynamic τ (α>0)**: V1 default eligibility is **α=0 only**. α>0 is a stretch goal, admitted
      only if boundary unit tests (T3, values within 1 ulp of a confidence) prove decision-identity
      on this toolchain; otherwise excluded and recorded, not forced.
    - **activationFlag stays MLX-side in V1** — computed from the kernel's `nextWindow` output via
      the existing `activationFlagFor` helper. It is a handful of post-kernel ops; keeping it out
      of kernel scope removes a derived-float hazard at negligible cost. May move in-kernel in V2
      if integerized with its own decision-identity proof.
    - `meanConf` (stats[2]) is diagnostics-only (both stats-driven control reads use integer
      stats[3]): tolerance-gated 1e-6 rel, documented deviation — never a G1 blocker.
  - `meanConf` aside, **G1 requires byte-for-byte identity on every discrete output and every
    control-relevant float output**, flag-on vs flag-off, across all parity + E2E tests.
- **Purity constraint.** `denoisePhase` chains K `windowStep`s into one lazy graph before eval;
  overshoot past a break is discarded via snapshots. The kernel must therefore be a **pure
  function** — MLXArrays in/out per step, no persistent `MTLBuffer`, no side effects — with any
  state threaded through the graph exactly like `frozenMask` is (`SlotRun` → `specFrozenMask` →
  `windowStep` → `WindowStepResult`). V1 adds `SlotRun.controlState: MLXArray [1,B] UInt32`
  (zeros, passed through unchanged) as inert plumbing so V2's packed state machine is a
  kernel-only change later.
- **Kernel scope boundary: post-reduction only.** Stock keeps the vocab-V=157184 work (softmax,
  argMax, max) on MLX; the kernel consumes only the tiny `[1,B]` tensors downstream of sampling
  (`windowActive`, `x0`, `x0p`, masks, `posts`, state) and fuses the selection chain from
  `activeMask` through `stats`, plain-Γ path only.
- **Feature-incompatibility policy is graded, not a blanket precondition** (André's amendment).
  Incompatible combos (ICE, credit, JOT, S2D2, nBuf>1, eosEarlyExit, temporalVoting, elasticCache,
  subBlockCommit, dynamicTauAlpha>0 in V1, B>32, `onTrace != nil`) are handled per context, never
  by a silent crash from a preset combination:

  | Context | Behavior |
  |---|---|
  | Tests / debug builds | `assert` (fires in debug test runs, compiled out of release) |
  | CLI benchmark (`diffusion-bench`) | explicit configuration error **before** execution — a bench must never silently measure the wrong arm |
  | Serving / library API | kernel disabled, stock path runs, `Metrics.effectiveControlKernel=false` **plus** `Metrics.controlKernelIncompatibility: String?` machine-readable reason (e.g. `"ice"`, `"dynamicTau"`) |

## §2. Pre-registered gates (pre-registered 2026-07-20, before any kernel code ran)

V1, scoped to the host the campaign actually runs on (the Studio M2 Ultra, per F1), per §0.1.

| Gate | Class | Criterion |
|---|---|---|
| **G0** feasibility | hard, pre-implementation | a no-op kernel with the planned I/O signature dispatches inside a K-chained lazy graph; `.bool` I/O (or a documented cast fallback); `[1]`-shaped Bool flags survive the `[2K]` concat seam |
| **G1** parity | hard | **Discrete outputs** (tokens, windows, bitsets, integer stats, flags): byte-for-byte identical flag-on vs flag-off across all parity + E2E tests. **Control-relevant float outputs**: bitwise identical. **Diagnostics-only `meanConf`**: rel error ≤ 1e-6, documented, never blocking |
| **G2** ms/step | hard | denoise ms/step (`denoiseSeconds/logicalStepsTotal`), repeated-median protocol (≥5 envValid reps, median + spread reported; overlap read against the spread, not a point estimate); trajectory-matched by construction via G1. **PASS: kernel ≤ stock + 2%** |
| **G4** no added syncs | hard | zero additional host synchronization on the kernel-on path: `syncPoints` metric identical flag-on vs flag-off; no new `.item()`/`.asArray` in the kernel branch (code-reviewed + T7 assert) |
| G3a node count | **diagnostic** | static op-node count of the segment (stock ≈ 35–40 → kernel target ≤ ~8 incl. casts); recorded, not gating — node count is not execution cost |
| G3b microbench | **diagnostic (strong prior)** | kernel segment vs stock on synthetic `[1,32]` inputs; a poor result predicts G2 failure but does not itself ship/block |

TPS/steady-forward deltas are reported, not gated. Every bench row carries `hw.model`,
`toolchain.mlxSwiftPackage`/`mlxCoreVersion`, `effectiveControlKernel` (+ incompatibility reason
when false). A G2 failure is a **documented negative** (Step-5 precedent): the kernel stays
landed, default-off, inert, and the logbook records it as such — it is not deleted.

**Cheap-exit point**: if T1.5 (the stock-segment GPU-time profile) shows the plain-branch
selection segment is a negligible share of per-step GPU time — i.e. MLX has already fused most of
it, or the segment's absolute cost is too small for G2's 2% budget to plausibly hide a kernel win
— the campaign **may stop after T1.5** without building T2. This is **André's call**, not an
automatic exit; T1.5's numbers go in §4 and the decision gets logged here either way.

## §3. Task ledger

| # | Task | Tier | Description | Status |
|---|---|---|---|---|
| T0 | Logbook skeleton | Sonnet | Decisions, pre-registered gates, empty findings table — this file | **done** (72973df) |
| T1 | Seam refactor | Fable | Pure code motion of the plain-branch selection chain into internal `stockSelectionUpdate(inputs) -> SelectionOutputs`; identical graph; full suite green; own commit before any kernel code | **done** (30a35eb; suite 137 tests / 26 env-gated skips / 0 failures) |
| T1.5 | Stock-segment profile | Sonnet | In-situ Metal capture of the stock selection segment (dispatch count, GPU time, command-buffer boundaries, existing MLX fusion) — informs whether G2 upside exists before T2 is built; cheap-exit checkpoint | **done — F4: segment ≈0.42 ms/step vs ≈32.2 ms forward (1.2–1.8%); CHEAP-EXIT recommended, awaiting André** |
| T2a | G0 feasibility spike | Sonnet | Minimal no-op `MLXFast.metalKernel` with the planned I/O signature (incl. `.bool` arrays, u32 state passthrough, FP_CONTRACT pragma) dispatched inside a K-chained lazy graph | **done — G0 PASS** (F2; `ControlKernelFeasibilityTests`, 4 tests / 0 failures) |
| T2 | `BlockControlKernelRunner` | Fable | Full MSL + Swift per design; `Tools/seed-metallib.sh` after builds | pending |
| T3 | Seam A/B parity tests | Sonnet | Adversarial set: τ boundaries, all/zero/single-mask windows, argMax ties, posts==maxPostSteps, tokenChanged=false at high conf, prompt-tail blocks, hasNextBlock both ways, B∈{16,32}; α>0 boundary tests double as the dyn-τ admission test | pending |
| T4 | MLX-semantics pin tests | Haiku | argMax first-index tie-break, strict-`>` with −inf — converts inferred claims to sourced-by-test | **done** (5a15067; F3; 4 tests / 0 failures) |
| T5 | Engine wiring | Sonnet | Params knob, run-level eligibility resolution + graded incompatibility policy, `SlotRun.controlState` threading, `windowStep` branch, Metrics echo + incompatibility field, bench JSONL echo | pending |
| T6 | E2E trajectory tests | Sonnet | Toy-config both-arms identity; real-weight-gated identity (`NEODIFFUSION_LLADA_REAL=1`), ≥8 prompts, Q mode, plain (+eosEarlyStop, +α=0.6 if admitted) | pending |
| T7 | Segment microbench (diagnostic) | Sonnet | Stock chain vs kernel on synthetic `[1,32]` inputs; static op-node counts; assert syncPoints identical flag-on/off (feeds G4) | pending |
| T8 | Bench campaign | Sonnet | `diffusion-bench llada` on this host (Studio, per F1), Q mode, flag off/on, repeated-median (≥5 reps), envValid-only, warmup excluded, toolchain + effective echoes; evaluate gates; logbook findings | pending |
| T9 | Close-out | Sonnet | Logbook verdict, CLAUDE.md status line, 3-section report to André | pending |

T4 can run any time in parallel with the rest. T1.5 and T2a can run in parallel with each other
(both depend only on T1).

## §4. Findings

| F# | Date | Finding | Evidence | Consequence |
|---|---|---|---|---|
| F0 | 2026-07-20 | The engine already runs the entire Γ/Δ selection chain as MLX GPU array ops, with host readback already batched to one `[2K]`-shaped flags read per K-step speculative batch (plus one per block commit) — there is no per-step CPU decision loop to eliminate. | `DiffusionEngine+Step.swift:219` (batched `[2K]` flags read); selection chain at `DiffusionEngine+Step.swift:397-682` | This campaign's value proposition is **fusion of already-GPU-resident ops + enabling persistent kernel state for V2**, not "moving control off the CPU" — that framing would be wrong going into T1. Sets the honest scope for §0's goal statement and for T1.5's cheap-exit check (if the segment MLX already runs turns out cheap/already-fused, the fusion upside may not clear G2's 2% budget). |
| F1 | 2026-07-20 | This campaign runs on the **Mac Studio M2 Ultra**, not the M1 dev box the plan assumed (sourced: `sysctl hw.model` = Mac14,14, 192 GB; both mini and flash weight artefacts local). | sysctl output, session 2026-07-20 | G2/T8 wall-clock verdicts are serving-host verdicts. Served default here is `speculationK=1` (host-aware, CLAUDE.md F-l) — T8 must bench the served K, and any K>1-specific claims are out of scope on this host. Plan/logbook "dev-M1" wording corrected; flagged to André. |
| F2 | 2026-07-20 | **G0 PASS, all four sub-items** — bool MLXArray I/O works natively for metalKernel inputs *and* outputs (traced to `custom_kernel.cpp` `write_signature` → Metal `bool`, generic path); `#pragma STDC FP_CONTRACT OFF` compiles in the JIT body splice; K=4 pure lazy chaining with one eval + one readback works; `[1]`-Bool concat seam intact. API facts for T2: inputs with < 8 elements are declared `constant` (read-only — `posts`/`rtScalars` must never be written through), outputs are always `device`; `source` is spliced verbatim inside the generated function body. | `Tests/DiffusionCoreTests/ControlKernelFeasibilityTests.swift` (4 tests / 0 failures, suite line 2026-07-20 08:18) | The planned kernel interface needs **no dtype fallbacks**. T2 proceeds with bool tensors as designed. |
| F3 | 2026-07-20 | MLX semantics pinned by test: argMax tie-break = first index (incl. all-equal and all-−inf → 0); `.>` is strict (τ==conf and −inf never pass); `which` is elementwise ternary. | `Tests/DiffusionGenerationTests/MLXSemanticsPinTests.swift` (4 tests / 0 failures) | The kernel's ballot+ctz first-index reduction and strict compares are implementing *sourced* semantics, not guesses. |
| F4 | 2026-07-20 | **The stock selection segment costs ≈0.42 ms/step against a ≈32.2 ms/step forward (share 1.2–1.8%) on the Studio** — measured two independent ways that agree: (a) in-situ instrument-mode run, real 4-bit weights, Q-cached, gen-128, warmup excluded (forward 31.5–33.2 ms, sampler ≈0.53 ms, selection 0.41–0.43 ms, loop-control ≈0.30 ms per step; eval-inflation caveat applies to shares); (b) standalone release-build microbench of the reconstructed op chain on synthetic `[1,32]` (median 421.8 µs/iter over 5 runs, ≈5% spread). Gputrace skipped — methods 1–2 conclusive and mutually corroborating. Provenance: `hw.model=Mac14,14`, `mlxCoreVersion=0.31.1`, `mlxSwiftPackage=0.31.6`, all rows `envValid:true`. | scratchpad `t1_5_bench.jsonl` + `t1_5_run.log` (session scratchpad — host-local, not tracked); `Tests/DiffusionCoreTests/ControlKernelBench.swift` (1 test / 0 failures per run, `NEODIFFUSION_CONTROL_BENCH=1`) | **Ceiling analysis: a perfect zero-cost fused kernel saves at most 1.2–1.8% of step time — inside G2's own ±2% pass band and at/below observed run noise.** The wall-clock case for T2 is dead on this host; the pre-registered cheap-exit checkpoint fires. Decision is André's (logbook §2): close the campaign, or proceed with T2 on the V2-enablement rationale alone (persistent-state substrate for the margin/agreement commit policy), with G2 reframed as pure no-regression. |

## §5. Provenance footer

- Every bench row produced by this campaign (T1.5, T7, T8) must carry: `hw.model`,
  `toolchain.mlxSwiftPackage` + `toolchain.mlxCoreVersion`, and `effectiveControlKernel` (plus
  `controlKernelIncompatibility` reason string whenever `effectiveControlKernel=false`).
- Provenance labels (sourced / inferred / speculative) apply to every non-trivial claim in this
  log and in code comments touching the kernel path, per house style.
- §0.1 protocol reminder: G0/G1/G4 are hardware-independent; G2's wall-clock is host-scoped —
  measured on the Studio M2 Ultra (F1) and not a portable verdict for the M1 (or any other host)
  without its own repeat of G2.
- F0 is currently *sourced* (line-referenced). All later findings must cite their evidence file
  the same way — a bench JSONL path, a test name + suite line, or a capture bundle path — never a
  bare assertion.
