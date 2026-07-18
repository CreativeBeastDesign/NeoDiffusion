# Step 5 hand-off — the fused MoE-gather kernel, entry state and evidence

**Status**: Step 4 (three cheap probes) CLOSED — all negative. **Step 5a (occupancy profile) CLOSED — decisive GO.** Step 5 itself (write the inline `MLXFast.metalKernel`) is **not started**. This doc is the entry point for that work in a fresh context.
**Created**: 2026-07-18. **Owns**: the transition from "profile says GO" into actually authoring the kernel.
**Prereq reading**: `Plans/pre-kernel-handoff.md` §3–4 (Step 4 + Step 5a results, now fully recorded there) · `Plans/gather_qmm_handoff.md` §4 (Case C — delivery mechanism, no mlx-swift fork) + §10/§12 (the ~7.5 ms sizing this all traces back to) · `Plans/final_optimisations_plans.md` §2 (plan of record).

---

## 1. Where things stand, in one paragraph

The road to the kernel is fully paved. Step 4's three cheap probes (2026-07-18) closed negative: no upstream MLX bump available (already on the newest mlx-swift, 0.31.6), the `sortedIndices` wiring was already optimal, and no cheaper quantization format (8-bit, mxfp4, FP16) beats the shipped 4-bit affine g64 artefact. That left exactly one open question — the Xcode Metal-debugger occupancy profile the hand-off has asked for since day one — and it came back **decisive**: the 4-bit gather is occupancy-limited by register pressure from its own dequant chain, confirmed by three independent, mutually-reinforcing counters (not just "not ruled out" — actually positively confirmed). **Nothing is blocking Step 5 anymore.** The only work left in this document's scope is writing the kernel itself.

## 2. The Step 5a verdict (full detail; also recorded in `pre-kernel-handoff.md` §4)

Captured via Xcode Metal debugger on `scratch/captures/moe_block_t32.gputrace` (a real `LLaDA2SparseMoEBlock` forward, production quantization, T=32 — **not** an isolated `gatherQuantizedMM` call; see §3 for why). Studio `Mac14,14`, MLX core 0.31.1 / mlx-swift 0.31.6.

| signal | reading | reads as |
|---|---|---|
| Kernel | `affine_gather_qmv_fast_float_gs_64_b_4`, 50.9% of encoder cost (259.75 µs of 510.22 µs) | unambiguous — bound buffers `w`=128 MiB, `scales`=16 MiB, `biases`=16 MiB match `E·I·H·bits/8` and `E·I·(H/64)·4B` exactly |
| Kernel Occupancy | **~32.8%** | low — well under half the GPU's thread-level parallelism in use |
| Registers | 100 allocated, 100 high, **0 bytes spilled** | no spilling, but high enough to cap occupancy — classic register-limited-occupancy |
| ALU Limiter / Integer&Complex Limiter | 69.6% / 57.1% | high — the dequant bit-unpack/shift is a real ALU cost, not just an indirect tax |
| F32 / F16 Utilization | 15.1% / 0% | low — raw float throughput is NOT the bottleneck |
| Kernel shape | `gather_qmv` — matrix-**vector**, not matrix-matrix | structurally can't use `simdgroup_matrix`/MMA: each of the 256 token×expert pairs needs a *distinct* weight block, so there's no shared-weight tile for tensor cores to batch over |

**Design implication, not just a verdict**: the kernel target is *reducing per-thread register pressure in the dequant+accumulate inner loop* — e.g. stage scale/bias lookups through threadgroup memory instead of holding them in per-thread registers, shorten the live range of intermediate dequantized values before the FMA — to raise occupancy above 32.8% and let more memory requests overlap. **"Force MMA" is not a viable lever** (structurally a GEMV, confirmed above) — don't spend time on it.

**Expected payoff** (unchanged from original sizing, `gather_qmm_handoff.md` §10/§12): **+10–20% TPS** if the profile's cause is fixable — it is the cause, per above. Ceiling **~+37%** end-to-end if the kernel closes the full gap to the 433 GB/s the FP16 sibling demonstrates. **Accept a recorded negative if the actual MSL doesn't move the needle** — the profile predicts a fixable cause, it doesn't guarantee the fix is cheap to write correctly.

## 3. Methodology note — read before attempting to re-profile anything

**Isolated `gatherQuantizedMM` captures do not work — go straight to in-situ.** Three independent, careful attempts to capture the op in complete isolation (via `MLX.GPU.startCapture`/`stopCapture` around a bare `gatherQuantizedMM` call, same Mini shapes as `LLaDAMoEDispatchBench.runGatherQMMVariants`) all failed the same way — a 40 KB empty trace, vs. the FP16 sibling's 2.7–4 GB every time:
1. Shared capture window (both arms in one trace) — only the FP16 dispatch showed up.
2. Separate trace files per arm — 4-bit arm still empty.
3. Completely fresh, never-before-evaluated inputs + forced synchronous `.item()` readback (ruling out both caching/memoization AND async-scheduling as explanations) — still empty.

Root cause was never diagnosed; it's structural to capturing that specific op in isolation, not a timing/caching artifact. **What worked**: capture a real `LLaDA2SparseMoEBlock` forward instead (`LLaDAMoEDispatchBench.testCaptureMoEBlockTrace`, production quantization, T=32) — the same code path Step 4's entire wall-clock campaign already validated to cost real GPU time. That captured cleanly on the first attempt. **If any future profiling is needed (e.g., re-profiling after a kernel change), use the in-situ MoE-block route from the start.**

The (env-gated, harmless, never runs by default) test harness lives in `Tests/DiffusionCoreTests/LLaDAMoEDispatchBench.swift`:
- `testCaptureGatherQMMTrace` — the isolated attempt (kept for provenance / in case someone wants to re-attempt with a different technique; currently known not to work for the 4-bit arm).
- `testCaptureMoEBlockTrace` — **the one that works.** Re-run: `MTL_CAPTURE_ENABLED=1 NEODIFFUSION_GPU_CAPTURE=1 swift test --filter LLaDAMoEDispatchBench/testCaptureMoEBlockTrace`. Output: `$NEODIFFUSION_CAPTURE_DIR` (default `<cwd>/scratch/captures`)`/moe_block_t32.gputrace`.

**Reading a trace once captured**: Xcode → open the `.gputrace` bundle → **Replay** (keep "Profile after replay" checked) → wait for load (large files, can take minutes) → **Performance** tab → **Shaders** sub-tab (sortable table, gives per-dispatch cost/registers/spill — this is where the register data above came from) → **Counters** tab → **Occupancy** (hover the graph during the target kernel's time window in the **Timeline** sub-tab for the exact %; the Timeline's "Shaders" track shows which colored block corresponds to which kernel).

## 4. Repo state — what's committed, what's not, what to clean up

**Already committed at HEAD** (`b4fae28`, branch `pre-kernel`) — verified via `git show HEAD:<path>`, safe, nothing to do:
- `Packages/DiffusionModel/Sources/DiffusionModel.swift` — mxfp4 plumbing (`QuantizationConfig.mode`, `quantizeModel(quantMode:)`) **and** the register-mode bugfix. Worth knowing about even though it's committed: a `quantizeModel(mode: QuantizationMode = .affine)` parameter named `mode` gets silently shadowed inside a filter closure that returns a tuple with a `mode:` label — it resolved to a stale cross-module default-argument thunk (`.mxfp4` instead of `.affine`) after repeated signature edits during incremental builds, crashing one test. Renamed the parameter to `quantMode` + a full `swift package clean` rebuild fixed it. The comment at `DiffusionModel.swift:126` explains it in place. **If a similar "the default argument value is wrong despite the source clearly being correct" symptom ever recurs, suspect a stale build artifact before suspecting the code — try `swift package clean` first.**
- `Tools/convert_weights_streaming.py` — `--mode {affine,mxfp4}` support (bias-less, g32/4-bit, e8m0 U8 scales for mxfp4).

**Still uncommitted** (working tree, `git status --short`):
- `Tests/DiffusionCoreTests/LLaDAMoEDispatchBench.swift` — the GPU-capture test harness (§3 above; both the isolated-attempt and working MoE-block methods).
- `Plans/pre-kernel-handoff.md` — Step 4 + Step 5a results recorded in §2–4.

Recommend committing both together once this hand-off is reviewed — they're finished, tested, and the harness is inert by default (env-gated).

**Disk cleanup candidates** (all host-local, gitignored, safe to delete once no longer needed):
| path | size | keep for |
|---|---|---|
| `scratch/captures/moe_block_t32.gputrace` | 1.2 GB | **keep** — the working, read trace; useful to re-open or diff against a future kernel's trace |
| `scratch/captures/gather_t{32,64}_{4bit,fp16}.gputrace` | ~40 KB + 4 GB each | delete — empty/superseded isolated-capture attempts (§3) |
| `scratch/captures_old/` | 5.3 GB | delete — superseded by the above |
| `models/llada2-1-mini-8bit-experts` | 17 GB | Step 4c-i diagnostic artefact (8-bit experts, confirmed +3.5% ≈ tie vs 4-bit — nibble-unpacking is not the cost). Keep only if you want to re-run that comparison; otherwise delete. |
| `models/llada2-1-mini-mxfp4` | 9.0 GB | Step 4c-ii diagnostic artefact (mxfp4, confirmed +7.9% slower). Same — delete unless re-running. |

## 5. Step 5 itself — what's actually left to do

Not started. Per `gather_qmm_handoff.md` §4 (Case C, already resolved): route (b) — **an inline-MSL kernel via `MLXFast.metalKernel`**, Metal source as inline Swift strings compiled by MLX at runtime. **No mlx-swift fork** — a fork needs an Xcode DerivedData rebuild of mlx-swift + reseed on every single edit, which is a non-starter for iteration speed.

Design target, from §2 above: a persistent/restructured dequant+gather-GEMV kernel that reduces live register count in the unpack-then-FMA inner loop (candidate techniques: cache scales/biases in threadgroup shared memory rather than per-thread registers since they're shared across the K picks touching the same expert; shorten the dequantized-value live range so the compiler doesn't need to hold it alongside accumulator state; consider restructuring the unpack to batch more of the 4-bit nibbles per instruction). The Alpha-MoE *ethos* (persistent kernel, threadgroup residency, fused dequant-into-GEMM) legitimately applies here; its Hopper-specific mechanisms do not transfer to Apple Silicon (refuted in `gather_qmm_handoff.md` §3).

**Effort**: the largest item in the whole optimisation plan (~8-9/10 per the original estimate) — moving-target MLX internals + inline-MSL authoring + iterative occupancy profiling using the now-proven in-situ capture method (§3). This is why it's last.

**Suggested first move in the fresh context**: re-read this doc + `gather_qmm_handoff.md` §4, then go into plan mode to scope the actual kernel design and implementation steps — this doc deliberately stops at "what to build and why," not "how to write it," since that's a substantial design task in its own right.

## 6. Fast start for a fresh context

1. Read this doc (you just did) + `Plans/gather_qmm_handoff.md` §4 (delivery mechanism) and §10/§12 (why ~7.5 ms, the sizing this all serves).
2. Decide: commit the two pending files (§4) now, or fold them into the kernel's eventual commit — either is fine, just don't lose them.
3. Optionally clean up the disk artefacts in §4's table if space matters.
4. Go into plan mode for Step 5's actual design — the register-pressure-reduction target from §2 is the brief; the specific MSL/threadgroup-memory design is the open question.
5. Report per the CLAUDE.md three-section rule once there's something to report (What you should know / What you should do / Next steps), provenance-complete (host, both MLX versions, envValid handling).
