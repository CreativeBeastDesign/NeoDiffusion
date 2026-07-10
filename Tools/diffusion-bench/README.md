# diffusion-bench

NeoDiffusion benchmark CLI. Two modes: **llada** (LLaDA2.1-mini, the phase-2 M6 metric
set) and **sumi** (sumi-plan.md §5 S3.1).

## LLaDA mode (M6)

```
swift run -c release diffusion-bench llada                  # q-cached + s-cached arms,
                                                            # 3 suites × 3 runs, variance gate
swift run -c release diffusion-bench llada --runs 1 --suites chat --gen-length 64
swift run -c release diffusion-bench llada --mask-diagnostic   # §6-item-4 strict vs
                                                               # referenceBias outputs
swift run -c release diffusion-bench llada \
    --arm q-uncached --arm-mode q --uncached                # cache-off diagnostic arm
```

Defaults: model `models/llada2-1-mini-4bit`, tokenizer `models/llada2-1-mini`, block 32,
gen-length 128, `eos_early_stop` on, chat-templated prompts from
`Tools/diffusion-bench/PromptSuites/{chat,reasoning,code}.json`. Results append to
`scratch/llada_bench.jsonl`, one JSON object per (arm, suite, prompt, run); a per-arm
cross-run variance check (<5% max deviation, the M6 acceptance) prints at the end.

### M6 metric set (per JSONL row)

| Metric | Field(s) | Notes |
|---|---|---|
| TPS | `tps` | trimmed tokens / wall-clock |
| TPF | `tpfHonest`, `tpfLogical` | honest = tokens / forwards *evaluated* (counts speculative overshoot + per-commit capture forwards + prompt prefill); logical = tokens / denoising steps (reference-comparable; run `--speculation-k 1` for reference-identical step economy) |
| steps/block | `stepsPerBlockMean`, `logicalSteps` | **+1 vs the reference trace on budget-break blocks** (handoff §4 — the engine counts the break iteration; tokens unaffected) |
| post-steps/block | `postStepsPerBlockMean`, `postSteps` | `post_steps` at loop exit = mask-free (refinement) iterations incl. the breaking one |
| sync points | `syncPoints` | blocking readbacks (M5(c) audit: ≤1 per K steps + 1 per commit) |
| peak memory | `peakMemoryGB` | `GPU.peakMemory`, reset per generation |
| per-phase wall-clock | `prefillSeconds`, `denoiseSeconds`, `commitSeconds` | real only while instrumented (default; the engine `eval`s at phase boundaries — `--no-instrument` to measure totals without them) |

Caveats: the 4-bit artefact is **task-level only, never parity** (gotcha 5); numbers
frozen on the M1 are the *dev* baseline — re-freeze on the Studio. Laptop thermals can
fail the 5% gate on >10-minute arms regardless of `--cooldown` (Sumi campaign quirk 4).

## Sumi mode

```
swift run -c release diffusion-bench sumi                 # frozen baseline suite, 3 runs/arm
swift run -c release diffusion-bench sumi --runs 1        # quick pass
swift run -c release diffusion-bench sumi \
    --arm my-experiment --sampler adaptive --canvas 512 --k 8 --steps 8 --freeze
```

Results append to `scratch/sumi_bench.jsonl` (one JSON object per arm-run) and print as a
human-readable table with a cross-run variance check (acceptance: <5% max deviation of
total time per arm).

## Baseline arms (frozen 2026-07-09, M1 numbers in sumi-plan.md §5)

| Arm | Sampler | Canvas | k | Steps | Notes |
|---|---|---|---|---|---|
| `recipe-adaptive-k4` | adaptive, t=0 | 1024 | 4 | 16 | the measured M1 operating point |
| `quality-adaptive-k1` | adaptive, t=0 | 1024 | 1 | 64 | coherence reference |
| `reference-ancestral` | ancestral, t=0.7 | 1024 | 1 | 64 | reference-default sampler |

Adaptive arms set `denoise_end = prompt + budget + 2` (concentrates the step budget on the
content window — required for coverage, see sumi-plan.md §5 lever experiments).

## Metric mapping vs the LLaDA M6 set

| M6 metric | Sumi mode | Why |
|---|---|---|
| TPS | `effectiveTPS` = budget / total wall-clock | budget tokens are the useful output |
| TPF | n/a | no forward-per-token structure; every step is a full-canvas forward |
| steps/block, post-steps/block | `steps` (fixed) | no blocks; the step count is a parameter, not an outcome |
| sync-point count | `syncPoints` = steps + 1 (analytic) | 1 blocking `eval` per step + 1 final trim readback |
| peak memory | `peakMemoryGB` (`GPU.peakMemory`) | same |
| per-phase wall-clock | load / `firstStepSeconds` (canvas init + kernel compile) / steady-state step stats (mean, std, min, max, p95) | Sumi has no prefill/commit phases |

## Known quirk: the Cmlx metallib bundle (updated 2026-07-10, m8-logbook)

On the current toolchain (Swift 6.3.3 / Xcode 26 SDK), **`swift build` never compiles
mlx-swift 0.31.6's `.metal` sources at all** — no `mlx-swift_Cmlx.bundle` is produced by
CLI builds. Binaries and tests then die with `MLX error: Failed to load the default
metallib`. The bundle must be seeded from an Xcode build of this workspace (DerivedData),
into TWO places depending on what you run:

```
D=$(ls -td ~/Library/Developer/Xcode/DerivedData/Diffusion-*/Build/Products/Debug/mlx-swift_Cmlx.bundle | head -1)
# executables (bench/server): next to the binary
cp -R "$D" .build/arm64-apple-macosx/release/
# XCTest: inside the test bundle's Resources
cp -R "$D" .build/arm64-apple-macosx/debug/NeoDiffusionPackageTests.xctest/Contents/Resources/
```

The xctest copy must be repeated after builds that relink the test bundle. Related quirk:
never delete parts of `.build` selectively — `.build/build.db` (llbuild state) survives
`swift package clean` and partial `rm`s, after which builds silently skip work and link
mismatched objects (segfaults). Delete artifacts + `build.db` + plugin state together.

## Profiling

For kernel-level insight (which ops eat a step), wrap a run in Instruments:

```
xctrace record --template 'Metal System Trace' --launch -- \
    .build/release/diffusion-bench sumi --runs 1
```

House rule (metal-shader-guide §TL;DR): no kernel work without the trace saying the target
region is hot — wall-clock and FLOP intuition diverge badly here (see sumi-plan.md §4
Finding 2 history).
