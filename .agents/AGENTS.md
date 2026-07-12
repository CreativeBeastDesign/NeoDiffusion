# Workspace Rules for NeoDiffusion

Any agent working on the NeoDiffusion codebase must strictly adhere to the following development, measurement, and logging methodologies.

## 1. Provenance & Labeling Discipline (House Style)
- **Label all Claims**: Mark every technical assertion, performance claim, or design argument in documentation, pull requests, and non-trivial comments as either **sourced / inferred / speculative**.
  - **Sourced**: Must cite the exact command, test case, log file path, or database query that reproduces the claim.
  - **Inferred**: Derived logically or mathematically from sourced facts (e.g. subtracting sub-module timings to find attention overhead).
  - **Speculative**: An educated guess or hypothesis that has not yet been measured or proven.
- **Document Negative Results**: Never delete failed experiments or optimization paths from the planning/logbook history. Document them with the same structure and metric precision as successful arms. Record them in the appropriate logbooks (e.g., `Plans/elastic-cache-logbook.md`) as negative results.
- **Write Wiki Drafts**: After completing any experiment or optimization work package (successful or negative), the agent must write a corresponding wiki draft document under `Plans/wiki-drafts/` (e.g. `wp-4a-temporal-self-consistency-voting.md`) summarizing what was built, the results, key findings, and recurring lessons learned.

## 2. "Paranoid" Benchmarking & Logging
- **Environment Validity (`envValid`)**: On Apple Silicon, background processes, memory pressure, thermal throttling, and macOS paging inject severe timing anomalies (e.g., a ~50s/forward swap pathology on 16GB machines).
  - Every benchmark run must capture system telemetry (swap used before/after, free memory pages, and OS thermal state).
  - A benchmark row is only considered valid if:
    - Swap growth during the run is $\le 256$ MB.
    - Free memory at the start is $\ge 1024$ MB.
    - OS thermal state is `nominal` or `fair`.
  - Staging/analysis scripts must only draw conclusions and evaluate performance gates using valid rows.
- **Engine Effective Echoes**: JSONL logs and benchmark output must record the *effective parameters* actually executed by the generation engine (e.g., `nBuf`, `tauAdd`, `speculationK`), not the raw CLI command-line inputs. This prevents silent execution bugs (e.g., unquoted CLI strings falling back to defaults) from contaminating the results.
- **Warmup Exclusion**: The first generation run in any new process pays a ~19-second kernel compilation and graph setup penalty. Exclude the first run (`warmupIncluded: true` rows) from all steady-state performance comparison calculations.
- **Content-Sensitivity**: Diffusion step counts, token throughput (TPS), and tokens-per-forward (TPF) are highly dependent on prompt content (e.g., a simple QA block takes ~2 steps/block, whereas a code or essay block takes ~21 steps/block). Never cite a single-prompt TPS. Always compare performance metrics across the standard prompt suites: `chat`, `reasoning`, and `code`.

## 3. Strict Acceptance Gates for Landing Code
To change defaults or merge optimizations to `main`, changes must pass the following progression:
1. **Toy-Config Parity**: Passes all unit tests on seeded random weights in FP32.
2. **BF16 Token-for-Token Parity**: Must match reference PyTorch outputs token-for-token at temperature 0 across at least 20 test prompts, first with prefix cache disabled, then with prefix cache enabled.
3. **Hardware-Independent Metrics Gate (§0.1)**: Algorithmic checks (e.g., step reduction, logical TPF, dual-active phase percentage) must demonstrate improvements on the M1 dev host.
4. **Target Hardware wall-clock TPS Gate**: The actual serving throughput (TPS) must clear the target gate (e.g., $\ge 15-20\%$ speedup) on the target Mac Studio M2 Ultra. If the dev host has high step-latency multipliers due to compute-bound scaling, the TPS verdict is deferred to the Studio backfill.

## 4. Coding Style and Toolchain Quirks
- **Precision Rules**: Never quantize the router, normalization layers, embeddings, or shared experts. Fused attention, router sigmoids, norms, and logit softmaxes must remain in FP32.
- **Compiler Metallib Seeding**: Xcode/SwiftPM toolchains on this project do not build the custom Metal library automatically. Always run `Tools/seed-metallib.sh` after any build command.
- **No Partial Builds**: Never partially delete the `.build` folder. Doing so poisons the build database, leading to silent linkage failures.
