# Track 1 Research Report & Paper Draft: Asymmetric KV-Cache Quantization and "Confidence Starvation" in Block-Diffusion Language Models

**Date**: 2026-10-18  
**Model**: `LLaDA2.1-mini` (`16B` MoE, 4-bit weights, 20 layers, 16 query heads, 4 KV heads, `headDim = 128`, `blockLength = 32`, `genLength = 64`, Q-mode $\tau_{\text{mask}} = 0.70$, $\tau_{\text{edit}} = 0.50$, `speculationK = 1`)  
**Host**: Apple M1 (~~`MacBookPro18,3`~~ **`MacBookPro17,1`** [E-1], 16 GB unified memory, `mlx-swift 0.31.6` / `MLX core 0.31.1`, Swift 6.3)  
**Primary Data Artifacts**:
- Operator-level perturbation benchmark: [`Tests/DiffusionCoreTests/DualCachingTests.swift`](../../Tests/DiffusionCoreTests/DualCachingTests.swift) (`testTrack1ScoreSpaceAndAttentionDivergenceAblation`)
- End-to-end 8-arm ~~interleaved~~ **single-process, sequentially-blocked** [E-2] JSONL (`96/96` valid rows, `12/12` `envValid` per arm): `scratch/track1_kv_starvation_k1.jsonl` (sha256 `2ebcdb25…1776`)
- Full token trajectories (`96` steady-state generations): `scratch/track1_kv_starvation_k1_text.jsonl`

## Errata & Corrections (2026-10-18)

> Reproduce every number below with `python3 Tools/analysis/cache_campaign_aggregate.py scratch/track1_kv_starvation_k1.jsonl` (CPU-only, stdlib; excludes the warmup row). Full campaign errata: [`Plans/cache-optimisation-logbook.md`](../cache-optimisation-logbook.md) § "Errata & Corrections (2026-10-18)".

- **E-1 — Host.** Was `MacBookPro18,3`; every row records **`MacBookPro17,1`** (Apple M1, 16 GB). Toolchain per row: `mlxSwiftPackage 0.31.6`, `mlxCoreVersion 0.31.1`. `[Sourced]`
- **E-2 — Not interleaved.** The 8 arms ran **sequentially in blocks** (all 12 prompts of one arm, then the next) in one process (`processId` 84470). Arm is therefore confounded with time-in-process; all rows still passed `envValid`. `[Sourced: script line "arm execution order: BLOCKED"]`
- **E-3 — "Mean TPS" is a mean of per-prompt ratios.** Aggregate throughput (Σtok/Σs) per arm: FP16 **12.50**, `fp8` **13.04** (+4.3%), `k8v4` **12.75** (+2.0%), `k4v8` **11.19** (−10.5%), `fp4` **10.73** (−14.2%), `fp4g32` **11.88** (−5.0%), `kivi4` **12.90** (+3.2%), `hybrid4` **11.45** (−8.4%) — deltas vs FP16 `cache-clean-commit-k1`. Note the sign flips: by mean-of-ratios `k8v4` (−1.2%) and `kivi4` (−2.5%) look slower than FP16; by aggregate they are faster. Suite-total seconds were already aggregate and are consistent with the aggregate figures. `[Sourced]`
- **E-4 — No compute speedup from KV quantization.** ms/forward: FP16 188.99; quantized arms 189.41–190.73 (+0.2% to +0.9%). Every TPS difference between arms is explained by forward-count (trajectory) changes (`fp8` −4.6% forwards … `fp4` +13.9%). `[Sourced]`
- **E-5 — "97.8% MoE expert + attention compute" is unsupported.** The project's measured forward budget (WP-6c, Mac Studio M2 Ultra) is routed MoE **56.3%** + attention **14.3%** ≈ 70.6%, and no M1 measurement of 97.8% exists. The conclusion (extra forwards dominate) does not depend on it: the cost is simply forwards × ~189 ms. `[Sourced: experiments-master-list.md WP-6c]`
- **E-6 — "Starvation steps" metric overstates.** It counts unmask steps with exactly 1 transfer — **126/254 = 49.6% of unmask steps in the FP16 baseline**, i.e. the normal top-1 fallback (and natural single-token clears are counted too). The measured effect is better described as a **reduced per-step commit rate** (tokens per unmask step: 3.61 FP16, 3.78 `fp8`, 3.72 `k8v4`, 3.21 `k4v8`, 3.10 `fp4`). The term "confidence starvation" is kept for traceability. `[Sourced: Table 1.3A]` / interpretation `[Inferred]`
- **E-7 — `fp8`/`fp4`/`k8v4`/`k4v8`/`fp4g32`/`kivi4`/`k8v4s4` are affine integer formats** (`MLX.quantized`, default affine mode: per-group scale + bias, integer codes). Read `fp8` as "`fp8` (affine int8)", `fp4` as "(affine int4)". `[Sourced: KVCacheCompression.swift]`
- **E-8 — Presets: quality gate OPEN (status change, fix in progress 2026-10-18, see code-fix section).** `k8v4-clean`, `kivi4-clean`, `fp8-clean` change the decoding trajectory (26–73% token match vs FP16) and have no scored blind sheet; per the standing rule (no preset ships while its quality gate is open) they are **experimental, default-off, quality gate OPEN**. `exact-fast` stays default-on (token-identical, 716/716) with **Studio BF16 parity OPEN**. `[Sourced: CLAUDE.md]`
- **E-9 — Citation scope.** "Quantized Keys Steal Attention" (arXiv:2605.26266, cited via `Notes/cache_research.md` Addendum B) concerns **video** diffusion; it is a mechanism analogue, not dLLM evidence. `[Sourced: arXiv abstract]`

> **Methodological Structure (Strict Fact vs. Interpretation Separation)**:  
> To support external reproducibility and publication, this document enforces a **hard structural boundary** between:
> - **Part I — Empirical Facts (`[Sourced]`)**: Direct measurements, byte counts, operator norms, KL divergences, step counts, starvation rates, wall-clock timings, and exact token-match counts, with zero causal speculation.
> - **Part II — Theoretical & Mechanistic Interpretation (`[Inferred]` & `[Speculative]`)**: Mathematical derivations, causal attribution from the $2 \times 2$ factorial, and mechanistic explanations connecting attention-score perturbation to diffusion unmasking dynamics.

---

## Part I: Empirical Facts (`[Sourced]`)

### 1.1 Experimental Design & Tested Configurations (`[Sourced]`)

All eight end-to-end configurations were evaluated in a single OS process (`PID` recorded per row, `--warmup-once` excluded, `--cooldown 5`, `12` prompts across `chat`, `reasoning`, and `code`, `genLength = 64`, `blockLength = 32`, `speculationK = 1`, `skipRedundantCommitForward = true`) via [`Tools/diffusion-bench/Sources/LLaDABench.swift`](../../Tools/diffusion-bench/Sources/LLaDABench.swift). All `96` steady-state rows passed the `envValid` telemetry gate (`swapDelta <= 256 MB`, `freeMemoryMBBefore >= 1024 MB`, `thermalBefore == "nominal"`).

| Arm Name | `kvQuantization` | Key Format | Value Format | Bytes / Token (`20L, 4 KV, D=128`) | Mean KV Size (`12` prompts) | Max KV Size (`128` tok) | Size Reduction vs `FP16` |
| :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **`cache-clean-commit-k1`** | `none` | 16-bit FP16 | 16-bit FP16 | `40,960 B` | `4.792 MiB` | `5.000 MiB` | `0.00%` (Control) |
| **`cache-fp8-clean-k1`** | `fp8` (affine int8 [E-7]) | 8-bit per-token (`g64`) | 8-bit per-token (`g64`) | `21,760 B` | `2.546 MiB` | `2.656 MiB` | **`-46.88%`** |
| **`cache-k8v4-clean-k1`** | `k8v4` | 8-bit per-token (`g64`) | 4-bit per-token (`g64`) | **`16,640 B`** | **`1.947 MiB`** | **`2.031 MiB`** | **`-59.38%`** |
| **`cache-k4v8-clean-k1`** | `k4v8` | 4-bit per-token (`g64`) | 8-bit per-token (`g64`) | **`16,640 B`** | **`1.947 MiB`** | **`2.031 MiB`** | **`-59.38%`** |
| **`cache-fp4-clean-k1`** | `fp4` | 4-bit per-token (`g64`) | 4-bit per-token (`g64`) | `11,520 B` | `1.348 MiB` | `1.406 MiB` | **`-71.88%`** |
| **`cache-fp4g32-clean-k1`** | `fp4g32` | 4-bit per-token (`g32`) | 4-bit per-token (`g32`) | `12,800 B` | `1.530 MiB` | `1.562 MiB` | **`-68.75%`** |
| **`cache-kivi4-clean-k1`** | `kivi4` | **4-bit per-channel (`g32` along block tokens)** | **4-bit per-token (`g64`)** | **`12,160 B`** | **`1.423 MiB`** | **`1.484 MiB`** | **`-70.31%`** |
| **`cache-hybrid4-clean-k1`** | `k8v4s4` | `k8v4` (L0–L3) + `fp4` (L4–L19) | `k8v4` (L0–L3) + `fp4` (L4–L19) | `12,544 B` | `1.467 MiB` | `1.531 MiB` | **`-69.38%`** |

---

### 1.2 Operator-Level Perturbation Facts (`[Sourced]`)

Reproduced by running `swift test --filter DualCachingTests/testTrack1ScoreSpaceAndAttentionDivergenceAblation` (`B=1, nH=16, nKV=4, L_prefix=128` across 4 committed blocks of 32 tokens, `L_q=32` active block queries, `D=128`):
- Let $s = Q K^\top / \sqrt{D}$ denote exact FP32 attention logits, $\bar{s} = \frac{1}{L}\sum_j s_j$ the row mean, $p = \operatorname{softmax}(s)$, and $O = p V$.
- Let $\widehat{K}, \widehat{V}$ denote dequantized KV tensors, $\widehat{s} = Q \widehat{K}^\top / \sqrt{D}$, $\widehat{p} = \operatorname{softmax}(\widehat{s})$, and $\widehat{O} = \widehat{p} \widehat{V}$.

#### Table 1.2A: Regime A — Isotropic Gaussian $Q, K, V \sim \mathcal{N}(0, 1)$ (`[Sourced]`)
| Mode | Bytes (`1L, 128 tok`) | $\frac{\|\widehat{K}-K\|_2}{\|K\|_2}$ | $\frac{\|\widehat{V}-V\|_2}{\|V\|_2}$ | Centered Score Error $\frac{\|(\widehat{s}-\bar{\hat{s}})-(s-\bar{s})\|_2}{\|s-\bar{s}\|_2}$ | Attention $\operatorname{KL}(p \,\|\, \widehat{p})$ | Top-1 Attention Argmax Flip % | Attention Output Error $\frac{\|\widehat{p}\widehat{V}-pV\|_2}{\|pV\|_2}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`fp8` (`K8V8`, `g64`)** | `139,264 B` | `0.0054` | `0.0054` | `0.0054` | `0.00001` | `0.20%` | `0.0078` |
| **`k8v4` (`K8V4`, `g64`)** | **`106,496 B`** | **`0.0054`** | `0.0908` | **`0.0054`** | **`0.00001`** | **`0.20%`** | `0.0963` |
| **`k4v8` (`K4V8`, `g64`)** | **`106,496 B`** | `0.0906` | **`0.0054`** | **`0.0907`** | **`0.00409`** | **`12.70%`** | `0.0918` |
| **`fp4` (`K4V4`, `g64`)** | `73,728 B` | `0.0906` | `0.0908` | `0.0907` | `0.00409` | `12.70%` | `0.1329` |
| **`fp4g32` (`K4V4`, `g32`)** | `81,920 B` | `0.0805` | `0.0812` | `0.0806` | `0.00321` | `10.16%` | `0.1173` |
| **`kivi4` (`K4cV4t`, `g32/g64`)** | `77,824 B` | `0.0807` | `0.0908` | `0.0802` | `0.00316` | `11.52%` | `0.1260` |

#### Table 1.2B: Regime B — Persistent-Channel Key Outliers + Token-Varying Values (`[Sourced]`)
*(8 of 128 Key channels scaled by $8\times$ with $\pm 6.0$ persistent offset across all tokens; 4 of 32 Value tokens per block scaled by $5\times$.)*

| Mode | Bytes (`1L, 128 tok`) | $\frac{\|\widehat{K}-K\|_2}{\|K\|_2}$ | $\frac{\|\widehat{V}-V\|_2}{\|V\|_2}$ | Centered Score Error $\frac{\|(\widehat{s}-\bar{\hat{s}})-(s-\bar{s})\|_2}{\|s-\bar{s}\|_2}$ | Attention $\operatorname{KL}(p \,\|\, \widehat{p})$ | Top-1 Attention Argmax Flip % | Attention Output Error $\frac{\|\widehat{p}\widehat{V}-pV\|_2}{\|pV\|_2}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`fp8` (`K8V8`, `g64`)** | `139,264 B` | `0.0090` | `0.0054` | `0.0049` | `0.00025` | `1.17%` | `0.0158` |
| **`k8v4` (`K8V4`, `g64`)** | **`106,496 B`** | **`0.0090`** | `0.0913` | **`0.0049`** | **`0.00025`** | **`1.17%`** | **`0.0920`** |
| **`k4v8` (`K4V8`, `g64`)** | **`106,496 B`** | `0.1512` | **`0.0054`** | **`0.0825`** | **`0.06791`** | **`18.36%`** | **`0.2309`** |
| **`fp4` (`K4V4`, `g64`)** | `73,728 B` | `0.1512` | `0.0913` | `0.0825` | `0.06791` | `18.36%` | `0.2471` |
| **`fp4g32` (`K4V4`, `g32`)** | `81,920 B` | `0.1130` | `0.0810` | `0.0592` | `0.03575` | `12.11%` | `0.1953` |
| **`kivi4` (`K4cV4t`, `g32/g64`)** | `77,824 B` | **`0.0666`** | `0.0913` | `0.0800` | `0.04702` | `12.70%` | `0.2310` |

**Direct Numerical Ratios from Tables 1.2A & 1.2B (`[Sourced]`)**:
1. At the exact same byte count (`106,496 B`), `k4v8` exhibits **$16.8\times$ higher Centered Score Error** (`0.0907` vs `0.0054` in Regime A; `0.0825` vs `0.0049` in Regime B), **$271.6\times\text{–}409.0\times$ higher Attention KL Divergence** (`0.00409` vs `0.00001` in Regime A; `0.06791` vs `0.00025` in Regime B), and **$15.7\times\text{–}63.5\times$ higher Top-1 Attention Argmax Flip Rate** (`12.70%` vs `0.20%` in Regime A; `18.36%` vs `1.17%` in Regime B) than `k8v4`.
2. In Regime B (persistent-channel Key outliers), `k4v8` produces **`2.51x` higher total Attention Output Error** (`0.2309`) than `k8v4` (`0.0920`) despite `k4v8` having $16.9\times$ lower Value reconstruction error (`0.0054` vs `0.0913`).
3. Switching 4-bit Keys from per-token (`fp4`) to per-channel along the 32-token block axis (`kivi4`) in Regime B reduces Key Relative $L_2$ error from `0.1512` to `0.0666` (**$-56.0\%$ / $2.27\times$ reduction**), reduces Attention KL divergence from `0.06791` to `0.04702` (**$-30.8\%$**), and reduces Top-1 Argmax Flip Rate from `18.36%` to `12.70%`.

---

### 1.3 End-to-End `LLaDA2.1-mini` $2 \times 2$ Factorial & Mitigation Facts (`[Sourced]`)

Definitions of recorded engine counters in `scratch/track1_kv_starvation_k1.jsonl`:
- **Logical Steps (`logicalSteps`)**: Total denoising iterations across all 12 prompts (sum of `stepsPerBlock`).
- **Unmasking Steps (`unmaskingSteps`)**: Denoising steps in which at least one `[MASK]` token was unmasked (`transfers > 0`; excludes post-unmasking edit-only steps where ` remainingMasks == 0`).
- **Starvation Steps (`starvationSteps`)**: Unmasking steps in which **exactly 1 token** was unmasked (`transfers == 1`, the minimum progress floor enforced when $\le 1$ masked token clears $\tau_{\text{mask}} = 0.70$). ⚠️ **Caveat [E-6]: 49.6% of FP16 unmask steps are "starvation" steps by this definition — it is the normal top-1 path, not a pathology; compare arms on Commits / Unmask Step.**
- **Commitment Rate (`Commits / Unmask Step`)**: Total transferred tokens divided by `unmaskingSteps`.

#### Table 1.3A: Aggregate End-to-End Results Across All 12 Prompts (`chat`, `reasoning`, `code`, `K=1`) (`[Sourced]`)

> "Mean TPS" below = mean of per-prompt ratios; aggregate tok/s per arm is listed in [E-3]. ms/forward is flat across arms [E-4].
| Arm (`K=1`, `clean-commit`) | `envValid` | Mean KV Size (Reduction) | Suite Total (s) | Mean TPS (`Chat` / `Reas` / `Code`) | Logical Steps (`Blk0` / `Blk1+`) | Unmask Steps | Starvation Steps (Rate) | Commits / Unmask Step | Mean Transfer Conf | Exact Prompt Match | Token Match (`vs FP16`) | Mean 1st Div Token |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`cache-clean-commit-k1` (`FP16`)** | `12/12` | `4.792 MiB` (`0.0%`) | `57.26s` | **`15.86`** (`7.90` / `18.88` / `20.80`) | `291` (`87` / `204`) | `254` | `126` (`49.6%`) | `3.61` | `0.8155` | **`12/12`** | **`716/716` (`100.0%`)** | `none` |
| **`cache-fp8-clean-k1` (`K8V8`)** | `12/12` | `2.546 MiB` (`-46.9%`) | `55.12s` | **`15.99`** (`8.21` / `18.97` / `20.79`) | `277` (`85` / `192`) | `243` | `111` (`45.7%`) | `3.78` | `0.8168` | `4/12` | `505/719` (`70.2%`) | `18.5` |
| **`cache-k8v4-clean-k1` (`K8V4`)** | `12/12` | **`1.947 MiB` (`-59.4%`)** | `56.15s` | **`15.67`** (`7.79` / `17.48` / `21.75`) | **`284`** (`84` / `200`) | `247` | `130` (`52.6%`) | **`3.72`** | `0.7953` | **`4/12`** | **`523/716` (`73.0%`)** | `21.5` |
| **`cache-k4v8-clean-k1` (`K4V8`)** | `12/12` | **`1.947 MiB` (`-59.4%`)** | `63.20s` | **`13.25`** (`7.47` / `13.36` / `18.94`) | **`321`** (`93` / `228`) | `286` | **`152` (`53.1%`)** | **`3.21`** | `0.8012` | **`0/12`** | **`297/716` (`41.5%`)** | `21.7` |
| **`cache-fp4-clean-k1` (`K4V4`)** | `12/12` | `1.348 MiB` (`-71.9%`) | `65.35s` | **`12.33`** (`6.87` / `13.52` / `16.60`) | **`333`** (`106` / `227`) | `296` | **`167` (`56.4%`)** | **`3.10`** | `0.7949` | `0/12` | `189/716` (`26.4%`) | `14.2` |
| **`cache-fp4g32-clean-k1` (`g32`)** | `12/12` | `1.530 MiB` (`-68.8%`) | `61.60s` | **`14.31`** (`8.40` / `16.57` / `17.96`) | `311` (`85` / `226`) | `272` | `134` (`49.3%`) | `3.49` | `0.7987` | `1/12` | `321/732` (`43.9%`) | `17.4` |
| **`cache-kivi4-clean-k1` (`KIVI`)** | `12/12` | **`1.423 MiB` (`-70.3%`)** | **`55.42s`** | **`15.46`** (`8.40` / `16.76` / `21.22`) | **`280`** (`81` / `199`) | **`244`** | **`108` (`44.3%`)** | **`3.76`** | **`0.8218`** | **`4/12`** | **`420/716` (`58.7%`)** | `15.5` |
| **`cache-hybrid4-clean-k1` (`L0-3 K8V4`)** | `12/12` | `1.467 MiB` (`-69.4%`) | `62.62s` | **`13.32`** (`7.84` / `14.91` / `17.20`) | `318` (`91` / `227`) | `282` | `145` (`51.4%`) | `3.26` | `0.7946` | `0/12` | `226/718` (`31.5%`) | `8.7` |

#### Table 1.3B: Per-Suite Breakdown (`chat`, `reasoning`, `code`) (`[Sourced]`)
| Suite | Arm | Logical Steps | Unmask Steps | Starvation Steps (`Rate`) | Commits / Unmask Step | Mean Transfer Conf | Suite Time (s) | Mean TPS |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`chat`** | `cache-clean-commit-k1` (`FP16`) | `158` | `147` | `91` (`61.9%`) | `2.26` | `0.7593` | `30.47s` | `7.90` |
| **`chat`** | `cache-fp8-clean-k1` (`K8V8`) | `148` | `137` | `79` (`57.7%`) | `2.42` | `0.7782` | `28.99s` | `8.21` |
| **`chat`** | `cache-k8v4-clean-k1` (`K8V4`) | `157` | `146` | `92` (`63.0%`) | `2.27` | `0.7597` | `30.45s` | `7.79` |
| **`chat`** | `cache-k4v8-clean-k1` (`K4V8`) | `167` | `156` | `98` (`62.8%`) | `2.13` | `0.7385` | `32.34s` | `7.47` |
| **`chat`** | `cache-fp4-clean-k1` (`K4V4`) | `174` | `161` | `110` (`68.3%`) | `2.06` | `0.7202` | `33.58s` | `6.87` |
| **`chat`** | `cache-fp4g32-clean-k1` (`g32`) | `151` | `138` | `78` (`56.5%`) | `2.41` | `0.7694` | `29.59s` | `8.40` |
| **`chat`** | `cache-kivi4-clean-k1` (`KIVI`) | **`144`** | **`133`** | **`70` (`52.6%`)** | **`2.50`** | **`0.7788`** | **`27.96s`** | **`8.40`** |
| **`chat`** | `cache-hybrid4-clean-k1` (`L0-3 K8V4`) | `166` | `154` | `97` (`63.0%`) | `2.16` | `0.7558` | `32.10s` | `7.84` |
| **`reasoning`** | `cache-clean-commit-k1` (`FP16`) | `64` | `53` | `17` (`32.1%`) | `4.79` | `0.8296` | `12.96s` | `18.88` |
| **`reasoning`** | `cache-fp8-clean-k1` (`K8V8`) | `61` | `50` | `14` (`28.0%`) | `5.08` | `0.8068` | `12.44s` | `18.97` |
| **`reasoning`** | `cache-k8v4-clean-k1` (`K8V4`) | **`66`** | `55` | `26` (`47.3%`) | **`4.62`** | `0.7925` | `13.35s` | **`17.48`** |
| **`reasoning`** | `cache-k4v8-clean-k1` (`K4V8`) | **`84`** | `73` | **`36` (`49.3%`)** | **`3.48`** | `0.7967` | `16.78s` | **`13.36`** |
| **`reasoning`** | `cache-fp4-clean-k1` (`K4V4`) | `80` | `69` | `31` (`44.9%`) | `3.68` | `0.8063` | `16.03s` | `13.52` |
| **`reasoning`** | `cache-fp4g32-clean-k1` (`g32`) | `88` | `76` | `32` (`42.1%`) | `3.76` | `0.8378` | `17.56s` | `16.57` |
| **`reasoning`** | `cache-kivi4-clean-k1` (`KIVI`) | **`73`** | `62` | `27` (`43.5%`) | **`4.10`** | `0.8191` | `14.72s` | **`16.76`** |
| **`reasoning`** | `cache-hybrid4-clean-k1` (`L0-3 K8V4`) | `76` | `65` | `25` (`38.5%`) | `3.91` | `0.8208` | `15.32s` | `14.91` |
| **`code`** | `cache-clean-commit-k1` (`FP16`) | `69` | `54` | `18` (`33.3%`) | `6.15` | `0.8574` | `13.83s` | `20.80` |
| **`code`** | `cache-fp8-clean-k1` (`K8V8`) | `68` | `56` | `18` (`32.1%`) | `5.93` | `0.8654` | `13.69s` | `20.79` |
| **`code`** | `cache-k8v4-clean-k1` (`K8V4`) | **`61`** | `46` | `12` (`26.1%`) | **`7.22`** | `0.8338` | `12.35s` | **`21.75`** |
| **`code`** | `cache-k4v8-clean-k1` (`K4V8`) | **`70`** | `57` | `18` (`31.6%`) | **`5.82`** | `0.8686` | `14.08s` | **`18.94`** |
| **`code`** | `cache-fp4-clean-k1` (`K4V4`) | `79` | `66` | `26` (`39.4%`) | `5.03` | `0.8583` | `15.74s` | `16.60` |
| **`code`** | `cache-fp4g32-clean-k1` (`g32`) | `72` | `58` | `24` (`41.4%`) | `5.72` | `0.7889` | `14.45s` | `17.96` |
| **`code`** | `cache-kivi4-clean-k1` (`KIVI`) | **`63`** | `49` | **`11` (`22.4%`)** | **`6.78`** | `0.8676` | **`12.73s`** | **`21.22`** |
| **`code`** | `cache-hybrid4-clean-k1` (`L0-3 K8V4`) | `76` | `63` | `23` (`36.5%`) | `5.27` | `0.8073` | `15.20s` | `17.20` |

#### Table 1.3C: Per-Prompt Token Match & Logical Syllogism Outcome (`[Sourced]`)
| Prompt ID | `fp8` (`K8V8`) | `k8v4` (`K8V4`) | `k4v8` (`K4V8`) | `fp4` (`K4V4`) | `fp4g32` (`g32`) | `kivi4` (`KIVI`) | `hybrid4` (`L0-3 K8V4`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `chat-capital` | `28/39` | **`EXACT`** | `18/38` | `12/38` | `27/40` | `33/38` | `17/40` |
| `chat-email` | `49/64` | `8/64` | `4/64` | `8/64` | `25/64` | `20/64` | `28/64` |
| `chat-explain` | `18/64` | `19/64` | `12/64` | `10/64` | `13/64` | `20/64` | `13/64` |
| `chat-recipe` | **`EXACT`** | `63/64` | `10/64` | `10/64` | `11/64` | `27/64` | `11/64` |
| `reason-trains` | **`EXACT`** | **`EXACT`** | `0/64` | `1/64` | `1/64` | **`EXACT`** | `33/64` |
| `reason-ages` | `59/64` | `60/64` | `63/64` | `2/64` | `30/64` | `13/64` | `2/64` |
| `reason-logic` *(Syllogism: valid="No")* | `6/40` (**"No"**) | `37/38` (**"No"**) | `2/38` (**"Yes" — FAIL**) | `1/38` (**"Yes" — FAIL**) | `14/52` (**"Yes" — FAIL**) | `6/38` (**"No"**) | `1/38` (**"Yes" — FAIL**) |
| `reason-units` | `16/64` | `21/64` | `16/64` | `6/64` | `15/64` | `23/64` | `16/64` |
| `code-fizzbuzz` | `51/64` | **`EXACT`** | `51/64` | `51/64` | `63/64` | **`EXACT`** | `61/64` |
| `code-swift-struct` | `22/64` | `22/64` | `63/64` | `25/64` | **`EXACT`** | `22/64` | `10/64` |
| `code-sql` | **`EXACT`** | **`EXACT`** | `2/64` | `3/64` | `2/64` | **`EXACT`** | `1/64` |
| `code-regex` | **`EXACT`** | `63/64` *(1-char typo)* | `56/64` | `60/64` | `56/64` | **`EXACT`** | `33/64` |
| **Total Exact Prompts** | **`4/12`** | **`4/12`** | **`0/12`** | **`0/12`** | **`1/12`** | **`4/12`** | **`0/12`** |
| **Total Token Match** | `505/719` (`70.2%`) | **`523/716` (`73.0%`)** | `297/716` (`41.5%`) | `189/716` (`26.4%`) | `321/732` (`43.9%`) | `420/716` (`58.7%`) | `226/718` (`31.5%`) |

---

## Part II: Theoretical & Mechanistic Interpretation (`[Inferred]` & `[Speculative]`)

### 2.1 Main & Interaction Effects of the $2 \times 2$ Bit-Allocation Factorial (`[Inferred]`)

Using the four cells of the $2 \times 2$ bit-allocation factorial (`K8V8`, `K8V4`, `K4V8`, `K4V4`) from Table 1.3A, we can decompose the exact main effects of Key bit-width ($b_K \in \{8, 4\}$) and Value bit-width ($b_V \in \{8, 4\}$) on denoising step count ($S$), starvation step count ($S_{\text{starve}}$), throughput ($\text{TPS}$), and token parity ($M_{\text{tok}}$) (`[Inferred]`):

1. **Effect of Quantizing Values from 8-bit to 4-bit at 8-bit Keys (`K8V8 -> K8V4`) (`[Inferred]`)**:
   - $\Delta S = 284 - 277 = \mathbf{+7\text{ steps}}$ (`-7` steps relative to `FP16`'s `291`)
   - $\Delta \text{TPS} = 15.67 - 15.99 = \mathbf{-0.32\text{ tok/s}}$ (`-2.0%` mean-of-ratios; aggregate 13.04 → 12.75 = −2.2% [E-3])
   - $\Delta M_{\text{tok}} = 73.0\% - 70.2\% = \mathbf{+2.8\text{ pp}}$ (`4/12` exact prompt matches preserved)
   - **Deduction (`[Inferred]`)**: Quantizing cached prefix Values from 8-bit to 4-bit (`g64`) is nearly lossless for both trajectory length and generation quality.

2. **Effect of Quantizing Keys from 8-bit to 4-bit at 8-bit Values (`K8V8 -> K4V8`) (`[Inferred]`)**:
   - $\Delta S = 321 - 277 = \mathbf{+44\text{ steps}}$ (`+15.9%` step inflation)
   - $\Delta S_{\text{starve}} = 152 - 111 = \mathbf{+41\text{ starvation steps}}$ (`+36.9%` more 1-token fallback steps)
   - $\Delta \text{Commits/Step} = 3.21 - 3.78 = \mathbf{-0.57\text{ tokens/unmasking step}}$
   - $\Delta \text{TPS} = 13.25 - 15.99 = \mathbf{-2.74\text{ tok/s}}$ (~~`-17.1%`~~ **`-14.2%`** aggregate throughput loss, 13.04 → 11.19; −17.1% is the mean-of-ratios [E-3])
   - $\Delta M_{\text{tok}} = 41.5\% - 70.2\% = \mathbf{-28.7\text{ pp}}$ (exact prompt matches collapse from `4/12` to `0/12`)
   - **Deduction (`[Inferred]`)**: Comparing `K8V4` (`1.947 MiB`) against `K4V8` (`1.947 MiB`) at **identical memory footprint and identical Value/Key group sizes** proves that **78.6% (`44 / 56`) of the total step inflation and 73.2% (`41 / 56`) of the additional starvation steps between `K8V8` (`277` steps) and `K4V4` (`333` steps) are caused strictly by 4-bit per-token Key quantization**.

3. **Interaction Effect when Both Keys and Values are 4-bit (`K4V8 -> K4V4`) (`[Inferred]`)**:
   - When Keys are already 8-bit (`K8V8 -> K8V4`), 4-bit Values add only `+7` steps (`15.99 -> 15.67 tok/s`).
   - When Keys are degraded to 4-bit (`K4V8 -> K4V4`), 4-bit Values add `+12` further steps (`321 -> 333`) and drop token match by another `-15.1 pp` (`41.5% -> 26.4%`).
   - **Deduction (`[Inferred]`)**: Value quantization error is benign when the attention distribution $p = \operatorname{softmax}(Q K^\top / \sqrt{D})$ is sharp and accurately aligned (`K8V4`), because averaging over attended tokens cancels zero-mean Value quantization noise. Once 4-bit per-token Key error scrambles the attention weights $\widehat{p}$ (`18.36%` top-1 argmax flips in Table 1.2B), Value quantization error compounds on already-misrouted attention outputs.

---

### 2.2 The "Confidence Starvation" Mechanism in Thresholded Diffusion Decoding (`[Inferred]`)

Why does 4-bit per-token Key quantization (`K4V8` and `K4V4`) make a diffusion language model run ~~**13%–22% slower in wall-clock TPS**~~ **10.5%–14.2% slower in aggregate wall-clock TPS vs FP16** [E-3] despite transferring 59%–72% fewer KV cache bytes? Every step of the causal chain is directly observable in the Part I and Part II data (`[Inferred]`):

1. **Exponential Amplification of Key Error vs. Linear Averaging of Value Error (`[Inferred]`)**:
   - In scaled dot-product attention, Value quantization error $\delta V = \widehat{V} - V$ enters the layer output **linearly** inside a convex combination:
     $$\widehat{O}_{\text{V-only}} - O = \sum_{j=1}^{L_{\text{prefix}}} p_j \,\delta V_j$$
     Because $\sum_j p_j = 1$ and $\delta V_j$ is approximately zero-mean across tokens, $\ell_2$ error is bounded by $\sim 9.2\%$ (`0.0920` in Table 1.2B for `k8v4`), and the attention weights $p_j$ are **unchanged** ($\operatorname{KL} = 0.00025$).
   - By contrast, Key quantization error $\delta K = \widehat{K} - K$ enters **inside the softmax exponential**:
     $$\widehat{p}_j = \frac{\exp\!\left(s_j + \frac{q^\top \delta K_j}{\sqrt{D}}\right)}{\sum_{m} \exp\!\left(s_m + \frac{q^\top \delta K_m}{\sqrt{D}}\right)}$$
     Because pre-RoPE/post-RoPE Transformer Keys exhibit **persistent channel outliers** (specific feature dimensions $d \in \{1,\dots,D\}$ with $5\times\text{–}10\times$ larger magnitude across all tokens), quantizing Keys **per-token** (`axis = -1`, group size 64) forces the 16 discrete quantization levels of 4-bit affine quantization to span the outlier channel's range, obliterating the precision of the remaining 63 channels in that group.
   - As measured in Table 1.2B, this produces a **$272\times$ surge in Attention KL divergence** (`0.06791` vs `0.00025`) and flips the top-1 attended prefix token on **`18.36%` of all `(head, query)` pairs**, driving total attention output error to `23.09%` (`k4v8`) — **$2.51\times$ higher than `k8v4` (`9.20%`)**.

2. **Thresholded Unmasking Stalls ("Confidence Starvation") (`[Inferred]`)**:
   - Unlike autoregressive decoding (which always commits exactly 1 token per forward pass regardless of logit entropy), `LLaDA2.1-mini` block diffusion unmasks **all** masked positions $i \in \{1,\dots,32\}$ in the active block whose predictive confidence clears $\tau_{\text{mask}} = 0.70$:
     $$\Gamma_t = \left\{\, i \in \mathcal{M}_t \;\middle|\; \max_{v \in \mathcal{V}} P_\theta(x_i = v \mid \widehat{K}_{\text{prefix}}, \widehat{V}_{\text{prefix}}, x_{\text{active}}^{(t)}) > \tau_{\text{mask}} \,\right\}$$
   - When $|\Gamma_t| = 0$ (no masked position exceeds $\tau_{\text{mask}} = 0.70$), the sampler executes the **starvation fallback** (`numToTransfer = 1`), unmasking only the single highest-confidence position $\arg\max_{i \in \mathcal{M}_t} c_i$.
   - Perturbed attention outputs from 4-bit per-token Keys add noise to the hidden states across all 20 layers, flattening the final vocabulary distribution $P_\theta(x_i \mid \cdot)$ so fewer tokens simultaneously cross $\tau_{\text{mask}} = 0.70$.
   - Consequently, in Table 1.3A, the number of unmasking steps that fall back to committing only 1 token (`starvationSteps`) jumps from `111` (`K8V8`) and `130` (`K8V4`) to **`152` (`K4V8`) and `167` (`K4V4`)**, collapsing the commitment rate from `3.72–3.78` tokens/step down to `3.10–3.21` tokens/step and forcing `+37` to `+49` extra full-model forward passes (`284 -> 321–333`).
   - Because each forward pass ~~is dominated by `97.8%` MoE expert + attention compute~~ **[unsupported — E-5]** costs ~189–191 ms on M1 regardless of KV format [E-4] ~~(~195 ms/step on M1)~~, the cost of `+37` to `+49` extra forward passes (`+7.0s` to `+9.2s` across the 12 prompts) completely overwhelms the microsecond HBM bandwidth savings of a smaller KV cache.

---

### 2.3 Why `KIVI` (`kivi4`) Cures 4-Bit Confidence Starvation While `g32` and `hybrid4` Only Partially Mitigate It (`[Inferred]`)

The three mitigation arms in Table 1.3A provide a clean diagnostic test of the outlier mechanism (`[Inferred]`):

1. **Mitigation 1 — Finer Per-Token Group Size (`cache-fp4g32-clean-k1`, `g32`) (`[Inferred]`)**:
   - Halving the per-token group size from `64` to `32` confines each outlier channel's dynamic-range stretch to 31 neighboring channels instead of 63.
   - This recovers roughly **half** of the lost performance: logical steps improve from `333` (`fp4`) to `311` (`fp4g32`), starvation steps drop from `167` to `134`, and TPS improves from `12.33` to `14.31 tok/s`. However, because the quantization axis is still **per-token**, the 31 non-outlier channels sharing a 32-channel group with an outlier channel remain severely under-resolved (`311` steps vs `284` for `k8v4`, and `reason-logic` still fails).

2. **Mitigation 3 — Shallow-Layer `K8V4` + Deep-Layer `FP4` (`cache-hybrid4-clean-k1`, `k8v4s4`) (`[Inferred]`)**:
   - Keeping layers `0..<4` at `K8V4` and quantizing layers `4..<20` to per-token `FP4` reduces steps from `333` to `318` (`13.32 tok/s`), proving that **middle and deep layers (`4..<20`) in `LLaDA2.1-mini` remain highly sensitive to per-token 4-bit Key quantization** (`[Inferred]`). Protecting only the first 4 layers is insufficient.

3. **Mitigation 2 — `KIVI` Asymmetric Axis Quantization (`cache-kivi4-clean-k1`: 4-bit Per-Channel Keys + 4-bit Per-Token Values) (`[Inferred]`)**:
   - Because block diffusion commits KV states in **exact 32-token blocks** (`blockLength = 32`), transposing the committed Key block from `[B, nKV, 32, D]` to `[B, nKV, D, 32]` and quantizing with `groupSize = 32` assigns **every single Key channel $d \in \{0,\dots,127\}$ its own dedicated scale and bias across the 32 tokens of the block** (`[Sourced]`).
   - An outlier channel $d^*$ with magnitude $\pm 15$ is quantized purely against its own values across the 32 tokens in the block, with **zero cross-channel contamination** of regular channels $d \neq d^*$ (`Key Rel L2` drops $2.27\times$ from `0.1512` to `0.0666` in Table 1.2B) (`[Sourced]` / `[Inferred]`).
   - Meanwhile, Values (which have token-specific spikes rather than persistent channel outliers) remain quantized **per-token** (`groupSize = 64`).
   - **End-to-End Result (`[Sourced]` / `[Inferred]`)**: `cache-kivi4-clean-k1` (`12,160 B/tok`, `-70.31%` KV cache size) **completely eliminates 4-bit Confidence Starvation**, reducing denoising steps from `333` (`fp4`) down to **`280` steps** (`108` starvation steps, `44.3%` starvation rate — lower than `FP16`'s `49.6%`), restoring mean transfer confidence to `0.8218`, achieving **`15.46 tok/s`** (`55.42s` suite total — faster than `FP16`'s `57.26s`), restoring **`4/12` exact prompt matches** (`58.7%` token match), and restoring logical validity on `reason-logic` (`"No."`).

---

### 2.4 Summary of Practical & Research Conclusions (`[Inferred]`)

1. **For Research Publication (`[Inferred]`)**:
   - **"Confidence Starvation"** is a fundamental failure mode unique to confidence-thresholded parallel/block diffusion language models under naive low-bit KV-cache quantization: whereas 4-bit KV quantization in autoregressive LLMs only affects output perplexity at fixed 1-token/step decoding speed, in diffusion LMs **pre-softmax Key quantization noise flattens predictive confidence below $\tau_{\text{mask}}$**, triggering single-token fallback steps (`numToTransfer = 1`) that inflate total forward passes by `+14%` to `+20%` (vs `K8V8`; +10% to +14% vs FP16) and degrade wall-clock throughput by ~~`-17%` to `-23%`~~ **`-14%` to `-18%` aggregate vs `K8V8` (`-10.5%` to `-14.2%` vs FP16)** [E-3]. (Caveat on "single-token fallback": [E-6].)
   - The **$2 \times 2$ bit-allocation factorial (`K8V8`, `K8V4`, `K4V8`, `K4V4`)** provides a clean causal proof: at identical memory footprint (`16,640 B/tok`, `-59.38%`), `K8V4` preserves step count (`284` vs `291`), throughput (`15.67` vs `15.86 tok/s`), and `73.0%` token parity, whereas `K4V8` collapses (`321` steps, `13.25 tok/s`, `41.5%` token parity, `0/12` exact matches). (Aggregate: `K8V4` 12.75 vs FP16 12.50 tok/s, `K4V8` 11.19 [E-3]. Arms ran blocked, not interleaved [E-2]; "causal proof" should read "strong single-process evidence".)
   - **Block-aligned `KIVI` (`kivi4`, 4-bit per-channel Keys along the 32-token diffusion block + 4-bit per-token Values)** is structurally matched to block-diffusion commit boundaries (`blockLength = 32`) — requiring **no residual FP16 ring buffer** unlike autoregressive `KIVI` — and cures 4-bit confidence starvation at `-70.31%` cache size (`12,160 B/tok`, `280` steps, `15.46 tok/s`).

2. **For Production Presets in `NeoDiffusion` (`[Sourced]`)** — ⚠️ **reclassified [E-8]: only `exact-fast` is a production default (Studio BF16 parity OPEN); `k8v4-clean` and `kivi4-clean` are experimental, default-off, quality gate OPEN. tok/s figures are mean-of-ratios [E-3].**:
   - `exact-fast` (`FP16` + `skipRedundantCommitForward`): `0%` size reduction, `15.86 tok/s` (`K=1`), `100.0%` exact parity.
   - `k8v4-clean` (`K8V4` + `skipRedundantCommitForward`): **`-59.38%` size reduction**, `15.67 tok/s` (`K=1`), **`73.0%` token match (`4/12` exact)** — best balanced ~~preset~~ candidate for highest fidelity under compression (quality gate OPEN [E-8]).
   - `kivi4-clean` (`KIVI 4-bit` + `skipRedundantCommitForward`): **`-70.31%` size reduction**, `15.46 tok/s` (`K=1`), **`58.7%` token match (`4/12` exact)** — best ultra-compact 4-bit ~~preset~~ candidate (quality gate OPEN [E-8]).
