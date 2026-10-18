# A-0 — Related work: KV quantization and decoding parallelism in threshold-decoded dLLMs

Search: arXiv, OpenReview/proceedings, general web (date in the S records). Tags: **[sourced]** = stated in the cited
record; **[inferred]** = our reading across records; **[speculative]** = not checked.

## What is known

- Threshold decoding makes tokens-committed-per-step a function of per-position confidence. [sourced: S-009]
- KV-cache quantization has been applied to dLLMs, but only in the last few months and only for block
  diffusion, where committed-block KV is persistent: Q-MaskAhead (4-bit KV, includes LLaDA-2.0-mini)
  [sourced: S-001], SoloQ (KV quantized at block commit; threshold-decoded models in the evaluation)
  [sourced: S-002], QWaF (sub-2-bit KV on SDAR; claims first dLLM KV quantizer) [sourced: S-003].
- All three report task accuracy only — no step counts, tokens per step, NFE or threshold-crossing rates
  for quantized vs full-precision KV. [sourced: S-001, S-002, S-003 (abstract only)]
- SoloQ notes that quantization changes a threshold-decoded model's EOS trajectory, and handles it for
  latency by replaying a fixed BF16 forward sequence — holding the forward count fixed. [sourced: S-002]
- Weight/activation PTQ for dLLMs (S-004–S-008): FAIR-Calib reports quantization flipping borderline
  commit decisions and measures commit-step logit margins [sourced: S-004]. None quantizes KV or reports
  step counts; S-005 lists steps-vs-quantization as future work. [sourced: S-004, S-005, S-006]
- Key-side asymmetry: Q-MaskAhead finds 2-bit keys cost more accuracy than 2-bit values at similar cache
  size, on block diffusion including LLaDA-2.0-mini. [sourced: S-001] In AR LLMs, per-channel key /
  per-token value quantization is established [sourced: S-013, S-014]; in video diffusion, quantized keys
  bias attention mass toward cached positions [sourced: S-015].
- Per-channel (KIVI-style) keys are already used in dLLMs: both Q-MaskAhead and SoloQ quantize keys
  per-channel by default. [sourced: S-001, S-002] Neither compares per-token vs per-channel keys.
- Caching/reuse work (S-010–S-012, S-016–S-018) studies staleness or sparsity, not precision. [sourced]

## What is not known (as far as this search reached)

- Whether KV quantization changes the **number of denoising steps / tokens committed per step** under
  confidence-threshold decoding. No verified paper measures it. [inferred from the records above]
- Whether a K-vs-V asymmetry exists **in that quantity** (as opposed to accuracy). [inferred]
- Whether per-channel or pre-gain key quantization restores steps per block. [inferred]
- Confidence shifts at masked positions under KV quantization; the only margin analysis found is for
  W4A4 weights/activations. [sourced: S-004; inferred: gap]

## Closest prior work

1. **S-001 (Q-MaskAhead)** — LLaDA-2.0-mini, 4-bit KV, K-vs-V ablation, per-channel keys; accuracy and
   memory only. [sourced]
2. **S-002 (SoloQ)** — KV quantized at block commit with KIVI-style keys, threshold-decoded block-diffusion
   models, observes trajectory change; does not measure step count. [sourced]
3. **S-004 (FAIR-Calib)** — decision flips and margins under quantization; weights/activations only. [sourced]

## Verdict on novelty

- **(a) Step inflation ("parallelism erosion") from KV quantization under threshold decoding**: not
  reported in any verified source. Likely novel as a measured effect. [inferred] Risk: S-003's full text was
  not read; an unindexed NFE table could exist. [speculative]
- **(b) K-vs-V asymmetry under threshold decoding**: the *accuracy* asymmetry (keys more sensitive) is
  already reported by S-001 on LLaDA-2.0-mini, so a keys-are-worse claim alone is not novel. Novel only
  if stated for steps/parallelism and with a mechanism. [inferred]
- **(c) Per-channel key quantization as a fix in dLLMs**: not novel as a technique — S-001 and S-002 use it
  by default, after S-013/S-014. A claim that it removes step inflation would be new evidence, not a new
  method; a pre-gain (qk-norm) key quantization variant was not found in any source. [inferred]

## Unverified / not checked

- QWaF full paper (workshop abstract only). [S-003]
- Fast-dLLM v2, EntropyCache, d2Cache, Flash-dLLM, GroupKV: snippet-level only; not recorded. [speculative]
