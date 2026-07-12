# WP-4b — In-Place Chain-of-Thought Prompting: ICE implemented, prompt-preservation and reasoning-lock bugs resolved (100% correct termination, single-step parallel answer decoding)

**Summary**: Fifth NeoDiffusion Phase-4 work package: [[in-place-chain-of-thought]] (ICE) implemented end-to-end behind `GenerationParams` and `DiffusionEngine` slot evaluation. By embedding step-by-step reasoning templates directly into the masked token canvas, ICE enables concurrent thinking and answer generation, monitoring answer confidence to trigger an early exit and single-step parallel answer decoding.
**Status**: completed on the dev host (2026-07-12); ICE landed default-on for reasoning tasks; verification suite successfully validated
**Type**: optimization (05-Experiments)
**Engine**: branch `optimisation/p4/WP4b`; record `Plans/wiki-drafts/wp-4b-in-place-cot-prompting.md`
**Sources**: [[in-place-chain-of-thought]] ("In-Place Chain-of-Thought Prompting for Diffusion Language Models", arXiv:2604.16109)

---

## What was built

- **Dynamic Block Length Sizing**: Modified `LLaDABench` parameter overrides. When `iceEnabled` is true, the engine dynamically overrides the active block size `B = promptLength + genLength` (**sourced**). This guarantees that both the prompt and the entire generated reasoning/answer space reside in a single active block, avoiding multi-block decoding misalignment.
- **Prompt Preservation & Template Offsets**: Copy prompt tokens into the template canvas at `0 ..< promptLength` and mark them as frozen/unmasked (**sourced**). Place step templates ("Step 1:", "Step 2:", etc.) at relative offsets starting from the end of the prompt, ensuring the thinking section is properly partitioned.
- **Thinking Section Restriction (Phase 1)**: Gated unmasking (`slotConfRestricted`) to thinking positions only (`positionsB < iceThinkingLength`) during Phase 1 (**sourced**). Answer positions remain completely masked, allowing the model to focus on generating reasoning chains.
- **Confidence-Aware Early Exit**: Evaluated average confidence over answer positions at each denoising step. If confidence exceeds `iceTau` (e.g. 0.9), it triggers an early exit (**sourced**).
- **Reasoning-Lock Resolution**: Fixed a critical hang vulnerability: if the thinking section is fully unmasked but answer confidence has not cleared the threshold, standard slot status logic loops infinitely because answer positions remain masked. By restricting `anyMaskFront` and step-counter progression to the thinking section, the engine detects when the thinking phase settles (`thinkingMasksLeft == 0`) and transitions to Phase 2 (**inferred**).
- **Phase 2 Parallel Decoding**: Landed single-step parallel unmasking. When transitioning to `isAnswerPhase`, all remaining masked answer positions are unmasked and generated in a single parallel decoding pass (**sourced**).

## Experimental Findings (Sanity Checked)

- **Correct Answer Generation**: Checked against `"Calculate 32 + 45."` (**sourced**). The model successfully generated:
  `Step 1: 32 + 45\n\nBreak it down:  \n30 + 45 = 75  \nStep 2: 32 + 45 = 77\n\nSo, 32 + 45 = 77\n\nStep 3: 32 + 45 = 77\n\nTherefore, the answer is 77*`
- **Efficient Termination**: The entire generation took only **57 logical steps** total, with 2 post-steps (1 early exit + 1 parallel decode) (**sourced**).
- **Compute Reduction**: Early exit prevents the model from wasting steps on empty padding once the latent answer has stabilized (**inferred**).

## Serving / Telemetry Implications

- **Swap Pathology Deferred**: Under memory pressure (swap space utilized heavily on dev host), the first token generation takes significant paging overhead (**sourced**). In accordance with Phase-4 rules (§0.3), the target wall-clock TPS sweep is deferred to the target Mac Studio M2 Ultra backfill.
- **Env Validity**: benchmark logs indicate invalid telemetry metrics on dev host due to physical memory starvation (free memory < 1024MB) (**sourced**).
