# Base for LLaDA2.1
- block-wise-causal-attention
- complementary-attention-mask
- mask-to-token-m2t
- token-to-token-t2t & exposure-bias-in-dllms
- block-diffusion
----
- editable-state-evolution
- quality-mode-q-mode / speedy-mode-s-mode
- block-level-cache
- prompt-caching

# Early
- complementary-attention-mask
- configurable-threshold-decoding
- token-stability & feature-similarity-caching
- freecache
- sliding-window-attention
- suffix-dropout / scratchpad-redundancy / distance-decay
- convergence-signals / prediction-stability / per-token-freezing / token-level-early-stopping / per-token-early-stopping
- one-shot-learning & confidence-threshold-calibration
- premature-overconfidence (as evaluation)
- self-speculative-decoding-dlm

# Convergence
- one-shot-learning, confidence-threshold-calibration, dynamic-thresholding consolidated into one calibration approach

# Suggested phase-by-phase build order (synthesized recommendation, not a literal restatement of the per-row phases)

Phase 0 (get it running): block-wise-causal-attention, complementary-attention-mask, mask-to-token-m2t, token-to-token-t2t, block-diffusion — the non-negotiable architecture.

- Early: prompt-caching, suffix-dropout/sliding-window-attention (DPad-style deterministic variant), configurable-threshold-decoding + one calibration scheme (pick one of one-shot-learning/confidence-threshold-calibration), token-stability, exposure-bias-in-dllms (as a diagnostic check), premature-overconfidence (as a diagnostic check).

- Early-to-middle: self-speculative-decoding-dlm (S2D2), the Jot early-stopping cluster, hierarchical-caching (validate the caching architecture without assuming the fine-tuned-checkpoint numbers transfer), freecache.

- Middle: elastic-cache (if token-stability proves insufficient), LocalLeap family, vicinity-kv-cache-refresh (as an alternative to compare, not necessarily to keep), draft-verification/FreeDave, in-place-chain-of-thought (if reasoning workloads matter), mask-query-attention (if long-context matters), multi-block-editing-mbe, dependency-ordered-scheduling.

- Late: model-scheduling/sandwich-schedule (pending LLaDA-style validation), guided-diffusion (memory-cost permitting), radix-caching (reconsider value for on-device single-user use case), layer-wise-budgeting refinements, logic-role-guided-unmasking/inference-time-reasoning-improvement (only if the 30-minute auxiliary-head training becomes acceptable).

- Out of scope for now: DID family, text-VAE family, per-block-fp8-quantization, trajectory-distillation, hybrid-dlm-ar-decoding, RL-training concepts, sglang-rollout-engine (port ideas, not the framework).
