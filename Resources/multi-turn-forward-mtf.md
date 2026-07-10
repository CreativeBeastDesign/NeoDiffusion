# Multi-Turn Forward (MTF)

**Summary**: Data augmentation technique that exposes the model to diverse editing scenarios by simulating multiple forward passes through the diffusion process during training.  
**Aliases**: MTF, multi-pass training  
**Status**: stable  
**Last updated**: 2026-04-16  
**Sources**:
- [[llada2-1-tech-report]]

---

## Definition

Multi-Turn Forward (MTF) is a training data augmentation method used in LLaDA2.1's CPT and SFT stages. It involves running multiple forward passes through the diffusion process on the same training example, generating different intermediate states and thus exposing the model to a wider variety of masked and noisy token configurations. This enhances the model's editing capabilities by showing it more scenarios where it must recover from perturbations.

## Why it matters

MTF is crucial for teaching the model the "Draft-and-Edit" paradigm. Without sufficient exposure to editing scenarios during training, the model would not learn to effectively perform T2T operations at inference. By artificially creating more editing opportunities, MTF ensures the model develops robust correction abilities.

## Mechanism

- Take a training sequence.
- Apply several diffusion steps to generate intermediate states with various masking patterns.
- For each intermediate state, compute loss using the Mixture of M2T and T2T objectives.
- This effectively simulates multiple forward trajectories from the same starting point.

## Trade-offs

- **Training cost**: More forward passes per example → increased compute.
- **Coverage**: Better coverage of state space → stronger editing performance.
- **Overfitting risk**: If not properly randomized, may encourage memorization of specific trajectories.

## Apple Silicon implications

- MTF is training-time only; inference does not use it directly.
- However, the model weights resulting from MTF training expect editable inference; kernels must support T2T.
- Training infrastructure (dFactory) would need to be ported to Metal for on-device training; MTF would benefit from Metal Performance Shaders' compute capabilities.

## Related concepts
- [[mixture-of-m2t-and-t2t-objective]]
- [[editable-state-evolution]]
- [[multi-block-editing-mbe]]

## Open questions
- How many MTF turns are optimal for convergence on hardware with limited memory (like Apple Silicon)?
- Can MTF be approximated during inference via on-the-fly augmentation to improve quality without full RL?