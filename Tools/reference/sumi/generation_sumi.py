# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Iterative denoising sampler for the uniform-only GIDD diffusion language model.

For uniform-only GIDD the noise distribution is constant ``π = 1/V`` and the
forward marginal is::

    q(z_t = v | x) = α_t · 1[v = x] + (1 - α_t) / V

Given a model that predicts ``x̂(z_t)``, ``z_s`` (with ``s < t`` in the denoising
direction) is sampled from the analytic ancestral posterior::

    p(z_s = v | z_t, x̂) ∝ q(z_s = v | x̂) · q(z_t | z_s = v)

This is the Hugging Face port of ``tools/run_uniform_diffusion_sampling.py``;
the same two samplers (``"ancestral"`` and ``"adaptive"``) are exposed through
:meth:`SumiGenerationMixin.generate`.
"""

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F

from transformers.cache_utils import Cache
from transformers.generation import GenerationConfig, GenerationMixin
from transformers.utils import ModelOutput


@dataclass
class SumiGenerationOutput(ModelOutput):
    """Output class for uniform diffusion generation."""

    sequences: torch.LongTensor
    past_key_values: Cache | None = None
    logits: None = None
    hidden_states: None = None
    canvas: torch.LongTensor | None = None  # full untrimmed denoised canvas


class SumiGenerationConfig(GenerationConfig):
    """Generation configuration for uniform diffusion language models."""

    def __init__(
        self,
        num_denoising_steps: int = 128,
        sampler: str = "ancestral",
        schedule: str = "linear",
        min_log_snr: float = -9.0,
        max_log_snr: float = 9.0,
        tokens_per_step: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if num_denoising_steps < 1:
            raise ValueError("num_denoising_steps must be a positive integer.")
        self.num_denoising_steps = num_denoising_steps
        self.sampler = sampler
        self.schedule = schedule
        self.min_log_snr = min_log_snr
        self.max_log_snr = max_log_snr
        self.tokens_per_step = tokens_per_step

    @classmethod
    def from_model_config(cls, model_config):
        # transformers>=5.12's GenerationConfig.from_model_config compares the generation
        # config against a base GenerationConfig() that lacks this subclass's custom fields
        # (num_denoising_steps, sampler, ...), raising AttributeError. Build via the base
        # class, then re-wrap so our extra defaults are preserved. Robust on 5.8-5.12+.
        base = GenerationConfig.from_model_config(model_config)
        return cls(**base.to_dict())


def _make_log_snr_schedule(
    num_steps: int,
    schedule: str,
    min_log_snr: float,
    max_log_snr: float,
    device: torch.device,
) -> torch.Tensor:
    """``[num_steps + 1]`` log-SNR values, ascending (noisy → clean)."""
    if schedule == "linear":
        alpha_ts = torch.linspace(1e-4, 1.0 - 1e-4, num_steps + 1, device=device)
    elif schedule == "cosine":
        ts = torch.linspace(1.0 - 1e-3, 1e-3, num_steps + 1, device=device)
        alpha_ts = 0.5 + 0.5 * torch.cos(ts * math.pi)
    else:
        raise ValueError(f"Unknown schedule: {schedule!r}")
    alpha_ts = alpha_ts.clamp(min=1e-6, max=1.0 - 1e-6)
    log_snrs = torch.log(alpha_ts) - torch.log1p(-alpha_ts)
    return log_snrs.clamp(min=min_log_snr, max=max_log_snr)


@torch.no_grad()
def _compute_logits(
    model,
    z: torch.Tensor,
    attention_mask: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """Forward pass → raw ``[B, S, vocab_size]`` logits truncated to the real vocab.

    Entries beyond ``vocab_size`` are dropped (matching how training masks padded
    vocab); returned as fp32. Shared by both the ancestral path (which softmaxes
    with temperature, via :func:`_compute_x_hat`) and the adaptive path (which
    needs the raw logits to separate the un-tempered selection distribution from
    the tempered commit distribution).
    """
    logits = model(
        input_ids=z,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits
    return logits[..., :vocab_size].float()


@torch.no_grad()
def _compute_x_hat(
    model,
    z: torch.Tensor,
    attention_mask: torch.Tensor,
    vocab_size: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Forward pass → ``softmax(logits / temperature)`` truncated to the real vocab.

    ``temperature`` sharpens (``< 1``) or flattens (``> 1``) the model's predicted
    clean-token distribution before it feeds the analytic ancestral posterior.
    ``1.0`` reproduces full posterior sampling; lower values make denoising more
    decisive. The ancestral step that samples ``z_s`` from the posterior keeps its
    own inherent stochasticity either way.
    """
    logits = _compute_logits(model, z, attention_mask, vocab_size)
    if temperature != 1.0:
        logits = logits / max(float(temperature), 1e-6)
    return F.softmax(logits, dim=-1)


def _ancestral_step(
    z_t: torch.Tensor,
    x_hat: torch.Tensor,
    log_snr_t: float,
    log_snr_s: float,
    vocab_size: int,
    generator: Optional[torch.Generator],
    eps: float = 1e-12,
) -> torch.Tensor:
    """One ancestral denoising step ``z_t → z_s`` (with ``s < t``).

    Shapes: ``z_t [B, S]``, ``x_hat [B, S, V]``. Returns ``[B, S]``.
    """
    device = z_t.device
    dtype = x_hat.dtype

    alpha_t = torch.sigmoid(torch.tensor(log_snr_t, device=device, dtype=dtype))
    alpha_s = torch.sigmoid(torch.tensor(log_snr_s, device=device, dtype=dtype))
    alpha_t_s = alpha_t / alpha_s.clamp(min=eps)
    beta_t = 1.0 - alpha_t
    beta_s = 1.0 - alpha_s
    beta_t_s = (1.0 - alpha_t_s).clamp(min=0.0)

    inv_v = 1.0 / vocab_size
    u_t = beta_t * inv_v
    u_s = beta_s * inv_v
    u_t_s = beta_t_s * inv_v

    q_s = alpha_s * x_hat + u_s

    one_hot_zt = F.one_hot(z_t, num_classes=vocab_size).to(dtype)
    q_t_given_s = alpha_t_s * one_hot_zt + u_t_s

    x_hat_at_zt = x_hat.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)
    q_t_at_zt = (alpha_t * x_hat_at_zt + u_t).clamp(min=eps)

    posterior = q_s * q_t_given_s / q_t_at_zt.unsqueeze(-1)
    posterior = posterior.clamp(min=0.0)

    flat = posterior.reshape(-1, vocab_size)
    sampled = torch.multinomial(flat, num_samples=1, generator=generator).squeeze(-1)
    return sampled.view_as(z_t)


def _adaptive_step(
    z_t: torch.Tensor,
    logits: torch.Tensor,
    noise_mask: torch.Tensor,
    tokens_per_step: int,
    temperature: float,
    generator: Optional[torch.Generator],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One confidence-based ("adaptive") denoising step ``z_t → z_s``.

    Implements the generalized confidence heuristic of von Rütte et al. (ICLR 2026,
    "Scaling Behavior of Discrete Diffusion Language Models", App. A.1, Eq. 9):

        conf(z_t) = p_prior(z_t) · ( max_z' p_θ(x=z'|z_t) − p_θ(x=z_t|z_t) )

    For the UNIFORM-only model the prior is constant ``π = 1/V`` over the whole vocab
    (there is no mask token), so it drops out of the ranking — the confidence reduces
    to ``p_max − p_curr`` ("how much the model wants to change this token").

    Unlike :func:`_ancestral_step`, which resamples EVERY denoise position each step,
    this commits ONLY the top ``tokens_per_step`` highest-confidence denoise positions
    to their predicted token (argmax when ``temperature == 0``, else sampled from
    ``softmax(logits/temperature)``) and leaves all other positions unchanged. The SNR
    schedule is unused, so more steps than tokens are allowed — a position may be
    revisited and overwritten on a later step.

    Shapes: ``z_t [B, S]``, ``logits [B, S, V]`` (un-tempered, fp32),
    ``noise_mask [B, S]`` (True = denoise here, False = frozen prompt/anchor).
    Returns ``(z_s [B, S], selected_pos [B, k])``. Frozen positions are guaranteed
    unchanged.
    """
    vocab_size = logits.shape[-1]

    # Selection distribution: un-tempered softmax (the confidence compares the model's
    # raw belief in the best token vs. the current token).
    x_hat = F.softmax(logits, dim=-1)
    p_max = x_hat.max(dim=-1).values                                  # [B, S]
    p_curr = x_hat.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)          # [B, S]
    conf = p_max - p_curr                                             # >= 0 (uniform prior const)
    # Never pick a frozen position: force its confidence below any real one.
    conf = conf.masked_fill(~noise_mask, float("-inf"))

    k = max(1, min(int(tokens_per_step), conf.shape[-1]))
    next_pos = torch.topk(conf, k, dim=-1).indices                   # [B, k]

    # Commit distribution: tempered (temperature == 0 -> greedy argmax).
    if temperature and float(temperature) > 0.0:
        probs = F.softmax(logits / max(float(temperature), 1e-6), dim=-1)
        pred = torch.multinomial(
            probs.reshape(-1, vocab_size), num_samples=1, generator=generator
        ).view_as(z_t)
    else:
        pred = logits.argmax(dim=-1)

    z_s = z_t.clone()
    batch_idx = torch.arange(z_t.shape[0], device=z_t.device).unsqueeze(-1)
    z_s[batch_idx, next_pos] = pred[batch_idx, next_pos]
    # Restore every frozen position (prompt + anchors) unconditionally.
    return torch.where(noise_mask, z_s, z_t), next_pos


def _greedy_step(
    z_t: torch.Tensor,
    logits: torch.Tensor,
    noise_mask: torch.Tensor,
) -> torch.Tensor:
    """One greedy denoising step ``z_t → z_s`` (the original naive Sumi sampler).

    Every denoise position is overwritten each step with the model's argmax
    prediction; there is no SNR schedule, no temperature and no stochasticity.
    Frozen positions (prompt + anchors) are restored via ``noise_mask``.

    Shapes: ``z_t [B, S]``, ``logits [B, S, V]``. Returns ``[B, S]``.
    """
    pred = logits.argmax(dim=-1)
    return torch.where(noise_mask, pred, z_t)


def _build_noise_mask(
    generated_ids: torch.Tensor,
    prompt_length: int,
    frozen: Optional[List] = None,
    denoise_end: Optional[List[int]] = None,
) -> torch.Tensor:
    """``[B, S]`` mask (``True`` = denoise here, ``False`` = frozen prompt/anchor).

    Prompt positions ``[:prompt_length]`` are frozen. ``frozen`` pins
    ``(position, token_id)`` pairs at fixed canvas positions and writes the token id
    into ``generated_ids`` in place (a flat ``[(pos, tid), ...]`` shared by every row,
    or a PER-ROW ``[[(pos, tid), ...], ...]`` of length ``batch_size``). ``denoise_end``
    (per-row exclusive upper bound) freezes the tail at its prior-random init so the
    step budget concentrates on the content window.
    """
    batch_size, total_length = generated_ids.shape
    device = generated_ids.device
    noise_mask = torch.zeros((batch_size, total_length), dtype=torch.bool, device=device)
    noise_mask[:, prompt_length:] = True

    if frozen:
        per_row = len(frozen) > 0 and isinstance(frozen[0], list)
        if per_row and len(frozen) != batch_size:
            raise ValueError(
                f"per-row frozen must have batch_size ({batch_size}) entries, got {len(frozen)}"
            )
        for i in range(batch_size):
            row_frozen = frozen[i] if per_row else frozen
            for pos, tid in row_frozen:
                if 0 <= pos < total_length and bool(noise_mask[i, pos]):  # not frozen by a prompt
                    generated_ids[i, pos] = tid
                    noise_mask[i, pos] = False

    if denoise_end is not None:
        if len(denoise_end) != batch_size:
            raise ValueError(
                f"denoise_end must have batch_size ({batch_size}) entries, got {len(denoise_end)}"
            )
        for i in range(batch_size):
            end = int(denoise_end[i])
            if 0 <= end < total_length:
                noise_mask[i, end:] = False  # tail stays at its prior-random init

    return noise_mask


class SumiGenerationMixin(GenerationMixin):
    """Generation mixin for uniform diffusion language models."""

    @staticmethod
    def _resolve(value, name, default, generation_config, self_generation_config):
        """Explicit kwarg → passed generation_config → ``self.generation_config`` → default."""
        if value is not None:
            return value
        for config in (generation_config, self_generation_config):
            if config is not None:
                resolved = getattr(config, name, None)
                if resolved is not None:
                    return resolved
        return default

    def _trim_at_eos(self, generated_ids, prompt_length):
        """Cut each row at the first EOS in its generated region (dropping the EOS, the
        anchored BOS delimiter, and the denoised tail), then right-pad the batch back into
        a rectangular tensor. Prompt tokens are kept. Mirrors the eval harness's
        ``_extract_text`` (`gen_ids[:gen_ids.index(eos)]`). The pad filler is a special
        token, so ``decode(..., skip_special_tokens=True)`` yields clean text."""
        eos_id = self.config.eos_token_id
        if eos_id is None:
            return generated_ids
        pad_id = self.config.pad_token_id
        pad_id = eos_id if pad_id is None else pad_id
        batch_size = generated_ids.shape[0]
        rows, max_len = [], prompt_length
        for i in range(batch_size):
            row = generated_ids[i]
            hit = (row[prompt_length:] == eos_id).nonzero(as_tuple=True)[0]
            cut = prompt_length + int(hit[0].item()) if hit.numel() > 0 else row.shape[0]
            rows.append(row[:cut])
            max_len = max(max_len, cut)
        out = torch.full((batch_size, max(max_len, 1)), pad_id,
                         dtype=generated_ids.dtype, device=generated_ids.device)
        for i, row in enumerate(rows):
            if row.shape[0] > 0:
                out[i, : row.shape[0]] = row
        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        max_length: Optional[int] = None,
        num_denoising_steps: Optional[int] = None,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        sampler: Optional[str] = None,
        schedule: Optional[str] = None,
        min_log_snr: Optional[float] = None,
        max_log_snr: Optional[float] = None,
        temperature: Optional[float] = None,
        tokens_per_step: Optional[int] = None,
        frozen: Optional[List] = None,
        denoise_end: Optional[List[int]] = None,
        canvas_length: Optional[int] = None,
        anchor_eosbos: Optional[bool] = None,
        trim_at_eos: Optional[bool] = None,
        progress_callback: Optional[Callable] = None,
        **kwargs,
    ) -> SumiGenerationOutput:
        generation_config = kwargs.pop("generation_config", None)
        inputs = kwargs.pop("inputs", None)
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        self_generation_config = getattr(self, "generation_config", None)

        if inputs_embeds is not None:
            raise ValueError("Diffusion generation currently requires input_ids; inputs_embeds is not supported.")
        if input_ids is None:
            input_ids = inputs

        kwargs.pop("return_dict_in_generate", None)

        num_denoising_steps = self._resolve(
            num_denoising_steps, "num_denoising_steps", 128, generation_config, self_generation_config
        )
        if num_denoising_steps < 1:
            raise ValueError("num_denoising_steps must be a positive integer.")

        sampler = self._resolve(sampler, "sampler", "ancestral", generation_config, self_generation_config)
        if sampler not in ("ancestral", "adaptive", "greedy"):
            raise ValueError(
                f"sampler must be 'ancestral', 'adaptive' or 'greedy', got {sampler!r}"
            )
        schedule = self._resolve(schedule, "schedule", "linear", generation_config, self_generation_config)
        min_log_snr = self._resolve(min_log_snr, "min_log_snr", -9.0, generation_config, self_generation_config)
        max_log_snr = self._resolve(max_log_snr, "max_log_snr", 9.0, generation_config, self_generation_config)
        temperature = self._resolve(temperature, "temperature", 1.0, generation_config, self_generation_config)
        tokens_per_step = self._resolve(
            tokens_per_step, "tokens_per_step", 1, generation_config, self_generation_config
        )
        canvas_length = self._resolve(canvas_length, "canvas_length", 2048, generation_config, self_generation_config)
        anchor_eosbos = self._resolve(anchor_eosbos, "anchor_eosbos", True, generation_config, self_generation_config)
        trim_at_eos = self._resolve(trim_at_eos, "trim_at_eos", True, generation_config, self_generation_config)

        device = self.model.embed_tokens.weight.device
        if input_ids is None:
            bos_token_id = kwargs.pop("bos_token_id", None)
            if bos_token_id is None and generation_config is not None:
                bos_token_id = getattr(generation_config, "bos_token_id", None)
            if bos_token_id is None:
                bos_token_id = self.config.bos_token_id
            if bos_token_id is None:
                raise ValueError("input_ids is required when bos_token_id is not configured.")
            input_ids = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
        elif not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        else:
            input_ids = input_ids.to(device=device, dtype=torch.long)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape (batch_size, sequence_length).")

        batch_size, prompt_length = input_ids.shape
        if max_new_tokens is None and generation_config is not None:
            max_new_tokens = getattr(generation_config, "max_new_tokens", None)
        if max_new_tokens is None and self_generation_config is not None:
            max_new_tokens = getattr(self_generation_config, "max_new_tokens", None)
        if max_new_tokens is None:
            if max_length is None and generation_config is not None:
                max_length = getattr(generation_config, "max_length", None)
            if max_length is None and self_generation_config is not None:
                max_length = getattr(self_generation_config, "max_length", None)
            if max_length is None:
                raise ValueError("Either max_new_tokens or max_length must be provided for diffusion generation.")
            max_new_tokens = max_length - prompt_length

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative.")
        if max_new_tokens == 0:
            return SumiGenerationOutput(sequences=input_ids, canvas=input_ids)

        # The model is trained on a packed, fixed-length canvas, so generation runs on a
        # full `canvas_length` canvas (default 2048) rather than just prompt+max_new_tokens.
        # `max_new_tokens` is the content budget: the EOS,BOS document delimiter is anchored
        # at prompt_length+max_new_tokens, the rest of the canvas is denoised as context, and
        # decoding is cut at the first EOS (trim_at_eos, default True).
        ceiling = self.config.max_position_embeddings
        canvas_length = min(int(canvas_length), ceiling)
        reserve = 2 if anchor_eosbos else 0
        if prompt_length + reserve >= canvas_length:
            raise ValueError(
                f"prompt_length ({prompt_length}) leaves no room in canvas_length ({canvas_length})."
            )
        budget = max(1, min(max_new_tokens, canvas_length - prompt_length - reserve))
        total_length = canvas_length
        completion_length = total_length - prompt_length

        # Anchor the EOS,BOS delimiter at the end of the content budget. The caller can
        # override by passing an explicit `frozen` set (then it takes precedence).
        if anchor_eosbos and frozen is None:
            eos_pos = prompt_length + budget
            frozen = [(eos_pos, self.config.eos_token_id), (eos_pos + 1, self.config.bos_token_id)]

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        elif not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.tensor(attention_mask, device=device)
        else:
            attention_mask = attention_mask.to(device=device)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids.")

        if generator is None and seed is not None:
            try:
                generator = torch.Generator(device=device)
            except RuntimeError:
                generator = torch.Generator()
            generator.manual_seed(seed)

        completion_shape = (batch_size, completion_length)
        try:
            completion_ids = torch.randint(
                low=0,
                high=self.vocab_size,
                size=completion_shape,
                dtype=torch.long,
                device=device,
                generator=generator,
            )
        except RuntimeError:
            completion_ids = torch.randint(
                low=0,
                high=self.vocab_size,
                size=completion_shape,
                dtype=torch.long,
                generator=generator,
            ).to(device)

        generated_ids = torch.cat([input_ids, completion_ids], dim=-1)
        completion_attention_mask = torch.ones(
            completion_shape,
            dtype=attention_mask.dtype,
            device=device,
        )
        generation_attention_mask = torch.cat([attention_mask, completion_attention_mask], dim=-1)

        # noise_mask follows the reference _init_z: prompt frozen at the front, plus
        # optional fixed anchors (frozen) and a per-row active-region bound (denoise_end).
        noise_mask = _build_noise_mask(generated_ids, prompt_length, frozen, denoise_end)

        log_snrs = _make_log_snr_schedule(
            num_denoising_steps, schedule, min_log_snr, max_log_snr, device
        )

        was_training = self.training
        self.eval()
        try:
            for step in range(num_denoising_steps):
                selected_pos = None
                log_snr_t = log_snr_s = None
                if sampler == "ancestral":
                    log_snr_t = float(log_snrs[step])
                    log_snr_s = float(log_snrs[step + 1])
                    x_hat = _compute_x_hat(
                        self, generated_ids, generation_attention_mask, self.vocab_size, temperature
                    )
                    if progress_callback is not None:
                        top1_prob, top1_z = x_hat.max(dim=-1)
                    else:
                        top1_prob = top1_z = None
                    z_new = _ancestral_step(
                        generated_ids, x_hat, log_snr_t, log_snr_s, self.vocab_size, generator
                    )
                else:
                    # adaptive / greedy both read the raw logits; the SNR schedule is unused.
                    logits = _compute_logits(self, generated_ids, generation_attention_mask, self.vocab_size)
                    if progress_callback is not None:
                        top1_prob, top1_z = F.softmax(logits, dim=-1).max(dim=-1)
                    else:
                        top1_prob = top1_z = None
                    if sampler == "adaptive":
                        # Confidence-based commit (Eq. 9).
                        z_new, selected_pos = _adaptive_step(
                            generated_ids, logits, noise_mask, tokens_per_step, temperature, generator
                        )
                    else:  # greedy: overwrite every denoise position with the argmax.
                        z_new = _greedy_step(generated_ids, logits, noise_mask)

                generated_ids = torch.where(noise_mask, z_new, generated_ids)

                if progress_callback is not None:
                    progress_callback(
                        {
                            "step": step + 1,
                            "num_steps": num_denoising_steps,
                            "log_snr_t": log_snr_t,
                            "log_snr_s": log_snr_s,
                            "z": generated_ids,
                            "top1_z": top1_z,
                            "top1_prob": top1_prob,
                            "noise_mask": noise_mask,
                            # Adaptive only: the [B, k] confidence-selected positions
                            # this step committed (None for ancestral).
                            "selected_pos": selected_pos,
                        }
                    )
        finally:
            if was_training:
                self.train()

        sequences = self._trim_at_eos(generated_ids, prompt_length) if trim_at_eos else generated_ids
        return SumiGenerationOutput(sequences=sequences, canvas=generated_ids)


__all__ = [
    "SumiGenerationOutput",
    "SumiGenerationConfig",
    "SumiGenerationMixin",
]
