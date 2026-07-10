# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.generic import maybe_autocast, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from .configuration_sumi import SumiConfig
from .generation_sumi import SumiGenerationConfig, SumiGenerationMixin


try:  # Transformer Engine is an optional GPU-only backend (see attn_implementation).
    import transformer_engine.pytorch as te

    _HAS_TRANSFORMER_ENGINE = True
except ImportError:
    te = None
    _HAS_TRANSFORMER_ENGINE = False


logger = logging.get_logger(__name__)


@dataclass
class SumiMaskGenerationOutput(ModelOutput):
    """
    Output class for Sumi mask generation.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` (and `t`) are provided):
            Uniform-diffusion (GIDD) training loss: the per-token diffusion ELBO averaged over the
            non-ignored positions. This is the pretraining objective, not masked-LM cross entropy.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. See the [cache documentation](https://huggingface.co/docs/transformers/en/kv_cache)
            for more details. Only returned when caching is enabled; the bidirectional diffusion model does not use an
            incremental cache during generation.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attention weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


class SumiRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        SumiRMSNorm is equivalent to T5LayerNorm.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class SumiRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: SumiConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: SumiConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SumiMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def softmax_one(
    input: torch.Tensor,
    dim: int = -1,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Compute exp(x_i) / (1 + sum_j exp(x_j)) in a numerically stable way."""
    logits = input.to(dtype=dtype) if dtype is not None else input
    dim = dim if dim >= 0 else logits.dim() + dim
    sink = torch.zeros_like(logits.narrow(dim, 0, 1))
    probabilities = torch.softmax(torch.cat((logits, sink), dim=dim), dim=dim)
    return probabilities.narrow(dim, 0, logits.size(dim))


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attention_mask_slice = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + attention_mask_slice

    attn_weights = softmax_one(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def _te_padding_mask(attention_mask: Optional[torch.Tensor]):
    """Translate Sumi's attention mask into a Transformer Engine mask.
    Returns ``(te_mask, attn_mask_type)``. ``te_mask`` is ``None`` for the fully
    bidirectional (unpadded) case — TE then uses its fastest ``"no_mask"`` kernel —
    or a ``[B, 1, 1, S]`` boolean tensor whose ``True`` entries are padding key
    positions to ignore (``"padding"``). Accepts either the 4D additive float mask
    produced by :func:`_prepare_attention_mask` or a raw 2D ``[B, S]`` mask.
    """
    if attention_mask is None:
        return None, "no_mask"
    if attention_mask.dim() == 4:
        # [B, 1, Sq, Skv] additive mask: padded keys carry a large negative bias.
        key_mask = attention_mask[:, 0, 0, :] < 0
    elif attention_mask.dim() == 2:
        # [B, S] with 1 = keep, 0 = pad.
        key_mask = attention_mask == 0
    else:
        raise ValueError("attention_mask must be 2D or 4D for the transformer_engine path.")
    if not bool(key_mask.any()):
        return None, "no_mask"
    return key_mask[:, None, None, :], "padding"


def transformer_engine_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    """Off-by-one attention via Transformer Engine's fused ``DotProductAttention``.
    Mirrors the Megatron training path (``softmax_type="off-by-one"``, TE >= 2.8) so
    the converted model reuses the same fused kernel instead of the eager
    ``softmax_one`` fallback. ``query``/``key``/``value`` arrive as ``[B, H, S, D]``
    (key/value un-repeated for GQA — TE expands them internally) and TE consumes the
    ``bshd`` layout. The softmax scale and dropout are already baked into the
    pre-built ``module.te_attention`` module, so the ``scaling``/``dropout`` passed by
    the dispatcher are ignored here. Returns ``[B, S, H * D]``.
    """
    te_attention = module.te_attention
    query_states = query.transpose(1, 2).contiguous()  # [B, S, H, D]
    key_states = key.transpose(1, 2).contiguous()
    value_states = value.transpose(1, 2).contiguous()

    te_mask, attn_mask_type = _te_padding_mask(attention_mask)
    attn_output = te_attention(
        query_states,
        key_states,
        value_states,
        attention_mask=te_mask,
        attn_mask_type=attn_mask_type,
    )
    return attn_output, None


if _HAS_TRANSFORMER_ENGINE:
    try:  # Register so `from_pretrained(..., attn_implementation="transformer_engine")` validates.
        from transformers import AttentionInterface

        AttentionInterface.register("transformer_engine", transformer_engine_attention_forward)
    except (ImportError, AttributeError, TypeError):  # pragma: no cover - registry API differences
        pass


def _prepare_attention_mask(
    attention_mask: Optional[torch.Tensor],
    inputs_embeds: torch.Tensor,
    past_seen_tokens: int = 0,
) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None

    batch_size, query_length = inputs_embeds.shape[:2]
    expected_key_value_length = past_seen_tokens + query_length

    if attention_mask.dim() == 4:
        if past_seen_tokens > 0 and attention_mask.shape[-1] == query_length:
            attention_mask = nn.functional.pad(attention_mask, (past_seen_tokens, 0), value=0.0)
        elif attention_mask.shape[-1] < expected_key_value_length:
            raise ValueError(
                "The 4D attention_mask key length must cover the cached and current tokens."
            )
        return attention_mask
    if attention_mask.dim() != 2:
        raise ValueError("attention_mask must be 2D or 4D.")

    if past_seen_tokens > 0 and attention_mask.shape[-1] == query_length:
        attention_mask = nn.functional.pad(attention_mask, (past_seen_tokens, 0), value=1)
    elif attention_mask.shape[-1] < expected_key_value_length:
        raise ValueError("attention_mask must cover the cached and current tokens.")

    key_value_length = attention_mask.shape[-1]
    min_dtype = torch.finfo(inputs_embeds.dtype).min
    expanded_mask = attention_mask[:, None, None, :].to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
    expanded_mask = (1.0 - expanded_mask) * min_dtype
    return expanded_mask.expand(batch_size, 1, query_length, key_value_length)


class SumiAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: SumiConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False

        qkv_bias = config.attention_bias or getattr(config, "add_qkv_bias", False)
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        # Optional fused off-by-one attention via Transformer Engine (TE >= 2.8).
        self.te_attention = None
        if getattr(config, "_attn_implementation", "eager") == "transformer_engine":
            if not _HAS_TRANSFORMER_ENGINE:
                raise ImportError(
                    "attn_implementation='transformer_engine' requires Transformer Engine "
                    "(>= 2.8 for the off-by-one softmax). Install it or use the eager path."
                )
            self.te_attention = te.DotProductAttention(
                num_attention_heads=config.num_attention_heads,
                kv_channels=self.head_dim,
                num_gqa_groups=config.num_key_value_heads,
                attention_dropout=config.attention_dropout,
                qkv_format="bshd",
                attn_mask_type="no_mask",
                softmax_scale=self.scaling,
                softmax_type="off-by-one",
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        # The off-by-one softmax is honoured by both the eager fallback (softmax_one)
        # and the fused Transformer Engine path (softmax_type="off-by-one").
        attention_interface = eager_attention_forward
        if self.te_attention is not None:
            attention_interface = transformer_engine_attention_forward
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class SumiEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: SumiConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = SumiAttention(config=config, layer_idx=layer_idx)

        self.mlp = SumiMLP(config)
        self.input_layernorm = SumiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = SumiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class SumiPreTrainedModel(PreTrainedModel):
    config: SumiConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["SumiEncoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    # Transformer Engine's DotProductAttention adds an `_extra_state` entry (FP8/attention
    # metadata) that is absent from converted checkpoints; it is runtime backend state, not
    # a trained weight, so suppress the "missing key" report when the TE path is selected.
    _keys_to_ignore_on_load_missing = [r"\.te_attention\._extra_state$"]
    # Sumi relies on the off-by-one softmax, available in the eager fallback and in the
    # fused Transformer Engine path (attn_implementation="transformer_engine"); the
    # stock flash/sdpa/flex kernels cannot represent it.
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False

    _can_compile_fullgraph = True
    _supports_attention_backend = False
    _can_record_outputs = {
        "hidden_states": SumiEncoderLayer,
        "attentions": SumiAttention,
    }


@auto_docstring
class SumiModel(SumiPreTrainedModel):
    def __init__(self, config: SumiConfig):
        attn_impl = getattr(config, "_attn_implementation", "eager")
        if attn_impl == "transformer_engine" and not _HAS_TRANSFORMER_ENGINE:
            logger.warning_once(
                "attn_implementation='transformer_engine' was requested but Transformer "
                "Engine is not installed; falling back to the eager off-by-one softmax."
            )
            config._attn_implementation = "eager"
        elif attn_impl not in ("eager", "transformer_engine"):
            logger.warning_once(
                "Sumi uses an off-by-one softmax, available only in the 'eager' or "
                "'transformer_engine' attention paths. Falling back to the eager implementation."
            )
            config._attn_implementation = "eager"
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [SumiEncoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = SumiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = SumiRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        attention_mask = _prepare_attention_mask(
            attention_mask,
            inputs_embeds,
            past_seen_tokens=past_seen_tokens,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for encoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


def uniform_gidd_loss(
    logits: torch.Tensor,
    z_t: torch.LongTensor,
    labels: torch.LongTensor,
    t: torch.Tensor,
    vocab_size: int,
    beta_is: float = 1.0,
    z_loss_strength: float | None = None,
    ignore_index: int = -100,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Per-token uniform-only GIDD (discrete-diffusion) training loss.
    This is the single-process, full-vocab form of the objective the model was pretrained with
    (Megatron `--uniform-diffusion --fused-gidd-loss`). It is the exact per-token loss used by the
    shipped evaluation NELBO (`sumi_eval.nelbo`) and is equivalent to the reference Megatron loss
    with tensor-parallel world size 1.
    The uniform forward (noising) process is
        ``q(z_t = v | x) = alpha * 1[v = x] + (1 - alpha) / V``,   ``alpha = 1 - t``,
    and the per-token loss is
        ``loss = w * KL[q(.|x) || q(.|x_hat)] + beta_is * w * D_IS(z_t)``   (+ z-loss),
    where ``x_hat = softmax(logits)`` is the model's predicted clean-token distribution and ``w`` is
    the GIDD weight (the posterior probability that the position was noised).
    Args:
        logits: ``[B, S, V]`` denoiser logits over the clean token (truncated to its first ``vocab_size``).
        z_t: ``[B, S]`` noised tokens actually fed to the model (i.e. the model's ``input_ids``).
        labels: ``[B, S]`` clean target tokens ``x``; positions equal to ``ignore_index`` are dropped
            from the returned ``valid_mask`` (and excluded from the reduction by the caller).
        t: diffusion time in ``(0, 1)``, shape ``[B]`` / ``[B, 1]`` (one level per sequence) or
            ``[B, S]`` (per-token / diffusion-forcing).
        vocab_size: real vocabulary size ``V``.
        beta_is: weight of the Itakura-Saito reconstruction term.
        z_loss_strength: optional coefficient of the ``logsumexp(logits) ** 2`` z-loss.
        ignore_index: label value marking positions to exclude.
        eps: numerical clamp for ``log`` / ratios.
    Returns:
        ``(per_token_loss, valid_mask)``, both ``[B, S]``; reduce as a ``valid_mask``-weighted mean to
        obtain the scalar training loss.
    """
    logits = logits[..., :vocab_size].float()

    t = t.float()
    if t.dim() == 1:
        t = t[:, None]
    alpha = (1.0 - t).clamp(min=eps, max=1.0 - eps).expand_as(z_t)
    u = (1.0 - alpha) / float(vocab_size)

    x_hat = torch.softmax(logits, dim=-1)
    log_q_v = torch.log(alpha.unsqueeze(-1) * x_hat + u.unsqueeze(-1))

    valid_mask = labels != ignore_index
    safe_labels = labels.clamp(min=0)  # avoid gathering at ignore_index; masked out below
    log_q_at_x = log_q_v.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    sum_log_q = log_q_v.sum(dim=-1)
    x_hat_at_zt = x_hat.gather(-1, z_t.unsqueeze(-1)).squeeze(-1)

    # KL[q(.|x) || q(.|x_hat)]: H_ce collapses since q(.|x) has only the two values (alpha + u, u).
    h_ce = -alpha * log_q_at_x - u * sum_log_q
    alpha_plus_u = alpha + u
    h_q = -(
        alpha_plus_u * torch.log(alpha_plus_u.clamp(min=eps))
        + (vocab_size - 1) * u * torch.log(u.clamp(min=eps))
    )
    kl = h_ce - h_q

    # Itakura-Saito divergence at the observed noised token z_t.
    is_zt_eq_x = (z_t == labels).to(alpha.dtype)
    q_zt_x = alpha * is_zt_eq_x + u
    q_zt_x_hat = alpha * x_hat_at_zt + u
    log_ratio = torch.log(q_zt_x.clamp(min=eps)) - torch.log(q_zt_x_hat.clamp(min=eps))
    is_div = torch.exp(log_ratio) - log_ratio - 1.0

    # Per-token GIDD weight = posterior probability the position was noised.
    w_kept = (1.0 - alpha) / (1.0 + (vocab_size - 1) * alpha)
    w = torch.where(z_t == labels, w_kept, torch.ones_like(w_kept))

    loss = w * kl + beta_is * w * is_div
    if z_loss_strength is not None and z_loss_strength > 0.0:
        log_z = torch.logsumexp(logits, dim=-1)
        loss = loss + float(z_loss_strength) * log_z * log_z
    return loss, valid_mask


@auto_docstring(
    custom_intro="""
    The Sumi model with a language modeling head for uniform-diffusion (mask) generation. The base model is run with
    full bidirectional attention and a language modeling head predicts the clean token at every position, so the model
    is trained to denoise a uniformly-noised sequence rather than to predict the next token autoregressively.
    """
)
class SumiForMaskGeneration(SumiPreTrainedModel, SumiGenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    generation_config_class = SumiGenerationConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = SumiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        t: torch.Tensor | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SumiMaskGenerationOutput:
        r"""
        t (`torch.FloatTensor` of shape `(batch_size,)`, `(batch_size, 1)` or `(batch_size, sequence_length)`, *optional*):
            Per-sequence or per-token diffusion time in `(0, 1)` used to compute the uniform-diffusion (GIDD)
            training loss. Required whenever `labels` is provided: `input_ids` is then interpreted as the noised
            sequence z_t and `labels` as the clean targets x, so the data collator is responsible for the forward
            noising. Ignored during generation.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
        Example:
        ```python
        >>> from transformers import AutoModelForMaskGeneration, AutoTokenizer
        >>> model = AutoModelForMaskGeneration.from_pretrained(
        ...     "tohoku-nlp/open-uniform-diffusion",
        ...     trust_remote_code=True,
        ... )
        >>> tokenizer = AutoTokenizer.from_pretrained(
        ...     "tohoku-nlp/open-uniform-diffusion",
        ...     trust_remote_code=True,
        ... )
        >>> prompt = "Hello, my name is"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> # Uniform diffusion appends `max_new_tokens` randomly-initialized tokens to the
        >>> # prompt and iteratively denoises them. `num_denoising_steps` sets how many
        >>> # refinement passes are run; the prompt itself is kept fixed.
        >>> outputs = model.generate(**inputs, max_new_tokens=16, num_denoising_steps=128)
        >>> tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)  # doctest: +SKIP
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Sumi is a uniform-diffusion (GIDD) model: the training loss is the per-token diffusion
            # ELBO, not next-token / masked-LM cross entropy. `input_ids` is the noised sequence z_t,
            # `labels` is the clean target x, and `t` is the (per-token) diffusion time supplied by the
            # data collator. Defaults (beta_is, z_loss_strength) are kept faithful to pretraining.
            if t is None:
                raise ValueError(
                    "Sumi computes a uniform-diffusion (GIDD) loss: pass the diffusion time `t` "
                    "(shape `[B]`, `[B, 1]`, or `[B, S]`) together with `labels`. `input_ids` is "
                    "treated as the noised sequence z_t and `labels` as the clean targets x; the data "
                    "collator performs the forward noising (see the model card)."
                )
            if input_ids is None:
                raise ValueError(
                    "Computing the uniform-diffusion loss requires `input_ids` (the noised tokens z_t); "
                    "it cannot be derived from `inputs_embeds` alone."
                )
            z_t = input_ids[:, slice_indices]
            selected_labels = labels[:, slice_indices]
            selected_t = t
            if torch.is_tensor(t) and t.dim() == 2 and t.size(1) == labels.size(1):
                selected_t = t[:, slice_indices]
            per_token_loss, valid_mask = uniform_gidd_loss(
                logits=logits,
                z_t=z_t,
                labels=selected_labels,
                t=selected_t,
                vocab_size=self.config.vocab_size,
                beta_is=getattr(self.config, "uniform_diffusion_beta_is", 1.0),
                z_loss_strength=getattr(self.config, "uniform_diffusion_z_loss_strength", 1e-5),
            )
            mask = valid_mask.to(per_token_loss.dtype)
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)

        return SumiMaskGenerationOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

__all__ = [
    "SumiForMaskGeneration",
    "SumiMaskGenerationOutput",
    "SumiModel",
    "SumiPreTrainedModel",
]
