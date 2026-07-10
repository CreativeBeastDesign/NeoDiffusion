# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, DynamicCache
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.generation.utils import GenerationConfig


class StableDiffcoderForCausalLM(LlamaForCausalLM):
    def _get_num_transfer_tokens(self, mask_map, steps):
        # Only bs == 1 is supported for now
        mask_num = mask_map.sum().long().item()

        base = mask_num // steps
        remainder = mask_num % steps

        num_transfer_tokens = torch.full(
            (steps,), fill_value=base, device=mask_map.device, dtype=torch.long
        )

        num_transfer_tokens[:remainder] += 1

        return num_transfer_tokens

    def _make_block_causal_mask(
        self, seq_len, block_size=2, device=None, dtype=torch.bfloat16
    ):
        num_blocks = (seq_len + block_size - 1) // block_size
        block_mask = torch.tril(
            torch.ones((num_blocks, num_blocks), dtype=torch.bool, device=device)
        )
        local_block = torch.ones(
            (block_size, block_size), dtype=torch.bool, device=device
        )
        mask = block_mask.kron(local_block)[:seq_len, :seq_len]

        attention_mask = mask.float()
        attention_mask.masked_fill_(~mask, -torch.inf)
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).to(dtype)
        return attention_mask

    def _get_transfer_index(
        self,
        logits,
        temperature,
        remasking,
        mask_index,
        x,
        num_transfer_token,
        threshold=None,
        shift=False,
    ):
        def add_gumbel_noise(logits, temperature):
            if temperature == 0:
                return logits
            logits = logits.to(torch.float64)
            noise = torch.rand_like(logits, dtype=torch.float64)
            gumbel_noise = (-torch.log(noise)) ** temperature
            return logits.exp() / gumbel_noise

        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
        if shift:
            x0 = torch.cat([x[:, :1], x0[:, :-1]], dim=-1)
            pad = torch.zeros_like(logits[:, :1])
            logits = torch.cat([pad, logits[:, :-1]], dim=1)
        if remasking == "low_confidence":
            p = F.softmax(logits.to(torch.float64), dim=-1)
            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
            )  # b, l
        elif remasking == "random":
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        else:
            raise NotImplementedError(remasking)

        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)

        transfer_map = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        if threshold is not None:
            num_transfer_token = mask_index.sum(dim=1, keepdim=True)
        _, select_index = torch.topk(confidence[0], k=num_transfer_token)
        transfer_map[0, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_token):
                if confidence[0, select_index[k]] < threshold:
                    transfer_map[0, select_index[k]] = False
        return x0, transfer_map

    @torch.no_grad()
    def generate_block(
        self,
        input_ids: torch.LongTensor,
        steps=128,
        gen_length=128,
        block_length=4,
        temperature=0.0,
        remasking="low_confidence",
        tokenizer=None,
        mask_id=5,
        threshold=0.95,
        shift=False,
        eos_id=None,
    ):
        x = torch.cat(
            [
                input_ids,
                torch.full(
                    (input_ids.shape[0], gen_length),
                    mask_id,
                    dtype=torch.long,
                    device=input_ids.device,
                ),
            ],
            dim=1,
        )

        assert gen_length % block_length == 0, (
            "gen_length must be divisible by block_length"
        )
        gen_blocks = gen_length // block_length

        assert steps % gen_blocks == 0, (
            "steps must be divisible by the number of generation blocks"
        )
        steps = steps // gen_blocks

        assert x.shape[0] == 1, (
            "Only batch size of 1 is supported for block-wise generation currently."
        )

        prompt_length = input_ids.shape[1]
        gen_block_list = [block_length for _ in range(gen_blocks)]

        # Fix 3: Only handle residual blocks if the prompt length is NOT cleanly divisible
        remainder = prompt_length % block_length
        if remainder != 0:
            res_block = block_length - remainder
            gen_block_list = [res_block] + gen_block_list
            gen_block_list[-1] = block_length - res_block
            gen_blocks += 1
        cum_block = [sum(gen_block_list[: i + 1]) for i in range(len(gen_block_list))]

        block_diffusion_attention_mask = self._make_block_causal_mask(
            prompt_length + gen_length,
            block_length,
            self.device,
            dtype=torch.bfloat16,
        )

        past_key_values = DynamicCache()

        nfe = 0
        final_flag = False
        prefill_length = prompt_length // block_length * block_length

        if prefill_length > 0:
            cur_attn_mask = block_diffusion_attention_mask[
                ..., :prefill_length, :prefill_length
            ]
            # Fix 1: Explicitly pass cache_position for newer transformers prefill
            # actually not necessary since transformers will automatically generate it for prefilling
            # if unspecified, but the official `generate` method does pass it,
            # so we follow that for consistency and to avoid potential issues in future transformers updates
            cache_pos = torch.arange(prefill_length, device=x.device)
            self(
                x[:, :prefill_length],
                past_key_values=past_key_values,
                attention_mask=cur_attn_mask,
                use_cache=True,
                cache_position=cache_pos,
            )

        for block_id, block_size in enumerate(gen_block_list):
            block_start = (
                prompt_length + cum_block[block_id - 1]
                if block_id > 0
                else prefill_length
            )
            block_end = prompt_length + cum_block[block_id]

            block_mask_map = x[:, block_start:block_end] == mask_id
            num_transfer_tokens = self._get_num_transfer_tokens(block_mask_map, steps)

            replace_position = torch.zeros_like(x, dtype=torch.bool)
            replace_position[:, block_start:block_end] = True

            for token_count in num_transfer_tokens:
                if token_count > 0:
                    nfe += 1
                    mask_map = x[:, block_start:block_end] == mask_id
                    attention_mask = block_diffusion_attention_mask[
                        ..., block_start:block_end, :block_end
                    ]
                    output = self(
                        x[:, block_start:block_end],
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        cache_position=replace_position.nonzero(as_tuple=True)[1],
                    )
                    logits = output.logits

                    past_key_values.crop(block_start)

                    x0, transfer_map = self._get_transfer_index(
                        logits,
                        temperature,
                        remasking,
                        mask_map,
                        x[:, block_start:block_end],
                        token_count.item() if threshold is None else None,
                        threshold,
                        shift=shift,
                    )
                    x[:, block_start:block_end][transfer_map] = x0[transfer_map]

                if (x[:, block_start:block_end] == mask_id).sum() == 0:

                   # Fix 2: Calculate where the generated tokens ACTUALLY start in this block
                    gen_start = max(block_start, prompt_length)

                    if (
                        eos_id is not None
                        and gen_start < block_end
                        and (x[:, gen_start:block_end] == eos_id).sum() > 0
                    ):
                        final_flag = True
                        x = x[:, :block_end]
                        eos_pos = (x[:, gen_start:block_end] == eos_id).nonzero(as_tuple=True)[1][0].item() + gen_start
                        x[0, eos_pos:] = eos_id
                        break

                    nfe += 1
                    self(
                        x[:, block_start:block_end],
                        attention_mask=block_diffusion_attention_mask[
                            ..., block_start:block_end, :block_end
                        ],
                        past_key_values=past_key_values,
                        use_cache=True,
                        cache_position=replace_position.nonzero(as_tuple=True)[1],
                    )
                    break

            if final_flag:
                break

        return x, nfe

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        generation_config: GenerationConfig = None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids must be provided")

        if generation_config is None:
            generation_config = self.generation_config

        output_ids, nfe = self.generate_block(
            input_ids=input_ids,
            **kwargs,
        )

        return output_ids
