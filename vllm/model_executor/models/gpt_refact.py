# coding=utf-8
"""Inference-only Refact model compatible with HuggingFace weights.

The input of the model is flattened to a 1D tensor of tokens. The model uses
InputMetadata to extract the original 2D shape of the input.
"""
import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import LlamaConfig

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearMethodBase,
                                               RowParallelLinear)
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
                                              hf_model_weights_iterator, convert_pyslice_to_tensor)
from vllm.sequence import SamplerOutput

KVCache = Tuple[torch.Tensor, torch.Tensor]


def _get_alibi_slopes(total_num_heads: int) -> torch.Tensor:
    closest_power_of_2 = 2 ** math.floor(math.log2(total_num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        dtype=torch.float32,
    )
    powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != total_num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            dtype=torch.float32,
        )
        num_remaining_heads = min(closest_power_of_2,
                                  total_num_heads - closest_power_of_2)
        extra_powers = torch.arange(start=1,
                                    end=1 + 2 * num_remaining_heads,
                                    step=2,
                                    dtype=torch.int32)
        slopes = torch.cat(
            [slopes, torch.pow(extra_base, extra_powers)], dim=0)
    return slopes


class LayerNormWithoutBias(nn.LayerNorm):

    def __init__(
            self,
            normalized_shape,
            eps: float = 1e-5,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(normalized_shape, eps, elementwise_affine=True, **factory_kwargs)
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.normalized_shape, self.weight, None, self.eps)


class RefactMLP(nn.Module):

    def __init__(
            self,
            hidden_size: int,
            mult: float,
            linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        multiple_of = 256
        intermediate_size = int(2 * (hidden_size * mult) / 3)
        self.intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size,
            2 * self.intermediate_size,
            bias=False,
            linear_method=linear_method
        )
        self.c_proj = RowParallelLinear(
            self.intermediate_size,
            hidden_size,
            bias=False,
            linear_method=linear_method
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.c_proj(x)
        return x


class RefactAttention(nn.Module):

    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.tensor_model_parallel_world_size = (
            get_tensor_model_parallel_world_size())
        tp_rank = get_tensor_model_parallel_rank()
        self.num_heads = (self.total_num_heads //
                          self.tensor_model_parallel_world_size)
        assert self.num_heads % self.tensor_model_parallel_world_size == 0
        self.head_dim = hidden_size // self.total_num_heads
        self.scaling = self.head_dim ** -0.5
        self.num_kv_heads = 1
        self.kv_dim = self.head_dim
        self.q = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            gather_output=False,
            linear_method=linear_method,
        )
        self.kv = nn.Linear(
            self.hidden_size,
            2 * self.kv_dim,
            bias=False
        )
        self.c_proj = RowParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            linear_method=linear_method,
        )
        head_start = tp_rank * self.num_heads
        head_end = (tp_rank + 1) * self.num_heads
        alibi_slopes = _get_alibi_slopes(self.num_heads)
        alibi_slopes = alibi_slopes[head_start:head_end].tolist()
        self.sa = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            alibi_slopes=alibi_slopes,
            num_kv_heads=self.num_kv_heads
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            kv_cache: KVCache,
            input_metadata: InputMetadata,
    ) -> torch.Tensor:
        q, _ = self.q(hidden_states)
        k, v = self.kv(hidden_states).split([self.kv_dim, self.kv_dim], dim=-1)
        k_cache, v_cache = kv_cache
        attn_output = self.sa(q, k, v, k_cache, v_cache, input_metadata)
        output, _ = self.c_proj(attn_output)
        return output


class RefactDecoderLayer(nn.Module):

    def __init__(
            self,
            config: LlamaConfig,
            linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attn = RefactAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            linear_method=linear_method,
        )
        self.mlp = RefactMLP(
            hidden_size=self.hidden_size,
            mult=4.0,
            linear_method=linear_method,
        )
        self.ln_1 = LayerNormWithoutBias(
            self.hidden_size,
            eps=config.layer_norm_epsilon,
        )
        self.ln_2 = LayerNormWithoutBias(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            kv_cache: KVCache,
            input_metadata: InputMetadata,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        hidden_states = self.attn(
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class RefactModel(nn.Module):

    def __init__(
            self,
            config: LlamaConfig,
            linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.wte = VocabParallelEmbedding(vocab_size, config.hidden_size)
        self.h = nn.ModuleList([
            RefactDecoderLayer(config, linear_method)
            for _ in range(config.num_hidden_layers)
        ])

    def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            kv_caches: List[KVCache],
            input_metadata: InputMetadata,
    ) -> torch.Tensor:
        hidden_states = self.wte(input_ids)
        for i in range(len(self.h)):
            layer = self.h[i]
            hidden_states = layer(
                hidden_states,
                kv_caches[i],
                input_metadata,
            )
        return hidden_states


class GPTRefactForCausalLM(nn.Module):

    def __init__(
            self,
            config: LlamaConfig,
            linear_method: Optional[LinearMethodBase] = None,
    ):
        super().__init__()
        self.config = config
        self.transformer = RefactModel(config, linear_method)
        vocab_size = ((config.vocab_size + 63) // 64) * 64
        self.ln_f = LayerNormWithoutBias(config.hidden_size, eps=config.layer_norm_epsilon)
        self.lm_head = ColumnParallelLinear(config.hidden_size,
                                            vocab_size,
                                            bias=False,
                                            linear_method=linear_method)
        self.sampler = Sampler(config.vocab_size)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[KVCache],
            input_metadata: InputMetadata,
    ) -> SamplerOutput:
        hidden_states = self.transformer(input_ids, positions, kv_caches,
                                         input_metadata)
        return self.ln_f(hidden_states)

    def sample(
            self,
            hidden_states: torch.Tensor,
            sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                   sampling_metadata)
        return next_tokens

    def load_weights(self,
                     model_name_or_path: str,
                     cache_dir: Optional[str] = None,
                     load_format: str = "auto",
                     revision: Optional[str] = None):
        tp_size = get_tensor_model_parallel_world_size()
        state_dict = self.state_dict()

        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, load_format, revision):
            loaded_weight = convert_pyslice_to_tensor(loaded_weight)

            if "wte.weight" in name or "lm_head" in name:
                param = state_dict[name]
                # Consider padding in the vocab size.
                padded_vocab_size = (param.shape[0] * tp_size)
                num_extra_rows = padded_vocab_size - self.config.vocab_size
                extra_rows = torch.empty(num_extra_rows,
                                         loaded_weight.shape[1])
                extra_rows = extra_rows.to(loaded_weight)
                loaded_weight = torch.cat([loaded_weight, extra_rows], dim=0)

            param = state_dict[name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
