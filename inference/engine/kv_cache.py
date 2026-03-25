import torch
from typing import Tuple

class KVCacheBase:
    def update(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

class StandardKVCache(KVCacheBase):
    def __init__(
            self,
            max_batch_size: int,
            n_layers: int,
            max_seq_len: int,
            n_kv_heads: int,
            head_dim: int,
            device: torch.device,
            dtype: torch.dtype,
    ):
        self.max_batch_size = max_batch_size
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.device = device

        self.cache_k = torch.zeros(
            (self.n_layers, self.max_batch_size, self.n_kv_heads, self.max_seq_len, self.head_dim),
            dtype=dtype,
            device=device,
        )

        self.cache_v = torch.zeros(
            (self.n_layers, self.max_batch_size, self.n_kv_heads, self.max_seq_len, self.head_dim),
            dtype=dtype,
            device=device,
        )

    def update(self, layer_idx: int, start_pos: int, xk: torch.Tensor, xv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, n_heads, seq_len, head_dim = xk.shape

        self.cache_k[layer_idx, :bs, :, start_pos:start_pos + seq_len, :] = xk
        self.cache_v[layer_idx, :bs, :, start_pos:start_pos + seq_len, :] = xv

        keys = self.cache_k[layer_idx, :bs, :, :start_pos + seq_len, :]
        values = self.cache_v[layer_idx, :bs, :, :start_pos + seq_len, :]

        return keys, values

class LatentKVCache(KVCacheBase):
    def __init__(
            self,
            n_layers: int,
            max_batch_size: int,
            max_seq_len: int,
            kv_lora_rank: int,
            rope_head_dim: int,
            device: torch.device,
            dtype: torch.dtype,
    ):
        self.n_layers = n_layers
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.kv_lora_rank = kv_lora_rank
        self.rope_head_dim = rope_head_dim

        self.cache_latent = torch.zeros(
            (self.n_layers, self.max_batch_size, self.max_seq_len, self.kv_lora_rank),
            dtype=dtype,
            device=device,
        )

        self.cache_k_rope = torch.zeros(
            (self.n_layers, self.max_batch_size, self.max_seq_len, self.rope_head_dim),
            dtype=dtype,
            device=device,
        )

    def update(self, layer_idx: int, start_pos: int, c_kv: torch.Tensor, k_rope: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, seq_len, kv_lora_rank = c_kv.shape

        self.cache_latent[layer_idx, :bs, start_pos:start_pos + seq_len, :] = c_kv
        self.cache_k_rope[layer_idx, :bs, start_pos:start_pos + seq_len, :] = k_rope

        full_c_kv = self.cache_latent[layer_idx, :bs, :start_pos + seq_len, :]
        full_k_rope = self.cache_k_rope[layer_idx, :bs, :start_pos + seq_len, :]

        return full_c_kv, full_k_rope


