from typing import Tuple
import torch

class KVCacheBase:
    def update(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

class StandardKVCache(KVCacheBase):
    def __init__(self,
                 max_batch_size: int,
                 max_seq_len: int,
                 n_layers: int,
                 n_kv_heads: int,
                 head_dim: int,
                 device: torch.device,
                 dtype: torch.dtype,
                 ):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        self.cache_k = torch.zeros(
            (
                self.n_layers,
                self.max_batch_size,
                self.n_kv_heads,
                self.max_seq_len,
                self.head_dim
            ),
            dtype=self.dtype,
            device=self.device,
        )

        self.cache_v = torch.zeros(
            (
                self.n_layers,
                self.max_batch_size,
                self.n_kv_heads,
                self.max_seq_len,
                self.head_dim
            ),
            dtype=self.dtype,
            device=self.device,
        )

    def update(self,
               layer_idx: int,
               start_pos: int,
               xk: torch.Tensor,
               xv: torch.Tensor,
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, num_heads, seq_len, head_dim = xk.shape
        self.cache_k[layer_idx, :bs, :, start_pos: start_pos + seq_len, :] = xk
        self.cache_v[layer_idx, :bs, :, start_pos: start_pos + seq_len, :] = xv

        keys = self.cache_k[layer_idx, :bs, :, start_pos: start_pos + seq_len, :]
        values = self.cache_v[layer_idx, :bs, :, start_pos: start_pos + seq_len, :]
        return keys, values


