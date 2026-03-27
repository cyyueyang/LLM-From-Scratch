import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Sequence

from inference.engine.kv_cache import StandardKVCache, LatentKVCache
from models.blocks.positional_encoding.positional_encoding import RoPE, RoPEConfig
from models.blocks.normalization.normalization import RMSNorm

class StandardAttention(nn.Module):
    def __init__(self, args):
        super(StandardAttention, self).__init__()

        self.n_heads: int = args.heads
        self.n_kv_heads: int = args.n_kv_heads
        assert self.n_heads % self.n_kv_heads == 0

        self.dim: int = args.dim
        assert self.dim % self.n_heads == 0

        self.head_dim: int = self.dim // self.n_heads
        self.n_rep: int = self.n_heads // self.n_kv_heads

        self.w_qkv = nn.Linear(self.dim, self.head_dim * (self.n_heads + 2 * self.n_kv_heads), bias=False)
        self.w_o = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        mask = torch.ones((1, 1, args.max_seq_len, args.max_seq_len), dtype=torch.bool)
        mask = torch.tril(mask)
        self.register_buffer("mask", mask)

    def forward(
            self, x: torch.Tensor,
            rope: RoPE,
            layer_idx: int,
            kv_cache: Optional[StandardKVCache] = None,
            start_pos: int = 0,
            paged_attention_inputs: Optional[Sequence[torch.Tensor]] = None,
            **kwargs
    ):
        if paged_attention_inputs is not None:
            return self._forward_paged(x, rope, layer_idx, paged_attention_inputs)

        bs, seq_len, dim = x.shape
        xq, xk, xv = self.w_qkv(x).split(
            [
                self.n_heads * self.head_dim,
                self.n_kv_heads * self.head_dim,
                self.n_kv_heads * self.head_dim
            ],
            dim=-1,
        )
        xq = xq.view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(bs, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(bs, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        xq = rope.apply_rotary_emb(xq)
        xk = rope.apply_rotary_emb(xk)

        if kv_cache is not None:
            keys, values = kv_cache.update(
                layer_idx,
                start_pos,
                xk,
                xv
            )
        else:
            keys, values = xk, xv

        if self.n_rep > 1:
            keys = keys.repeat_interleave(self.n_rep, dim=1)
            values = values.repeat_interleave(self.n_rep, dim=1)

        scores = torch.matmul(xq, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # seq_len > 1 说明是在prefill阶段
        if seq_len > 1:
            current_seq_len = start_pos + seq_len
            scores = scores.masked_fill(~self.mask[:, :, start_pos: current_seq_len, :current_seq_len], -float('inf'))

        probs = F.softmax(scores, dim=-1)
        output = torch.matmul(probs, values)
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, dim)
        output = self.w_o(output)

        return output

    def _forward_paged(
            self, x: torch.Tensor,
            rope: RoPE,
            layer_idx: int,
            paged_attention_inputs: Optional[Sequence[torch.Tensor]] = None
    ):
        # x shape [num_tokens, d_model]
        # positions shape [num_tokens, ]
        # tokens_per_seq shape [batchsize, ]
        # context_lengths shape [batch_size, ]
        # k/v cache shape [num_blocks, num_layers, n_kv_heads, block_size, head_dim]
        # block_tables [batch_size, max_num_blocks_per_seq]

        positions, tokens_per_seq, context_lengths, k_cache, v_cache, block_tables = paged_attention_inputs

        xq, xk, xv = self.w_qkv(x).split(
            [
                self.n_heads * self.head_dim,
                self.n_kv_heads * self.head_dim,
                self.n_kv_heads * self.head_dim
            ],
            dim=-1,
        )

        xq = xq.view(-1, self.n_heads, self.head_dim)
        xk = xk.view(-1, self.n_kv_heads, self.head_dim)
        xv = xv.view(-1, self.n_kv_heads, self.head_dim)

        xq = rope.apply_rotary_emb_paged(xq, positions)
        xk = rope.apply_rotary_emb_paged(xk, positions)

        block_size = k_cache.shape[3]
        token_idx = 0
        for seq_idx, num_tokens in enumerate(tokens_per_seq):
            num_tokens = num_tokens.item()
            current_ctx_len = context_lengths[seq_idx].item()
            start_pos = current_ctx_len - num_tokens

            for i in range(num_tokens):
                pos = start_pos + i
                block_idx = block_tables[seq_idx, pos // block_size].item()
                offset = pos % block_size

                k_cache[block_idx, layer_idx, :, offset, :] = xk[token_idx]
                v_cache[block_idx, layer_idx, :, offset, :] = xv[token_idx]

                token_idx += 1
        # [num_tokens, n_heads, head_dim]
        output = torch.zeros_like(xq)

        token_idx = 0
        for seq_idx, num_tokens in enumerate(tokens_per_seq):
            num_tokens = num_tokens.item()
            seq_len = context_lengths[seq_idx].item()

            gathered_k = torch.zeros(self.n_kv_heads, seq_len, self.head_dim, device=x.device, dtype=x.dtype)
            gathered_v = torch.zeros(self.n_kv_heads, seq_len, self.head_dim, device=x.device, dtype=x.dtype)

            for pos in range(seq_len):
                block_idx = block_tables[seq_idx, pos // block_size].item()
                offset = pos % block_size
                gathered_k[:, pos, :] = k_cache[block_idx, layer_idx, :, offset, :]
                gathered_v[:, pos, :] = v_cache[block_idx, layer_idx, :, offset, :]

            if self.n_rep > 1:
                gathered_k = gathered_k.repeat_interleave(self.n_rep, dim=0)
                gathered_v = gathered_v.repeat_interleave(self.n_rep, dim=0)

            q_curr = xq[token_idx: token_idx + num_tokens].transpose(0, 1)

            scores = torch.matmul(q_curr, gathered_k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            if num_tokens > 1:
                q_pos = positions[token_idx: token_idx + num_tokens]
                k_pos = torch.arange(seq_len, device=x.device)
                mask = q_pos.unsqueeze(1) < k_pos.unsqueeze(0)
                scores = scores.masked_fill(mask.unsqueeze(0), -float('inf'))

            probs = F.softmax(scores, dim=-1)
            attn_out = torch.matmul(probs, gathered_v)
            output[token_idx: token_idx + num_tokens] = attn_out.transpose(0, 1)

            token_idx += num_tokens

        output_flat = output.view(-1, self.n_heads * self.head_dim)

        return self.w_o(output_flat)

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dim: int = args.dim
        self.n_heads: int = args.n_heads
        self.q_lora_rank: int = args.q_lora_rank
        self.kv_lora_rank: int = args.kv_lora_rank
        self.nope_head_dim: int = args.nope_head_dim
        self.rope_head_dim: int  = args.rope_head_dim
        self.v_head_dim: int = args.v_head_dim

        if self.q_lora_rank > 0:
            self.wq_down = nn.Linear(self.dim, self.q_lora_rank, bias=False)
            self.q_norm = RMSNorm(self.q_lora_rank, args.norm_eps)
            self.wq_up = nn.Linear(self.q_lora_rank, self.n_heads * self.nope_head_dim, bias=False)
            self.wq_rope = nn.Linear(self.q_lora_rank, self.n_heads * self.rope_head_dim, bias=False)

        else:
            self.wq_up = nn.Linear(self.dim, self.n_heads * self.nope_head_dim, bias=False)
            self.wq_rope = nn.Linear(self.dim, self.n_heads * self.rope_head_dim, bias=False)

        self.wkv_down = nn.Linear(self.dim, self.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, args.norm_eps)

        self.wkv_up = nn.Linear(self.kv_lora_rank, self.n_heads * (self.nope_head_dim + self.v_head_dim), bias=False)
        self.wk_rope = nn.Linear(self.dim, self.rope_head_dim, bias=False)

        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)

        mask = torch.ones((1, 1, args.max_seq_len, args.max_seq_len), dtype=torch.bool)
        mask = torch.tril(mask)
        self.register_buffer("mask", mask)

    def forward(
            self,
            x: torch.Tensor,
            rope: RoPE,
            layer_idx: int,
            kv_cache: Optional[LatentKVCache] = None,
            start_pos: int = 0,
            paged_attention_inputs=None,
            **kwargs
    ):
        bs, seq_len, d_model = x.shape
        # kv cache + decode stage 进行优化
        if kv_cache is not None and seq_len == 1:
            return self._forward_inference_optimized(x, rope, layer_idx, kv_cache, start_pos)

        if paged_attention_inputs is not None:
            raise NotImplementedError("Paged attention for MLA is not implemented")

        if self.q_lora_rank > 0:
            q_compressed = self.wq_down(x)
            q_compressed = self.q_norm(q_compressed)
            q_nope = self.wq_up(q_compressed).view(bs, seq_len, self.n_heads, self.nope_head_dim)
            q_pe = self.wq_rope(q_compressed).view(bs, seq_len, self.n_heads, self.rope_head_dim)
        else:
            q_nope = self.wq_up(x).view(bs, seq_len, self.n_heads, self.nope_head_dim)
            q_pe = self.wq_rope(x).view(bs, seq_len, self.n_heads, self.rope_head_dim)

        kv_compressed = self.wkv_down(x)
        kv_compressed = self.kv_norm(kv_compressed)

        kv_up = self.wkv_up(kv_compressed).view(bs, seq_len, self.n_heads, self.nope_head_dim + self.v_head_dim)
        k_nope, v = kv_up.split([self.nope_head_dim, self.v_head_dim], dim=-1)

        k_rope_shared = self.wk_rope(x).view(bs, seq_len, 1, self.rope_head_dim)

        if kv_cache is not None:
            k_rope_for_cache = k_rope_shared.squeeze(2)
            kv_cache.update(layer_idx, start_pos, kv_compressed, k_rope_for_cache)

        # rope stage
        q_pe = q_pe.transpose(1, 2)
        k_rope_shared = k_rope_shared.transpose(1, 2)
        q_pe = rope.apply_rotary_emb(q_pe)
        k_rope_shared = rope.apply_rotary_emb(k_rope_shared)

        # 对 nope 和 v 进行转置 把头维度放在前面

        q_nope = q_nope.transpose(1, 2)
        k_nope = k_nope.transpose(1, 2)
        v = v.transpose(1, 2)

        # 对 nope rope concat 起来
        q = torch.cat([q_nope, q_pe], dim=-1)
        k_rope = k_rope_shared.expand(-1, -1, self.n_heads, -1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        # attention stage
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.nope_head_dim + self.rope_head_dim)

        if seq_len > 1:
            local_mask = self.mask[:, :, :seq_len, :seq_len]
            scores = scores.masked_fill(~local_mask, -float("inf"))

        probs = torch.softmax(scores, dim=-1)
        output = torch.matmul(probs, v)
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.wo(output)

    def _forward_inference_optimized(
            self,
            x: torch.Tensor,
            rope: RoPE,
            layer_idx: int,
            kv_cache: LatentKVCache,
            start_pos: int
    ):
        """
        scores = s_pe + s_nope 的优化 并且进行矩阵吸收
        """
        bs, seq_len, d_model = x.shape
        assert seq_len == 1

        if self.q_lora_rank > 0:
            q_compressed = self.q_norm(self.wq_down(x))
            q_nope = self.wq_up(q_compressed).view(bs, seq_len, self.n_heads, self.nope_head_dim)
            q_pe = self.wq_rope(q_compressed).view(bs, seq_len, self.n_heads, self.rope_head_dim)

        else:
            q_nope = self.wq_up(x).view(bs, seq_len, self.n_heads, self.nope_head_dim)
            q_pe = self.wq_rope(x).view(bs, seq_len, self.n_heads, self.rope_head_dim)

        kv_compressed = self.kv_norm(self.wkv_down(x))
        # 为什么没有1 那个 维度 方便存 kvcache
        k_rope_shared = self.wk_rope(x).view(bs, seq_len, self.rope_head_dim)

        # rope stage

        positions = torch.arange(start_pos, start_pos + seq_len, dtype=torch.long, device=x.device)
        positions = positions.unsqueeze(0).expand(bs, -1).flatten()

        q_pe_flat = q_pe.view(bs * seq_len, self.n_heads, self.rope_head_dim)
        q_pe_out = rope.apply_rotary_emb_paged(q_pe_flat, positions)
        # nope rope 分开算 所以分开转置
        q_pe = q_pe_out.view(bs, seq_len, self.n_heads, self.rope_head_dim).transpose(1, 2)

        k_rope_flat = k_rope_shared.view(bs * seq_len, 1, self.rope_head_dim)
        k_rope_out = rope.apply_rotary_emb_paged(k_rope_flat, positions)
        # 为什么没有1 那个 维度 方便存 kvcache
        k_rope_shared = k_rope_out.view(bs, seq_len, self.rope_head_dim)

        full_c_kv, full_k_rope = kv_cache.update(layer_idx, start_pos, kv_compressed, k_rope_shared)

        k_rope = full_k_rope.unsqueeze(1)
        # 计算s_pe
        score_pe = torch.matmul(q_pe, k_rope.transpose(-2, -1))

        # 计算s_nope

        w_up_weight = None
        if hasattr(self.wkv_up, '_packed_params'):
            if hasattr(self.wkv_up._packed_params, 'unpack'):
                w_up_weight, _ = self.wkv_up._packed_params.unpack()
            else:
                # 兜底：使用全局算子 (针对 Windows/FBGEMM 后端)
                try:
                    w_up_weight, _ = torch.ops.quantized.linear_unpack(self.wkv_up._packed_params)
                except Exception as e:
                    # 如果连这也失败了，那确实是环境问题了
                    raise RuntimeError(f"Failed to unpack quantized weights: {e}")
        else:
            # 标准 nn.Linear
            w_up_weight = self.wkv_up.weight

        head_dim_total = self.nope_head_dim + self.v_head_dim

        w_up_reshaped = w_up_weight.view(self.n_heads, head_dim_total, self.kv_lora_rank)

        w_uk = w_up_reshaped[:, :self.nope_head_dim, :]
        w_uv = w_up_reshaped[:, self.nope_head_dim:, :]

        q_nope = q_nope.transpose(1, 2)
        q_absorbed = torch.einsum("b h t d, h d r -> b h t r", q_nope, w_uk)
        scores_nope = torch.matmul(q_absorbed, full_c_kv.transpose(-2, -1).unsqueeze(1))

        scores = (scores_nope + score_pe) / math.sqrt(self.nope_head_dim + self.rope_head_dim)

        probs = torch.softmax(scores.float(), dim=-1).type_as(x)

        latent_output = torch.matmul(probs, full_c_kv.unsqueeze(1))
        output = torch.einsum("b h t r, h v r -> b h t v", latent_output, w_uv)

        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.wo(output)
























