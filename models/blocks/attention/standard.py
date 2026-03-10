import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from typing import Tuple

from ..positional_encoding.positional_encoding import RoPE
from inference.engine.kv_cache import StandardKVCache, LatentKVCache
from ..normalization.normalization import RMSNorm

import sys
from pathlib import Path
project_dir = Path(__file__).parent.parent.parent
sys.path.append(str(project_dir))


class StandardAttention(nn.Module):
    def __init__(self, args):
        super(StandardAttention, self).__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.d_model = args.d_model

        assert self.d_model % self.n_heads == 0
        self.head_dim = self.d_model // self.n_heads

        assert self.n_heads % self.n_kv_heads == 0
        self.n_rep = self.n_heads // self.n_kv_heads

        self.w_q = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.w_kv = nn.Linear(self.d_model, 2 * self.n_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(self.n_heads * self.head_dim, args.d_model, bias=False)

        mask = torch.ones((1, 1, args.max_seq_len, args.max_seq_len), dtype=torch.bool)
        mask = torch.tril(mask)
        self.register_buffer('mask', mask)

    def forward(self, x: torch.Tensor, rope: RoPE, layer_idx: int, kv_cache: StandardKVCache = None, start_pos=0, paged_attention_inputs=None, **kwargs) -> torch.Tensor:
        if paged_attention_inputs is not None:
            return self._forward_paged(x, rope, layer_idx, paged_attention_inputs)

        bs, seq_len, d_model = x.size()

        x_q = self.w_q(x).view(bs, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        x_k, x_v = self.w_kv(x).chunk(2, dim=-1)
        x_k = x_k.view(bs, seq_len, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        x_v = x_v.view(bs, seq_len, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        x_q = rope.apply_rotary_emb(x_q)
        x_k = rope.apply_rotary_emb(x_k)

        if kv_cache is not None:
            keys, values = kv_cache.update(layer_idx, start_pos, x_k, x_v)
        else:
            keys, values = x_k, x_v

        if self.n_rep > 1:
            keys = keys.repeat_interleave(self.n_rep, dim=1)
            values = values.repeat_interleave(self.n_rep, dim=1)

        scores = torch.matmul(x_q, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 针对prefill 阶段 因为 decode 阶段 不需要mask
        if seq_len > 1:
            current_seq_len = start_pos + seq_len
            mask = self.mask[:, :, start_pos: current_seq_len, :current_seq_len]

            scores = scores.masked_fill(~mask, -1e9)

        probs = F.softmax(scores.float(), dim=-1).type_as(x_q)
        output = torch.matmul(probs, values)
        output = output.permute(0, 2, 1, 3).contiguous().view(bs, seq_len, -1)

        return self.w_o(output)

    def _forward_paged(self, x: torch.Tensor, rope: RoPE, layer_idx: int, paged_inputs) -> torch.Tensor:
        """
        paged_attention 前向传播，支持边长序列的高效推理
        通过 Block Table 将逻辑序列位置映射到物理kv缓存块， 实现动态的内存管理
        和序列之间的内存共享

        :param x: 形状[num_tokens, d_model]
        :param rope: ROPE模块，提供  apply_rotary_emb_paged 方法
        :param layer_idx: 层索引 用于kv cache中
        :param paged_inputs: 包含分页注意力需要的6个元素的元组
            - positions (torch.Tensor): 每个token的绝对位置 [num_tokens]
            - tokens_per_seq (torch.Tensor): batch中每个序列的token数量 [batch_size]
            - context_lengths (torch.Tensor): 每个序列的完整上下文长度，用于计算attention的key长度 [batch_size]
            - k_cache (torch.Tensor): 分页键缓存 [num_blocks, num_layers, n_kv_heads, block_size, head_dim]
            - v_cache (torch.Tensor): 分页值缓存 [num_blocks, num_layers, n_kv_heads, block_size, head_dim]
            - block_tables (torch.Tensor): 块映射表 [batch_size, max_num_blocks]  per-squence 而不是per-layers
        :return: 注意力输出，形状 [total_num_tokens, d_model]，与输入 x 的
        第一维相同，已包含残差 dropout 和输出投影。
        """

        positions, tokens_per_seq, context_lengths, k_cache, v_cache, block_tables = paged_inputs
        x_q = self.w_q(x).view(-1, self.n_heads, self.head_dim)
        x_k, x_v = self.w_kv(x).chunk(2, dim=-1)
        x_k = x_k.view(-1, self.n_kv_heads, self.head_dim)
        x_v = x_v.view(-1, self.n_kv_heads, self.head_dim)

        x_q = rope.apply_rotary_emb_paged(x_q, positions)
        x_k = rope.apply_rotary_emb_paged(x_k, positions)

        block_size = k_cache.size(-2)

        token_idx = 0

        for seq_idx, num_tokens in enumerate(tokens_per_seq):
            num_tokens = num_tokens.item()
            current_ctx_length = context_lengths[seq_idx].item()
            start_pos = current_ctx_length - num_tokens

            for i in range(num_tokens):
                pos = start_pos + i
                block_idx = block_tables[seq_idx, pos // block_size].item()
                offset = pos % block_size

                k_cache[block_idx, layer_idx, :, offset, :] = x_k[token_idx]
                v_cache[block_idx, layer_idx, :, offset, :] = x_v[token_idx]

                token_idx += 1

        output = torch.zeros_like(x_q)
        token_idx = 0
        for seq_idx, num_tokens in enumerate(tokens_per_seq):
            num_tokens = num_tokens.item()
            seq_len = context_lengths[seq_idx].item()

            gathered_k = torch.zeros(
                self.n_kv_heads,
                seq_len,
                self.head_dim,
                dtype=x.dtype,
                device=x.device
            )
            gathered_v = torch.zeros(
                self.n_kv_heads,
                seq_len,
                self.head_dim,
                dtype=x.dtype,
                device=x.device
            )

            for pos in range(seq_len):
                block_idx = block_tables[seq_idx, pos // block_size].item()
                offset = pos % block_size
                gathered_k[:, pos, :] = k_cache[block_idx, layer_idx, :, offset, :]
                gathered_v[:, pos, :] = v_cache[block_idx, layer_idx, :, offset, :]

            if self.n_rep > 1:
                gathered_k = gathered_k.repeat_interleave(self.n_rep, dim=0)
                gathered_v = gathered_v.repeat_interleave(self.n_rep, dim=0)

            q_curr = x_q[token_idx: token_idx + num_tokens].transpose(0, 1)
            scores = torch.matmul(q_curr, gathered_k.transpose(1, 2))  / math.sqrt(self.head_dim)

            if num_tokens > 1:
                q_pos = positions[token_idx: token_idx + num_tokens]
                k_pos = torch.arange(seq_len, device=x.device)

                mask = q_pos.unsqueeze(1) < k_pos.unsqueeze(0)
                scores = scores.masked_fill(mask.unsqueeze(0), -1e9)

            probs = F.softmax(scores.float(), dim=-1).type_as(x_q)
            attn_out = torch.matmul(probs, gathered_v)

            output[token_idx: token_idx + num_tokens] = attn_out.transpose(0, 1)
            token_idx += num_tokens

        output_flat = output.view(-1, self.n_heads * self.head_dim)

        return self.w_o(output_flat)


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadLatentAttention, self).__init__()

        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank

        self.n_heads = args.n_heads
        self.d_model = args.d_model
        self.nope_head_dim = args.nope_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.v_head_dim = args.v_head_dim

        if self.q_lora_rank > 0:
            self.wq_down = nn.Linear(self.d_model, self.q_lora_rank, bias=False)
            self.wq_up = nn.Linear(self.q_lora_rank, self.n_heads * self.nope_head_dim, bias=False)
            self.wq_rope = nn.Linear(self.q_lora_rank, self.n_heads * self.rope_head_dim, bias=False)
            self.q_norm = RMSNorm(self.q_lora_rank, eps=args.norm_eps)

        else:
            self.wq_up = nn.Linear(self.d_model, self.n_heads * self.nope_head_dim, bias=False)
            self.wq_rope = nn.Linear(self.d_model, self.n_heads * self.rope_head_dim, bias=False)

        self.wkv_down = nn.Linear(self.d_model, self.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=args.norm_eps)

        self.wkv_up = nn.Linear(self.kv_lora_rank, self.n_heads * (self.nope_head_dim + self.v_head_dim), bias=False)

        self.wk_rope = nn.Linear(self.d_model, self.rope_head_dim, bias=False)

        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.d_model, bias=False)

        mask = torch.ones((1, 1, args.max_seq_len, args.max_seq_len), dtype=torch.bool)
        mask = torch.tril(mask)
        self.register_buffer("mask", mask)

    def forward(self,
                x: torch.Tensor,
                rope: RoPE,
                layer_idx: int,
                kv_cache: LatentKVCache,
                start_pos: int = 0,
                paged_attention_inputs: Tuple[torch.Tensor, ...] = None) -> torch.Tensor:
        bs, seq_len, _ = x.size()

        if seq_len == 1 and kv_cache is not None:
            return self._forward_inference_optimized(x, rope, layer_idx, kv_cache, start_pos)

        if paged_attention_inputs is not None:
            raise NotImplementedError
        # 这部分 只用于 prefill or training
        if self.q_lora_rank > 0:
            q_compressed = self.q_norm(self.wq_down(x))
            q_nope = self.wq_up(q_compressed).view(bs, seq_len, self.n_heads, self.nope_head_dim)
            q_rope = self.wq_rope(q_compressed).view(bs, seq_len, self.n_heads, self.rope_head_dim)
        else:
            q_nope = self.wq_up(x).view(bs, seq_len, self.n_heads, self.nope_head_dim)
            q_rope = self.wq_rope(x).view(bs, seq_len, self.n_heads, self.rope_head_dim)

        kv_compressed = self.kv_norm(self.wkv_down(x))

        kv_up = self.wkv_up(kv_compressed).view(bs, seq_len, self.n_heads, self.nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv_up, [self.nope_head_dim, self.v_head_dim], dim=-1)

        k_rope_shared = self.wk_rope(x).view(bs, seq_len, 1, self.rope_head_dim)

        if kv_cache is not None:
            k_rope_for_cache = k_rope_shared.squeeze(2)
            kv_cache.update(layer_idx, start_pos, kv_compressed, k_rope_for_cache)

        q_rope = q_rope.transpose(1, 2)
        q_rope = rope.apply_rotary_emb(q_rope)

        k_rope_shared = k_rope_shared.transpose(1, 2)
        k_rope_shared = rope.apply_rotary_emb(k_rope_shared)

        k_rope = k_rope_shared.expand(-1, self.n_heads, -1, -1)

        q_nope = q_nope.transpose(1, 2)
        k_nope = k_nope.transpose(1, 2)
        v = v.transpose(1, 2)

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.nope_head_dim + self.rope_head_dim)

        if seq_len > 1:
            mask = self.mask[:, :, :seq_len, :seq_len]
            scores = scores.masked_fill(~mask, -1e9)

        probs = F.softmax(scores.float(), dim=-1).type_as(x)
        output = torch.matmul(probs, v)
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.wo(output)

    def _forward_inference_optimized(self,
                                     x: torch.Tensor,
                                     rope: RoPE,
                                     layer_idx: int,
                                     kv_cache: LatentKVCache,
                                     start_pos: int = 0) -> torch.Tensor:
        bs, seq_len, _ = x.size()
        assert seq_len == 1

        if self.q_lora_rank > 0:
            q_compressed = self.q_norm(self.wq_down(x))
            q_nope = self.wq_up(q_compressed).view(bs, seq_len, self.n_heads, self.nope_head_dim)
            q_rope = self.wq_rope(q_compressed).view(bs, seq_len, self.n_heads, self.rope_head_dim)
        else:
            q_nope = self.wq_up(x).view(bs, seq_len, self.n_heads, self.nope_head_dim)
            q_rope = self.wq_rope(x).view(bs, seq_len, self.n_heads, self.rope_head_dim)

        kv_compressed = self.kv_norm(self.wkv_down(x))

        k_rope_shared = self.wk_rope(x).view(bs, seq_len, self.rope_head_dim)

        positions = torch.arange(start_pos, start_pos + seq_len, device=x.device, dtype=torch.long)
        positions = positions.unsqueeze(0).expand(bs, -1).flatten()

        q_rope_flat = q_rope.view(bs * seq_len, self.n_heads, self.rope_head_dim)
        q_rope_out = rope.apply_rotary_emb_paged(q_rope_flat, positions)
        q_rope = q_rope_out.view(bs, seq_len, self.n_heads, self.rope_head_dim).transpose(1, 2)

        k_rope_flat = k_rope_shared.view(bs * seq_len, 1, self.rope_head_dim)
        k_rope_out = rope.apply_rotary_emb_paged(k_rope_flat, positions)
        k_rope_shared = k_rope_out.view(bs, seq_len, self.rope_head_dim)

        full_c_kv, full_k_rope = kv_cache.update(layer_idx, start_pos, kv_compressed, k_rope_shared)

        k_rope_hist_heads = full_k_rope.unsqueeze(1)
        score_pe = torch.matmul(q_rope, k_rope_hist_heads.transpose(-2, -1))

        w_up_weight = None
        if hasattr(self.wkv_up, '_packed_params'):
            if hasattr(self.wkv_up._packed_params, 'unpack'):
                w_up_weight, _ = self.wkv_up._packed_params.unpack()

            else:
                try:
                    w_up_weight, _ = torch.ops.quantized.linear_unpack(self.wkv_up._packed_params)

                except Exception as e:
                    raise RuntimeError(f"Failed to unpacked quantized weight {e}")

        else:
            w_up_weight = self.wkv_up.weight

        head_dim_total = self.nope_head_dim + self.v_head_dim
        w_up_weight = w_up_weight.view(self.n_heads, head_dim_total, self.kv_lora_rank)

        w_uk = w_up_weight[:, :self.nope_head_dim, :]
        w_uv = w_up_weight[:, self.nope_head_dim:, :]

        q_nope_heads = q_nope.transpose(1, 2)

        q_absorbed = torch.einsum('b h t d, h d r -> b h t r', q_nope_heads, w_uk)

        score_nope = torch.matmul(q_absorbed, full_c_kv.transpose(1, 2).unsqueeze(1))

        scores = (score_nope + score_pe) / math.sqrt(self.nope_head_dim + self.rope_head_dim)

        probs = F.softmax(scores.float(), dim=-1).type_as(x)

        latent_output = torch.matmul(probs, full_c_kv.unsqueeze(1))

        output = torch.einsum('b h t r, h v r -> b h t v', latent_output, w_uv)

        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        return self.wo(output)



























