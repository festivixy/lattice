"""Grouped-query attention with rotary embeddings and a fused KV cache."""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class AttentionConfig:
    d_model: int = 4096
    n_heads: int = 32
    n_kv_heads: int = 8
    rope_theta: float = 500_000.0
    max_seq_len: int = 8192
    dropout: float = 0.0


def precompute_freqs(dim: int, seq_len: int, theta: float) -> torch.Tensor:
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    pairs = x.float().reshape(*x.shape[:-1], -1, 2)
    rotated = torch.view_as_complex(pairs) * freqs.unsqueeze(0).unsqueeze(2)
    return torch.view_as_real(rotated).flatten(-2).type_as(x)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    batch, seq, heads, dim = x.shape
    expanded = x[:, :, :, None, :].expand(batch, seq, heads, n_rep, dim)
    return expanded.reshape(batch, seq, heads * n_rep, dim)


class KVCache(nn.Module):
    def __init__(self, batch: int, seq_len: int, n_kv_heads: int, head_dim: int, dtype):
        super().__init__()
        shape = (batch, seq_len, n_kv_heads, head_dim)
        self.register_buffer("k", torch.zeros(shape, dtype=dtype), persistent=False)
        self.register_buffer("v", torch.zeros(shape, dtype=dtype), persistent=False)

    def update(self, start: int, k: torch.Tensor, v: torch.Tensor):
        end = start + k.size(1)
        self.k[:, start:end] = k
        self.v[:, start:end] = v
        return self.k[:, :end], self.v[:, :end]


class Attention(nn.Module):
    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        self.cache = None

    def forward(self, x, freqs, mask=None, start_pos: int = 0):
        batch, seq, _ = x.shape
        q = self.wq(x).view(batch, seq, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch, seq, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(batch, seq, self.n_kv_heads, self.head_dim)
        q, k = apply_rope(q, freqs), apply_rope(k, freqs)

        if self.cache is not None:
            k, v = self.cache.update(start_pos, k, v)

        k, v = repeat_kv(k, self.n_rep), repeat_kv(v, self.n_rep)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=mask is None and seq > 1,
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: AttentionConfig, hidden: int):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model, eps=1e-5)
        self.attn = Attention(cfg)
        self.ffn_norm = nn.RMSNorm(cfg.d_model, eps=1e-5)
        self.ffn = SwiGLU(cfg.d_model, hidden)

    def forward(self, x, freqs, mask=None, start_pos: int = 0):
        h = x + self.attn(self.attn_norm(x), freqs, mask, start_pos)
        return h + self.ffn(self.ffn_norm(h))
