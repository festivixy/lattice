"""Decoder-only transformer built from attention.Block, with tied embeddings."""
import math

import torch
from torch import nn

from .attention import AttentionConfig, Block, precompute_freqs


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, n_layers: int, cfg: AttentionConfig, ffn_mult: float = 2.67):
        super().__init__()
        self.cfg = cfg
        hidden = int(ffn_mult * cfg.d_model)
        hidden = 256 * ((hidden + 255) // 256)  # round up for tensor-core friendly shapes
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(Block(cfg, hidden) for _ in range(n_layers))
        self.norm = nn.RMSNorm(cfg.d_model, eps=1e-5)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        head_dim = cfg.d_model // cfg.n_heads
        self.register_buffer(
            "freqs", precompute_freqs(head_dim, cfg.max_seq_len, cfg.rope_theta), persistent=False
        )
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        seq = tokens.size(1)
        h = self.embed(tokens)
        freqs = self.freqs[start_pos:start_pos + seq]
        for layer in self.layers:
            h = layer(h, freqs, start_pos=start_pos)
        return self.head(self.norm(h)).float()

    def num_params(self, exclude_embeddings: bool = True) -> int:
        total = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            total -= self.embed.weight.numel()
        return total

    @torch.no_grad()
    def generate(self, prompt: torch.Tensor, max_new: int, temperature: float = 0.8, top_p: float = 0.95):
        tokens = prompt
        for _ in range(max_new):
            logits = self(tokens[:, -self.cfg.max_seq_len:])[:, -1] / max(temperature, 1e-5)
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, order = probs.sort(dim=-1, descending=True)
            keep = sorted_probs.cumsum(dim=-1) - sorted_probs < top_p
            sorted_probs = sorted_probs * keep
            choice = torch.multinomial(sorted_probs / sorted_probs.sum(-1, keepdim=True), 1)
            tokens = torch.cat([tokens, order.gather(-1, choice)], dim=1)
        return tokens
