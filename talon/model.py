"""
The decoder-only Transformer architecture that powers Talon.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTModelConfig:
    vocab_size: int
    block_size: int
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTModelConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # The causal mask prevents each token from attending to future tokens.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_dim = x.shape
        qkv = self.qkv(x)
        query, key, value = qkv.split(embedding_dim, dim=2)

        query = query.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.n_head, self.head_dim).transpose(1, 2)

        # Standard scaled dot-product attention with masking for autoregressive generation.
        attention = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = attention.masked_fill(self.mask[:, :, :sequence_length, :sequence_length] == 0, float("-inf"))
        attention = F.softmax(attention, dim=-1)
        attention = self.attn_dropout(attention)

        output = attention @ value
        output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_dim)
        output = self.resid_dropout(self.proj(output))
        return output


class FeedForward(nn.Module):
    def __init__(self, config: GPTModelConfig) -> None:
        super().__init__()
        inner_dim = 4 * config.n_embd
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connections keep gradients stable as the stack gets deeper.
        x = x + self.attn(self.ln_1(x))
        x = x + self.ff(self.ln_2(x))
        return x


class TalonGPT(nn.Module):
    def __init__(self, config: GPTModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying reduces parameters and usually helps language models generalize.
        self.token_embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, sequence_length = idx.shape
        if sequence_length > self.config.block_size:
            raise ValueError("Sequence length exceeds model block size")

        positions = torch.arange(sequence_length, device=idx.device)
        token_embeddings = self.token_embedding(idx)
        # Positional embeddings tell the model where each token appears in the sequence.
        position_embeddings = self.position_embedding(positions)[None, :, :]
        x = self.dropout(token_embeddings + position_embeddings)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(batch_size * sequence_length, -1), targets.view(batch_size * sequence_length))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            # Only the most recent window matters because the model has a fixed context size.
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            logits = logits / max(temperature, 1e-5)

            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")

            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx
