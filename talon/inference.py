"""
Shared checkpoint-loading and text-generation helpers for Talon runtimes.
"""

from __future__ import annotations

from pathlib import Path

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is not installed. Run '.\\.venv\\Scripts\\pip.exe install -r requirements.txt' first."
    ) from exc

from .config import TalonConfig
from .model import GPTModelConfig, TalonGPT
from .tokenizer import CharTokenizer


def load_checkpoint(checkpoint_dir: str | Path, device: str) -> tuple[TalonConfig, CharTokenizer, TalonGPT]:
    checkpoint_path = Path(checkpoint_dir)
    config = TalonConfig.load_json(checkpoint_path / "config.json")
    tokenizer = CharTokenizer.load(checkpoint_path / "tokenizer.json")
    model_config = GPTModelConfig(
        vocab_size=config.vocab_size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_layer=config.n_layer,
        dropout=config.dropout,
    )
    model = TalonGPT(model_config).to(device)
    # map_location lets the same checkpoint load on CPU or GPU without conversion steps.
    checkpoint = torch.load(checkpoint_path / "model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return config, tokenizer, model


def trim_prompt_to_block_size(prompt: str, tokenizer: CharTokenizer, block_size: int) -> str:
    token_ids = tokenizer.encode(prompt)
    if len(token_ids) <= block_size:
        return prompt
    # Keep the newest part of the prompt because it is most relevant for generation.
    trimmed_ids = token_ids[-block_size:]
    return tokenizer.decode(trimmed_ids, skip_special_tokens=False)


def generate_text(
    model: TalonGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    include_prompt: bool = False,
) -> str:
    prompt = trim_prompt_to_block_size(prompt, tokenizer, model.config.block_size)
    prompt_tokens = tokenizer.encode(prompt)
    # Generation starts from the encoded prompt, then samples additional tokens.
    x = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    y = model.generate(
        idx=x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    output_tokens = y[0].tolist()
    if include_prompt:
        return tokenizer.decode(output_tokens)
    return tokenizer.decode(output_tokens[len(prompt_tokens) :])
