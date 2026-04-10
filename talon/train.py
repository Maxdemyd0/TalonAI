"""
Command-line training loop for building a Talon checkpoint from markdown.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import random

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is not installed. Run '.\\.venv\\Scripts\\pip.exe install -r requirements.txt' first."
    ) from exc

from .config import TalonConfig
from .corpus import load_markdown_documents
from .model import GPTModelConfig, TalonGPT
from .tokenizer import CharTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Talon markdown language model.")
    parser.add_argument("--data-dir", default="data/knowledge")
    parser.add_argument("--output-dir", default="artifacts/talon-base")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # Keep CUDA runs reproducible enough for debugging and comparison.
        torch.cuda.manual_seed_all(seed)


def build_token_stream(tokenizer: CharTokenizer, documents: list[str]) -> list[int]:
    token_ids: list[int] = []
    for document in documents:
        token_ids.extend(tokenizer.encode(document))
        # EOS markers give the model a boundary between markdown documents.
        token_ids.append(tokenizer.eos_id)
    return token_ids


def split_tokens(token_ids: list[int], train_split: float, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    minimum_window = block_size + 1
    if len(token_ids) < minimum_window * 2:
        raise ValueError(
            f"Need at least {minimum_window * 2} tokens for train/validation splits, found {len(token_ids)}. "
            "Add more markdown files or lower --block-size."
        )

    split_index = int(len(token_ids) * train_split)
    split_index = max(split_index, minimum_window)
    split_index = min(split_index, len(token_ids) - minimum_window)

    train_ids = torch.tensor(token_ids[:split_index], dtype=torch.long)
    val_ids = torch.tensor(token_ids[split_index:], dtype=torch.long)
    return train_ids, val_ids


def sample_batch(token_tensor: torch.Tensor, batch_size: int, block_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(token_tensor) - block_size - 1
    starts = torch.randint(0, max_start + 1, (batch_size,))
    # Inputs and targets are the same window shifted by one token.
    inputs = torch.stack([token_tensor[start : start + block_size] for start in starts])
    targets = torch.stack([token_tensor[start + 1 : start + block_size + 1] for start in starts])
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def estimate_loss(
    model: TalonGPT,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    batch_size: int,
    block_size: int,
    eval_batches: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    losses: dict[str, float] = {}
    for split_name, token_tensor in {"train": train_tokens, "val": val_tokens}.items():
        split_losses = torch.zeros(eval_batches)
        for step in range(eval_batches):
            xb, yb = sample_batch(token_tensor, batch_size, block_size, device)
            _, loss = model(xb, yb)
            split_losses[step] = loss.item()
        losses[split_name] = split_losses.mean().item()
    model.train()
    return losses


def main() -> None:
    args = parse_args()
    config = TalonConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        train_split=args.train_split,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        seed=args.seed,
        device=args.device,
    )

    set_seed(config.seed)

    documents = load_markdown_documents(config.data_dir)
    texts = [document.text for document in documents]
    tokenizer = CharTokenizer.fit(texts)
    token_ids = build_token_stream(tokenizer, texts)
    train_tokens, val_tokens = split_tokens(token_ids, config.train_split, config.block_size)
    config.vocab_size = tokenizer.vocab_size

    model_config = GPTModelConfig(
        vocab_size=config.vocab_size,
        block_size=config.block_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_layer=config.n_layer,
        dropout=config.dropout,
    )
    model = TalonGPT(model_config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(documents)} markdown files from {config.data_dir}")
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Training tokens: {len(train_tokens)} | Validation tokens: {len(val_tokens)}")
    print(f"Using device: {config.device}")

    for step in range(config.max_steps):
        if step % config.eval_interval == 0 or step == config.max_steps - 1:
            losses = estimate_loss(
                model=model,
                train_tokens=train_tokens,
                val_tokens=val_tokens,
                batch_size=config.batch_size,
                block_size=config.block_size,
                eval_batches=config.eval_batches,
                device=config.device,
            )
            print(f"step {step:04d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

        xb, yb = sample_batch(train_tokens, config.batch_size, config.block_size, config.device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save everything needed to reload the exact model later.
    checkpoint = {
        "config": asdict(config),
        "model_state_dict": model.state_dict(),
    }
    torch.save(checkpoint, output_dir / "model.pt")
    config.save_json(output_dir / "config.json")
    tokenizer.save(output_dir / "tokenizer.json")
    print(f"Saved Talon checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
