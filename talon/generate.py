"""
Simple CLI for sampling text from a trained Talon checkpoint.
"""

from __future__ import annotations

import argparse

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is not installed. Run '.\\.venv\\Scripts\\pip.exe install -r requirements.txt' first."
    ) from exc

from .inference import generate_text, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained Talon checkpoint.")
    parser.add_argument("--checkpoint-dir", default="artifacts/talon-base")
    parser.add_argument("--prompt", default="# Talon\n\n")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, tokenizer, model = load_checkpoint(args.checkpoint_dir, args.device)
    # include_prompt=True makes the output easier to read directly in the terminal.
    print(
        generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            include_prompt=True,
        )
    )


if __name__ == "__main__":
    main()
