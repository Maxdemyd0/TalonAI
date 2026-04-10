"""
Single-question answer CLI that combines retrieval with optional generation.
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
from .retrieval import (
    NO_RELEVANT_INFORMATION_MESSAGE,
    build_extractive_answer,
    build_special_case_answer,
    render_context,
    render_sources,
    search_local,
    search_web,
)


def max_new_tokens_for_length(answer_length: str, default_value: int) -> int:
    return {
        "short": min(default_value, 96),
        "medium": default_value,
        "long": max(default_value, 320),
    }.get(answer_length, default_value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Answer a question with local and optional web retrieval.")
    parser.add_argument("--checkpoint-dir", default="artifacts/talon-base")
    parser.add_argument("--question", required=True)
    parser.add_argument("--data-dir", default="data/knowledge")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-top-k", type=int, default=3)
    parser.add_argument(
        "--web",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retrieve live web pages during answer time.",
    )
    parser.add_argument("--web-search-limit", type=int, default=5)
    parser.add_argument("--web-fetch-limit", type=int, default=2)
    parser.add_argument("--web-top-k", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument(
        "--strict-sources",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a stricter allowlist for live web sources.",
    )
    parser.add_argument("--show-sources", action="store_true")
    parser.add_argument(
        "--answer-length",
        choices=("short", "medium", "long"),
        default="medium",
        help="Control how brief or detailed Talon's answer should be.",
    )
    parser.add_argument(
        "--tone",
        choices=("casual", "balanced", "formal"),
        default="balanced",
        help="Control the tone of Talon's answer.",
    )
    parser.add_argument(
        "--style",
        choices=("balanced", "logical", "creative"),
        default="balanced",
        help="Control whether Talon sounds more balanced, logical, or creative.",
    )
    parser.add_argument(
        "--explanation-level",
        choices=("low", "medium", "high", "advanced"),
        default="medium",
        help="Control how simple or advanced Talon's explanations should be.",
    )
    parser.add_argument(
        "--extractive-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return a readable retrieval-based answer without generation.",
    )
    return parser.parse_args()


def build_answer_prompt(
    question: str,
    context: str,
    answer_length: str,
    tone: str,
    style: str,
    explanation_level: str,
) -> str:
    parts = [
        "# Talon",
        "",
        "## Instructions",
        "Answer the user's question using the context when it helps.",
        "Be concise, practical, and honest about uncertainty.",
        f"Target answer length: {answer_length}.",
        f"Target tone: {tone}.",
        f"Target style: {style}.",
        f"Target explanation level: {explanation_level}.",
        "",
        "## Question",
        question,
        "",
    ]
    if context:
        # Retrieved snippets are appended as plain text context for the model.
        parts.extend(
            [
                "## Context",
                context,
                "",
            ]
        )
    parts.extend(
        [
            "## Answer",
            "Talon:",
        ]
    )
    return "\n".join(parts)


def compute_context_budget(question: str, block_size: int) -> int:
    base_prompt = build_answer_prompt(
        question=question,
        context="",
        answer_length="medium",
        tone="balanced",
        style="balanced",
        explanation_level="medium",
    )
    return max(0, block_size - len(base_prompt) - 1)


def main() -> None:
    args = parse_args()
    config, tokenizer, model = load_checkpoint(args.checkpoint_dir, args.device)

    local_results = search_local(query=args.question, data_dir=args.data_dir, top_k=args.local_top_k)
    web_results = []
    web_warning = ""
    if args.web:
        try:
            web_results = search_web(
                query=args.question,
                search_limit=args.web_search_limit,
                fetch_limit=args.web_fetch_limit,
                top_k=args.web_top_k,
                timeout=args.timeout,
                strict_sources=args.strict_sources,
            )
        except Exception as exc:
            # Retrieval should degrade gracefully instead of breaking the whole answer command.
            web_warning = f"Web retrieval unavailable, using local knowledge only: {exc}"

    all_results = [*local_results, *web_results]
    special_case_answer = (
        build_special_case_answer(
            query=args.question,
            passages=all_results,
            answer_length=args.answer_length,
            tone=args.tone,
            style=args.style,
            explanation_level=args.explanation_level,
        )
        if all_results
        else None
    )
    use_extractive = args.extractive_only or config.block_size < 512
    if not all_results:
        answer = NO_RELEVANT_INFORMATION_MESSAGE
    elif special_case_answer:
        # Special cases handle topics like "What is X?" with more natural summaries.
        answer = special_case_answer
    elif use_extractive and all_results:
        answer = build_extractive_answer(
            query=args.question,
            passages=all_results,
            answer_length=args.answer_length,
            tone=args.tone,
            style=args.style,
            explanation_level=args.explanation_level,
        )
    else:
        context_budget = compute_context_budget(question=args.question, block_size=config.block_size)
        context = render_context(all_results, max_chars=context_budget)
        prompt = build_answer_prompt(
            question=args.question,
            context=context,
            answer_length=args.answer_length,
            tone=args.tone,
            style=args.style,
            explanation_level=args.explanation_level,
        )
        answer = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=args.device,
            max_new_tokens=max_new_tokens_for_length(args.answer_length, args.max_new_tokens),
            temperature=args.temperature,
            top_k=args.top_k,
        )

    if args.show_sources:
        if web_warning:
            print(web_warning)
            print("")
        print("Sources:")
        print(render_sources(all_results))
        print("")

    print(answer)


if __name__ == "__main__":
    main()
