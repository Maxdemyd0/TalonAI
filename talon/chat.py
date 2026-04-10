"""
Interactive terminal chat loop for Talon.
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
from .learning import learn_from_command, learn_from_correction, render_learning_acknowledgement
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


def normalize_question_key(question: str) -> str:
    return " ".join(question.lower().split())


def count_previous_question_asks(question: str, history: list[tuple[str, str]]) -> int:
    normalized_question = normalize_question_key(question)
    return sum(1 for previous_question, _ in history if normalize_question_key(previous_question) == normalized_question)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive Talon chat with local and optional web retrieval.")
    parser.add_argument("--checkpoint-dir", default="artifacts/talon-base")
    parser.add_argument("--data-dir", default="data/knowledge")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--web",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start chat with live web retrieval enabled.",
    )
    parser.add_argument("--show-sources", action="store_true", help="Print source list after each answer.")
    parser.add_argument("--local-top-k", type=int, default=3)
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
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--history-turns", type=int, default=3)
    parser.add_argument(
        "--answer-length",
        choices=("short", "medium", "long"),
        default="medium",
        help="Control how brief or detailed Talon's answers should be.",
    )
    parser.add_argument(
        "--tone",
        choices=("casual", "balanced", "formal"),
        default="balanced",
        help="Control the tone of Talon's answers.",
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
        help="Always use retrieval summaries instead of generation.",
    )
    return parser.parse_args()


def render_history(history: list[tuple[str, str]], max_turns: int) -> str:
    if not history:
        return ""
    recent_history = history[-max_turns:]
    # Only keep the most recent turns so the prompt stays within Talon's context window.
    lines = ["## Recent Conversation"]
    for user_message, assistant_message in recent_history:
        lines.append(f"User: {user_message}")
        lines.append(f"Talon: {assistant_message}")
        lines.append("")
    return "\n".join(lines).strip()


def build_chat_prompt(
    question: str,
    context: str,
    history: list[tuple[str, str]],
    history_turns: int,
    answer_length: str,
    tone: str,
    style: str,
    explanation_level: str,
    variation_index: int,
) -> str:
    parts = [
        "# Talon",
        "",
        "## Instructions",
        "Answer the user's latest message using the retrieved context when it helps.",
        "Be concise, practical, and honest about uncertainty.",
        f"Target answer length: {answer_length}.",
        f"Target tone: {tone}.",
        f"Target style: {style}.",
        f"Target explanation level: {explanation_level}.",
        "If this question has been asked before, vary the wording while keeping the facts consistent."
        if variation_index > 0
        else "Keep the wording clear and natural.",
        "",
    ]
    history_block = render_history(history, max_turns=history_turns)
    if history_block:
        parts.extend([history_block, ""])
    parts.extend(["## Latest User Message", question, ""])
    if context:
        parts.extend(["## Retrieved Context", context, ""])
    parts.extend(["## Answer", "Talon:"])
    return "\n".join(parts)


def compute_chat_context_budget(
    question: str,
    history: list[tuple[str, str]],
    history_turns: int,
    block_size: int,
    answer_length: str,
    tone: str,
    style: str,
    explanation_level: str,
    variation_index: int,
) -> int:
    base_prompt = build_chat_prompt(
        question=question,
        context="",
        history=history,
        history_turns=history_turns,
        answer_length=answer_length,
        tone=tone,
        style=style,
        explanation_level=explanation_level,
        variation_index=variation_index,
    )
    return max(0, block_size - len(base_prompt) - 1)


def build_chat_answer(
    *,
    question: str,
    history: list[tuple[str, str]],
    use_web: bool,
    show_sources: bool,
    args: argparse.Namespace,
    block_size: int,
    tokenizer,
    model,
) -> tuple[str, str]:
    variation_index = count_previous_question_asks(question, history)
    local_results = search_local(query=question, data_dir=args.data_dir, top_k=args.local_top_k)
    web_results = []
    web_warning = ""
    if use_web:
        try:
            web_results = search_web(
                query=question,
                search_limit=args.web_search_limit,
                fetch_limit=args.web_fetch_limit,
                top_k=args.web_top_k,
                timeout=args.timeout,
                strict_sources=args.strict_sources,
            )
        except Exception as exc:
            web_warning = f"Web retrieval unavailable, using local knowledge only: {exc}"
    all_results = [*local_results, *web_results]

    special_case_answer = (
        build_special_case_answer(
            query=question,
            passages=all_results,
            answer_length=args.answer_length,
            tone=args.tone,
            style=args.style,
            explanation_level=args.explanation_level,
            variation_index=variation_index,
        )
        if all_results
        else None
    )
    use_extractive = args.extractive_only or block_size < 512
    if not all_results:
        answer = NO_RELEVANT_INFORMATION_MESSAGE
    elif special_case_answer:
        answer = special_case_answer
    elif use_extractive and all_results:
        # The extractive path is safer for Talon's current small checkpoint.
        answer = build_extractive_answer(
            query=question,
            passages=all_results,
            answer_length=args.answer_length,
            tone=args.tone,
            style=args.style,
            explanation_level=args.explanation_level,
            variation_index=variation_index,
        )
    else:
        context_budget = compute_chat_context_budget(
            question=question,
            history=history,
            history_turns=args.history_turns,
            block_size=block_size,
            answer_length=args.answer_length,
            tone=args.tone,
            style=args.style,
            explanation_level=args.explanation_level,
            variation_index=variation_index,
        )
        context = render_context(all_results, max_chars=context_budget)
        prompt = build_chat_prompt(
            question=question,
            context=context,
            history=history,
            history_turns=args.history_turns,
            answer_length=args.answer_length,
            tone=args.tone,
            style=args.style,
            explanation_level=args.explanation_level,
            variation_index=variation_index,
        )
        answer = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=args.device,
            max_new_tokens=max_new_tokens_for_length(args.answer_length, args.max_new_tokens),
            temperature=args.temperature,
            top_k=args.top_k,
        ).strip()

    if show_sources:
        source_parts: list[str] = []
        if web_warning:
            source_parts.append(web_warning)
        source_parts.append(render_sources(all_results))
        sources_text = "\n".join(part for part in source_parts if part).strip()
    else:
        sources_text = ""
    return answer.strip() or NO_RELEVANT_INFORMATION_MESSAGE, sources_text


def print_help() -> None:
    print("Commands:")
    print("  /help           Show chat commands")
    print("  /learn <fact>   Save a fact directly into Talon's markdown knowledge")
    print("  /length <mode>  Set answer length to short, medium, or long")
    print("  /tone <mode>    Set tone to casual, balanced, or formal")
    print("  /style <mode>   Set style to balanced, logical, or creative")
    print("  /explain <lvl>  Set explanation level to low, medium, high, or advanced")
    print("  /strict on      Only use stricter allowed live web sources")
    print("  /strict off     Allow the normal live web source pool")
    print("  /quit           Exit chat")
    print("  /web on         Enable live web retrieval")
    print("  /web off        Disable live web retrieval")
    print("  /sources on     Show sources after each answer")
    print("  /sources off    Hide sources after each answer")


def main() -> None:
    args = parse_args()
    config, tokenizer, model = load_checkpoint(args.checkpoint_dir, args.device)
    active_device = str(next(model.parameters()).device)
    use_web = args.web
    show_sources = args.show_sources
    history: list[tuple[str, str]] = []

    print(f"Talon chat ready on device: {active_device}")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(
        f"Web retrieval: {'on' if use_web else 'off'} | Sources: {'on' if show_sources else 'off'} | "
        f"Strict sources: {'on' if args.strict_sources else 'off'} | Length: {args.answer_length} | "
        f"Tone: {args.tone} | Style: {args.style} | Explain: {args.explanation_level}"
    )
    print("Explicit corrections like 'Actually, the sky is blue.' are learned automatically.")
    print("Type /help for commands.")

    while True:
        try:
            user_input = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input == "/quit":
            print("Bye.")
            break
        if user_input == "/help":
            print_help()
            continue
        if user_input.lower().startswith("/length "):
            requested_length = user_input[8:].strip().lower()
            if requested_length not in {"short", "medium", "long"}:
                print("Talon> Please use /length short, /length medium, or /length long.")
            else:
                args.answer_length = requested_length
                print(f"Talon> Answer length set to {requested_length}.")
            continue
        if user_input.lower().startswith("/tone "):
            requested_tone = user_input[6:].strip().lower()
            if requested_tone not in {"casual", "balanced", "formal"}:
                print("Talon> Please use /tone casual, /tone balanced, or /tone formal.")
            else:
                args.tone = requested_tone
                print(f"Talon> Tone set to {requested_tone}.")
            continue
        if user_input.lower().startswith("/style "):
            requested_style = user_input[7:].strip().lower()
            if requested_style not in {"balanced", "logical", "creative"}:
                print("Talon> Please use /style balanced, /style logical, or /style creative.")
            else:
                args.style = requested_style
                print(f"Talon> Style set to {requested_style}.")
            continue
        if user_input.lower().startswith("/explain "):
            requested_level = user_input[9:].strip().lower()
            if requested_level not in {"low", "medium", "high", "advanced"}:
                print("Talon> Please use /explain low, /explain medium, /explain high, or /explain advanced.")
            else:
                args.explanation_level = requested_level
                print(f"Talon> Explanation level set to {requested_level}.")
            continue
        if user_input.lower().startswith("/learn "):
            learned_fact = learn_from_command(user_input[7:].strip(), args.data_dir)
            if learned_fact is None:
                answer = "Please give /learn a simple fact like '/learn The sky is blue.'"
            else:
                answer = render_learning_acknowledgement(learned_fact)
            print(f"Talon> {answer}")
            history.append((user_input, answer))
            continue
        if user_input == "/web on":
            use_web = True
            print("Live web retrieval enabled.")
            continue
        if user_input == "/web off":
            use_web = False
            print("Live web retrieval disabled.")
            continue
        if user_input == "/strict on":
            args.strict_sources = True
            print("Strict live source filtering enabled.")
            continue
        if user_input == "/strict off":
            args.strict_sources = False
            print("Strict live source filtering disabled.")
            continue
        if user_input == "/sources on":
            show_sources = True
            print("Source display enabled.")
            continue
        if user_input == "/sources off":
            show_sources = False
            print("Source display disabled.")
            continue

        if history:
            learned_fact = learn_from_correction(user_input, args.data_dir)
            if learned_fact is not None:
                # Corrections are stored immediately so the next question can retrieve them.
                answer = render_learning_acknowledgement(learned_fact)
                print(f"Talon> {answer}")
                history.append((user_input, answer))
                continue

        answer, sources_text = build_chat_answer(
            question=user_input,
            history=history,
            use_web=use_web,
            show_sources=show_sources,
            args=args,
            block_size=config.block_size,
            tokenizer=tokenizer,
            model=model,
        )
        print(f"Talon> {answer}")
        if show_sources and sources_text:
            print("Sources:")
            print(sources_text)
        history.append((user_input, answer))


if __name__ == "__main__":
    main()
