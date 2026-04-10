"""
A small Tkinter desktop window for chatting with Talon.
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path
import sys
from tkinter import BooleanVar, END, BOTH, LEFT, RIGHT, X, Y, Frame, Label, StringVar, Text, Tk, TclError
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is not installed. Run '.\\.venv\\Scripts\\pip.exe install -r requirements.txt' first."
    ) from exc

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from talon.chat import build_chat_answer
    from talon.inference import load_checkpoint
    from talon.learning import learn_from_command, learn_from_correction, render_learning_acknowledgement
else:
    from .chat import build_chat_answer
    from .inference import load_checkpoint
    from .learning import learn_from_command, learn_from_correction, render_learning_acknowledgement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small Talon desktop chat window.")
    parser.add_argument("--checkpoint-dir", default="artifacts/talon-base")
    parser.add_argument("--data-dir", default="data/knowledge")
    parser.add_argument("--device", default="cuda(gpu)" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--web", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-sources", action="store_false")
    parser.add_argument("--local-top-k", type=int, default=3)
    parser.add_argument("--web-search-limit", type=int, default=5)
    parser.add_argument("--web-fetch-limit", type=int, default=2)
    parser.add_argument("--web-top-k", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--strict-sources", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--history-turns", type=int, default=3)
    parser.add_argument("--answer-length", choices=("short", "medium", "long"), default="medium")
    parser.add_argument("--tone", choices=("casual", "balanced", "formal"), default="balanced")
    parser.add_argument("--style", choices=("balanced", "logical", "creative"), default="balanced")
    parser.add_argument("--explanation-level", choices=("low", "medium", "high", "advanced"), default="medium")
    parser.add_argument("--extractive-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


class TalonWindow:
    def __init__(self, root: Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.config, self.tokenizer, self.model = load_checkpoint(args.checkpoint_dir, args.device)
        self.history: list[tuple[str, str]] = []
        self.active_device = str(next(self.model.parameters()).device)
        self.busy = False

        self.web_var = BooleanVar(value=args.web)
        self.sources_var = BooleanVar(value=args.show_sources)
        self.strict_sources_var = BooleanVar(value=args.strict_sources)
        self.answer_length_var = StringVar(value=args.answer_length)
        self.tone_var = StringVar(value=args.tone)
        self.style_var = StringVar(value=args.style)
        self.explanation_level_var = StringVar(value=args.explanation_level)
        self.status_var = StringVar(
            value=f"Ready on {self.active_device} | Checkpoint: {args.checkpoint_dir}"
        )

        self.root.title("Talon")
        self.root.geometry("1600x900")
        self.root.minsize(600, 420)

        self._build_layout()
        self._append_system(
            "Talon is ready. Type a message below.\n"
            f"Device: {self.active_device}\n"
            f"Web retrieval: {'on' if self.web_var.get() else 'off'} | "
            f"Sources: {'on' if self.sources_var.get() else 'off'} | "
            f"Strict sources: {'on' if self.strict_sources_var.get() else 'off'} | "
            f"Length: {self.answer_length_var.get()} | Tone: {self.tone_var.get()} | "
            f"Style: {self.style_var.get()} | Explain: {self.explanation_level_var.get()}\n"
            "Explicit corrections like 'Actually, the sky is blue.' are learned automatically."
        )

    def _build_layout(self) -> None:
        main_frame = Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=BOTH, expand=True)

        top_bar = Frame(main_frame)
        top_bar.pack(fill=X, pady=(0, 8))

        ttk.Checkbutton(top_bar, text="Web Retrieval", variable=self.web_var).pack(side=LEFT, padx=(0, 12))
        ttk.Checkbutton(top_bar, text="Show Sources", variable=self.sources_var).pack(side=LEFT, padx=(0, 12))
        ttk.Checkbutton(
            top_bar,
            text="Strict Sources",
            variable=self.strict_sources_var,
            command=self._on_strict_sources_changed,
        ).pack(side=LEFT, padx=(0, 12))
        Label(top_bar, text="Length").pack(side=LEFT, padx=(4, 6))
        length_box = ttk.Combobox(
            top_bar,
            textvariable=self.answer_length_var,
            values=("short", "medium", "long"),
            state="readonly",
            width=8,
        )
        length_box.pack(side=LEFT, padx=(0, 12))
        length_box.bind("<<ComboboxSelected>>", self._on_length_changed)
        Label(top_bar, text="Tone").pack(side=LEFT, padx=(4, 6))
        tone_box = ttk.Combobox(
            top_bar,
            textvariable=self.tone_var,
            values=("casual", "balanced", "formal"),
            state="readonly",
            width=9,
        )
        tone_box.pack(side=LEFT, padx=(0, 12))
        tone_box.bind("<<ComboboxSelected>>", self._on_tone_changed)
        Label(top_bar, text="Style").pack(side=LEFT, padx=(4, 6))
        style_box = ttk.Combobox(
            top_bar,
            textvariable=self.style_var,
            values=("balanced", "logical", "creative"),
            state="readonly",
            width=10,
        )
        style_box.pack(side=LEFT, padx=(0, 12))
        style_box.bind("<<ComboboxSelected>>", self._on_style_changed)
        Label(top_bar, text="Explanation").pack(side=LEFT, padx=(4, 6))
        explain_box = ttk.Combobox(
            top_bar,
            textvariable=self.explanation_level_var,
            values=("low", "medium", "high", "advanced"),
            state="readonly",
            width=11,
        )
        explain_box.pack(side=LEFT, padx=(0, 12))
        explain_box.bind("<<ComboboxSelected>>", self._on_explanation_level_changed)
        ttk.Button(top_bar, text="Clear Chat", command=self.clear_chat).pack(side=RIGHT)

        self.output = ScrolledText(main_frame, wrap="word", state="disabled", height=24)
        self.output.pack(fill=BOTH, expand=True)

        input_frame = Frame(main_frame)
        input_frame.pack(fill=X, pady=(8, 0))

        self.input_box = Text(input_frame, height=4, wrap="word")
        self.input_box.pack(side=LEFT, fill=BOTH, expand=True)
        self.input_box.bind("<Control-Return>", self._on_send_shortcut)
        self.input_box.bind("<Control-KP_Enter>", self._on_send_shortcut)

        button_frame = Frame(input_frame, padx=8)
        button_frame.pack(side=RIGHT, fill=Y)
        self.send_button = ttk.Button(button_frame, text="Send", command=self.send_message)
        self.send_button.pack(fill=X)

        help_label = Label(
            main_frame,
            text="Ctrl+Enter to send. Use /learn The sky is blue. to save a fact directly.",
            anchor="w",
        )
        help_label.pack(fill=X, pady=(6, 2))

        status_label = Label(main_frame, textvariable=self.status_var, anchor="w")
        status_label.pack(fill=X)

    def _append_block(self, prefix: str, message: str) -> None:
        self.output.configure(state="normal")
        # All chat output goes through one helper so the transcript stays formatted consistently.
        self.output.insert(END, f"{prefix}\n{message.strip()}\n\n")
        self.output.see(END)
        self.output.configure(state="disabled")

    def _append_user(self, message: str) -> None:
        self._append_block("You>", message)

    def _append_talon(self, message: str) -> None:
        self._append_block("Talon>", message)

    def _append_sources(self, message: str) -> None:
        self._append_block("Sources>", message)

    def _append_system(self, message: str) -> None:
        self._append_block("System>", message)

    def _on_send_shortcut(self, event) -> str:
        self.send_message()
        return "break"

    def clear_chat(self) -> None:
        self.history.clear()
        self.output.configure(state="normal")
        self.output.delete("1.0", END)
        self.output.configure(state="disabled")
        self._append_system("Chat cleared.")

    def _on_length_changed(self, event=None) -> None:
        self.args.answer_length = self.answer_length_var.get()
        self._append_system(f"Answer length set to {self.args.answer_length}.")

    def _on_tone_changed(self, event=None) -> None:
        self.args.tone = self.tone_var.get()
        self._append_system(f"Tone set to {self.args.tone}.")

    def _on_style_changed(self, event=None) -> None:
        self.args.style = self.style_var.get()
        self._append_system(f"Style set to {self.args.style}.")

    def _on_explanation_level_changed(self, event=None) -> None:
        self.args.explanation_level = self.explanation_level_var.get()
        self._append_system(f"Explanation level set to {self.args.explanation_level}.")

    def _on_strict_sources_changed(self) -> None:
        self.args.strict_sources = self.strict_sources_var.get()
        mode = "enabled" if self.args.strict_sources else "disabled"
        self._append_system(f"Strict live source filtering {mode}.")

    def send_message(self) -> None:
        if self.busy:
            return

        message = self.input_box.get("1.0", END).strip()
        if not message:
            return

        self.input_box.delete("1.0", END)
        self._append_user(message)

        if message.lower().startswith("/learn "):
            learned_fact = learn_from_command(message[7:].strip(), self.args.data_dir)
            if learned_fact is None:
                answer = "Please give /learn a simple fact like '/learn The sky is blue.'"
            else:
                answer = render_learning_acknowledgement(learned_fact)
            self.history.append((message, answer))
            self._append_talon(answer)
            return

        if self.history:
            learned_fact = learn_from_correction(message, self.args.data_dir)
            if learned_fact is not None:
                answer = render_learning_acknowledgement(learned_fact)
                self.history.append((message, answer))
                self._append_talon(answer)
                return

        self._set_busy(True, "Talon is thinking...")

        # Run retrieval and generation in a background thread so the window stays responsive.
        worker = threading.Thread(target=self._answer_in_background, args=(message,), daemon=True)
        worker.start()

    def _answer_in_background(self, message: str) -> None:
        try:
            answer, sources = build_chat_answer(
                question=message,
                history=self.history,
                use_web=self.web_var.get(),
                show_sources=self.sources_var.get(),
                args=self.args,
                block_size=self.config.block_size,
                tokenizer=self.tokenizer,
                model=self.model,
            )
            self.root.after(0, self._finish_answer, message, answer, sources)
        except Exception as exc:
            self.root.after(0, self._show_error, str(exc))

    def _finish_answer(self, message: str, answer: str, sources: str) -> None:
        self.history.append((message, answer))
        self._append_talon(answer)
        if self.sources_var.get() and sources:
            self._append_sources(sources)
        # Tk widgets must be updated on the main thread, so background work hands back here.
        self._set_busy(False, f"Ready on {self.active_device}")

    def _show_error(self, error_message: str) -> None:
        self._append_system(f"Error: {error_message}")
        self._set_busy(False, "Ready after error")

    def _set_busy(self, is_busy: bool, status: str) -> None:
        self.busy = is_busy
        state = "disabled" if is_busy else "normal"
        self.send_button.configure(state=state)
        self.input_box.configure(state=state)
        self.status_var.set(status)


def main() -> None:
    args = parse_args()
    root = Tk()
    app = TalonWindow(root, args)
    try:
        root.mainloop()
    except TclError as exc:
        raise SystemExit(f"Failed to start the Talon window: {exc}") from exc


if __name__ == "__main__":
    main()
