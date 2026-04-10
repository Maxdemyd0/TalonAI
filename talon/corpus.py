"""
Helpers for loading Talon's markdown knowledge files from disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MarkdownDocument:
    path: Path
    text: str


def load_markdown_documents(directory: str | Path) -> list[MarkdownDocument]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Markdown directory not found: {root}")

    documents: list[MarkdownDocument] = []
    # Talon learns from every markdown file under the knowledge directory tree.
    for path in sorted(root.rglob("*.md")):
        if not path.is_file():
            continue

        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        documents.append(MarkdownDocument(path=path, text=text))

    if not documents:
        raise ValueError(f"No markdown files with content found under {root}")

    return documents
