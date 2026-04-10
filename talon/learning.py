"""
Utilities for saving user-taught facts back into Talon's markdown knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


AUTO_CORRECTION_PATTERNS = (
    re.compile(r"^(?:actually[\s,]+)?(?P<subject>.+?)\s+is\s+actually\s+(?P<predicate>.+?)[.!?]*$", re.IGNORECASE),
    re.compile(r"^actually[\s,]+(?P<subject>.+?)\s+is\s+(?P<predicate>.+?)[.!?]*$", re.IGNORECASE),
    re.compile(r"^no[\s,]+(?P<subject>.+?)\s+is\s+(?:actually\s+)?(?P<predicate>.+?)[.!?]*$", re.IGNORECASE),
    re.compile(r"^correction[\s,:-]+(?P<subject>.+?)\s+is\s+(?:actually\s+)?(?P<predicate>.+?)[.!?]*$", re.IGNORECASE),
)
SIMPLE_FACT_PATTERN = re.compile(r"^(?P<subject>.+?)\s+is\s+(?P<predicate>.+?)[.!?]*$", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class LearnedFact:
    title: str
    statement: str
    path: Path
    was_new: bool


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def normalize_statement_text(text: str) -> str:
    cleaned = normalize_whitespace(text).strip(" .!?")
    if not cleaned:
        return ""
    return cleaned[0].upper() + cleaned[1:] + "."


def title_from_subject(subject: str) -> str:
    cleaned = normalize_whitespace(subject).strip(" .!?")
    return cleaned.title() if cleaned else "Learned Fact"


def slugify(text: str) -> str:
    parts = re.findall(r"[a-z0-9]+", text.lower())
    slug = "-".join(parts[:8])
    return slug or "learned-fact"


def extract_fact_from_statement(message: str) -> tuple[str, str] | None:
    stripped = normalize_whitespace(message)
    if not stripped or stripped.endswith("?"):
        return None

    match = SIMPLE_FACT_PATTERN.match(stripped)
    if not match:
        return None

    subject = normalize_whitespace(match.group("subject")).strip(" .!?")
    predicate = normalize_whitespace(match.group("predicate")).strip(" .!?")
    if not subject or not predicate:
        return None

    statement = normalize_statement_text(f"{subject} is {predicate}")
    title = title_from_subject(subject)
    return title, statement


def extract_correction_fact(message: str) -> tuple[str, str] | None:
    stripped = normalize_whitespace(message)
    if not stripped or stripped.endswith("?"):
        return None

    for pattern in AUTO_CORRECTION_PATTERNS:
        match = pattern.match(stripped)
        if not match:
            continue
        subject = normalize_whitespace(match.group("subject")).strip(" .!?")
        predicate = normalize_whitespace(match.group("predicate")).strip(" .!?")
        if not subject or not predicate:
            return None
        statement = normalize_statement_text(f"{subject} is {predicate}")
        title = title_from_subject(subject)
        return title, statement
    return None


def save_learned_fact(data_dir: str | Path, title: str, statement: str) -> LearnedFact:
    knowledge_dir = Path(data_dir) / "user_facts"
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    path = knowledge_dir / f"{slugify(title)}.md"
    normalized_statement = normalize_statement_text(statement)
    block = f"# {title}\n\n## Learned Facts\n\n- {normalized_statement}\n"

    if not path.exists():
        # New subjects get their own markdown file so retrieval can find them immediately.
        path.write_text(block, encoding="utf-8")
        return LearnedFact(title=title, statement=normalized_statement, path=path, was_new=True)

    existing_text = path.read_text(encoding="utf-8")
    existing_lines = {normalize_statement_text(line[2:]) for line in existing_text.splitlines() if line.startswith("- ")}
    if normalized_statement not in existing_lines:
        # Existing subjects keep accumulating bullet-point facts in one place.
        updated_text = existing_text.rstrip() + f"\n- {normalized_statement}\n"
        path.write_text(updated_text, encoding="utf-8")
        return LearnedFact(title=title, statement=normalized_statement, path=path, was_new=True)

    return LearnedFact(title=title, statement=normalized_statement, path=path, was_new=False)


def learn_from_correction(message: str, data_dir: str | Path) -> LearnedFact | None:
    extracted = extract_correction_fact(message)
    if not extracted:
        return None
    title, statement = extracted
    return save_learned_fact(data_dir=data_dir, title=title, statement=statement)


def learn_from_command(command_text: str, data_dir: str | Path) -> LearnedFact | None:
    extracted = extract_fact_from_statement(command_text)
    if not extracted:
        return None
    title, statement = extracted
    return save_learned_fact(data_dir=data_dir, title=title, statement=statement)


def render_learning_acknowledgement(learned_fact: LearnedFact) -> str:
    if learned_fact.was_new:
        return f"Thanks, I saved that fact: {learned_fact.statement}"
    return f"I already had that fact saved: {learned_fact.statement}"
