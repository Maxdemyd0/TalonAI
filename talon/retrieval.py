"""
Retrieval, ranking, and lightweight summary logic for Talon's answer path.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log
from pathlib import Path
import re

from .corpus import MarkdownDocument, load_markdown_documents
from .web import SearchResult, fetch_search_results, fetch_url_text, html_to_markdown, is_allowed_source


WORD_PATTERN = re.compile(r"[a-zA-Z0-9_]{2,}")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
USE_QUESTION_PATTERN = re.compile(r"what\s+is\s+(?P<subject>.+?)\s+used\s+for\??", re.IGNORECASE)
KNOW_ABOUT_PATTERN = re.compile(r"what\s+(?P<category>.+?)\s+do\s+you\s+know\s+about\??", re.IGNORECASE)
TELL_ABOUT_PATTERN = re.compile(r"(?:can\s+you\s+)?tell\s+me\s+about\s+(?P<subject>.+?)\??$", re.IGNORECASE)
WHAT_IS_SUBJECT_PATTERN = re.compile(r"what\s+(?:is|are)\s+(?P<subject>.+?)\??$", re.IGNORECASE)
CREDENTIAL_NOISE_PATTERN = re.compile(
    r"\b(professor|associate vice president|dean|graduate school|university|department)\b",
    re.IGNORECASE,
)
STOPWORDS = {
    "about",
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "know",
    "me",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "their",
    "tell",
    "this",
    "to",
    "use",
    "used",
    "user",
    "users",
    "was",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
    "you",
    "your",
}
REASON_KEYWORDS = {
    "availability": "a large library ecosystem",
    "community": "strong community support",
    "ecosystem": "a large library ecosystem",
    "libraries": "a large library ecosystem",
    "readability": "readable syntax",
    "readable": "readable syntax",
    "simple": "simple syntax",
    "simplicity": "simple syntax",
    "versatile": "its versatility across domains",
    "versatility": "its versatility across domains",
}
DISPLAY_NAME_OVERRIDES = {
    "cs2": "Counter-Strike 2",
    "fortnite": "Fortnite",
    "minecraft": "Minecraft",
    "roblox": "Roblox",
}
NO_RELEVANT_INFORMATION_MESSAGE = "Sorry, no relevant information was found."
ANSWER_LENGTH_SENTENCE_LIMITS = {
    "short": 1,
    "medium": 3,
    "long": 5,
}
ANSWER_LENGTH_PART_LIMITS = {
    "short": 1,
    "medium": 3,
    "long": 5,
}
TONE_CHOICES = {"casual", "balanced", "formal"}
STYLE_CHOICES = {"balanced", "logical", "creative"}
EXPLANATION_LEVEL_CHOICES = {"low", "medium", "high", "advanced"}


@dataclass(frozen=True)
class RetrievedPassage:
    source: str
    title: str
    text: str
    score: float
    origin: str


@dataclass(frozen=True)
class Passage:
    source: str
    title: str
    text: str
    origin: str


@dataclass(frozen=True)
class MarkdownSection:
    heading: str
    lines: list[str]


@dataclass
class SummaryPart:
    text: str | None = None
    heading: str | None = None
    bullets: list[str] | None = None


def tokenize_words(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_PATTERN.finditer(text)]


def normalize_term(term: str) -> str:
    if len(term) > 4 and term.endswith("ing"):
        return term[:-3]
    if len(term) > 3 and term.endswith("ed"):
        return term[:-2]
    if len(term) > 3 and term.endswith("es"):
        return term[:-2]
    if len(term) > 2 and term.endswith("s"):
        return term[:-1]
    return term


def normalize_phrase(text: str) -> str:
    terms = [normalize_term(token) for token in tokenize_words(text)]
    return " ".join(term for term in terms if term and term not in STOPWORDS)


def extract_query_terms(text: str) -> list[str]:
    terms: list[str] = []
    for raw_term in tokenize_words(text):
        normalized = normalize_term(raw_term)
        if normalized in STOPWORDS:
            continue
        terms.append(normalized)
    return terms


def normalized_counter(text: str) -> Counter[str]:
    counter: Counter[str] = Counter()
    for raw_term in tokenize_words(text):
        normalized = normalize_term(raw_term)
        if normalized in STOPWORDS:
            continue
        counter[normalized] += 1
    return counter


def split_markdown_into_passages(document: MarkdownDocument, max_chars: int = 420) -> list[Passage]:
    sections: list[Passage] = []
    current_title = document.path.stem.replace("_", " ").title()
    current_lines: list[str] = []

    def flush() -> None:
        if not current_lines:
            return
        text = "\n".join(line for line in current_lines if line.strip()).strip()
        current_lines.clear()
        if not text:
            return
        # Large markdown sections are chunked so retrieval can rank smaller, more focused passages.
        for index in range(0, len(text), max_chars):
            chunk = text[index : index + max_chars].strip()
            if chunk:
                sections.append(
                    Passage(
                        source=str(document.path),
                        title=current_title,
                        text=chunk,
                        origin="local",
                    )
                )

    for raw_line in document.text.splitlines():
        line = raw_line.strip()
        if line.startswith("#"):
            flush()
            current_title = line.lstrip("#").strip() or current_title
            continue
        if not line:
            flush()
            continue
        current_lines.append(line)

    flush()
    return sections


def build_local_passages(data_dir: str | Path) -> list[Passage]:
    passages: list[Passage] = []
    for document in load_markdown_documents(data_dir):
        passages.extend(split_markdown_into_passages(document))
    return passages


def is_useful_web_passage(passage: Passage) -> bool:
    word_count = len(tokenize_words(passage.text))
    if word_count < 20:
        return False
    if len(passage.text) < 140:
        return False
    if is_header_like(passage.text.strip(), passage.title):
        return False
    return True


def score_passages(query: str, passages: list[Passage], top_k: int) -> list[RetrievedPassage]:
    query_terms = extract_query_terms(query)
    if not query_terms or not passages:
        return []

    doc_frequencies: Counter[str] = Counter()
    passage_counters: list[Counter[str]] = []
    for passage in passages:
        counter = normalized_counter(f"{passage.title}\n{passage.text}")
        passage_counters.append(counter)
        doc_frequencies.update(counter.keys())

    total_docs = len(passages)
    results: list[RetrievedPassage] = []
    for passage, term_counts in zip(passages, passage_counters, strict=True):
        if not term_counts:
            continue
        score = 0.0
        overlap_count = 0
        for term in query_terms:
            tf = term_counts.get(term, 0)
            if not tf:
                continue
            overlap_count += 1
            # A simple TF-IDF style score works well enough for Talon's local markdown corpus.
            idf = log((1 + total_docs) / (1 + doc_frequencies[term])) + 1.0
            score += (1 + log(tf)) * idf

        if score <= 0 or overlap_count == 0:
            continue

        title_terms = set(extract_query_terms(passage.title))
        title_overlap = len(title_terms & set(query_terms))
        score += overlap_count * 1.5
        score += title_overlap * 3.0

        # Favor compact passages that are easier to fit into Talon's limited context window.
        score = score / (1 + (len(passage.text) / 500))
        results.append(
            RetrievedPassage(
                source=passage.source,
                title=passage.title,
                text=passage.text,
                score=score,
                origin=passage.origin,
            )
        )

    results.sort(key=lambda item: item.score, reverse=True)
    unique_results: list[RetrievedPassage] = []
    seen_sources: set[str] = set()
    for result in results:
        # Keep one top passage per source to avoid a single file flooding the answer.
        if result.source in seen_sources:
            continue
        seen_sources.add(result.source)
        unique_results.append(result)
        if len(unique_results) >= top_k:
            break
    return unique_results


def search_local(query: str, data_dir: str | Path, top_k: int = 3) -> list[RetrievedPassage]:
    passages = build_local_passages(data_dir)
    return score_passages(query=query, passages=passages, top_k=top_k)


def build_web_passages(
    query: str,
    search_limit: int,
    fetch_limit: int,
    timeout: int,
    strict_sources: bool = False,
) -> list[Passage]:
    search_results = fetch_search_results(
        query=query,
        limit=search_limit,
        timeout=timeout,
        strict_sources=strict_sources,
    )
    passages: list[Passage] = []
    accepted_sources = 0
    for result in search_results:
        if accepted_sources >= fetch_limit:
            break
        try:
            final_url, html = fetch_url_text(result.url, timeout=timeout)
        except Exception:
            continue
        if not is_allowed_source(final_url, result.title, strict_sources=strict_sources):
            continue
        markdown = html_to_markdown(final_url, html)
        synthetic_doc = MarkdownDocument(path=Path(result.url), text=markdown)
        source_passages: list[Passage] = []
        for passage in split_markdown_into_passages(synthetic_doc, max_chars=1200):
            web_passage = Passage(
                source=result.url,
                title=result.title or passage.title,
                text=passage.text,
                origin="web",
            )
            if is_useful_web_passage(web_passage):
                source_passages.append(web_passage)
        if not source_passages:
            continue
        # Web pages are merged into one larger passage per source for cleaner source listings.
        combined_text = "\n".join(passage.text for passage in source_passages[:4]).strip()
        passages.append(
            Passage(
                source=result.url,
                title=result.title or source_passages[0].title,
                text=combined_text,
                origin="web",
            )
        )
        accepted_sources += 1
    return passages


def search_web(
    query: str,
    search_limit: int = 5,
    fetch_limit: int = 3,
    top_k: int = 3,
    timeout: int = 20,
    strict_sources: bool = False,
) -> list[RetrievedPassage]:
    passages = build_web_passages(
        query=query,
        search_limit=search_limit,
        fetch_limit=fetch_limit,
        timeout=timeout,
        strict_sources=strict_sources,
    )
    return score_passages(query=query, passages=passages, top_k=top_k)


def render_context(passages: list[RetrievedPassage], max_chars: int) -> str:
    lines: list[str] = []
    remaining = max_chars
    for index, passage in enumerate(passages, start=1):
        header = f"[{passage.origin.upper()} {index}] {passage.title} | {passage.source}"
        body_budget = max(40, remaining - len(header) - 4)
        snippet = passage.text.replace("\n", " ")
        if len(snippet) > body_budget:
            snippet = snippet[: body_budget - 3].rstrip() + "..."
        block = f"{header}\n{snippet}\n"
        if len(block) > remaining and lines:
            break
        lines.append(block)
        remaining -= len(block) + 1
        if remaining <= 0:
            break
    return "\n".join(lines).strip()


def render_sources(passages: list[RetrievedPassage]) -> str:
    if not passages:
        return "No sources retrieved."
    lines = []
    for index, passage in enumerate(passages, start=1):
        lines.append(f"{index}. [{passage.origin}] {passage.title} - {passage.source}")
    return "\n".join(lines)


def is_header_like(sentence: str, title: str) -> bool:
    normalized_sentence = [normalize_term(token) for token in tokenize_words(sentence)]
    normalized_title = [normalize_term(token) for token in tokenize_words(title)]
    if normalized_sentence and normalized_sentence == normalized_title:
        return True
    words = sentence.split()
    if len(words) <= 8 and not sentence.endswith((".", "!", "?")):
        return True
    return False


def normalize_list_item(item: str, lowercase_initial: bool = False) -> str:
    cleaned = item.rstrip().rstrip(".,;:").strip()
    if not cleaned:
        return ""
    if lowercase_initial and len(cleaned) > 1 and cleaned[0].isupper() and cleaned[1].islower():
        cleaned = cleaned[0].lower() + cleaned[1:]
    return cleaned


def format_list(items: list[str], lowercase_initial: bool = False) -> str:
    cleaned_items = [
        normalize_list_item(item, lowercase_initial=lowercase_initial)
        for item in items
        if item.strip()
    ]
    cleaned_items = [item for item in cleaned_items if item]
    if not cleaned_items:
        return ""
    if len(cleaned_items) == 1:
        return cleaned_items[0]
    if len(cleaned_items) == 2:
        return f"{cleaned_items[0]} and {cleaned_items[1]}"
    return f"{', '.join(cleaned_items[:-1])}, and {cleaned_items[-1]}"


def strip_trailing_period(text: str) -> str:
    return text.rstrip().rstrip(".")


def clamp_answer_length(answer_length: str) -> str:
    normalized = answer_length.strip().lower()
    if normalized in ANSWER_LENGTH_SENTENCE_LIMITS:
        return normalized
    return "medium"


def clamp_tone(tone: str) -> str:
    normalized = tone.strip().lower()
    if normalized in TONE_CHOICES:
        return normalized
    return "balanced"


def clamp_style(style: str) -> str:
    normalized = style.strip().lower()
    if normalized in STYLE_CHOICES:
        return normalized
    return "balanced"


def clamp_explanation_level(explanation_level: str) -> str:
    normalized = explanation_level.strip().lower()
    if normalized in EXPLANATION_LEVEL_CHOICES:
        return normalized
    return "medium"


def choose_tone_variant(tone: str, *, casual: str, balanced: str, formal: str) -> str:
    normalized = clamp_tone(tone)
    if normalized == "casual":
        return casual
    if normalized == "formal":
        return formal
    return balanced


def choose_style_variant(style: str, *, balanced: str, logical: str, creative: str) -> str:
    normalized = clamp_style(style)
    if normalized == "logical":
        return logical
    if normalized == "creative":
        return creative
    return balanced


def explanation_item_limit(explanation_level: str) -> int:
    normalized = clamp_explanation_level(explanation_level)
    return {
        "low": 3,
        "medium": 4,
        "high": 5,
        "advanced": 6,
    }[normalized]


def explanation_part_limit(answer_length: str, explanation_level: str) -> int:
    base_limit = ANSWER_LENGTH_PART_LIMITS[clamp_answer_length(answer_length)]
    normalized = clamp_explanation_level(explanation_level)
    level_cap = {
        "low": 2,
        "medium": 3,
        "high": 4,
        "advanced": 5,
    }[normalized]
    return min(base_limit, level_cap)


def explanation_overview_limit(answer_length: str, explanation_level: str) -> int:
    if clamp_answer_length(answer_length) == "short":
        return 0
    normalized = clamp_explanation_level(explanation_level)
    return {
        "low": 1,
        "medium": 1,
        "high": 2,
        "advanced": 3,
    }[normalized]


def select_variant(options: list[str], variation_index: int) -> str:
    if not options:
        return ""
    return options[variation_index % len(options)]


def rotate_items(items: list[str], variation_index: int) -> list[str]:
    if not items:
        return []
    offset = variation_index % len(items)
    return items[offset:] + items[:offset]


def sentence_case(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    if cleaned[0].isalpha():
        return cleaned[0].upper() + cleaned[1:]
    return cleaned


def build_creative_intro(title: str, text: str) -> str:
    cleaned = strip_trailing_period(text).strip()
    lowered_title = title.lower()
    lowered_cleaned = cleaned.lower()
    if lowered_cleaned.startswith(f"{lowered_title} is "):
        return f"You can think of {lowered_title} as {cleaned[len(title) + 3:].lstrip()}"
    if lowered_cleaned.startswith(f"{lowered_title} are "):
        return f"You can think of {lowered_title} as {cleaned[len(title) + 4:].lstrip()}"
    return f"You can think of {lowered_title} like this: {cleaned}"


def source_name_to_display(source: str, title: str) -> str | None:
    source_path = Path(source)
    stem = source_path.stem.lower()
    if stem in DISPLAY_NAME_OVERRIDES:
        return DISPLAY_NAME_OVERRIDES[stem]
    if stem and stem not in {"index", "overview"}:
        return source_path.stem.replace("_", " ").title()
    if title:
        cleaned = title.strip()
        return cleaned if len(cleaned.split()) <= 6 else None
    return None


def extract_overview_items(text: str) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("- "):
            continue
        item = line[2:].strip()
        if not item:
            continue
        normalized = item.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        items.append(item)
    return items


def clean_topic_label(label: str) -> str | None:
    cleaned = re.sub(r"\s+", " ", label).strip(" -:;,.")
    if not cleaned:
        return None
    if len(cleaned.split()) > 5:
        return None
    normalized = normalize_phrase(cleaned)
    if not normalized:
        return None
    return cleaned.lower()


def extract_topic_candidates(passages: list[RetrievedPassage]) -> list[str]:
    topics: list[str] = []
    seen: set[str] = set()
    for passage in passages:
        for raw_line in passage.text.splitlines():
            line = raw_line.strip(" -")
            if ":" not in line:
                continue
            left, right = line.split(":", 1)
            if len(right.strip()) < 20:
                continue
            topic = clean_topic_label(left)
            if not topic or topic in seen:
                continue
            seen.add(topic)
            topics.append(topic)
    return topics


def extract_reason_phrases(passages: list[RetrievedPassage]) -> list[str]:
    reasons: list[str] = []
    seen: set[str] = set()
    combined_text = " ".join(passage.text for passage in passages).lower()
    for keyword, phrase in REASON_KEYWORDS.items():
        if keyword in combined_text and phrase not in seen:
            seen.add(phrase)
            reasons.append(phrase)
    return reasons


def extract_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for sentence in SENTENCE_SPLIT_PATTERN.split(text.replace("\n", " ")):
        cleaned = sentence.strip(" -")
        if cleaned:
            sentences.append(cleaned)
    return sentences


def is_noisy_sentence(sentence: str) -> bool:
    if CREDENTIAL_NOISE_PATTERN.search(sentence) and ";" in sentence:
        return True
    if sentence.count(";") >= 3:
        return True
    return False


def extract_markdown_headings(source: str) -> list[str]:
    path = Path(source)
    if not path.exists() or path.suffix.lower() != ".md":
        return []

    headings: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("##"):
            continue
        heading = line.lstrip("#").strip()
        if heading:
            headings.append(heading)
    return headings


def parse_markdown_sections(source: str) -> tuple[str, list[str], list[MarkdownSection]]:
    path = Path(source)
    if not path.exists() or path.suffix.lower() != ".md":
        return "", [], []

    title = path.stem.replace("_", " ").title()
    intro_lines: list[str] = []
    sections: list[MarkdownSection] = []
    current_heading = ""
    current_lines: list[str] = []
    saw_section = False

    def flush_section() -> None:
        nonlocal current_lines, current_heading
        if current_heading:
            sections.append(MarkdownSection(heading=current_heading, lines=current_lines[:]))
        current_heading = ""
        current_lines = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip() or title
            continue
        if stripped.startswith("## "):
            flush_section()
            current_heading = stripped[3:].strip()
            saw_section = True
            continue
        if current_heading:
            current_lines.append(line)
        elif not saw_section:
            intro_lines.append(line)

    flush_section()
    return title, intro_lines, sections


def first_sentence_from_lines(lines: list[str]) -> str | None:
    text = " ".join(line.strip() for line in lines if line.strip())
    if not text:
        return None
    sentences = extract_sentences(text)
    return sentences[0] if sentences else text.strip()


def bullet_items(lines: list[str]) -> list[str]:
    items: list[str] = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if stripped.startswith("- "):
            item = stripped[2:].strip()
            if item:
                items.append(item)
    return items


def find_section(sections: list[MarkdownSection], keywords: tuple[str, ...]) -> MarkdownSection | None:
    for section in sections:
        heading = section.heading.lower()
        if any(keyword in heading for keyword in keywords):
            return section
    return None


def summarize_framework_items(
    items: list[str],
    limit: int = 4,
    tone: str = "balanced",
    variation_index: int = 0,
    explanation_level: str = "medium",
) -> SummaryPart | None:
    if not items:
        return None
    summarized: list[str] = []
    item_limit = min(limit, explanation_item_limit(explanation_level))
    for item in rotate_items(items, variation_index)[:item_limit]:
        if ":" in item:
            name, detail = item.split(":", 1)
            cleaned_detail = detail.strip()
            if cleaned_detail.lower().startswith("also "):
                cleaned_detail = cleaned_detail[5:]
            summarized.append(f"{name.strip()} for {cleaned_detail}")
        else:
            summarized.append(item)
    return SummaryPart(
        heading=select_variant(
            [
                choose_tone_variant(
                    tone,
                    casual="Useful frameworks and tools include:",
                    balanced="Common frameworks and tools include:",
                    formal="Common frameworks and tools include:",
                ),
                choose_tone_variant(
                    tone,
                    casual="Some useful tools and frameworks are:",
                    balanced="Other frameworks and tools include:",
                    formal="Additional frameworks and tools include:",
                ),
            ],
            variation_index,
        ),
        bullets=[normalize_list_item(item, lowercase_initial=True) for item in summarized],
    )


def summarize_feature_items(
    items: list[str],
    limit: int = 4,
    tone: str = "balanced",
    variation_index: int = 0,
    explanation_level: str = "medium",
) -> SummaryPart | None:
    if not items:
        return None
    explanation_level = clamp_explanation_level(explanation_level)
    normalized_items: list[str] = []
    item_limit = min(limit, explanation_item_limit(explanation_level))
    for item in rotate_items(items, variation_index)[:item_limit]:
        cleaned = item.strip()
        if cleaned == "No `{}` of `;`":
            normalized_items.append("not relying on braces or semicolons")
        else:
            normalized_items.append(cleaned.lower())
    return SummaryPart(
        heading=select_variant(
            [
                {
                    "low": "Main ideas:",
                    "medium": choose_tone_variant(
                        tone,
                        casual="A few key points:",
                        balanced="Key points include:",
                        formal="Key points:",
                    ),
                    "high": "Key points include:",
                    "advanced": "Core points:",
                }[explanation_level],
                choose_tone_variant(
                    tone,
                    casual="A few things worth knowing:",
                    balanced="A few notable points:",
                    formal="Notable points:",
                ),
            ],
            variation_index,
        ),
        bullets=[normalize_list_item(item, lowercase_initial=True) for item in normalized_items],
    )


def summarize_syntax_section(
    section: MarkdownSection | None,
    tone: str = "balanced",
    variation_index: int = 0,
    explanation_level: str = "medium",
) -> SummaryPart | None:
    if section is None:
        return None
    text = " ".join(line.strip() for line in section.lines if line.strip() and not line.strip().startswith("```"))
    lowered = text.lower()
    mentions: list[str] = []
    if "print(" in lowered or "print text" in lowered or "prints" in lowered:
        mentions.append("basic printing")
    if "var =" in lowered or "variable" in lowered:
        mentions.append("variable assignment")
    if mentions:
        explanation_level = clamp_explanation_level(explanation_level)
        return SummaryPart(
            heading=select_variant(
                [
                    {
                        "low": "Simple examples:",
                        "medium": choose_tone_variant(
                            tone,
                            casual="A couple of basic examples:",
                            balanced="Basic examples include:",
                            formal="Basic examples include:",
                        ),
                        "high": "Basic examples include:",
                        "advanced": "Illustrative examples:",
                    }[explanation_level],
                    choose_tone_variant(
                        tone,
                        casual="A few basic examples:",
                        balanced="Simple examples include:",
                        formal="Illustrative examples include:",
                    ),
                ],
                variation_index,
            ),
            bullets=[
                normalize_list_item(item, lowercase_initial=True)
                for item in rotate_items(mentions, variation_index)[: explanation_item_limit(explanation_level)]
            ],
        )
    sentence = first_sentence_from_lines(section.lines)
    if sentence:
        return SummaryPart(text=sentence)
    return None


def summarize_generic_section(
    section: MarkdownSection,
    title: str,
    tone: str = "balanced",
    variation_index: int = 0,
    explanation_level: str = "medium",
) -> SummaryPart | None:
    explanation_level = clamp_explanation_level(explanation_level)
    items = bullet_items(section.lines)
    if not items:
        sentence = first_sentence_from_lines(section.lines)
        return SummaryPart(text=sentence) if sentence else None

    heading = section.heading.lower()
    limited_items = rotate_items(items, variation_index)[: explanation_item_limit(explanation_level)]
    if "core" in heading or "key" in heading:
        return SummaryPart(
            heading=select_variant(
                [
                    {
                        "low": "Main ideas:",
                        "medium": choose_tone_variant(
                            tone,
                            casual="A few key points:",
                            balanced="Key points include:",
                            formal="Key points:",
                        ),
                        "high": "Key points include:",
                        "advanced": "Core points:",
                    }[explanation_level],
                    choose_tone_variant(
                        tone,
                        casual="A few things worth knowing:",
                        balanced="A few notable points:",
                        formal="Notable points:",
                    ),
                ],
                variation_index,
            ),
            bullets=[normalize_list_item(item, lowercase_initial=True) for item in limited_items],
        )
    if heading.startswith("why ") or "why " in heading or "importance" in heading:
        return SummaryPart(
            heading=select_variant(
                [
                    choose_tone_variant(
                        tone,
                        casual=f"Why {title.lower()} matter:",
                        balanced=f"{title} matter for several reasons:",
                        formal=f"Reasons {title.lower()} matter:",
                    ),
                    choose_tone_variant(
                        tone,
                        casual=f"Why {title.lower()} are important:",
                        balanced=f"Why {title.lower()} matter:",
                        formal=f"Key reasons {title.lower()} matter:",
                    ),
                ],
                variation_index,
            ),
            bullets=[normalize_list_item(item, lowercase_initial=True) for item in limited_items],
        )
    if "common" in heading or "type" in heading or "group" in heading:
        return SummaryPart(
            heading=select_variant(
                [
                    choose_tone_variant(
                        tone,
                        casual="Common examples:",
                        balanced="Common examples include:",
                        formal="Common examples include:",
                    ),
                    choose_tone_variant(
                        tone,
                        casual="Some common examples:",
                        balanced="Examples include:",
                        formal="Representative examples include:",
                    ),
                ],
                variation_index,
            ),
            bullets=[normalize_list_item(item, lowercase_initial=True) for item in limited_items],
        )
    if "growth" in heading or "need" in heading:
        return SummaryPart(
            heading=select_variant(
                [
                    choose_tone_variant(
                        tone,
                        casual="What they usually need:",
                        balanced="They usually need:",
                        formal="Typical requirements include:",
                    ),
                    choose_tone_variant(
                        tone,
                        casual="What helps them grow:",
                        balanced="They tend to need:",
                        formal="Conditions that support them include:",
                    ),
                ],
                variation_index,
            ),
            bullets=[normalize_list_item(item, lowercase_initial=True) for item in limited_items],
        )
    return SummaryPart(
        heading=choose_tone_variant(
            tone,
            casual=f"{section.heading}:",
            balanced=f"{section.heading} include:",
            formal=f"{section.heading} include:",
        ),
        bullets=[normalize_list_item(item, lowercase_initial=True) for item in limited_items],
    )


def heading_to_overview_sentence(part: SummaryPart, tone: str, variation_index: int = 0) -> str | None:
    if not part.heading or not part.bullets:
        return None

    first_items = [item for item in part.bullets[:2] if item]
    if not first_items:
        return None

    heading = part.heading.lower()
    if "key points" in heading:
        return choose_tone_variant(
            tone,
            casual=select_variant(
                [
                    f"A few big things to know are {format_list(first_items)}.",
                    f"A couple of big takeaways are {format_list(first_items)}.",
                ],
                variation_index,
            ),
            balanced=select_variant(
                [
                    f"Important points include {format_list(first_items)}.",
                    f"One useful summary is that {format_list(first_items)}.",
                ],
                variation_index,
            ),
            formal=select_variant(
                [
                    f"Notable characteristics are that {format_list(first_items)}.",
                    f"A concise summary is that {format_list(first_items)}.",
                ],
                variation_index,
            ),
        )
    if "reasons" in heading or "why " in heading:
        return choose_tone_variant(
            tone,
            casual=f"They matter because {format_list(first_items)}.",
            balanced=f"They matter because {format_list(first_items)}.",
            formal=f"They are important because {format_list(first_items)}.",
        )
    if "common examples" in heading:
        return choose_tone_variant(
            tone,
            casual=f"Common examples are {format_list(first_items)}.",
            balanced=f"Common examples include {format_list(first_items)}.",
            formal=f"Common examples include {format_list(first_items)}.",
        )
    if "usually need" in heading or "requirements" in heading:
        return choose_tone_variant(
            tone,
            casual=f"They usually need {format_list(first_items)}.",
            balanced=f"They usually need {format_list(first_items)}.",
            formal=f"Typical requirements include {format_list(first_items)}.",
        )
    return None


def build_lead_summary(
    parts: list[SummaryPart],
    answer_length: str,
    tone: str,
    style: str,
    explanation_level: str,
    variation_index: int = 0,
) -> list[str]:
    cleaned_parts = [
        part
        for part in parts
        if (part.text and part.text.strip()) or (part.heading and part.bullets)
    ]
    if not cleaned_parts:
        return []

    lead_lines: list[str] = []
    intro_text = cleaned_parts[0].text or cleaned_parts[0].heading or ""
    if intro_text:
        lead_lines.append(strip_trailing_period(intro_text) + ".")

    if answer_length == "short":
        return lead_lines

    overview_limit = explanation_overview_limit(answer_length, explanation_level)
    for part in cleaned_parts[1:]:
        overview = heading_to_overview_sentence(part, tone, variation_index=variation_index)
        if not overview:
            continue
        overview = choose_style_variant(
            style,
            balanced=overview,
            logical=overview.replace("A couple of big takeaways are", "Two key points are").replace(
                "A few big things to know are",
                "Two key points are",
            ).replace(
                "One useful summary is that",
                "A direct summary is that",
            ).replace(
                "They matter because",
                "They are important because",
            ),
            creative=overview.replace("Important points include", "One way to think about them is").replace(
                "One useful summary is that",
                "You can think of them like this:",
            ).replace(
                "They matter because",
                "Part of what makes them important is that",
            ),
        )
        lead_lines.append(strip_trailing_period(sentence_case(overview)) + ".")
        if len(lead_lines) >= overview_limit + 1:
            break

    return lead_lines


def format_summary_parts(
    summary_parts: list[SummaryPart],
    answer_length: str,
    tone: str,
    style: str,
    explanation_level: str,
    variation_index: int = 0,
) -> str | None:
    cleaned_parts = [
        part
        for part in summary_parts
        if (part.text and part.text.strip()) or (part.heading and part.bullets)
    ]
    if not cleaned_parts:
        return None
    first_part = cleaned_parts[0]
    intro_text = first_part.text or first_part.heading or ""
    if answer_length == "short" or len(cleaned_parts) == 1:
        return strip_trailing_period(intro_text) + "."

    lines = build_lead_summary(
        cleaned_parts,
        answer_length,
        tone,
        style,
        explanation_level,
        variation_index=variation_index,
    )
    for part in cleaned_parts[1:]:
        if part.text:
            prose_line = strip_trailing_period(part.text) + "."
            if prose_line not in lines:
                lines.append(prose_line)
            continue
        if part.heading and part.bullets:
            lines.append(part.heading)
            lines.extend(f"• {strip_trailing_period(item)}." for item in part.bullets if item)
    return "\n".join(lines)


def build_intro_summary(
    title: str,
    intro: str,
    features_section: MarkdownSection | None,
    frameworks_section: MarkdownSection | None,
    tone: str = "balanced",
    style: str = "balanced",
    explanation_level: str = "medium",
    variation_index: int = 0,
) -> str:
    lowered_intro = intro.lower()
    feature_items = bullet_items(features_section.lines) if features_section else []
    has_frameworks = bool(bullet_items(frameworks_section.lines)) if frameworks_section else False
    style = clamp_style(style)
    explanation_level = clamp_explanation_level(explanation_level)

    if "programming language" in lowered_intro:
        simple_signal = any("simple" in item.lower() for item in feature_items) or "beginner" in lowered_intro
        if simple_signal and has_frameworks:
            base = select_variant(
                [
                    choose_tone_variant(
                        tone,
                        casual=f"{title} is a programming language that's easy to pick up but still strong enough for serious projects.",
                        balanced=f"{title} is a programming language that is easy to pick up but still flexible enough for serious projects.",
                        formal=f"{title} is a programming language that is approachable for beginners while remaining capable enough for serious projects.",
                    ),
                    choose_tone_variant(
                        tone,
                        casual=f"{title} is a language that's friendly at the start but still powerful for serious work.",
                        balanced=f"{title} is a programming language that is approachable at first and still practical for serious work.",
                        formal=f"{title} is a programming language that remains accessible while still supporting serious work.",
                    ),
                ],
                variation_index,
            )
            base_result = choose_style_variant(
                style,
                balanced=base,
                logical=f"{title} is a programming language that is approachable early on and still well suited to serious work.",
                creative=f"{title} is a programming language that feels easy to start with while still growing into serious projects.",
            )
            if explanation_level == "low":
                return f"{title} is a programming language that is easy to start learning and useful for serious projects."
            if explanation_level == "advanced":
                return base_result.replace("serious work", "substantial technical work")
            return base_result
        if simple_signal:
            base = select_variant(
                [
                    choose_tone_variant(
                        tone,
                        casual=f"{title} is a programming language that's easy to learn and pleasant to work with.",
                        balanced=f"{title} is a programming language that is easy to learn and comfortable to work with.",
                        formal=f"{title} is a programming language that is relatively easy to learn and practical to work with.",
                    ),
                    choose_tone_variant(
                        tone,
                        casual=f"{title} is a programming language that's fairly easy to learn and use.",
                        balanced=f"{title} is a programming language that is straightforward to learn and practical to use.",
                        formal=f"{title} is a programming language that is generally straightforward to learn and practical to use.",
                    ),
                ],
                variation_index,
            )
            base_result = choose_style_variant(
                style,
                balanced=base,
                logical=f"{title} is a programming language that is relatively easy to learn and practical to use.",
                creative=f"{title} is a programming language that is easy to get into and pleasant to build with.",
            )
            if explanation_level == "low":
                return f"{title} is a programming language that is fairly easy to learn and use."
            if explanation_level == "advanced":
                return base_result.replace("practical to use", "effective across many technical tasks")
            return base_result
        base = select_variant(
            [
                choose_tone_variant(
                    tone,
                    casual=f"{title} is a programming language used for lots of different tasks.",
                    balanced=f"{title} is a programming language used for a wide range of tasks.",
                    formal=f"{title} is a programming language used across a wide range of tasks.",
                ),
                choose_tone_variant(
                    tone,
                    casual=f"{title} is a programming language people use for many different kinds of work.",
                    balanced=f"{title} is a programming language used across many different kinds of work.",
                    formal=f"{title} is a programming language applied across many different kinds of work.",
                ),
            ],
            variation_index,
        )
        base_result = choose_style_variant(
            style,
            balanced=base,
            logical=f"{title} is a programming language used across many different tasks.",
            creative=f"{title} is a programming language people reach for in many different kinds of work.",
        )
        if explanation_level == "low":
            return f"{title} is a programming language used for many different tasks."
        if explanation_level == "advanced":
            return base_result.replace("many different tasks", "a wide range of technical tasks")
        return base_result

    rewritten = intro
    replacements = [
        (" is meant to be ", " is designed to be "),
        ("beginner-friendly", "easy to get started with"),
        ("while also being", "while still being"),
        ("deep enough for experts and professional developers", "powerful enough for advanced and professional work"),
    ]
    for old, new in replacements:
        rewritten = rewritten.replace(old, new)
    if tone == "casual":
        rewritten = rewritten.replace(" is ", " is ").replace("usually", "often")
    elif tone == "formal":
        rewritten = rewritten.replace("usually", "typically")
    base_result = choose_style_variant(
        style,
        balanced=rewritten,
        logical=rewritten.replace("often", "typically").replace("usually", "typically"),
        creative=build_creative_intro(title, rewritten),
    )
    if explanation_level == "low":
        return (
            base_result.replace("living organisms", "living things")
            .replace("organisms", "living things")
            .replace("typically", "usually")
        )
    if explanation_level == "advanced":
        return base_result.replace("living things", "organisms")
    return base_result


def build_local_markdown_subject_summary(
    source: str,
    answer_length: str = "medium",
    tone: str = "balanced",
    style: str = "balanced",
    explanation_level: str = "medium",
    variation_index: int = 0,
) -> str | None:
    title, intro_lines, sections = parse_markdown_sections(source)
    intro = first_sentence_from_lines(intro_lines)
    if not intro:
        return None
    answer_length = clamp_answer_length(answer_length)
    tone = clamp_tone(tone)
    style = clamp_style(style)
    explanation_level = clamp_explanation_level(explanation_level)

    features_section = find_section(sections, ("feature", "trait", "distinct"))
    syntax_section = find_section(sections, ("syntax", "example", "usage"))
    frameworks_section = find_section(sections, ("framework", "library", "tool"))
    generic_sections = [
        section
        for section in sections
        if section is not features_section and section is not syntax_section and section is not frameworks_section
    ]

    intro_summary = build_intro_summary(
        title=title,
        intro=intro,
        features_section=features_section,
        frameworks_section=frameworks_section,
        tone=tone,
        style=style,
        explanation_level=explanation_level,
        variation_index=variation_index,
    )
    summary_parts = [SummaryPart(text=intro_summary)]

    features_summary = (
        summarize_feature_items(
            bullet_items(features_section.lines),
            tone=tone,
            explanation_level=explanation_level,
            variation_index=variation_index,
        )
        if features_section
        else None
    )
    if features_summary:
        summary_parts.append(features_summary)

    syntax_summary = summarize_syntax_section(
        syntax_section,
        tone=tone,
        variation_index=variation_index,
        explanation_level=explanation_level,
    )
    if syntax_summary:
        summary_parts.append(syntax_summary)

    frameworks_summary = (
        summarize_framework_items(
            bullet_items(frameworks_section.lines),
            tone=tone,
            explanation_level=explanation_level,
            variation_index=variation_index,
        )
        if frameworks_section
        else None
    )
    if frameworks_summary:
        summary_parts.append(frameworks_summary)

    for section in rotate_items(generic_sections, variation_index):
        section_summary = summarize_generic_section(
            section,
            title,
            tone=tone,
            variation_index=variation_index,
            explanation_level=explanation_level,
        )
        if section_summary:
            summary_parts.append(section_summary)

    max_parts = explanation_part_limit(answer_length, explanation_level)
    return format_summary_parts(
        summary_parts[:max_parts],
        answer_length,
        tone,
        style,
        explanation_level,
        variation_index=variation_index,
    )


def build_subject_summary(
    query: str,
    passages: list[RetrievedPassage],
    answer_length: str = "medium",
    tone: str = "balanced",
    style: str = "balanced",
    explanation_level: str = "medium",
    variation_index: int = 0,
) -> str | None:
    if USE_QUESTION_PATTERN.search(query.strip()):
        return None
    answer_length = clamp_answer_length(answer_length)
    tone = clamp_tone(tone)
    style = clamp_style(style)
    explanation_level = clamp_explanation_level(explanation_level)

    match = TELL_ABOUT_PATTERN.search(query.strip())
    if not match:
        match = WHAT_IS_SUBJECT_PATTERN.search(query.strip())
    if not match or not passages:
        return None

    subject = match.group("subject").strip()
    subject_terms = set(extract_query_terms(subject))
    if not subject_terms:
        return None

    filtered_passages: list[RetrievedPassage] = []
    for passage in passages:
        combined_terms = set(extract_query_terms(f"{passage.title}\n{passage.text}\n{passage.source}"))
        if subject_terms & combined_terms:
            filtered_passages.append(passage)

    if not filtered_passages:
        return None

    normalized_subject = normalize_phrase(subject)
    exact_local_matches = [
        passage
        for passage in filtered_passages
        if passage.origin == "local"
        and (
            normalize_phrase(passage.title) == normalized_subject
            or normalize_phrase(Path(passage.source).stem) == normalized_subject
        )
    ]
    if exact_local_matches:
        # Prefer the user's own markdown topic file over looser web matches.
        best_passage = max(exact_local_matches, key=lambda item: item.score)
    else:
        best_passage = max(filtered_passages, key=lambda item: item.score)

    local_summary = build_local_markdown_subject_summary(
        best_passage.source,
        answer_length=answer_length,
        tone=tone,
        style=style,
        explanation_level=explanation_level,
        variation_index=variation_index,
    )
    if local_summary:
        return local_summary

    sentences = [
        sentence for sentence in extract_sentences(best_passage.text)
        if not is_header_like(sentence, best_passage.title) and not is_noisy_sentence(sentence)
    ]
    if not sentences:
        return None

    sentence_limit = ANSWER_LENGTH_SENTENCE_LIMITS[answer_length]
    summary = " ".join(sentences[:sentence_limit])
    headings = extract_markdown_headings(best_passage.source)
    if headings and answer_length != "short":
        normalized_headings = [heading.lower() for heading in headings[:4]]
        summary += " " + choose_style_variant(
            style,
            balanced=choose_tone_variant(
                tone,
                casual=f"There are also sections on {format_list(normalized_headings)}.",
                balanced=f"I also have notes on {format_list(normalized_headings)}.",
                formal=f"Related sections include {format_list(normalized_headings)}.",
            ),
            logical=f"Related sections include {format_list(normalized_headings)}.",
            creative=f"There are also sections that explore {format_list(normalized_headings)}.",
        )
    return summary


def build_use_case_summary(
    query: str,
    passages: list[RetrievedPassage],
    answer_length: str = "medium",
    tone: str = "balanced",
    style: str = "balanced",
    explanation_level: str = "medium",
    variation_index: int = 0,
) -> str | None:
    match = USE_QUESTION_PATTERN.search(query.strip())
    if not match or not passages:
        return None
    answer_length = clamp_answer_length(answer_length)
    tone = clamp_tone(tone)
    style = clamp_style(style)
    explanation_level = clamp_explanation_level(explanation_level)

    subject = match.group("subject").strip()
    subject_name = subject[0].upper() + subject[1:] if subject else "It"
    topic_limit = min({"short": 3, "medium": 6, "long": 8}[answer_length], explanation_item_limit(explanation_level) + 2)
    topics = rotate_items(extract_topic_candidates(passages), variation_index)[:topic_limit]
    reasons = rotate_items(extract_reason_phrases(passages), variation_index)[:3]

    if not topics:
        return None

    summary = choose_style_variant(
        style,
        balanced=choose_tone_variant(
            tone,
            casual=f"{subject_name} is often used for {format_list(topics)}.",
            balanced=f"{subject_name} is commonly used for {format_list(topics)}.",
            formal=f"{subject_name} is commonly used for {format_list(topics)}.",
        ),
        logical=f"{subject_name} is commonly used for {format_list(topics)}.",
        creative=f"{subject_name} often shows up in work involving {format_list(topics)}.",
    )
    if reasons and answer_length != "short":
        summary += " " + choose_style_variant(
            style,
            balanced=choose_tone_variant(
                tone,
                casual=f"People like it because of {format_list(reasons)}.",
                balanced=f"It is popular because of {format_list(reasons)}.",
                formal=f"It is widely used because of {format_list(reasons)}.",
            ),
            logical=f"It is widely used because of {format_list(reasons)}.",
            creative=f"Part of the appeal comes from {format_list(reasons)}.",
        )
    return summary


def build_known_items_summary(
    query: str,
    passages: list[RetrievedPassage],
    answer_length: str = "medium",
    tone: str = "balanced",
    style: str = "balanced",
    explanation_level: str = "medium",
    variation_index: int = 0,
) -> str | None:
    match = KNOW_ABOUT_PATTERN.search(query.strip())
    if not match or not passages:
        return None
    answer_length = clamp_answer_length(answer_length)
    tone = clamp_tone(tone)
    style = clamp_style(style)
    explanation_level = clamp_explanation_level(explanation_level)

    category = match.group("category").strip().lower()
    if "game" not in category:
        return None

    items: list[str] = []
    seen: set[str] = set()
    game_directories = {
        Path(passage.source).parent
        for passage in passages
        if "games" in passage.source.lower()
    }
    for game_directory in sorted(game_directories):
        if not game_directory.exists():
            continue
        for file_path in sorted(game_directory.glob("*.md")):
            if file_path.stem.lower() in {"overview", "index"}:
                continue
            display_name = source_name_to_display(str(file_path), file_path.stem)
            if not display_name:
                continue
            normalized = display_name.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            items.append(display_name)

    if items:
        items = rotate_items(items, variation_index)[: explanation_item_limit(explanation_level) + 1]
        summary = choose_style_variant(
            style,
            balanced=choose_tone_variant(
                tone,
                casual=f"Games covered right now include {format_list(items)}.",
                balanced=f"Games covered right now include {format_list(items)}.",
                formal=f"Current game coverage includes {format_list(items)}.",
            ),
            logical=f"Current game coverage includes {format_list(items)}.",
            creative=f"Right now the game lineup includes {format_list(items)}.",
        )
        if answer_length != "short":
            summary += " " + choose_style_variant(
                style,
                balanced=choose_tone_variant(
                    tone,
                    casual="I can also explain what each one is like, what makes it distinct, and why people play it.",
                    balanced="I can also explain what each one is like, what makes it distinct, and why people play it.",
                    formal="I can also explain what each one is like, what makes it distinct, and why people play it.",
                ),
                logical="I can also explain what each one is like, what makes it distinct, and why people play it.",
                creative="I can also break down what each one feels like, what makes it stand out, and why people keep coming back to it.",
            )
        return summary

    for passage in passages:
        if "games" not in passage.source.lower():
            continue
        source_path = Path(passage.source)
        if source_path.stem.lower() in {"overview", "index"}:
            for item in extract_overview_items(passage.text):
                normalized = item.lower()
                if normalized in seen:
                    continue
                seen.add(normalized)
                items.append(item)
            continue
        display_name = source_name_to_display(passage.source, passage.title)
        if not display_name:
            continue
        normalized = display_name.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        items.append(display_name)

    if not items:
        return None

    items = rotate_items(items, variation_index)[: explanation_item_limit(explanation_level) + 1]
    summary = choose_style_variant(
        style,
        balanced=choose_tone_variant(
            tone,
            casual=f"Games covered right now include {format_list(items)}.",
            balanced=f"Games covered right now include {format_list(items)}.",
            formal=f"Current game coverage includes {format_list(items)}.",
        ),
        logical=f"Current game coverage includes {format_list(items)}.",
        creative=f"Right now the game lineup includes {format_list(items)}.",
    )
    if answer_length != "short":
        summary += " " + choose_style_variant(
            style,
            balanced=choose_tone_variant(
                tone,
                casual="I can also explain what each one is like, what makes it distinct, and why people play it.",
                balanced="I can also explain what each one is like, what makes it distinct, and why people play it.",
                formal="I can also explain what each one is like, what makes it distinct, and why people play it.",
            ),
            logical="I can also explain what each one is like, what makes it distinct, and why people play it.",
            creative="I can also break down what each one feels like, what makes it stand out, and why people keep coming back to it.",
        )
    return summary


def build_special_case_answer(
    query: str,
    passages: list[RetrievedPassage],
    answer_length: str = "medium",
    tone: str = "balanced",
    style: str = "balanced",
    explanation_level: str = "medium",
    variation_index: int = 0,
) -> str | None:
    answer_length = clamp_answer_length(answer_length)
    tone = clamp_tone(tone)
    style = clamp_style(style)
    explanation_level = clamp_explanation_level(explanation_level)
    subject_summary = build_subject_summary(
        query,
        passages,
        answer_length=answer_length,
        tone=tone,
        style=style,
        explanation_level=explanation_level,
        variation_index=variation_index,
    )
    if subject_summary:
        return subject_summary

    known_items_summary = build_known_items_summary(
        query,
        passages,
        answer_length=answer_length,
        tone=tone,
        style=style,
        explanation_level=explanation_level,
        variation_index=variation_index,
    )
    if known_items_summary:
        return known_items_summary

    use_case_summary = build_use_case_summary(
        query,
        passages,
        answer_length=answer_length,
        tone=tone,
        style=style,
        explanation_level=explanation_level,
        variation_index=variation_index,
    )
    if use_case_summary:
        return use_case_summary

    return None


def build_general_summary(
    query: str,
    passages: list[RetrievedPassage],
    answer_length: str = "medium",
    tone: str = "balanced",
    style: str = "balanced",
    explanation_level: str = "medium",
    variation_index: int = 0,
) -> str | None:
    query_terms = set(extract_query_terms(query))
    if not query_terms or not passages:
        return None
    answer_length = clamp_answer_length(answer_length)
    tone = clamp_tone(tone)
    style = clamp_style(style)
    explanation_level = clamp_explanation_level(explanation_level)
    max_points = min(
        ANSWER_LENGTH_SENTENCE_LIMITS[answer_length],
        {"low": 2, "medium": 3, "high": 4, "advanced": 5}[explanation_level],
    )

    points: list[str] = []
    seen: set[str] = set()
    for passage in passages:
        for sentence in SENTENCE_SPLIT_PATTERN.split(passage.text.replace("\n", " ")):
            cleaned = sentence.strip(" -")
            if len(cleaned) < 40 or is_header_like(cleaned, passage.title):
                continue
            if is_noisy_sentence(cleaned):
                continue
            sentence_terms = set(extract_query_terms(cleaned))
            overlap = len(query_terms & sentence_terms)
            if overlap == 0:
                continue
            normalized = normalize_phrase(cleaned)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            points.append(cleaned)
            if len(points) >= max_points:
                break
        if len(points) >= max_points:
            break

    if not points:
        return None

    body = " ".join(points)
    if tone == "formal":
        return body
    if tone == "casual":
        return body
    sentence_pool = rotate_items(points, variation_index)
    body = " ".join(sentence_pool[:max_points])
    return choose_style_variant(
        style,
        balanced=body,
        logical=body,
        creative=body.replace(" is ", " can be ").replace(" are ", " can be "),
    )


def build_extractive_answer(
    query: str,
    passages: list[RetrievedPassage],
    answer_length: str = "medium",
    tone: str = "balanced",
    style: str = "balanced",
    explanation_level: str = "medium",
    variation_index: int = 0,
) -> str:
    answer_length = clamp_answer_length(answer_length)
    tone = clamp_tone(tone)
    style = clamp_style(style)
    explanation_level = clamp_explanation_level(explanation_level)
    max_sentences = ANSWER_LENGTH_SENTENCE_LIMITS[answer_length]
    special_case_answer = build_special_case_answer(
        query,
        passages,
        answer_length=answer_length,
        tone=tone,
        style=style,
        explanation_level=explanation_level,
        variation_index=variation_index,
    )
    if special_case_answer:
        return special_case_answer

    general_summary = build_general_summary(
        query,
        passages,
        answer_length=answer_length,
        tone=tone,
        style=style,
        explanation_level=explanation_level,
        variation_index=variation_index,
    )
    if general_summary:
        return general_summary

    query_terms = set(extract_query_terms(query))
    if not query_terms or not passages:
        return NO_RELEVANT_INFORMATION_MESSAGE

    scored_sentences: list[tuple[float, str]] = []
    for passage in passages:
        for sentence in SENTENCE_SPLIT_PATTERN.split(passage.text.replace("\n", " ")):
            cleaned = sentence.strip(" -")
            if len(cleaned) < 30:
                continue
            if is_header_like(cleaned, passage.title):
                continue
            if is_noisy_sentence(cleaned):
                continue
            sentence_terms = set(extract_query_terms(cleaned))
            overlap = len(query_terms & sentence_terms)
            if overlap == 0:
                continue
            score = overlap + passage.score
            scored_sentences.append((score, cleaned))

    if not scored_sentences:
        best_passage = passages[0].text.replace("\n", " ").strip()
        if not best_passage:
            return NO_RELEVANT_INFORMATION_MESSAGE
        return best_passage[:280] + ("..." if len(best_passage) > 280 else "")

    answer_lines: list[str] = []
    seen_sentences: set[str] = set()
    for _, sentence in sorted(scored_sentences, key=lambda item: item[0], reverse=True):
        if sentence in seen_sentences:
            continue
        seen_sentences.add(sentence)
        answer_lines.append(f"- {sentence}")
        if len(answer_lines) >= max_sentences:
            break

    return "\n".join(answer_lines)
