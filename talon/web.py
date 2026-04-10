"""
Low-level web fetching and HTML-to-markdown conversion for Talon.
"""

from __future__ import annotations

from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen
import re


USER_AGENT = "TalonAI/0.1 (+https://local.talon)"
BLOCKED_SOURCE_HOSTS = {
    "wikipedia.org",
    "fandom.com",
    "wikia.com",
}
BLOCKED_SOURCE_TITLE_KEYWORDS = {
    "wikipedia",
    "fandom",
}
STRICT_ALLOWED_SOURCE_HOSTS = {
    "apple.com",
    "britannica.com",
    "cdc.gov",
    "datacamp.com",
    "developer.mozilla.org",
    "django-project.com",
    "djangoproject.com",
    "docs.djangoproject.com",
    "docs.oracle.com",
    "fastapi.tiangolo.com",
    "flask.palletsprojects.com",
    "geeksforgeeks.org",
    "hibernate.org",
    "jakarta.ee",
    "learn.microsoft.com",
    "microsoft.com",
    "mozilla.org",
    "nasa.gov",
    "nih.gov",
    "noaa.gov",
    "numpy.org",
    "openai.com",
    "oracle.com",
    "pandas.pydata.org",
    "palletsprojects.com",
    "python.org",
    "pytorch.org",
    "scipy.org",
    "si.edu",
    "spring.io",
    "support.apple.com",
    "tiangolo.com",
    "un.org",
    "usgs.gov",
    "who.int",
}


class SearchResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[SearchResult] = []
        self._current_href: str | None = None
        self._current_text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self._current_href = href
            self._current_text_parts = []

    def handle_data(self, data: str) -> None:
        if self._current_href is not None:
            cleaned = normalize_whitespace(data)
            if cleaned:
                self._current_text_parts.append(cleaned)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self._current_href is None:
            return
        # DuckDuckGo's HTML results expose links as regular anchors, so we collect them here.
        title = normalize_whitespace(" ".join(self._current_text_parts))
        url = unwrap_search_result_url(self._current_href)
        if title and url:
            self.results.append(SearchResult(title=title, url=url))
        self._current_href = None
        self._current_text_parts = []


class SearchResult:
    def __init__(self, title: str, url: str) -> None:
        self.title = title
        self.url = url


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def hostname_matches_blocklist(hostname: str) -> bool:
    normalized = hostname.lower().strip(".")
    return any(normalized == blocked or normalized.endswith(f".{blocked}") for blocked in BLOCKED_SOURCE_HOSTS)


def hostname_matches_allowlist(hostname: str) -> bool:
    normalized = hostname.lower().strip(".")
    if normalized.endswith(".gov") or normalized.endswith(".edu"):
        return True
    return any(normalized == allowed or normalized.endswith(f".{allowed}") for allowed in STRICT_ALLOWED_SOURCE_HOSTS)


def is_blocked_source(url: str, title: str = "") -> bool:
    parsed = urlparse(url)
    hostname = parsed.netloc.split("@")[-1].split(":")[0]
    if hostname and hostname_matches_blocklist(hostname):
        return True
    lowered_title = title.lower()
    return any(keyword in lowered_title for keyword in BLOCKED_SOURCE_TITLE_KEYWORDS)


def is_allowed_source(url: str, title: str = "", strict_sources: bool = False) -> bool:
    parsed = urlparse(url)
    hostname = parsed.netloc.split("@")[-1].split(":")[0]
    if is_blocked_source(url, title):
        return False
    if not strict_sources:
        return True
    if not hostname:
        return False
    return hostname_matches_allowlist(hostname)


def slugify_url(url: str) -> str:
    parsed = urlparse(url)
    parts = [parsed.netloc.replace(":", "_")]
    if parsed.path and parsed.path != "/":
        parts.extend(part for part in parsed.path.split("/") if part)
    slug = "-".join(parts)
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", slug).strip("-_.")
    return slug or "web-page"


def unwrap_search_result_url(url: str) -> str | None:
    if url.startswith("//"):
        url = f"https:{url}"
    parsed = urlparse(url)
    if not parsed.scheme:
        return None
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [None])[0]
        return unquote(target) if target else None
    return url if parsed.scheme in {"http", "https"} else None


class HTMLDocumentParser(HTMLParser):
    block_tags = {"p", "div", "section", "article", "main", "header", "footer", "aside", "li", "pre", "code"}
    heading_tags = {"h1", "h2", "h3", "h4", "h5", "h6"}
    ignored_tags = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self._inside_title = False
        self._ignored_depth = 0
        self._current_parts: list[str] = []
        self.blocks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.ignored_tags:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag == "title":
            self._inside_title = True
            return
        if tag in self.heading_tags | self.block_tags and self._current_parts:
            self._flush_block()

    def handle_endtag(self, tag: str) -> None:
        if tag in self.ignored_tags and self._ignored_depth:
            self._ignored_depth -= 1
            return
        if self._ignored_depth:
            return
        if tag == "title":
            self._inside_title = False
            return
        if tag in self.heading_tags | self.block_tags:
            self._flush_block()

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        cleaned = normalize_whitespace(data)
        if not cleaned:
            return
        if self._inside_title:
            self.title = f"{self.title} {cleaned}".strip()
        else:
            # Blocks are stitched together later into markdown paragraphs.
            self._current_parts.append(cleaned)

    def _flush_block(self) -> None:
        if not self._current_parts:
            return
        text = normalize_whitespace(" ".join(self._current_parts))
        if text:
            self.blocks.append(text)
        self._current_parts.clear()

    def finish(self) -> tuple[str, list[str]]:
        self._flush_block()
        deduped_blocks: list[str] = []
        previous = None
        for block in self.blocks:
            if block == previous:
                continue
            deduped_blocks.append(block)
            previous = block
        return self.title or "Untitled Page", deduped_blocks


def fetch_url_text(url: str, timeout: int = 20) -> tuple[str, str]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        html = response.read().decode(charset, errors="replace")
        final_url = response.geturl()
    return final_url, html


def fetch_search_results(
    query: str,
    limit: int = 5,
    timeout: int = 20,
    strict_sources: bool = False,
) -> list[SearchResult]:
    search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    _, html = fetch_url_text(search_url, timeout=timeout)
    parser = SearchResultParser()
    parser.feed(html)
    parser.close()

    unique_results: list[SearchResult] = []
    seen_urls: set[str] = set()
    for result in parser.results:
        if result.url in seen_urls:
            continue
        if not is_allowed_source(result.url, result.title, strict_sources=strict_sources):
            continue
        seen_urls.add(result.url)
        unique_results.append(result)
        if len(unique_results) >= limit:
            break
    return unique_results


def html_to_markdown(source_url: str, html: str) -> str:
    parser = HTMLDocumentParser()
    parser.feed(html)
    parser.close()
    title, blocks = parser.finish()
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        f"# {title}",
        "",
        # Metadata at the top makes imported pages easier to trace later.
        f"- Source: {source_url}",
        f"- Fetched: {fetched_at}",
        "",
    ]

    for block in blocks:
        lines.append(block)
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def save_web_markdown(
    url: str,
    output_dir: str | Path,
    timeout: int = 20,
    strict_sources: bool = False,
) -> Path:
    final_url, html = fetch_url_text(url, timeout=timeout)
    if not is_allowed_source(final_url, strict_sources=strict_sources):
        raise ValueError(f"Blocked source for Talon web knowledge: {final_url}")
    markdown = html_to_markdown(final_url, html)
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{slugify_url(final_url)}.md"
    destination.write_text(markdown, encoding="utf-8")
    return destination
