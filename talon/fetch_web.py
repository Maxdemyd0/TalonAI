"""
CLI for importing webpages into Talon's markdown knowledge folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .web import save_web_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch webpages into markdown files for Talon training.")
    parser.add_argument("--url", action="append", dest="urls", help="A webpage URL to fetch. Repeat for multiple URLs.")
    parser.add_argument("--from-file", dest="from_file", help="Optional text file containing one URL per line.")
    parser.add_argument("--output-dir", default="data/knowledge/web")
    parser.add_argument("--timeout", type=int, default=20)
    return parser.parse_args()


def load_urls(args: argparse.Namespace) -> list[str]:
    urls = list(args.urls or [])
    if args.from_file:
        file_path = Path(args.from_file)
        lines = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines()]
        urls.extend(line for line in lines if line and not line.startswith("#"))
    # Preserve order while removing duplicate URLs.
    unique_urls = list(dict.fromkeys(urls))
    if not unique_urls:
        raise SystemExit("Provide at least one --url or use --from-file with a URL list.")
    return unique_urls


def main() -> None:
    args = parse_args()
    urls = load_urls(args)
    for url in urls:
        # Each fetched page is converted into markdown so Talon can train on it later.
        destination = save_web_markdown(url=url, output_dir=args.output_dir, timeout=args.timeout)
        print(f"Saved {url} -> {destination}")


if __name__ == "__main__":
    main()
