"""Comic scraping utility for atm166.org.

This module provides a CLI entry-point capable of downloading the entire
comic found at https://atm166.org/comicdetail/32165976.  The scraper will
discover all chapters listed on the root page and then download every image
for each chapter, storing the results under ``/Users/hyhh/Desktop/爬虫`` by
default.  The implementation tries to be resilient against small HTML
structure differences by relying on heuristics when extracting chapter and
image URLs.

Example usage::

    python scripts/comic_scraper.py \
        --root-url https://atm166.org/comicdetail/32165976 \
        --output-dir /Users/hyhh/Desktop/爬虫

The default argument values already match the example above, so running the
script without arguments will target the comic requested by the user.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


DEFAULT_ROOT_URL = "https://atm166.org/comicdetail/32165976"
DEFAULT_OUTPUT_DIR = Path("/Users/hyhh/Desktop/爬虫")
LOGGER = logging.getLogger(__name__)


def _sanitize_filename(name: str) -> str:
    """Convert *name* into a filesystem friendly representation."""

    sanitized = re.sub(r"[\\/:*?\"<>|]", "_", name)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    return sanitized or "chapter"


def _chapter_sort_key(url: str, title: str) -> Tuple:
    """Generate a sort key that keeps chapters in reading order."""

    numbers = re.findall(r"\d+", url) or re.findall(r"\d+", title)
    if numbers:
        return tuple(int(n) for n in numbers)
    return (title,)


def _candidate_attributes(tag) -> Iterable[str]:
    for attr in ("data-src", "data-original", "data-url", "data-echo",
                 "src"):
        value = tag.get(attr)
        if value:
            yield value


@dataclass
class Chapter:
    title: str
    url: str


class ComicScraper:
    """Scrape a comic hosted on atm166.org."""

    def __init__(self, root_url: str, output_dir: Path, *, delay: float = 0.0):
        self.root_url = root_url
        self.output_dir = Path(output_dir)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/118.0 Safari/537.36"
                ),
                "Referer": "https://atm166.org/",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            }
        )

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
    def fetch(self, url: str) -> str:
        """Return the HTML for *url*, raising an informative error on failure."""

        LOGGER.debug("Fetching %s", url)
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network code
            raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc
        return response.text

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def extract_chapters(self, html: str) -> List[Chapter]:
        """Parse the root page *html* and return discovered chapters."""

        soup = BeautifulSoup(html, "html.parser")
        candidates = []

        selectors = [
            "div.chapter-list a",
            "ul.chapter__list a",
            "div#chapterlist a",
            "div.list-block a",
        ]
        for selector in selectors:
            candidates.extend(soup.select(selector))

        if not candidates:
            candidates = soup.find_all("a", href=True)

        chapters: List[Chapter] = []
        seen = set()
        root_path = urlparse(self.root_url).path.rstrip("/")

        for anchor in candidates:
            href = anchor.get("href")
            if not href or "javascript" in href.lower():
                continue

            url = urljoin(self.root_url, href)
            if url in seen:
                continue

            if not self._is_same_comic(root_path, url):
                continue

            title = anchor.get("title") or anchor.get_text(strip=True)
            title = title or Path(urlparse(url).path).name

            seen.add(url)
            chapters.append(Chapter(title=title, url=url))

        chapters.sort(key=lambda chapter: _chapter_sort_key(chapter.url, chapter.title))
        return chapters

    def _is_same_comic(self, root_path: str, chapter_url: str) -> bool:
        parsed = urlparse(chapter_url)
        if parsed.netloc and "atm166.org" not in parsed.netloc:
            return False

        if root_path:
            return root_path in parsed.path
        return True

    def extract_image_urls(self, html: str, base_url: str) -> List[str]:
        soup = BeautifulSoup(html, "html.parser")
        images: List[str] = []

        selectors = [
            "div#viewer img",
            "div.comic-contain img",
            "div.comic-imgs img",
            "div.reading-area img",
            "div#images img",
        ]

        for selector in selectors:
            images.extend(soup.select(selector))

        if not images:
            images = soup.find_all("img")

        urls: List[str] = []
        for img in images:
            for candidate in _candidate_attributes(img):
                candidate = candidate.strip()
                if not candidate:
                    continue
                if candidate.startswith("data:image"):
                    continue
                absolute = urljoin(base_url, candidate)
                if absolute not in urls:
                    urls.append(absolute)
                break
        return urls

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------
    def download_image(self, url: str, destination: Path) -> None:
        LOGGER.debug("Downloading image %s -> %s", url, destination)
        try:
            with self.session.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=65536):
                        if chunk:
                            fh.write(chunk)
        except requests.RequestException as exc:  # pragma: no cover - network code
            LOGGER.error("Failed to download %s: %s", url, exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def scrape(self) -> None:
        LOGGER.info("Fetching comic root %s", self.root_url)
        root_html = self.fetch(self.root_url)
        chapters = self.extract_chapters(root_html)

        if not chapters:
            LOGGER.warning("No chapters were discovered on %s", self.root_url)
            return

        LOGGER.info("Discovered %d chapters", len(chapters))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for index, chapter in enumerate(chapters, start=1):
            LOGGER.info("[%d/%d] Downloading chapter: %s", index, len(chapters), chapter.title)
            self.scrape_chapter(chapter)
            if self.delay:
                time.sleep(self.delay)

    def scrape_chapter(self, chapter: Chapter) -> None:
        try:
            html = self.fetch(chapter.url)
        except RuntimeError as exc:  # pragma: no cover - network code
            LOGGER.error("Skipping chapter %s due to error: %s", chapter.title, exc)
            return

        image_urls = self.extract_image_urls(html, chapter.url)

        if not image_urls:
            LOGGER.warning("No images found for chapter %s (%s)", chapter.title, chapter.url)
            return

        chapter_dir = self.output_dir / _sanitize_filename(chapter.title)
        chapter_dir.mkdir(parents=True, exist_ok=True)

        for position, image_url in enumerate(image_urls, start=1):
            ext = Path(urlparse(image_url).path).suffix or ".jpg"
            filename = f"{position:03d}{ext}"
            destination = chapter_dir / filename
            self.download_image(image_url, destination)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download comics from atm166.org")
    parser.add_argument(
        "--root-url",
        default=DEFAULT_ROOT_URL,
        help="Root page for the comic to download",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to store downloaded chapters",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Optional delay (in seconds) between chapter downloads",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (e.g. INFO, DEBUG)",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    configure_logging(args.log_level)

    scraper = ComicScraper(
        root_url=args.root_url,
        output_dir=args.output_dir,
        delay=args.delay,
    )

    try:
        scraper.scrape()
    except RuntimeError as exc:  # pragma: no cover - network errors
        LOGGER.error("Scraping failed: %s", exc)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
