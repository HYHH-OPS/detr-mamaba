"""Utility for downloading manga chapters from a root page and exporting per-chapter PDFs.

The scraper is intentionally configurable so that it can adapt to different manga web sites.
It discovers chapter links from the supplied root URL, downloads the images of each chapter
concurrently, and writes a PDF per chapter where the filename combines the manga title and
chapter title.

Example
-------
python scripts/manga_scraper.py https://example.com/manga/demo \
    --chapter-selector "div.chapter-list a" \
    --image-selector "div.page img" \
    --processes 4
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image
from tqdm import tqdm


DEFAULT_CHAPTER_PATTERN = re.compile(
    r"(chapter|ch\.?|第\s*\d+\s*[话話章]|\b\d{1,4}\b)", re.IGNORECASE
)
DEFAULT_IMAGE_PATTERN = re.compile(r"\.(jpe?g|png|gif|webp)", re.IGNORECASE)
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


@dataclass
class ChapterInfo:
    """Information about an individual chapter to download."""

    title: str
    url: str


class MangaScraper:
    """Scraper capable of downloading manga chapters and exporting PDFs."""

    def __init__(
        self,
        root_url: str,
        output_dir: Path,
        chapter_selector: Optional[str] = None,
        chapter_pattern: Optional[re.Pattern[str]] = None,
        image_selector: Optional[str] = None,
        image_pattern: Optional[re.Pattern[str]] = None,
        timeout: int = 20,
    ) -> None:
        self.root_url = root_url
        self.output_dir = output_dir
        self.chapter_selector = chapter_selector
        self.chapter_pattern = chapter_pattern or DEFAULT_CHAPTER_PATTERN
        self.image_selector = image_selector
        self.image_pattern = image_pattern or DEFAULT_IMAGE_PATTERN
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, processes: int = 4) -> None:
        """Discover chapters and download them using a worker pool."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        manga_title = self._discover_manga_title()
        chapters = self._discover_chapters()

        if not chapters:
            raise RuntimeError(
                "No chapters were discovered. Adjust the chapter selector/pattern."
            )

        self._download_chapters_concurrently(manga_title, chapters, processes)

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------
    def _fetch_soup(self, url: str) -> BeautifulSoup:
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")

    def _discover_manga_title(self) -> str:
        soup = self._fetch_soup(self.root_url)
        title_text = soup.title.string if soup.title else "manga"
        title_text = title_text.strip() if title_text else "manga"
        # Remove common separators ("-", "|") to isolate the manga name.
        for sep in ("-", "|"):
            if sep in title_text:
                title_text = title_text.split(sep)[0].strip()
        return _sanitize_filename(title_text or "manga")

    def _discover_chapters(self) -> List[ChapterInfo]:
        soup = self._fetch_soup(self.root_url)
        anchors: Iterable = (
            soup.select(self.chapter_selector)
            if self.chapter_selector
            else soup.find_all("a", href=True)
        )

        seen = set()
        chapters: List[ChapterInfo] = []

        for anchor in anchors:
            href = anchor.get("href")
            text = anchor.get_text(strip=True)
            if not href:
                continue

            full_url = urljoin(self.root_url, href)
            if not self._is_same_domain(full_url):
                continue

            link_text = text or href
            if not self.chapter_pattern.search(link_text):
                continue

            normalized_url = self._normalize_url(full_url)
            if normalized_url in seen:
                continue
            seen.add(normalized_url)

            chapter_title = link_text
            chapters.append(ChapterInfo(title=chapter_title, url=normalized_url))

        chapters = self._sort_chapters(chapters)
        return chapters

    def _is_same_domain(self, url: str) -> bool:
        root_domain = urlparse(self.root_url).netloc
        return urlparse(url).netloc == root_domain

    def _normalize_url(self, url: str) -> str:
        parsed = urlparse(url)
        normalized_path = re.sub(r"/+$", "", parsed.path)
        return parsed._replace(path=normalized_path).geturl()

    def _sort_chapters(self, chapters: Sequence[ChapterInfo]) -> List[ChapterInfo]:
        def sort_key(chapter: ChapterInfo) -> Tuple[int, str]:
            numbers = [int(num) for num in re.findall(r"\d+", chapter.title)]
            primary = numbers[0] if numbers else sys.maxsize
            return (primary, chapter.title)

        return sorted(chapters, key=sort_key)

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------
    def _download_chapters_concurrently(
        self, manga_title: str, chapters: Sequence[ChapterInfo], processes: int
    ) -> None:
        tasks = [
            (
                manga_title,
                chapter,
                self.image_selector,
                self.image_pattern,
                self.output_dir,
                self.timeout,
                self.session.headers,
            )
            for chapter in chapters
        ]

        if processes <= 1:
            for task in tqdm(tasks, desc="Chapters", unit="chapter"):
                _download_single_chapter(task)
            return

        with mp.get_context("spawn").Pool(processes) as pool:
            for _ in tqdm(
                pool.imap_unordered(_download_single_chapter, tasks),
                total=len(tasks),
                desc="Chapters",
                unit="chapter",
            ):
                pass


def _download_single_chapter(task: Tuple) -> None:
    (
        manga_title,
        chapter,
        image_selector,
        image_pattern,
        output_dir,
        timeout,
        headers,
    ) = task

    session = requests.Session()
    session.headers.update(headers)

    soup = _fetch_soup_with_session(session, chapter.url, timeout)
    chapter_title = _extract_chapter_title(soup, chapter.title)
    image_urls = _extract_image_urls(soup, chapter.url, image_selector, image_pattern)

    if not image_urls:
        raise RuntimeError(f"No images found for chapter {chapter.url}")

    images = list(_download_images(session, image_urls, timeout))
    pdf_path = output_dir / f"{manga_title}_{chapter_title}.pdf"
    _images_to_pdf(images, pdf_path)


def _fetch_soup_with_session(
    session: requests.Session, url: str, timeout: int
) -> BeautifulSoup:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def _extract_chapter_title(soup: BeautifulSoup, fallback: str) -> str:
    title_candidates = [
        soup.find("h1"),
        soup.find("h2"),
        soup.title,
    ]
    for candidate in title_candidates:
        if candidate and candidate.get_text(strip=True):
            return _sanitize_filename(candidate.get_text(strip=True))
    return _sanitize_filename(fallback)


def _extract_image_urls(
    soup: BeautifulSoup,
    base_url: str,
    image_selector: Optional[str],
    image_pattern: re.Pattern[str],
) -> List[str]:
    elements = soup.select(image_selector) if image_selector else soup.find_all("img")
    urls = []
    for element in elements:
        src = element.get("data-src") or element.get("data-original") or element.get("src")
        if not src:
            continue
        absolute = urljoin(base_url, src)
        if not image_pattern.search(absolute):
            continue
        urls.append(absolute)
    # Ensure deterministic ordering by removing duplicates while preserving order.
    seen = set()
    ordered_urls = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            ordered_urls.append(url)
    return ordered_urls


def _download_images(
    session: requests.Session, image_urls: Sequence[str], timeout: int
) -> Iterable[Image.Image]:
    for url in image_urls:
        with session.get(url, timeout=timeout, stream=True) as response:
            response.raise_for_status()
            content = response.content
        with BytesIO(content) as buffer:
            image = Image.open(buffer)
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            else:
                image = image.copy()
        yield image


def _images_to_pdf(images: Sequence[Image.Image], pdf_path: Path) -> None:
    if not images:
        raise ValueError("No images to save")

    first, *rest = images
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    first.save(pdf_path, "PDF", save_all=True, append_images=rest)


def _sanitize_filename(name: str) -> str:
    name = re.sub(r"\s+", " ", name)
    return re.sub(r"[^\w\- .]", "_", name).strip()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_url", help="Root URL of the manga series")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("downloads"),
        help="Directory to store the generated PDFs (default: downloads)",
    )
    parser.add_argument(
        "--chapter-selector",
        help="CSS selector to locate chapter links on the root page",
    )
    parser.add_argument(
        "--chapter-pattern",
        help="Regex pattern to identify chapter links based on their text",
    )
    parser.add_argument(
        "--image-selector",
        help="CSS selector to locate images on a chapter page",
    )
    parser.add_argument(
        "--image-pattern",
        help="Regex pattern to filter image URLs (default targets common formats)",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=4,
        help="Number of worker processes to use (default: 4)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="Request timeout in seconds (default: 20)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    chapter_pattern = (
        re.compile(args.chapter_pattern, re.IGNORECASE) if args.chapter_pattern else None
    )
    image_pattern = (
        re.compile(args.image_pattern, re.IGNORECASE) if args.image_pattern else None
    )

    scraper = MangaScraper(
        root_url=args.root_url,
        output_dir=args.output,
        chapter_selector=args.chapter_selector,
        chapter_pattern=chapter_pattern,
        image_selector=args.image_selector,
        image_pattern=image_pattern,
        timeout=args.timeout,
    )
    scraper.run(processes=args.processes)


if __name__ == "__main__":
    main()
