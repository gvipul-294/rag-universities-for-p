"""Ingestion pipeline: select sources -> fetch pages -> chunk -> embed -> write index.

Each stage is a separate function so it can be run, tested, or replaced on its own.
`build_index` wires them together for the `ingest` command.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .embeddings import embed_passages
from .models import UniversitySource
from .settings import Settings

log = logging.getLogger(__name__)

USER_AGENT = "phd-rag-research-bot/0.1 (educational; contact: local-user)"
INDEX_FILE = "university_phd.faiss"
CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "manifest.json"
FETCH_LOG_FILE = "fetch_log.jsonl"
# Pages with less visible text than this are usually JavaScript-rendered shells.
THIN_PAGE_CHARS = 2000


@dataclass
class FetchedPage:
    source: UniversitySource
    text: str
    fetched_at: str


# --- Stage 1: sources -------------------------------------------------------


def load_registry(path: Path) -> list[UniversitySource]:
    return [UniversitySource.model_validate(item) for item in json.loads(path.read_text())]


def select_sources(
    sources: list[UniversitySource],
    countries: set[str] | None = None,
    limit: int | None = None,
) -> list[UniversitySource]:
    if countries:
        wanted = {country.casefold() for country in countries}
        sources = [item for item in sources if item.country.casefold() in wanted]
    if limit is not None:
        sources = sources[:limit]
    if not sources:
        raise ValueError("No sources matched the requested country filter.")
    return sources


# --- Stage 2: fetch ---------------------------------------------------------


def _safe_filename(source: UniversitySource) -> str:
    digest = hashlib.sha256(str(source.url).encode()).hexdigest()[:12]
    slug = re.sub(r"[^a-z0-9]+", "-", source.university.lower()).strip("-")
    return f"{source.country.lower()}-{slug}-{digest}.txt"


def extract_page_text(url: str, timeout: int = 30) -> str:
    """Fetch one public, official page and return its visible text.

    This intentionally does not bypass access controls, robots policies, paywalls,
    or JavaScript challenges. Replace blocked entries with an allowed official page.
    """
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    # Parse bytes, not response.text: when a server omits the charset header, requests
    # assumes ISO-8859-1 and garbles UTF-8 pages. BeautifulSoup reads the BOM and <meta charset>.
    soup = BeautifulSoup(response.content, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        tag.decompose()
    return " ".join(soup.stripped_strings)


def _describe_error(error: Exception) -> tuple[str, int | None]:
    """Return a short error kind (e.g. 'HTTP 404', 'ConnectionError') and the HTTP status if any."""
    if isinstance(error, requests.HTTPError) and error.response is not None:
        return f"HTTP {error.response.status_code}", error.response.status_code
    return type(error).__name__, None


def fetch_pages(
    settings: Settings,
    sources: list[UniversitySource],
    delay_seconds: float = 0.4,
) -> tuple[list[FetchedPage], list[dict[str, object]]]:
    """Download each source, logging every outcome as it happens.

    Each result is printed to the log and appended to artifacts/fetch_log.jsonl
    immediately, so the record survives even if a later stage crashes.
    """
    settings.raw_pages_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    pages: list[FetchedPage] = []
    failures: list[dict[str, object]] = []
    total = len(sources)
    with (settings.artifacts_dir / FETCH_LOG_FILE).open("w", encoding="utf-8") as fetch_log:
        for number, source in enumerate(sources, start=1):
            entry: dict[str, object] = {
                "at": datetime.now(timezone.utc).isoformat(),
                "university": source.university,
                "country": source.country,
                "url": str(source.url),
            }
            try:
                text = extract_page_text(str(source.url))
            except Exception as error:  # one bad page must not abort the whole run
                kind, status = _describe_error(error)
                entry.update(ok=False, error_kind=kind, status=status, error=str(error))
                failures.append(entry)
                log.warning("[%d/%d] FAIL %s — %s (%s)", number, total, source.university, kind, source.url)
            else:
                (settings.raw_pages_dir / _safe_filename(source)).write_text(text, encoding="utf-8")
                pages.append(FetchedPage(source=source, text=text, fetched_at=entry["at"]))
                entry.update(ok=True, chars=len(text))
                thin = " (thin page, may need JavaScript)" if len(text) < THIN_PAGE_CHARS else ""
                log.info("[%d/%d] OK   %s — %d chars%s", number, total, source.university, len(text), thin)
            fetch_log.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fetch_log.flush()
            time.sleep(delay_seconds)
    log.info("Fetched %d/%d pages, %d failed", len(pages), total, len(failures))
    return pages, failures


# --- Stage 3: chunk ---------------------------------------------------------


def split_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=180,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [chunk for chunk in splitter.split_text(text) if len(chunk.strip()) >= 100]


def chunk_pages(pages: list[FetchedPage]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for page in pages:
        filename = _safe_filename(page.source)
        for chunk_number, chunk in enumerate(split_text(page.text)):
            records.append(
                {
                    "id": f"{filename}:{chunk_number}",
                    "text": chunk,
                    "country": page.source.country,
                    "university": page.source.university,
                    "url": str(page.source.url),
                    "focus": page.source.focus,
                    "fetched_at": page.fetched_at,
                }
            )
    log.info("Split %d pages into %d chunks", len(pages), len(records))
    return records


# --- Stage 4: embed + write -------------------------------------------------


def write_index(settings: Settings, records: list[dict[str, str]], vectors: np.ndarray) -> None:
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(settings.artifacts_dir / INDEX_FILE))
    with (settings.artifacts_dir / CHUNKS_FILE).open("w", encoding="utf-8") as handle:
        handle.write("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n")


def build_index(
    settings: Settings,
    countries: set[str] | None = None,
    limit: int | None = None,
    delay_seconds: float = 0.4,
) -> dict[str, object]:
    """Download configured official pages, chunk them, and replace the local index."""
    sources = select_sources(load_registry(settings.registry_path), countries, limit)
    pages, failures = fetch_pages(settings, sources, delay_seconds)
    records = chunk_pages(pages)
    if not records:
        raise RuntimeError(f"No pages could be converted to usable text; see artifacts/{FETCH_LOG_FILE}.")

    log.info("Embedding %d chunks with %s", len(records), settings.embedding_model)
    vectors = embed_passages(settings, [record["text"] for record in records])
    write_index(settings, records, vectors)
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "sources_requested": len(sources),
        "sources_succeeded": len(pages),
        "chunks": len(records),
        "failures": failures,
    }
    (settings.artifacts_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Index written to %s", settings.artifacts_dir)
    return manifest
