from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from tqdm import tqdm

from .models import UniversitySource
from .settings import Settings, require_api_key

USER_AGENT = "phd-rag-research-bot/0.1 (educational; contact: local-user)"
INDEX_FILE = "university_phd.faiss"
CHUNKS_FILE = "chunks.jsonl"
MANIFEST_FILE = "manifest.json"


def load_registry(path: Path) -> list[UniversitySource]:
    return [UniversitySource.model_validate(item) for item in json.loads(path.read_text())]


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
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        tag.decompose()
    return " ".join(soup.stripped_strings)


def split_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=180,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [chunk for chunk in splitter.split_text(text) if len(chunk.strip()) >= 100]


def _embed(client: OpenAI, model: str, texts: list[str]) -> np.ndarray:
    vectors: list[list[float]] = []
    for offset in range(0, len(texts), 96):
        response = client.embeddings.create(model=model, input=texts[offset : offset + 96])
        vectors.extend(item.embedding for item in response.data)
    matrix = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(matrix)
    return matrix


def build_index(
    settings: Settings,
    countries: set[str] | None = None,
    limit: int | None = None,
    delay_seconds: float = 0.4,
) -> dict[str, object]:
    """Download configured official pages, chunk them, and replace the local index."""
    require_api_key()
    sources = load_registry(settings.registry_path)
    if countries:
        wanted = {country.casefold() for country in countries}
        sources = [item for item in sources if item.country.casefold() in wanted]
    if limit is not None:
        sources = sources[:limit]
    if not sources:
        raise ValueError("No sources matched the requested country filter.")

    settings.raw_pages_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for source in tqdm(sources, desc="Fetching official pages"):
        try:
            text = extract_page_text(str(source.url))
            (settings.raw_pages_dir / _safe_filename(source)).write_text(text, encoding="utf-8")
            fetched_at = datetime.now(timezone.utc).isoformat()
            for chunk_number, chunk in enumerate(split_text(text)):
                records.append(
                    {
                        "id": f"{_safe_filename(source)}:{chunk_number}",
                        "text": chunk,
                        "country": source.country,
                        "university": source.university,
                        "url": str(source.url),
                        "focus": source.focus,
                        "fetched_at": fetched_at,
                    }
                )
        except requests.RequestException as error:
            failures.append({"university": source.university, "url": str(source.url), "error": str(error)})
        time.sleep(delay_seconds)
    if not records:
        raise RuntimeError("No pages could be converted to usable text; inspect the failures in manifest.json.")

    client = OpenAI()
    vectors = _embed(client, settings.embedding_model, [record["text"] for record in records])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(settings.artifacts_dir / INDEX_FILE))
    with (settings.artifacts_dir / CHUNKS_FILE).open("w", encoding="utf-8") as handle:
        handle.write("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n")
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": settings.embedding_model,
        "sources_requested": len(sources),
        "sources_succeeded": len(sources) - len(failures),
        "chunks": len(records),
        "failures": failures,
    }
    (settings.artifacts_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
