from __future__ import annotations

import json
from dataclasses import dataclass

import faiss
from openai import OpenAI

from .embeddings import embed_query
from .ingest import CHUNKS_FILE, INDEX_FILE, MANIFEST_FILE
from .models import CandidateProfile
from .settings import Settings, require_llm_key


@dataclass
class RetrievedChunk:
    score: float
    record: dict[str, str]


SYSTEM_PROMPT = """You are a cautious PhD programme research assistant.
Use only the supplied official-university context. Recommend at most five well-matched
options, explain the fit and uncertainties, and include citations in the exact form
[University — source URL]. Never claim admission likelihood or invent programme,
supervisor, funding, eligibility, deadline, or visa facts. Say when a detail must be
verified on the linked source. Recommendations are research leads, not admissions advice."""


def _load_chunks(settings: Settings) -> list[dict[str, str]]:
    path = settings.artifacts_dir / CHUNKS_FILE
    if not path.exists():
        raise RuntimeError("Index not found. Run `phd-rag ingest` first.")
    # Split on "\n" only: str.splitlines() also breaks on characters such as U+0085
    # and U+2028, which json.dumps(ensure_ascii=False) leaves unescaped inside records.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]


def _check_index_model(settings: Settings) -> None:
    """Query and index vectors must come from the same embedding model."""
    manifest_path = settings.artifacts_dir / MANIFEST_FILE
    if not manifest_path.exists():
        return
    built_with = json.loads(manifest_path.read_text(encoding="utf-8")).get("embedding_model")
    if built_with and built_with != settings.embedding_model:
        raise RuntimeError(
            f"Index was built with {built_with!r} but EMBEDDING_MODEL is {settings.embedding_model!r}. "
            "Re-run `ingest` or change EMBEDDING_MODEL back."
        )


def search(
    settings: Settings,
    query: str,
    top_k: int = 12,
    countries: list[str] | None = None,
) -> list[RetrievedChunk]:
    index_path = settings.artifacts_dir / INDEX_FILE
    if not index_path.exists():
        raise RuntimeError("Index not found. Run `phd-rag ingest` first.")
    _check_index_model(settings)
    query_vector = embed_query(settings, query)
    index = faiss.read_index(str(index_path))
    # Over-fetch before applying metadata filters, as IndexFlatIP has no native
    # filter support. This is adequate for the small local corpus; use a vector
    # service with server-side filters once the corpus or traffic grows.
    search_k = min(max(top_k * 8, top_k), index.ntotal)
    scores, positions = index.search(query_vector, search_k)
    chunks = _load_chunks(settings)
    results = [
        RetrievedChunk(score=float(score), record=chunks[int(position)])
        for score, position in zip(scores[0], positions[0])
        if position >= 0
    ]
    if countries:
        wanted = {country.casefold() for country in countries}
        filtered = [item for item in results if item.record["country"].casefold() in wanted]
        if filtered:
            results = filtered
    return results[:top_k]


def _context(results: list[RetrievedChunk]) -> str:
    sections = []
    for result in results:
        item = result.record
        sections.append(
            f"SOURCE: {item['university']} | {item['country']} | {item['url']}\n"
            f"RETRIEVAL SCORE: {result.score:.3f}\n{item['text']}"
        )
    return "\n\n---\n\n".join(sections)


def answer(
    settings: Settings,
    question: str,
    top_k: int = 12,
    countries: list[str] | None = None,
) -> tuple[str, list[RetrievedChunk]]:
    results = search(settings, question, top_k=top_k, countries=countries)
    # Chat Completions is the API every OpenAI-compatible provider supports.
    client = OpenAI(api_key=require_llm_key(), base_url=settings.llm_base_url)
    response = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question:\n{question}\n\nOfficial context:\n{_context(results)}"},
        ],
    )
    return response.choices[0].message.content or "", results


def profile_to_question(profile: CandidateProfile) -> str:
    countries = ", ".join(profile.preferred_countries) or "No preference stated"
    interests = ", ".join(profile.research_interests)
    return (
        "Recommend PhD research leads for this candidate, using the official pages.\n"
        f"Field: {profile.field}\nDegree/background: {profile.degree}\n"
        f"Research interests: {interests}\nPreferred countries: {countries}\n"
        f"Citizenship/residency: {profile.citizenship_or_residency or 'Not provided'}\n"
        f"Funding need: {profile.budget_or_funding_need or 'Not provided'}\n"
        f"Test scores: {profile.test_scores or 'Not provided'}\n"
        f"Constraints: {profile.constraints or 'Not provided'}"
    )
