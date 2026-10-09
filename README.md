# PhD Programme RAG

A runnable, source-backed RAG prototype for discovering PhD research leads from official university graduate/admissions pages. It starts with a curated registry of 100 major research universities across ten countries, downloads only the pages in that registry, and records the exact source URL and fetch time with every chunk.

This is deliberately **not** an admissions decision engine. It returns research leads and citations, never an admission probability or a claim that should be checked against the official page.

## Why FAISS here—and when not to use it

The requested corpus is roughly 100 public admissions/research pages. FAISS is the best first database choice for that scale because it is local, free, fast, transparent, and has no hosting or data-retention concern. `IndexFlatIP` with L2-normalized OpenAI embeddings gives exact cosine-similarity retrieval; it is simple to inspect and rebuild.

| Situation | Recommended store | Reason |
| --- | --- | --- |
| This prototype: ~100–10,000 chunks, single user | **FAISS** | Minimal operational overhead; exact local search. |
| Existing relational product with filters and audit needs | **Postgres + pgvector** | Joins, access control, structured metadata filters, transactions. |
| Managed production RAG with filters and growth | **Qdrant Cloud** | Server-side payload filters, hybrid search, snapshots, and scalable operations. |
| Large managed multi-tenant search | **Pinecone** | Fully managed scaling and namespace isolation. |

The code already stores country, university, URL, focus, and fetch time as chunk metadata. With FAISS, country filtering is applied after over-fetching; move to pgvector/Qdrant before relying on complex filters, many users, incremental writes, or millions of chunks.

## What is in the seed registry

`data/universities.json` contains 10 sources in each of: Australia, Canada, China, France, Germany, India, Japan, Netherlands, United Kingdom, and United States. The selection is a starting list of major research universities, **not a claim of a current ranking**. Each URL targets an official university graduate, doctoral, or admissions page. At ingestion, the downloaded official text—not a hand-authored university profile—is what enters the vector database.

University websites change frequently. Review the registry and re-ingest before any consequential use. Failed or blocked pages are retained in `artifacts/manifest.json` so they cannot quietly disappear from the corpus.

## Architecture

```text
Official URLs (100) -> requests + HTML text extraction -> RecursiveCharacterTextSplitter
       -> OpenAI embeddings -> normalized FAISS index + JSONL metadata
       -> retrieve relevant chunks -> Responses API with constrained prompt -> cited research leads
```

The pipeline follows the proposed RAG pattern while keeping every layer replaceable:

1. Fetch public official pages with an identifiable user-agent and cache raw text.
2. Split text with a recursive character splitter (1,200 chars, 180-char overlap).
3. Embed chunks using OpenAI embeddings and store normalized vectors in FAISS.
4. Retrieve relevant passages, optionally apply the candidate's preferred-country metadata filter, then ask the language model to synthesize only the retrieved evidence.
5. Print the retrieved official source URLs with similarity scores after every answer.

## Quick start

Requires Python 3.9+ and an OpenAI API key.

```bash
cd /Users/vipulgupta/work/mvp-saas/phd-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add OPENAI_API_KEY
```

The documented commands use `PYTHONPATH=src python -m phd_rag.cli`, so they work without a package install. With a recent pip that supports PEP 660, `pip install -e .` also installs a `phd-rag` command.

Start with two pages to validate credentials, network access, and extraction:

```bash
PYTHONPATH=src python -m phd_rag.cli ingest --limit 2
PYTHONPATH=src python -m phd_rag.cli ask "Which official pages mention doctoral research applications?"
```

Build the full 100-source index (be considerate of university sites; the default delay is 0.4 seconds):

```bash
PYTHONPATH=src python -m phd_rag.cli ingest
```

Ask for a recommendation using the included candidate profile:

```bash
PYTHONPATH=src python -m phd_rag.cli recommend --profile data/example_candidate.json
```

Useful operations:

```bash
PYTHONPATH=src python -m phd_rag.cli sources # inspect all 100 official source URLs
PYTHONPATH=src python -m phd_rag.cli ingest --country Canada --country Germany
PYTHONPATH=src python -m phd_rag.cli ask "What funding or supervisor details should I verify for an AI PhD?"
PYTHONPATH=src python -m unittest discover -s tests
```

## Candidate profiles

`data/example_candidate.json` is the input contract. The required fields are `field`, `degree`, and `research_interests`; the rest make retrieval and the final advice more useful. Keep it free of unnecessary personal data. API responses, raw pages, and generated indexes are local and ignored by Git.

## Data-quality and safety guardrails

- Only ingest official university domains from the reviewed registry. Do not use social posts, aggregators, or ranking pages as admission evidence.
- The loader does not bypass robots controls, authentication, paywalls, or bot challenges.
- The answer prompt prohibits invented funding, eligibility, deadline, supervisor, visa, and admission-likelihood claims.
- Treat citations as a verification checklist; programme requirements are volatile and often department-specific.
- Before production, add a URL-review workflow, periodic refreshes, department/faculty pages, multilingual extraction, PII minimization, evaluation queries, and human review.

## Configuration

The default models are set in `.env.example`:

```text
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-5-mini
```

The OpenAI embedding model can be changed only when rebuilding the index. The query embedding must match the index embedding model. Model availability and pricing should be verified in the [official OpenAI documentation](https://developers.openai.com/api/docs/models) before deployment.
