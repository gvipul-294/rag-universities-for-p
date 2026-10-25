# PhD Programme RAG

> New here? [HOW_IT_WORKS.md](HOW_IT_WORKS.md) gives a one-page walkthrough of each step and the file that does it.

A runnable, source-backed RAG prototype for discovering PhD research leads from official university graduate/admissions pages. It starts with a curated registry of 100 major research universities across ten countries, downloads only the pages in that registry, and records the exact source URL and fetch time with every chunk.

This is deliberately **not** an admissions decision engine. It returns research leads and citations, never an admission probability or a claim that should be checked against the official page.

## What this project does

- Builds a local search index from public, official graduate, doctoral, and admissions pages.
- Matches natural-language questions or a structured candidate profile against that evidence.
- Produces a short list of research leads, with the official URLs used to support the answer.
- Keeps generated data local: raw page text, FAISS vectors, chunks, and an ingestion manifest are never committed.

It does **not** scrape behind logins, bypass website restrictions, select a supervisor, calculate an admission chance, or replace checking the linked departmental pages.

## Why FAISS here—and when not to use it

The requested corpus is roughly 100 public admissions/research pages. FAISS is the best first database choice for that scale because it is local, free, fast, transparent, and has no hosting or data-retention concern. `IndexFlatIP` with L2-normalized embeddings gives exact cosine-similarity retrieval; it is simple to inspect and rebuild.

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
       -> local fastembed embeddings (bge-small-en-v1.5) -> normalized FAISS index + JSONL metadata
       -> retrieve relevant chunks -> either:
          - an OpenAI-compatible chat API (Groq by default) for `ask` / `recommend`, or
          - MCP tools + local Ollama agent for `agent`
```

The pipeline follows the proposed RAG pattern while keeping every layer replaceable:

1. Fetch public official pages with an identifiable user-agent and cache raw text.
2. Split text with a recursive character splitter (1,200 chars, 180-char overlap).
3. Embed chunks locally with fastembed (`BAAI/bge-small-en-v1.5`) and store normalized vectors in FAISS.
4. Retrieve relevant passages, optionally apply the candidate's preferred-country metadata filter, then synthesize only the retrieved evidence.
5. Use either the hosted open-weight answer path (Groq) or the MCP-backed local agent path, and print the official source URLs with similarity scores after every answer.

### Retrieval lifecycle

```text
1. Registry     data/universities.json
2. Ingestion    fetch -> extract visible text -> cache raw text
3. Indexing     split -> embed -> normalize -> FAISS IndexFlatIP
4. Retrieval    embed query -> top-k similarity search -> optional country filter
5. Answering    retrieved context + guardrailed prompt -> cited research leads
```

The index is intentionally rebuilt from source pages rather than incrementally mutating embeddings. That makes runs reproducible and makes it easier to replace stale or failed URLs. An ingestion run replaces the files in `artifacts/`; with the default local embeddings it does not send the university content anywhere.

## Project layout

| Path | Purpose |
| --- | --- |
| `data/universities.json` | Reviewed seed list of official university URLs and country metadata. |
| `data/example_candidate.json` | Example input profile for a recommendation query. |
| `src/phd_rag/ingest.py` | Web extraction, recursive chunking, and FAISS persistence. |
| `src/phd_rag/embeddings.py` | Embedding backends: local fastembed (default) or the OpenAI embeddings API. |
| `src/phd_rag/retrieve.py` | Similarity search, metadata narrowing, prompt construction, and answer generation. |
| `src/phd_rag/mcp_server.py` | MCP server exposing source search, candidate mapping, and index-status tools. |
| `src/phd_rag/agentic.py` | Ollama tool-calling loop that invokes those tools through MCP. |
| `src/phd_rag/api.py` | FastAPI HTTP boundary for the PhD Finder web frontend. |
| `src/phd_rag/cli.py` | The `sources`, `ingest`, `ask`, `recommend`, and `agent` commands. |
| `docs/api.md` | Browser-facing API contract, cURL examples, and remote-server checks. |
| `artifacts/` | Generated FAISS index, chunk metadata, and the latest manifest; ignored by Git. |
| `raw_pages/` | Cached page text for inspection and debugging; ignored by Git. |

## Quick start

Requires Python 3.10+ and a free [Groq API key](https://console.groq.com/keys) for answer generation. Embeddings run locally and need no key. Python 3.10+ is required by the MCP Python SDK used for the agentic path.

```bash
cd /Users/vipulgupta/work/mvp-saas/phd-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set LLM_API_KEY to your Groq key
```

The documented commands use `PYTHONPATH=src python -m phd_rag.cli`, so they work without a package install. With a recent pip that supports PEP 660, `pip install -e .` also installs a `phd-rag` command.

Never commit `.env`; it is ignored by default. The CLI reads `LLM_API_KEY` and the model settings from that file when it starts. The first `ingest` downloads the embedding model (under 100 MB) from Hugging Face and caches it.

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

## HTTP API for the web frontend

The web frontend should call this API, never the LLM provider or FAISS directly. The API exposes the two flows shown in the PhD Finder mockup: a free-form question and a structured candidate match. It returns the generated answer and the official-source passages needed to render linked evidence cards.

```bash
# An index must exist before asking questions.
PYTHONPATH=src python -m phd_rag.cli ingest --limit 2
PYTHONPATH=src uvicorn phd_rag.api:app --host 127.0.0.1 --port 8000

# Health/index check, then a browser-equivalent request.
curl --fail-with-body http://127.0.0.1:8000/api/v1/health
curl --fail-with-body --request POST http://127.0.0.1:8000/api/v1/ask \
  --header 'Content-Type: application/json' \
  --data '{"question":"What IELTS score do UK universities ask for a PhD?","countries":["United Kingdom"]}'
```

Open `http://127.0.0.1:8000/docs` while the server is running to inspect and test the generated OpenAPI documentation. See [the API guide](docs/api.md) for the complete request/response contract, the candidate-match cURL example, and EC2 deployment checks. For how production runs (the EC2 service, Vercel, the API token and memory limits), see [docs/deployment.md](docs/deployment.md).

`CORS_ALLOW_ORIGINS` restricts which browser origins can call the service. Add the exact Vercel production URL before deployment. `PHD_RAG_API_TOKEN` is optional for local development, but must be set before a public deployment; a Vercel server-side proxy should hold it, not client-side JavaScript.

### Frontend (Next.js / Vercel)

The Vercel-ready frontend lives in `frontend/`. It includes Ask and Match flows, evidence cards, and a server-side API proxy so browser code never sees the backend bearer token. For local development, start the FastAPI server as above, then:

```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000`. For Vercel, import the GitHub repository and set the project root to `frontend`. Configure `PHD_RAG_API_URL` and `PHD_RAG_API_TOKEN` as server-side Vercel environment variables. Keep the backend API token and LLM API key out of `NEXT_PUBLIC_*` variables and browser code. The API must also be reachable from Vercel over HTTPS; do not point production Vercel at a raw, unauthenticated EC2 HTTP port.

## MCP-backed open-source RAG agent

The `agent` command is an agentic PhD-matching system. It runs an open-weight model locally through Ollama, gives that model a set of MCP tools, and lets the model decide which approved retrieval tool to call before forming a recommendation.

```text
Candidate question/profile
        |
        v
Ollama local model (`mistral` by default)
        |  tool call
        v
MCP client  ->  PhD RAG MCP server  ->  FAISS + local query embedding
        ^             |                     |
        |             +-- official excerpts -+
        +----- tool result / source URLs -----+
        |
        v
Evidence-grounded university matches + citations
```

MCP does not replace the LLM or itself make model calls: it is the standard tool interface between the model-hosting agent and the RAG capabilities. Here the Python agent uses real MCP `list_tools` and `call_tool` operations against the project’s in-process MCP server. The same server can also be launched over standard input/output for another MCP-capable host.

By default, the agent uses Ollama's `mistral` model. Ollama documents it as a 7B Mistral model distributed under an Apache license; the current default tag is about 4.4 GB. See the [Ollama Mistral library page](https://ollama.com/library/mistral) for current tags and hardware expectations.

### MCP tools available to the agent

| Tool | When the agent should use it | Output |
| --- | --- | --- |
| `search_official_university_sources` | A user asks a free-form PhD question. | Official excerpts, source URLs, countries, scores, and fetch times. |
| `map_candidate_to_universities` | A structured candidate profile is supplied. | Candidate-aware retrieved sources, narrowed by preferred country where provided. |
| `get_rag_index_status` | The agent needs to check whether a current index exists or inspect ingestion failures. | Latest manifest and index availability. |

The system prompt requires at least one retrieval call before a university recommendation. It treats page excerpts as untrusted data, so instructions embedded in university web pages cannot alter the agent’s behavior.

### Prerequisites

- An existing local FAISS index, built with `ingest`.
- [Ollama](https://ollama.com/) installed and running locally. The final answer is generated by the local model; nothing is sent to a hosted LLM.

Pull the default model once, then run the agent:

```bash
ollama pull mistral

# Ask a direct question.
PYTHONPATH=src python -m phd_rag.cli agent "Which Canadian AI PhD sources should I investigate?" --country Canada

# Or ask with the structured candidate profile.
PYTHONPATH=src python -m phd_rag.cli agent --profile data/example_candidate.json
```

Choose a different locally available Ollama model without changing code:

```bash
PYTHONPATH=src python -m phd_rag.cli agent "Find PhD options in robotics" --model mistral
```

An agent run prints its MCP activity, model name, final answer, and retrieved official sources. The agent is intentionally constrained to retrieval tools instead of allowing arbitrary web browsing. That keeps university mapping tied to the reviewed registry, lowers prompt-injection risk from retrieved webpages, and preserves source citations.

### Run the RAG tools as a standalone MCP server

This is useful when an MCP-capable desktop application or another local agent should use the same approved PhD-research tools:

```bash
PYTHONPATH=src python -m phd_rag.mcp_server
```

The server uses the standard `stdio` transport. A typical host configuration should launch the project environment. With the default local embeddings it needs no API key:

```json
{
  "mcpServers": {
    "phd-rag": {
      "command": "/absolute/path/to/phd-rag/.venv/bin/python",
      "args": ["-m", "phd_rag.mcp_server"],
      "cwd": "/absolute/path/to/phd-rag"
    }
  }
}
```

Do not expose this server on a public network without authentication and access controls: it can query your local index and, if `EMBEDDING_PROVIDER=openai`, consumes embedding API usage when a tool is called.

## Command reference

```bash
# Inspect all 100 official source URLs before ingesting.
PYTHONPATH=src python -m phd_rag.cli sources

# Index only selected countries or run a small smoke test.
PYTHONPATH=src python -m phd_rag.cli ingest --country Canada --country Germany
PYTHONPATH=src python -m phd_rag.cli ingest --limit 2 --delay 1

# Ask an evidence-backed question after an index exists.
PYTHONPATH=src python -m phd_rag.cli ask "What funding or supervisor details should I verify for an AI PhD?"
PYTHONPATH=src python -m phd_rag.cli ask "What IELTS score is required?" --country "United Kingdom"

# Recommend PhD research leads for a JSON profile.
PYTHONPATH=src python -m phd_rag.cli recommend --profile data/example_candidate.json --top-k 12

# Use the Ollama + MCP agentic matcher.
PYTHONPATH=src python -m phd_rag.cli agent "Find funded computer-science PhD research leads" --country Germany

# Run the registry validation.
PYTHONPATH=src python -m unittest discover -s tests
```

Retrieval keeps at most 3 chunks per university (`MAX_CHUNKS_PER_UNIVERSITY` in `retrieve.py`) so one long page cannot crowd out others. `--delay` spaces page requests to be considerate of university websites. `--top-k` controls how many chunks are supplied to the answer model; increasing it can improve coverage but also increases answer-context size and cost.

## Candidate profiles

`data/example_candidate.json` is the input contract. The required fields are `field`, `degree`, and `research_interests`; the rest make retrieval and the final advice more useful. Keep it free of unnecessary personal data. API responses, raw pages, and generated indexes are local and ignored by Git.

```json
{
  "field": "Computer Science",
  "degree": "Degree, grades, and relevant preparation",
  "research_interests": ["machine learning", "natural language processing"],
  "citizenship_or_residency": "Optional",
  "preferred_countries": ["Canada", "Germany"],
  "budget_or_funding_need": "Optional",
  "test_scores": "Optional",
  "constraints": "Optional"
}
```

`preferred_countries` narrows returned passages after FAISS over-fetches candidates. It is a convenience filter for this small corpus, not a substitute for database-native filtering in a larger deployment.

## Index outputs and provenance

After a successful `ingest` command, inspect these local outputs:

| File | Contents | Why it matters |
| --- | --- | --- |
| `artifacts/university_phd.faiss` | The normalized dense-vector FAISS index. | Required for search. |
| `artifacts/chunks.jsonl` | One text chunk per line plus URL, university, country, focus, and fetch time. | Makes retrieval explainable and auditable. |
| `artifacts/manifest.json` | Build timestamp, model name, source/chunk counts, and failures. | Reveals blocked or changed sources. |
| `raw_pages/*.txt` | Visible text extracted from each successful page. | Lets you review extraction quality without re-fetching. |

If a website blocks a request, change its registry entry to another official, publicly accessible page and rebuild. Do not silently substitute a third-party source.

## Models and how to change them

The system uses two models by default, both open-weight:

| Role | Default | Where it runs | Setting |
| --- | --- | --- | --- |
| Embeddings (indexing and every query) | `BAAI/bge-small-en-v1.5` via [fastembed](https://github.com/qdrant/fastembed) | Locally on CPU, ~360 MB RAM | `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL` |
| Answer writing for `ask` / `recommend` | `openai/gpt-oss-120b` on Groq | Groq's OpenAI-compatible API | `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` |
| Answer writing for `agent` | `mistral` via Ollama | Locally, needs several GB of RAM | `OLLAMA_BASE_URL`, `OLLAMA_MODEL` |

```text
EMBEDDING_PROVIDER=fastembed
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=your_groq_api_key
LLM_MODEL=openai/gpt-oss-120b
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral
```

The answer step uses the Chat Completions API, so any OpenAI-compatible provider works (OpenRouter, Together, Ollama's `/v1` endpoint, or OpenAI itself with `LLM_BASE_URL` left empty). Provider model lists change; if a call returns `model_not_found`, pick another model from the provider's list.

To use OpenAI embeddings instead, set `EMBEDDING_PROVIDER=openai`, `OPENAI_API_KEY`, and optionally `EMBEDDING_MODEL=text-embedding-3-small`.

Keep the same embedding model for indexing and querying. When you change `EMBEDDING_PROVIDER` or `EMBEDDING_MODEL`, rebuild `artifacts/` by running `ingest` again; the CLI refuses to query an index built with a different model. The default splitter uses 1,200-character chunks with 180-character overlap. Tune chunk size, overlap, and `top-k` using representative candidate questions—not intuition alone.

`OLLAMA_BASE_URL` is the local Ollama server endpoint, and `OLLAMA_MODEL` is the model name installed in that server. The agent uses Ollama's `POST /api/chat` interface with non-streaming output and a low temperature for constrained query refinement and evidence-grounded synthesis. It has no Ollama API key configuration because the default endpoint is local.

## Cost, quality, and freshness

- Local embeddings cost nothing but CPU time; start with `--limit 2` to check extraction before a full ingestion.
- Answer latency and any provider rate limits grow with the retrieved context. Keep `top-k` modest and measure answer quality on a set of real questions.
- Re-run ingestion on a schedule appropriate to admissions cycles and whenever `manifest.json` reports failures.
- The seed list is not a ranking system. A university URL can be valid while still being too general for a field-specific recommendation; add official faculty, department, funding, and programme pages after review.
- Treat retrieved sources as leads to verify. Admission requirements, funding, deadlines, and supervisor availability can change quickly.

## Data-quality and safety guardrails

- Only ingest official university domains from the reviewed registry. Do not use social posts, aggregators, or ranking pages as admission evidence.
- The loader does not bypass robots controls, authentication, paywalls, or bot challenges.
- The answer prompt prohibits invented funding, eligibility, deadline, supervisor, visa, and admission-likelihood claims.
- Treat citations as a verification checklist; programme requirements are volatile and often department-specific.
- Before production, add a URL-review workflow, periodic refreshes, department/faculty pages, multilingual extraction, PII minimization, evaluation queries, and human review.

## Troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `LLM_API_KEY is missing` | No answer-model key in `.env` or environment. | Copy `.env.example` to `.env` and set `LLM_API_KEY`. |
| `model_not_found` from the answer API | The provider retired or never offered `LLM_MODEL`. | List the provider's models and set `LLM_MODEL` to an available one. |
| `Index not found` | Query ran before ingestion. | Run `ingest` first. |
| A university is in `failures` | The page moved, blocked the request, or needs JavaScript. | Review the URL and use a different official public page. |
| Results feel too broad | The page corpus is generic or the query lacks a field. | Add field-specific official pages and provide a more detailed profile. |
| `Index was built with ...` error | `EMBEDDING_MODEL` changed since the last `ingest`. | Rebuild the index with `ingest` or restore the previous model. |
| The local agent cannot reach Ollama | Ollama is stopped, unavailable at the configured URL, or its model is missing. | Start Ollama, run `ollama pull mistral`, or change `OLLAMA_BASE_URL`/`OLLAMA_MODEL`. |
| The agent returns insufficient evidence | The matching excerpts failed score or source-diversity gates. | Add/review field-specific official pages, rebuild the index, then tune the evidence thresholds using labelled queries if appropriate. |
| `mcp` import error | Dependencies were installed with an older Python environment. | Recreate the environment with Python 3.10+ and run `pip install -r requirements.txt`. |

## Production next steps

For a decision-support product, add server-side metadata filters, a reviewed content pipeline, scheduled refreshes, retrieval/answer evaluations, access controls, observability, rate limiting, and a human admissions-review workflow. Move from FAISS to pgvector or Qdrant before running a shared, multi-user service or relying on complex filter combinations.
