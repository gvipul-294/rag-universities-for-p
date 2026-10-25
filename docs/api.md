# PhD Finder HTTP API

The browser frontend uses this API instead of calling the LLM provider or FAISS directly. The API keeps the LLM API key (Groq by default) and local index on the backend, then returns an LLM answer plus the official evidence used for it.

## Run locally

Build an index first, then start the server:

```bash
PYTHONPATH=src python -m phd_rag.cli ingest --limit 2
PYTHONPATH=src uvicorn phd_rag.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the generated interactive API documentation.

## Endpoints

### `GET /api/v1/health`

Returns whether both the FAISS index and chunk metadata are present. It does not call the LLM provider and is suitable for a deployment health check.

```bash
curl --fail-with-body http://127.0.0.1:8000/api/v1/health
```

### `GET /api/v1/index/status`

Returns non-secret metadata from the latest ingestion manifest.

```bash
curl --fail-with-body http://127.0.0.1:8000/api/v1/index/status
```

### `POST /api/v1/ask`

Use this endpoint for the **Ask a question** tab in the PDF mockup.

```bash
curl --fail-with-body --request POST http://127.0.0.1:8000/api/v1/ask \
  --header 'Content-Type: application/json' \
  --data '{
    "question": "What IELTS score do UK universities ask for a PhD?",
    "countries": ["United Kingdom"],
    "top_k": 12
  }'
```

### `POST /api/v1/match`

Use this endpoint for the **Match me to programmes** tab. `field`, `degree`, and `research_interests` are required.

```bash
curl --fail-with-body --request POST http://127.0.0.1:8000/api/v1/match \
  --header 'Content-Type: application/json' \
  --data '{
    "field": "Computer Science",
    "degree": "BTech in Computer Science; 3.7/4 GPA",
    "research_interests": ["machine learning", "responsible AI"],
    "budget_or_funding_need": "Needs a funded position",
    "preferred_countries": ["Canada", "Netherlands", "Germany"],
    "top_k": 12
  }'
```

Every successful `ask` and `match` response has this shape:

```json
{
  "answer": "Evidence-grounded LLM response with citations.",
  "sources": [
    {
      "university": "Example University",
      "country": "Canada",
      "url": "https://example.edu/phd",
      "score": 0.8123,
      "excerpt": "Retrieved official-page passage.",
      "fetched_at": "2026-10-21T00:00:00+00:00"
    }
  ],
  "source_count": 1,
  "chat_model": "gpt-5-mini"
}
```

The frontend should render `answer` and show a source card for every item in `sources`. It must link `url` directly and label the result as a research lead, not an admission decision.

## Remote server checks

Only use the public address after the server has been configured to listen on port `8000` and the EC2 security group permits that port from the intended clients. Replace the port if your reverse proxy uses another one.

```bash
export PHD_RAG_API_URL=http://16.171.147.74:8000
curl --fail-with-body "$PHD_RAG_API_URL/api/v1/health"
```

To bind Uvicorn beyond localhost for a reverse proxy or controlled network:

```bash
PYTHONPATH=src uvicorn phd_rag.api:app --host 0.0.0.0 --port 8000
```

Do not expose the raw API to the public internet without setting `PHD_RAG_API_TOKEN`, restricting network access, and placing HTTPS authentication in front of it. For a Vercel frontend, keep the token in a server-side Vercel environment variable and proxy calls through a server route; never send it to browser JavaScript.

When `PHD_RAG_API_TOKEN` is set, protect non-health requests with:

```bash
curl --fail-with-body --request POST "$PHD_RAG_API_URL/api/v1/ask" \
  --header "Authorization: Bearer $PHD_RAG_API_TOKEN" \
  --header 'Content-Type: application/json' \
  --data '{"question":"Find Canadian computer science PhD research leads"}'
```
