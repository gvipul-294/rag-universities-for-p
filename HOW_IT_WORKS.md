# How it works

PhD Finder reads official university PhD admissions pages and answers questions about them, citing the source page for every answer.

This page lists each step and the file that does it. For setup details and options, see [README.md](README.md).

All code is in `src/phd_rag/`. Every step runs through one command-line entry point:

```bash
PYTHONPATH=src python -m phd_rag.cli <command>
```

## Part 1: Build the index (`ingest` command, run once)

| Step | What happens | File · function | Output |
| --- | --- | --- | --- |
| 1. Load settings | Read models, keys and paths from `.env` | `settings.py` · `get_settings()` | — |
| 2. Load the source list | Read the 100 university URLs | `ingest.py` · `load_registry()`, `select_sources()` | — |
| 3. Download pages | Fetch each page and keep only the visible text. Each result is logged as soon as it finishes. | `ingest.py` · `fetch_pages()`, `extract_page_text()` | `raw_pages/*.txt`, `artifacts/fetch_log.jsonl` |
| 4. Split into chunks | Cut each page into pieces of about 1,200 characters | `ingest.py` · `chunk_pages()`, `split_text()` | — |
| 5. Turn chunks into vectors | Convert each chunk to numbers, locally, with `bge-small-en-v1.5` | `embeddings.py` · `embed_passages()` | — |
| 6. Save the index | Store vectors and chunk text for searching | `ingest.py` · `write_index()` | `artifacts/university_phd.faiss`, `artifacts/chunks.jsonl`, `artifacts/manifest.json` |

`ingest.py` · `build_index()` runs steps 2–6 in order.

## Part 2: Answer a question (`ask` and `recommend` commands)

| Step | What happens | File · function |
| --- | --- | --- |
| 1. Read the input | A question, or a candidate profile turned into a question | `cli.py` · `main()`, `retrieve.py` · `profile_to_question()` |
| 2. Turn the question into a vector | Same embedding model as indexing | `embeddings.py` · `embed_query()` |
| 3. Find matching chunks | Search the index for the 12 closest chunks, filtered by country for profiles | `retrieve.py` · `search()` |
| 4. Write the answer | Send the question, chunks and rules to the answer model (Groq `gpt-oss-120b`), which writes a cited reply | `retrieve.py` · `answer()` |

## Optional: agent mode (`agent` command)

A local Ollama model decides which search tool to call, then writes the answer itself. This needs Ollama and several GB of RAM.

| Part | File |
| --- | --- |
| Search tools exposed over MCP | `mcp_server.py` · `search_official_university_sources()`, `map_candidate_to_universities()`, `get_rag_index_status()` |
| Agent loop that calls those tools | `agentic.py` · `run_agentic_matcher()` |

## Other files

| File | Purpose |
| --- | --- |
| `data/universities.json` | The 100 official source URLs |
| `data/example_candidate.json` | A sample candidate profile |
| `models.py` | The shape of a source (`UniversitySource`) and a profile (`CandidateProfile`) |
| `tests/test_registry.py` | Checks the source list is valid |
| `.env.example` | Template for your `.env` settings |

## Quick commands

```bash
PYTHONPATH=src python -m phd_rag.cli sources                 # list the 100 sources
PYTHONPATH=src python -m phd_rag.cli ingest                  # build the index
PYTHONPATH=src python -m phd_rag.cli ask "your question"     # ask a question
PYTHONPATH=src python -m phd_rag.cli recommend --profile data/example_candidate.json
PYTHONPATH=src python -m unittest discover -s tests          # run tests
```
