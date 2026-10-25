# Deployment

How the PhD Finder runs in production, and how to update it.

```text
Browser ──> Vercel (frontend/, Next.js)
              │  server-side proxy: /api/backend/* , adds the bearer token
              v
           EC2 :8000  phd-rag-api.service (FastAPI/uvicorn, src/phd_rag/api.py)
              │  local: fastembed query embedding + FAISS index (artifacts/)
              v
           Groq API (answer model, LLM_API_KEY)
```

## EC2 host layout

| Path | Contents |
| --- | --- |
| `~/phd-rag/src`, `data`, `tests`, `requirements.txt`, `pyproject.toml` | Code, copied from a reviewed commit. No `.git` on the server. |
| `~/phd-rag/REVISION` | The commit that is deployed. |
| `~/phd-rag/.env` | Server secrets and settings (`LLM_API_KEY`, `PHD_RAG_API_TOKEN`, …). Never committed. |
| `~/phd-rag/.venv` | Python 3.12 virtual environment. |
| `~/phd-rag/.cache/fastembed` | Embedding model weights (downloaded on first use). |
| `~/phd-rag/artifacts`, `raw_pages` | The FAISS index and fetched page text, built by `ingest`. |
| `/etc/systemd/system/phd-rag-api.service` | The service definition, a copy of [`deploy/phd-rag-api.service`](../deploy/phd-rag-api.service). |

Do not edit code on the server. Change it in git, merge, then deploy.

## Update the code

1. Copy the committed run files of the merged commit to `~/phd-rag` (`src`, `data`, `tests`, `requirements.txt`, `pyproject.toml`), and write the commit hash to `~/phd-rag/REVISION`. Leave `.env`, `.venv`, `.cache`, `artifacts` and `raw_pages` untouched.
2. On the server:
   ```bash
   cd ~/phd-rag
   .venv/bin/pip install -q -r requirements.txt
   PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
   sudo systemctl restart phd-rag-api
   curl -s http://127.0.0.1:8000/api/v1/health
   ```

The frontend deploys itself: Vercel builds every pull request as a preview and publishes `main` to production.

## Install or change the API service

```bash
# Copy deploy/phd-rag-api.service from the repo to the server (scp), or paste it into:
sudo nano /etc/systemd/system/phd-rag-api.service
sudo systemctl daemon-reload
sudo systemctl enable --now phd-rag-api      # first install; use `restart` afterwards
systemctl status phd-rag-api --no-pager
```

| Task | Command |
| --- | --- |
| Restart after a deploy or `.env` change | `sudo systemctl restart phd-rag-api` |
| Recent logs | `journalctl -u phd-rag-api -n 50 --no-pager` |
| Follow logs | `journalctl -u phd-rag-api -f` |
| Memory in use | `systemctl show phd-rag-api -p MemoryCurrent` |

`deploy/` is not copied to `~/phd-rag` with the run files; when the unit file changes, update `/etc/systemd/system/phd-rag-api.service` by hand and run `sudo systemctl daemon-reload && sudo systemctl restart phd-rag-api`.

## API token

The API checks `Authorization: Bearer <PHD_RAG_API_TOKEN>` on every endpoint except `/api/v1/health`. Without a token configured, anyone who finds port 8000 can spend the Groq quota.

Set it in this order, or the live site returns 401 in between:

1. Generate a value: `openssl rand -hex 32`.
2. In Vercel → Project → Settings → Environment Variables, add `PHD_RAG_API_TOKEN` with that value (Production and Preview), then **redeploy** (environment changes apply only to new deployments).
3. On the server, add `PHD_RAG_API_TOKEN=<same value>` to `~/phd-rag/.env` and run `sudo systemctl restart phd-rag-api`.
4. Check: a request without the token is refused, and the site still answers.
   ```bash
   curl -s -o /dev/null -w "%{http_code}\n" -X POST http://<ec2-ip>:8000/api/v1/ask \
     -H 'Content-Type: application/json' -d '{"question":"test question"}'   # expect 401
   ```

To rotate the token, repeat steps 1–3. The token travels over plain HTTP between Vercel and EC2 until the API is put behind HTTPS (for example Caddy with a domain name).

## Memory on the 1 GB instance

The API uses about 100 MB at start and about 450–500 MB after the first question, once the embedding model is loaded. `MemoryMax=700M` in the unit restarts it if it grows past that. Embedding calls are serialised and uvicorn caps in-flight requests at 4, so concurrent questions do not each add working memory.

Optional safety net, a 1 GB swap file:

```bash
sudo dd if=/dev/zero of=/swapfile bs=1M count=1024 && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab
```

`ingest` also needs ~450 MB. Run it while traffic is low, or stop the API service during a rebuild.
