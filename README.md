# Peeranalysis App

Clean production repo for the peer explorer app.

## What is included

- FastAPI/Vercel API in `api/index.py`
- Local development UI/server in `peer_ui.py`
- Production peer data loader in `vercel_peer_data.py`
- Direct OpenAI rerank pipeline:
  - `40 + 40` product/company candidate union
  - Azure OpenAI final selection to 10 from the full deduped union
  - Optional Cohere + OpenAI comparison mode via `rerank_mode=cohere_openai`
- Compact deployment data in `vercel_data/`
- Small public UI assets in `public/`

## Required production environment variables

```text
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_KEY
AZURE_OPENAI_API_VERSION
AZURE_OPENAI_DEPLOYMENT
AZURE_OPENAI_REASONING_EFFORT=low
AZURE_OPENAI_FINAL_MAX_OUTPUT_TOKENS=4000
AZURE_OPENAI_TIMEOUT=30
```

Optional for `rerank_mode=cohere_openai` comparison runs:

```text
COHERE_API_KEY
COHERE_RERANK_TIMEOUT=20
```

## Local setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
.venv/bin/python run_local.py
```

Open `http://127.0.0.1:8766`.

## API

```bash
curl "http://127.0.0.1:8766/api/peers?company=INFOSYS%20LIMITED&limit=20"
```

Use `company` for a company name or CIN, and use `k` or `limit` to request 1-40
peers. The public API reranks by default using the deduped 40+40 union directly
with Azure OpenAI. For raw first-stage results, add `rerank=false`; for the old
Cohere comparison flow, add `rerank_mode=cohere_openai`.

## Tests

```bash
.venv/bin/python -m unittest test_openai_final_rerank.py test_rerank_candidate_pool.py test_api_rerank_modes.py
```

## CLI top peers

```bash
.venv/bin/python top_peers.py "HDFC Bank" --top 5
```

By default this prints the current pipeline top peers and also asks Azure OpenAI
to select a separate top 10 from a compact CSV of the enriched universe. This
full-universe call is slower and more expensive than the normal pipeline. For a
cheaper current-pipeline-only run:

```bash
.venv/bin/python top_peers.py "HDFC Bank" --top 5 --skip-universe-chatgpt
```

To test whether Cohere improves when it can see candidate-source and similarity
score evidence:

```bash
.venv/bin/python top_peers.py "Tata Consultancy Services" --top 10 --cohere-retrieval-evidence --skip-universe-chatgpt
```

## Deploy

This repo is configured for Vercel with `vercel.json`.
