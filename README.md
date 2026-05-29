# Peeranalysis App

Clean production repo for the peer explorer app.

## What is included

- Flask/Vercel API in `api/index.py`
- Local development UI/server in `peer_ui.py`
- Production peer data loader in `vercel_peer_data.py`
- Two-stage rerank pipeline:
  - `40 + 40` product/company candidate union
  - Cohere rerank to 25
  - Azure OpenAI final selection to 10
- Compact deployment data in `vercel_data/`
- Small public UI assets in `public/`

## Required production environment variables

```text
COHERE_API_KEY
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_KEY
AZURE_OPENAI_API_VERSION
AZURE_OPENAI_DEPLOYMENT
AZURE_OPENAI_REASONING_EFFORT=low
AZURE_OPENAI_FINAL_MAX_OUTPUT_TOKENS=4000
COHERE_RERANK_TIMEOUT=20
AZURE_OPENAI_TIMEOUT=30
```

## Local setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
.venv/bin/python run_local.py
```

Open `http://127.0.0.1:8766`.

## Tests

```bash
.venv/bin/python -m unittest test_openai_final_rerank.py test_rerank_candidate_pool.py
```

## CLI top peers

```bash
.venv/bin/python top_peers.py "HDFC Bank" --top 5
```

## Deploy

This repo is configured for Vercel with `vercel.json`.
