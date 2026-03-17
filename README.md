# Vett – Local-First CV Scorer

Score your CV against job descriptions using AI. Everything runs locally.

## Quick Start

```bash
# 1. Configure your environment
cp .env.example .env
# Edit .env — set your preferred LLM provider + credentials

# 2. Launch with Docker Compose
docker compose up --build

# 3. Open the app
open http://localhost:5173
```

## Features

### Phase 1 (fully implemented)
| Feature | Detail |
|---------|--------|
| CV Management | Upload PDF, DOCX, DOC, MD, TXT — parsed and stored locally |
| Manual Scorer | Paste any JD → fit score (%), matched/missing keywords, gap analysis, rewrite suggestions |
| Score History | All results saved to local SQLite, viewable on Dashboard |
| LLM Switcher | Claude · OpenAI · Azure OpenAI · Ollama · LM Studio via LiteLLM |
| Settings | API keys, models, history toggle — persisted to .env |

### Phase 2 (scaffolded, not active)
| Feature | Status |
|---------|--------|
| Job Scraper | Playwright-based scraper for LinkedIn / Indeed / Naukri |
| Background Queue | Celery + Redis async task runner |
| Real-time Updates | SSE stream of scored results as they come in |
| Jobs Table | Filterable by score, work type, seniority, source |

To enable Phase 2, uncomment `celery_worker` in `docker-compose.yml` and implement the scrapers in `backend/app/tasks/scraper.py`.

## Architecture

```
┌─────────────────────┐     ┌──────────────────────┐
│  React + Vite       │────▶│  FastAPI              │
│  (Notion-style UI)  │◀────│  (REST + SSE)         │
│  :5173              │     │  :8000                │
└─────────────────────┘     └──────────┬───────────┘
                                        │
                            ┌───────────┼───────────┐
                            ▼           ▼           ▼
                        SQLite       LiteLLM      Redis
                        (local)    (multi-LLM)  (Phase 2)
```

## LLM Providers

Configure in Settings UI or `.env`:

| Provider | What to set |
|----------|-------------|
| **Ollama** (default) | `OLLAMA_BASE_URL`, `OLLAMA_MODEL` |
| **Claude** | `ANTHROPIC_API_KEY`, `CLAUDE_MODEL` |
| **OpenAI** | `OPENAI_API_KEY`, `OPENAI_MODEL` |
| **Azure OpenAI** | Key, endpoint, deployment, API version |
| **LM Studio** | `LM_STUDIO_BASE_URL`, `LM_STUDIO_MODEL` |

For Ollama/LM Studio inside Docker, use `host.docker.internal` as the hostname.

## Development (without Docker)

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

## Data

All data lives in `./data/`:
- `vett.db` — SQLite database (CV, history, jobs)
- `uploads/` — raw uploaded files

The `.env` file stores LLM credentials locally. Nothing is sent externally unless you configure a cloud provider.
