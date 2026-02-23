# DeepRead

DeepRead is an ML paper comprehension system that generates a six-section implementation briefing, then supports follow-up Q&A with tool-calling agents.

## Architecture

- Backend: FastAPI + LangChain + LangGraph + Gemini 2.5 (`gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`)
- Persistence: PostgreSQL via Prisma (`prisma-client-py`)
- Auth: Google OAuth + JWT httpOnly cookie
- Frontend: Next.js App Router + Tailwind + SSE streaming UI

## Key Flow

1. User logs in (Google OAuth).
2. User uploads PDF or submits arXiv ID.
3. Backend runs:
   - `ingestion_agent` (PDF parse + figure interpretation)
   - `comprehension_agent` (full InternalRepresentation)
   - `briefing_agent` (six sequential LangGraph sections with SSE `thinking` + `section_token`)
4. Briefing sections are saved progressively to DB and rendered progressively in the UI.
5. Q&A uses `qa_agent` with tool-calling and DB-backed conversation memory.

## Repo Structure

- `backend/agents`: ingestion/comprehension/briefing/qa/code agents + graph state definitions
- `backend/routers`: `auth.py`, `ingest.py`, `conversation.py`
- `backend/db`: Prisma client lifecycle and all DB query functions
- `backend/models`: `paper.py`, `briefing.py`, `artifacts.py`
- `backend/prompts`: comprehension, six briefing prompts, QA, code generation, figure prompts
- `prisma/schema.prisma`: DB schema
- `frontend/app`: hero, upload, and session pages + next-auth route
- `frontend/components`: PDF panel/lever, thinking stream, briefing document, chat input, artifact downloads

## Local Setup

1. Copy env:
   - `copy .env.example .env`
   - Set `GEMINI_API_KEY`, `DATABASE_URL`, OAuth/JWT values in `.env`
2. Python deps:
   - `python -m venv .venv`
   - `.venv\Scripts\activate`
   - `pip install -r requirements.txt`
3. Generate Prisma client:
   - `prisma generate --schema prisma/schema.prisma`
4. Node deps:
   - `npm install`
   - `cd frontend && npm install && cd ..`
5. Run:
   - `npm run dev`

## Core Endpoints

- `GET /auth/google`
- `GET /auth/google/callback`
- `GET /auth/me`
- `POST /ingest/upload`
- `POST /ingest/arxiv`
- `GET /ingest/{paper_id}/events`
- `GET /ingest/{paper_id}/pdf`
- `POST /conversation/{paper_id}/message`
- `GET /conversation/{paper_id}/state`
- `GET /conversation/{paper_id}/artifacts`
- `POST /conversation/{paper_id}/resolve-ambiguity`
