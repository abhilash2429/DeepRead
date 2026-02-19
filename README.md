# DeepRead

Conversational ML paper comprehension app with FastAPI backend and Next.js frontend.
Backend agents are built with LangChain + LangGraph and Gemini (`gemini-flash-latest`).

## Backend setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn backend.main:app --reload --port 8000
```

If you do not want LangSmith network calls in local/offline runs, set:

```bash
LANGCHAIN_TRACING_V2=false
```

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Set `NEXT_PUBLIC_API_BASE=http://localhost:8000` in frontend env if needed.

## One-command run (npm)

From project root:

```bash
npm install
npm run dev
```

This starts backend + frontend in parallel.

## One-command run (Windows .cmd alternative)

```bat
run-all.cmd
```

This launches backend and frontend in parallel in separate terminal windows.

## Key endpoints

- `POST /ingest/upload`
- `POST /ingest/arxiv`
- `GET /ingest/{session_id}/events`
- `POST /conversation/{session_id}/message`
- `GET /conversation/{session_id}/state`
- `GET /conversation/{session_id}/artifacts`
- `POST /conversation/{session_id}/resolve-ambiguity`
