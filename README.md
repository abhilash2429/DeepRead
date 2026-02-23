# DeepRead

DeepRead is an AI system for implementation-oriented understanding of machine learning papers.  
It ingests a PDF (uploaded directly or fetched from arXiv), builds a structured internal representation of the method, generates a six-section technical briefing, and supports follow-up Q&A grounded in the paper, generated briefing, and tool calls.

## What DeepRead Delivers

- Structured paper decomposition into:
  - problem statement
  - method summary
  - novelty
  - dependency graph of components
  - hyperparameter registry
  - ambiguity log
  - training procedure
  - prerequisite concepts
- Progressive six-section briefing generation streamed to the UI in real time.
- Component-level PyTorch snippet generation with provenance labels.
- Ambiguity resolution workflow where users can override unresolved implementation choices.
- Artifact exports:
  - `briefing.md`
  - `annotated_code.py`
  - `hyperparameters.csv`
  - `ambiguity_report.md`
- Authenticated, per-user paper sessions with free/paid plan gating.

## System Architecture

### Backend

- Framework: FastAPI (`backend/main.py`)
- Agent stack: LangChain + LangGraph + Google Gemini
- Data layer: PostgreSQL via Prisma (`prisma-client-py`)
- Auth: Google OAuth + JWT cookie (`/auth/*`)
- Streaming:
  - ingestion progress via SSE (`GET /ingest/{paper_id}/events`)
  - Q&A response tokens via POST-based SSE (`POST /conversation/{paper_id}/message`)

### Frontend

- Framework: Next.js App Router (React + TypeScript)
- Rendering:
  - upload workflow and auth state
  - session workspace with progressive briefing stream
  - integrated PDF panel (`react-pdf`)
  - ambiguity resolution and artifact download controls
- Theme system: `next-themes`
- Transport:
  - authenticated fetch calls with credentials
  - native EventSource for ingestion stream
  - custom SSE frame parser for POST streaming Q&A

## Core Pipeline

1. User uploads a PDF or submits an arXiv reference.
2. Ingestion pipeline runs:
   - PDF parsing into structured elements (sections/equations/figures/tables/pseudocode).
   - Figure interpretation with multimodal Gemini.
   - Primary task and prerequisite extraction.
3. Comprehension pipeline builds `InternalRepresentation`.
4. Briefing pipeline generates six sections sequentially and streams section tokens.
5. Structured outputs (briefing sections, hyperparameters, ambiguities, code snippets) are persisted.
6. User asks follow-up questions through a tool-calling Q&A agent.

## Six-Section Briefing Contract

1. What This Paper Actually Does
2. The Mechanism
3. What You Need To Already Know
4. The Full Implementation Map
5. What The Paper Left Out
6. How To Train It

Sections are generated with dedicated prompts (`backend/prompts/briefing_sections.py`) and model-tier selection per section (Pro for higher reasoning sections, Flash for speed-focused sections).

## Data Model (High-Level)

### `ParsedPaper` (`backend/models/paper.py`)

- Metadata: title, authors, abstract
- Content: full text + typed elements
- Enrichment: figure descriptions, primary task hint, raw prerequisites
- PDF payload stored in base64 for retrieval/rendering

### `InternalRepresentation` (`backend/models/briefing.py`)

- `problem_statement`, `method_summary`, `novelty`
- `component_graph` as dependency edges
- `hyperparameter_registry` with status labels
- `ambiguity_log` with confidence and optional user resolution
- `training_procedure`
- `prerequisite_concepts` as problem-solution-usage units

### Persistence Schema (`prisma/schema.prisma`)

- `User` with plan and paper usage counters
- `Paper` with `parsed_paper` + `internal_rep` JSON
- `Briefing` with six section fields and structured outputs
- `QAMessage` for persisted conversation history

## API Surface

### Authentication

- `GET /auth/google`
- `GET /auth/google/callback`
- `GET /auth/me`
- `POST /auth/logout`

### Ingestion

- `POST /ingest/upload`
- `POST /ingest/arxiv`
- `GET /ingest/{paper_id}/events`
- `GET /ingest/{paper_id}/pdf`

### Conversation & Artifacts

- `POST /conversation/{paper_id}/message`
- `GET /conversation/{paper_id}/state`
- `GET /conversation/{paper_id}/artifacts`
- `POST /conversation/{paper_id}/resolve-ambiguity`

### Health

- `GET /health`

## Tooling Available to Q&A Agent

- `paper_section_lookup`
- `equation_decoder`
- `figure_interpreter`
- `prerequisite_expander`
- `background_knowledge_lookup`
- `hyperparameter_extractor`
- `ambiguity_detector`
- `code_snippet_generator`

These tools are wired in `backend/tools/*` and attached during Q&A agent execution.

## Security and Access Model

- Google OAuth login flow issues a JWT in an `httpOnly` cookie.
- Backend routes for ingestion and conversation require authenticated user context.
- Paper access is user-scoped (`paper.user_id` checks in route handlers).
- Free-plan quota enforcement occurs before ingestion execution.

## Notable Implementation Characteristics

- Async-first backend orchestration for ingestion and generation tasks.
- Progressive generation UX through SSE events (`thinking`, `section_token`, `progress`, `done`, `error`).
- Robust arXiv ingestion strategy:
  - ID normalization
  - metadata fetch + fallback path
  - multiple PDF endpoints
  - network-error diagnostics
- Heuristic PDF extraction optimized for implementation details:
  - heading detection
  - equation signal detection
  - pseudocode and table detection
  - figure extraction and caption association

## Repository Layout

- `backend/agents`: ingestion, comprehension, briefing, Q&A, code-generation agents
- `backend/routers`: auth, ingestion, conversation APIs
- `backend/db`: Prisma lifecycle + query layer
- `backend/models`: Pydantic models for paper, briefing, artifacts
- `backend/prompts`: prompt contracts for comprehension, briefing, QA, figure/code tasks
- `backend/services`: PDF parsing, arXiv fetch, figure interpretation
- `backend/tools`: tool builders exposed to LangChain agents
- `frontend/app`: upload/session pages and app-level wiring
- `frontend/components`: briefing renderer, chat input, PDF panel, artifact controls
- `frontend/lib`: API client + shared types + auth/prisma utilities
- `prisma/schema.prisma`: relational schema and JSON-backed paper state

## Environment Contract (Reference)

DeepRead relies on environment configuration for runtime integrations, including:

- `GEMINI_API_KEY`
- `DATABASE_URL`
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `GOOGLE_REDIRECT_URI`
- `JWT_SECRET`
- `JWT_ALGORITHM`
- `JWT_EXPIRE_MINUTES`
- `NEXTAUTH_URL`
- `NEXTAUTH_SECRET`
- `NEXT_PUBLIC_API_BASE`
- `MAX_PAPER_SIZE_MB`
- optional LangSmith tracing variables

See `.env.example` for the full variable list and naming contract.
