# DeepRead — Agent Context Document

> Full architectural context for AI agents working on this codebase.
> Read this before making any changes.

---

## 1. Project Overview

**DeepRead** is an agentic ML-paper comprehension system. Users authenticate via Google or GitHub OAuth, upload a PDF (or paste an arXiv ID/URL), and the system runs a multi-agent pipeline that:

1. Parses the PDF into a structured representation (`ParsedPaper`)
2. Feeds the full text into a comprehension agent that produces a deep structured analysis (`InternalRepresentation`)
3. Generates a 6-section briefing document via a LangGraph pipeline, streaming each section to the frontend in real time
4. Exposes a tool-calling Q&A agent for follow-up questions grounded in the paper and briefing
5. Produces downloadable artifacts (`.md`, `.py`, `.csv`)

### Core Value Proposition

- **Structured comprehension**: Papers are decomposed into problem, method, novelty, component graph, hyperparameters, ambiguities, and prerequisites
- **Briefing document**: 6 specialist sections — plain-English summary, mechanism deep-dive, prerequisites, implementation map with labeled code, ambiguity report, and training recipe
- **Code generation**: Annotated PyTorch snippets with provenance labels (`paper-stated`, `inferred`, `assumed`, `missing`)
- **Ambiguity resolution**: Surfaces underspecified details and lets the user override agent-proposed defaults
- **Downloadable artifacts**: Architecture summary, annotated code, hyperparameter CSV, ambiguity report

---

## 2. Tech Stack

| Layer       | Technology                                                               |
| ----------- | ------------------------------------------------------------------------ |
| Backend     | Python 3.11+, FastAPI, Uvicorn                                           |
| LLM         | Google Gemini (`gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`) via `langchain-google-genai` |
| Agents      | LangChain (`langchain_classic` for `AgentExecutor`), LangGraph (state graphs) |
| Database    | PostgreSQL via Prisma (`prisma-client-py`, async)                        |
| Auth        | Google + GitHub OAuth (Authlib), JWT (python-jose), Starlette sessions   |
| PDF parse   | PyMuPDF (`fitz`), Pillow                                                 |
| arXiv       | `arxiv` library + `httpx` fallback                                       |
| Frontend    | Next.js 14 (App Router), React 18, TypeScript, TailwindCSS 3            |
| Markdown    | `react-markdown` + `remark-gfm` for rendering briefing sections          |
| PDF viewer  | `react-pdf`                                                              |
| Code blocks | `highlight.js`                                                           |
| Dev runner  | `concurrently` (root `package.json`)                                     |

### Environment Variables (`.env`)

| Variable                     | Purpose                                                        |
| ---------------------------- | -------------------------------------------------------------- |
| `GEMINI_API_KEY`             | **Required.** Google AI API key                                |
| `DATABASE_URL`               | **Required.** PostgreSQL connection string                     |
| `GOOGLE_CLIENT_ID`           | Google OAuth client ID                                         |
| `GOOGLE_CLIENT_SECRET`       | Google OAuth client secret                                     |
| `GOOGLE_REDIRECT_URI`        | Google OAuth callback URL                                      |
| `GITHUB_CLIENT_ID`           | GitHub OAuth client ID (optional)                              |
| `GITHUB_CLIENT_SECRET`       | GitHub OAuth client secret (optional)                          |
| `GITHUB_REDIRECT_URI`        | GitHub OAuth callback URL                                      |
| `JWT_SECRET`                 | JWT signing secret (>=32 chars in production)                  |
| `SESSION_SECRET`             | Starlette session signing secret (>=32 chars in production)    |
| `JWT_ALGORITHM`              | JWT algorithm (default `HS256`)                                |
| `JWT_EXPIRE_MINUTES`         | JWT expiry in minutes (default `10080` = 7 days)               |
| `APP_ENV`                    | `development` or `production`                                  |
| `COOKIE_SAMESITE`            | Cookie SameSite policy (`lax`, `strict`, `none`)               |
| `NEXTAUTH_URL`               | Frontend origin for CORS + auth redirects                      |
| `FRONTEND_ORIGINS`           | Comma-separated additional CORS origins                        |
| `NEXT_PUBLIC_API_BASE`       | Frontend → backend URL (default `http://localhost:8000`)        |
| `MAX_PAPER_SIZE_MB`          | Max PDF upload size (default `20`)                             |
| `INGEST_CONCURRENCY_LIMIT`   | Max concurrent ingestion pipelines (default `2`)               |
| `INGEST_PENDING_LIMIT`       | Max queued ingestion requests (default `20`)                   |
| `CAPACITY_LOCK`              | Backend: block `/ingest` + `/conversation` with 503            |
| `NEXT_PUBLIC_CAPACITY_LOCK`  | Frontend: show example walkthroughs instead of live analysis   |
| `LANGCHAIN_TRACING_V2`       | Enable LangSmith tracing (`true`/`false`)                      |
| `LANGCHAIN_API_KEY`          | LangSmith API key (optional)                                   |
| `LANGCHAIN_PROJECT`          | LangSmith project name                                         |

---

## 3. Repository Structure

```
DeepRead/
├── backend/
│   ├── main.py                          # FastAPI app, lifespan, middleware, router mounting
│   ├── agents/
│   │   ├── graph.py                     # BriefingState TypedDict + build_briefing_graph()
│   │   ├── ingestion_agent.py           # run_ingestion(): PDF parse + vision + task extraction
│   │   ├── comprehension_agent.py       # run_comprehension(): full paper → InternalRepresentation
│   │   ├── briefing_agent.py            # run_briefing_pipeline(): 6-section LangGraph pipeline
│   │   ├── qa_agent.py                  # run_qa_turn(): tool-calling Q&A agent
│   │   └── code_agent.py               # generate_component_code(): PyTorch CodeSnippet generation
│   ├── models/
│   │   ├── paper.py                     # ParsedPaper, PaperElement, ElementType, ProvenanceLabel
│   │   ├── briefing.py                  # InternalRepresentation, AmbiguityEntry, HyperparameterEntry, etc.
│   │   └── artifacts.py                 # CodeSnippet, ArtifactItem, ArtifactManifest
│   ├── prompts/
│   │   ├── briefing_sections.py         # SECTION_PROMPTS dict (system prompts per briefing section)
│   │   ├── comprehension.py             # COMPREHENSION_PROMPT (JSON schema for InternalRepresentation)
│   │   ├── code_gen.py                  # CODE_GEN_PROMPT (PyTorch code generation rules)
│   │   ├── figure.py                    # FIGURE_PROMPT (vision description rules)
│   │   └── qa.py                        # QA_PROMPT (Q&A agent system prompt)
│   ├── routers/
│   │   ├── auth.py                      # /auth/* OAuth + JWT + /me endpoint
│   │   ├── ingest.py                    # /ingest/* upload, arxiv, events SSE, pdf retrieval
│   │   └── conversation.py              # /conversation/* message SSE, state, artifacts, resolve-ambiguity
│   ├── services/
│   │   ├── pdf_parser.py                # parse_pdf(): PyMuPDF structured extraction
│   │   ├── vision_service.py            # describe_figure(): Gemini multimodal figure analysis
│   │   └── arxiv_fetcher.py             # arXiv metadata + PDF download with fallback strategies
│   ├── db/
│   │   ├── prisma.py                    # Prisma client singleton + async lifespan context
│   │   └── queries.py                   # All database operations (users, papers, briefings, QA messages)
│   ├── memory/
│   │   └── session_memory.py            # ConversationSummaryBufferMemory management for Q&A
│   ├── tools/
│   │   ├── __init__.py                  # Re-exports: build_paper_tools, build_knowledge_tools, etc.
│   │   ├── paper_tools.py               # paper_section_lookup, equation_decoder
│   │   ├── knowledge_tools.py           # prerequisite_expander, background_knowledge_lookup
│   │   ├── analysis_tools.py            # hyperparameter_extractor, ambiguity_detector
│   │   └── code_tools.py                # code_snippet_generator (wraps code_agent)
│   └── background_knowledge/
│       └── landmark_papers.py           # Built-in summaries of landmark papers (Transformer, ResNet, etc.)
├── frontend/
│   ├── app/
│   │   ├── layout.tsx                   # Root layout (Google Fonts: Syne, DM Mono, Instrument Serif)
│   │   ├── Providers.tsx                # ThemeProvider wrapper
│   │   ├── globals.css                  # Global styles + CSS variables
│   │   ├── page.tsx                     # Landing page (hero, pipeline SVG, examples, output cards)
│   │   ├── page.module.css              # Landing page styles
│   │   ├── signin/page.tsx              # OAuth sign-in page (Google + GitHub buttons)
│   │   ├── dashboard/page.tsx           # User dashboard (paper history, usage limits)
│   │   ├── upload/page.tsx              # PDF upload + arXiv input + ingestion progress
│   │   ├── session/[id]/page.tsx        # Session page (PDF viewer + briefing + Q&A + artifacts)
│   │   ├── examples/page.tsx            # Example walkthroughs index
│   │   ├── examples/[slug]/page.tsx     # Individual example walkthrough
│   │   └── api/auth/...                 # Next.js API route for auth
│   ├── components/
│   │   ├── BriefingDocument.tsx          # Full briefing document renderer (all 6 sections)
│   │   ├── BriefingSection.tsx           # Single briefing section with markdown rendering
│   │   ├── ChatInput.tsx                 # Q&A message input
│   │   ├── ThinkingStream.tsx            # Live agent thinking/progress stream display
│   │   ├── ArtifactDownloads.tsx         # Artifact download buttons
│   │   ├── AmbiguityCard.tsx             # Displays/resolves a single ambiguity entry
│   │   ├── CodeBlock.tsx                 # Syntax-highlighted Python code with provenance badge
│   │   ├── HyperparamTable.tsx           # Hyperparameter table with status color coding
│   │   ├── PrerequisiteCard.tsx          # Prerequisite concept explanation card
│   │   ├── PdfPanel.tsx                  # PDF viewer panel (react-pdf)
│   │   ├── PdfLever.tsx                  # PDF panel toggle control
│   │   ├── ExampleWalkthroughDocument.tsx # Example walkthrough document
│   │   └── ThemeToggle.tsx               # Dark/light theme toggle
│   ├── hooks/
│   │   └── useSSE.ts                     # Custom hook: POST-based SSE streaming
│   ├── lib/
│   │   ├── api.ts                        # API client (ingest, conversation, auth, artifacts)
│   │   ├── types.ts                      # TypeScript types mirroring backend models
│   │   ├── examples.ts                   # Example walkthrough data (Transformer, ResNet, BERT)
│   │   └── prisma.ts                     # Prisma client for Next.js (if server-side needed)
│   └── public/                           # Static assets
├── prisma/
│   └── schema.prisma                     # Database schema (User, Paper, Briefing, QAMessage)
├── .env.example
├── requirements.txt
├── package.json                          # Root: concurrently runs backend + frontend
├── README.md
└── agents.md                             # This file
```

---

## 4. Database Schema (Prisma)

```prisma
model User {
  id              String   @id @default(cuid())
  google_sub      String   @unique        // "github:{id}" for GitHub users
  email           String   @unique
  name            String
  avatar_url      String?
  plan            Plan     @default(FREE)  // FREE or PRO
  papers_analyzed Int      @default(0)
  papers          Paper[]
}

model Paper {
  id           String      @id @default(cuid())
  user_id      String
  title        String
  authors      String[]
  arxiv_id     String?
  status       PaperStatus @default(PROCESSING)  // PROCESSING | COMPLETE | FAILED
  parsed_paper Json        // serialized ParsedPaper
  internal_rep Json        // serialized InternalRepresentation
  briefing     Briefing?
  qa_messages  QAMessage[]
}

model Briefing {
  id              String   @id @default(cuid())
  paper_id        String   @unique
  section_1..6    String?  // markdown content per section
  hyperparameters Json?    // serialized HyperparameterEntry[]
  ambiguities     Json?    // serialized AmbiguityEntry[]
  code_snippets   Json?    // serialized CodeSnippet[]
}

model QAMessage {
  id         String   @id @default(cuid())
  paper_id   String
  role       String   // "user" | "assistant"
  content    String
  created_at DateTime @default(now())
}
```

**Free plan limit**: `FREE_PLAN_LIMIT = 3` papers per user (enforced in `db/queries.py`).

---

## 5. Architecture & Data Flow

### 5.1 Authentication

```
User clicks "Sign in with Google" or "Sign in with GitHub"
        │
        ▼
  GET /auth/google (or /auth/github)
    → Authlib redirects to OAuth provider
        │
        ▼
  GET /auth/google/callback (or /auth/github/callback)
    → Validates OAuth token
    → Upserts User in PostgreSQL
    → Creates JWT with user_id claim
    → Sets httpOnly cookie
    → Redirects to frontend /dashboard
```

All authenticated endpoints use `Depends(get_current_user)` which extracts the JWT from the `access_token` cookie and returns the user record.

### 5.2 Ingestion Pipeline

```
User uploads PDF / enters arXiv ID
        │
        ▼
  ┌─────────────────────────────────┐
  │  Router: POST /ingest/upload    │
  │          POST /ingest/arxiv     │
  │  • Validates auth + user limit  │
  │  • Creates Paper row (PROCESSING)│
  │  • Starts async _run_pipeline() │
  │  • Returns { paper_id }         │
  └────────────┬────────────────────┘
               │
               ▼
  ┌─────────────────────────────────┐
  │  _run_pipeline() (async task)   │
  │                                 │
  │  1. parse_pdf() → ParsedPaper   │
  │  2. describe_figure() per image │  ← Gemini vision (gemini-2.5-flash)
  │  3. Extract primary_task +      │  ← Gemini text
  │     prerequisites               │
  │  4. Save ParsedPaper to DB      │
  └────────────┬────────────────────┘
               │
               ▼
  ┌─────────────────────────────────┐
  │  run_comprehension()            │
  │  Full paper → structured JSON   │  ← Gemini (gemini-2.5-pro, 1M context)
  │  → InternalRepresentation       │
  │  Save to DB                     │
  └────────────┬────────────────────┘
               │
               ▼
  ┌─────────────────────────────────┐
  │  run_briefing_pipeline()        │
  │  LangGraph: 6 sequential nodes  │
  │  Each section streamed via SSE  │  ← gemini-2.5-pro (sections 2,4,5) or gemini-2.5-flash (1,3,6)
  │  Section 4 also generates code  │  ← code_agent for each component
  │  Save each section to Briefing  │
  │  Save structured data (hyper,   │
  │  ambiguities, code_snippets)    │
  └────────────┬────────────────────┘
               │
               ▼
  ┌─────────────────────────────────┐
  │  Paper.status → COMPLETE        │
  │  SSE "done" event sent          │
  │  Frontend redirects to          │
  │  /session/{paper_id}            │
  └─────────────────────────────────┘
```

**SSE streaming**: The ingestion pipeline emits events (`thinking`, `section`, `section_token`, `code_start`, `code_token`, `code_end`, `structured_data`, `done`, `error`) via an `asyncio.Queue`. Frontend listens on `GET /ingest/{paper_id}/events`.

### 5.3 Briefing Pipeline (LangGraph)

The briefing pipeline uses a linear `StateGraph` with `BriefingState`:

```python
class BriefingState(TypedDict, total=False):
    session_id: str
    paper_id: str
    internal_rep: InternalRepresentation
    parsed_paper: ParsedPaper
    completed_sections: dict[str, str]
    generation_progress: int
    code_snippets: list[CodeSnippet]
```

```
section_1 → section_2 → section_3 → section_4 → section_5 → section_6 → END
```

Each section node:
1. Builds context from `ParsedPaper` + `InternalRepresentation` (figures, equations, components)
2. Selects model: `gemini-2.5-pro` for sections 2, 4, 5 (deep reasoning), `gemini-2.5-flash` for 1, 3, 6 (speed)
3. Streams tokens via SSE events
4. Saves completed section markdown to `Briefing` table
5. Section 4 additionally generates `CodeSnippet` per component via `code_agent`

### 5.4 Q&A Agent

After the briefing is complete, the user can ask follow-up questions via the Q&A agent:

```
User message arrives at POST /conversation/{paper_id}/message
        │
        ▼
  ┌─────────────────────────────────┐
  │  Load from DB:                  │
  │  • ParsedPaper, InternalRep     │
  │  • Briefing markdown            │
  │  • Chat history (QAMessage)     │
  │  • Resolved ambiguities         │
  └────────────┬────────────────────┘
               │
               ▼
  ┌─────────────────────────────────┐
  │  run_qa_turn()                  │
  │  • Builds tool set              │
  │  • Creates AgentExecutor        │  ← LangChain tool-calling agent
  │  • LangGraph: QAState graph     │
  │  • Invokes with chat history    │
  │    (last 12 messages)           │
  └────────────┬────────────────────┘
               │
               ▼
  Response streamed as SSE tokens
  Chat history saved to QAMessage table
```

**Model**: `gemini-2.5-flash` for Q&A.  
**Memory**: `ConversationSummaryBufferMemory` (max 4000 tokens) using `gemini-2.5-flash-lite` for summarization.

### 5.5 Tool System

The Q&A agent has access to these tools during each conversation turn:

| Tool                           | Module                     | Purpose                                                     |
| ------------------------------ | -------------------------- | ----------------------------------------------------------- |
| `paper_section_lookup`         | `tools/paper_tools.py`     | Search parsed paper sections by heading or content keyword   |
| `equation_decoder`             | `tools/paper_tools.py`     | Retrieve equation text by label (e.g., `(1)`)               |
| `prerequisite_expander`        | `tools/knowledge_tools.py` | Explain a prerequisite concept from the InternalRepresentation |
| `background_knowledge_lookup`  | `tools/knowledge_tools.py` | Built-in summaries of landmark papers                        |
| `hyperparameter_extractor`     | `tools/analysis_tools.py`  | Return the full hyperparameter registry as text              |
| `ambiguity_detector`           | `tools/analysis_tools.py`  | Scan section text for ambiguity markers                      |
| `code_snippet_generator`       | `tools/code_tools.py`      | Generate annotated PyTorch code for a named component        |

Tools are factory-built functions (closures binding `parsed_paper`, `internal_rep`, or a code callback).

---

## 6. Key Data Models

### 6.1 `ParsedPaper` (`backend/models/paper.py`)

Structured representation of an ingested PDF:
- `title`, `authors`, `abstract`, `full_text`
- `elements: list[PaperElement]` — sections, equations, figures, tables, pseudocode
- `primary_task`, `prerequisites_raw`
- `pdf_bytes_b64` — base64-encoded PDF for browser rendering

Each `PaperElement` has: `id`, `element_type` (enum: Section, Equation, Figure, Table, Pseudocode), `section_heading`, `page_number`, `content`, `caption`, `equation_label`, `image_bytes_b64`, `figure_description`.

### 6.2 `InternalRepresentation` (`backend/models/briefing.py`)

LLM-generated deep analysis of the paper:
- `problem_statement`, `method_summary`, `novelty`
- `component_graph: list[DependencyEdge]` (parent/child relationships)
- `hyperparameter_registry: list[HyperparameterEntry]` (name, value, source, status, suggested default)
- `ambiguity_log: list[AmbiguityEntry]` (ambiguities with impact, agent resolution, confidence)
- `training_procedure` — free-text training recipe
- `prerequisite_concepts: list[ConceptExplanation]` (concept + problem/solution/usage)

### 6.3 `AmbiguityEntry` (`backend/models/briefing.py`)

```python
class AmbiguityEntry(BaseModel):
    ambiguity_id: str
    ambiguity_type: AmbiguityType  # missing_hyperparameter | undefined_notation | underspecified_architecture | missing_training_detail | ambiguous_loss_function
    title: str
    ambiguous_point: str
    section: str
    implementation_consequence: str
    agent_resolution: str
    reasoning: str
    confidence: float  # 0.0 - 1.0
    resolved: bool
    user_resolution: str | None
```

### 6.4 `CodeSnippet` (`backend/models/artifacts.py`)

```python
class CodeSnippet(BaseModel):
    component_name: str
    code: str
    provenance: ProvenanceLabel  # "paper-stated" | "inferred" | "assumed" | "missing"
    assumption_notes: list[str]
    source_sections: list[str]
    equation_references: list[str]
```

### 6.5 `ProvenanceLabel` (`backend/models/paper.py`)

```python
ProvenanceLabel = Literal["paper-stated", "inferred", "assumed", "missing"]
```

Used throughout to track information provenance.

---

## 7. API Endpoints

### Authentication

| Method | Path                       | Auth     | Description                                |
| ------ | -------------------------- | -------- | ------------------------------------------ |
| GET    | `/auth/google`             | No       | Initiate Google OAuth flow                 |
| GET    | `/auth/google/callback`    | No       | Google OAuth callback, sets JWT cookie     |
| GET    | `/auth/github`             | No       | Initiate GitHub OAuth flow                 |
| GET    | `/auth/github/callback`    | No       | GitHub OAuth callback, sets JWT cookie     |
| POST   | `/auth/logout`             | No       | Clear auth cookie                          |
| GET    | `/auth/me`                 | Yes      | Return current user profile + usage limits |

### Ingestion

| Method | Path                          | Auth | Description                                      |
| ------ | ----------------------------- | ---- | ------------------------------------------------ |
| POST   | `/ingest/upload`              | Yes  | Upload PDF. Returns `{ paper_id }`               |
| POST   | `/ingest/arxiv`               | Yes  | Ingest from arXiv ref. Returns `{ paper_id }`    |
| GET    | `/ingest/{paper_id}/events`   | Yes  | SSE stream of ingestion + briefing progress      |
| GET    | `/ingest/{paper_id}/pdf`      | Yes  | Retrieve stored PDF bytes for browser rendering  |

### Conversation

| Method | Path                                           | Auth | Description                                    |
| ------ | ---------------------------------------------- | ---- | ---------------------------------------------- |
| POST   | `/conversation/{paper_id}/message`             | Yes  | Send Q&A message; response streamed as SSE     |
| GET    | `/conversation/{paper_id}/state`               | Yes  | Get full paper state (briefing, hyper, ambig)  |
| GET    | `/conversation/{paper_id}/artifacts`           | Yes  | Get downloadable artifact manifest             |
| POST   | `/conversation/{paper_id}/resolve-ambiguity`   | Yes  | Resolve a specific ambiguity by ID             |

### Health

| Method | Path      | Description              |
| ------ | --------- | ------------------------ |
| GET    | `/health` | Returns `{"status":"ok"}`|

### SSE Event Types

**Ingestion stream** (`/ingest/{paper_id}/events`):

| Event             | Description                                        |
| ----------------- | -------------------------------------------------- |
| `thinking`        | Agent progress update (status message)             |
| `section`         | Briefing section start (section number + name)     |
| `section_token`   | Streamed token within a briefing section           |
| `code_start`      | Code snippet generation started for a component    |
| `code_token`      | Streamed code token                                |
| `code_end`        | Code snippet complete                              |
| `structured_data` | Hyperparameters + ambiguities + code snippets JSON |
| `done`            | Pipeline complete                                  |
| `error`           | Pipeline error                                     |

**Conversation stream** (`/conversation/{paper_id}/message`):

| Event   | Description                    |
| ------- | ------------------------------ |
| `token` | Streamed Q&A response token    |
| `done`  | Q&A response complete          |
| `error` | Error during Q&A processing    |

---

## 8. Frontend Architecture

### Pages

| Route                   | File                              | Description                                                  |
| ----------------------- | --------------------------------- | ------------------------------------------------------------ |
| `/`                     | `app/page.tsx`                    | Landing page: hero, pipeline SVG, example walkthroughs, output cards |
| `/signin`               | `app/signin/page.tsx`             | OAuth sign-in page (Google + GitHub buttons)                 |
| `/dashboard`            | `app/dashboard/page.tsx`          | User dashboard: paper history, usage limits, "Analyze" CTA  |
| `/upload`               | `app/upload/page.tsx`             | PDF upload + arXiv input + live ingestion progress           |
| `/session/[id]`         | `app/session/[id]/page.tsx`       | Session: PDF viewer + briefing document + Q&A chat + artifacts |
| `/examples`             | `app/examples/page.tsx`           | Example walkthroughs index (Transformer, ResNet, BERT)       |
| `/examples/[slug]`      | `app/examples/[slug]/page.tsx`    | Individual example walkthrough                                |

### Key Components

| Component                    | File                                 | Purpose                                                 |
| ---------------------------- | ------------------------------------ | ------------------------------------------------------- |
| `BriefingDocument`           | `components/BriefingDocument.tsx`    | Renders the complete 6-section briefing document        |
| `BriefingSection`            | `components/BriefingSection.tsx`     | Single section with `react-markdown` rendering          |
| `ChatInput`                  | `components/ChatInput.tsx`           | Q&A message input field                                 |
| `ThinkingStream`             | `components/ThinkingStream.tsx`      | Live agent thinking/progress display                    |
| `ArtifactDownloads`          | `components/ArtifactDownloads.tsx`   | Download buttons for generated artifacts                |
| `AmbiguityCard`              | `components/AmbiguityCard.tsx`       | Renders/resolves a single ambiguity entry               |
| `CodeBlock`                  | `components/CodeBlock.tsx`           | Syntax-highlighted code with provenance badge           |
| `HyperparamTable`            | `components/HyperparamTable.tsx`     | Hyperparameter table with color-coded statuses          |
| `PrerequisiteCard`           | `components/PrerequisiteCard.tsx`    | Prerequisite concept explanation card                   |
| `PdfPanel`                   | `components/PdfPanel.tsx`            | PDF viewer panel (`react-pdf`)                          |
| `ExampleWalkthroughDocument` | `components/ExampleWalkthroughDocument.tsx` | Example walkthrough renderer                     |
| `ThemeToggle`                | `components/ThemeToggle.tsx`         | Dark/light theme toggle                                 |

### Custom Hook

- **`useSSE`** (`hooks/useSSE.ts`): POST-based SSE streaming. Manually parses SSE frames from `fetch` response body stream. Handles `token`, `done`, `error` events. Exposes `streamPost` function.

### Frontend ↔ Backend Communication

- All API calls go through `lib/api.ts` using `credentials: "include"` for cookie-based auth
- Ingestion progress uses GET-based SSE (`EventSource`)
- Q&A messages use POST-based SSE via `useSSE` hook
- `lib/types.ts` mirrors backend Pydantic models

### Fonts & Theming

- **Syne** (`--font-sans`): Body text
- **DM Mono** (`--font-mono`): Code, labels, tags
- **Instrument Serif** (`--font-serif`): Headlines
- Dark theme by default, `next-themes` for toggle

---

## 9. Coding Conventions & Patterns

### Python Backend

1. **`from __future__ import annotations`** in all modules for PEP 604 type hints
2. **Pydantic v2** models throughout (`BaseModel`, `model_dump()`, `model_validate()`)
3. **Async-first**: All IO and LLM calls are async. FastAPI routes are async
4. **Fallback pattern**: Every LLM call has `try/except` with a fallback strategy
5. **Database persistence**: All state persisted to PostgreSQL via Prisma — no in-memory session store
6. **Type annotations**: All function signatures fully typed
7. **Import style**: Absolute imports from `backend.*` namespace
8. **`langchain_classic` import**: `AgentExecutor` and `create_tool_calling_agent` come from `langchain_classic.agents` — not `langchain.agents`. This is due to breaking changes between LangChain versions
9. **Tool definitions**: Use `@tool("tool_name")` decorator from `langchain_core.tools`. Tools are factory-built (closures binding `parsed_paper` or `internal_rep`)

### TypeScript Frontend

1. **Next.js App Router** with `"use client"` on all interactive components
2. **CSS Modules** for page-specific styles (e.g., `page.module.css`, `signin.module.css`)
3. **TailwindCSS** for utility styling (v3, configured in `tailwind.config.ts`)
4. **Types mirror backend**: `lib/types.ts` mirrors Pydantic models — keep them in sync
5. **No state management library**: React `useState` + prop drilling
6. **SSE handling**: Ingestion uses native `EventSource`; Q&A uses custom POST-based streaming via `useSSE`
7. **Cookie-based auth**: All API calls include `credentials: "include"` — no Authorization header

---

## 10. Common Modification Scenarios

### Adding a New Briefing Section

1. Add section system prompt in `backend/prompts/briefing_sections.py` → `SECTION_PROMPTS`
2. Update model selection in `_briefing_model_for_section()` in `briefing_agent.py`
3. Add node + edge in `build_briefing_graph()` in `graph.py`
4. Add `section_N` column to `Briefing` model in `prisma/schema.prisma`, run `prisma migrate`
5. Update `save_briefing_section()` in `db/queries.py`
6. Update `BriefingSectionKey` type in `frontend/lib/types.ts`
7. Update `BriefingDocument.tsx` rendering logic

### Adding a New Tool to the Q&A Agent

1. Create tool builder function in `backend/tools/` (use `@tool` decorator)
2. Export from `backend/tools/__init__.py`
3. Add to tool list in `_build_executor()` in `backend/agents/qa_agent.py`

### Changing the LLM Model

Models are instantiated directly in agent files:
- `briefing_agent.py` → `BRIEFING_MODEL_PRO` (gemini-2.5-pro), `BRIEFING_MODEL_FLASH` (gemini-2.5-flash)
- `qa_agent.py` → `QA_MODEL` (gemini-2.5-flash)
- `session_memory.py` → `MEMORY_MODEL` (gemini-2.5-flash-lite)
- `code_agent.py`, `comprehension_agent.py`, `ingestion_agent.py`, `vision_service.py` — each instantiate their own model

### Adding a New API Endpoint

1. Add route function in the appropriate router (`backend/routers/{auth,ingest,conversation}.py`)
2. Add `Depends(get_current_user)` for authenticated endpoints
3. Add corresponding client function in `frontend/lib/api.ts`
4. Update types if needed in `frontend/lib/types.ts`

### Modifying the InternalRepresentation Schema

1. Update fields in `InternalRepresentation` in `backend/models/briefing.py`
2. Update `COMPREHENSION_PROMPT` in `backend/prompts/comprehension.py` to match the new JSON schema
3. Update `frontend/lib/types.ts` to mirror the change
4. Update any tools or briefing section prompts that reference the changed fields

---

## 11. Known Gotchas & Edge Cases

1. **`langchain_classic` import**: `qa_agent.py` uses `langchain_classic.agents` for `AgentExecutor` and `create_tool_calling_agent`. If upgrading LangChain, verify this import still resolves correctly.

2. **`ConversationSummaryBufferMemory` import**: `session_memory.py` tries two import paths (`langchain.memory` → `langchain_classic.memory`). If both fail, memory is broken.

3. **Prisma JSON fields**: `parsed_paper`, `internal_rep`, `hyperparameters`, `ambiguities`, and `code_snippets` are stored as `Json` columns. The `_to_json()` helper in `queries.py` serializes Pydantic models. After retrieval, they are raw dicts — use `model_validate()` to reconstruct Pydantic instances.

4. **Auth cookie**: JWT is stored in an httpOnly cookie named `access_token`. In production, `Secure=true` and `SameSite` is configurable. In development, a random JWT secret is auto-generated per process.

5. **GitHub OAuth via `google_sub`**: GitHub users are stored with `google_sub = "github:{github_id}"`. This is a pragmatic reuse of an existing unique column.

6. **Free plan limit**: Users on the FREE plan can analyze at most 3 papers (`FREE_PLAN_LIMIT` in `db/queries.py`). The limit is checked in the ingestion router before starting the pipeline.

7. **Model selection per section**: The briefing pipeline uses `gemini-2.5-pro` for sections 2 (mechanism), 4 (implementation), and 5 (ambiguity) where deep reasoning matters, and `gemini-2.5-flash` for sections 1, 3, and 6 where speed is preferred.

8. **Capacity lock**: When `CAPACITY_LOCK=true`, the backend middleware returns 503 for all `/ingest` and `/conversation` requests. The frontend counterpart `NEXT_PUBLIC_CAPACITY_LOCK=true` shows example walkthroughs instead of the live analysis flow.

9. **Concurrency control**: Ingestion uses a semaphore (`INGEST_CONCURRENCY_LIMIT`) and pending counter (`INGEST_PENDING_LIMIT`) to prevent resource exhaustion during parallel paper analyses.

10. **PDF parsing heuristics**: Heading/equation/table detection is heuristic-based and may fail on unusual PDF layouts. The fallback is to classify everything as Section.

11. **Browser SSE handling**: Q&A uses POST-based SSE (not native `EventSource`). The `useSSE` hook manually parses SSE frames from the fetch response body.

12. **Production validation**: In production (`APP_ENV=production`), `main.py` validates that JWT_SECRET >= 32 chars, SESSION_SECRET >= 32 chars, all redirect/callback URLs use HTTPS, and no localhost URLs are used.

---

## 12. Gemini Model Usage Map

| Agent / Service           | Model                  | Purpose                                    |
| ------------------------- | ---------------------- | ------------------------------------------ |
| `ingestion_agent.py`      | `gemini-2.5-flash`     | Task extraction, prerequisites             |
| `vision_service.py`       | `gemini-2.5-flash`     | Figure description (multimodal)            |
| `comprehension_agent.py`  | `gemini-2.5-pro`       | InternalRepresentation (deep reasoning)    |
| `briefing_agent.py`       | `gemini-2.5-pro`       | Sections 2, 4, 5 (mechanism, impl, ambig)  |
| `briefing_agent.py`       | `gemini-2.5-flash`     | Sections 1, 3, 6 (summary, prereqs, training) |
| `code_agent.py`           | `gemini-2.5-flash`     | PyTorch code snippet generation            |
| `qa_agent.py`             | `gemini-2.5-flash`     | Q&A tool-calling agent                     |
| `session_memory.py`       | `gemini-2.5-flash-lite`| Conversation summary buffer                |

---

## 13. Dependencies Summary

### Python (`requirements.txt`)

| Package                  | Version    | Purpose                              |
| ------------------------ | ---------- | ------------------------------------ |
| `fastapi`                | 0.115.2    | Web framework                        |
| `uvicorn[standard]`      | 0.30.6     | ASGI server                          |
| `pymupdf`                | 1.24.10    | PDF parsing                          |
| `arxiv`                  | 2.1.3      | arXiv metadata API                   |
| `httpx`                  | >=0.27.2   | Async HTTP client                    |
| `langchain`              | >=0.3.0    | LLM framework                       |
| `langchain-community`    | >=0.3.0    | Community integrations               |
| `langchain-google-genai` | >=2.0.0    | Gemini LLM binding                   |
| `langgraph`              | >=0.2.0    | Agent state graphs                   |
| `sse-starlette`          | 2.1.3      | Server-Sent Events                   |
| `pydantic`               | 2.9.2      | Data validation                      |
| `python-multipart`       | 0.0.12     | File upload parsing                  |
| `Pillow`                 | 10.4.0     | Image processing                     |
| `prisma`                 | >=0.13.1   | Prisma ORM for PostgreSQL            |
| `authlib`                | >=1.3.2    | OAuth client                         |
| `python-jose`            | >=3.3.0    | JWT encoding/decoding                |
| `itsdangerous`           | >=2.2.0    | Session signing                      |

### Node.js (`frontend/package.json`)

| Package            | Version  | Purpose                        |
| ------------------ | -------- | ------------------------------ |
| `next`             | 14.2.15  | React framework                |
| `react`            | 18.3.1   | UI library                     |
| `react-dom`        | 18.3.1   | React DOM renderer             |
| `react-markdown`   | ^10.1.0  | Markdown rendering             |
| `remark-gfm`       | ^4.0.1   | GitHub Flavored Markdown       |
| `highlight.js`     | ^11.10.0 | Code syntax highlighting       |
| `react-pdf`        | ^9.1.1   | PDF rendering in browser       |
| `next-themes`      | ^0.4.6   | Theme toggling                 |
| `@prisma/client`   | ^6.6.0   | Prisma client (if server-side) |
| `tailwindcss`      | ^3.4.14  | Utility CSS framework          |
| `typescript`       | ^5.6.3   | Type checking                  |
