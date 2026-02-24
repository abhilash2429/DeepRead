# DeepRead

DeepRead is an AI system that turns machine learning research papers into implementation-ready technical briefings. It ingests a PDF (uploaded directly or fetched from arXiv), builds a structured internal representation of the paper's method, generates a six-section technical briefing streamed in real time, and then provides a tool-calling Q&A agent grounded in the paper, the generated briefing, and extracted artifacts.

The system is not a summarizer. It deconstructs a paper down to its component architecture, hyperparameter registry, ambiguity log, prerequisite dependency chain, and training recipe — the exact information an engineer needs to go from reading a paper to writing code.

---

## Agentic Pipeline

DeepRead operates as a multi-agent pipeline where each stage's output becomes the next stage's input. The entire flow is orchestrated through LangGraph state graphs backed by Google Gemini.

```
                            USER
                             |
                     Upload PDF / arXiv ID
                             |
                             v
               +----------------------------+
               |     INGESTION AGENT        |
               |                            |
               |  pdf_parser (PyMuPDF)      |
               |         |                  |
               |         v                  |
               |  Structural extraction     |
               |  (sections, equations,     |
               |   figures, tables,         |
               |   pseudocode)              |
               |         |                  |
               |         v                  |
               |  Vision Service            |
               |  (Gemini multimodal)       |
               |  parallel figure interp.   |
               |         |                  |
               |         v                  |
               |  Task + Prerequisite       |
               |  Extraction (Gemini)       |
               |                            |
               |  Output: ParsedPaper       |
               +-------------+--------------+
                             |
                             v
               +----------------------------+
               |   COMPREHENSION AGENT      |
               |                            |
               |  Full paper text           |
               |  + figure descriptions     |
               |         |                  |
               |         v                  |
               |  Gemini (structured JSON)  |
               |  single 1M-context call    |
               |                            |
               |  Output:                   |
               |  InternalRepresentation    |
               |  - problem_statement       |
               |  - method_summary          |
               |  - novelty                 |
               |  - component_graph         |
               |  - hyperparameter_registry |
               |  - ambiguity_log           |
               |  - training_procedure      |
               |  - prerequisite_concepts   |
               +-------------+--------------+
                             |
                             v
               +----------------------------+
               |     BRIEFING AGENT         |
               |                            |
               |  Six sections, sequential  |
               |  generation with per-      |
               |  section model-tier        |
               |  selection:                |
               |                            |
               |  S1: What It Does (flash)  |
               |  S2: Mechanism (pro)       |
               |  S3: Prerequisites (flash) |
               |  S4: Impl Map (pro)        |
               |  S5: What's Left Out (pro) |
               |  S6: How To Train (flash)  |
               |                            |
               |  Each section streamed as  |
               |  SSE tokens to frontend,   |
               |  persisted to DB per       |
               |  section completion        |
               +-------------+--------------+
                             |
                             v
               +----------------------------+
               |     QA AGENT               |
               |                            |
               |  LangChain tool-calling    |
               |  agent with 7 tools:       |
               |                            |
               |  - paper_section_lookup    |
               |  - equation_decoder        |
               |  - prerequisite_expander   |
               |  - background_knowledge    |
               |  - hyperparameter_extract  |
               |  - ambiguity_detector      |
               |  - code_snippet_generator  |
               |                            |
               |  Grounded in: paper text,  |
               |  briefing, IR, chat        |
               |  history with              |
               |  summary-buffer memory     |
               +-------------+--------------+
                             |
                             v
               +----------------------------+
               |     CODE AGENT             |
               |                            |
               |  On-demand PyTorch snippet |
               |  generation per component  |
               |  with provenance labels:   |
               |  paper-stated | inferred   |
               |  | assumed | missing       |
               +----------------------------+
```

Every agent emits structured SSE events (`thinking`, `section_token`, `progress`, `done`, `error`) so the frontend can render progressive results in real time. The ingestion and briefing agents stream status updates, while the QA agent streams response tokens.

---

## What DeepRead Produces

For every paper ingested, DeepRead generates and persists the following structured outputs.

### Six-Section Technical Briefing

Each section is generated with a dedicated prompt contract and a model-tier selection (Gemini Pro for reasoning-heavy sections, Gemini Flash for speed-focused sections).

| Section | Title | Purpose |
|---------|-------|---------|
| 1 | What This Paper Actually Does | Plain-language problem/proposal/significance. No equations, no jargon. |
| 2 | The Mechanism | Step-by-step method walkthrough. Decodes every equation symbol at first use, weaves in figure interpretation, chains explanations downward showing why each component is necessary. |
| 3 | What You Need To Already Know | Dependency-ordered prerequisite list. Each concept uses a 3-part structure: Problem, Solution, Usage in this paper. |
| 4 | The Full Implementation Map | All components in dependency order with plain-English roles, annotated PyTorch snippets with provenance labels, and implementation notes covering pitfalls and dimension checks. |
| 5 | What The Paper Left Out | Full ambiguity report organized by type: missing hyperparameters, undefined notation, underspecified architecture, missing training details, ambiguous loss functions. Each entry includes the section source, implementation consequence, and agent resolution with reasoning and confidence. |
| 6 | How To Train It | Complete hyperparameter table, full training loop recipe, preprocessing requirements, and implementation tricks. Proposes and justifies defaults for every missing value. |

### Internal Representation

The comprehension agent produces a structured JSON object that serves as the analytical backbone for all downstream agents:

- **Problem Statement** — what the paper addresses and why.
- **Method Summary** — concise description of the proposed approach.
- **Novelty** — what distinguishes this work from prior art.
- **Component Graph** — dependency edges between architectural components (parent/child relationships), visualizable as a directed graph.
- **Hyperparameter Registry** — every hyperparameter extracted from the paper, with its value, source section, status (`paper-stated`, `inferred`, `missing`), and a suggested default with reasoning when the value is absent.
- **Ambiguity Log** — every underspecified detail, typed by category, with an impact assessment, the agent's best-guess resolution, confidence score, and a slot for user override.
- **Training Procedure** — free-text training recipe covering optimizer, scheduler, loss, batch size, and convergence details.
- **Prerequisite Concepts** — foundational concepts the reader needs, structured as Problem/Solution/Usage-in-paper triples.

### Downloadable Artifacts

Users can export the following from the session workspace:

- `briefing.md` — the complete six-section briefing as markdown.
- `annotated_code.py` — all generated PyTorch snippets merged into a single file.
- `hyperparameters.csv` — the full hyperparameter registry as a spreadsheet.
- `ambiguity_report.md` — the complete ambiguity log with resolutions.

---

## System Architecture

### Backend

The backend is a Python application built on FastAPI, running fully async. All LLM orchestration uses LangChain and LangGraph with Google Gemini as the model provider.

| Layer | Technology |
|-------|-----------|
| Framework | FastAPI 0.115+, Uvicorn |
| LLM Provider | Google Gemini (gemini-2.5-pro and gemini-2.5-flash via langchain-google-genai) |
| Agent Orchestration | LangChain (tool-calling agents), LangGraph (state graphs) |
| PDF Parsing | PyMuPDF (fitz) with heuristic heading, equation, table, and pseudocode detection |
| Figure Interpretation | Gemini multimodal API via Pillow + base64 encoding |
| arXiv Fetching | arxiv library + httpx with multi-strategy fallback |
| Database | PostgreSQL via Prisma (prisma-client-py, async interface) |
| Authentication | Google OAuth + GitHub OAuth via Authlib, JWT in httpOnly cookies |
| Streaming | SSE via sse-starlette for ingestion progress and conversation tokens |
| Memory | LangChain ConversationSummaryBufferMemory (4000 token window) |

The backend enforces per-user paper access (all queries filter by `user_id`), free-plan quota enforcement before ingestion, and concurrency limits on simultaneous ingestion jobs.

### Frontend

The frontend is a Next.js 14 App Router application with React 18 and TypeScript.

| Layer | Technology |
|-------|-----------|
| Framework | Next.js 14 (App Router) |
| Styling | Tailwind CSS 3 + CSS Modules |
| PDF Rendering | react-pdf |
| Code Highlighting | highlight.js |
| Theming | next-themes (dark/light modes) |

The frontend consists of four primary views:

- **Landing Page** — animated hero with an inline SVG pipeline diagram, feature descriptions, and example paper walkthroughs.
- **Upload Page** — PDF upload with drag-and-drop, arXiv ID/URL input, and a live progress stream during ingestion.
- **Session Workspace** — split-panel layout with a resizable PDF viewer on the left, and the briefing document, Q&A chat, and artifact controls on the right. Briefing sections stream in progressively. The chat interface supports follow-up questions grounded in the full paper context.
- **Dashboard** — lists the authenticated user's previously analyzed papers with status indicators and quick-resume links.

Communication between frontend and backend uses two SSE patterns: native `EventSource` (GET-based) for ingestion status, and a custom POST-based SSE frame parser via the `useSSE` hook for conversation streaming (since `EventSource` does not support POST requests with a body).

### Persistence Schema

The database stores four entities through Prisma:

| Model | Purpose |
|-------|---------|
| `User` | Google/GitHub identity, plan tier (FREE/PRO), paper usage count |
| `Paper` | Parsed paper JSON, internal representation JSON, arXiv ID, processing status, linked to user |
| `Briefing` | Six section text fields, hyperparameters JSON, ambiguities JSON, code snippets JSON, linked 1:1 to paper |
| `QAMessage` | Role + content pairs forming the persisted conversation history per paper |

---

## Tool System

The QA agent has access to seven tools that are dynamically constructed per session, each binding the relevant paper and analysis data through closures.

### paper_section_lookup

Searches the parsed paper's structural elements by heading or content keyword. Returns matching sections with their page numbers and element types, giving the agent precise access to specific parts of the paper during conversation.

### equation_decoder

Retrieves equation text by label (e.g., `(1)`, `(3a)`). Matches against the extracted equation elements and returns the raw equation content with its surrounding context, enabling the agent to explain mathematical notation on demand.

### prerequisite_expander

Expands a prerequisite concept from the internal representation's concept list. Returns the full Problem/Solution/Usage-in-paper structure for a named concept, letting the agent provide deep explanations of foundational knowledge when the user asks about unfamiliar terms.

### background_knowledge_lookup

Queries a built-in knowledge base of landmark ML papers (Transformer, ResNet, etc.) for concise summaries. This gives the agent grounding in widely-referenced prior work without needing to re-derive context from the current paper.

### hyperparameter_extractor

Returns the full hyperparameter registry from the internal representation as formatted text. Includes names, values, source sections, statuses, and suggested defaults, providing the agent with the complete parameter landscape for implementation-oriented answers.

### ambiguity_detector

Scans the internal representation's ambiguity log and returns entries matching a query. Each entry includes the ambiguity type, implementation consequence, agent resolution, and confidence, allowing the agent to surface specific underspecified details during conversation.

### code_snippet_generator

Generates an annotated PyTorch code snippet for a named component by invoking the code agent. The code agent produces snippets with provenance labels (`paper-stated`, `inferred`, `assumed`, `missing`) on every implementation choice, assumption notes, and source section references.

---

## Authentication and Access Control

DeepRead uses a cookie-based OAuth flow supporting two identity providers:

1. **Google OAuth** — OpenID Connect flow via Authlib. On successful authentication, the backend upserts a `User` record in PostgreSQL and issues a signed JWT stored in an `httpOnly` cookie.
2. **GitHub OAuth** — standard OAuth2 authorization code flow. The backend fetches the user profile and primary email from the GitHub API, maps them to the same `User` model, and issues the same JWT cookie.

All ingestion and conversation endpoints require an authenticated user context extracted from the JWT cookie. Paper access is scoped to the owning user — every database query filters by `user_id`. Free-plan users have a hard cap on the number of papers they can analyze, enforced before the ingestion pipeline starts.

Production deployments enforce HTTPS-only cookies, validate that redirect URIs use HTTPS and non-localhost hostnames, and require JWT secrets of at least 32 characters.

---

## Streaming Architecture

DeepRead uses Server-Sent Events (SSE) for all real-time communication between backend and frontend.

**Ingestion streaming**: When a paper is submitted, the backend creates an `asyncio.Queue` and pushes status events (`thinking`, `progress`, `done`, `error`) as each pipeline stage completes. The frontend connects to `GET /ingest/{paper_id}/events` using a native `EventSource` and renders a live progress log.

**Briefing streaming**: During briefing generation, tokens are emitted per-section as `section_token` events. The frontend accumulates tokens and renders sections progressively, giving the user readable content within seconds of pipeline start rather than waiting for all six sections to complete.

**Conversation streaming**: Q&A responses are streamed token-by-token through `POST /conversation/{paper_id}/message`. Because `EventSource` does not support POST requests, the frontend uses a custom SSE frame parser in the `useSSE` hook that reads the fetch response body as a stream and manually parses SSE frames (`event:`, `data:`, double-newline delimiters).

---

## Repository Layout

```
DeepRead/
├── backend/
│   ├── main.py                  # FastAPI app, lifespan, middleware, router mounting
│   ├── agents/
│   │   ├── ingestion_agent.py   # PDF parse + vision + task extraction
│   │   ├── comprehension_agent.py # ParsedPaper -> InternalRepresentation
│   │   ├── briefing_agent.py    # LangGraph six-section generation pipeline
│   │   ├── qa_agent.py          # Tool-calling Q&A with LangChain AgentExecutor
│   │   ├── code_agent.py        # PyTorch snippet generation with provenance
│   │   └── graph.py             # Shared LangGraph utilities
│   ├── models/
│   │   ├── paper.py             # ParsedPaper, PaperElement, ElementType
│   │   ├── briefing.py          # InternalRepresentation, BriefingSection, and sub-models
│   │   └── artifacts.py         # CodeSnippet, ArtifactItem
│   ├── prompts/
│   │   ├── briefing_sections.py # Six section prompt contracts
│   │   ├── comprehension.py     # InternalRepresentation JSON schema prompt
│   │   ├── qa.py                # QA agent system prompt
│   │   ├── code_gen.py          # Code generation rules
│   │   └── figure.py            # Vision description rules
│   ├── routers/
│   │   ├── auth.py              # Google + GitHub OAuth, JWT, user endpoints
│   │   ├── ingest.py            # Upload, arXiv fetch, SSE events, PDF retrieval
│   │   └── conversation.py      # Message streaming, state, artifacts, ambiguity resolution
│   ├── services/
│   │   ├── pdf_parser.py        # PyMuPDF structural extraction
│   │   ├── arxiv_fetcher.py     # arXiv ID normalization, metadata, multi-strategy PDF download
│   │   └── vision_service.py    # Gemini multimodal figure interpretation
│   ├── tools/
│   │   ├── paper_tools.py       # paper_section_lookup, equation_decoder
│   │   ├── knowledge_tools.py   # prerequisite_expander, background_knowledge_lookup
│   │   ├── analysis_tools.py    # hyperparameter_extractor, ambiguity_detector
│   │   └── code_tools.py        # code_snippet_generator
│   ├── db/
│   │   ├── prisma.py            # Prisma client lifecycle
│   │   └── queries.py           # Database query layer
│   ├── memory/
│   │   └── session_memory.py    # ConversationSummaryBufferMemory management
│   └── background_knowledge/
│       └── landmark_papers.py   # Built-in summaries of Transformer, ResNet, etc.
├── frontend/
│   ├── app/
│   │   ├── page.tsx             # Landing page with pipeline SVG and examples
│   │   ├── upload/              # Upload page (PDF + arXiv input)
│   │   ├── session/[id]/        # Session workspace (PDF viewer + briefing + QA)
│   │   ├── dashboard/           # User paper history
│   │   ├── signin/              # Authentication page
│   │   └── layout.tsx           # Root layout, metadata, providers
│   ├── components/
│   │   ├── BriefingDocument.tsx  # Full briefing renderer
│   │   ├── BriefingSection.tsx   # Individual section with streaming
│   │   ├── ChatInput.tsx         # Q&A message input
│   │   ├── PdfPanel.tsx          # Resizable PDF viewer (react-pdf)
│   │   ├── ArtifactDownloads.tsx # Artifact export controls
│   │   ├── AmbiguityCard.tsx     # Ambiguity display + resolution input
│   │   ├── CodeBlock.tsx         # Syntax-highlighted code with provenance badges
│   │   ├── HyperparamTable.tsx   # Hyperparameter table with status coloring
│   │   ├── PrerequisiteCard.tsx  # Prerequisite concept display
│   │   ├── ThinkingStream.tsx    # Real-time thinking indicator
│   │   └── ThemeToggle.tsx       # Dark/light mode switch
│   ├── hooks/
│   │   └── useSSE.ts            # Custom POST-based SSE streaming hook
│   └── lib/
│       ├── api.ts               # Backend API client functions
│       ├── types.ts             # TypeScript types mirroring backend models
│       ├── prisma.ts            # Prisma client for frontend
│       └── examples.ts          # Example paper walkthrough data
├── prisma/
│   └── schema.prisma            # User, Paper, Briefing, QAMessage models
├── requirements.txt             # Python dependencies
└── package.json                 # Root-level dev runner (concurrently)
```

---

## API Surface

### Authentication

| Method | Path | Description |
|--------|------|-------------|
| GET | `/auth/google` | Initiate Google OAuth flow |
| GET | `/auth/google/callback` | Handle Google OAuth callback, issue JWT cookie |
| GET | `/auth/github` | Initiate GitHub OAuth flow |
| GET | `/auth/github/callback` | Handle GitHub OAuth callback, issue JWT cookie |
| GET | `/auth/me` | Return current authenticated user profile |
| POST | `/auth/logout` | Clear authentication cookie |

### Ingestion

| Method | Path | Description |
|--------|------|-------------|
| POST | `/ingest/upload` | Upload a PDF file, returns `paper_id` and SSE URL |
| POST | `/ingest/arxiv` | Ingest from arXiv reference, returns `paper_id` and SSE URL |
| GET | `/ingest/{paper_id}/events` | SSE stream of ingestion + briefing progress events |
| GET | `/ingest/{paper_id}/pdf` | Retrieve stored PDF bytes for frontend rendering |

### Conversation and Artifacts

| Method | Path | Description |
|--------|------|-------------|
| POST | `/conversation/{paper_id}/message` | Send a message, response streamed as SSE |
| GET | `/conversation/{paper_id}/state` | Get paper status, title, briefing availability |
| GET | `/conversation/{paper_id}/artifacts` | Get downloadable artifact files |
| POST | `/conversation/{paper_id}/resolve-ambiguity` | Submit a user resolution for a specific ambiguity |

### Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Returns `{"status": "ok"}` |
