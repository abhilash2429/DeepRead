# DeepRead — Agent Context Document

> This file provides full architectural context for AI agents working on the DeepRead codebase.
> Read this before making any changes.

---

## 1. Project Overview

**DeepRead** is a conversational ML-paper comprehension application. Users upload a PDF (or paste an arXiv ID/URL), and the system ingests, parses, and deeply analyzes the paper. It then guides users through a multi-stage, interactive conversation — explaining the problem, architecture, implementation details, ambiguities, and training recipe — powered by LLM agents built with LangChain, LangGraph, and Google Gemini.

### Core Value Proposition

- **Structured comprehension**: Papers are broken into an internal representation (problem, method, novelty, component graph, hyperparameters, ambiguities, prerequisites).
- **Stage-based tutoring**: Conversation flows through `orientation → architecture → implementation → ambiguity → training` stages, with a freeform Q&A fallback.
- **Code generation**: On-demand annotated PyTorch code snippets per component, with provenance labels (`paper-stated`, `inferred`, `assumed`, `missing`).
- **Ambiguity resolution**: Surfaces underspecified details one at a time and lets the user lock in decisions.
- **Downloadable artifacts**: Architecture summaries, annotated code, hyperparameter CSVs, ambiguity reports.

---

## 2. Tech Stack

| Layer      | Technology                                                        |
| ---------- | ----------------------------------------------------------------- |
| Backend    | Python 3.11+, FastAPI, Uvicorn                                    |
| LLM        | Google Gemini (`gemini-flash-latest` via `langchain-google-genai`) |
| Agents     | LangChain (tool-calling agents), LangGraph (state graph)          |
| PDF parse  | PyMuPDF (`fitz`), Pillow                                          |
| arXiv      | `arxiv` library + `httpx` for fallback PDF download               |
| Frontend   | Next.js 14 (App Router), React 18, TypeScript, TailwindCSS 3     |
| Extras     | `react-pdf` (PDF viewer), `reactflow` (component graph), `highlight.js` (code blocks) |
| Dev runner | `concurrently` (root `package.json`) or Windows `.cmd` scripts    |

### Environment Variables (`.env`)

| Variable                | Purpose                                    |
| ----------------------- | ------------------------------------------ |
| `GEMINI_API_KEY`        | **Required.** Google AI API key.           |
| `MAX_PAPER_SIZE_MB`     | Max PDF upload size (default `20`).        |
| `SESSION_DIR`           | Directory for persisted session data.      |
| `LANGCHAIN_TRACING_V2`  | Enable LangSmith tracing (`true`/`false`). |
| `LANGCHAIN_API_KEY`     | LangSmith API key (optional).              |
| `LANGCHAIN_PROJECT`     | LangSmith project name.                    |
| `NEXT_PUBLIC_API_BASE`  | Frontend → backend URL (default `http://localhost:8000`). |

---

## 3. Repository Structure

```
DeepRead/
├── backend/
│   ├── __init__.py
│   ├── main.py                      # FastAPI app creation, lifespan, middleware, router mounting
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── graph.py                 # LangGraph state graph: router → stage nodes → END
│   │   ├── comprehension_agent.py   # Builds InternalRepresentation from ParsedPaper
│   │   ├── conversation_agent.py    # Orchestrates a single conversation turn via the graph
│   │   ├── ingestion_agent.py       # PDF parse + vision + task extraction pipeline
│   │   └── code_agent.py            # Generates annotated PyTorch CodeSnippets
│   ├── models/
│   │   ├── __init__.py
│   │   ├── paper.py                 # ParsedPaper, PaperElement, ElementType, ProvenanceLabel
│   │   ├── conversation.py          # ConversationState, InternalRepresentation, Stage, etc.
│   │   └── artifacts.py             # CodeSnippet, ArtifactItem, ArtifactManifest
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── stages.py                # STAGE_PROMPTS dict (system prompts per stage)
│   │   ├── comprehension.py         # COMPREHENSION_PROMPT (JSON schema for InternalRepresentation)
│   │   ├── code_gen.py              # CODE_GEN_PROMPT (PyTorch code generation rules)
│   │   └── figure.py                # FIGURE_PROMPT (vision description rules)
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── ingest.py                # /ingest/* endpoints (upload, arxiv, events, pdf)
│   │   └── conversation.py          # /conversation/* endpoints (message, state, artifacts, resolve-ambiguity)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── gemini_client.py         # GeminiClient: LangChain wrapper for Gemini API
│   │   ├── pdf_parser.py            # parse_pdf(): PyMuPDF-based structured PDF extraction
│   │   ├── arxiv_fetcher.py         # arXiv metadata + PDF download with fallback strategies
│   │   ├── artifact_builder.py      # build_artifacts(): assembles downloadable artifact manifest
│   │   └── vision_service.py        # describe_figure(): multimodal figure analysis
│   ├── store/
│   │   ├── __init__.py
│   │   └── session_store.py         # In-memory + disk-persisted session storage with TTL cleanup
│   ├── memory/
│   │   └── session_memory.py        # ConversationSummaryBufferMemory management (LangChain)
│   └── tools/
│       ├── __init__.py              # Re-exports all tool builder functions
│       ├── paper_tools.py           # paper_section_lookup, equation_decoder
│       ├── knowledge_tools.py       # prerequisite_expander, background_knowledge_lookup
│       ├── analysis_tools.py        # hyperparameter_extractor, ambiguity_detector
│       └── code_tools.py            # code_snippet_generator (wraps code_agent)
├── frontend/
│   ├── app/
│   │   ├── globals.css
│   │   ├── layout.tsx               # Root layout (imports TailwindCSS, sets metadata)
│   │   ├── page.tsx                 # Home page: arXiv input + PDF upload + progress tracking
│   │   └── session/[id]/page.tsx    # Session page: PDF viewer + chat + graph + artifacts
│   ├── components/
│   │   ├── ChatPanel.tsx            # Main chat UI: stage tabs, message list, SSE streaming, code blocks
│   │   ├── ArtifactPanel.tsx        # Dropdown to download generated artifacts
│   │   ├── AmbiguityCard.tsx        # Displays/resolves a single ambiguity entry
│   │   ├── CodeBlock.tsx            # Syntax-highlighted Python code with provenance badge
│   │   ├── ComponentGraph.tsx       # ReactFlow-based component dependency graph
│   │   └── HyperparamTable.tsx      # Hyperparameter table with status color coding
│   ├── hooks/
│   │   └── useSSE.ts                # Custom hook: POST-based SSE streaming (token/stage/done/error)
│   ├── lib/
│   │   ├── api.ts                   # API client functions (ingest, conversation, artifacts)
│   │   └── types.ts                 # TypeScript types mirroring backend Pydantic models
│   ├── package.json
│   ├── next.config.js
│   ├── tailwind.config.ts
│   ├── postcss.config.js
│   └── tsconfig.json
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt                 # Python dependencies
├── package.json                     # Root: concurrently runs backend + frontend
├── run-all.cmd                      # Windows: starts both services in separate terminals
├── run-backend.cmd
├── run-frontend.cmd
└── sample_test.pdf                  # Test PDF for local development
```

---

## 4. Architecture & Data Flow

### 4.1 Ingestion Pipeline

```
User uploads PDF / enters arXiv ID
        │
        ▼
  ┌─────────────────────────────────┐
  │  Router: /ingest/upload or      │
  │          /ingest/arxiv           │
  │  Creates session, starts async  │
  │  pipeline, returns session_id   │
  └────────────┬────────────────────┘
               │
               ▼
  ┌─────────────────────────────────┐
  │  run_ingestion()                │
  │  1. parse_pdf() → ParsedPaper   │
  │  2. describe_figure() per image │  ← Gemini multimodal
  │  3. Extract primary_task +      │  ← Gemini text
  │     prerequisites               │
  └────────────┬────────────────────┘
               │
               ▼
  ┌─────────────────────────────────┐
  │  run_comprehension()            │
  │  Produces InternalRepresentation│  ← Gemini (structured JSON)
  │  (problem, method, novelty,     │
  │   component_graph, hyperparams, │
  │   ambiguity_log, training,      │
  │   prerequisites)                │
  └────────────┬────────────────────┘
               │
               ▼
  ┌─────────────────────────────────┐
  │  ConversationState initialized  │
  │  Session persisted to store     │
  │  SSE "done" event sent          │
  │  Frontend redirects to          │
  │  /session/{session_id}          │
  └─────────────────────────────────┘
```

**Status streaming**: The ingestion pipeline emits SSE events (`status`, `done`, `error`) via an `asyncio.Queue`. The frontend listens on `GET /ingest/{session_id}/events`.

### 4.2 Conversation Flow (LangGraph)

```
User message arrives at POST /conversation/{session_id}/message
        │
        ▼
  ┌─────────────────────────────────┐
  │  conversation_agent.run_turn()  │
  │  Loads memory, calls graph      │
  └────────────┬────────────────────┘
               │
               ▼
  ┌─────────────────────────────────┐
  │  LangGraph: DeepReadState       │
  │                                 │
  │  Entry: router_node             │
  │    └→ _classify_route()         │  ← Gemini classifies intent
  │       Returns: orientation |    │
  │       architecture |            │
  │       implementation |          │
  │       ambiguity | training |    │
  │       freeqa                    │
  │                                 │
  │  Conditional edge → stage node  │
  │    └→ run_stage_node()          │
  │       • Builds tool set         │
  │       • Creates AgentExecutor   │  ← LangChain tool-calling agent
  │       • Invokes with chat       │
  │         history + user message  │
  │       • If implementation:      │
  │         generates CodeSnippet   │
  │       • If ambiguity:           │
  │         surfaces next question  │
  │                                 │
  │  Each stage node → END          │
  └────────────┬────────────────────┘
               │
               ▼
  Response streamed to frontend as SSE tokens
  (event types: progress, stage, clarifying, token, done, error)
```

### 4.3 Tool System

The LangGraph agent has access to these tools within each conversation turn:

| Tool                        | Module                  | Purpose                                                    |
| --------------------------- | ----------------------- | ---------------------------------------------------------- |
| `paper_section_lookup`      | `tools/paper_tools.py`  | Search parsed paper sections by heading or content keyword  |
| `equation_decoder`          | `tools/paper_tools.py`  | Retrieve equation text by label (e.g., `(1)`)              |
| `prerequisite_expander`     | `tools/knowledge_tools.py` | Explain a prerequisite concept from the IR                |
| `background_knowledge_lookup` | `tools/knowledge_tools.py` | Built-in summaries of landmark papers (Transformer, ResNet, etc.) |
| `hyperparameter_extractor`  | `tools/analysis_tools.py` | Return the full hyperparameter registry as text            |
| `ambiguity_detector`        | `tools/analysis_tools.py` | Scan section text for ambiguity markers                    |
| `code_snippet_generator`    | `tools/code_tools.py`   | Generate annotated PyTorch code for a named component      |

---

## 5. Key Data Models

### 5.1 `ParsedPaper` (`backend/models/paper.py`)

The structured representation of an ingested PDF:

- `title`, `authors`, `abstract`, `full_text`
- `elements: list[PaperElement]` — sections, equations, figures, tables, pseudocode
- `primary_task`, `prerequisites_raw`

Each `PaperElement` has: `id`, `element_type` (enum), `section_heading`, `page_number`, `content`, `caption`, `equation_label`, `image_bytes_b64`, `figure_description`.

### 5.2 `InternalRepresentation` (`backend/models/conversation.py`)

The LLM-generated deep analysis of the paper:

- `problem_statement`, `method_summary`, `novelty`
- `component_graph: list[DependencyEdge]` — parent/child component relationships
- `hyperparameter_registry: list[HyperparameterEntry]` — name, value, source, status, suggested default
- `ambiguity_log: list[AmbiguityEntry]` — ambiguities with impact, best guess, resolution status
- `training_procedure` — free-text training recipe
- `prerequisite_concepts: list[ConceptExplanation]` — concept + explanation pairs

### 5.3 `ConversationState` (`backend/models/conversation.py`)

Per-session state tracking:

- `session_id`, `current_stage` (Stage enum)
- `message_history: list[ChatMessage]`
- `resolved_ambiguities: dict[str, str]`
- `internal_representation: InternalRepresentation`
- `user_level` (`"student"` or `"practitioner"`, auto-detected from vocabulary)
- `pending_question`, `last_component_focus`, `current_component_index`, `metadata`

### 5.4 `Stage` Enum

```python
class Stage(str, Enum):
    ORIENTATION = "orientation"
    ARCHITECTURE = "architecture"
    IMPLEMENTATION = "implementation"
    AMBIGUITY = "ambiguity"
    TRAINING = "training"
```

Note: `"freeqa"` is a valid route key but not a Stage enum member; the graph handles it as a node.

### 5.5 `CodeSnippet` (`backend/models/artifacts.py`)

- `component_name`, `code`, `provenance` (ProvenanceLabel), `assumption_notes`, `source_sections`

### 5.6 `ProvenanceLabel` (`backend/models/paper.py`)

```python
ProvenanceLabel = Literal["paper-stated", "inferred", "assumed", "missing"]
```

Used throughout the system to track information provenance.

---

## 6. API Endpoints

### Ingestion

| Method | Path                           | Description                                      |
| ------ | ------------------------------ | ------------------------------------------------ |
| POST   | `/ingest/upload`               | Upload a PDF file. Returns `session_id` + SSE URL |
| POST   | `/ingest/arxiv`                | Ingest from arXiv ref. Returns `session_id` + SSE URL |
| GET    | `/ingest/{session_id}/events`  | SSE stream of ingestion progress events          |
| GET    | `/ingest/{session_id}/pdf`     | Retrieve the stored PDF bytes for rendering      |

### Conversation

| Method | Path                                          | Description                                    |
| ------ | --------------------------------------------- | ---------------------------------------------- |
| POST   | `/conversation/{session_id}/message`          | Send a message; response streamed as SSE       |
| GET    | `/conversation/{session_id}/state`            | Get full conversation state JSON               |
| GET    | `/conversation/{session_id}/artifacts`        | Get downloadable artifact manifest             |
| POST   | `/conversation/{session_id}/resolve-ambiguity`| Resolve a specific ambiguity by ID             |

### Health

| Method | Path      | Description       |
| ------ | --------- | ----------------- |
| GET    | `/health` | Returns `{"status": "ok"}` |

### SSE Event Types (Conversation)

| Event        | Payload                             | Description                             |
| ------------ | ----------------------------------- | --------------------------------------- |
| `progress`   | `{ message: string }`              | Thinking/processing indicator           |
| `stage`      | `{ current_stage, reason }`        | Stage transition notification           |
| `clarifying` | `{ question: string }`             | Ambiguity question for user             |
| `token`      | `{ text: string }`                 | Streamed response token                 |
| `done`       | `{ message_id: number }`           | Response complete                       |
| `error`      | `{ message: string }`              | Error during processing                 |

---

## 7. Session & Memory System

### SessionStore (`backend/store/session_store.py`)

- **Dual-layer storage**: In-memory `dict` + disk persistence (JSON/binary files per session).
- **TTL cleanup**: Background task cleans up sessions older than 2 hours (configurable).
- **Serialization**: Pydantic models (`ParsedPaper`, `ConversationState`, `InternalRepresentation`, `CodeSnippet` lists) are serialized with type metadata wrappers. Binary data (PDF bytes) stored as `.bin` files.
- **Disk path**: `{SESSION_DIR}/{session_id}/{key}.json` or `.bin`.

### SessionMemoryManager (`backend/memory/session_memory.py`)

- Uses LangChain's `ConversationSummaryBufferMemory` (max 4000 tokens).
- Persists message history as JSON at `{SESSION_DIR}/{session_id}/memory.json`.
- Falls back across LangChain import paths (`langchain.memory` → `langchain_classic.memory`).

---

## 8. Frontend Architecture

### Pages

- **`/` (Home)**: Upload PDF or enter arXiv reference. Shows progress bar and live event log during ingestion. On success, redirects to session page.
- **`/session/[id]`**: Three-panel layout:
  - **Left**: Resizable PDF viewer (`react-pdf`).
  - **Right top**: Header bar with stage badge, user level, component graph toggle, artifact panel.
  - **Right main**: `ChatPanel` — stage tabs, message history, streaming responses, code blocks, hyperparameter table (training stage), ambiguity cards (ambiguity stage).

### Key Components

| Component          | File                            | Purpose                                                   |
| ------------------ | ------------------------------- | --------------------------------------------------------- |
| `ChatPanel`        | `components/ChatPanel.tsx`      | Full chat interface: send messages, stream SSE responses, render text/code segments, show stage-specific panels |
| `ArtifactPanel`    | `components/ArtifactPanel.tsx`  | Dropdown that fetches and offers artifact downloads        |
| `AmbiguityCard`    | `components/AmbiguityCard.tsx`  | Renders a single ambiguity with resolve input              |
| `CodeBlock`        | `components/CodeBlock.tsx`      | Syntax-highlighted code with provenance badge, copy button, assumption notes toggle |
| `ComponentGraph`   | `components/ComponentGraph.tsx` | ReactFlow graph of component dependencies; clicking a node triggers architecture explanation |
| `HyperparamTable`  | `components/HyperparamTable.tsx`| Rendered table with color-coded provenance statuses        |

### Custom Hook

- **`useSSE`** (`hooks/useSSE.ts`): Performs a `POST` fetch with `Accept: text/event-stream`, manually parses SSE frames from the response body stream. Handles `token`, `stage`, `progress`, `clarifying`, `done`, `error` events. Exposes `connected` state and `streamPost` function.

### Frontend ↔ Backend Communication

- All API calls go through `lib/api.ts` which uses `NEXT_PUBLIC_API_BASE` (defaults to `http://localhost:8000`).
- Ingestion progress uses native `EventSource` (GET-based SSE).
- Conversation messages use a custom POST-based SSE pattern via `useSSE` hook (since `EventSource` doesn't support POST).

---

## 9. How to Run

### Prerequisites

- Python 3.11+ with a virtual environment
- Node.js 18+
- A valid `GEMINI_API_KEY`

### Quick Start (both services)

```bash
# From project root
cp .env.example .env         # Fill in GEMINI_API_KEY
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
npm install                   # Root concurrently
cd frontend && npm install && cd ..
npm run dev                   # Starts backend (port 8000) + frontend (port 3000)
```

### Backend Only

```bash
uvicorn backend.main:app --reload --port 8000
```

### Frontend Only

```bash
cd frontend
npm run dev
```

### Windows `.cmd` Alternative

```
run-all.cmd       # Opens two terminal windows
```

---

## 10. Coding Conventions & Patterns

### Python Backend

1. **All modules use `from __future__ import annotations`** for PEP 604 type hints.
2. **Pydantic v2** models throughout (`BaseModel`, `model_dump()`, `model_validate()`).
3. **Async-first**: All IO and LLM calls are `async`. FastAPI routes are async.
4. **Fallback pattern**: Every LLM call has a `try/except` with a fallback strategy (simpler prompt, keyword matching, etc.). This is critical — never remove fallback blocks.
5. **No database**: State is entirely in-memory + filesystem. No SQL, no Redis.
6. **Type annotations**: All function signatures are fully typed.
7. **Import style**: Absolute imports from `backend.*` namespace.
8. **Tool definitions**: Use `@tool("tool_name")` decorator from `langchain_core.tools`. Tools are factory-built (closures binding `parsed_paper` or `internal_rep`).

### TypeScript Frontend

1. **Next.js App Router** with `"use client"` directives on all interactive components.
2. **TailwindCSS** for styling (v3, configured in `tailwind.config.ts`).
3. **Types mirror backend**: `lib/types.ts` mirrors Pydantic models. Keep them in sync.
4. **No state management library**: React `useState` + prop drilling.
5. **SSE handling**: Ingestion uses native `EventSource`; conversation uses custom POST-based stream parsing in `useSSE`.
6. **Custom events**: `deepread-component-focus` dispatched from `ComponentGraph` → caught by `ChatPanel` to trigger architecture stage explanations.

---

## 11. Key Implementation Details

### PDF Parsing (`services/pdf_parser.py`)

- Uses **PyMuPDF** (`fitz`) to extract text blocks, detect headings (by font size/boldness), equations (Greek chars, equation tokens), tables, and pseudocode.
- Images are extracted per page, converted to PNG, base64-encoded, and stored as `PaperElement` entries.
- Heading detection uses `_looks_like_heading()` with font size threshold (`>= max(12.0, body_size + 1.0)` or bold ratio ≥ 0.8).
- Equation detection: numbered patterns `(N)` at end of line, Greek character count ≥ 2, or math tokens.

### LangGraph State (`agents/graph.py`)

- `DeepReadState` is a `TypedDict` (not Pydantic) — required by LangGraph.
- The graph compiles a new instance per `run_graph_turn()` call (stateless compilation).
- Route classification uses Gemini with a `RouteDecision` Pydantic model; falls back to keyword matching on failure.
- `AgentExecutor` uses `create_tool_calling_agent` from `langchain_classic.agents` — note the `langchain_classic` import, not `langchain`.

### GeminiClient (`services/gemini_client.py`)

- Wraps `ChatGoogleGenerativeAI` from `langchain-google-genai`.
- Provides: `generate_text()`, `stream_text()`, `generate_multimodal_text()`, `generate_json()`.
- `generate_json()` has a two-pass strategy: parse → if fail, ask Gemini to repair → parse again.
- `extract_json_candidate()` strips markdown fences and finds JSON boundaries.

### arXiv Fetcher (`services/arxiv_fetcher.py`)

- Normalizes arXiv references (URLs, bare IDs, with/without version suffixes).
- Three-tier fetching: `arxiv.Client` → abs page metadata fallback → multiple PDF URL attempts.
- Handles `WinError 10013` (Windows firewall) with a specific user-friendly error message.

### Artifact Builder (`services/artifact_builder.py`)

Produces an `ArtifactManifest` with:
- `architecture_summary.md` — problem/method/novelty
- Per-component `.py` files
- `annotated_code.py` — all snippets merged
- `hyperparameters.csv` — full registry
- `ambiguity_report.md` — all ambiguity entries

---

## 12. Common Modification Scenarios

### Adding a New Conversation Stage

1. Add to `Stage` enum in `backend/models/conversation.py`.
2. Add system prompt in `backend/prompts/stages.py` → `STAGE_PROMPTS`.
3. Add to `RouteDecision.intent` literal in `backend/agents/graph.py`.
4. Add node + edge in `build_conversation_graph()` in `graph.py`.
5. Add keyword fallback in `_classify_route()`.
6. Update `Stage` type in `frontend/lib/types.ts`.
7. Add to `stages` array in `frontend/components/ChatPanel.tsx`.

### Adding a New Tool

1. Create tool builder function in `backend/tools/` (use `@tool` decorator).
2. Export from `backend/tools/__init__.py`.
3. Call builder in `run_stage_node()` in `backend/agents/graph.py` and add to `tools` list.

### Changing the LLM Model

1. Update `model_name` in `backend/main.py` lifespan → `GeminiClient(model_name=...)`.
2. Or set a new env var and read it in `main.py`.

### Adding a New API Endpoint

1. Add route function in the appropriate router (`backend/routers/ingest.py` or `conversation.py`).
2. Add corresponding client function in `frontend/lib/api.ts`.
3. Update types if needed in `frontend/lib/types.ts`.

### Modifying the InternalRepresentation Schema

1. Update fields in `InternalRepresentation` in `backend/models/conversation.py`.
2. Update `COMPREHENSION_PROMPT` in `backend/prompts/comprehension.py` to match the new JSON schema.
3. Update `frontend/lib/types.ts` to mirror the change.
4. Update any tools or artifact builders that reference the changed fields.

---

## 13. Known Gotchas & Edge Cases

1. **`langchain_classic` import**: The codebase uses `langchain_classic.agents` (not `langchain.agents`) for `AgentExecutor` and `create_tool_calling_agent`. This is due to API changes between LangChain versions. If upgrading LangChain, verify this import.

2. **`ConversationSummaryBufferMemory` import**: `session_memory.py` tries two import paths. If both fail, the feature is disabled. Check LangChain version compatibility.

3. **`freeqa` is not a `Stage`**: It's a valid `route_key` in the graph but not in the `Stage` enum. The router treats it as a node, and it shares `run_stage_node` logic with the other stages.

4. **Session store does not auto-recover**: If the server restarts, in-memory session data is lost. Disk-persisted data is reloaded on first access, but active SSE connections will break.

5. **No authentication**: All endpoints are open. CORS is set to `allow_origins=["*"]`.

6. **PDF parsing heuristics**: Heading/equation/table detection is heuristic-based and may fail on unusual PDF layouts. The fallback is to classify everything as `Section`.

7. **Gemini rate limits**: No built-in rate limiting or retry logic for Gemini API calls. Bulk ingestion may hit quota limits.

8. **Browser SSE handling**: Conversation uses POST-based SSE (not native `EventSource`). The `useSSE` hook manually parses SSE frames from the fetch response body. If modifying streaming behavior, be aware of this custom implementation.

9. **Image processing**: Figure images are extracted, converted to PNG via Pillow, and base64-encoded. Very large papers with many figures may cause significant memory usage during ingestion.

10. **`set` type in Pydantic**: `ConversationState.prerequisites_explained` uses `set[str]` with Pydantic, which serializes to a list. Be careful with set operations after deserialization.

---

## 14. Testing

There are currently no automated tests in the repository. To manually test:

1. Start both services (`npm run dev` from root).
2. Upload `sample_test.pdf` or use an arXiv ID (e.g., `2310.06825`).
3. Wait for ingestion to complete (watch SSE progress).
4. Ask questions across different stages.
5. Check artifacts download.
6. Resolve an ambiguity.

---

## 15. Dependencies Summary

### Python (`requirements.txt`)

| Package                | Version    | Purpose                           |
| ---------------------- | ---------- | --------------------------------- |
| `fastapi`              | 0.115.2    | Web framework                     |
| `uvicorn[standard]`    | 0.30.6     | ASGI server                       |
| `pymupdf`              | 1.24.10    | PDF parsing                       |
| `arxiv`                | 2.1.3      | arXiv metadata API                |
| `httpx`                | >=0.27.2   | Async HTTP client                 |
| `langchain`            | >=0.3.0    | LLM framework                    |
| `langchain-community`  | >=0.3.0    | Community integrations            |
| `langchain-google-genai` | >=2.0.0  | Gemini LLM binding                |
| `langgraph`            | >=0.2.0    | Agent state graphs                |
| `sse-starlette`        | 2.1.3      | Server-Sent Events                |
| `pydantic`             | 2.9.2      | Data validation                   |
| `python-multipart`     | 0.0.12     | File upload parsing               |
| `Pillow`               | 10.4.0     | Image processing                  |

### Node.js (`frontend/package.json`)

| Package        | Version  | Purpose                     |
| -------------- | -------- | --------------------------- |
| `next`         | 14.2.15  | React framework             |
| `react`        | 18.3.1   | UI library                  |
| `react-dom`    | 18.3.1   | React DOM renderer          |
| `highlight.js` | ^11.10.0 | Code syntax highlighting    |
| `reactflow`    | ^11.11.4 | Node graph visualization    |
| `react-pdf`    | ^9.1.1   | PDF rendering in browser    |
| `tailwindcss`  | ^3.4.14  | Utility CSS framework       |
| `typescript`   | ^5.6.3   | Type checking               |
