# AGENTS.md — ML Paper Comprehension Agent
### Instructions for Codex-5.3

---

## What You Are Building

A web application where a user submits an ML research paper (PDF upload or arXiv link) and gets back an **Intelligent Briefing** — a single, dense, structured document that contains everything a practitioner or student needs to understand and implement the paper. The agent reads the paper, builds an internal structured understanding, then generates the briefing in one pass. The briefing is the primary deliverable. A chat interface sits beneath it for follow-up questions.

**The core requirement:** The agent must fill the gap between what the paper states and what a practitioner or student needs to know to implement it. When the paper says "standard layer normalization," the agent explains what layer norm is, why it's used here, and shows the implementation inline. When notation is undefined, the agent decodes it at the point of use. When a hyperparameter is missing, the agent proposes a justified default. Everything is labeled: **paper-stated**, **inferred**, or **assumed/missing**.

The briefing is not a summary. It is a transformed version of the paper — same information density, but reorganized around implementation understanding rather than novelty demonstration.

---

## LLM and Agent Framework

Use the **Gemini 2.5 model family** as the LLM provider via `langchain-google-genai`. All LLM calls go through LangChain.

**Gemini Setup**
```python
from langchain_google_genai import ChatGoogleGenerativeAI

def get_model(model_name: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=os.environ["GEMINI_API_KEY"],
    )
```

**Model Assignment — use the right model for each workload**

Every agent uses a specific model chosen for its workload. Do not use a single model everywhere.

| Agent / Task | Model | Reason |
|---|---|---|
| Comprehension agent (full paper  InternalRepresentation) | `gemini-2.5-pro` | Most critical call. Exhaustive ambiguity detection and hyperparameter extraction require the strongest reasoning in the family. Everything downstream depends on this output. |
| Briefing Section 2 — The Mechanism | `gemini-2.5-pro` | Equation decoding, symbol-by-symbol explanation, and prerequisite chaining are deep multi-step reasoning tasks. Downgrading produces shallow output. |
| Briefing Section 4 — Implementation Map | `gemini-2.5-pro` | Code generation with inline equation citations and correctly labeled ASSUMED/INFERRED annotations requires careful step-by-step reasoning. |
| Briefing Section 5 — What The Paper Left Out | `gemini-2.5-pro` | Second most important reasoning task after comprehension. Weak models miss subtle ambiguities. |
| Briefing Sections 1, 3, 6 | `gemini-2.5-flash` | Plain-English summary, prerequisites list, and training recipe are structured but not deep reasoning tasks. Flash handles these at lower cost and higher speed. |
| Vision service (figure interpretation) | `gemini-2.5-flash` | Multimodal single-turn description task. Pro is overkill here. |
| Code agent (on-demand snippets from Q&A) | `gemini-2.5-flash` | Code generation without the heavy annotation burden of Section 4. Flash is sufficient. |
| Q&A agent (conversational tool calling) | `gemini-2.5-flash` | Low latency matters in chat. Flash's output speed is a real advantage here. |
| Memory summarization | `gemini-2.5-flash-lite` | Background compression task. Cheapest model in the family, quality irrelevant. |

Instantiate each model once at module level and reuse — do not create new instances per request.

**Agent Framework — LangChain Primitives to Use**
- Use **LangGraph** for the two-phase workflow: Phase 1 is briefing generation (a linear chain of nodes), Phase 2 is free-form Q&A (a single node with tool access). There are no stages, no stage routing, no stage transitions.
- Use **LangChain LCEL** (LangChain Expression Language) for all chain construction — `prompt | llm | output_parser` pipelines throughout. Do not write manual LLM call loops.
- Use **LangChain Tools** (`@tool` decorator) for all discrete agent capabilities: `paper_section_lookup`, `equation_decoder`, `prerequisite_expander`, `code_snippet_generator`, `hyperparameter_extractor`, `ambiguity_detector`, `figure_interpreter`, `background_knowledge_lookup`. The Q&A agent invokes these via `create_tool_calling_agent`.
- Use **LangChain Memory** — `ConversationSummaryBufferMemory` backed by `gemini-2.5-flash-lite` for managing Q&A message history. The briefing itself is not part of memory — it is stored in the database and loaded into context per request.
- Use **LangChain's `PydanticOutputParser`** for all structured outputs (InternalRepresentation, BriefingSection, CodeSnippet, AmbiguityEntry, HyperparameterEntry). Pass format instructions into every relevant prompt via `parser.get_format_instructions()`.
- Use **LangChain's `ChatPromptTemplate`** with `MessagesPlaceholder` for the Q&A agent prompt.
- Use **LangChain's document loaders and text splitters only for the background knowledge module** — the main paper text goes to `gemini-2.5-pro` in full, no splitting.

---

## Tech Stack

**Backend**
- `FastAPI` — API server with async support
- `pymupdf` (import as `fitz`) — PDF parsing, text extraction, figure/image cropping
- `arxiv` Python library — fetch paper metadata and PDF URL by arXiv ID
- `httpx` — async download of arXiv PDFs
- `langchain` — core LangChain framework
- `langchain-google-genai` — LangChain integration for Gemini 2.5 family
- `langchain-community` — community tools and utilities
- `langgraph` — two-phase agent workflow as a state graph
- `sse-starlette` — server-sent events for streaming agent responses to frontend
- `pydantic v2` — all data models, used with LangChain's PydanticOutputParser
- `python-multipart` — PDF file upload handling
- `Pillow` — image processing for extracted figures
- `prisma` — Prisma Python client for database access (`prisma-client-py`)
- `authlib` — Google OAuth2 flow handling
- `python-jose` — JWT token creation and validation
- `itsdangerous` — session signing

**Frontend**
- `Next.js` with TypeScript and App Router
- `next-auth` v5 — Google OAuth provider
- `@prisma/client` — Prisma client for frontend API routes
- Two-panel layout: lever-controlled PDF panel + full-width briefing/chat
- Chat streams token-by-token via SSE
- Code blocks render with syntax highlighting and provenance badges
- Hyperparameter table renders as an interactive grid with color-coded status
- Ambiguity cards show agent resolution with user override input
- Glass UI design system throughout (spec in frontend section)

---

## Project Structure

```
/
├── backend/
│   ├── main.py
│   ├── routers/
│   │   ├── ingest.py
│   │   ├── conversation.py
│   │   └── auth.py                         # Google OAuth callback + JWT issue
│   ├── agents/
│   │   ├── ingestion_agent.py
│   │   ├── comprehension_agent.py
│   │   ├── briefing_agent.py
│   │   ├── qa_agent.py
│   │   ├── code_agent.py
│   │   └── graph.py
│   ├── tools/
│   │   ├── paper_tools.py
│   │   ├── knowledge_tools.py
│   │   ├── code_tools.py
│   │   └── analysis_tools.py
│   ├── services/
│   │   ├── pdf_parser.py
│   │   ├── arxiv_fetcher.py
│   │   └── vision_service.py
│   ├── models/
│   │   ├── paper.py
│   │   ├── briefing.py
│   │   └── artifacts.py
│   ├── db/
│   │   ├── prisma.py                       # Prisma client singleton
│   │   └── queries.py                      # All DB read/write functions
│   ├── background_knowledge/
│   │   └── landmark_papers.py
│   ├── memory/
│   │   └── session_memory.py
│   └── prompts/
│       ├── comprehension.py
│       ├── briefing_sections.py
│       ├── code_gen.py
│       ├── qa.py
│       └── figure.py
├── prisma/
│   └── schema.prisma                       # Prisma schema — all models defined here
├── frontend/
│   ├── app/
│   │   ├── page.tsx                        # Hero
│   │   ├── upload/page.tsx                 # Paper input
│   │   ├── session/[id]/page.tsx           # Main session
│   │   ├── api/
│   │   │   └── auth/[...nextauth]/route.ts # next-auth handler
│   │   └── layout.tsx
│   ├── components/
│   ├── hooks/
│   └── lib/
│       ├── auth.ts                         # next-auth config
│       ├── prisma.ts                       # Prisma client singleton (frontend)
│       └── api.ts
├── .env.example
└── requirements.txt
```

---

## Backend: What Each File Must Do

### `services/pdf_parser.py`
- Open the PDF using `pymupdf`.
- Walk every page. For each page extract: text blocks with font size metadata, image crops (for figures), and table-like structures.
- Use font size heuristics to detect section headings — larger bold text that is short in character count.
- Detect pseudocode blocks by looking for "Algorithm" keyword patterns or monospace font usage.
- Detect equations by looking for equation-label patterns like `(1)`, `(2)` at line ends, or heavy use of Greek characters.
- For each extracted figure/image, crop it as a PNG bytes object and store it alongside its caption text.
- Return a structured `ParsedPaper` object containing: title, authors, abstract, and a flat list of typed elements (Section, Equation, Figure, Table, Pseudocode), each tagged with its section heading and page number.

### `services/arxiv_fetcher.py`
- Accept an arXiv ID (e.g., `2310.06825`) or full arXiv URL.
- Use the `arxiv` library to fetch paper metadata: title, authors, abstract.
- Download the PDF bytes using `httpx` from the arXiv PDF URL.
- Return the PDF bytes and metadata — hand off to `pdf_parser.py`.

### `services/vision_service.py`
- Accept a figure image (PNG bytes) and its caption string.
- Send both to Gemini via LangChain — construct a `HumanMessage` with two content parts: an image part (base64-encoded PNG) and a text part containing the caption and the interpretation prompt.
- Use `ChatGoogleGenerativeAI` directly here (not a full agent chain) since this is a single-turn vision call.
- The prompt must instruct the model to: identify all labeled components, describe connections/arrows between them, identify any dimension annotations, and state what the figure demonstrates in implementation terms. Note anything in the figure that contradicts or adds to the text description.
- Return a plain-text description stored on the `PaperElement` as `figure_description`.

### `agents/ingestion_agent.py`
- Orchestrates the full ingestion pipeline using a **LangChain LCEL chain**: receives PDF bytes + metadata, calls `pdf_parser.py`, then calls `vision_service.py` for every Figure element.
- After parsing, runs a Gemini chain via LCEL — `ChatPromptTemplate | ChatGoogleGenerativeAI | PydanticOutputParser` — over the full extracted text to identify the primary task being solved and the list of foundational prerequisite concepts.
- Returns a complete `ParsedPaper` object with all figure descriptions populated.

### `agents/comprehension_agent.py`
- Runs **once** after ingestion, before any conversation starts. This is the most important agent.
- Implemented as a **LangChain LCEL chain**: `ChatPromptTemplate | ChatGoogleGenerativeAI | PydanticOutputParser(InternalRepresentation)`.
- Feed it the full paper text (all section elements concatenated) plus all figure descriptions — pass as a single large context string in the prompt. Do not chunk.
- It must produce an `InternalRepresentation` object containing:
  - `problem_statement` — one paragraph, plain English
  - `method_summary` — 3-5 sentences, the core idea
  - `novelty` — what this paper does differently from prior work, cited by section
  - `component_graph` — list of architectural components with dependency relationships (e.g., TransformerBlock depends on MultiHeadAttention which depends on ScaledDotProductAttention)
  - `hyperparameter_registry` — every hyperparameter found anywhere in the paper (including footnotes and appendices), with value if stated, source section, and status: `paper-stated` / `inferred` / `missing`
  - `ambiguity_log` — every point where the paper is underspecified in a way that affects implementation. Each entry must include: what is ambiguous, which section it is in, what the implementation consequence is if resolved incorrectly, and the agent's best-guess resolution with reasoning
  - `training_procedure` — synthesized from all mentions across the entire paper: optimizer, scheduler, loss function, batch size, number of steps, data augmentation, regularization, hardware
  - `prerequisite_concepts` — foundational concepts the paper assumes, each with a plain-English explanation appended by the agent
- The prompt (in `prompts/comprehension.py`) passes `parser.get_format_instructions()` and instructs the model to output structured JSON matching `InternalRepresentation` exactly. It must be exhaustive on hyperparameters and ambiguities.

### `agents/briefing_agent.py` and `agents/graph.py`

The briefing agent generates the six-section briefing document. It is a **LangGraph linear pipeline** — six nodes executed in sequence, each generating one section. This is not a conversation. It runs once per session after the comprehension pass and produces a static document.

**State Definition**
Define a `BriefingState` TypedDict containing: `session_id`, `internal_rep`, `parsed_paper`, `completed_sections` (dict keyed by section name), and `generation_progress` (integer 0-6 for progress indicator).

**The Six Briefing Sections — one LangGraph node each**

Each node receives the full `InternalRepresentation` and the relevant portion of `ParsedPaper` as context, generates its section content, and writes it to `completed_sections` in the state. Each section streams its output to the frontend as it generates — do not wait for all six sections to complete before showing anything.

**Section 1 — What This Paper Actually Does**
One paragraph. No jargon. No prior ML knowledge assumed. Explains the problem, what the paper proposes, and why it matters. Written as if explaining to an intelligent non-specialist. No equations. No architecture terminology. If the user reads nothing else, this paragraph tells them what the paper is about.

**Section 2 — The Mechanism**
How the core idea works. Walks through the method step by step. Every equation that appears is decoded inline — every symbol is defined at the point of use, not in a glossary. Every figure referenced is accompanied by the vision model's interpretation. Prerequisite concepts are explained at the point they are needed, not in a separate list. The explanation chains downward: not just what each component does but why it is necessary given what came before it.

**Section 3 — What You Need To Already Know**
A dependency-ordered list of prerequisite concepts the paper assumes. Each concept is explained in a 3-part structure: (1) the problem that existed before this concept, (2) what the concept does to solve it, (3) how this paper uses it specifically. Concepts are ordered from foundational to paper-specific — start with the lowest-level dependency and build up. This section is generated from `prerequisite_concepts` in `InternalRepresentation`, expanded by the agent.

**Section 4 — The Full Implementation Map**
Every architectural component in dependency order (leaf components first, composed components last). For each component: a plain-English description, the code snippet generated by `code_agent.py` with inline equation citations and provenance labels, and a list of implementation notes (things to watch out for, common mistakes, dimensions to track). Components that depend on each other are visually grouped.

**Section 5 — What The Paper Left Out**
The full ambiguity report rendered as the primary content of this section, not a footnote. For each ambiguity: what is missing, which section it comes from, what the implementation consequence is, and the agent's resolution with reasoning and confidence level. Organized by ambiguity type: `missing_hyperparameter`, `undefined_notation`, `underspecified_architecture`, `missing_training_detail`, `ambiguous_loss_function`. This section is front and center — it is one of the most valuable outputs of the tool.

**Section 6 — How To Train It**
The complete training procedure synthesized from all paper sections including footnotes, appendix, and result table captions. Rendered as: the full hyperparameter table (name, value, source section, status), the training loop description (optimizer, scheduler, loss, batch size, steps), data preprocessing requirements, and any reported tricks or implementation details that affect results. For every missing hyperparameter, state the agent's suggested default and the reasoning behind it.

**Streaming and DB writes**
Each section node calls `.astream_events()` and pipes tokens to the frontend via SSE as it generates. As each section completes its full content, call `save_briefing_section(paper_id, section_number, content)` to persist it. The frontend renders sections progressively — Section 1 appears first while Sections 2-6 are still generating. When all six sections are done, call `save_briefing_structured_data` to persist hyperparameters, ambiguities, and code snippets. Then call `update_paper_status(paper_id, "COMPLETE")`.

**Thinking Events — required for the ThinkingStream UI**
Every LangGraph node must emit `thinking` SSE events throughout its execution — not just at start and end, but at meaningful internal steps. These are short plain-English strings describing what the agent is actually doing at that exact moment. They must be real, not predefined. Examples of what nodes should emit:

- Ingestion node: "Extracting text from page 4...", "Found Algorithm 1 block on page 6...", "Interpreting Figure 3 — encoder-decoder diagram..."
- Comprehension node: "Identifying core architectural components...", "Found 3 undefined hyperparameters...", "Checking appendix for training details...", "Mapping component dependencies..."
- Section 1 node: "Distilling the core contribution into plain language..."
- Section 2 node: "Decoding equation (4) — attention weight computation...", "Tracing data flow through the encoder..."
- Section 4 node: "Generating PyTorch snippet for MultiHeadAttention...", "Labeling inferred assumptions in feed-forward block..."
- Section 5 node: "Evaluating ambiguity in dropout placement...", "Flagging missing learning rate schedule..."

Emit these via `astream_events` custom event type `"thinking"` with the string as the event data. The frontend `ThinkingStream.tsx` listens for this event type specifically and renders the stream. Section content tokens come through a separate `"section_token"` event type. The `generation_progress` field in state drives the progress indicator.

---

### `agents/qa_agent.py`

The Q&A agent handles all user questions after the briefing is generated. It is a **LangGraph single node** with tool access.

- Implemented using `create_tool_calling_agent` with all tools bound.
- Has access to the full `InternalRepresentation`, the full `ParsedPaper` text, and the generated briefing as context.
- Memory: `ConversationSummaryBufferMemory` — compresses old Q&A turns automatically when context grows long.
- The agent must use the briefing as the first source of truth for any question — if the answer is already in the briefing, point to it and expand. If not, use `paper_section_lookup` to retrieve the relevant section and answer from source.
- The agent must never say "the paper doesn't say" without first calling `paper_section_lookup` on the appendix and every footnote. Appendices contain critical details authors deprioritize.
- If a user asks for a code snippet not in the briefing, call `code_snippet_generator` and return the result with full provenance labels.
- If a user resolves an ambiguity differently than the agent did, update the `resolved_ambiguities` in session state and note the implementation consequence of the different choice.

### `agents/code_agent.py`
- Implemented as a **LangChain LCEL chain**: `ChatPromptTemplate | ChatGoogleGenerativeAI | PydanticOutputParser(CodeSnippet)`.
- Accepts a component name, its description, relevant paper sections, relevant equations, and any user-resolved ambiguities.
- Generates a PyTorch implementation snippet.
- Every line corresponding to a paper equation must have an inline comment: `# Eq. (N)`.
- Every assumption the agent made must be labeled: `# ASSUMED: <reason>`.
- Every inferred detail must be labeled: `# INFERRED: <reason>`.
- Never silently invent values for missing hyperparameters — always flag them.
- Prefer readable code over compact code. This is for learning, not production.
- Returns a `CodeSnippet` Pydantic object with: code string, provenance label, list of assumption notes, and source section references.

### `routers/ingest.py`
- `POST /ingest/upload` — accepts PDF file upload, runs ingestion + comprehension + briefing pipeline. Requires authenticated user (JWT middleware). Returns `paper_id`.
- `POST /ingest/arxiv` — accepts arXiv ID or URL, runs same pipeline. Requires authenticated user. Returns `paper_id`.
- On receipt, create a `Paper` record in the database immediately with status `processing`. Update status to `complete` when briefing finishes or `failed` on error.
- Before running the pipeline, check the user's `papers_analyzed_count` against their plan limit. If at limit, return 429 with the limit details.
- Increment `papers_analyzed_count` on the user record after a successful ingestion.

### `routers/auth.py`
- `GET /auth/google` — redirects user to Google OAuth consent screen.
- `GET /auth/google/callback` — handles OAuth callback, exchanges code for tokens, fetches Google profile (email, name, avatar).
- On callback: upsert a `User` record in the database using the Google `sub` as the unique identifier. If new user, set `plan = "free"` and `papers_analyzed_count = 0`.
- Issue a signed JWT containing `user_id`, `email`, and `plan`. Return it as an `httpOnly` cookie with `SameSite=Lax`.
- `POST /auth/logout` — clears the cookie.
- `GET /auth/me` — returns current user profile from JWT + database record (plan, usage count, limit).

---

## Database — Prisma Schema

Define the schema in `prisma/schema.prisma`. Use PostgreSQL as the provider. All data that was previously written to disk as JSON files now lives in the database.

```prisma
generator client {
  provider             = "prisma-client-py"
  interface            = "asyncio"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model User {
  id                   String   @id @default(cuid())
  google_sub           String   @unique
  email                String   @unique
  name                 String
  avatar_url           String?
  plan                 Plan     @default(FREE)
  papers_analyzed      Int      @default(0)
  created_at           DateTime @default(now())
  updated_at           DateTime @updatedAt
  papers               Paper[]
}

enum Plan {
  FREE
  PRO
}

model Paper {
  id                   String        @id @default(cuid())
  user_id              String
  user                 User          @relation(fields: [user_id], references: [id])
  title                String
  authors              String[]
  arxiv_id             String?
  status               PaperStatus   @default(PROCESSING)
  parsed_paper         Json          // serialized ParsedPaper object
  internal_rep         Json          // serialized InternalRepresentation object
  created_at           DateTime      @default(now())
  updated_at           DateTime      @updatedAt
  briefing             Briefing?
  qa_messages          QAMessage[]
}

enum PaperStatus {
  PROCESSING
  COMPLETE
  FAILED
}

model Briefing {
  id                   String   @id @default(cuid())
  paper_id             String   @unique
  paper                Paper    @relation(fields: [paper_id], references: [id])
  section_1            String?  // What This Paper Actually Does
  section_2            String?  // The Mechanism
  section_3            String?  // What You Need To Already Know
  section_4            String?  // The Full Implementation Map
  section_5            String?  // What The Paper Left Out
  section_6            String?  // How To Train It
  hyperparameters      Json?    // serialized HyperparameterEntry list
  ambiguities          Json?    // serialized AmbiguityEntry list
  code_snippets        Json?    // serialized CodeSnippet list
  created_at           DateTime @default(now())
  updated_at           DateTime @updatedAt
}

model QAMessage {
  id                   String   @id @default(cuid())
  paper_id             String
  paper                Paper    @relation(fields: [paper_id], references: [id])
  role                 String   // "user" or "assistant"
  content              String
  created_at           DateTime @default(now())
}
```

**Plan limits — enforce in `routers/ingest.py` and `routers/auth.py`:**

| Plan | papers_analyzed limit | Notes |
|---|---|---|
| FREE | 3 | Lifetime, not monthly |
| PRO | unlimited | Future paid tier, schema ready |

When a FREE user hits the limit, return a clear error message with the limit details. Do not silently fail. The frontend must display this as a visible upgrade prompt, not a generic error.

### `db/prisma.py`
- Single Prisma client instance, initialized once at startup.
- Use `@asynccontextmanager` for connection lifecycle.

### `db/queries.py`
All database operations live here as async functions. No raw Prisma calls scattered across routers. Functions needed at minimum:
- `get_or_create_user(google_sub, email, name, avatar_url) -> User`
- `get_user_by_id(user_id) -> User`
- `check_user_limit(user_id) -> bool` — returns True if user can analyze another paper
- `increment_paper_count(user_id)`
- `create_paper(user_id, title, authors, arxiv_id) -> Paper`
- `update_paper_status(paper_id, status)`
- `save_parsed_paper(paper_id, parsed_paper)`
- `save_internal_rep(paper_id, internal_rep)`
- `save_briefing_section(paper_id, section_number, content)`
- `save_briefing_structured_data(paper_id, hyperparameters, ambiguities, code_snippets)`
- `get_paper_with_briefing(paper_id, user_id) -> Paper + Briefing`
- `get_user_papers(user_id) -> list[Paper]`
- `append_qa_message(paper_id, role, content)`
- `get_qa_history(paper_id) -> list[QAMessage]`

---

### `memory/session_memory.py`
- Defines how LangChain memory is constructed per request from the database.
- Use `ConversationSummaryBufferMemory` with `gemini-2.5-flash-lite` as the summarizer and `max_token_limit=4000`.
- On every Q&A request, load the user's Q&A history from the database via `get_qa_history(paper_id)`, reconstruct the memory object from those messages, and pass it into the Q&A agent.
- After the agent responds, call `append_qa_message` to persist both the user message and agent response to the database.
- There are no files written to disk. All persistence goes through the database.

### `routers/conversation.py`
- All endpoints require authenticated user via JWT middleware. Validate that the requesting user owns the paper before returning any data.
- `POST /conversation/{paper_id}/message` — accepts user message, loads Q&A history from DB, feeds into `qa_agent` via `.astream_events()`, streams token-level output via SSE. Persists both messages to DB after response completes.
- `GET /conversation/{paper_id}/state` — returns current paper state from DB: briefing completion status per section, resolved ambiguities, hyperparameter table.
- `GET /conversation/{paper_id}/artifacts` — assembles and returns downloadable artifacts from DB briefing data.
- `POST /conversation/{paper_id}/resolve-ambiguity` — accepts ambiguity_id and user resolution, updates the `ambiguities` JSON in the Briefing record.

---

## Frontend: Full UI Specification

---

### Page 1 — Hero (`app/page.tsx`)

Full-screen dark landing page. This is the first thing the user sees.

**Content:**
- Project name **DeepRead** centered, large, clean typography.
- One sentence beneath it: the value proposition. Example: "Drop a research paper. Get everything you need to implement it."
- A single **Get Started** button below that. No other navigation, no feature list, no testimonials.
- Subtle background: very faint animated gradient or particle field — dark, not distracting. The page should feel like a tool, not a marketing site.
- On click, **Get Started** navigates to Page 2.

---

### Page 2 — Paper Input (`app/upload/page.tsx`)

Small, centered card on a dark background. Nothing else on the page.

**The card:**
- Compact. Not full screen. Roughly 480px wide, vertically centered.
- Two input options inside the card, visually separated by an "or" divider:
  - **Drag and drop zone** for PDF upload. On hover, border glows. On drop, shows file name.
  - **Text input** for arXiv link. Placeholder: `arxiv.org/abs/...` or just the ID.
- A single **Analyze** button below both inputs. Disabled until either a file is dropped or a link is entered.
- Glass UI style: the card has a frosted glass appearance — semi-transparent dark background, `backdrop-filter: blur(12px)`, a very subtle white border at 10-15% opacity, and a soft inner shadow. This is the visual language for all cards in the app.
- On submit, navigate to Page 3 (session page) and begin the pipeline.

---

### Page 3 — Main Session (`app/session/[id]/page.tsx`)

This is the primary interface. Three zones:

**Zone layout:**
- Left margin: **15%** — contains the PDF lever toggle, always present.
- Center: **70%** — the main briefing + chat interface.
- Right margin: **15%** — empty by default, reserved for the sticky section TOC and artifact download buttons.

When the PDF panel is open, it slides in from the left and takes **35% of total screen width**. The center panel shrinks from 70% to the remaining space. The right margin disappears on smaller screens. The transition is smooth: `transition: all 300ms ease`.

**Zone 1 — Left margin: `PdfLever.tsx`**
- Vertically centered in the left margin.
- Renders a diamond shape (rotated 45° square, white outline, no fill) with a small arrow icon beneath it.
- When PDF is closed: arrow points right, tooltip says "Open PDF".
- When PDF is open: arrow points left, tooltip says "Close PDF".
- Click toggles the PDF panel.
- No other elements in this margin.

**Zone 2 — Center: Main Interface**
The center panel is the entire product. Glass UI: `background: rgba(255,255,255,0.04)`, `backdrop-filter: blur(16px)`, `border: 1px solid rgba(255,255,255,0.08)`, `border-radius: 16px`, subtle box shadow outward. This panel has two states:

**State A — Thinking (while briefing generates):**
Replaces the briefing content area with the `ThinkingStream.tsx` component. Once sections start completing, they replace the thinking stream from the top down. Section 1 replaces the thinking area as soon as it's done; the thinking stream continues below it for the remaining sections.

**State B — Briefing loaded:**
Full briefing document rendered top to bottom. Chat input anchored to the bottom.

**Zone 3 — Right margin:**
- Sticky mini table-of-contents showing all six section names. Greyed out while pending, bold when complete. Click scrolls to that section.
- Below TOC: download buttons for artifacts, available as each section completes.
- Hidden on mobile.

---

### `PdfPanel.tsx`
- Slides in from the left with `transform: translateX`.
- Width: 35vw desktop, full-width overlay on mobile.
- Contains `react-pdf` viewer.
- **Zoom controls**: `+` and `−` buttons at the bottom of the panel. Current zoom percentage shown between them (e.g., `100%`). Also supports pinch-to-zoom on touch.
- **Scroll**: natural scroll within the panel. Page number indicator at the bottom (`Page 3 of 14`). Previous/next page buttons.
- Thin `×` close button at the top right of the panel as a secondary dismiss (primary dismiss is the lever).
- The panel has the same glass border style as the center panel.

---

### `ThinkingStream.tsx`
This replaces the progress bar entirely. It is a live feed of what the model is doing at the exact moment — not a predefined list of steps, but actual real-time status messages streamed from the backend.

**How it works:**
- The backend SSE stream emits two event types: `thinking` events (short status strings from the agent mid-process) and `section_token` events (actual briefing content tokens).
- `ThinkingStream` listens to `thinking` events and renders them as a flowing stream of single-line status updates, newest at the top, older ones fading out downward.
- Examples of what these strings look like (these come from the backend, not hardcoded in the frontend): "Reading abstract and identifying core claim...", "Extracting equations from Section 3...", "Interpreting Figure 2 — attention diagram...", "Found 4 undefined hyperparameters in Appendix B...", "Generating implementation for MultiHeadAttention...", "Checking footnotes for training details..."
- Each new message fades in at the top with a subtle animation. Previous messages shift down and reduce opacity gradually. After 4-5 messages, the oldest disappears.
- Beneath the streaming messages, a single pulsing dot (like a typing indicator) signals the model is still running.
- The backend must emit these `thinking` events from the LangGraph nodes — each node emits status strings at key steps using `astream_events` with custom event types. This is not cosmetic — the strings come from what the agent is actually doing.

**Transition to content:**
When Section 1 completes, its content fades in above the thinking stream. The thinking stream continues beneath it for the remaining sections. As each section completes, it slots in above the remaining thinking stream. When all six are done, the thinking stream disappears and only the briefing remains.

---

### `BriefingDocument.tsx`
The full briefing rendered as a clean document. Sections load in progressively — do not dump all content at once. Each section's text streams in token by token as it arrives from SSE, like a typewriter but smooth (use `opacity` and slight `translateY` entrance per paragraph block, not per character).

Sections are visually separated by a thin horizontal rule with the section number and name. The document has comfortable reading width (max ~680px), centered in the 70% panel with generous padding on both sides.

---

### `BriefingSection.tsx`
Renders one section. Handles mixed content types: prose paragraphs, `CodeBlock`, `HyperparamTable`, `AmbiguityCard` list, `PrerequisiteCard` list, figure callout boxes. Each content type enters with a subtle fade-in as it loads — no layout jumps.

---

### `CodeBlock.tsx`
- Syntax-highlighted with `highlight.js` or `prism`. Dark theme matching the overall UI.
- Provenance badge top-right corner: green = `paper-stated`, yellow = `inferred`, red = `assumed`.
- Collapsible ASSUMED/INFERRED notes panel below the code block.
- Copy button top-right. Equation and section citation badges below the block.
- The code block itself uses the glass card style: slightly lighter background than the surrounding panel, same border treatment.

---

### `HyperparamTable.tsx`
- Rendered inside Section 6.
- Columns: Name, Value, Source, Status.
- Status column color-coded: green / yellow / red.
- Missing values shown in italic muted text with agent default inline.
- CSV export button. Tooltips on missing rows explaining the agent's default reasoning.

---

### `AmbiguityCard.tsx`
- Rendered inside Section 5. Glass card style.
- Header row: ambiguity type badge (color by category) + short title of what's ambiguous.
- Body: implementation impact text, agent's resolution, confidence indicator.
- Footer: text input + confirm button for user override. Collapses on resolution.

---

### `PrerequisiteCard.tsx`
- Rendered inside Section 3. Glass card style.
- Three labeled rows: **Problem**, **Solution**, **Usage in this paper**.
- Collapsible. Concepts the user marks "I know this" collapse permanently for the session.

---

### `ChatInput.tsx`
- Fixed to the bottom of the center panel, always visible.
- Glass style: frosted input bar with subtle border.
- Placeholder: "Ask anything about this paper..."
- Send on Enter or button click.
- When a response is generating, the input is disabled and shows a pulsing indicator.
- Chat messages render in the same scroll container as the briefing, below all six sections. User messages right-aligned, agent messages left-aligned. Each message block fades in as it arrives — token by token for agent messages, immediate for user messages.
- Agent messages use the same glass card style as the rest of the UI.

---

### `ArtifactDownloads.tsx`
- Rendered in the right margin, below the TOC.
- Four download buttons, each labeled and icon-tagged: Briefing (`.md`), Code (`.py`), Hyperparams (`.csv`), Ambiguity Report (`.md`).
- Each button activates as soon as its relevant section completes. Inactive buttons are greyed out with a pending state.

---

### Visual Design System (apply consistently everywhere)

**Color palette:** Near-black background (`#0a0a0f`). White text at 90% opacity for primary, 50% for secondary. Accent color: a single muted blue or violet (`#6366f1` or similar) used only for active states, badges, and the send button.

**Glass UI recipe (apply to all cards and panels):**
```
background: rgba(255, 255, 255, 0.04);
backdrop-filter: blur(16px);
-webkit-backdrop-filter: blur(16px);
border: 1px solid rgba(255, 255, 255, 0.08);
border-radius: 16px;
box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255,255,255,0.06);
```

**Typography:** Single font — `Inter` or `Geist`. No mixed fonts. Code blocks use `JetBrains Mono` or `Fira Code`.

**Animations:** All transitions `300ms ease`. Entrance animations on content blocks: `opacity 0 → 1` + `translateY(8px) → 0`. No bounces, no spring physics. Clean and fast.

**No progress bars anywhere.** The thinking stream replaces all progress indicators.

---

## Prompts: What Each Prompt Must Accomplish

### `prompts/comprehension.py`
Instruct the model to:
- Read the entire paper and produce all fields of `InternalRepresentation` as structured JSON.
- Be exhaustive on hyperparameters — check every section, table, footnote, and appendix.
- Be exhaustive on ambiguities — flag anything requiring an implementation decision the paper does not cover.
- For each ambiguity, reason explicitly about what breaks if the wrong choice is made.
- For prerequisite concepts, assume the reader is a student and explain each from scratch in 2-4 sentences.

### `prompts/briefing_sections.py`
One prompt per briefing section (six total). Each must:
- Include the full `InternalRepresentation` as context.
- Include the specific paper elements relevant to that section (e.g., the method section text and figure descriptions for Section 2, the hyperparameter table for Section 6).
- Define the agent persona: precise, pedagogical, zero tolerance for undefined terms.
- Instruct the agent to decode every mathematical symbol at the point of first use — never leave a symbol undefined.
- Instruct the agent to explain prerequisite concepts inline, at the point they are needed, using the 3-part structure: problem → solution → paper-specific usage.
- Instruct the agent to cite paper sections and equation numbers inline in every claim.
- Section 5 prompt specifically: instruct the model to be **more aggressive about ambiguity detection** than in the comprehension pass — the briefing is the right place to surface every concern in full.

### `prompts/qa.py`
Instruct the Q&A agent to:
- Use the briefing as the first source of truth. If the answer is in the briefing, point to the section and expand.
- Call `paper_section_lookup` before saying the paper doesn't address something.
- Call `code_snippet_generator` if the user asks for a code example not in the briefing.
- Explain prerequisites when they come up in questions, using the same 3-part structure as the briefing.
- Infer user knowledge level from vocabulary in their questions and adjust explanation depth accordingly.
- Never contradict the briefing's provenance labels. If a detail was labeled `assumed` in the briefing, it remains assumed in the Q&A.

### `prompts/code_gen.py`
Instruct the model to:
- Generate PyTorch code only.
- Comment every equation reference inline: `# Eq. (N)`.
- Label every assumption: `# ASSUMED: <reason>`.
- Label every inference: `# INFERRED: <reason>`.
- Never invent values for missing hyperparameters silently.
- Prefer readable code over compact code.

### `prompts/figure.py`
Instruct the vision model to:
- Identify every labeled component in the figure.
- Describe data flow direction where arrows are present.
- Identify any dimension annotations.
- State what the figure demonstrates in implementation terms.
- Note anything in the figure that contradicts or adds to the text description.

---

## Data Flow

```
User visits hero → clicks Get Started → Google OAuth → JWT cookie issued
User record upserted in DB (google_sub as unique key)
        ↓
User submits PDF or arXiv ID (Page 2 upload card)
Limit check: query DB for user.papers_analyzed vs plan limit
If at limit → 429 → frontend shows upgrade prompt
        ↓
Paper record created in DB with status=PROCESSING
        ↓
arxiv_fetcher.py (if arXiv) → PDF bytes
        ↓
pdf_parser.py → ParsedPaper (typed elements: Section, Equation, Figure, Table, Pseudocode)
        ↓
vision_service.py → populates figure_description on each Figure element via gemini-2.5-flash vision
        ↓
ingestion_agent.py → complete ParsedPaper saved to Paper.parsed_paper in DB
        ↓
comprehension_agent.py (gemini-2.5-pro) → InternalRepresentation saved to Paper.internal_rep in DB
        ↓
Frontend navigates to /session/{paper_id}, SSE connection opens
        ↓
briefing_agent.py (LangGraph pipeline, gemini-2.5-pro for Sections 2/4/5 and gemini-2.5-flash for Sections 1/3/6) → runs six nodes in sequence
  — each node emits `thinking` SSE events throughout execution
  — each node emits `section_token` SSE events as content generates
  — each completed section written to Briefing record in DB immediately
        ↓
Frontend renders ThinkingStream → slots in sections as they arrive from DB writes
All six sections complete → Paper.status = COMPLETE → ThinkingStream disappears
User.papers_analyzed incremented
        ↓
qa_agent.py (gemini-2.5-flash) handles follow-up questions
  — loads QAMessage history from DB on each request
  — calls tools as needed
  — persists user message + agent response to QAMessage table after each turn
        ↓
GET /artifacts → assembles downloadable files from Briefing record in DB
```

---

## Environment Variables (`.env.example`)

```
# Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/deepread

# Google OAuth
GOOGLE_CLIENT_ID=your_google_oauth_client_id
GOOGLE_CLIENT_SECRET=your_google_oauth_client_secret
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback

# JWT
JWT_SECRET=your_long_random_secret_string
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=10080  # 7 days

# next-auth (frontend)
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=your_nextauth_secret

# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=deepread

# App
MAX_PAPER_SIZE_MB=20
```

Enable LangSmith tracing in development. It gives full visibility into every LangGraph node execution, every tool call, and every LLM call. Do not require it for production.

---

## Non-Negotiable Implementation Rules

1. **Never hallucinate paper content.** If the paper does not state something, label it `assumed` or `missing`. This is the tool's core integrity guarantee.

2. **The comprehension pass runs once and is cached in the database.** After `InternalRepresentation` is generated, it is written to the `Paper` record. On all subsequent requests — including Q&A turns and page reloads — load it from the database. Never regenerate it.

3. **Feed the full paper text to gemini-2.5-pro.** Do not chunk or summarize the paper before the comprehension pass. gemini-2.5-pro's large context window handles standard ML papers in full. Summarization loses footnotes, appendix hyperparameters, and caption details.

4. **The briefing generates first, Q&A comes after.** The chat input is visible during generation but disabled until at least Section 1 is written to the database. Do not let the user ask questions before any content exists to answer from.

5. **The ambiguity log is the primary differentiator.** Every paper has underspecified points. Surface all of them. A tool that hides ambiguities to produce clean output creates false confidence in a potentially wrong implementation.

6. **Prerequisite expansion is mandatory.** When any concept from `prerequisite_concepts` is first encountered, the `prerequisite_expander` tool must be called and its output included in the response before continuing.

7. **Enforce user limits at the router level.** Check `check_user_limit` at the start of every ingest endpoint before doing any work. Return a structured 429 response the frontend can render as an upgrade prompt. Do not check limits anywhere else — one place, one check.

8. **All authentication is JWT via httpOnly cookie.** Every backend endpoint that touches user data requires the JWT cookie. Validate it in a FastAPI dependency and inject the `user_id` into route handlers. Never trust user-supplied IDs in request bodies.

9. **All LLM calls go through LangChain.** No direct google-generativeai SDK calls anywhere. All calls go through langchain-google-genai. Use the correct model for each workload as specified in the model assignment table — gemini-2.5-pro for the three high-reasoning tasks, gemini-2.5-flash for speed-sensitive tasks, gemini-2.5-flash-lite for background tasks.

---

## Project Philosophy — Translate These Into Agent Behavior

This section defines what DeepRead must be at its core. Every architectural decision, every prompt, every UI choice must be traceable back to one of these principles. Do not treat these as soft guidelines — implement each one as a concrete system behavior.

---

### 1. The Agent Exists to Fill the Implementation Gap

ML papers are written for reviewers, not implementers. They omit initialization schemes, skip optimizer details, assume fluency with prerequisite concepts, and bury critical details in footnotes. The agent's entire purpose is to fill that gap.

**In code this means:**
- The comprehension agent must actively hunt for what is missing, not just extract what is present. The `ambiguity_log` and `hyperparameter_registry` are not optional outputs — they are the primary outputs.
- When the agent generates a code snippet, it must ask: "What does an engineer need to know to write this correctly that the paper did not say?" and surface those answers as `ASSUMED` or `INFERRED` labels.
- The agent should never produce output that looks more complete than the paper actually is. False completeness is worse than acknowledged incompleteness.

---

### 2. Honesty Over Polish — Enforce Labeling Discipline Everywhere

The tool's credibility is entirely in its labeling. A clean-looking output that hides assumptions actively misleads the user. An output that clearly marks uncertainty empowers them.

**In code this means:**
- The three provenance labels — `paper-stated`, `inferred`, `assumed` — must be present on every code snippet, every hyperparameter entry, and every architectural claim the agent makes.
- `paper-stated` = the paper explicitly states this value or behavior, with a section/equation reference attached.
- `inferred` = the paper implies this through context, convention, or related work — the agent must state the reasoning.
- `assumed` = the paper is silent on this, the agent made a choice — the agent must state what it chose and why, and flag that a different choice would produce different results.
- The frontend must make these labels visually prominent, not small footnotes. Color coding is mandatory (green / yellow / red).
- The agent must never aggregate multiple assumed details into one unlabeled paragraph. Each assumed detail is a separate labeled item.

---

### 3. Prerequisite Expansion Must Chain Downward

When the paper uses a concept, the agent must not just define it — it must explain the chain of reasoning that makes it necessary. The difference between a definition and an explanation is the chain: what problem does this solve, what property does it provide, why does this paper use it specifically.

**In code this means:**
- The `prerequisite_concepts` list in `InternalRepresentation` must include not just top-level concepts (e.g., "multi-head attention") but their sub-dependencies (e.g., "scaled dot-product attention", "query-key-value decomposition").
- When the briefing agent generates Section 3, and when the Q&A agent encounters a prerequisite concept in a follow-up question, both must use the 3-part explanation structure: (1) what the problem was before this existed, (2) what this concept does to solve it, (3) why this paper specifically uses it in this context.
- The depth of explanation in Section 3 is always student-level — full from-first-principles. In Q&A, the agent should infer user knowledge level from vocabulary in their questions and adjust depth accordingly — technical vocabulary signals the user can skip the basics.

---

### 4. Foundational Paper Knowledge Must Be Built In

Great papers cite prior work and assume the reader has read it. DeepRead cannot require the user to go read every cited paper. The agent must have background knowledge on landmark ML papers and activate it inline.

**In code this means:**
- Build a `background_knowledge` module that contains plain-English implementation summaries of landmark papers: Attention Is All You Need, ResNet, Adam, Batch Norm, Dropout, BERT, GPT-2, ViT, DDPM, and others as relevant.
- When the comprehension agent or conversation agent detects a citation or reference to one of these papers, it must pull the relevant background knowledge and include it as context.
- This does not mean reproducing the cited paper — it means explaining the specific concept being borrowed, enough that the user does not need to context-switch to another paper.
- If a cited paper is not in `background_knowledge`, the agent should say so explicitly and describe what it can infer from context.

---

### 5. The Ambiguity Report Is a First-Class Artifact

The ambiguity log is not a side output — it is one of the most valuable things DeepRead produces. Across many papers, these logs constitute a dataset of where ML papers systematically fail to provide enough information to reproduce their results.

**In code this means:**
- Every ambiguity entry must have a structured schema: `ambiguity_id`, `description`, `section`, `implementation_impact`, `agent_resolution`, `user_resolution`, `resolution_confidence`.
- `implementation_impact` must be specific — not "results may vary" but "the choice of activation function here (ReLU vs GELU) affects gradient flow through sparse activations and will produce measurably different loss curves."
- The final ambiguity report artifact (downloadable `.md`) must be formatted so it is readable as a standalone document, not just an internal log dump.
- Design the schema so ambiguity reports from multiple papers can eventually be aggregated. Use consistent field names and categorize ambiguity types: `missing_hyperparameter`, `undefined_notation`, `underspecified_architecture`, `missing_training_detail`, `ambiguous_loss_function`.

---

### 6. Session Persistence Is Required

A user implementing a paper returns to it over days or weeks. Losing conversation state between sessions destroys the tool's utility for real implementation work.

**In code this means:**
- All persistence goes through Prisma and PostgreSQL. No files written to disk anywhere. `ParsedPaper`, `InternalRepresentation`, all six briefing sections, resolved ambiguities, and Q&A message history are all database records.
- On page reload, the frontend fetches the paper and briefing from the database via `GET /conversation/{paper_id}/state`. The briefing renders immediately from stored content — no regeneration, no loading state.
- The Q&A agent reconstructs its memory from `QAMessage` records on each request.
- Sessions are replaced by authenticated paper records. The user's paper history is available at any time via `GET /papers` — shows title, arXiv ID if applicable, date, and completion status.

---

### 7. Paper Comparison Mode

Users frequently need to understand how a new paper relates to a prior method — what changed, what improved, and what the implementation delta is.

**In code this means:**
- Support a comparison mode where two sessions can be loaded side by side.
- The comparison agent takes the two `InternalRepresentation` objects and produces a diff: architectural changes, changed hyperparameters, new components, removed components, and claimed improvement vs implementation change.
- This is a stretch goal — implement it after core single-paper flow is complete and stable. But design the `InternalRepresentation` schema from the start to support diffing. Use consistent component naming conventions so two representations of related architectures can be compared programmatically.

---

## Definition of Done

- Hero page loads → Get Started → Google OAuth → user created in DB → redirected to upload page.
- FREE users blocked at 3 papers with a visible upgrade prompt. PRO path is schema-ready but not implemented.
- User submits arXiv link or PDF → paper record created in DB → navigates to session page → ThinkingStream begins immediately showing real-time agent status strings.
- Briefing sections appear progressively as they stream and write to DB — Section 1 is readable within seconds.
- All six sections render completely: plain-English summary, mechanism with decoded equations, prerequisites explained inline, full implementation map with labeled code, ambiguity report with user-resolvable cards, complete training recipe with hyperparameter table.
- Every code snippet has inline equation citations and ASSUMED/INFERRED labels with reasons.
- Hyperparameter table color-codes every entry: green/yellow/red.
- Ambiguity cards are interactive — user can override agent resolutions, stored in DB.
- Chat input accepts free-form questions after Section 1 completes. Q&A history persists in DB across page reloads.
- Page reload restores full briefing instantly from DB — no regeneration.
- All artifacts downloadable: briefing `.md`, code `.py`, hyperparameter table `.csv`, ambiguity report `.md`.
- User's paper history accessible from the hero page — list of analyzed papers with title and date.
