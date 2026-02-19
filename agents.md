# AGENTS.md — ML Paper Comprehension Agent
### Instructions for Codex-5.3

---

## What You Are Building

A conversational web application where a user submits an ML research paper (PDF upload or arXiv link) and a Gemini-powered agent helps them fully understand and implement it. The agent does **not** run any code. It reads the paper, builds a structured internal understanding, then walks the user through architecture, implementation, hyperparameters, and ambiguities in a staged conversational workflow.

**The core requirement:** The agent must fill the gap between what the paper states and what a practitioner or student needs to know to implement it. When the paper says "standard layer normalization," the agent explains what layer norm is, why it's used here, and shows the implementation. When notation is undefined, the agent decodes it. When a hyperparameter is missing, the agent proposes a justified default. Everything is labeled: **paper-stated**, **inferred**, or **assumed/missing**.

---

## LLM and Agent Framework

Use **Google Gemini** (free tier) as the LLM, accessed exclusively through **LangChain** — not through the raw `google-generativeai` SDK. LangChain is the agent framework for the entire backend. All LLM calls, memory, tool use, and agent orchestration go through LangChain primitives.

**LLM Setup**
- Use `langchain-google-genai` to instantiate the LLM: `ChatGoogleGenerativeAI(model="gemini-flash-latest")`.
- `gemini-flash-latest` is the default model for all agents — 1M token context window is critical for feeding full paper text without chunking.
- For vision tasks (figure interpretation), use the same model with image content passed as a LangChain `HumanMessage` with inline base64 image data — `gemini-flash-latest` supports multimodal inputs natively through LangChain.
- Do not use any other LLM provider.

**Agent Framework — LangChain Primitives to Use**
- Use **LangGraph** for the multi-stage conversational workflow. Each conversation stage (Orientation, Architecture, Implementation, Ambiguity Resolution, Training Recipe) is a **LangGraph node**. Stage transitions are graph edges with conditional routing based on user intent detection.
- Use **LangChain LCEL** (LangChain Expression Language) for all chain construction — `prompt | llm | output_parser` pipelines throughout. Do not write manual LLM call loops.
- Use **LangChain Tools** (`@tool` decorator) for all discrete agent capabilities: `paper_section_lookup`, `equation_decoder`, `prerequisite_expander`, `code_snippet_generator`, `hyperparameter_extractor`, `ambiguity_detector`, `figure_interpreter`. The conversation agent invokes these via `create_tool_calling_agent`.
- Use **LangChain Memory** — `ConversationSummaryBufferMemory` backed by Gemini for managing message history within a session. Do not manage message history manually anywhere.
- Use **LangChain's `PydanticOutputParser`** for all structured outputs (InternalRepresentation, CodeSnippet, AmbiguityEntry, HyperparameterEntry). Pass the format instructions into every relevant prompt via `parser.get_format_instructions()`.
- Use **LangChain's `ChatPromptTemplate`** with `MessagesPlaceholder` for all prompts that include conversation history.
- Use **LangChain's document loaders and text splitters only for the background knowledge module** — the main paper text goes to Gemini in full, no splitting.

---

## Tech Stack

**Backend**
- `FastAPI` — API server with async support
- `pymupdf` (import as `fitz`) — PDF parsing, text extraction, figure/image cropping
- `arxiv` Python library — fetch paper metadata and PDF URL by arXiv ID
- `httpx` — async download of arXiv PDFs
- `langchain` — core LangChain framework
- `langchain-google-genai` — LangChain integration for Gemini
- `langchain-community` — community tools and utilities
- `langgraph` — multi-stage agent workflow as a state graph
- `sse-starlette` — server-sent events for streaming agent responses to frontend
- `pydantic v2` — all data models, used with LangChain's PydanticOutputParser
- `python-multipart` — PDF file upload handling
- `Pillow` — image processing for extracted figures before sending to vision model

**Frontend**
- `Next.js` with TypeScript
- Two-panel layout: PDF viewer on the left, chat interface on the right
- Chat streams token-by-token via SSE
- Code blocks render with syntax highlighting and a provenance badge (`paper-stated` / `inferred` / `assumed`)
- Hyperparameter table renders as an interactive grid with color-coded status
- Ambiguity cards show agent question + input for user resolution

---

## Project Structure

```
/
├── backend/
│   ├── main.py
│   ├── routers/
│   │   ├── ingest.py
│   │   └── conversation.py
│   ├── agents/
│   │   ├── ingestion_agent.py
│   │   ├── comprehension_agent.py
│   │   ├── conversation_agent.py       # LangGraph state machine lives here
│   │   ├── code_agent.py
│   │   └── graph.py                    # LangGraph graph definition and compilation
│   ├── tools/
│   │   ├── paper_tools.py              # @tool: section_lookup, equation_decoder
│   │   ├── knowledge_tools.py          # @tool: prerequisite_expander, background_knowledge_lookup
│   │   ├── code_tools.py               # @tool: code_snippet_generator
│   │   └── analysis_tools.py           # @tool: hyperparameter_extractor, ambiguity_detector
│   ├── services/
│   │   ├── pdf_parser.py
│   │   ├── arxiv_fetcher.py
│   │   └── vision_service.py
│   ├── models/
│   │   ├── paper.py
│   │   ├── conversation.py
│   │   └── artifacts.py
│   ├── memory/
│   │   └── session_memory.py           # LangChain memory setup and session persistence
│   └── prompts/
│       ├── comprehension.py
│       ├── stages.py
│       ├── code_gen.py
│       └── figure.py
├── frontend/
│   ├── app/
│   ├── components/
│   ├── hooks/
│   └── lib/
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

### `agents/conversation_agent.py` and `agents/graph.py`

The conversational workflow is a **LangGraph state machine**. Build it as follows:

**State Definition**
Define a `PaperLensState` TypedDict for the LangGraph state containing: `session_id`, `messages` (LangChain message list), `current_stage`, `internal_rep`, `resolved_ambiguities`, `current_component_index`, `user_level` (inferred from conversation), and `pending_question`.

**Graph Nodes — one per stage**
Each stage is a node function that receives the state, calls the appropriate LangChain chain, and returns a state update:
- `orientation_node` — introduces the paper, plain English, no code
- `architecture_node` — walks `component_graph` top-down, decodes equations, describes figures
- `implementation_node` — calls `code_snippet_generator` tool per component, attaches provenance
- `ambiguity_node` — surfaces `ambiguity_log` items one at a time, awaits user resolution
- `training_node` — synthesizes full training procedure and hyperparameter table
- `freeqa_node` — handles off-stage questions using full paper context

**Routing**
Add a `router_node` that runs before every user message. It uses a lightweight Gemini chain to classify user intent into one of: `continue_current_stage`, `jump_to_stage(N)`, `ask_about_component`, `free_question`. This classification drives which node handles the message next. This is the LangGraph conditional edge.

**Tools**
Each node invokes tools via `create_tool_calling_agent`. Bind the following tools to the conversation agent:
- `paper_section_lookup(section_name)` — retrieves the relevant section text from `ParsedPaper`
- `equation_decoder(equation_id)` — returns the equation with every symbol decoded
- `prerequisite_expander(concept_name)` — returns a 3-part explanation of the concept
- `background_knowledge_lookup(paper_name)` — returns the built-in summary of a landmark paper
- `code_snippet_generator(component_name)` — calls `code_agent.py`, returns `CodeSnippet`
- `hyperparameter_extractor()` — returns the full hyperparameter registry formatted as a table
- `ambiguity_detector(section_text)` — runs the ambiguity detection chain on a given section

All tools are defined in the `tools/` directory using the `@tool` decorator.

**Memory**
Attach `ConversationSummaryBufferMemory` to the graph — backed by `ChatGoogleGenerativeAI` for summarization. Pass memory into each node via the state's `messages` field. LangGraph manages message history across turns; the memory layer compresses old turns when context gets long.

**Streaming**
Use LangGraph's `.astream_events()` for token-level streaming. Pipe events to the SSE endpoint in `routers/conversation.py`.

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
- `POST /ingest/upload` — accepts PDF file upload, runs ingestion + comprehension pipeline, returns `session_id`.
- `POST /ingest/arxiv` — accepts arXiv ID or URL, runs same pipeline, returns `session_id`.
- Both endpoints store `ParsedPaper` and `InternalRepresentation` in a memory dict keyed by `session_id`. No database required.
- Return incremental status so the frontend can show a progress indicator during the comprehension pass.

### `memory/session_memory.py`
- Defines how LangChain memory is constructed and persisted per session.
- Use `ConversationSummaryBufferMemory` with `ChatGoogleGenerativeAI` as the summarizer LLM and `max_token_limit=4000`.
- On session creation, initialize a new memory instance and attach it to the session's LangGraph graph instance.
- On session save, serialize the memory's `chat_memory.messages` list to JSON and write to `sessions/{session_id}/memory.json`.
- On session reload, deserialize the messages and reconstruct the memory object. Do not re-run the comprehension pass.
- The `InternalRepresentation` is stored separately in `sessions/{session_id}/internal_rep.json` and loaded independently of memory.

### `routers/conversation.py`
- `POST /conversation/{session_id}/message` — accepts user message, feeds it into the LangGraph graph via `.astream_events()`, streams token-level output to the client via SSE.
- `GET /conversation/{session_id}/state` — returns current conversation state: stage, resolved ambiguities, hyperparameter table.
- `GET /conversation/{session_id}/artifacts` — returns all generated artifacts: architecture summary, annotated code file, hyperparameter table CSV, ambiguity report.
- `POST /conversation/{session_id}/resolve-ambiguity` — accepts ambiguity_id and user resolution, updates state, triggers the ambiguity node to move to the next item.

---

## Frontend: What Each Component Must Do

### Two-Panel Layout (`app/session/[id]/page.tsx`)
- Left panel: PDF viewer using `react-pdf`. Supports page navigation.
- Right panel: chat interface. Full remaining width.
- On mobile, stack vertically with PDF viewer collapsible.

### `ChatPanel.tsx`
- Renders message history with clear visual distinction between user and agent messages.
- Streams agent responses token by token as SSE arrives — do not wait for full response before rendering.
- When the agent asks a clarifying question, render it in a visually distinct card above the text input.
- Stage indicator at the top showing which of the 5 stages is active, clickable to jump between stages.

### `CodeBlock.tsx`
- Syntax-highlighted code using `highlight.js` or `prism`.
- Provenance badge in top-right: green = `paper-stated`, yellow = `inferred`, red = `assumed`.
- Expandable inline notes panel showing all ASSUMED and INFERRED comments as a list.
- Copy button.

### `HyperparamTable.tsx`
- Table with columns: Name, Value, Source Section, Status.
- Color-coded Status column: green = paper-stated, yellow = inferred, red = missing.
- For missing rows, show the agent's suggested default in the Value column in muted/italic style.

### `AmbiguityCard.tsx`
- Shows: what is ambiguous, implementation impact, agent's resolution.
- If unresolved: text input + confirm button for user override.
- Resolved cards are visually collapsed with a checkmark.

### `ComponentGraph.tsx`
- Directed graph of the `component_graph` using `react-flow`.
- Nodes are clickable — clicking a node jumps the conversation to that component's explanation in Stage 2 or 3.
- Show dependency arrows.

### `ArtifactPanel.tsx`
- Collapsible drawer.
- Downloadable artifacts: architecture summary `.md`, annotated code `.py`, hyperparameter table `.csv`, ambiguity report `.md`.
- Generated on demand via `GET /conversation/{session_id}/artifacts`.

---

## Prompts: What Each Prompt Must Accomplish

### `prompts/comprehension.py`
Instruct the model to:
- Read the entire paper and produce all fields of `InternalRepresentation` as structured JSON.
- Be exhaustive on hyperparameters — check every section, table, footnote, and appendix.
- Be exhaustive on ambiguities — flag anything requiring an implementation decision not covered by the paper.
- For each ambiguity, reason explicitly about what breaks if the wrong choice is made.
- For prerequisite concepts, assume the reader is a student and explain each from scratch in 2-4 sentences.

### `prompts/stages.py`
One prompt per stage. Each must:
- Include the full `InternalRepresentation` as context.
- Define the agent persona: patient, precise, pedagogical.
- Instruct the agent to always expand acronyms on first use.
- Instruct the agent to decode every mathematical symbol before using it.
- Instruct the agent to cite paper sections and equation numbers inline.
- Instruct the agent to ask one clarifying question at a time when ambiguity arises mid-explanation.
- Instruct the agent to explain foundational prerequisites before using them, scaled to student level.

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
User submits PDF or arXiv ID
        ↓
arxiv_fetcher.py (if arXiv) → PDF bytes
        ↓
pdf_parser.py → ParsedPaper (typed elements)
        ↓
vision_service.py → populates figure_description on each Figure element
        ↓
ingestion_agent.py → complete ParsedPaper stored by session_id
        ↓
comprehension_agent.py → InternalRepresentation stored by session_id
        ↓
Frontend receives session_id, loads session page
        ↓
conversation_agent.py handles all subsequent user messages
        ↓
code_agent.py called per component during Stage 3
        ↓
artifact_builder.py assembles final downloadable outputs on request
```

---

## Environment Variables (`.env.example`)

```
GEMINI_API_KEY=your_gemini_api_key_here
MAX_PAPER_SIZE_MB=20
SESSION_DIR=sessions

# Optional: LangSmith tracing for debugging agent runs
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=paperlens
```

Enable LangSmith tracing in development. It gives full visibility into every LangGraph node execution, every tool call, and every LLM call — essential for debugging the multi-stage agent flow. Do not require it for production.

---

## Non-Negotiable Implementation Rules

1. **Never hallucinate paper content.** If the paper does not state something, label it `assumed` or `missing`. This is the tool's core integrity guarantee.

2. **The comprehension pass runs once and is cached.** Store the `InternalRepresentation` to disk after the first run. On all subsequent conversation turns and session reloads, load it from disk. Never re-run it.

3. **Feed the full paper text to Gemini.** Do not chunk or summarize the paper before the comprehension pass. `gemini-flash-latest`'s 1M token window exists for this exact use case. Summarization loses footnotes, appendix hyperparameters, and caption details.

4. **Stage flow is a default, not a gate.** The LangGraph router must detect when a user wants to jump stages and route accordingly. The stage system guides; it does not block.

5. **The ambiguity log is the primary differentiator.** Every paper has underspecified points. Surface all of them. A tool that hides ambiguities to produce clean output creates false confidence in a potentially wrong implementation.

6. **Prerequisite expansion is mandatory.** When any concept from `prerequisite_concepts` is first encountered, the `prerequisite_expander` tool must be called and its output included in the response before continuing.

7. **Gemini free tier rate limits.** The comprehension pass is one large call. Conversation turns are smaller. Do not make redundant LLM calls — check if a tool result is already in session state before calling again. Batch figure interpretation where possible.

8. **All LLM calls go through LangChain.** No direct `google-generativeai` SDK calls anywhere in the codebase. This ensures all calls are observable via LangSmith and the memory layer stays consistent.

---

## Project Philosophy — Translate These Into Agent Behavior

This section defines what PaperLens must be at its core. Every architectural decision, every prompt, every UI choice must be traceable back to one of these principles. Do not treat these as soft guidelines — implement each one as a concrete system behavior.

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
- When the conversation agent encounters a prerequisite concept, it must generate a 3-part explanation: (1) what the problem was before this existed, (2) what this concept does to solve it, (3) why this paper specifically uses it in this context.
- The depth of explanation scales to user level. For a student, explain from first principles. For a practitioner, start from the known concept and explain only the paper-specific usage. The agent should infer user level from how they phrase questions — technical vocabulary signals higher level.

---

### 4. Foundational Paper Knowledge Must Be Built In

Great papers cite prior work and assume the reader has read it. PaperLens cannot require the user to go read every cited paper. The agent must have background knowledge on landmark ML papers and activate it inline.

**In code this means:**
- Build a `background_knowledge` module that contains plain-English implementation summaries of landmark papers: Attention Is All You Need, ResNet, Adam, Batch Norm, Dropout, BERT, GPT-2, ViT, DDPM, and others as relevant.
- When the comprehension agent or conversation agent detects a citation or reference to one of these papers, it must pull the relevant background knowledge and include it as context.
- This does not mean reproducing the cited paper — it means explaining the specific concept being borrowed, enough that the user does not need to context-switch to another paper.
- If a cited paper is not in `background_knowledge`, the agent should say so explicitly and describe what it can infer from context.

---

### 5. The Ambiguity Report Is a First-Class Artifact

The ambiguity log is not a side output — it is one of the most valuable things PaperLens produces. Across many papers, these logs constitute a dataset of where ML papers systematically fail to provide enough information to reproduce their results.

**In code this means:**
- Every ambiguity entry must have a structured schema: `ambiguity_id`, `description`, `section`, `implementation_impact`, `agent_resolution`, `user_resolution`, `resolution_confidence`.
- `implementation_impact` must be specific — not "results may vary" but "the choice of activation function here (ReLU vs GELU) affects gradient flow through sparse activations and will produce measurably different loss curves."
- The final ambiguity report artifact (downloadable `.md`) must be formatted so it is readable as a standalone document, not just an internal log dump.
- Design the schema so ambiguity reports from multiple papers can eventually be aggregated. Use consistent field names and categorize ambiguity types: `missing_hyperparameter`, `undefined_notation`, `underspecified_architecture`, `missing_training_detail`, `ambiguous_loss_function`.

---

### 6. Session Persistence Is Required

A user implementing a paper returns to it over days or weeks. Losing conversation state between sessions destroys the tool's utility for real implementation work.

**In code this means:**
- Store sessions on disk, not just in memory. Use a simple JSON file store keyed by `session_id` as the first implementation — no database required initially.
- Persist: `ParsedPaper`, `InternalRepresentation`, full `ConversationState` including message history, all resolved ambiguities, and all generated code snippets.
- On session reload, the agent must resume from where the conversation left off — restore stage, restore resolved ambiguities, do not re-run the comprehension pass.
- Sessions should be loadable from a session list on the landing page. Show paper title, arXiv ID if applicable, date, and current stage.

---

### 7. Paper Comparison Mode

Users frequently need to understand how a new paper relates to a prior method — what changed, what improved, and what the implementation delta is.

**In code this means:**
- Support a comparison mode where two sessions can be loaded side by side.
- The comparison agent takes the two `InternalRepresentation` objects and produces a diff: architectural changes, changed hyperparameters, new components, removed components, and claimed improvement vs implementation change.
- This is a stretch goal — implement it after core single-paper flow is complete and stable. But design the `InternalRepresentation` schema from the start to support diffing. Use consistent component naming conventions so two representations of related architectures can be compared programmatically.

---

## Definition of Done

- User pastes arXiv link → loading indicator → session opens with PDF on left, agent on right.
- Agent delivers Orientation message without any user prompting.
- Agent walks through architecture, decodes every equation symbol, generates labeled code snippets with inline citations.
- Every code snippet has provenance badges and ASSUMED/INFERRED labels.
- Hyperparameter table shows green/yellow/red status for every parameter found.
- Ambiguity cards surface every underspecified implementation decision and accept user input.
- Free-form Q&A works at any point in the conversation.
- Final artifacts are downloadable: code file, hyperparameter table, ambiguity report, architecture summary.
