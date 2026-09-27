# Curriculum → implementation map

Every topic from the learning path, what it means, what was built for it in
LegalLens, how it was built, and which technology runs it.

Legend for the **Proof** column: how you can verify the claim yourself.

---

## Shared Foundation

### Module 1 — AI & LLM Fundamentals

| Topic | What it means | What was built | How | Technology | Proof |
|---|---|---|---|---|---|
| AI / ML / Deep learning | Nested fields: AI ⊃ ML ⊃ DL | Two DL models run locally in the app (a bi-encoder and a cross-encoder) | Loaded through sentence-transformers on the Apple GPU | PyTorch (MPS), sentence-transformers | `core/embeddings.py`, `retrieval/rerank.py` |
| Generative AI | Models that produce new text rather than classify | The answer generator | Excerpts + instruction → generated prose with citations | gpt-5 / claude-opus-5 | `graph/nodes.py · generate` |
| How LLMs work | Next-token prediction over a context window | Why RAG exists here at all: the model has no contract knowledge, so we supply it per request | Retrieved clauses injected into the user turn | — | `prompts/templates.py` |
| Transformer architecture | Self-attention over a sequence | Exploited directly: the cross-encoder attends *across* query and clause in one pass | Joint `[CLS] query [SEP] clause` forward pass | ms-marco-MiniLM-L-6-v2 | `retrieval/rerank.py` |
| Tokens & tokenisation | Text → integer IDs; billing and limits are per token | Exact pre-flight token counting | Provider-correct: Anthropic's count endpoint; tiktoken for OpenAI | `messages.count_tokens`, tiktoken | `core/llm.py`, `core/llm_openai.py` |
| Context windows | Finite input budget per request | Retrieve 40 candidates, inject only 6 | Re-ranking narrows before the prompt is built | — | `config.py` (`retrieval_top_k`, `final_top_k`) |
| Prompt engineering basics | Instructions shape output | A prompt library, not inline strings | Parameterised `build_*` functions, versioned and testable | — | `prompts/templates.py` |
| Temperature / Top-P | Sampling randomness | Deliberately not used — reasoning models reject it | Depth controlled by `effort` instead; sampling only sent to models that accept it | `output_config.effort`, `reasoning_effort` | `core/llm.py`, `core/llm_openai.py` |
| Hallucinations | Confident, unsupported claims | Two independent defences | (1) regex citation validation, (2) claim-level groundedness audit with one regenerate | Structured output | `core/guardrails.py`, `graph/nodes.py · verify` |
| Fine-tuning vs prompting vs RAG | Three ways to add knowledge | RAG chosen, and the reasoning is documented | Corpus changes per user + answers must cite a source — neither works with a fine-tune | — | `docs/TECHNIQUES.md` |

### Module 2 — LLM APIs & Model Providers

| Topic | What it means | What was built | How | Technology | Proof |
|---|---|---|---|---|---|
| Anthropic Claude | Claude's Messages API | Full adapter | Adaptive thinking, `cache_control` breakpoints, effort control | `anthropic` SDK | `core/llm.py` |
| OpenAI APIs | Chat Completions | Second full adapter | Absorbs the differences: `max_completion_tokens`, `developer` role, automatic caching | `openai` SDK | `core/llm_openai.py` |
| Provider abstraction | Swap vendors without rewrites | `LLMClient` ABC with 4 methods | Adding a provider is a subclass; no call site changed when OpenAI was added | Python ABC | `core/llm.py` + 28 tests |
| Chat completion API | Request/response messaging | `complete()` | Uniform `LLMResult` across both vendors | both SDKs | `core/llm.py` |
| Structured outputs | Schema-constrained generation | Every routing/grading decision | Pydantic schema → constrained decoding | `messages.parse`, `chat.completions.parse` | `graph/schemas.py` |
| JSON mode | Guaranteed parseable JSON | Same mechanism | Router cannot return a shape the graph does not understand | Pydantic | `graph/schemas.py` |
| Function calling | Model invokes your functions | 5 read-only research tools | Defined once in Anthropic shape, translated for OpenAI | tool schemas, `strict: true` | `agents/tools.py` |
| Streaming responses | Token-by-token output | Live answer streaming | `asyncio.Queue` drained into SSE frames | SSE, sse-starlette | `core/service.py`, `api/routes_chat.py` |
| Build: chat application | — | The Research page | Multi-turn, session-persisted, streamed | Next.js + FastAPI | `app/page.tsx` |
| Build: summarisation tool | — | Two of them | The `summarize` intent, plus rolling conversation summaries | fast model | `graph/nodes.py`, `memory/chat_memory.py` |

### Module 3 — Prompt Engineering

| Topic | What it means | What was built | How | Technology | Proof |
|---|---|---|---|---|---|
| Zero-shot | Instruction, no examples | The answer instructions | Task is well specified, examples not needed | — | `prompts/templates.py` |
| One-shot | A single example | Query rewriting | One worked example teaches the output shape | — | `prompts/templates.py` |
| Few-shot | Several examples | The answer prompt | 4 examples: found / partial / **absent** / advice-seeking | — | `prompts/templates.py` |
| Chain of thought | Reason before answering | Answer + groundedness judge | "Work out which clause governs before committing" | — | `prompts/templates.py` |
| Role prompting | Assign a persona | "Research assistant, not a lawyer" | This is what makes the system decline advice | — | `prompts/templates.py` |
| Output formatting | Constrain the shape | The `[n]` citation contract | Machine-checkable, which is what makes validation possible | regex | `core/guardrails.py` |
| Prompt templates | Parameterised, reusable | `build_*` functions | System prompt kept **byte-stable** so caching works | — | `prompts/templates.py` |

### Module 4 — Embeddings & Vector Search

| Topic | What it means | What was built | How | Technology | Proof |
|---|---|---|---|---|---|
| What are embeddings | Text → vector where distance = meaning | 4,200 clause vectors | 768-dim, generated locally and free | BAAI/bge-base-en-v1.5 | `core/embeddings.py` |
| Dense vs sparse retrieval | Meaning vs literal terms | Both arms, running in parallel | Dense finds "exit rights", sparse finds "Siemens" and "$5,000,000" | BGE + rank_bm25 | `retrieval/retriever.py` |
| Similarity search | Nearest neighbours in vector space | Top-k over the corpus | HNSW index inside Chroma | chromadb | `retrieval/vectorstore.py` |
| Cosine similarity | Angle between vectors | The distance metric | Vectors L2-normalised at encode → cosine becomes a dot product | numpy | `core/embeddings.py` |
| Semantic search | Meaning-based retrieval | The dense arm | Asymmetric prefix: queries get an instruction prefix, documents do not | BGE | `core/embeddings.py` |
| Vector DB: Chroma | Embedded vector store | **Default backend** | No server; persists to disk | chromadb | `retrieval/vectorstore.py` |
| Vector DB: PGVector | Postgres extension | Alternate backend | Lazy import; compose profile provided | langchain-postgres, pgvector | `retrieval/vectorstore.py`, `docker-compose.yml` |
| Vector DB: Pinecone | Managed service | Alternate backend | Lazy import behind the same interface | langchain-pinecone | `retrieval/vectorstore.py` |
| Vector DB: Weaviate | Hybrid-native engine | Alternate backend | Lazy import behind the same interface | weaviate-client | `retrieval/vectorstore.py` |
| Build: semantic search engine | — | The whole retrieval pipeline | — | — | `retrieval/` |

### Module 5 — Retrieval Augmented Generation

| Topic | What it means | What was built | How | Technology | Proof |
|---|---|---|---|---|---|
| RAG architecture | Retrieve → augment → generate | The full pipeline, split offline/online | Ingest once; query many times | — | `ingestion/pipeline.py`, `graph/build.py` |
| Chunking strategies | How documents are split | **Three**, switchable and comparable | `recursive` / `legal_structural` (default) / `semantic` | langchain-text-splitters + custom | `ingestion/chunking.py` |
| Metadata | Structured fields on each chunk | Two tiers | Document-level (parties, type, law) + chunk-level (41 clause tags) | — | `ingestion/metadata.py`, `taxonomy.py` |
| Retrieval pipeline | Query → ranked chunks | Hybrid + fusion + re-rank | Measured, not assumed — see the ablation | — | `retrieval/retriever.py` |
| Context injection | Putting chunks in the prompt | Excerpts first, instruction last | Ordered so the cache prefix stays stable | — | `prompts/templates.py` |
| Re-ranking | Reorder with a better model | Cross-encoder pass over top 20 | Retrieve wide, re-rank narrow | ms-marco-MiniLM-L-6-v2 | `retrieval/rerank.py` |
| Build: PDF chatbot | — | The application | PDF upload → indexed → cited answers | pdfplumber | `api/routes_documents.py` |
| Build: knowledge base assistant | — | The same application at corpus scale | 510 contracts | — | `scripts/ingest_cuad.py` |

### Module 6 — LangChain

| Topic | What it means | What was built | How | Technology | Proof |
|---|---|---|---|---|---|
| Chains | Composed steps | The ingestion chain | load → chunk → enrich → embed → index | — | `ingestion/pipeline.py` |
| Prompt templates | Reusable prompts | The prompt library | — | — | `prompts/templates.py` |
| Output parsers | Text → typed object | Pydantic schemas everywhere | Constrained decoding rather than parsing prose | Pydantic v2 | `graph/schemas.py` |
| Tools | Callable functions for a model | 5 tools, exposed twice | Native schemas + LangChain `StructuredTool`, one implementation | langchain-core | `agents/tools.py` |
| Memory | Conversation history | Windowed buffer + rolling summary | Recent turns verbatim, older turns summarised | — | `memory/chat_memory.py` |
| Retrievers | Standard retrieval interface | `build_langchain_retriever` | Wraps the hybrid retriever as a `BaseRetriever` | langchain-core | `retrieval/retriever.py` |
| Build: multi-tool assistant | — | The research agent | Model picks its own searches | — | `agents/orchestrator.py` |

### Module 7 — LangGraph

| Topic | What it means | What was built | How | Technology | Proof |
|---|---|---|---|---|---|
| State management | Typed state passed between nodes | `GraphState` | TypedDict with reducers; checkpointed per thread | langgraph, SQLite saver | `graph/state.py` |
| Nodes | Units of work | **11 nodes** | guard, route, rewrite, retrieve, broaden, grade, generate, verify, finalize, refuse, answer_meta | langgraph | `graph/nodes.py` |
| Edges | Flow between nodes | Static + conditional | — | langgraph | `graph/build.py` |
| Conditional routing | Branch on state | 4 decision points | Every branch keys off a structured-output field, never a regex on prose | langgraph | `graph/build.py` |
| Multi-step workflows | Cycles, not just chains | **2 bounded retry loops** | retrieve↔grade and generate↔verify, each with a hard ceiling | langgraph | `graph/build.py` |
| Build: classification agent | — | The intent router | 6-way classification into the graph's branches | structured output | `graph/nodes.py · route` |

### Module 8 — AI Evaluation & Observability

| Topic | What it means | What was built | How | Technology | Proof |
|---|---|---|---|---|---|
| Hallucination detection | Catch unsupported claims | The `verify` node | Claim-level entailment against retrieved excerpts | fast model | `graph/nodes.py` |
| Response validation | Check output shape/content | Output guardrails | Citation indices validated as a set-membership check | regex | `core/guardrails.py` |
| Ground truth evaluation | Score against known answers | **CUAD golden set** | 510 contracts × 41 lawyer-marked span sets | — | `evaluation/golden_set.py` |
| LLM evaluation metrics | Quantify quality | 5 retrieval metrics + 3 judge axes | hit_rate, span_recall, precision, MRR, nDCG | numpy | `evaluation/metrics.py` |
| Tool: LangSmith | LLM tracing | Optional integration | Env-flag gated | langsmith | `core/observability.py` |
| Tool: Helicone | Proxy logging | Optional integration | `base_url` swap — zero code at call sites | — | `core/llm.py` |
| Tool: OpenTelemetry | Vendor-neutral tracing | Spans on every stage | Traces embedding/retrieval time too, not just LLM calls | opentelemetry-sdk | `core/observability.py` |
| Evaluate RAG responses | End-to-end scoring | The evaluation runner | Two modes: free retrieval-only, and full with generation | — | `evaluation/runner.py` |

### Module 9 — Advanced

| Topic | What it means | What was built | How | Technology | Proof |
|---|---|---|---|---|---|
| Agentic AI | Model decides its own steps | `ResearchAgent` | Bounded tool-calling loop | tool calling | `agents/orchestrator.py` |
| Sub-agents | Delegating to child agents | `Orchestrator` | Decompose → fan out **concurrently** → synthesise | asyncio.gather | `agents/orchestrator.py` |
| Orchestration layer | Coordinating the parts | The LangGraph workflow | Each sub-agent reads only its own results, so none drowns in context | langgraph | `graph/build.py` |

---

## Tier 1 — AI Application Developer

### Module 1 — Production RAG

| Topic | What was built | How | Technology | Proof |
|---|---|---|---|---|
| Advanced chunking | Structural + semantic splitters | Splits on `ARTICLE II > 5.2 > 5.2.1`; carries the section path into citations | custom | `ingestion/chunking.py` |
| Metadata filtering | 41-category clause taxonomy | Comma-wrapped CSV tags so `,non_compete,` cannot match `,non_competition,` | store `where` clauses | `ingestion/taxonomy.py`, `retrieval/bm25.py` |
| Hybrid search | RRF fusion of dense + sparse | Fuses on rank, because cosine and BM25 scores are not comparable | rank_bm25 | `retrieval/hybrid.py` |
| Document processing | PDF / DOCX / TXT / MD | Page-confidence flags: `exact` for PDFs, `estimated` for text extracts | pdfplumber, python-docx | `ingestion/loaders.py` |

### Module 2 — AI Application Development

| Topic | What was built | How | Technology | Proof |
|---|---|---|---|---|
| API integration | **26 REST endpoints** | Typed request/response models throughout | FastAPI, Pydantic | `api/` |
| Streaming | SSE token streaming | Queue passed through `config.configurable` so nodes can emit mid-execution | sse-starlette | `core/service.py` |
| Chat memory | Window + rolling summary | Prevents quadratic cost growth and history crowding out excerpts | fast model | `memory/chat_memory.py` |
| Session management | SQLite-backed sessions | Survives restart; TTL extends on activity | SQLite | `memory/session.py` |
| Rate limiting | Token bucket per client | Chosen over fixed windows, which allow a double burst at the boundary | custom | `core/rate_limit.py` |

### Module 3 — Agent Development

| Topic | What was built | How | Technology | Proof |
|---|---|---|---|---|
| Tool calling | 5 read-only tools | `search_contracts`, `list_documents`, `get_document_info`, `list_clause_types`, `compare_clause` | tool schemas | `agents/tools.py` |
| Function calling | `strict: true` schemas | Guarantees arguments validate | both SDKs | `agents/tools.py` |
| Multi-step agents | `ResearchAgent` | Iterative loop; model chooses each next search | — | `agents/orchestrator.py` |
| Agent guardrails | 4 structural limits | `max_steps`, `max_cost_usd`, `fan_out`, and **every tool read-only** | — | `agents/orchestrator.py` |

### Module 4 — AI Testing

| Topic | What was built | How | Technology | Proof |
|---|---|---|---|---|
| Prompt testing | Guardrail + refusal tests | Injection patterns, advice detection, citation contracts | pytest | `tests/test_guardrails.py` |
| RAG testing | Retrieval maths tests | RRF, MMR, BM25, metric bounds | pytest | `tests/test_retrieval.py` |
| Regression testing | `compare_runs` + config fingerprint | Two runs comparable only if you know what differed | SQLite | `evaluation/runner.py` |
| Evaluation frameworks | The runner + ablation sweep | Stratified sampling, concurrency, persisted results | asyncio | `evaluation/runner.py`, `scripts/ablate.py` |
| Automated tests | **189 tests** | Wired into CI | pytest | `tests/` |

### Module 5 — Deployment & Monitoring

| Topic | What was built | How | Technology | Proof |
|---|---|---|---|---|
| Docker | Two multi-stage images | Build deps discarded; non-root user; readiness healthcheck | Docker | `docker/` |
| CI/CD | 3-job pipeline | Backend tests, import sweep, frontend typecheck+build, both images | GitHub Actions | `.github/workflows/ci.yml` |
| AI monitoring | Spans + latency percentiles | Every stage traced; p50/p95 surfaced in the UI | OpenTelemetry | `core/observability.py` |
| Cost tracking | Per-stage ledger | Cache reads tracked separately; daily circuit breaker | SQLite | `core/cost.py` |

---

## Things built that the curriculum did not ask for

| What | Why it was needed |
|---|---|
| Ablation harness (`scripts/ablate.py`) | Hybrid search, MMR and re-ranking are all standard advice. One of them measurably hurt. Only measuring found it. |
| Metadata accuracy harness (`scripts/eval_metadata.py`) | The extractors feed metadata filters; a wrong value silently hides the right document. |
| Process-wide torch lock (`core/torch_lock.py`) | MPS is not thread-safe across models. Per-model locks were not enough — the conflict is at the device level. |
| Provider-agnostic `LLMClient` | Proved when OpenAI was added: not one call site changed. |
| Page-confidence flags | Two thirds of CUAD text extracts have no recoverable page numbers. Inventing them would be fabricating a citation. |
