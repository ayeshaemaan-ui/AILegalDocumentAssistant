# LegalLens — what was built, step by step

This document walks the system in the order a question actually travels through
it, and explains **what** each step does, **why** it is there, and **where** the
code lives. At the end there is a coverage table mapping every topic in the
curriculum to its implementation, and an honest account of the measured results
and the known limitations.

The governing principle throughout: a legal research tool is only useful if the
user can check it. Every design decision here serves that — clause-level
chunking so citations land on a real section, citation validation so a
fabricated reference is caught mechanically, an evaluation harness built on
lawyer annotations rather than model opinion, and an abstain-rather-than-guess
policy on metadata.

---

## 0. The dataset decides the design

**CUAD v1** — 510 commercial contracts, annotated by lawyers across 41 clause
categories, with the exact text spans that answer each one.

That choice shapes everything downstream:

- The 41 categories become the **metadata vocabulary** used for filtering, the
  facets in the UI, and the join key in the evaluation.
- The marked spans become the **ground truth** for retrieval scoring. A chunk
  either overlaps a lawyer-marked passage or it does not — no model judges it.
- The `master_clauses.csv` short answers ("Germany", "Yes") become **reference
  answers** for grading generated text.

Most RAG projects evaluate by asking a model whether its own answer looks good.
That measures plausibility. This measures correctness.

---

## 1. Document processing

**Code:** [`app/ingestion/loaders.py`](../backend/app/ingestion/loaders.py)

PDF, DOCX, TXT and Markdown in; normalised text plus a page map out.

Normalisation is deliberately conservative — whitespace collapsed, ligatures
expanded, words de-hyphenated across line breaks — but nothing is ever dropped.
A mangled clause is worse than an ugly one.

**The honest bit about page numbers.** PDFs give exact ones; `pdfplumber` knows
where each page starts. The CUAD text extracts do not — only about two thirds
carry a recoverable page sequence, and many repeat an exhibit number at every
break, which naively parsed would label all 60 chunks "page 18".

So each page carries a `confidence` flag (`exact` / `recovered` / `estimated`),
and the UI renders an estimated page as `~p.7`, never `p.7`. The primary
locator is the **section path**, which is both reliable and what a lawyer would
actually cite.

---

## 2. Chunking — the highest-leverage decision

**Code:** [`app/ingestion/chunking.py`](../backend/app/ingestion/chunking.py)

Retrieval can only ever return a chunk. If a chunk splits an obligation from
its exception, no amount of re-ranking or prompt engineering recovers the
meaning. Three strategies ship so the evaluation can *prove* which is best
rather than us asserting it:

| Strategy | How it splits | When it wins |
|---|---|---|
| `recursive` | Fixed windows on paragraph/sentence boundaries | Baseline — fast, structure-blind |
| `legal_structural` *(default)* | On the contract's own hierarchy: `ARTICLE II` → `5.2` → `5.2.1` | Commercial contracts, which are numbered |
| `semantic` | Where embedding distance between adjacent sentences spikes | Unstructured documents with no numbering |

The structural splitter carries the **section path** into chunk metadata, so a
citation reads `§ 2 > 2.3 > 2.3.1 Opportunity` instead of "chunk 47". Oversized
clauses are window-split with overlap but keep the parent path, so citation
survives the split. Sub-minimum fragments (`12. MISCELLANEOUS` alone on a line)
are merged forward — indexed on their own they are noise that outranks real
clauses on short queries.

---

## 3. Metadata — what turns search into research

**Code:** [`taxonomy.py`](../backend/app/ingestion/taxonomy.py),
[`metadata.py`](../backend/app/ingestion/metadata.py)

Without metadata every question searches all 510 contracts. With it, "what is
the liability cap in the Accuray agreement?" becomes a filtered search over one
document's liability clauses.

Two tiers:

- **Document level** — parties, contract type, governing law, dates. Extracted
  once, from the opening pages where they always live.
- **Chunk level** — clause-type tags from the 41-category taxonomy, plus content
  signals (`has_money`, `has_date`, `has_duration`).

The clause rules are **high precision, not high recall**: it is better to leave
a clause untagged (dense retrieval still finds it) than to mislabel it and have
a metadata filter hide the right answer.

The same principle drove three fixes found by measuring rather than assuming:

1. **Contract type reads the title first.** The Accuray distributor agreement
   name-drops a "Strategic Alliance Agreement" in its recitals; matching on the
   body labelled it wrongly. Titles are authoritative.
2. **Party extraction is case-aware.** A global `re.IGNORECASE` made `[A-Z]`
   match lowercase, so "a Delaware corporation" was extracted as a party. The
   suffix match is case-insensitive via a *scoped* flag; the name anchor is not.
3. **Governing law abstains.** "a corporation formed under the laws of Germany"
   is where a party was incorporated, not what governs the contract. And "the
   laws thereof" points back at a jurisdiction named earlier in the sentence.
   Both are now handled, and anything that does not clearly read as a
   governing-law clause returns `None`.

---

## 4. Embeddings

**Code:** [`app/core/embeddings.py`](../backend/app/core/embeddings.py)

`BAAI/bge-base-en-v1.5`, 768 dimensions, running locally. Free and offline,
which matters more than it sounds: re-indexing 510 contracts while tuning
chunking costs nothing and hits no rate limit.

Two details that are easy to get wrong:

- **Asymmetric prefixes.** BGE was trained with an instruction prefix on the
  query side only. Applying it to documents too measurably hurts recall, which
  is why `embed_query` and `embed_documents` are deliberately different.
- **Cosine, not Euclidean.** Cosine measures angle, not magnitude, so a one-line
  clause and a three-paragraph clause about the same obligation score as
  similar. Vectors are L2-normalised at encode time, which makes the cosine a
  plain dot product.

---

## 5. Vector storage

**Code:** [`app/retrieval/vectorstore.py`](../backend/app/retrieval/vectorstore.py)

All four backends from the curriculum sit behind one interface — **Chroma**
(default, embedded), **PGVector**, **Pinecone**, **Weaviate** — switchable with
one env var, with lazy imports so the app never requires a driver it will not
use. What differs between them is operational, not semantic: Chroma needs no
server, PGVector rides your existing Postgres, Pinecone is managed, Weaviate has
hybrid search inside the engine.

---

## 6. Hybrid retrieval

**Code:** [`bm25.py`](../backend/app/retrieval/bm25.py),
[`hybrid.py`](../backend/app/retrieval/hybrid.py),
[`retriever.py`](../backend/app/retrieval/retriever.py)

Dense embeddings are good at meaning and bad at **identifiers**. In legal text
the identifiers are often the whole question: a party name, `Section 12.3`,
`$5,000,000`, "Delaware". A 768-dim vector smooths those into the surrounding
semantics — ask for "Siemens" and dense retrieval happily returns a distributor
clause naming someone else.

BM25 has the opposite bias. Running both and fusing them is the entire argument
for hybrid search.

**Reciprocal Rank Fusion** solves the score-comparability problem by discarding
scores and fusing on rank:

```
RRF(chunk) = Σ  weight / (k + rank_in_that_list)
```

A chunk ranked #1 by one arm and #30 by the other beats a chunk ranked #15 by
both — agreement is good, but a strong signal from either arm should survive.

The tokeniser is legal-aware: `$5,000,000` and `12.3` survive as single tokens
rather than being shredded by a naive `\w+` split.

> **A bug worth recording.** Candidate selection originally used `score > 0`.
> BM25's IDF term goes *negative* for a token appearing in every document, so on
> a small corpus the most common legal terms silently returned nothing — exactly
> the case the sparse arm exists to help. Candidacy is now decided by actual
> lexical overlap, and scores are min-max normalised.

---

## 7. Re-ranking

**Code:** [`app/retrieval/rerank.py`](../backend/app/retrieval/rerank.py)

Retrieval and re-ranking answer different questions, which is why doing both
beats doing either well.

The embedding model is a **bi-encoder**: it encodes query and clause separately
and compares vectors. That is what makes it fast enough to search 40,000 chunks
— but it never sees the two texts together, so it cannot notice that a clause
says "*Supplier* may terminate" when the question asked what *Customer* may do.

A **cross-encoder** reads both in one forward pass. Far more accurate, far too
slow for the whole corpus — so it runs only over the ~20 candidates hybrid
search already shortlisted. Retrieve wide, re-rank narrow.

> **A second bug worth recording.** Running the evaluation at concurrency 4
> segfaulted the process: Apple's MPS backend is not safe under concurrent
> forward passes. Per-model locks were not enough, because the conflict is at
> the *device* level — the bi-encoder and the cross-encoder share one GPU. The
> fix is a single process-wide lock in
> [`torch_lock.py`](../backend/app/core/torch_lock.py).

---

## 8. Prompting

**Code:** [`app/prompts/templates.py`](../backend/app/prompts/templates.py)

Every technique from the prompting module is used, each for a reason:

| Technique | Where | Why there |
|---|---|---|
| **Role prompting** | Answer system prompt | Pins the model to "research assistant, not lawyer" — this is what makes it decline advice |
| **Zero-shot** | The answer instructions | The task is well specified; examples are not needed for it |
| **One-shot** | Query rewriting | One worked example is enough to teach the output shape |
| **Few-shot** | Answer prompt | Four examples, each teaching a different failure mode: found, partial, **absent**, advice-seeking |
| **Chain of thought** | Answer + groundedness judge | Reason about which clause governs before committing |
| **Output formatting** | The `[n]` citation contract | Machine-checkable, which is what makes validation possible |
| **Prompt templates** | `build_*` functions | Parameterised, versioned, testable |

The "absent" few-shot example is the one that matters most. Without it, a model
asked about a liability cap that is not in the excerpts will reach for an
insurance clause and present it as an answer. With it, "the excerpts do not
state a cap" becomes a correct and expected response.

**One structural rule governs the whole file: the system prompt is byte-stable.**
It is the cached prefix of every request, so anything volatile — the question,
the retrieved chunks, the date — goes in the user turn. A single changing
character would invalidate the cache on every call and multiply input cost by
roughly ten.

---

## 9. Orchestration — the LangGraph workflow

**Code:** [`app/graph/`](../backend/app/graph/)

```
                        START
                          │
                     guard_input
                     ╱         ╲
               (blocked)      (ok)
                   │             │
                refuse         route ──────────────┐
                   │          ╱      ╲             │
                  END   (advice/  (document/     (meta)
                         out of    summarize/      │
                         scope)    compare)    answer_meta
                            │         │             │
                         refuse  rewrite_query     END
                            │         │
                           END    retrieve ◄──────┐
                                      │            │
                                 grade_context     │
                                  ╱        ╲       │
                            (enough)   (not enough)│
                                 │          ╲      │
                                 │          broaden┘
                                 │
                              generate ◄────────┐
                                 │               │
                              verify             │
                              ╱     ╲            │
                       (grounded) (ungrounded)   │
                             │          ╲────────┘
                          finalize
                             │
                            END
```

Two cycles make this a workflow rather than a chain, and both are bounded:

- `retrieve → grade → broaden → retrieve` recovers from a bad first search
- `generate → verify → generate` recovers from an ungrounded answer

That self-correction is the practical difference between a demo and something
you would trust. A plain chain generates from whatever it retrieved and reports
whatever it generated. This one checks its own work at both stages and gets a
second attempt — with a hard ceiling, so a persistent failure ends in a degraded
answer rather than an unbounded bill.

Each decision node returns **structured output**, not prose
([`schemas.py`](../backend/app/graph/schemas.py)). An edge that depends on
regexing "yes" out of a sentence is a bug waiting for the model to rephrase
itself; with a constrained schema, the model physically cannot return a shape
the router does not understand.

---

## 10. Guardrails

**Code:** [`app/core/guardrails.py`](../backend/app/core/guardrails.py)

Input guardrails are cheap, deterministic, and run first — length limits,
prompt-injection patterns, and an advice-seeking pre-screen. They exist to stop
a request before it costs a model call.

Output guardrails validate what came back. The important one is **citation
validation**: the prompt promises every claim carries a `[n]` pointing at a real
excerpt, and this checks that promise mechanically. A model that cites `[7]`
when given 6 excerpts has fabricated its own evidence — a parsing problem, not
one needing an LLM judge.

Two deliberate choices:

- Guardrails **annotate rather than silently rewrite**. A blocked answer is
  replaced; a low-confidence one passes through with a visible warning, because
  hiding a weak answer is worse for a research tool than showing its weakness.
- An answer that correctly says "the excerpts do not address X" is **exempt from
  the citation requirement**. Refusing to invent is the behaviour we want, and
  penalising it would train the system toward confident guessing.

Structurally, user text only ever enters the *user* turn, never the system
prompt — so an injected instruction competes with a cached system prompt rather
than becoming part of it.

---

## 11. Agents

**Code:** [`app/agents/`](../backend/app/agents/)

The RAG graph answers one question with one retrieval. Some questions are not
that shape: "compare the termination rights across all three distributor
agreements" needs several searches whose parameters depend on earlier results.

- **`ResearchAgent`** — a bounded tool-calling loop over five read-only tools
  (`search_contracts`, `list_documents`, `get_document_info`,
  `list_clause_types`, `compare_clause`). The model chooses its own searches.
- **`Orchestrator`** — decomposes a broad question into independent
  sub-questions, runs a sub-agent per sub-question **concurrently**, then
  synthesises. The reason is context, not speed: each sub-agent reads only its
  own results, so none drowns in a window full of other sub-questions' excerpts.

Guardrails here are structural: `max_steps` bounds the loop, `max_cost_usd`
bounds spend, `fan_out` bounds decomposition, and **every tool is read-only** —
an agent loop can be wrong; it should not be able to be destructive.

Tools are defined once and exposed twice: as Anthropic tool schemas with
`strict: true`, and as LangChain `StructuredTool`s — same functions, no second
implementation.

---

## 12. Memory and sessions

**Code:** [`app/memory/`](../backend/app/memory/)

The Messages API is stateless. Anything that feels like memory is state the
application keeps and replays.

Naive memory replays the whole transcript every turn, which breaks twice: input
cost grows quadratically, and eventually the history crowds out the retrieved
excerpts — the part that actually answers. The strategy here is a **windowed
buffer plus a rolling summary**:

```
[ rolling summary of turns 1..n-6 ]  +  [ last 6 turns verbatim ]
```

Recent turns stay exact because follow-ups depend on precise wording ("what
about clause 9.2?"). Older turns compress into a summary that keeps the facts a
follow-up might need and drops the phrasing. Summarisation runs on the cheap
model, off the critical path.

Sessions are persisted to SQLite rather than a dict, so a restart does not wipe
open conversations, two workers see the same session, and an evaluation run can
replay real history.

---

## 13. Streaming

**Code:** [`service.py`](../backend/app/core/service.py),
[`routes_chat.py`](../backend/app/api/routes_chat.py)

LangGraph runs nodes to completion, so it cannot emit tokens mid-node. The
service creates an `asyncio.Queue`, passes it through `config.configurable`, and
runs the graph as a background task; the `generate` node pushes tokens onto the
queue as Claude produces them while the service drains it into SSE frames.

The same graph therefore serves both the live UI and the batch evaluation
runner — the only difference is whether a queue was supplied.

SSE rather than WebSockets because this is one-directional: SSE is plain HTTP,
survives proxies, and reconnects on its own. A WebSocket would be more machinery
for less benefit.

One subtlety: when the graph regenerates after a failed groundedness check, the
client would otherwise append the retry to the draft. A `restart` frame tells it
to clear the buffer.

---

## 14. Evaluation

**Code:** [`app/evaluation/`](../backend/app/evaluation/)

Two modes, because they answer different questions:

- **`retrieval_only`** — scores retrieval against lawyer-marked spans. No model
  calls, so it is **free**. This is the mode to live in while tuning chunking or
  retrieval weights, because it isolates the variable you changed.
- **`full`** — also generates answers and grades them (faithfulness + judge
  against the lawyer's reference). Costs money; run it when measuring end to end.

**Metrics, and why more than one:**

| Metric | Question it answers |
|---|---|
| `hit_rate` | Did *any* relevant clause make the top k? The floor — if this is low, nothing downstream can save the answer |
| `span_recall` | What fraction of the lawyer-marked passages did we retrieve? |
| `precision` | What fraction of what we returned was relevant? Low precision burns context on noise |
| `mrr` | How high did the first relevant chunk rank? |
| `ndcg` | Rank-weighted relevance — the best single-number summary |

The pair to watch is **hit_rate vs MRR**. High hit_rate with low MRR means the
retriever is finding the right clause and burying it — a re-ranking problem, not
a retrieval problem, and the most common failure in a system like this.

> **A metrics bug worth recording.** The first run reported `recall = 2.0` and
> `nDCG = 1.448` — both impossible. The cause was conflating two units: a gold
> **span** is one lawyer-marked passage, a **chunk** is one retrieval result, and
> a single span routinely spills across several adjacent chunks. Using chunk
> counts as the recall denominator let recall exceed 1. Span recall is now
> measured in spans; precision, MRR and nDCG in chunks.

Every run is stored with the **configuration that produced it**, which is what
turns evaluation into regression testing — two runs are only comparable if you
know what differed. `compare_runs` reports the deltas and the config diff.

---

## 15. Observability and cost

**Code:** [`observability.py`](../backend/app/core/observability.py),
[`cost.py`](../backend/app/core/cost.py)

A RAG answer is the output of eight or nine steps. When one is wrong, "the
answer was bad" is useless — you need to know *which*. Every stage opens a span.

Three tools, each for what it is good at: **OpenTelemetry** for vendor-neutral
spans across LLM and non-LLM work alike (the only one that traces embedding and
retrieval time, which is usually where the latency actually is); **LangSmith**
for prompt/completion-level LLM tracing; **Helicone** as a reverse proxy, which
needs no code at the call sites at all. All three are optional — with none
configured, spans still land in local SQLite, so the tracing story works offline.

Every model call writes one row to a cost ledger, broken down **by stage**. That
breakdown is the useful part: it shows routing, rewriting and grading cost
pennies while generation is the bill. Cache reads are tracked separately because
caching is the biggest available lever — `cache_hit_rate` is how you prove it is
working rather than silently invalidating.

---

## 16. Production concerns

**Rate limiting** ([`rate_limit.py`](../backend/app/core/rate_limit.py)) — a
token bucket per client, chosen over a fixed window because fixed windows allow
a double burst across the boundary. Plus a **cost circuit breaker**: rate
limiting alone does not bound spend, since 30 requests/minute against a large
context is a large bill.

**Deployment** — multi-stage Dockerfiles (build deps discarded from the runtime
image, non-root user, healthcheck that tests *readiness* rather than liveness),
and a compose stack that needs no external services by default.

**CI** ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)) — tests, an
import sweep over every module (catches circular imports the tests may not
reach), frontend typecheck and build, and both Docker images.

---

## Measured results

### Metadata extraction, 500 contracts vs lawyer-written answers

| Field | Result |
|---|---|
| Parties | **93.4%** recall (returned nothing on 13) |
| Contract type | **95.6%** match |
| Governing law | **95.8%** precision when it answers; 63.7% recall, abstains on 33% |

The governing-law profile is intentional. A wrong value becomes a metadata
filter that silently hides the right document, so the extractor abstains rather
than guessing.

### Retrieval ablation

80 questions, scored against lawyer-marked spans, retrieval scoped to the
correct document. Reproduce with
[`scripts/ablate.py`](../backend/scripts/ablate.py).

| Configuration | hit_rate | span_recall | precision | MRR | nDCG |
|---|---|---|---|---|---|
| dense only | **0.713** | **0.643** | 0.173 | 0.526 | 0.570 |
| dense + MMR | 0.675 | 0.581 | 0.133 | 0.508 | 0.545 |
| hybrid | 0.650 | 0.590 | 0.163 | 0.505 | 0.536 |
| hybrid + MMR | 0.688 | 0.588 | 0.135 | 0.511 | 0.550 |
| dense + rerank | 0.675 | 0.603 | 0.167 | 0.538 | 0.566 |
| **hybrid + rerank** *(default)* | 0.700 | 0.631 | **0.175** | **0.540** | **0.573** |
| hybrid + MMR + rerank | 0.662 | 0.586 | 0.152 | 0.525 | 0.556 |

**What this actually shows, stated carefully:**

1. **MMR hurt in every pairing** — −0.024, −0.019 and −0.014 nDCG against its
   comparable baseline. That is the one large, consistent, directionally stable
   effect in the table, so **MMR is now off by default**. The reason is
   legible: these questions are scoped to a single contract, where near-duplicate
   clauses are usually *all* relevant, so trading relevance for diversity costs
   more than it returns. It remains configurable, and is worth re-testing for
   unscoped search across all 510 contracts, where one document can flood the
   candidate pool.

2. **The gap between the top two configurations is not significant.**
   hybrid + rerank (0.573) versus dense only (0.570) is a 0.003 difference at
   n=80 — noise. It would be dishonest to present that as hybrid search winning.
   Hybrid + re-ranking is kept as the default on two other grounds: it has the
   best MRR (0.540) and precision (0.175), meaning it puts the right clause
   higher and wastes fewer context tokens; and the ablation **handicaps BM25**,
   because retrieval-only mode runs without the LLM query rewriter, so the sparse
   arm receives a conversational question instead of the extracted keyword query
   it gets in production. Its measured floor here is therefore lower than its
   real behaviour.

3. **Hybrid without re-ranking was worse than dense alone** (0.536 vs 0.570),
   which is consistent with point 2 — unrewritten conversational queries are
   BM25's weak case, and the cross-encoder is what recovers the ranking.

The broader point is the method, not the numbers: hybrid search, MMR and
cross-encoder re-ranking are all standard advice, and standard advice is not
evidence. One of the three measurably hurt on this corpus, and it was only
findable by measuring.

## Known limitations

Stated plainly, because a research tool that overstates itself is worse than one
that does not exist:

1. **Retrieval is good, not solved.** Roughly a third of questions do not surface
   a lawyer-marked span in the top 6. The weakest categories are the diffuse ones
   — competitive-restriction carve-outs, affiliate licence grants — where the
   answer is spread across several clauses rather than stated in one.
2. **Page numbers are estimated for text extracts.** Flagged in the data and
   marked `~` in the UI, but they are still estimates. PDFs give exact pages.
3. **Clause tagging is rule-based**, so it is high precision and moderate recall.
   Untagged clauses are still found by dense retrieval; they just cannot be
   filtered on.
4. **The LLM judge is a proxy.** It is useful for comparing two versions of this
   system, not for claiming an absolute quality level.
5. **BM25 is an in-memory full rebuild.** Fine at 4,200 chunks, and fine at
   40,000; at a much larger scale it belongs in Postgres full-text or Weaviate's
   native hybrid.
6. **Single-process assumptions.** The rate limiter, job board and BM25 index are
   in-process. Multiple replicas would need Redis for the first two and a shared
   index for the third.
7. **English only**, and tuned on US-style commercial contracts. The clause
   patterns and the embedding model both reflect that.

---

## Curriculum coverage

### Shared Foundation

| Module | Topic | Where |
|---|---|---|
| **1. AI & LLM Fundamentals** | Tokens & tokenisation | `count_tokens` via the API — never `tiktoken`, which is OpenAI's tokeniser and undercounts Claude by 15–20% ([`llm.py`](../backend/app/core/llm.py)) |
| | Context windows | Retrieve 40 candidates, inject 6 — context is the scarce resource ([`retriever.py`](../backend/app/retrieval/retriever.py)) |
| | Temperature / sampling | Effort control via `output_config` rather than temperature on current models ([`llm.py`](../backend/app/core/llm.py)) |
| | Hallucinations | Groundedness audit + citation validation ([`guardrails.py`](../backend/app/core/guardrails.py), `verify` node) |
| | Fine-tuning vs prompting vs RAG | RAG chosen: the corpus changes per user and answers must cite a source — neither is true of a fine-tune |
| **2. LLM APIs** | Anthropic Claude | [`llm.py`](../backend/app/core/llm.py) |
| | Provider abstraction | `LLMClient` ABC — a second provider is a subclass, not a refactor |
| | Chat completion | `complete()` |
| | Structured outputs / JSON mode | `structured()` + [`schemas.py`](../backend/app/graph/schemas.py) |
| | Function calling | [`agents/tools.py`](../backend/app/agents/tools.py) |
| | Streaming | `stream()` → SSE |
| | Summarisation build | `summarize` intent + rolling memory summary |
| **3. Prompt Engineering** | Zero / one / few-shot, CoT, role, formatting, templates | [`templates.py`](../backend/app/prompts/templates.py) — see §8 |
| **4. Embeddings & Vector Search** | Embeddings, cosine, semantic search | [`embeddings.py`](../backend/app/core/embeddings.py) |
| | Dense vs sparse | [`bm25.py`](../backend/app/retrieval/bm25.py) vs dense — see §6 |
| | Chroma / PGVector / Pinecone / Weaviate | [`vectorstore.py`](../backend/app/retrieval/vectorstore.py) — all four |
| | Semantic search build | The retrieval pipeline itself |
| **5. RAG** | Architecture | Offline ingest / online query split ([`pipeline.py`](../backend/app/ingestion/pipeline.py)) |
| | Chunking strategies | Three, comparable by evaluation ([`chunking.py`](../backend/app/ingestion/chunking.py)) |
| | Metadata | 41-category taxonomy ([`taxonomy.py`](../backend/app/ingestion/taxonomy.py)) |
| | Retrieval pipeline | [`retriever.py`](../backend/app/retrieval/retriever.py) |
| | Context injection | `build_answer_user_turn` — excerpts first, instruction last |
| | Re-ranking | [`rerank.py`](../backend/app/retrieval/rerank.py) |
| | PDF chatbot / KB assistant builds | The application |
| **6. LangChain** | Retrievers | `build_langchain_retriever` |
| | Tools | `build_langchain_tools` |
| | Memory | `build_langchain_memory` — same store as the FastAPI path |
| | Prompt templates, output parsers | [`templates.py`](../backend/app/prompts/templates.py), Pydantic schemas |
| | Text splitters | `RecursiveCharacterTextSplitter` baseline |
| **7. LangGraph** | State management | [`state.py`](../backend/app/graph/state.py) — typed, reducers, checkpointed |
| | Nodes | [`nodes.py`](../backend/app/graph/nodes.py) — 11 |
| | Edges & conditional routing | [`build.py`](../backend/app/graph/build.py) |
| | Multi-step workflows | Two bounded self-correction cycles |
| **8. Evaluation & Observability** | Hallucination detection | `verify` node, claim-level verdicts |
| | Response validation | [`guardrails.py`](../backend/app/core/guardrails.py) |
| | Ground truth evaluation | CUAD spans ([`golden_set.py`](../backend/app/evaluation/golden_set.py)) |
| | LLM evaluation metrics | [`metrics.py`](../backend/app/evaluation/metrics.py), [`llm_judge.py`](../backend/app/evaluation/llm_judge.py) |
| | LangSmith / Helicone / OpenTelemetry | [`observability.py`](../backend/app/core/observability.py) — all three |
| **9. Advanced** | Agentic AI | [`orchestrator.py`](../backend/app/agents/orchestrator.py) |
| | Multiple sub-agents | `Orchestrator` — parallel fan-out |
| | Orchestration layer | Decompose → dispatch → synthesise |

### Tier 1 — AI Application Developer

| Module | Topic | Where |
|---|---|---|
| **1. Production RAG** | Advanced chunking | Structural + semantic strategies |
| | Metadata filtering | `RetrievalFilters` → store `where` clauses |
| | Hybrid search | RRF fusion ([`hybrid.py`](../backend/app/retrieval/hybrid.py)) |
| | Document processing | [`loaders.py`](../backend/app/ingestion/loaders.py) |
| | KB assistant build | The application |
| **2. AI Application Development** | API integration | [`api/`](../backend/app/api/) — 26 endpoints |
| | Streaming | SSE — see §13 |
| | Chat memory | Windowed buffer + rolling summary |
| | Session management | [`session.py`](../backend/app/memory/session.py) |
| | Rate limiting | [`rate_limit.py`](../backend/app/core/rate_limit.py) |
| **3. Agent Development** | Tool calling / function calling | [`tools.py`](../backend/app/agents/tools.py) |
| | Multi-step agents | `ResearchAgent` |
| | Agent guardrails | Step, cost and fan-out ceilings; read-only tools |
| **4. AI Testing** | Prompt testing | `tests/test_guardrails.py` |
| | RAG testing | `tests/test_retrieval.py`, `tests/test_evaluation.py` |
| | Regression testing | `compare_runs` + config fingerprinting |
| | Evaluation frameworks | [`runner.py`](../backend/app/evaluation/runner.py) |
| | Automated tests | 156 tests, wired into CI |
| **5. Deployment & Monitoring** | Docker | [`docker/`](../docker/) — multi-stage, non-root, healthchecked |
| | CI/CD | [`ci.yml`](../.github/workflows/ci.yml) |
| | AI monitoring | Spans, latency percentiles, the Observability page |
| | Cost tracking | Per-stage ledger + daily circuit breaker |
