# LegalLens

Retrieval-Augmented Generation over commercial contracts. Ask a question in plain
English, get an answer drawn only from the indexed documents, with a citation to
the clause it came from.

Built on **CUAD v1** — 510 real commercial contracts annotated by lawyers across
41 clause categories. Those annotations are what make the evaluation in this
project objective rather than self-assessed: retrieval is scored against
human-marked text spans, not against a model's opinion of its own output.

> LegalLens is a **document research tool**. It reports what contracts say. It
> does not give legal advice, and it declines to when asked.

---

## What it does

- **Hybrid retrieval** — dense embeddings for meaning, BM25 for exact terms
  (party names, `$5,000,000`, `Section 12.3`), fused by Reciprocal Rank Fusion.
- **Clause-aware chunking** — splits on the contract's own hierarchy
  (`ARTICLE II > 5.2 > 5.2.1`) so a chunk is a clause, and citations point at a
  section a lawyer would recognise.
- **Metadata filtering** — every chunk is tagged against CUAD's 41 clause
  categories, plus document, contract type and governing law.
- **Self-correcting pipeline** — a LangGraph workflow that grades its own
  retrieval before generating, and re-generates if the answer is not grounded in
  its own sources.
- **Verified citations** — every `[n]` is validated against what was actually
  retrieved. A fabricated citation is blocked, not shipped.
- **Measured, not asserted** — an evaluation harness scoring hit rate, span
  recall, MRR and nDCG against lawyer-annotated spans.

---

## Quick start

### Requirements

- Python 3.12 (PyTorch has no 3.14 wheels yet)
- Node 22+
- ~2GB disk for the embedding and re-ranking models
- An `ANTHROPIC_API_KEY` — only for generating answers. **Indexing, retrieval
  and retrieval evaluation all run with no key and no network.**

### Setup

```bash
git clone <repo> && cd AILegalDocumentAssistant
cp .env.example .env          # add ANTHROPIC_API_KEY, set CUAD_DIR

python3.12 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt

cd frontend && npm install && cd ..
```

### Index the corpus

```bash
cd backend
PYTHONPATH=$PWD ../.venv/bin/python -m scripts.ingest_cuad --limit 50   # ~6 min
PYTHONPATH=$PWD ../.venv/bin/python -m scripts.ingest_cuad --limit 510  # full corpus
```

### Run

```bash
# Terminal 1 — API on :8000
cd backend && PYTHONPATH=$PWD ../.venv/bin/uvicorn app.main:app --reload

# Terminal 2 — UI on :3400
cd frontend && npm run dev   # http://localhost:3400
```

Or the whole stack in Docker:

```bash
docker compose up --build
```

---

## Evaluate

Retrieval evaluation makes **no model calls**, so it is free and runs offline:

```bash
cd backend
PYTHONPATH=$PWD ../.venv/bin/python -m scripts.run_eval --retrieval-only -n 80
PYTHONPATH=$PWD ../.venv/bin/python -m scripts.ablate -n 80        # component sweep
PYTHONPATH=$PWD ../.venv/bin/python -m scripts.eval_metadata       # extractors vs ground truth
```

Full evaluation additionally generates and grades answers, and costs money:

```bash
PYTHONPATH=$PWD ../.venv/bin/python -m scripts.run_eval -n 25
```

Every run is recorded with the configuration that produced it, so two runs can
be compared as a regression check:

```bash
PYTHONPATH=$PWD ../.venv/bin/python -m scripts.run_eval --list
PYTHONPATH=$PWD ../.venv/bin/python -m scripts.run_eval --compare RUN_A RUN_B
```

---

## Measured results

Metadata extractors, scored against CUAD's lawyer-written answers across **500
contracts**:

| Field | Result |
|---|---|
| Parties | **93.4%** recall |
| Contract type | **95.6%** match |
| Governing law | **95.8%** precision when it answers (abstains on 33%) |

The governing-law extractor abstains rather than guessing. A wrong value becomes
a metadata filter that silently hides the right document, which is worse than no
filter at all.

Retrieval, scored against lawyer-marked clause spans, is reported in
[`docs/TECHNIQUES.md`](docs/TECHNIQUES.md#measured-results) along with the
ablation showing what each component is actually worth.

---

## Project layout

```
backend/
  app/
    config.py            every tunable knob, one place
    core/                LLM, embeddings, guardrails, cost, rate limits, tracing
    ingestion/           loaders, chunking strategies, clause taxonomy, metadata
    retrieval/           vector stores, BM25, hybrid fusion, re-ranking
    graph/               LangGraph state, nodes, conditional edges
    agents/              tool definitions, research agent, orchestrator
    memory/              sessions, chat memory
    evaluation/          golden set, metrics, LLM judge, runner
    prompts/             the prompt library
    api/                 FastAPI routes
  scripts/               ingest, evaluate, ablate
  tests/                 156 tests
frontend/                Next.js 16 UI
docker/                  backend + frontend images
docs/TECHNIQUES.md       what was built, step by step, and why
```

---

## Documentation

**[docs/TECHNIQUES.md](docs/TECHNIQUES.md)** — a step-by-step walk through every
technique used, what it does, why it is there, and where the code lives.
