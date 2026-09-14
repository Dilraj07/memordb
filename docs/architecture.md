# Temporal Memory Engine — Final Architecture & Build Plan

(Consolidated version — supersedes prior architecture drafts)

## 1. What This Is

An embedded, single-file Python library that gives LLM agents a memory layer which
knows the difference between what was once true and what's true now — plus a
companion pytest plugin that lets any developer (using this engine or any other
memory backend) test for that specific bug class in CI.

Two deliverables, not one:
- **temporal_memory** — the engine (store, extractor, resolver, retriever).
- **pytest-temporal-memory** — a plugin that tests any backend for fact-drift/multi-hop correctness, with this engine as its reference implementation.

## 2. Why This Problem, Stated Precisely

The bug: an agent answers using a fact that was true once but no longer is,
because nothing in the pipeline tracked when it stopped being true.

Why a smarter LLM doesn't fix it: the failure happens before the model sees
anything. A retrieval layer returns facts by similarity-to-query, not by "what's
currently true." If it hands the model a stale fact — or hands it both the old and
new fact but scrambled out of chronological order — the model is reasoning correctly
over wrong or disordered input. Documented research on LLM temporal reasoning
(event-ordering studies from late 2025) shows models specifically degrade when
chronological signals are disrupted, which is exactly what similarity-ranked
retrieval does to a fact's timeline. This is an information-access problem, not an
intelligence problem, and it does not shrink as models get smarter.

Proof this persists at the frontier today: Mem0's own V3 upgrade — tested against
current frontier models — measured its largest single improvement specifically on
temporal queries (+29.6 pts) and multi-hop reasoning (+23.1 pts) after adding a
dedicated resolution layer. If model intelligence alone solved this, adding a
resolver wouldn't have moved that number.

Why it doesn't disappear as context windows grow: stuffing full history into
every call doesn't scale on cost, and models measurably lose accuracy on information
buried in very long contexts regardless of window size. Selective, correct
retrieval remains necessary no matter how large context windows get.

## 3. Why 2026, Specifically

- Agent memory has become a first-class architectural component with its own
benchmark suite (LoCoMo, LongMemEval) and research literature — it's no longer a
side feature, it's a recognized discipline with measurable trade-offs.
- The mechanism this project uses — bitemporal versioning (valid_from/valid_to)
— is a 40-year-old, proven database pattern from banking/insurance/healthcare.
Applying it to LLM memory is a smart application of settled theory, not a fragile
new bet — the part of this project least likely to look dated in two years.
- MCP has become the default way local tools plug into Claude Desktop, Cursor, and
Claude Code by mid-2026 — a real, low-effort distribution channel that didn't
exist in this form a year ago.
- The eval/testing side of the agent ecosystem (DeepEval, pytest-evals, Inspect AI)
has matured into pytest-native, plugin-based tooling — meaning "ship a plugin for
the ecosystem's existing test runner" is now a normal, expected pattern, not a
novel ask.

## 4. Honest Market Positioning

What's already solved by incumbents, don't claim to fix it:
- Mem0 (Apache-2.0) already runs fully embedded/local (Chroma + Ollama, no Docker
required) and already does bitemporal-style conflict resolution (V3, 91.6 LoCoMo).
- Evermind.ai benchmarks above Mem0 on the same tasks, also open-source/self-hosted.
- DeepEval (15.7k stars) already owns general pytest-native LLM/agent evaluation.

What's genuinely still open:
- No incumbent ships a single-SQLite-file engine (Mem0's simplest config still
uses a separate Chroma directory alongside its SQLite history file) — real,
narrow value for games, CLI tools, edge/offline apps, anything wanting true
one-file portability.
- No incumbent ships a dedicated, drop-in pytest plugin specifically for temporal
fact-drift and multi-hop contradiction testing, usable against arbitrary memory
backends. General agent-eval tools test hallucination/faithfulness/tool-use — none
specialize in this one bug class the way pytest-asyncio specializes in async.

The pitch to actually use:
"I reimplemented the core mechanism production memory systems use — from scratch,
benchmarked against my own test set — and built the missing correctness-testing tool
for it. Here's precisely where Mem0/Evermind already cover this ground, and here's
the narrow gap I'm filling instead of pretending to replace them."

Realistic goal: real usage among a specific community (indie builders, hackathon
teams, LangGraph/agent devs wanting a lightweight CI check) — not numpy/FastAPI-scale
adoption. That bar isn't reachable through engineering choices alone in a space this
saturated, and claiming otherwise would set you up to be caught out by the first
person who asks a hard question.

## 5. System Architecture

```
                    ┌──────────────────────┐
   chat message ───►│   ExtractionWorker    │  (background thread + queue.Queue,
                    │  (non-blocking submit)│   NOT asyncio -- store is sync sqlite3)
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │   Fact Extractor      │  LLM call, JSON mode,
                    │ Pydantic-validated,   │  Tenacity retry w/ backoff
                    │ Tenacity-retried      │  on 429s / malformed JSON
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │  Conflict Resolver     │  fast path: known single-valued
                    │ ADD / UPDATE /         │  predicates (lives_in, works_at)
                    │ DUPLICATE              │  slow path: LLM judge for novel ones
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │  Bitemporal Store      │  SQLite, WAL mode, owner_id-scoped,
                    │ valid_from/valid_to,   │  PRAGMA user_version migrations,
                    │ owner_id isolation     │  composite index (owner,subj,pred,valid_to)
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
   query ──────────►│  Hybrid Retriever      │  vector (TF-IDF/embeddings) +
                    │ vector+BM25+entity,    │  recursive-CTE
                    │ recursive-CTE          │  multi-hop, LLM rerank pass;
                    │ multi-hop, LLM rerank  │  WITH RECURSIVE for 2-3 hop traversal
                    └──────────┬───────────┘
                               ▼
                          answer to user

   ┌─────────────────────────────────────────────────────────┐
   │  pytest-temporal-memory  (separate package, adapter-based)│
   │  - fixtures: planted-contradiction conversations           │
   │  - assertions: assert_fact_active(), assert_no_contradiction()│
   │  - works against THIS engine or any other backend via adapter│
   └─────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────┐
   │  MCP adapter (thin, secondary interface)                    │
   │  - exposes MemoryEngine as a local stdio MCP server         │
   │  - lets it plug into Claude Desktop / Cursor / Claude Code  │
   │  - NOT the headline pitch -- crowded, skepticism-prone niche│
   └─────────────────────────────────────────────────────────┘
```

## 6. How It Works, End to End

- **Write**: a message comes in, gets queued (worker.submit()), returns
immediately — extraction happens off the main thread.
- **Extract**: the queued job hits the LLM with a strict JSON-schema prompt;
Pydantic validates the shape; Tenacity retries transient failures (rate limits,
malformed JSON) with exponential backoff before giving up and logging loudly.
- **Resolve**: for each extracted fact, check active facts for the same
(owner_id, subject, predicate). Known single-valued predicates (location, job)
go straight to UPDATE if the object differs. Novel predicates fall back to an
LLM judge call. Identical facts are DUPLICATEs and are skipped.
- **Store**: ADD inserts a new row (valid_to = NULL). UPDATE stamps the old row's
valid_to = now() and inserts the new one. Nothing is ever deleted by default —
history is preserved unless pruning is explicitly enabled.
- **Read**: a query triggers the hybrid retriever — vector similarity, BM25 keyword
match, and entity overlap are computed, normalized, and fused into one score; the
top pool is reranked by an LLM judge; multi-hop questions walk the graph via
WITH RECURSIVE over the same table, no separate graph database needed.
- **Verify**: the pytest plugin runs a fixed set of planted-contradiction and
multi-hop test conversations through whichever backend it's pointed at (adapter
pattern), and asserts the answer reflects only currently-valid facts.

## 7. Build Plan (dependency-correct order)

| Phase | Description | Depends on | Status |
|---|---|---|---|
| 1 | Bitemporal store + extractor + resolver (owner-scoped, WAL, migrations, retry-hardened) | — | Built — needs a live run with a real API key to confirm end-to-end |
| 2 | Mini benchmark (10-20 planted-contradiction cases) | Phase 1 | Not started — do this next |
| 3 | Hybrid retriever (vector+BM25+entity, LLM rerank) | Phase 1 | Built, not yet wired into the worker flow |
| 4 | RAG Integration (Hybrid Retriver + reranking) | Phase 2, 3 | Done |
| 5 | Multi-hop Traversal (GraphRAG-lite) | Phase 4 | Done |
| 6 | Consolidation/pruning (opt-in flag, never default-on) | Phase 4 | Done |
| 7 | Formal packaging (pyproject.toml, rename to temporal_memory, stable public API) | All internals stable | Do last, not first |
| 8 | pytest-temporal-memory plugin + backend adapter interface | Phase 5 | The actual differentiator — build once core engine is proven |
| 9 | MCP adapter (thin stdio server) | Phase 7 | Last — demo-reach layer, not the pitch |

Why this order, not the original one: benchmark moves early (right after the
resolver exists) so bugs get caught before three more layers are built on top of
them. Packaging moves to last because the public API will keep shifting while
you're still building — locking it early just means re-locking it later.

## 8. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| DB | SQLite3, WAL mode, PRAGMA user_version migrations | Single file, no server, no Alembic/SQLAlchemy needed |
| Extraction | Groq/OpenAI/Ollama LLM call, Pydantic-validated, Tenacity retry | Correct-by-default; GLiNER-Relex optional "fast mode" only, documented as lower-recall on implicit facts |
| Retrieval | TF-IDF or sentence-transformers + rank_bm25 + entity overlap, LLM rerank on top-N pool only | Fusion beats any single signal alone; reranking the whole corpus doesn't scale |
| Multi-hop | SQLite WITH RECURSIVE | Kuzu (the obvious embedded-graph-DB pick) was acquired by Apple and archived Oct 2025 — don't bet on unmaintained infra |
| Concurrency | queue.Queue + threading.Thread | Matches sync sqlite3 calls; asyncio.Queue would block the event loop on every DB call |
| Testing | Custom benchmark → ported to pytest → CI gate on every PR | Turns claims into evidence |

## 9. Known Limitations & Risks (say these out loud, don't hide them)

- History growth is bounded by opt-in `engine.prune(older_than_days=N)` which
archives dead facts to `facts_archive` and hard-deletes from the hot table in
batches of 500. Developers must call this explicitly (e.g. in a CRON job) —
the engine will never silently destroy audit history.
- GLiNER-Relex (if you ever add the "fast mode") misses facts that require
implicit world knowledge — document this trade-off, don't market it as a free
speed upgrade.
- The embedded-graph-DB space is currently unstable (Kuzu's fate) — recursive
CTEs are the safe choice for now; revisit only if a real workload needs 4+ hop
traversal and the landscape has settled.
- This will not reach numpy/FastAPI-scale adoption. State this to yourself and
to anyone asking, so the actual achievable goal — real niche usage plus a
benchmark that proves the mechanism works — stays the target instead of a bar
that isn't reachable through engineering alone.

## 10. What "Done" Looks Like

- `test_runner.py` (or similar) runs cleanly, shows correct UPDATE/ADD/DUPLICATE decisions, and proves owner isolation between two users.
- A 15-20 case benchmark scores this engine against a naive-RAG baseline with real numbers, not assertions.
- `pytest-temporal-memory` runs against at least two backends (this engine + one other, even a toy one) via the adapter interface, proving it's not hard-coded to only work with itself.
- A README that leads with the precise, honest pitch from Section 4 — not an overclaim that falls apart on the first hard question.
