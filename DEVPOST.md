# Memor: Local-First Bitemporal Memory Engine for LLM Agents

## Inspiration

When conversational AI agents interact with humans across multiple sessions, days, or months, they inevitably run into a critical failure mode: **temporal drift**.

Consider a user who tells an agent in January:
> *"I live in San Francisco and work as a kernel developer."*

Six months later, in July, they say:
> *"I just relocated to Seattle for a new role."*

In conventional Retrieval-Augmented Generation (RAG) architectures, dialogue turns are chunked, transformed into dense embeddings, and indexed into a standard vector database. When the user later asks, *"Where do I live?"*, vector similarity retrieves **both** chunks because they share nearly identical semantic proximity to the query. 

As a result, the model is handed two diametrically opposed statements scrambled out of chronological order. Research into temporal reasoning demonstrates that frontier LLMs degrade sharply when chronological signals are disrupted. **This is an information-access defect, not a reasoning failure.** Increasing model intelligence or expanding context windows cannot fix what the retrieval layer scrambles before prompt construction even begins.

We realized that this problem had already been solved 40 years ago in financial accounting, insurance records, and medical informatics through **bitemporal database modeling**. Rather than relying on naive, atemporal vector chunks, we asked: 
*What if an agent's memory tracked world validity time and transaction recording time natively, executing graph traversals and contradiction resolution entirely inside a local, embedded SQLite database?*

That question inspired **Memor**.

---

## What it does

**Memor** is an embedded, local-first bitemporal memory and retrieval engine for autonomous AI agents and conversational assistants.

* **Zero-Infrastructure Single-File Storage**: Operates entirely within a local SQLite database (`memory.db`) in Write-Ahead Logging (WAL) mode. No external vector clusters, daemon processes, or cloud databases are required.
* **Bitemporal Fact Audit Log**: Deconstructs raw conversational dialogue into atomic Subject-Predicate-Object ($S, P, O$) triples, recording both world validity intervals and system transaction timestamps. Current truths are retrieved cleanly with `valid_to IS NULL`, while superseded facts are retained for complete historical auditability.
* **Dual-Path Contradiction Resolution**:
  * *Fast Path (Deterministic Heuristic)*: Instantly invalidates outdated rows for single-valued predicates (e.g., `lives_in`, `works_at`, `current_phone`) with zero LLM inference cost.
  * *Slow Path (LLM Judge)*: Routes complex or ambiguous predicates (e.g., dietary restrictions, hobbies) through a structured LLM judge to determine whether new facts replace or append to existing knowledge.
* **Two-Stage Hybrid Retrieval**: Combines disk-backed SQLite FTS5 BM25 lexical search ($\approx 4.4\,\text{ms}$), dense vector similarity (`sentence-transformers`), and entity overlap scoring, bounded by candidate pre-filtering.
* **Recursive Multi-Hop Graph Traversal**: Traverses associative relationship chains (e.g., $\text{user} \rightarrow \text{brother} \rightarrow \text{Alex} \rightarrow \text{adopted} \rightarrow \text{golden retriever}$) natively via SQLite `WITH RECURSIVE` Common Table Expressions (CTEs) without requiring an external graph database.
* **Model Context Protocol (MCP) Server**: Provides a native `stdio` MCP interface (`memor-server`) allowing autonomous agents in environments like Claude Desktop and Cursor to interface directly with Memor.
* **Drop-in Pytest Benchmarking Plugin (`pytest-temporal-memory`)**: Ships with a registered `pytest11` plugin that allows developers to benchmark any custom memory backend against planted-contradiction suites via an abstract adapter interface.

---

## How we built it

Memor was built from the ground up in Python 3.10+ using a modular, dependency-minimal architecture:

```
                  ┌──────────────────────┐
 dialogue turn ──►│   ExtractionWorker   │  (Non-blocking thread + bounded queue)
                  └──────────┬───────────┘
                             ▼
                  ┌──────────────────────┐
                  │    Fact Extractor    │  (Structured Pydantic JSON + Tenacity retries)
                  └──────────┬───────────┘
                             ▼
                  ┌──────────────────────┐
                  │  Conflict Resolver   │  (Fast path: known single-valued predicates;
                  │  ADD / UPDATE / DUP  │   Slow path: LLM Judge)
                  └──────────┬───────────┘
                             ▼
                  ┌──────────────────────┐
                  │   Bitemporal Store   │  (SQLite WAL, PRAGMA user_version,
                  │  memory.db (WAL)     │   composite index: user, subj, pred, valid_to)
                  └──────────┬───────────┘
                             ▼
                  ┌──────────────────────┐
 query text ─────►│   Hybrid Retriever   │  (FTS5 BM25 + dense vectors + graph CTE
                  │  & Batched Reranker  │   fused into Top-K context)
                  └──────────────────────┘
```

### 1. Bitemporal Mathematical Formulation

Every extracted fact $f$ is stored as a tuple:
$$f = \big( \text{id}, \text{user\_id}, s, p, o, t_{\text{valid\_from}}, t_{\text{valid\_to}}, t_{\text{recorded\_at}} \big)$$

Where:
* $[t_{\text{valid\_from}}, t_{\text{valid\_to}})$ defines the **valid time** interval during which the proposition is true in the physical world.
* $t_{\text{recorded\_at}}$ is the **system transaction time** indicating when the fact was ingested.

A fact is considered **active** at query time $t_q$ if and only if:
$$t_{\text{valid\_from}} \le t_q < t_{\text{valid\_to}} \quad \text{where } t_{\text{valid\_to}} = \infty \ (\text{or } \texttt{NULL})$$

When an update occurs that supersedes fact $f_{\text{old}}$ with new fact $f_{\text{new}}$ at time $t_{\text{event}}$:
$$\text{UPDATE } f_{\text{old}} \implies t_{\text{valid\_to}} \leftarrow t_{\text{event}}$$
$$\text{INSERT } f_{\text{new}} \implies t_{\text{valid\_from}} \leftarrow t_{\text{event}}, \quad t_{\text{valid\_to}} \leftarrow \texttt{NULL}$$

No history is destructively overwritten.

### 2. High-Precision Hybrid Scoring & Candidate Bounding

To eliminate the latency of computing dense vector similarities across an unbounded corpus, Memor implements a two-stage candidate retrieval pipeline bounding the candidate pool to $M \le 50$ before vector scoring:

1. **Entity Overlap Score**:
   $$\text{Score}_{\text{entity}}(q, f) = \frac{|\mathcal{T}(q) \cap (\mathcal{T}(f_s) \cup \mathcal{T}(f_o))|}{|\mathcal{T}(f_s) \cup \mathcal{T}(f_o)|}$$
   where $\mathcal{T}(x)$ represents the set of alphanumeric tokens in string $x$.

2. **Min-Max Score Normalization**:
   For any raw score metric $s_i \in S$:
   $$\hat{s}_i = \begin{cases} 
   \frac{s_i - \min(S)}{\max(S) - \min(S)}, & \text{if } \max(S) > \min(S) \\ 
   0.0, & \text{otherwise} 
   \end{cases}$$

3. **Reinforcement Boost**:
   Facts repeatedly mentioned by the user increment a `reinforcement_count` ($r$), applying a sub-linear confidence multiplier:
   $$\beta(r) = 1.0 + 0.05 \times \min(r - 1, 5)$$

4. **Fused Candidate Ranking**:
   $$S_{\text{fused}} = \Big( w_{\text{vector}} \cdot \hat{s}_{\text{vector}} + w_{\text{bm25}} \cdot \hat{s}_{\text{bm25}} + w_{\text{entity}} \cdot \text{Score}_{\text{entity}} \Big) \times \beta(r)$$
   *(Default tuned weights: $w_{\text{vector}} = 0.4$, $w_{\text{bm25}} = 0.4$, $w_{\text{entity}} = 0.2$)*.

### 3. Cycle-Safe Multi-Hop Graph Traversal

Multi-hop relations are computed directly within SQLite using recursive common table expressions. To prevent infinite recursion on cyclical graphs, we track visited nodes through a path-accumulator string with SQL boundary delimiters:

```sql
WITH RECURSIVE entity_graph(entity, depth, path) AS (
    SELECT :seed_entity, 0, '/' || :seed_entity || '/'
    UNION ALL
    SELECT 
        CASE WHEN f.subject = eg.entity THEN f.object ELSE f.subject END,
        eg.depth + 1,
        eg.path || CASE WHEN f.subject = eg.entity THEN f.object ELSE f.subject END || '/'
    FROM facts f
    JOIN entity_graph eg ON (f.subject = eg.entity OR f.object = eg.entity)
    WHERE f.user_id = :user_id 
      AND f.valid_to IS NULL
      AND eg.depth < :max_depth
      AND INSTR(eg.path, '/' || CASE WHEN f.subject = eg.entity THEN f.object ELSE f.subject END || '/') = 0
)
SELECT DISTINCT entity, depth FROM entity_graph;
```

---

## Challenges we ran into

1. **Concurrency and SQLite Thread Safety**:
   Early prototypes attempted to run synchronous SQLite operations inside Python's asynchronous `asyncio` event loop. Under continuous chat turn ingestion, this caused event loop blocking and `sqlite3.OperationalError: database is locked`. We resolved this by redesigning the ingestion pipeline around dedicated OS threads (`threading.Thread`) combined with bounded thread-safe queues (`queue.Queue`), WAL mode, and double-checked locking for embedding cache access.

2. **Recursive Traversal Loops**:
   In complex conversation graphs with reciprocal associations (e.g., *"Alice is Bob's colleague"* and *"Bob works with Alice"*), early recursive queries entered infinite loops, crashing the SQLite engine. We overcame this by implementing path-history tracking via `INSTR(path, '/' || id || '/') = 0`, ensuring cyclic references terminate cleanly in $O(V + E)$ time.

3. **Disambiguating Additive vs. Mutually Exclusive Facts**:
   Distinguishing between updates that overwrite a state (e.g., *"I bought an Android phone"* replacing *"I use an iPhone"*) versus updates that are additive (e.g., *"I also play the cello"* supplementing *"I play the violin"*) was difficult for naive prompts. We addressed this by implementing strict predicate canonicalization (`plays_instrument` vs. `general_hobby`) and engineering explicit few-shot domain rules into the conflict resolution judge.

4. **Reranking Overhead**:
   Individual LLM calls to score each candidate fact produced prohibitive latency and quickly triggered API rate limits. We refactored the reranker into a single-pass batched judge that scores all top candidates in one structured JSON output, reducing API calls by $87.5\%$.

---

## Accomplishments that we're proud of

* **100% Contradiction Resolution**: In our 12-case benchmark suite comparing Memor against a Naive-RAG baseline, Memor achieved a **100% pass rate** across all location updates, job changes, diet shifts, and multi-hop contradictions, cleanly eliminating stale facts.
* **Single-File Zero-Dependency Architecture**: Packaged an entire temporal graph retrieval engine, FTS5 full-text index, and vector cache into a single `.db` file without requiring external vector database engines or cluster setups.
* **Sub-5ms FTS5 Hybrid Search**: Designed SQLite trigger-synchronized FTS5 indexing with query-seeded multi-hop graph traversal that runs in single-digit milliseconds.
* **Standardized Testing Tool**: Published `pytest-temporal-memory` as a registered `pytest11` entry point, giving the broader developer community a standardized tool to test their own agent memory systems for fact drift.
* **Native MCP Support**: Provided full Model Context Protocol integration, allowing local agent environments like Claude Desktop to leverage persistent, drift-free memory with zero configuration.

---

## What we learned

* **Bitemporality is Fundamental to Agent Reliability**: Classical database theory from the 1980s provides a much cleaner solution to temporal drift than simply scaling model parameters. Preserving both event validity time and transaction time provides agents with an exact chronology of the world.
* **The Power of Modern SQLite**: Between FTS5 virtual tables, recursive CTEs, trigger-based index updates, and WAL mode, SQLite can match or exceed specialized vector and graph databases on local-first workloads while avoiding infrastructure bloat.
* **Decoupled Ingestion & Flush Synchronization**: Decoupling write ingestion into background queues while providing an explicit synchronous `flush()` barrier ensures zero latency on interactive chat loops while guaranteeing complete read-after-write consistency.

---

## What's next for Memor

* **Offline Small Language Model (SLM) Extraction**: Adding built-in adapters for local quantized models (e.g., Llama-3-8B or SmolLM via Ollama) to support completely offline, zero-cloud fact extraction and conflict resolution.
* **Ebbinghaus Forgetting Curves**: Implementing mathematical memory decay models based on the cognitive forgetting curve:
  $$R(t) = e^{-\frac{t}{S}}$$
  where memory retention $R$ decays over elapsed time $t$ relative to stability $S$, which increases dynamically with each reinforcement event.
* **Time-Travel Query API**: Exposing an explicit historical query endpoint:
  ```python
  engine.query_at_time(user_id, "Where did I live?", as_of="2026-02-01")
  ```
  allowing agents to inspect what was believed to be true at any specific point in history for auditing and counterfactual reasoning.
* **Transitive Closure Graph Optimization**: Pre-computing transitive closure tables for frequently traversed relationship paths to support 4+ hop graph queries at instant lookup speeds.
