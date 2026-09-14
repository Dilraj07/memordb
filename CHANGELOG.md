# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-09-09
### Added
- Disk-backed SQLite FTS5 virtual table (`facts_fts`) synchronized with `facts` via triggers for zero-RAM BM25 searches in ~4.4ms.
- Two-stage candidate pre-filtering pipeline (FTS5 + entity seed + seeded multihop) bounding dense vector scoring to $M \le 50$.
- Query-seeded multi-hop graph traversal with cycle prevention (`INSTR` boundary tracking) stopping infinite recursive loops on cyclic knowledge graphs.
- Schema V6: Added `event_time`, `confidence`, `reinforcement_count`, `access_count`, `last_accessed_at`, and `context_qualifiers`.
- Duplicate fact reinforcement: repeated facts increment `reinforcement_count`, strengthening memory retrieval weights over time.
- New scale benchmark (`benchmark/test_scale.py`) testing 1,000+ facts ingestion, FTS5 latency, and circular graph stress.

### Improved
- Disambiguated musical instrument extraction (`plays_instrument` vs `hobby`) in extraction prompt.
- Added device replacement and singular preference guidance to conflict resolver judge prompt.
- Provider-agnostic LLM error handling (`TransientLLMError`, `PermanentLLMError`) and dynamic Groq model configuration.

## [0.1.3] - 2026-07-20
### Improved
- Batched LLM reranker: 1 API call instead of 8 per query (~87.5% fewer calls, faster retrieval).
- Compressed extraction and eval prompts for lower token usage.
- Added `max_tokens=256` to all LLM calls to prevent wasted output budget.
- Added composite DB index on `(user_id, subject, predicate, valid_to)` for faster fact lookups.
- Added explicit `conn.rollback()` on database write failures.
- Fixed retriever cache race condition with double-checked locking.

### Added
- Token usage logging at `DEBUG` level via `memor` logger.
- `memor.__version__` exposed for standard Python packaging.
- Multi-hop graph traversal section added to demo.
- Community profile: LICENSE (MIT), CODE_OF_CONDUCT, SECURITY.md, issue & PR templates.

## [0.1.2] - 2026-07-15
### Fixed
- Fixed broken logo image rendering on PyPI by using absolute raw GitHub URLs in `README.md`.

## [0.1.1] - 2026-07-15
### Added
- Added `get_all_memories` and `clear_memories` to `MemoryEngine` for full profile dumps and GDPR-compliant hard deletes.
- Integrated `get_user_profile` and `forget_memory` tools into the Model Context Protocol (MCP) server for autonomous agents.

## [0.1.0] - 2026-07-15
### Added
- Initial release of `memor-db`.
- Local-first bitemporal SQLite memory storage.
- LLM-powered contradiction resolution and dual-path updates.
- Hybrid Retrieval Pipeline (BM25 + Semantic + Entity).
- Multi-hop graph traversal via SQLite CTEs.
- `pytest-temporal-memory` benchmarking plugin.
- Standard MCP server adapter.
