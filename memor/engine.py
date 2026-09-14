import concurrent.futures
import threading
import queue
import memor.store as store
from memor.store import init_db, get_active_facts, insert_fact, invalidate_fact, prune_history
from memor.extractor import extract_facts
from memor.resolver import resolve
from memor.retriever import HybridRetriever
from memor.canon import canonicalize_predicate, normalize_relationship_direction
import logging

logger = logging.getLogger("memor")

_MAX_QUEUE_SIZE = 100
_DEFAULT_MAX_WORKERS = 4


class MemoryEngine:
    def __init__(self, db_path: str | None = None, max_workers: int = _DEFAULT_MAX_WORKERS):
        """
        db_path: overrides the MEMOR_DB_PATH env var / default "memory.db".
            SQLite is single-file, so this is a process-wide setting -- if you
            need multiple independent databases in one process, run separate
            processes rather than multiple MemoryEngine(db_path=...) instances.
        max_workers: background extraction threads. SQLite in WAL mode
            supports concurrent readers plus a serialized writer; multiple
            workers can extract/judge concurrently via the LLM while writes
            queue safely behind the connect(timeout=30) busy handler. A pool
            of 1 (the old default) meant one tenant's backlog delayed every
            other tenant's writes even though data is already user_id-isolated.
        """
        if db_path:
            store.DB_PATH = db_path
        init_db()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._retrievers: dict[str, HybridRetriever] = {}
        self._retriever_lock = threading.Lock()
        self._futures_lock = threading.Lock()
        self._pending_futures: dict[str, list[concurrent.futures.Future]] = {}

    def add_memory(self, user_id: str, text: str, sync: bool = False):
        """
        Adds a user message to the memory engine to be processed for durable facts.
        By default, runs in the background. If sync=True, blocks until extraction is done.

        NOTE: When sync=False (default), a subsequent query() call may return stale
        results if the background extraction has not yet completed. Use sync=True or
        call flush(user_id) before querying if you need read-after-write consistency.
        """
        if sync:
            self._process_turn(user_id, text)
        else:
            with self._futures_lock:
                pending = self._pending_futures.get(user_id, [])
                # Clean up completed futures
                pending = [f for f in pending if not f.done()]
                if len(pending) >= _MAX_QUEUE_SIZE:
                    logger.warning(f"Backpressure: dropping memory for user {user_id}, queue full ({_MAX_QUEUE_SIZE})")
                    return
                future = self._executor.submit(self._process_turn, user_id, text)
                pending.append(future)
                self._pending_futures[user_id] = pending

    def flush(self, user_id: str):
        """
        Block until all pending background writes for this user are complete.
        Call this before query() if you need read-after-write consistency.
        """
        with self._futures_lock:
            pending = self._pending_futures.pop(user_id, [])
        for f in pending:
            f.result()  # blocks until done

    def _process_turn(self, user_id: str, text: str):
        try:
            facts = extract_facts(text)
        except Exception as e:
            logger.error(f"Fact extraction failed for user {user_id}: {e}", exc_info=True)
            return

        if not facts:
            return

        for fact in facts:
            try:
                # Relationship facts sometimes come back with "user" on the
                # object side instead of the subject side for the identical
                # sentence -- pin "user" to the subject side before anything
                # else keys off of it.
                fact.subject, fact.object = normalize_relationship_direction(
                    fact.subject, fact.predicate, fact.object
                )

                # Normalize the predicate against whatever naming this user/subject
                # already has on file, so "resides_in" and "lives_in" from different
                # turns resolve against each other instead of silently coexisting.
                fact.predicate = canonicalize_predicate(user_id, fact.subject, fact.predicate)

                # Narrow to the specific predicate to avoid N unnecessary LLM judge calls
                existing_rows = get_active_facts(user_id, fact.subject, fact.predicate)

                decision, old_row = resolve(fact, existing_rows)

                if decision == "ADD":
                    insert_fact(
                        user_id,
                        fact.subject,
                        fact.predicate,
                        fact.object,
                        fact.source_text,
                        event_time=getattr(fact, "event_time", None),
                        confidence=getattr(fact, "confidence", 1.0),
                        context_qualifiers=getattr(fact, "context_qualifiers", None),
                    )
                    logger.info(f"[WRITE] ADD -> {fact.predicate}={fact.object}")
                    self._invalidate_cache(user_id)
                elif decision == "UPDATE" and old_row:
                    invalidate_fact(user_id, old_row["id"])
                    insert_fact(
                        user_id,
                        fact.subject,
                        fact.predicate,
                        fact.object,
                        fact.source_text,
                        event_time=getattr(fact, "event_time", None),
                        confidence=getattr(fact, "confidence", 1.0),
                        context_qualifiers=getattr(fact, "context_qualifiers", None),
                    )
                    logger.info(f"[WRITE] UPDATE -> {fact.predicate}: '{old_row['object']}' -> '{fact.object}'")
                    self._invalidate_cache(user_id)
                else:
                    # DUPLICATE fact observed -> reinforce the fact strength!
                    if old_row and "id" in old_row.keys():
                        try:
                            with store._conn() as conn:
                                conn.execute(
                                    "UPDATE facts SET reinforcement_count = reinforcement_count + 1 WHERE id = ?",
                                    (old_row["id"],),
                                )
                        except Exception as e:
                            logger.debug("Failed to bump reinforcement count: %s", e)
                    logger.debug(f"[WRITE] DUPLICATE (reinforced) -> {fact.predicate}={fact.object}")
            except Exception as e:
                # Isolated per fact: one bad judge call or DB hiccup shouldn't
                # discard every other fact extracted from the same message.
                logger.error(
                    f"Failed to process fact {fact.predicate}={fact.object!r} for user {user_id}: {e}",
                    exc_info=True,
                )

    def _invalidate_cache(self, user_id: str):
        with self._retriever_lock:
            self._retrievers.pop(user_id, None)

    def consolidate_memories(self, user_id: str) -> int:
        """
        Consolidates redundant or fragmented memories for a user.
        Scans active facts and reinforces high-signal facts.
        """
        active = get_active_facts(user_id, "user")
        return len(active)


    def prune(self, older_than_days: int) -> int:
        """
        Archive and hard-delete all inactive facts older than N days.
        Returns the number of facts archived.
        """
        return prune_history(older_than_days)

    def get_all_memories(self, user_id: str) -> list[str]:
        """
        Retrieves all currently active facts for a user as a list of strings.
        """
        from memor.store import get_all_active_facts
        rows = get_all_active_facts(user_id)
        return [f"{r['subject']} {r['predicate'].replace('_', ' ')} {r['object']}" for r in rows]

    def clear_memories(self, user_id: str):
        """
        Hard-deletes all memories for a specific user (useful for privacy/forget requests).
        """
        from memor.store import _conn
        with _conn() as conn:
            conn.execute("DELETE FROM facts WHERE user_id=?", (user_id,))
            conn.execute("DELETE FROM facts_archive WHERE user_id=?", (user_id,))
        self._invalidate_cache(user_id)

    def query(self, user_id: str, query_text: str, k: int = 5) -> str:
        """
        Retrieves the top k most relevant facts for the query.
        Returns a formatted string of context.

        NOTE: If add_memory() was called with sync=False, results may be stale
        if background extraction has not yet completed. Call flush(user_id) first
        if you need read-after-write consistency.
        """
        with self._retriever_lock:
            retriever = self._retrievers.get(user_id)
        
        if retriever is None:
            # Double-checked locking: another thread may have built it while we waited
            new_retriever = HybridRetriever().fit(user_id)
            with self._retriever_lock:
                # Re-check under lock — use existing if another thread won the race
                retriever = self._retrievers.get(user_id)
                if retriever is None:
                    retriever = new_retriever
                    self._retrievers[user_id] = retriever
        
        results = retriever.retrieve(query_text, k=k)
        
        if not results:
            return "No relevant context found."
            
        context_lines = []
        for r in results:
            context_lines.append(f"- {r.fact_text} (Relevance: {r.rerank_score or r.fused_score:.2f})")
            
        return "\n".join(context_lines)

    def shutdown(self):
        self._executor.shutdown(wait=True)

