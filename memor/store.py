"""
Bitemporal fact store. Every fact has:
  - valid_from / valid_to  -> when the fact was true in the real world
  - recorded_at            -> when *we* learned about it (lets you replay
                               "what did the system believe on date X")

This is the piece that separates a real memory engine from a vector store with
extra steps: a vector DB has no concept of "this fact used to be true."
"""
import os
import re
import json
import sqlite3
from datetime import datetime, timezone
from contextlib import contextmanager

# MEMOR_DB_PATH lets a deployment point the whole engine at a specific file
# (e.g. a data volume path) without code changes. MemoryEngine(db_path=...)
# overrides this at construction time by reassigning this module attribute --
# same pattern the test suite already uses (see tests/conftest.py).
DB_PATH = os.environ.get("MEMOR_DB_PATH", "memory.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    subject_norm TEXT,
    object_norm TEXT,
    valid_from TEXT NOT NULL,
    valid_to TEXT,              -- NULL means "still active"
    recorded_at TEXT NOT NULL,
    source_text TEXT,
    event_time TEXT,
    confidence REAL DEFAULT 1.0,
    reinforcement_count INTEGER DEFAULT 1,
    access_count INTEGER DEFAULT 0,
    last_accessed_at TEXT,
    context_qualifiers TEXT,
    embedding BLOB
);

CREATE TABLE IF NOT EXISTS facts_archive (
    id INTEGER PRIMARY KEY, -- keep original id
    user_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    valid_from TEXT NOT NULL,
    valid_to TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    source_text TEXT,
    archived_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    user_id UNINDEXED, subject, predicate, object, source_text
);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, user_id, subject, predicate, object, source_text)
    VALUES (new.id, new.user_id, new.subject, new.predicate, new.object, new.source_text);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    DELETE FROM facts_fts WHERE rowid = old.id;
END;
"""

@contextmanager
def _conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    # journal_mode=WAL is persisted in the database file header once set (see
    # init_db) -- re-issuing it on every connection is a no-op round trip we
    # don't need. synchronous is NOT persisted, so it does need setting per
    # connection: NORMAL is the standard, safe pairing with WAL (an app crash
    # can't corrupt the DB; only OS-level power loss could lose the most
    # recent commit) and avoids an fsync on every single write.
    conn.execute("PRAGMA synchronous=NORMAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with _conn() as conn:
        conn.execute("PRAGMA journal_mode=WAL")

        # Check current schema version
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        
        if version == 0:
            # First time setup or upgrading from old V0 (initial schema)
            # Create table if it doesn't exist
            conn.executescript(SCHEMA)
            
            # If the table already existed but was missing user_id (V0 schema), we need to add it
            # We can check if the column exists by PRAGMA table_info
            columns = [info["name"] for info in conn.execute("PRAGMA table_info(facts)").fetchall()]
            if columns and "user_id" not in columns:
                # Upgrade existing V0 table
                conn.execute("ALTER TABLE facts ADD COLUMN user_id TEXT DEFAULT 'default'")
            
            # Bump version to 1
            conn.execute("PRAGMA user_version = 1")
            version = 1
            
        if version == 1:
            # Upgrade to V2: Create facts_archive
            conn.execute("CREATE TABLE IF NOT EXISTS facts_archive ("
                         "id INTEGER PRIMARY KEY,"
                         "user_id TEXT NOT NULL,"
                         "subject TEXT NOT NULL,"
                         "predicate TEXT NOT NULL,"
                         "object TEXT NOT NULL,"
                         "valid_from TEXT NOT NULL,"
                         "valid_to TEXT NOT NULL,"
                         "recorded_at TEXT NOT NULL,"
                         "source_text TEXT,"
                         "archived_at TEXT NOT NULL)")
            conn.execute("PRAGMA user_version = 2")
            version = 2

        if version == 2:
            # Upgrade to V3: Add composite index for fast fact lookups
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_lookup "
                "ON facts(user_id, subject, predicate, valid_to)"
            )
            conn.execute("PRAGMA user_version = 3")
            version = 3

        if version == 3:
            # Upgrade to V4: pre-normalized subject/object columns + indexes so the
            # multi-hop WITH RECURSIVE join (store.get_multihop_paths) can index-seek
            # on equality instead of running LOWER(TRIM()) over every row on every hop.
            columns = [info["name"] for info in conn.execute("PRAGMA table_info(facts)").fetchall()]
            if "subject_norm" not in columns:
                conn.execute("ALTER TABLE facts ADD COLUMN subject_norm TEXT")
            if "object_norm" not in columns:
                conn.execute("ALTER TABLE facts ADD COLUMN object_norm TEXT")
            conn.execute(
                "UPDATE facts SET subject_norm = LOWER(TRIM(subject)), object_norm = LOWER(TRIM(object)) "
                "WHERE subject_norm IS NULL"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_subject_norm ON facts(user_id, subject_norm, valid_to)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_facts_object_norm ON facts(user_id, object_norm, valid_to)"
            )
            conn.execute("PRAGMA user_version = 4")
            version = 4

        if version == 4:
            # Upgrade to V5: cache the embedding vector alongside each fact so the
            # retriever only has to encode genuinely new facts on fit(), instead of
            # re-embedding the entire active corpus on every cache invalidation.
            columns = [info["name"] for info in conn.execute("PRAGMA table_info(facts)").fetchall()]
            if "embedding" not in columns:
                conn.execute("ALTER TABLE facts ADD COLUMN embedding BLOB")
            conn.execute("PRAGMA user_version = 5")
            version = 5

        if version == 5:
            # Upgrade to V6: Real-world complex memory + Disk-backed FTS5 search
            columns = [info["name"] for info in conn.execute("PRAGMA table_info(facts)").fetchall()]
            if "event_time" not in columns:
                conn.execute("ALTER TABLE facts ADD COLUMN event_time TEXT")
            if "confidence" not in columns:
                conn.execute("ALTER TABLE facts ADD COLUMN confidence REAL DEFAULT 1.0")
            if "reinforcement_count" not in columns:
                conn.execute("ALTER TABLE facts ADD COLUMN reinforcement_count INTEGER DEFAULT 1")
            if "access_count" not in columns:
                conn.execute("ALTER TABLE facts ADD COLUMN access_count INTEGER DEFAULT 0")
            if "last_accessed_at" not in columns:
                conn.execute("ALTER TABLE facts ADD COLUMN last_accessed_at TEXT")
            if "context_qualifiers" not in columns:
                conn.execute("ALTER TABLE facts ADD COLUMN context_qualifiers TEXT")

            # Create FTS5 virtual table
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5("
                "user_id UNINDEXED, subject, predicate, object, source_text"
                ")"
            )
            # Create triggers
            conn.execute(
                "CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN "
                "INSERT INTO facts_fts(rowid, user_id, subject, predicate, object, source_text) "
                "VALUES (new.id, new.user_id, new.subject, new.predicate, new.object, new.source_text); "
                "END;"
            )
            conn.execute(
                "CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN "
                "DELETE FROM facts_fts WHERE rowid = old.id; "
                "END;"
            )
            # Backfill existing facts into facts_fts if any
            conn.execute(
                "INSERT OR IGNORE INTO facts_fts(rowid, user_id, subject, predicate, object, source_text) "
                "SELECT id, user_id, subject, predicate, object, source_text FROM facts "
                "WHERE id NOT IN (SELECT rowid FROM facts_fts)"
            )
            conn.execute("PRAGMA user_version = 6")
            version = 6



def get_active_facts(user_id: str, subject: str, predicate: str | None = None) -> list[sqlite3.Row]:
    with _conn() as conn:
        if predicate:
            return conn.execute(
                "SELECT * FROM facts WHERE user_id=? AND subject=? AND predicate=? AND valid_to IS NULL",
                (user_id, subject, predicate),
            ).fetchall()
        return conn.execute(
            "SELECT * FROM facts WHERE user_id=? AND subject=? AND valid_to IS NULL", (user_id, subject)
        ).fetchall()


def insert_fact(
    user_id: str,
    subject: str,
    predicate: str,
    obj: str,
    source_text: str = "",
    event_time: str | None = None,
    confidence: float = 1.0,
    context_qualifiers: dict | str | None = None,
):
    now = datetime.now(timezone.utc).isoformat()
    cq_json = json.dumps(context_qualifiers) if isinstance(context_qualifiers, dict) else context_qualifiers
    with _conn() as conn:
        conn.execute(
            "INSERT INTO facts (user_id, subject, predicate, object, subject_norm, object_norm, "
            "valid_from, valid_to, recorded_at, source_text, event_time, confidence, "
            "reinforcement_count, access_count, last_accessed_at, context_qualifiers) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, 1, 0, ?, ?)",
            (
                user_id,
                subject,
                predicate,
                obj,
                subject.strip().lower(),
                obj.strip().lower(),
                now,
                now,
                source_text,
                event_time,
                confidence,
                now,
                cq_json,
            ),
        )


def search_facts_fts(user_id: str, query: str, limit: int = 50) -> list[sqlite3.Row]:
    """Native SQLite FTS5 search. Queries the disk-backed inverted index,
    scoped to active facts for user_id, ordered by BM25 relevance score."""
    tokens = re.findall(r"[a-zA-Z0-9]+", query.lower())
    if not tokens:
        return []

    # Prefix match tokens with OR to retrieve broad candidate pool
    fts_query = " OR ".join(f'"{t}"*' for t in tokens)
    with _conn() as conn:
        try:
            return conn.execute(
                """
                SELECT f.*, bm25(facts_fts) AS bm25_score
                FROM facts_fts
                JOIN facts f ON f.id = facts_fts.rowid
                WHERE facts_fts MATCH ? AND f.user_id = ? AND f.valid_to IS NULL
                ORDER BY bm25_score ASC
                LIMIT ?
                """,
                (fts_query, user_id, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []


def touch_fact_access(user_id: str, fact_ids: list[int]):
    """Update access metrics for retrieved facts to support recency and decay."""
    if not fact_ids:
        return
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        placeholders = ",".join("?" for _ in fact_ids)
        conn.execute(
            f"UPDATE facts SET access_count = access_count + 1, last_accessed_at = ? "
            f"WHERE user_id = ? AND id IN ({placeholders})",
            [now, user_id] + fact_ids,
        )


def get_query_seeded_multihop_paths(user_id: str, seed_entities: list[str], max_depth: int = 2) -> list[str]:
    """Query-anchored graph traversal with cycle prevention.
    Instead of calculating the global Cartesian product of all paths in the graph,
    we only traverse outgoing edges from entities present in the query."""
    if not seed_entities:
        return []

    seeds = list({s.strip().lower() for s in seed_entities if s.strip()})
    if not seeds:
        return []

    placeholders = ",".join("?" for _ in seeds)
    query = f"""
    WITH RECURSIVE multihop(subject, object, object_norm, path, visited, depth) AS (
        -- Base case: Active facts touching seed entities
        SELECT subject, object, object_norm,
               subject || ' ' || REPLACE(predicate, '_', ' ') || ' ' || object,
               '/' || subject_norm || '/' || object_norm || '/',
               1
        FROM facts
        WHERE user_id = ? AND valid_to IS NULL
          AND (subject_norm IN ({placeholders}) OR object_norm IN ({placeholders}))

        UNION ALL

        -- Recursive step: traverse matching outgoing hops, with cycle prevention
        SELECT f.subject, f.object, f.object_norm,
               m.path || ' -> ' || f.subject || ' ' || REPLACE(f.predicate, '_', ' ') || ' ' || f.object,
               m.visited || f.object_norm || '/',
               m.depth + 1
        FROM facts f
        JOIN multihop m ON m.object_norm = f.subject_norm
        WHERE f.user_id = ? AND f.valid_to IS NULL
          AND m.depth < ?
          AND INSTR(m.visited, '/' || f.object_norm || '/') = 0
    )
    SELECT DISTINCT path FROM multihop WHERE depth > 1;
    """
    params = [user_id] + seeds + seeds + [user_id, max_depth]
    with _conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [r["path"] for r in rows]



def update_fact_embeddings(updates: list[tuple[int, bytes]]):
    """Persist newly computed embedding vectors for facts that didn't have one
    cached yet, so the next retriever fit() doesn't need to re-encode them."""
    if not updates:
        return
    with _conn() as conn:
        conn.executemany("UPDATE facts SET embedding=? WHERE id=?", [(blob, fid) for fid, blob in updates])


def invalidate_fact(user_id: str, fact_id: int):
    """Set valid_to on a fact, but only if it belongs to the given user (defense-in-depth)."""
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        conn.execute("UPDATE facts SET valid_to=? WHERE id=? AND user_id=?", (now, fact_id, user_id))


def get_distinct_predicates(user_id: str, subject: str) -> list[str]:
    """Predicate names already in use for this (user_id, subject), active facts
    only. Used by memor.canon to canonicalize newly extracted predicates against
    whatever naming is already on file, instead of matching by exact string."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT predicate FROM facts WHERE user_id=? AND subject=? AND valid_to IS NULL",
            (user_id, subject),
        ).fetchall()
        return [r["predicate"] for r in rows]


def get_all_active_facts(user_id: str) -> list[sqlite3.Row]:
    """Corpus for the retriever -- every currently-valid fact, scoped to a specific user."""
    with _conn() as conn:
        return conn.execute("SELECT * FROM facts WHERE user_id=? AND valid_to IS NULL", (user_id,)).fetchall()


def all_facts_including_history(user_id: str, subject: str) -> list[sqlite3.Row]:
    """Useful for the 'what did we used to believe' queries / debugging the resolver."""
    with _conn() as conn:
        return conn.execute(
            "SELECT * FROM facts WHERE user_id=? AND subject=? ORDER BY recorded_at", (user_id, subject)
        ).fetchall()


def get_multihop_paths(user_id: str, max_depth: int = 2) -> list[str]:
    """
    Graph traversal using SQLite WITH RECURSIVE.
    Returns concatenated text paths for all connected facts up to max_depth.
    """
    query = """
    WITH RECURSIVE multihop(subject, object, object_norm, path, depth) AS (
        -- Base case: All active facts
        SELECT subject, object, object_norm,
               subject || ' ' || REPLACE(predicate, '_', ' ') || ' ' || object, 1
        FROM facts
        WHERE user_id = ? AND valid_to IS NULL

        UNION

        -- Recursive step: find facts whose subject matches the previous object.
        -- Joins on the pre-normalized, indexed subject_norm column (idx_facts_subject_norm)
        -- instead of computing LOWER(TRIM()) over every row on every hop.
        SELECT f.subject, f.object, f.object_norm,
               m.path || ' -> ' || f.subject || ' ' || REPLACE(f.predicate, '_', ' ') || ' ' || f.object, m.depth + 1
        FROM facts f
        JOIN multihop m ON m.object_norm = f.subject_norm
        WHERE f.user_id = ? AND f.valid_to IS NULL AND m.depth < ?
    )
    -- We only return paths of length > 1, because length 1 facts are already in the base corpus.
    SELECT path FROM multihop WHERE depth > 1;
    """
    
    args = [user_id, user_id, max_depth]
    
    with _conn() as conn:
        rows = conn.execute(query, tuple(args)).fetchall()
        return [r["path"] for r in rows]

def prune_history(older_than_days: int) -> int:
    """
    Archive and hard-delete inactive facts older than N days.
    Operates in batches of 500 to prevent locking the database.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if older_than_days < 0:
        raise ValueError("older_than_days must be >= 0")
        
    batch_size = 500
    total_archived = 0
    now = datetime.now(timezone.utc)
    
    # Calculate cutoff in Python so it matches the isoformat exactly
    from datetime import timedelta
    cutoff = (now - timedelta(days=older_than_days)).isoformat()
    now_iso = now.isoformat()
    
    # We use a separate connection loop because we want to commit between batches
    # so we don't hold the write lock during the entire prune.
    while True:
        with _conn() as conn:
            # 1. Identify a batch of stale rows
            select_query = f"""
                SELECT id, user_id, subject, predicate, object, valid_from, valid_to, recorded_at, source_text
                FROM facts
                WHERE valid_to IS NOT NULL AND valid_to < ?
                LIMIT {batch_size}
            """
            stale_rows = conn.execute(select_query, (cutoff,)).fetchall()
            
            if not stale_rows:
                break
                
            # 2. Archive them
            archive_query = """
                INSERT INTO facts_archive 
                (id, user_id, subject, predicate, object, valid_from, valid_to, recorded_at, source_text, archived_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            archive_data = [
                (r["id"], r["user_id"], r["subject"], r["predicate"], r["object"], 
                 r["valid_from"], r["valid_to"], r["recorded_at"], r["source_text"], now_iso)
                for r in stale_rows
            ]
            conn.executemany(archive_query, archive_data)
            
            # 3. Delete them from the hot table
            ids_to_delete = [(r["id"],) for r in stale_rows]
            conn.executemany("DELETE FROM facts WHERE id = ?", ids_to_delete)
            
            total_archived += len(stale_rows)
            logger.info(f"Archived {len(stale_rows)} stale facts...")
            
    if total_archived > 0:
        logger.info(f"Prune complete. Total facts archived: {total_archived}")
    
    return total_archived


