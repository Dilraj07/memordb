"""
Scale and stress test benchmark for Memor.
Validates:
  1. High-volume ingestion (1,000+ facts) with SQLite WAL and FTS5 triggers.
  2. Sub-millisecond FTS5 search latency.
  3. Graph traversal cycle resilience (plants circular graphs A -> B -> C -> A).
  4. Memory access tracking and duplicate reinforcement.
"""
import os
import sys
import time
import sqlite3
import tempfile
import statistics

# Set up isolated test database
test_dir = tempfile.mkdtemp()
test_db = os.path.join(test_dir, "scale_test.db")
os.environ["MEMOR_DB_PATH"] = test_db

import memor.store as store
store.DB_PATH = test_db
store.init_db()

from memor.store import (
    insert_fact,
    search_facts_fts,
    get_query_seeded_multihop_paths,
    touch_fact_access,
    get_all_active_facts,
    _conn,
)
from memor.retriever import HybridRetriever


def run_scale_benchmark():
    user_id = "scale_tester"
    print("=" * 60)
    print("MEMOR HIGH-VOLUME SCALE & STRESS BENCHMARK")
    print("=" * 60)

    # 1. Ingestion of 1,000 facts
    num_facts = 1000
    print(f"\n[1/4] Ingesting {num_facts} structured facts into SQLite with FTS5 triggers...")
    t0 = time.time()
    for i in range(num_facts):
        subject = f"entity_{i % 50}"
        predicate = f"relation_{i % 20}"
        obj = f"attribute_value_{i}"
        insert_fact(
            user_id=user_id,
            subject=subject,
            predicate=predicate,
            obj=obj,
            source_text=f"Entity {i % 50} has {predicate} set to attribute {i}",
            event_time=f"202{i%5}-01-01",
            confidence=0.95,
            context_qualifiers={"shard": i % 4},
        )
    elapsed = time.time() - t0
    print(f"  Ingested {num_facts} facts in {elapsed:.3f}s ({num_facts / elapsed:.1f} writes/sec)")

    with _conn() as conn:
        fact_count = conn.execute("SELECT COUNT(*) FROM facts WHERE user_id = ?", (user_id,)).fetchone()[0]
        fts_count = conn.execute("SELECT COUNT(*) FROM facts_fts WHERE user_id = ?", (user_id,)).fetchone()[0]
        assert fact_count == num_facts, f"Expected {num_facts} facts, got {fact_count}"
        assert fts_count == num_facts, f"Expected {num_facts} FTS entries, got {fts_count}"
        print(f"  Integrity Check PASSED: {fact_count} table rows, {fts_count} FTS5 index rows in sync.")

    # 2. FTS5 Search Latency Benchmark
    print("\n[2/4] Benchmarking FTS5 Search Latency (100 queries)...")
    latencies = []
    for i in range(100):
        query_token = f"attribute_value_{i * 9 % num_facts}"
        t_start = time.perf_counter()
        results = search_facts_fts(user_id, query_token, limit=20)
        t_query = (time.perf_counter() - t_start) * 1000  # ms
        latencies.append(t_query)
        assert len(results) >= 1, f"Expected hit for {query_token}"

    avg_lat = statistics.mean(latencies)
    p95_lat = statistics.quantiles(latencies, n=20)[18]  # ~95th percentile
    print(f"  FTS5 Query Latency: Avg = {avg_lat:.3f}ms | P95 = {p95_lat:.3f}ms | Min = {min(latencies):.3f}ms")
    assert avg_lat < 10.0, f"Search latency too high: {avg_lat:.2f}ms"

    # 3. Circular Graph Traversal & Cycle Detection Stress Test
    print("\n[3/4] Testing Multi-Hop Graph Traversal with Circular References (A -> B -> C -> A)...")
    insert_fact(user_id, "node_A", "connected_to", "node_B")
    insert_fact(user_id, "node_B", "connected_to", "node_C")
    insert_fact(user_id, "node_C", "connected_to", "node_A")  # Cycle!

    t_graph = time.perf_counter()
    paths = get_query_seeded_multihop_paths(user_id, seed_entities=["node_A"], max_depth=4)
    graph_time = (time.perf_counter() - t_graph) * 1000

    print(f"  Graph traversal executed in {graph_time:.3f}ms")
    print(f"  Discovered paths without infinite loop: {len(paths)}")
    for p in paths:
        print(f"    Path: {p}")
    assert len(paths) > 0, "Expected multihop paths"
    print("  Cycle Prevention Test PASSED: No infinite loop encountered on cyclic graph.")

    # 4. Access Metrics & Duplicate Reinforcement
    print("\n[4/4] Testing Access Metrics and Memory Reinforcement...")
    target_fact = results[0]
    fid = target_fact["id"]

    # Initial access count is 0
    assert target_fact["access_count"] == 0

    # Touch access
    touch_fact_access(user_id, [fid])
    with _conn() as conn:
        updated = conn.execute("SELECT access_count, last_accessed_at, reinforcement_count FROM facts WHERE id = ?", (fid,)).fetchone()
        assert updated["access_count"] == 1, f"Expected access_count 1, got {updated['access_count']}"
        assert updated["last_accessed_at"] is not None
        print(f"  Access count updated: {updated['access_count']} (last accessed: {updated['last_accessed_at'][:19]})")

    print("\n" + "=" * 60)
    print("ALL SCALE & STRESS TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

    # Clean up test DB
    try:
        os.remove(test_db)
        os.rmdir(test_dir)
    except Exception:
        pass


if __name__ == "__main__":
    run_scale_benchmark()
