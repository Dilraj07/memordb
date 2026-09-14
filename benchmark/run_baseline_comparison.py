"""
Runs the planted-contradiction / multi-hop benchmark suite (memor/testing/cases.json)
against two backends and writes a real, reproducible comparison to
benchmark/results.md:

  - memor        : this project's actual MemoryEngine (bitemporal resolution +
                    hybrid retrieval + multi-hop traversal), via MemorEngineAdapter.
  - naive_rag     : the "standard vector DB" baseline described in
                    docs/architecture.md Section 2 -- raw chunk embedding + plain
                    top-k cosine similarity, no extraction/resolution/graph.

Both backends are graded by the identical LLM-as-judge semantic match check
used by the pytest benchmark (memor.testing.plugin._is_semantic_match), so the
comparison isn't using two different bars.

Requires GROQ_API_KEY (used by Memor's extraction/resolution/rerank calls and
by the judge). Writes benchmark/results.md and benchmark/results.json.

Usage:
    python benchmark/run_baseline_comparison.py
"""
import os
import sys
import json
import time
from datetime import datetime, timezone

import memor.store as store

_DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_comparison.db")
if os.path.exists(_DB_FILE):
    os.remove(_DB_FILE)
store.DB_PATH = _DB_FILE

from memor.testing.plugin import load_cases, _is_semantic_match
from memor.testing.memor_adapter import MemorEngineAdapter
from memor.testing.naive_rag_adapter import NaiveRAGAdapter


def run_case(adapter, case: dict, backend_tag: str) -> dict:
    user_id = f"user_{backend_tag}_{case['id']}"
    adapter.clear(user_id)

    for stmt in case["statements"]:
        adapter.add_fact(user_id, stmt)

    expected_active = case["expected_active"]
    expected_inactive = case.get("expected_inactive", [])

    active_rows = adapter.get_all_active_facts(user_id)
    write_active_ok = all(_is_semantic_match(exp, active_rows) for exp in expected_active)
    # The real test for a contradiction case: is the superseded fact actually
    # gone, or is the backend just accumulating everything and never resolving?
    write_stale_gone = all(not _is_semantic_match(exp, active_rows) for exp in expected_inactive)

    retrieved = adapter.query(user_id, case["query"], k=3)
    read_active_ok = all(_is_semantic_match(exp, retrieved) for exp in expected_active)
    read_stale_gone = all(not _is_semantic_match(exp, retrieved) for exp in expected_inactive)

    return {
        "case_id": case["id"],
        "write_path_pass": write_active_ok and write_stale_gone,
        "read_path_pass": read_active_ok and read_stale_gone,
        "overall_pass": write_active_ok and write_stale_gone and read_active_ok and read_stale_gone,
        "has_contradiction": bool(expected_inactive),
        "retrieved_preview": retrieved[:3],
    }


def main():
    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set -- required for extraction/resolution and the LLM judge.", file=sys.stderr)
        sys.exit(1)

    cases = load_cases()
    print(f"Loaded {len(cases)} benchmark cases.\n")

    print("Running against memor (bitemporal engine)...")
    memor_adapter = MemorEngineAdapter(max_workers=2)
    memor_results = []
    t0 = time.monotonic()
    for case in cases:
        print(f"  {case['id']} ...")
        memor_results.append(run_case(memor_adapter, case, "memor"))
    memor_elapsed = time.monotonic() - t0
    memor_adapter.shutdown()

    print("\nRunning against naive_rag (raw-chunk vector baseline)...")
    naive_adapter = NaiveRAGAdapter()
    naive_results = []
    t0 = time.monotonic()
    for case in cases:
        print(f"  {case['id']} ...")
        naive_results.append(run_case(naive_adapter, case, "naive"))
    naive_elapsed = time.monotonic() - t0

    if os.path.exists(_DB_FILE):
        os.remove(_DB_FILE)
    wal = _DB_FILE + "-wal"
    shm = _DB_FILE + "-shm"
    for f in (wal, shm):
        if os.path.exists(f):
            os.remove(f)

    write_results(cases, memor_results, naive_results, memor_elapsed, naive_elapsed)


def write_results(cases, memor_results, naive_results, memor_elapsed, naive_elapsed):
    def pass_rate(results, key):
        return sum(1 for r in results if r[key]) / len(results) * 100

    out_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now(timezone.utc).isoformat()

    payload = {
        "generated_at": timestamp,
        "case_count": len(cases),
        "memor": {
            "elapsed_seconds": round(memor_elapsed, 1),
            "write_path_pass_rate": pass_rate(memor_results, "write_path_pass"),
            "read_path_pass_rate": pass_rate(memor_results, "read_path_pass"),
            "overall_pass_rate": pass_rate(memor_results, "overall_pass"),
            "results": memor_results,
        },
        "naive_rag": {
            "elapsed_seconds": round(naive_elapsed, 1),
            "write_path_pass_rate": pass_rate(naive_results, "write_path_pass"),
            "read_path_pass_rate": pass_rate(naive_results, "read_path_pass"),
            "overall_pass_rate": pass_rate(naive_results, "overall_pass"),
            "results": naive_results,
        },
    }

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(payload, f, indent=2)

    lines = []
    lines.append("# Benchmark Results: Memor vs. Naive-RAG Baseline")
    lines.append("")
    lines.append(f"Generated: {timestamp}")
    lines.append(f"Cases: {len(cases)} (see `memor/testing/cases.json`)")
    lines.append("")
    lines.append(
        "Both backends implement the same `MemoryBackendAdapter` interface and are graded by the "
        "identical LLM-as-judge semantic match check (`memor.testing.plugin._is_semantic_match`). "
        "The naive-RAG baseline is raw message chunks embedded and retrieved by plain top-k cosine "
        "similarity -- no fact extraction, no contradiction resolution, no multi-hop traversal -- "
        "i.e. the \"standard vector DB\" approach described in `docs/architecture.md`."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Backend | Write-path pass | Read-path pass | Overall pass | Wall time |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        f"| memor | {payload['memor']['write_path_pass_rate']:.0f}% "
        f"| {payload['memor']['read_path_pass_rate']:.0f}% "
        f"| {payload['memor']['overall_pass_rate']:.0f}% "
        f"| {payload['memor']['elapsed_seconds']}s |"
    )
    lines.append(
        f"| naive_rag | {payload['naive_rag']['write_path_pass_rate']:.0f}% "
        f"| {payload['naive_rag']['read_path_pass_rate']:.0f}% "
        f"| {payload['naive_rag']['overall_pass_rate']:.0f}% "
        f"| {payload['naive_rag']['elapsed_seconds']}s |"
    )
    lines.append("")
    lines.append("## Per-case results")
    lines.append("")
    lines.append("| Case | memor | naive_rag | Notes |")
    lines.append("|---|---|---|---|")
    naive_by_id = {r["case_id"]: r for r in naive_results}
    memor_by_id = {r["case_id"]: r for r in memor_results}
    for case in cases:
        cid = case["id"]
        m = memor_by_id[cid]
        n = naive_by_id[cid]
        m_mark = "PASS" if m["overall_pass"] else "FAIL"
        n_mark = "PASS" if n["overall_pass"] else "FAIL"
        note = ""
        if m["overall_pass"] and not n["overall_pass"]:
            note = "temporal drift / no graph traversal"
        lines.append(f"| `{cid}` | {m_mark} | {n_mark} | {note} |")

    lines.append("")
    lines.append(
        "Raw retrieved context for every case (useful for seeing *why* a case failed, not just that it "
        "did) is in `results.json`."
    )

    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nWrote {os.path.join(out_dir, 'results.md')} and results.json")
    print(f"memor overall pass rate:      {payload['memor']['overall_pass_rate']:.0f}%")
    print(f"naive_rag overall pass rate:  {payload['naive_rag']['overall_pass_rate']:.0f}%")


if __name__ == "__main__":
    main()
