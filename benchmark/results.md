# Benchmark Results: Memor vs. Naive-RAG Baseline

Generated: 2026-08-14T18:49:27.723849+00:00
Cases: 12 (see `memor/testing/cases.json`)

Both backends implement the same `MemoryBackendAdapter` interface and are graded by the identical LLM-as-judge semantic match check (`memor.testing.plugin._is_semantic_match`). The naive-RAG baseline is raw message chunks embedded and retrieved by plain top-k cosine similarity -- no fact extraction, no contradiction resolution, no multi-hop traversal -- i.e. the "standard vector DB" approach described in `docs/architecture.md`.

## Summary

| Backend | Write-path pass | Read-path pass | Overall pass | Wall time |
|---|---|---|---|---|
| memor | 100% | 100% | 100% | 110.1s |
| naive_rag | 100% | 100% | 100% | 72.2s |

## Per-case results

| Case | memor | naive_rag | Notes |
|---|---|---|---|
| `case_1_location_update` | PASS | PASS |  |
| `case_2_diet_additive` | PASS | PASS |  |
| `case_3_diet_contradiction` | PASS | PASS |  |
| `case_4_pet_additive` | PASS | PASS |  |
| `case_5_job_update` | PASS | PASS |  |
| `case_6_hobby_additive` | PASS | PASS |  |
| `case_7_relationship_update` | PASS | PASS |  |
| `case_8_language_additive` | PASS | PASS |  |
| `case_9_device_update` | PASS | PASS |  |
| `case_10_duplicate_fact` | PASS | PASS |  |
| `case_11_multihop_base` | PASS | PASS |  |
| `case_12_multihop_contradiction` | PASS | PASS |  |

Raw retrieved context for every case (useful for seeing *why* a case failed, not just that it did) is in `results.json`.
