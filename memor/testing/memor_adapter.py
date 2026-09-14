"""
Reference MemoryBackendAdapter wrapping Memor's own MemoryEngine.

This is what the benchmark suite (memor/testing/plugin.py) runs against by
default, and it doubles as the example other developers copy when wiring up
their own backend for the pytest-temporal-memory plugin.
"""
from typing import List, Dict

import memor.store as store
from memor.engine import MemoryEngine
from memor.testing.adapter import MemoryBackendAdapter


class MemorEngineAdapter(MemoryBackendAdapter):
    def __init__(self, db_path: str | None = None, max_workers: int = 2):
        self.engine = MemoryEngine(db_path=db_path, max_workers=max_workers)

    def add_fact(self, user_id: str, text: str) -> None:
        # sync=True ensures we wait for extraction and DB writes
        self.engine.add_memory(user_id, text, sync=True)

    def get_all_active_facts(self, user_id: str) -> List[Dict[str, str]]:
        active_rows = store.get_all_active_facts(user_id)
        return [{"predicate": r["predicate"], "object": r["object"]} for r in active_rows]

    def query(self, user_id: str, query_text: str, k: int = 3) -> List[str]:
        # engine.query() returns one newline-joined, "- "-prefixed string meant
        # for direct LLM prompt injection (see README) -- the adapter interface
        # requires List[str], so split it back into individual context lines.
        # Passing the joined blob straight through as a length-1 list previously
        # made the LLM-judge benchmark check unreliable: a single multi-line
        # string reads very differently to the judge than a proper list of
        # discrete facts does.
        result = self.engine.query(user_id, query_text, k=k)
        if result == "No relevant context found.":
            return []
        return [line[2:] if line.startswith("- ") else line for line in result.split("\n") if line.strip()]

    def clear(self, user_id: str) -> None:
        self.engine.clear_memories(user_id)

    def shutdown(self):
        self.engine.shutdown()
