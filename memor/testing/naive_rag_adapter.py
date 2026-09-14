"""
Naive-RAG baseline adapter: the "standard vector DB" approach Memor's own
architecture doc (docs/architecture.md, Section 2) names as the status quo it
improves on. Raw message chunks embedded and retrieved by plain cosine
similarity -- no fact extraction, no bitemporal contradiction resolution, no
multi-hop graph traversal, no reranking.

This exists so benchmark/run_baseline_comparison.py can produce a real,
reproducible number instead of an assertion: run the same planted-contradiction
and multi-hop cases through this adapter and through MemorEngineAdapter via the
same MemoryBackendAdapter interface, using the identical LLM-judge grading in
memor.testing.plugin, and compare pass rates.
"""
import threading
from typing import List, Dict

from sentence_transformers import SentenceTransformer, util

from memor.testing.adapter import MemoryBackendAdapter

_model = None
_model_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


class NaiveRAGAdapter(MemoryBackendAdapter):
    def __init__(self):
        self._model = _get_model()
        self._chunks: dict[str, list[str]] = {}
        self._embeddings: dict[str, object] = {}

    def add_fact(self, user_id: str, text: str) -> None:
        # No extraction: the raw message is the stored unit, same as chunking
        # a conversation straight into a vector store.
        self._chunks.setdefault(user_id, []).append(text)
        self._embeddings.pop(user_id, None)

    def _ensure_embeddings(self, user_id: str):
        if user_id not in self._embeddings:
            chunks = self._chunks.get(user_id, [])
            if chunks:
                self._embeddings[user_id] = self._model.encode(chunks, convert_to_tensor=True)

    def get_all_active_facts(self, user_id: str) -> List[Dict[str, str]]:
        # No structured facts exist in this baseline -- expose the raw chunks
        # under the same dict shape the adapter interface expects, so the
        # benchmark's write-path semantic-match check can still run against it.
        return [{"predicate": "raw_message", "object": c} for c in self._chunks.get(user_id, [])]

    def query(self, user_id: str, query_text: str, k: int = 3) -> List[str]:
        chunks = self._chunks.get(user_id, [])
        if not chunks:
            return []
        self._ensure_embeddings(user_id)
        q_emb = self._model.encode(query_text, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, self._embeddings[user_id])[0].cpu().tolist()
        ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
        return [text for text, _ in ranked[:k]]

    def clear(self, user_id: str) -> None:
        self._chunks.pop(user_id, None)
        self._embeddings.pop(user_id, None)

    def shutdown(self):
        pass
