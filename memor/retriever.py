"""
Hybrid retrieval over the fact store.

Why three signals instead of just vector similarity:
  - Vector (sentence-transformers) catches semantic/paraphrase matches: query "where does he live" matches fact "lives_in".
  - BM25 catches exact term matches that embeddings sometimes under-weight,
    especially for names and rare tokens: "Bangalore" should score high even if
    the embedding space doesn't separate cities well.
  - Entity overlap is a cheap, high-precision signal: if the query names a subject
    that also appears in the fact, that's strong evidence, and it doesn't need any
    model at all.

Fusing them (rather than picking one) is the actual lesson here -- vector-only
retrieval is the single most common reason "memory" systems retrieve the wrong
thing at the wrong time.


Swap-in points for later:
  - Replace TF-IDF with sentence-transformers or an API embedding model for real
    semantic similarity (drop-in: just change `_vectorize`).
  - Replace the LLM rerank pass with a trained cross-encoder if you want lower
    latency at scale.
"""
import re
import json
import sqlite3
import logging
import threading
from dataclasses import dataclass

from pydantic import BaseModel
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

logger = logging.getLogger(__name__)
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi

from memor.store import (
    get_all_active_facts,
    get_multihop_paths,
    update_fact_embeddings,
    search_facts_fts,
    get_query_seeded_multihop_paths,
    touch_fact_access,
)
from memor.config import chat, TransientLLMError

RERANK_PROMPT = """Query: "{query}"
Score each fact 0-10 for relevance. Reply JSON: {{"scores":[int,...]}}

{numbered_facts}
"""


class RerankResult(BaseModel):
    scores: list[int]


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((TransientLLMError, json.JSONDecodeError, ValueError)),
)
def _rerank_pool(query: str, pool: list) -> list[int]:
    """Batched LLM rerank of a candidate pool. Raises ValueError (retried by
    tenacity, same as extractor/resolver) on malformed JSON, a schema
    mismatch, or a scores array that doesn't line up 1:1 with the pool --
    positional index matching a wrong-length response is worse than failing
    loudly and falling back to the fused score."""
    numbered_facts = "\n".join(f"{i+1}. {r.fact_text}" for i, r in enumerate(pool))
    raw = chat(
        [{"role": "user", "content": RERANK_PROMPT.format(query=query, numbered_facts=numbered_facts)}],
        json_mode=True,
    )
    try:
        parsed = json.loads(raw)
        result = RerankResult(**parsed)
    except Exception as e:
        raise ValueError(f"Failed to parse or validate rerank output: {raw}") from e

    if len(result.scores) != len(pool):
        raise ValueError(
            f"Rerank returned {len(result.scores)} scores for {len(pool)} candidates -- refusing to "
            "positionally match a mismatched-length response."
        )
    return result.scores


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _fact_to_text(row) -> str:
    if row.get("predicate") == "multihop_path":
        return row["object"]
    return f"{row['subject']} {row['predicate'].replace('_', ' ')} {row['object']}"


@dataclass
class RetrievalResult:
    fact_text: str
    row: dict
    vector_score: float
    bm25_score: float
    entity_score: float
    fused_score: float
    rerank_score: float | None = None


_embedding_model = None
_embedding_model_lock = threading.Lock()

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        with _embedding_model_lock:
            # Re-check inside the lock: another thread may have already built
            # it while we were waiting (double-checked locking avoids two
            # concurrent first-queries both loading a full SentenceTransformer).
            if _embedding_model is None:
                from sentence_transformers import SentenceTransformer
                _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


class HybridRetriever:
    """
    Combines exact keyword matching (BM25) with semantic similarity (SentenceTransformers)
    for high-recall retrieval of facts.
    """
    
    def __init__(self, weights: dict[str, float] | None = None):
        # Tunable fusion weights -- worth an ablation table in your writeup
        # (show accuracy with each signal turned off).
        self.weights = weights or {"vector": 0.4, "bm25": 0.4, "entity": 0.2}
        self.user_id: str | None = None
        self._rows = []
        self._texts = []
        self.embedding_model = get_embedding_model()
        self._embeddings = None
        self._bm25 = None

    def fit(self, user_id: str, rows=None):
        self.user_id = user_id
        base_rows = rows if rows is not None else get_all_active_facts(user_id)
        self._rows = [dict(r) for r in base_rows]

        # Inject multi-hop paths into the corpus to enable GraphRAG-style retrieval
        try:
            paths = get_multihop_paths(user_id, max_depth=2)
            for p in paths:
                self._rows.append({
                    "subject": "multi-hop",
                    "predicate": "multihop_path",
                    "object": p
                })
        except Exception as e:
            logger.error(f"Failed to build multihop paths: {e}")

        self._texts = [_fact_to_text(r) for r in self._rows]

        if not self._texts:
            return self

        self._embeddings = self._build_embeddings()

        tokenized_corpus = [_tokenize(t) for t in self._texts]
        self._bm25 = BM25Okapi(tokenized_corpus)
        return self

    def _build_embeddings(self):
        """
        Encode only the facts that don't already have a cached embedding.
        Without this, invalidating the retriever cache after a single new
        fact (memor.engine._invalidate_cache) forces the *entire* active
        corpus to be re-embedded on the next query -- fine at demo scale,
        a real cost once a user has hundreds/thousands of facts.

        Cached vectors live in facts.embedding (BLOB, schema v5) so they
        survive process restarts too, not just retriever cache hits.
        Synthetic multi-hop rows have no fact id and are always re-encoded --
        they're derived from the current fact graph and aren't worth caching
        against a row that doesn't exist.
        """
        import numpy as np
        import torch

        vectors: list[np.ndarray | None] = [None] * len(self._rows)
        to_encode_idx = []
        to_encode_text = []

        for i, row in enumerate(self._rows):
            blob = row.get("embedding")
            if blob:
                vectors[i] = np.frombuffer(blob, dtype=np.float32)
            else:
                to_encode_idx.append(i)
                to_encode_text.append(self._texts[i])

        if to_encode_text:
            new_vecs = self.embedding_model.encode(to_encode_text, convert_to_numpy=True).astype(np.float32)
            persist_updates = []
            for offset, idx in enumerate(to_encode_idx):
                vectors[idx] = new_vecs[offset]
                fact_id = self._rows[idx].get("id")
                if fact_id is not None:
                    persist_updates.append((fact_id, new_vecs[offset].tobytes()))

            if persist_updates:
                try:
                    update_fact_embeddings(persist_updates)
                except Exception as e:
                    logger.error(f"Failed to persist fact embeddings: {e}")

            logger.debug(f"Embedded {len(to_encode_text)} new facts, reused {len(self._rows) - len(to_encode_text)} cached")

        stacked = np.stack(vectors).astype(np.float32)
        return torch.from_numpy(stacked)

    def _entity_score(self, query: str, row) -> float:
        q_tokens = set(_tokenize(query))
        entity_tokens = set(_tokenize(row["subject"])) | set(_tokenize(row["object"]))
        if not entity_tokens:
            return 0.0
        overlap = q_tokens & entity_tokens
        return len(overlap) / len(entity_tokens)

    def _normalize(self, scores: list[float]) -> list[float]:
        if not scores:
            return scores
        lo, hi = min(scores), max(scores)
        if hi == lo:
            return [0.0 for _ in scores]
        return [(s - lo) / (hi - lo) for s in scores]

    def retrieve(
        self,
        query: str,
        k: int = 5,
        rerank: bool = True,
        rerank_pool: int = 8,
    ) -> list[RetrievalResult]:
        if not self._rows and not self.user_id:
            return []

        # Two-Stage Candidate Retrieval
        candidates: dict[str, dict] = {}
        q_tokens = _tokenize(query)

        # 1. Native Disk-Backed FTS5 Match
        if self.user_id:
            try:
                fts_matches = search_facts_fts(self.user_id, query, limit=50)
                for r in fts_matches:
                    r_dict = dict(r)
                    candidates[f"fact_{r_dict['id']}"] = r_dict
            except Exception as e:
                logger.debug("FTS search fallback: %s", e)

        # 2. Query-Seeded Multi-Hop Graph Traversal (with Cycle Prevention)
        if self.user_id:
            try:
                seed_entities = ["user"] + list(q_tokens)
                paths = get_query_seeded_multihop_paths(self.user_id, seed_entities=seed_entities, max_depth=2)
                for p in paths:
                    candidates[f"hop_{p}"] = {
                        "subject": "multi-hop",
                        "predicate": "multihop_path",
                        "object": p,
                    }
            except Exception as e:
                logger.debug("Seeded multihop fallback: %s", e)

        # 3. Base active rows fallback if candidate pool is small or for small corpora
        if self._rows:
            for r in self._rows:
                key = f"fact_{r.get('id')}" if r.get("id") else f"hop_{r.get('object')}"
                if key not in candidates:
                    if len(candidates) < 50 or self._entity_score(query, r) > 0:
                        candidates[key] = r

        candidate_rows = list(candidates.values())
        if not candidate_rows:
            return []

        candidate_texts = [_fact_to_text(r) for r in candidate_rows]

        # 4. Dense Vector Scoring across Bounded Candidate Pool
        import numpy as np
        import torch

        q_emb = self.embedding_model.encode(query, convert_to_tensor=True)

        cand_vectors = [None] * len(candidate_rows)
        to_encode_idx = []
        to_encode_text = []

        for idx, row in enumerate(candidate_rows):
            blob = row.get("embedding")
            if blob:
                cand_vectors[idx] = np.frombuffer(blob, dtype=np.float32)
            else:
                to_encode_idx.append(idx)
                to_encode_text.append(candidate_texts[idx])

        if to_encode_text:
            new_vecs = self.embedding_model.encode(to_encode_text, convert_to_numpy=True).astype(np.float32)
            persist_updates = []
            for offset, idx in enumerate(to_encode_idx):
                cand_vectors[idx] = new_vecs[offset]
                fid = candidate_rows[idx].get("id")
                if fid is not None:
                    persist_updates.append((fid, new_vecs[offset].tobytes()))
            if persist_updates:
                try:
                    update_fact_embeddings(persist_updates)
                except Exception as e:
                    logger.debug("Failed to persist embeddings: %s", e)

        stacked = np.stack(cand_vectors).astype(np.float32)
        cand_tensors = torch.from_numpy(stacked).to(q_emb.device)
        vector_scores = util.cos_sim(q_emb, cand_tensors)[0].cpu().tolist()

        # BM25 scores across candidate pool
        tokenized_candidates = [_tokenize(t) for t in candidate_texts]
        bm25_pool = BM25Okapi(tokenized_candidates)
        bm25_scores = bm25_pool.get_scores(q_tokens).tolist()

        entity_scores = [self._entity_score(query, r) for r in candidate_rows]

        v_norm = self._normalize(vector_scores)
        b_norm = self._normalize(bm25_scores)
        e_norm = entity_scores

        results = []
        for i, row in enumerate(candidate_rows):
            reinforcement = row.get("reinforcement_count", 1) or 1
            boost = 1.0 + 0.05 * min(reinforcement - 1, 5)

            fused = (
                self.weights["vector"] * v_norm[i]
                + self.weights["bm25"] * b_norm[i]
                + self.weights["entity"] * e_norm[i]
            ) * boost

            results.append(RetrievalResult(
                fact_text=candidate_texts[i],
                row=row,
                vector_score=v_norm[i],
                bm25_score=b_norm[i],
                entity_score=e_norm[i],
                fused_score=fused,
            ))

        results.sort(key=lambda r: r.fused_score, reverse=True)

        if not rerank:
            logger.debug(f"Retrieved {len(results)} facts (fast-path)")
            return results[:k]

        pool = results[:rerank_pool]

        try:
            scores = _rerank_pool(query, pool)
            for r, score in zip(pool, scores):
                r.rerank_score = score
            logger.debug(f"Reranked top {len(pool)} facts via batched LLM call")
        except Exception as e:
            logger.warning(f"LLM rerank failed after retries, falling back to fused score: {e}")
            for r in pool:
                r.rerank_score = r.fused_score * 10

        pool.sort(key=lambda r: r.rerank_score, reverse=True)

        if self.user_id:
            surfaced_ids = [r.row.get("id") for r in pool[:k] if r.row.get("id")]
            if surfaced_ids:
                try:
                    touch_fact_access(self.user_id, surfaced_ids)
                except Exception as e:
                    logger.debug("Failed to touch fact access: %s", e)

        return pool[:k]


