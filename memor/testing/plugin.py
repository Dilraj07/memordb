import pytest
import json
import os
import logging
from typing import List, Dict

from pydantic import BaseModel
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from memor.config import chat, TransientLLMError
from memor.testing.adapter import MemoryBackendAdapter

logger = logging.getLogger(__name__)

class PresentValuesResult(BaseModel):
    present_values: list[str] = []


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((TransientLLMError, json.JSONDecodeError, ValueError)),
)
def _extract_present_values(expected: dict, actual_facts) -> list[str]:
    """LLM call scoped to the one thing it's reliable at: naming which values
    in actual_facts share the expected fact's relation. The actual true/false
    decision is made in Python, not by the model -- see _is_semantic_match for
    why that split matters."""
    prompt = f"""Expected fact: {expected}
Actual memories: {actual_facts}

List every object/value in "Actual memories" that shares the same
predicate/relation as the expected fact (a clearly synonymous predicate
counts, e.g. speaks_natively/speaks_language, has_pet/owns_pet). If none of
the memories share that predicate/relation, reply with an empty list.

Reply JSON only: {{"present_values": ["..."]}}"""

    raw = chat([{"role": "user", "content": prompt}], json_mode=True, temperature=0.0)
    try:
        parsed = json.loads(raw)
        return PresentValuesResult(**parsed).present_values
    except Exception as e:
        raise ValueError(f"Failed to parse or validate present_values output: {raw}") from e


def _value_matches(expected_object: str, candidates: list[str]) -> bool:
    if not candidates:
        return False

    exp_norm = expected_object.strip().lower()
    for c in candidates:
        c_norm = c.strip().lower()
        if exp_norm == c_norm or exp_norm in c_norm or c_norm in exp_norm:
            return True

    # Fallback for true paraphrases that don't share substrings (e.g.
    # "carnivore" vs "meat eater"). Reuses the same embedding model the
    # retriever already loads, so this doesn't pull in a second model.
    from memor.retriever import get_embedding_model
    from sentence_transformers import util

    model = get_embedding_model()
    exp_emb = model.encode(expected_object, convert_to_tensor=True)
    cand_embs = model.encode(candidates, convert_to_tensor=True)
    scores = util.cos_sim(exp_emb, cand_embs)[0].cpu().tolist()
    return max(scores) >= 0.75


def _is_semantic_match(expected: dict, actual_facts: List[Dict[str, str]] | List[str]) -> bool:
    """
    Whether `expected` (a {"predicate", "object"} dict) is semantically
    present in actual_facts.

    Split into an LLM extraction step (find candidate values for this
    relation) followed by a deterministic Python comparison, instead of a
    single "is this a match, yes/no" LLM call. That single-call version was
    empirically unreliable on this model even when restructured into an
    explicit two-step chain-of-thought prompt: it would correctly list
    present_values containing the expected value and then still answer
    match=false, contradicting its own listed output. Moving the actual
    comparison out of the model and into code removes that failure mode
    entirely instead of trying to prompt around it.
    """
    if not actual_facts:
        return False
    present_values = _extract_present_values(expected, actual_facts)
    return _value_matches(expected["object"], present_values)


def load_cases():
    base_dir = os.path.dirname(__file__)
    cases_path = os.path.join(base_dir, "cases.json")
    with open(cases_path, "r") as f:
        return json.load(f)

CASES = load_cases()

def pytest_generate_tests(metafunc):
    """
    Automatically injects test_temporal_memory_benchmark into the pytest session
    if the function exists in the user's test suite, parametrizing it with all cases.
    Alternatively, we can just define the test function here directly, but pytest plugins
    usually provide fixtures or collect specific files.

    Actually, since we want to *provide* the test itself, we can define a base class
    that users can inherit from, or just define the test directly in this plugin namespace.
    """
    pass

# We define the actual test case that Pytest will collect if this plugin is loaded.
# Users must provide a `memor_adapter` fixture that yields an instance of MemoryBackendAdapter.

class BaseTemporalMemoryBenchmark:
    @pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
    def test_temporal_memory_benchmark(self, memor_adapter: MemoryBackendAdapter, case):
        """
        End-to-End Pipeline Evaluation for a single benchmark case.
        Validates both the Write Path and the Read Path using the provided MemoryBackendAdapter.
        """
        case_id = case["id"]
        statements = case["statements"]
        expected_active = case["expected_active"]
        expected_inactive = case.get("expected_inactive", [])
        query = case["query"]

        user_id = f"user_bench_{case_id}"

        memor_adapter.clear(user_id)

        # 1. INGESTION (WRITE PATH)
        for stmt in statements:
            memor_adapter.add_fact(user_id, stmt)

        # 2. VERIFY WRITE PATH (Storage State)
        active_rows = memor_adapter.get_all_active_facts(user_id)

        for exp in expected_active:
            assert _is_semantic_match(exp, active_rows), (
                f"Write Path Failure: Expected fact {exp} missing or contradicted in store.\nActual: {active_rows}"
            )

        # Contradiction cases must also prove the superseded fact is gone, not
        # just that the new one showed up alongside it -- a store that keeps
        # both "active" hasn't actually resolved anything.
        for exp in expected_inactive:
            assert not _is_semantic_match(exp, active_rows), (
                f"Write Path Failure: Superseded fact {exp} is still active in store.\nActual: {active_rows}"
            )

        # 3. VERIFY READ PATH (Hybrid Retrieval + Multi-hop)
        retrieved_texts = memor_adapter.query(user_id, query, k=3)

        for exp in expected_active:
            assert _is_semantic_match(exp, retrieved_texts), (
                f"Read Path Failure: Retriever failed to surface expected fact {exp} for query '{query}'.\nRetrieved context: {retrieved_texts}"
            )

        for exp in expected_inactive:
            assert not _is_semantic_match(exp, retrieved_texts), (
                f"Read Path Failure: Retriever surfaced a superseded fact {exp} for query '{query}'.\nRetrieved context: {retrieved_texts}"
            )

