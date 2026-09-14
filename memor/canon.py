"""
Predicate canonicalization.

The fact extractor names predicates freely per turn -- it's an LLM call, not a
fixed vocabulary, so it can just as easily emit "lives_in" one turn and
"resides_in" or "current_city" the next for the same relation. Resolution
matches predicates by exact string equality (memor.store.get_active_facts is
scoped to user_id+subject+predicate), so that drift means two facts that
should conflict never even get compared -- which reproduces the exact
stale-fact bug this project exists to prevent.

Two layers, cheapest first:
  1. A static alias table for synonyms the extraction model is known to
     alternate between -- zero cost, deterministic.
  2. An embedding-similarity fallback against predicates already stored for
     the same (user_id, subject) -- catches drift the static table doesn't
     cover, without needing another LLM call (reuses the same embedding
     model the retriever already loads).
"""
import logging
from memor.store import get_distinct_predicates

logger = logging.getLogger("memor")

# Keys and values are pre-normalized (snake_case). canonicalize_predicate()
# normalizes incoming predicates before checking this table.
_ALIAS_MAP = {
    "resides_in": "lives_in",
    "current_city": "lives_in",
    "current_residence": "lives_in",
    "based_in": "lives_in",
    "location": "lives_in",
    "current_location": "lives_in",
    "city": "lives_in",
    "employer": "works_at",
    "employed_at": "works_at",
    "company": "works_at",
    "job_title": "has_role",
    "occupation": "has_role",
    "role": "has_role",
    "job": "has_role",
    "attends": "studies_at",
    "school": "studies_at",
    "university": "studies_at",
    "college": "studies_at",
    "pet": "owns_pet",
    "has_pet": "owns_pet",
    "owns": "owns_pet",
    "favorite_movie": "likes_movie",
    "favourite_movie": "likes_movie",
    "favorite_food": "likes_food",
    "favourite_food": "likes_food",
    "favorite_color": "likes_color",
    "favourite_color": "likes_color",
    "favorite_hobby": "hobby",
    "hobbies": "hobby",
    "speaks": "speaks_language",
    "speaks_natively": "speaks_language",
    "language": "speaks_language",
    "languages": "speaks_language",
    "instrument": "plays_instrument",
}

# Conservative -- a false merge (treating two genuinely different relations as
# the same predicate) silently overwrites a fact via the UPDATE fast path,
# which is worse than an occasional missed match that just falls through to ADD.
_SIMILARITY_THRESHOLD = 0.82


def _normalize(predicate: str) -> str:
    return predicate.strip().lower().replace(" ", "_").replace("-", "_")


def canonicalize_predicate(user_id: str, subject: str, predicate: str) -> str:
    """
    Maps a freshly extracted predicate onto whatever name is already in use
    for this (user_id, subject) pair, so resolution compares like with like.

    Order: normalized exact match against existing predicates (no-op case) ->
    static alias table -> embedding-similarity match against predicates
    already on file for this subject. The similarity step only runs when
    there's something to compare against, so a brand-new subject costs
    nothing beyond the alias lookup.
    """
    norm = _normalize(predicate)
    existing = get_distinct_predicates(user_id, subject)

    if norm in existing:
        return norm

    aliased = _ALIAS_MAP.get(norm, norm)
    if aliased in existing:
        logger.debug("Predicate alias hit: %s -> %s", norm, aliased)
        return aliased

    if not existing:
        return aliased

    match = _best_similarity_match(aliased, existing)
    if match is not None:
        logger.info(
            "Predicate canonicalized by similarity: %s -> %s (user=%s, subject=%s)",
            norm, match, user_id, subject,
        )
        return match

    return aliased


# Relationship words are symmetric in meaning but the extractor isn't always
# consistent about which side it puts "user" on for a reflexive statement like
# "My brother is named Alex." (sometimes subject=user/object=Alex, sometimes
# subject=Alex/object=user for the identical sentence, across otherwise
# identical extraction calls). Multi-hop traversal depends on "user" being a
# stable anchor subject for these, so normalize deterministically rather than
# relying on the model to pick the same direction every time.
_SYMMETRIC_RELATIONSHIP_PREDICATES = {
    "brother", "sister", "sibling", "friend", "spouse", "husband", "wife",
    "partner", "cousin", "colleague", "coworker", "neighbor", "mother",
    "father", "parent", "son", "daughter", "child", "boss", "manager",
}


def normalize_relationship_direction(subject: str, predicate: str, obj: str) -> tuple[str, str]:
    """If a symmetric relationship fact has "user" on the object side, flip it
    onto the subject side. Returns (subject, object); predicate is unchanged."""
    if (
        _normalize(predicate) in _SYMMETRIC_RELATIONSHIP_PREDICATES
        and obj.strip().lower() == "user"
        and subject.strip().lower() != "user"
    ):
        return "user", subject
    return subject, obj


def _best_similarity_match(predicate: str, candidates: list[str]) -> str | None:
    from memor.retriever import get_embedding_model
    from sentence_transformers import util

    model = get_embedding_model()
    query_text = predicate.replace("_", " ")
    candidate_texts = [c.replace("_", " ") for c in candidates]

    query_emb = model.encode(query_text, convert_to_tensor=True)
    candidate_embs = model.encode(candidate_texts, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, candidate_embs)[0].cpu().tolist()

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    if scores[best_idx] >= _SIMILARITY_THRESHOLD:
        return candidates[best_idx]
    return None
