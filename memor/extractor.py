"""
Turns a raw conversation turn into structured, timestamped facts.

This is the first place naive memory projects cut corners — they either skip
extraction entirely (raw text -> embedding) or extract with no schema. Structured
extraction is what makes conflict resolution and temporal queries possible later.
"""
import json
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from memor.config import chat, TransientLLMError

EXTRACTION_PROMPT = """Extract durable facts as JSON: {{"facts":[{{"subject":"str","predicate":"str","object":"str","event_time":null,"confidence":1.0,"context_qualifiers":null}}]}}
- Skip questions; only extract declarative facts.
- Extract all facts: preferences, hobbies, relationships, job, devices, colors, pets, location, diet, skills, relationship/marital status.
- Casual expressions of preference ("I love X", "I'm a huge fan of X", "X is
  great") ARE durable facts worth extracting -- don't require an explicit
  "my favorite is" framing before extracting one.
- When a message names a relationship to another person (brother, sister,
  friend, colleague, etc.), extract subject = whoever is speaking (usually
  "user"), predicate = the relationship word itself (e.g. "brother",
  "friend"), object = that person's name. Example: "My brother is named
  Alex." -> {{"subject":"user","predicate":"brother","object":"Alex"}}. This
  lets later facts about that named person (subject = their name) chain onto
  this one for multi-hop lookups.
- subject: the person the fact is actually about. Use "user" only when the
  fact is about the person speaking. If a named person is the one doing the
  action or holding the attribute, the subject is their name -- even if
  that's the only mention of them in this message (e.g. "Sarah moved to
  Seattle." -> subject "Sarah", NOT "user"; don't default to "user" just
  because no other subject was established earlier in this message).
- predicate: a STABLE relation name. The value that can change over time belongs in
  "object" -- never bake it into the predicate name.
    Wrong: {{"predicate":"is_single","object":"yes"}}
    Wrong: {{"predicate":"got_married","object":"last week"}}
    Right: {{"predicate":"marital_status","object":"married"}}
  Reuse these exact predicate names whenever they apply, don't invent a synonym:
  lives_in, works_at, has_role, studies_at, owns_pet, marital_status, diet,
  uses_device, speaks_language, plays_instrument, likes_food, likes_movie,
  likes_color, hobby, allergic_to, favorite_color. Use a short snake_case name
  for anything else.
- For musical instruments (playing, learning, or practicing instruments like
  piano, guitar, drums, violin, etc.), ALWAYS use predicate "plays_instrument",
  NOT "hobby" (e.g. "started learning piano" -> predicate: "plays_instrument", object: "piano").
- For personal tech devices (phones, laptops, operating systems), use "uses_device".
- When a message describes a CHANGE (moved, quit, switched jobs, broke up),
  extract ONLY the new current-state fact with the stable predicate (e.g.
  works_at: OpenAI). Do not also invent a separate "previously_X" / "used_to_X"
  fact for the old value -- the store keeps that history automatically once
  the new fact supersedes it; a duplicate historical fact just pollutes the
  active set.
- event_time: if the utterance refers to a specific time or date in the real world
  (e.g., "In 2022", "Last year", "Next month"), capture it in ISO8601 or descriptive string; else null.
- context_qualifiers: if a fact applies conditionally or in a specific scope (e.g. {{"scope":"work"}}), capture as key-value pairs; else null.
- object: the value with useful identifying context ("dog named Max", not
  just "Max"), but not the whole sentence restated, and no timing filler
  ("today", "last week") -- that's tracked separately, not part of the value.
- No facts → {{"facts":[]}}

Message: "{message}"
"""


class Fact(BaseModel):
    subject: str
    predicate: str
    object: str
    event_time: str | None = None
    confidence: float = 1.0
    context_qualifiers: dict | None = None
    extracted_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_text: str = ""



class ExtractionResult(BaseModel):
    facts: list[Fact]


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((TransientLLMError, json.JSONDecodeError, ValueError))
)
def extract_facts(message: str) -> list[Fact]:
    raw = chat(
        [{"role": "user", "content": EXTRACTION_PROMPT.format(message=message)}],
        json_mode=True,
    )
    
    # Let Pydantic validate the structured output. If it fails (ValueError),
    # tenacity will automatically retry the LLM call up to 3 times.
    try:
        parsed = json.loads(raw)
        result = ExtractionResult(**parsed)
    except Exception as e:
        # Re-raise to trigger tenacity retry
        raise ValueError(f"Failed to parse or validate LLM output: {raw}") from e

    # Annotate with source text
    for f in result.facts:
        f.source_text = message
        
    return result.facts
