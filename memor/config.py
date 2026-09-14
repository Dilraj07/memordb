"""
Central config. Uses Groq's OpenAI-compatible client, but you can swap this for
any provider (OpenAI, Anthropic, local Ollama via OpenAI-compat shim) without
touching the rest of the engine — everything downstream calls `chat()` and
retries on the provider-agnostic exceptions defined here (TransientLLMError /
PermanentLLMError), never on the groq SDK's exception classes directly. That's
what actually makes the provider swap a one-file change: update get_client(),
chat(), and the two exception tuples below, and every retry decorator in
extractor.py/resolver.py/retriever.py keeps working unmodified.
"""
import os
from dotenv import load_dotenv
from groq import Groq
import groq
import logging

logger = logging.getLogger("memor")

load_dotenv()

_client = None
MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")


class LLMError(Exception):
    """Base class for all failures raised by chat(). Provider-agnostic: callers
    should never need to import the underlying SDK's exception types."""


class TransientLLMError(LLMError):
    """Worth retrying: rate limits, timeouts, connection drops, 5xx responses."""


class PermanentLLMError(LLMError):
    """Not worth retrying: bad auth, malformed request, etc."""


# Only this module needs to know which groq SDK exceptions map to which
# internal category. Swapping providers means updating get_client(), chat(),
# and these two tuples -- nothing else in the codebase touches groq.* directly.
_TRANSIENT_GROQ_ERRORS = (groq.APIConnectionError, groq.RateLimitError, groq.InternalServerError)
_PERMANENT_GROQ_ERRORS = (
    groq.AuthenticationError, groq.BadRequestError, groq.PermissionDeniedError,
    groq.NotFoundError, groq.UnprocessableEntityError, groq.ConflictError,
)


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file or set it as an environment variable. "
                "Get a free key at https://console.groq.com/keys"
            )
        _client = Groq(api_key=api_key)
    return _client


def chat(messages: list[dict], json_mode: bool = False, temperature: float = 0.0, max_tokens: int = 256) -> str:
    """Single entry point for all LLM calls in this project. Keep the rest of the
    codebase provider-agnostic by routing everything through here."""
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    client = get_client()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
    except _TRANSIENT_GROQ_ERRORS as e:
        raise TransientLLMError(str(e)) from e
    except _PERMANENT_GROQ_ERRORS as e:
        raise PermanentLLMError(str(e)) from e

    if resp.usage:
        logger.debug(
            f"Tokens: {resp.usage.prompt_tokens}in/{resp.usage.completion_tokens}out "
            f"(total: {resp.usage.total_tokens})"
        )
    return resp.choices[0].message.content

def chat_stream(messages: list[dict], temperature: float = 0.7):
    """Generator that yields streaming chunks for interactive chat."""
    client = get_client()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    for chunk in resp:
        content = chunk.choices[0].delta.content
        if content is not None:
            yield content
