"""
Tests for app.services.generator.

All tests here use a fake Groq client injected directly, so nothing
makes a real network call or requires a real API key. This tests our
own logic (prompt construction, grounding rules, error handling) --
not Groq's model quality, which is out of our control and not our
code to test.

Run from the repo root:
    pytest tests/test_generator.py -v
"""

import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.services.generator import (  # noqa: E402
    NOT_FOUND_MESSAGE,
    GenerationError,
    Generator,
    get_generator,
)
from app.services.retriever import RetrievedChunk  # noqa: E402


def _make_chunk(rank: int, text: str, page: int, similarity: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        rank=rank,
        chunk_id=f"chunk-{rank}",
        text=text,
        source_filename="handbook.pdf",
        page_number=page,
        similarity=similarity,
    )


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeCompletionResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeChatCompletions:
    """Records every call made to it, and returns a scripted answer."""

    def __init__(self, answer_text: str = "This is the generated answer.") -> None:
        self.answer_text = answer_text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeCompletionResponse(self.answer_text)


class _FakeChat:
    def __init__(self, completions: _FakeChatCompletions) -> None:
        self.completions = completions


class _FakeGroqClient:
    def __init__(self, answer_text: str = "This is the generated answer.") -> None:
        self.completions = _FakeChatCompletions(answer_text)
        self.chat = _FakeChat(self.completions)


@pytest.fixture
def generator_with_fake_client() -> tuple[Generator, _FakeGroqClient]:
    gen = Generator(api_key="fake-key-for-tests", model="fake-model")
    fake_client = _FakeGroqClient()
    gen._client = fake_client  # bypass the lazy Groq() construction entirely
    return gen, fake_client


# --- Prompt construction -------------------------------------------------


def test_build_prompt_includes_question_and_context(generator_with_fake_client):
    generator, _ = generator_with_fake_client
    chunks = [_make_chunk(1, "Students must maintain 75% attendance.", page=12)]

    system_prompt, user_prompt = generator.build_prompt("What is the attendance policy?", chunks)

    assert "What is the attendance policy?" in user_prompt
    assert "75% attendance" in user_prompt
    assert "Page 12" in user_prompt
    assert "handbook.pdf" in user_prompt


def test_system_prompt_states_all_grounding_rules():
    # Sanity-check the prompt actually contains the rules the spec
    # requires, not just prose that vaguely gestures at them.
    from app.services.generator import SYSTEM_PROMPT

    assert "only" in SYSTEM_PROMPT.lower()
    assert "do not invent" in SYSTEM_PROMPT.lower() or "invent" in SYSTEM_PROMPT.lower()
    assert "couldn't find enough information" in SYSTEM_PROMPT.lower()
    assert "page number" in SYSTEM_PROMPT.lower()


def test_context_block_lists_multiple_chunks_with_separators(generator_with_fake_client):
    generator, _ = generator_with_fake_client
    chunks = [
        _make_chunk(1, "First relevant fact.", page=3),
        _make_chunk(2, "Second relevant fact.", page=7),
    ]

    _, user_prompt = generator.build_prompt("some question", chunks)

    assert "First relevant fact." in user_prompt
    assert "Second relevant fact." in user_prompt
    assert "Page 3" in user_prompt
    assert "Page 7" in user_prompt


def test_no_chunks_context_block_says_nothing_found(generator_with_fake_client):
    generator, _ = generator_with_fake_client

    _, user_prompt = generator.build_prompt("some question", [])

    assert "no relevant content" in user_prompt.lower()


# --- Generation behaviour ------------------------------------------------


def test_generate_answer_returns_llm_response(generator_with_fake_client):
    generator, fake_client = generator_with_fake_client
    fake_client.completions.answer_text = "Attendance must be at least 75%, per Page 12."
    chunks = [_make_chunk(1, "Students must maintain 75% attendance.", page=12)]

    answer = generator.generate_answer("What is the attendance requirement?", chunks)

    assert answer == "Attendance must be at least 75%, per Page 12."
    assert len(fake_client.completions.calls) == 1


def test_generate_answer_passes_low_temperature_by_default(generator_with_fake_client):
    generator, fake_client = generator_with_fake_client
    chunks = [_make_chunk(1, "Some fact.", page=1)]

    generator.generate_answer("a question", chunks)

    call_kwargs = fake_client.completions.calls[0]
    assert call_kwargs["temperature"] <= 0.2  # fact-retrieval, not creative writing


def test_generate_answer_uses_configured_model(generator_with_fake_client):
    generator, fake_client = generator_with_fake_client
    chunks = [_make_chunk(1, "Some fact.", page=1)]

    generator.generate_answer("a question", chunks)

    assert fake_client.completions.calls[0]["model"] == "fake-model"


def test_no_retrieved_chunks_returns_honest_message_without_calling_api(generator_with_fake_client):
    generator, fake_client = generator_with_fake_client

    answer = generator.generate_answer("What is the attendance policy?", [])

    assert answer == NOT_FOUND_MESSAGE
    # The whole point of this design: no chunks means no API call at all.
    assert len(fake_client.completions.calls) == 0


def test_empty_answer_from_llm_falls_back_to_not_found_message(generator_with_fake_client):
    generator, fake_client = generator_with_fake_client
    fake_client.completions.answer_text = ""
    chunks = [_make_chunk(1, "Some fact.", page=1)]

    answer = generator.generate_answer("a question", chunks)

    assert answer == NOT_FOUND_MESSAGE


# --- Input validation and error handling -----------------------------


def test_empty_question_raises_value_error(generator_with_fake_client):
    generator, _ = generator_with_fake_client

    with pytest.raises(ValueError):
        generator.generate_answer("", [_make_chunk(1, "fact", page=1)])


def test_whitespace_only_question_raises_value_error(generator_with_fake_client):
    generator, _ = generator_with_fake_client

    with pytest.raises(ValueError):
        generator.generate_answer("   ", [_make_chunk(1, "fact", page=1)])


def test_missing_api_key_raises_generation_error_on_use():
    generator = Generator(api_key="", model="some-model")
    chunks = [_make_chunk(1, "fact", page=1)]

    with pytest.raises(GenerationError):
        generator.generate_answer("a question", chunks)


def test_api_call_failure_is_wrapped_in_generation_error():
    generator = Generator(api_key="fake-key", model="fake-model")

    class _BrokenCompletions:
        def create(self, **kwargs):
            raise RuntimeError("simulated network failure")

    class _BrokenChat:
        completions = _BrokenCompletions()

    class _BrokenClient:
        chat = _BrokenChat()

    generator._client = _BrokenClient()
    chunks = [_make_chunk(1, "fact", page=1)]

    with pytest.raises(GenerationError):
        generator.generate_answer("a question", chunks)


def test_constructing_generator_without_api_key_does_not_raise_immediately():
    # The client is lazy -- missing key should only surface once
    # generation is actually attempted, not at construction time.
    generator = Generator(api_key="")
    assert generator._client is None


def test_get_generator_returns_singleton():
    assert get_generator() is get_generator()
