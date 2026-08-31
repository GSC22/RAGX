"""
Generation service.

Takes a user's question and the chunks the retriever found for it,
builds a strict "answer only from this context" prompt, and calls
Groq to produce the final answer.

This is the ONLY place in the whole pipeline that talks to an LLM.
Everything upstream (loading, chunking, embedding, retrieval) is
deterministic, hand-written logic -- this file is deliberately the
single, well-defined boundary where an external AI model's judgment
enters the system, which makes the "where could hallucination sneak
in?" question easy to answer in an interview: only here, and only
constrained to the text we explicitly hand it.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from app.config import settings
from app.services.retriever import RetrievedChunk

NOT_FOUND_MESSAGE = (
    "I couldn't find enough information in the provided document to answer that question."
)

# Kept as one readable block rather than scattered f-strings -- an
# interviewer should be able to read this top to bottom and understand
# the entire grounding strategy in under a minute.
SYSTEM_PROMPT = """You are a careful, honest document assistant. You answer questions using ONLY the DOCUMENT CONTEXT provided in the user's message.

Rules you must always follow:
1. Use only information explicitly present in the DOCUMENT CONTEXT below. Never use outside knowledge, even if you are confident it is correct.
2. Do not invent, guess, or assume any fact, number, name, or date that is not stated in the context.
3. If the DOCUMENT CONTEXT does not contain enough information to answer the question, respond with exactly: "I couldn't find enough information in the provided document to answer that question." Do not attempt a partial or speculative answer instead.
4. Keep answers concise and directly useful. Do not pad with unnecessary caveats or repeat the question back.
5. When it is helpful, refer to the page number(s) the information came from, using the source labels given in the context.
6. Never claim to have read, seen, or know anything beyond what is present in the DOCUMENT CONTEXT given to you in this message."""

USER_PROMPT_TEMPLATE = """DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}"""


class GenerationError(Exception):
    """Raised when the LLM call itself fails (bad/missing API key,
    network error, Groq-side error) -- distinct from a ValueError,
    which means the caller passed bad input."""


class Generator:
    """Builds grounded prompts and calls Groq to answer questions."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        self.api_key = api_key or settings.groq_api_key
        self.model = model or settings.groq_model
        self._client = None  # lazily constructed on first real use

    @property
    def client(self):
        """Lazily construct the Groq client.

        Deferred to first use (rather than in __init__) so simply
        creating a Generator -- e.g. in a test that never calls the
        real API -- doesn't require a valid API key to be set.
        """
        if self._client is None:
            if not self.api_key:
                raise GenerationError(
                    "GROQ_API_KEY is not set. Add it to your .env file "
                    "(see .env.example) before asking questions."
                )
            from groq import Groq

            self._client = Groq(api_key=self.api_key)
        return self._client

    def build_prompt(self, question: str, chunks: List[RetrievedChunk]) -> tuple[str, str]:
        """Build the (system_prompt, user_prompt) pair sent to the LLM.

        Exposed as its own method (not inlined into generate_answer) so
        the exact prompt can be inspected/tested without needing to
        make a real API call -- useful both for tests and for live
        debugging during an interview.
        """
        context_block = self._build_context_block(chunks)
        user_prompt = USER_PROMPT_TEMPLATE.format(context=context_block, question=question)
        return SYSTEM_PROMPT, user_prompt

    @staticmethod
    def _build_context_block(chunks: List[RetrievedChunk]) -> str:
        """Render retrieved chunks into labeled context blocks.

        Each chunk is tagged with its real source filename, page
        number, and similarity score -- these labels are what let the
        LLM (and the prompt's rule #5) refer back to a specific page,
        and what lets a human verify no page number was fabricated
        (it always traces back to one of these labels).
        """
        if not chunks:
            return "(no relevant content was found in the document)"

        blocks = [
            f"[Source: {chunk.source_filename} | Page {chunk.page_number} | "
            f"similarity {chunk.similarity:.2f}]\n{chunk.text}"
            for chunk in chunks
        ]
        return "\n\n---\n\n".join(blocks)

    def generate_answer(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        temperature: float = 0.1,
        max_tokens: int = 600,
        timeout_seconds: float = 30.0,
    ) -> str:
        """Generate a grounded answer to `question` using `chunks` as context.

        temperature defaults low (0.1, not 0.7+) deliberately: this is
        a fact-retrieval task, not a creative one, so we want the
        model's most likely/consistent completion, not variety.

        timeout_seconds bounds how long we'll wait on Groq before
        giving up -- without it, a hung upstream call would hang the
        request indefinitely, tying up server resources for every
        caller waiting on a response that may never come.

        Raises:
            ValueError: for an empty question (bad input from caller).
            GenerationError: for a missing API key, a timeout, or a
                failed API call.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        if not chunks:
            # Nothing was retrieved at all -- answering honestly here in
            # our own code is more reliable (and free) than trusting an
            # LLM call to correctly say "I don't know" with no context.
            return NOT_FOUND_MESSAGE

        system_prompt, user_prompt = self.build_prompt(question, chunks)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout_seconds,
            )
        except Exception as exc:  # network error, auth error, timeout, Groq-side error, etc.
            raise GenerationError(f"Failed to generate an answer via Groq: {exc}") from exc

        answer = response.choices[0].message.content
        return answer.strip() if answer else NOT_FOUND_MESSAGE


@lru_cache(maxsize=1)
def get_generator() -> Generator:
    """Process-wide singleton, used as a FastAPI dependency in later phases."""
    return Generator()
