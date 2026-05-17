import re
import logging
from dataclasses import dataclass, field
from openai import AsyncOpenAI
from app.config import get_settings
from app.services.qdrant_client import SearchResult

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r'\[(\d+)\]')


@dataclass
class SourceAttribution:
    doc_id: str
    chunk_id: str
    score: float
    excerpt: str
    citation_number: int


@dataclass
class GenerationResult:
    answer: str
    sources: list[SourceAttribution] = field(default_factory=list)
    confidence: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    low_confidence: bool = False


class GeneratorService:
    """Generate answers from retrieved context with citation tracking."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = AsyncOpenAI(api_key=self._settings.openai_api_key)

    def _build_prompt(self, query: str, chunks: list[SearchResult]) -> tuple[str, str]:
        """Build system + user prompt with numbered context chunks."""
        system_prompt = (
            "You are a precise question-answering assistant. "
            "Answer the user's question using ONLY the provided context below. "
            "If the answer is not in the context, say explicitly: "
            "'I cannot find the answer to this question in the provided documents.' "
            "Cite your sources inline using [1], [2], etc. "
            "Only cite sources you actually use in your answer."
        )

        context_parts = []
        for i, chunk in enumerate(chunks, start=1):
            context_parts.append(f"[{i}] {chunk.text}")

        context_block = "\n\n".join(context_parts)
        user_prompt = f"Context:\n{context_block}\n\nQuestion: {query}"

        return system_prompt, user_prompt

    def _parse_citations(
        self, answer: str, chunks: list[SearchResult]
    ) -> list[SourceAttribution]:
        """
        Parse [N] citations from the answer.
        Only return sources that were actually cited — not all retrieved chunks.
        This is the key trust signal.
        """
        cited_numbers: set[int] = set()
        for match in _CITATION_RE.finditer(answer):
            n = int(match.group(1))
            cited_numbers.add(n)

        sources: list[SourceAttribution] = []
        for n in sorted(cited_numbers):
            idx = n - 1  # citations are 1-indexed
            if 0 <= idx < len(chunks):
                chunk = chunks[idx]
                sources.append(
                    SourceAttribution(
                        doc_id=chunk.doc_id,
                        chunk_id=chunk.chunk_id,
                        score=chunk.score,
                        excerpt=chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""),
                        citation_number=n,
                    )
                )
        return sources

    async def generate(
        self,
        query: str,
        context_chunks: list[SearchResult],
        confidence: float = 0.0,
        low_confidence: bool = False,
    ) -> GenerationResult:
        """Generate an answer with source attribution."""
        if not context_chunks:
            return GenerationResult(
                answer="I cannot find the answer to this question in the provided documents.",
                sources=[],
                confidence=0.0,
                low_confidence=True,
            )

        system_prompt, user_prompt = self._build_prompt(query, context_chunks)

        try:
            response = await self._client.chat.completions.create(
                model=self._settings.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,  # low temperature for factual answers
                max_tokens=1024,
            )
        except Exception as e:
            logger.error("OpenAI generation failed: %s", e)
            raise

        answer = response.choices[0].message.content or ""
        usage = response.usage

        sources = self._parse_citations(answer, context_chunks)

        return GenerationResult(
            answer=answer,
            sources=sources,
            confidence=confidence,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            low_confidence=low_confidence,
        )
