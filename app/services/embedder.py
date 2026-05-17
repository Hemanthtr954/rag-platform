import hashlib
import logging
from openai import AsyncOpenAI
from app.config import get_settings
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class EmbedderService:
    """Embed text using OpenAI embeddings with in-process caching."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = AsyncOpenAI(api_key=self._settings.openai_api_key)
        self._cache: dict[str, list[float]] = {}

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string. Returns cached result if available."""
        key = self._cache_key(text)
        if key in self._cache:
            logger.debug("Embedding cache hit for key %s", key[:8])
            return self._cache[key]

        response = await self._client.embeddings.create(
            input=text,
            model=self._settings.embedding_model,
        )
        embedding = response.data[0].embedding
        self._cache[key] = embedding
        return embedding

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Uses cache for already-embedded texts."""
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            response = await self._client.embeddings.create(
                input=uncached_texts,
                model=self._settings.embedding_model,
            )
            for j, item in enumerate(response.data):
                idx = uncached_indices[j]
                embedding = item.embedding
                results[idx] = embedding
                self._cache[self._cache_key(texts[idx])] = embedding

        return [r for r in results if r is not None]
