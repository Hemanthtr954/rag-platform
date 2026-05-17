import logging
from app.config import get_settings
from app.services.embedder import EmbedderService
from app.services.qdrant_client import QdrantService, SearchResult

logger = logging.getLogger(__name__)


class RetrievalService:
    """Orchestrates embedding + vector search with confidence scoring."""

    def __init__(self, embedder: EmbedderService, qdrant: QdrantService) -> None:
        self._embedder = embedder
        self._qdrant = qdrant
        self._settings = get_settings()

    async def retrieve(
        self,
        org_id: str,
        query: str,
        top_k: int | None = None,
    ) -> tuple[list[SearchResult], float, bool]:
        """
        Retrieve relevant chunks for a query within an org's collection.

        Returns:
            (results, confidence_score, low_confidence)
            - results: ranked list of SearchResult
            - confidence_score: mean of top-3 scores (0-1)
            - low_confidence: True if confidence < threshold
        """
        k = top_k or self._settings.top_k
        threshold = self._settings.confidence_threshold

        # Embed the query
        query_embedding = await self._embedder.embed_text(query)

        # Search within org's isolated collection
        results = await self._qdrant.search(
            org_id=org_id,
            query_embedding=query_embedding,
            top_k=k,
            score_threshold=0.0,  # return all, we compute confidence ourselves
        )

        # Confidence = mean of top-3 scores
        top_scores = [r.score for r in results[:3]]
        if top_scores:
            confidence = sum(top_scores) / len(top_scores)
        else:
            confidence = 0.0

        low_confidence = confidence < threshold

        if low_confidence:
            logger.warning(
                "Low confidence retrieval for org=%s query='%s' confidence=%.3f",
                org_id,
                query[:50],
                confidence,
            )

        return results, confidence, low_confidence
