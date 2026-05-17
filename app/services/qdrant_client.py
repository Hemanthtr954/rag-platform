import logging
from dataclasses import dataclass
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from app.config import get_settings

logger = logging.getLogger(__name__)

VECTOR_SIZE = 1536  # text-embedding-3-small dimension


@dataclass
class SearchResult:
    chunk_id: str
    doc_id: str
    score: float
    text: str


class QdrantService:
    """Async Qdrant wrapper with per-org collection isolation."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncQdrantClient(url=settings.qdrant_url)

    def _collection_name(self, org_id: str) -> str:
        """Each org gets its own collection — complete isolation."""
        return f"org_{org_id}"

    async def ensure_collection(self, org_id: str) -> None:
        """Create org collection if it doesn't exist."""
        name = self._collection_name(org_id)
        exists = await self._client.collection_exists(name)
        if not exists:
            await self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection: %s", name)

    async def upsert_chunks(
        self,
        org_id: str,
        doc_id: str,
        chunks_with_embeddings: list[tuple[str, str, list[float]]],
        # Each tuple: (chunk_id, chunk_text, embedding)
    ) -> None:
        """Upsert chunk vectors into the org collection."""
        collection = self._collection_name(org_id)
        points = [
            PointStruct(
                id=abs(hash(chunk_id)) % (2**63),  # Qdrant needs uint64
                vector=embedding,
                payload={
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "org_id": org_id,
                    "text": text,
                },
            )
            for chunk_id, text, embedding in chunks_with_embeddings
        ]
        if points:
            await self._client.upsert(collection_name=collection, points=points)
            logger.info("Upserted %d chunks for doc %s in org %s", len(points), doc_id, org_id)

    async def search(
        self,
        org_id: str,
        query_embedding: list[float],
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[SearchResult]:
        """Search within the org's collection. Cannot cross org boundaries."""
        collection = self._collection_name(org_id)
        try:
            results = await self._client.search(
                collection_name=collection,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )
        except Exception as e:
            logger.error("Qdrant search error for org %s: %s", org_id, e)
            return []

        return [
            SearchResult(
                chunk_id=hit.payload["chunk_id"],
                doc_id=hit.payload["doc_id"],
                score=hit.score,
                text=hit.payload["text"],
            )
            for hit in results
        ]

    async def delete_document(self, org_id: str, doc_id: str) -> None:
        """Delete all vectors belonging to a specific document in the org's collection."""
        collection = self._collection_name(org_id)
        await self._client.delete(
            collection_name=collection,
            points_selector=Filter(
                must=[
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                ]
            ),
        )
        logger.info("Deleted all chunks for doc %s in org %s", doc_id, org_id)

    async def close(self) -> None:
        await self._client.close()
