"""Tests for RetrievalService — mocked Qdrant and embedder."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.services.retriever import RetrievalService
from app.services.qdrant_client import SearchResult


def make_search_result(chunk_id: str, doc_id: str, score: float) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        score=score,
        text=f"Text for {chunk_id}",
    )


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed_text = AsyncMock(return_value=[0.1] * 1536)
    return embedder


@pytest.fixture
def mock_qdrant():
    qdrant = MagicMock()
    return qdrant


@pytest.fixture
def retrieval_service(mock_embedder, mock_qdrant):
    return RetrievalService(mock_embedder, mock_qdrant)


class TestConfidenceScoreCalculation:
    @pytest.mark.asyncio
    async def test_confidence_is_mean_of_top_3(self, retrieval_service, mock_qdrant):
        """Confidence = mean of top-3 result scores."""
        results = [
            make_search_result("c1", "d1", 0.9),
            make_search_result("c2", "d1", 0.8),
            make_search_result("c3", "d1", 0.7),
            make_search_result("c4", "d1", 0.1),  # 4th result not included in confidence
        ]
        mock_qdrant.search = AsyncMock(return_value=results)

        _, confidence, _ = await retrieval_service.retrieve("org1", "test query")

        expected = (0.9 + 0.8 + 0.7) / 3
        assert abs(confidence - expected) < 1e-6

    @pytest.mark.asyncio
    async def test_confidence_single_result(self, retrieval_service, mock_qdrant):
        """With only 1 result, confidence = that result's score."""
        results = [make_search_result("c1", "d1", 0.65)]
        mock_qdrant.search = AsyncMock(return_value=results)

        _, confidence, _ = await retrieval_service.retrieve("org1", "test query")

        assert abs(confidence - 0.65) < 1e-6

    @pytest.mark.asyncio
    async def test_confidence_two_results(self, retrieval_service, mock_qdrant):
        """With 2 results, confidence = mean of those 2."""
        results = [
            make_search_result("c1", "d1", 0.8),
            make_search_result("c2", "d1", 0.6),
        ]
        mock_qdrant.search = AsyncMock(return_value=results)

        _, confidence, _ = await retrieval_service.retrieve("org1", "test query")

        expected = (0.8 + 0.6) / 2
        assert abs(confidence - expected) < 1e-6

    @pytest.mark.asyncio
    async def test_zero_confidence_on_empty_results(self, retrieval_service, mock_qdrant):
        """No results → confidence = 0.0."""
        mock_qdrant.search = AsyncMock(return_value=[])

        _, confidence, low_confidence = await retrieval_service.retrieve("org1", "test query")

        assert confidence == 0.0
        assert low_confidence is True


class TestLowConfidenceFlag:
    @pytest.mark.asyncio
    async def test_low_confidence_true_when_below_threshold(self, retrieval_service, mock_qdrant):
        """low_confidence=True when mean top-3 score < 0.75 (default threshold)."""
        results = [
            make_search_result("c1", "d1", 0.5),
            make_search_result("c2", "d1", 0.4),
            make_search_result("c3", "d1", 0.3),
        ]
        mock_qdrant.search = AsyncMock(return_value=results)

        _, confidence, low_confidence = await retrieval_service.retrieve("org1", "low confidence query")

        assert confidence < 0.75
        assert low_confidence is True

    @pytest.mark.asyncio
    async def test_low_confidence_false_when_above_threshold(self, retrieval_service, mock_qdrant):
        """low_confidence=False when confidence >= 0.75."""
        results = [
            make_search_result("c1", "d1", 0.95),
            make_search_result("c2", "d1", 0.90),
            make_search_result("c3", "d1", 0.85),
        ]
        mock_qdrant.search = AsyncMock(return_value=results)

        _, confidence, low_confidence = await retrieval_service.retrieve("org1", "high confidence query")

        assert confidence >= 0.75
        assert low_confidence is False

    @pytest.mark.asyncio
    async def test_low_confidence_exactly_at_threshold(self, retrieval_service, mock_qdrant):
        """Exactly at threshold (0.75) → low_confidence=False."""
        results = [
            make_search_result("c1", "d1", 0.75),
            make_search_result("c2", "d1", 0.75),
            make_search_result("c3", "d1", 0.75),
        ]
        mock_qdrant.search = AsyncMock(return_value=results)

        _, confidence, low_confidence = await retrieval_service.retrieve("org1", "threshold query")

        assert abs(confidence - 0.75) < 1e-6
        assert low_confidence is False  # threshold is < not <=

    @pytest.mark.asyncio
    async def test_org_isolation_in_search_call(self, retrieval_service, mock_qdrant):
        """Qdrant search is called with the correct org_id — never crosses org boundaries."""
        mock_qdrant.search = AsyncMock(return_value=[])

        await retrieval_service.retrieve("org-alpha", "test query", top_k=3)

        mock_qdrant.search.assert_called_once()
        call_kwargs = mock_qdrant.search.call_args
        # org_id must match exactly
        assert call_kwargs.kwargs.get("org_id") == "org-alpha" or call_kwargs.args[0] == "org-alpha"

    @pytest.mark.asyncio
    async def test_embedder_called_with_query(self, retrieval_service, mock_qdrant, mock_embedder):
        """Query text is embedded before search."""
        mock_qdrant.search = AsyncMock(return_value=[])

        await retrieval_service.retrieve("org1", "what is the policy?")

        mock_embedder.embed_text.assert_called_once_with("what is the policy?")
