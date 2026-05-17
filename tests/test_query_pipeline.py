"""End-to-end query pipeline tests — all external services mocked."""
import os
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport

# Ensure test env vars before any app imports
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_pipeline.db"
os.environ["OPENAI_API_KEY"] = "sk-test-fake"
os.environ["QDRANT_URL"] = "http://localhost:6333"


from app.services.qdrant_client import SearchResult
from app.services.generator import GeneratorService, GenerationResult, SourceAttribution


# ── Citation parsing tests ────────────────────────────────────────────────────

class TestCitationParsing:
    """Test that GeneratorService._parse_citations only returns actually cited chunks."""

    def setup_method(self):
        self.gen = GeneratorService.__new__(GeneratorService)

    def _make_chunks(self, n: int) -> list[SearchResult]:
        return [
            SearchResult(
                chunk_id=f"chunk_{i}",
                doc_id=f"doc_{i % 2}",
                score=0.9 - i * 0.05,
                text=f"Context content for chunk {i}. " * 10,
            )
            for i in range(n)
        ]

    def test_single_citation_parsed(self):
        chunks = self._make_chunks(3)
        answer = "According to the policy [1], employees must comply."
        sources = self.gen._parse_citations(answer, chunks)
        assert len(sources) == 1
        assert sources[0].citation_number == 1
        assert sources[0].chunk_id == "chunk_0"

    def test_multiple_citations_parsed(self):
        chunks = self._make_chunks(5)
        answer = "The termination clause [1] and notice period [3] are key."
        sources = self.gen._parse_citations(answer, chunks)
        assert len(sources) == 2
        cited_numbers = {s.citation_number for s in sources}
        assert cited_numbers == {1, 3}

    def test_uncited_chunks_not_returned(self):
        """Only cited chunks are returned — not all retrieved chunks."""
        chunks = self._make_chunks(5)
        answer = "Only [2] is relevant here."
        sources = self.gen._parse_citations(answer, chunks)
        assert len(sources) == 1
        assert sources[0].citation_number == 2
        assert sources[0].chunk_id == "chunk_1"

    def test_no_citations_returns_empty(self):
        chunks = self._make_chunks(3)
        answer = "I cannot find the answer in the provided documents."
        sources = self.gen._parse_citations(answer, chunks)
        assert sources == []

    def test_out_of_range_citation_ignored(self):
        """Citation [99] when only 3 chunks exist — should be ignored."""
        chunks = self._make_chunks(3)
        answer = "See [1] and [99] for details."
        sources = self.gen._parse_citations(answer, chunks)
        # Only [1] is valid
        assert len(sources) == 1
        assert sources[0].citation_number == 1

    def test_duplicate_citations_deduplicated(self):
        """[1] cited twice in answer → only one source entry."""
        chunks = self._make_chunks(3)
        answer = "As noted in [1], the policy [1] states clearly."
        sources = self.gen._parse_citations(answer, chunks)
        assert len(sources) == 1

    def test_excerpt_truncated_at_300_chars(self):
        long_text = "A" * 500
        chunks = [SearchResult(chunk_id="c1", doc_id="d1", score=0.9, text=long_text)]
        answer = "The answer is [1]."
        sources = self.gen._parse_citations(answer, chunks)
        assert len(sources[0].excerpt) <= 303  # 300 + "..."


# ── Full pipeline tests ───────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestFullQueryPipeline:
    """End-to-end test with mocked embedder, Qdrant, and OpenAI."""

    async def _get_app(self):
        """Build app with mocked DB and services."""
        from app.main import app
        return app

    @pytest.fixture(autouse=True)
    def mock_services(self):
        """Patch all external services at module level."""
        fake_results = [
            SearchResult(chunk_id="c1", doc_id="doc1", score=0.92, text="The notice period is 30 days."),
            SearchResult(chunk_id="c2", doc_id="doc1", score=0.85, text="Employees must provide written notice."),
            SearchResult(chunk_id="c3", doc_id="doc2", score=0.80, text="Severance pay is calculated monthly."),
        ]

        # Mock OpenAI embeddings
        mock_embed_response = MagicMock()
        mock_embed_response.data = [MagicMock(embedding=[0.1] * 1536)]

        # Mock OpenAI completion
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="The notice period is 30 days [1]. Written notice is required [2]."))]
        mock_completion.usage = MagicMock(prompt_tokens=100, completion_tokens=50)

        # Patch routers.query service instances
        with patch("app.routers.query._embedder") as mock_emb, \
             patch("app.routers.query._qdrant") as mock_qd, \
             patch("app.routers.query._retrieval") as mock_ret, \
             patch("app.routers.query._generator") as mock_gen, \
             patch("app.routers.query._langfuse") as mock_lf:

            mock_emb.embed_text = AsyncMock(return_value=[0.1] * 1536)
            mock_qd.search = AsyncMock(return_value=fake_results)

            # Retrieval returns (results, confidence, low_confidence)
            mock_ret.retrieve = AsyncMock(return_value=(fake_results, 0.857, False))

            # Generator returns a GenerationResult
            gen_result = GenerationResult(
                answer="The notice period is 30 days [1]. Written notice is required [2].",
                sources=[
                    SourceAttribution(doc_id="doc1", chunk_id="c1", score=0.92, excerpt="The notice period is 30 days.", citation_number=1),
                    SourceAttribution(doc_id="doc1", chunk_id="c2", score=0.85, excerpt="Employees must provide written notice.", citation_number=2),
                ],
                confidence=0.857,
                prompt_tokens=100,
                completion_tokens=50,
                low_confidence=False,
            )
            mock_gen.generate = AsyncMock(return_value=gen_result)

            mock_lf.create_trace = MagicMock(return_value="trace-abc-123")
            mock_lf.log_retrieval = MagicMock()
            mock_lf.log_generation = MagicMock()

            yield

    async def test_query_response_structure(self, mock_services):
        """Full pipeline returns correct response shape."""
        from app.main import app
        from app.database import init_db

        await init_db()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Create org first
            org_resp = await client.post("/orgs", json={"name": "Test Corp", "slug": "test-corp-pipeline"})
            assert org_resp.status_code == 201
            org_id = org_resp.json()["id"]

            # Query
            resp = await client.post(
                f"/orgs/{org_id}/query",
                json={"query": "What is the notice period?", "top_k": 3},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "sources" in data
        assert "confidence" in data
        assert "low_confidence" in data
        assert "trace_id" in data
        assert "latency_ms" in data

    async def test_query_returns_only_cited_sources(self, mock_services):
        """Sources in response should match cited chunks, not all retrieved chunks."""
        from app.main import app
        from app.database import init_db

        await init_db()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            org_resp = await client.post("/orgs", json={"name": "Test Corp 2", "slug": "test-corp-citations"})
            org_id = org_resp.json()["id"]

            resp = await client.post(
                f"/orgs/{org_id}/query",
                json={"query": "What is the notice period?"},
            )

        data = resp.json()
        # Generator mock returns 2 cited sources (out of 3 retrieved)
        assert len(data["sources"]) == 2
        assert data["sources"][0]["citation_number"] == 1
        assert data["sources"][1]["citation_number"] == 2

    async def test_confidence_score_in_response(self, mock_services):
        """Confidence score is correctly included in response."""
        from app.main import app
        from app.database import init_db

        await init_db()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            org_resp = await client.post("/orgs", json={"name": "Conf Corp", "slug": "conf-corp-score"})
            org_id = org_resp.json()["id"]

            resp = await client.post(
                f"/orgs/{org_id}/query",
                json={"query": "What are the benefits?"},
            )

        data = resp.json()
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["low_confidence"] is False  # mock returns 0.857 > 0.75

    async def test_low_confidence_response_flag(self, mock_services):
        """When retrieval confidence is low, low_confidence=True in response."""
        from app.main import app
        from app.database import init_db
        from app.services.generator import GenerationResult

        await init_db()

        # Override retrieval to return low confidence
        with patch("app.routers.query._retrieval") as mock_ret2, \
             patch("app.routers.query._generator") as mock_gen2:

            mock_ret2.retrieve = AsyncMock(return_value=([], 0.2, True))
            mock_gen2.generate = AsyncMock(return_value=GenerationResult(
                answer="I cannot find the answer to this question in the provided documents.",
                sources=[],
                confidence=0.2,
                prompt_tokens=50,
                completion_tokens=20,
                low_confidence=True,
            ))

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                org_resp = await client.post("/orgs", json={"name": "Low Conf Org", "slug": "low-conf-org"})
                org_id = org_resp.json()["id"]

                resp = await client.post(
                    f"/orgs/{org_id}/query",
                    json={"query": "What is the meaning of life?"},
                )

        data = resp.json()
        assert data["low_confidence"] is True
        assert data["confidence"] < 0.75

    async def test_unknown_org_returns_404(self):
        """Querying a non-existent org returns 404."""
        from app.main import app
        from app.database import init_db

        await init_db()

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post(
                "/orgs/nonexistent-org-id/query",
                json={"query": "test"},
            )

        assert resp.status_code == 404
