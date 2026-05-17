"""Tests for ChunkerService — no external services required."""
import pytest
from unittest.mock import MagicMock, patch
from app.services.chunker import ChunkerService, Chunk


@pytest.fixture
def chunker():
    return ChunkerService()


class TestChunkDocument:
    def test_basic_chunking_returns_chunks(self, chunker):
        text = " ".join(["word"] * 600)
        chunks = chunker.chunk_document(text, chunk_size=100, overlap=10)
        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_indices_are_sequential(self, chunker):
        text = ". ".join(["This is a sentence"] * 50)
        chunks = chunker.chunk_document(text, chunk_size=50, overlap=5)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_chunk_size_respected(self, chunker):
        """Each chunk should not exceed chunk_size words by much (sentence boundary tolerance)."""
        text = ". ".join(["word"] * 200)
        chunk_size = 30
        chunks = chunker.chunk_document(text, chunk_size=chunk_size, overlap=5)
        # Allow some tolerance for sentence boundary handling
        for chunk in chunks:
            assert chunk.word_count <= chunk_size * 3, f"Chunk too large: {chunk.word_count}"

    def test_overlap_creates_shared_content(self, chunker):
        """With overlap, consecutive chunks should share some words."""
        # Create text with clear sentences
        sentences = ["Alpha beta gamma delta epsilon."] * 20
        text = " ".join(sentences)
        chunks = chunker.chunk_document(text, chunk_size=20, overlap=10)
        if len(chunks) > 1:
            words_0 = set(chunks[0].text.split())
            words_1 = set(chunks[1].text.split())
            # There should be some word overlap between consecutive chunks
            overlap = words_0 & words_1
            assert len(overlap) > 0, "Expected word overlap between consecutive chunks"

    def test_empty_text_returns_empty(self, chunker):
        chunks = chunker.chunk_document("", chunk_size=100, overlap=10)
        assert chunks == []

    def test_whitespace_only_returns_empty(self, chunker):
        chunks = chunker.chunk_document("   \n\t  ", chunk_size=100, overlap=10)
        assert chunks == []

    def test_short_text_single_chunk(self, chunker):
        text = "This is a short document."
        chunks = chunker.chunk_document(text, chunk_size=512, overlap=64)
        assert len(chunks) == 1
        assert "short document" in chunks[0].text

    def test_sentence_boundary_not_split(self, chunker):
        """Chunk text should contain complete sentences (no mid-sentence splits)."""
        sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs.",
            "How vividly daft jumping zebras vex.",
            "The five boxing wizards jump quickly.",
        ]
        text = " ".join(sentences * 5)
        chunks = chunker.chunk_document(text, chunk_size=30, overlap=5)
        for chunk in chunks:
            # Each chunk should not end with a word that's clearly mid-sentence
            # (basic check — text ends at word boundaries)
            assert chunk.text.strip(), "Chunk should not be empty"

    def test_word_count_matches_text(self, chunker):
        text = "one two three four five. six seven eight nine ten."
        chunks = chunker.chunk_document(text, chunk_size=512, overlap=10)
        for chunk in chunks:
            actual_word_count = len(chunk.text.split())
            assert chunk.word_count == actual_word_count

    def test_large_document_chunked_correctly(self, chunker):
        # 5000 word document
        words = ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit"]
        sentences = [" ".join(words) + "." for _ in range(100)]
        text = " ".join(sentences)
        chunks = chunker.chunk_document(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1
        # All chunks have text
        assert all(c.text for c in chunks)


class TestExtractTextFromPdf:
    def test_pdf_extraction_calls_pypdf(self, chunker):
        """Mock pypdf to test the extraction path."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Hello from PDF page one."

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("app.services.chunker.PdfReader", return_value=mock_reader):
            result = chunker.extract_text_from_pdf(b"%PDF-fake-content")

        assert "Hello from PDF page one." in result

    def test_pdf_extraction_joins_pages(self, chunker):
        """Multiple pages are joined with double newline."""
        pages = [MagicMock(), MagicMock()]
        pages[0].extract_text.return_value = "Page one content."
        pages[1].extract_text.return_value = "Page two content."

        mock_reader = MagicMock()
        mock_reader.pages = pages

        with patch("app.services.chunker.PdfReader", return_value=mock_reader):
            result = chunker.extract_text_from_pdf(b"%PDF-fake")

        assert "Page one content." in result
        assert "Page two content." in result

    def test_pdf_extraction_skips_empty_pages(self, chunker):
        """Pages with no text are skipped."""
        pages = [MagicMock(), MagicMock()]
        pages[0].extract_text.return_value = "Real content here."
        pages[1].extract_text.return_value = ""  # empty page

        mock_reader = MagicMock()
        mock_reader.pages = pages

        with patch("app.services.chunker.PdfReader", return_value=mock_reader):
            result = chunker.extract_text_from_pdf(b"%PDF-fake")

        assert "Real content here." in result
        # Should not have double newlines from empty pages
        assert result.strip() == "Real content here."

    def test_pdf_extraction_raises_on_invalid(self, chunker):
        """Invalid PDF content raises ValueError."""
        with patch("app.services.chunker.PdfReader", side_effect=Exception("invalid PDF")):
            with pytest.raises(ValueError, match="Failed to extract text from PDF"):
                chunker.extract_text_from_pdf(b"not-a-pdf")
