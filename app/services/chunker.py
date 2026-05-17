import re
import logging
from dataclasses import dataclass
from io import BytesIO

logger = logging.getLogger(__name__)

# Sentence-ending punctuation pattern
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


@dataclass
class Chunk:
    text: str
    index: int
    word_count: int


class ChunkerService:
    """Word-based sliding window chunker that respects sentence boundaries."""

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences preserving structure."""
        # Normalize whitespace
        text = re.sub(r'\n+', '\n', text.strip())
        text = re.sub(r'[ \t]+', ' ', text)

        # Split on sentence boundaries
        raw_sentences = _SENTENCE_END.split(text)
        sentences: list[str] = []
        for s in raw_sentences:
            s = s.strip()
            if s:
                sentences.append(s)
        return sentences

    def chunk_document(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 64,
    ) -> list[Chunk]:
        """
        Chunk a document using a word-based sliding window.
        Respects sentence boundaries — never splits mid-sentence.
        """
        if not text.strip():
            return []

        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        # Build word lists per sentence for efficient counting
        sentence_words: list[list[str]] = [s.split() for s in sentences]

        chunks: list[Chunk] = []
        chunk_index = 0
        sentence_start = 0  # index into sentences[]

        while sentence_start < len(sentences):
            # Accumulate sentences until we reach chunk_size words
            current_words: list[str] = []
            current_sentences: list[str] = []
            i = sentence_start

            while i < len(sentences):
                candidate_words = sentence_words[i]
                if (
                    current_words
                    and len(current_words) + len(candidate_words) > chunk_size
                ):
                    break
                current_words.extend(candidate_words)
                current_sentences.append(sentences[i])
                i += 1

            if not current_words:
                # Single sentence is larger than chunk_size — split by words
                words = sentence_words[sentence_start]
                chunk_text = " ".join(words[:chunk_size])
                chunks.append(Chunk(text=chunk_text, index=chunk_index, word_count=len(words[:chunk_size])))
                chunk_index += 1
                sentence_start += 1
                continue

            chunk_text = " ".join(current_sentences)
            chunks.append(
                Chunk(text=chunk_text, index=chunk_index, word_count=len(current_words))
            )
            chunk_index += 1

            # Move forward by (chunk_size - overlap) words to find new sentence_start
            target_words_to_skip = max(1, chunk_size - overlap)
            skipped = 0
            new_start = sentence_start
            while new_start < i and skipped < target_words_to_skip:
                skipped += len(sentence_words[new_start])
                new_start += 1

            if new_start == sentence_start:
                new_start = sentence_start + 1  # always make progress

            sentence_start = new_start

        return chunks

    def extract_text_from_pdf(self, content: bytes) -> str:
        """Extract plain text from PDF bytes using pypdf."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(BytesIO(content))
            parts: list[str] = []
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        parts.append(page_text)
                except Exception as e:
                    logger.warning("Failed to extract text from page %d: %s", page_num, e)

            return "\n\n".join(parts)
        except Exception as e:
            logger.error("PDF extraction failed: %s", e)
            raise ValueError(f"Failed to extract text from PDF: {e}") from e
