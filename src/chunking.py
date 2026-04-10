from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        
        # Split on sentence boundaries: ". ", "! ", "? ", or ".\n"
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        
        for sentence in sentences:
            current_chunk.append(sentence)
            if len(current_chunk) >= self.max_sentences_per_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text:
            return []
        
        # If text is within chunk_size, return it as is
        if len(current_text) <= self.chunk_size:
            return [current_text]
        
        # If no separators left, return the text as is (forced)
        if not remaining_separators:
            return [current_text]
        
        # Try the first separator
        separator = remaining_separators[0]
        
        if separator in current_text:
            # Split by this separator
            pieces = current_text.split(separator)
            good_chunks = []
            bad_chunks = []
            
            for piece in pieces:
                if not piece:
                    continue
                if len(piece) <= self.chunk_size:
                    good_chunks.append(piece)
                else:
                    bad_chunks.append(piece)
            
            # Recursively split the bad chunks with next separator
            if remaining_separators[1:]:
                split_bad_chunks = []
                for bad_chunk in bad_chunks:
                    split_bad_chunks.extend(self._split(bad_chunk, remaining_separators[1:]))
                return good_chunks + split_bad_chunks
            else:
                # No more separators, return both good and bad
                return good_chunks + bad_chunks
        else:
            # Separator not found, try next separator
            return self._split(current_text, remaining_separators[1:] if remaining_separators[1:] else [])


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    dot_product = _dot(vec_a, vec_b)
    
    # Compute magnitudes
    mag_a = math.sqrt(sum(x * x for x in vec_a))
    mag_b = math.sqrt(sum(x * x for x in vec_b))
    
    # Handle zero magnitude case
    if mag_a == 0 or mag_b == 0:
        return 0.0
    
    return dot_product / (mag_a * mag_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        fixed_chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=0)
        sentence_chunker = SentenceChunker(max_sentences_per_chunk=3)
        recursive_chunker = RecursiveChunker(chunk_size=chunk_size)
        
        # Run each chunker
        fixed_chunks = fixed_chunker.chunk(text)
        sentence_chunks = sentence_chunker.chunk(text)
        recursive_chunks = recursive_chunker.chunk(text)
        
        # Helper to compute stats
        def compute_stats(chunks: list[str]) -> dict:
            if not chunks:
                return {"count": 0, "avg_length": 0.0, "chunks": []}
            return {
                "count": len(chunks),
                "avg_length": sum(len(c) for c in chunks) / len(chunks),
                "chunks": chunks,
            }
        
        return {
            "fixed_size": compute_stats(fixed_chunks),
            "by_sentences": compute_stats(sentence_chunks),
            "recursive": compute_stats(recursive_chunks),
        }
