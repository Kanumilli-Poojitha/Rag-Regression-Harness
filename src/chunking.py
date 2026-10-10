"""Fixed-size word chunker (chunker_version = fixed-words-v1). Intentionally naive."""
from typing import List

CHUNKER_VERSION = "fixed-words-v1"


def chunk_words(text: str, size: int, overlap: int = 0) -> List[str]:
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks
