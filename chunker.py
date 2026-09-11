"""
Chunker Module for RAG Document Intelligence.
Splits document pages into semantically cohesive, overlapping chunks with metadata.
"""

import re
from typing import List, Dict, Any

class DocumentChunker:
    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 50):
        """
        chunk_size: approximate words per chunk
        chunk_overlap: approximate overlapping words between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunks a list of extracted pages while preserving page numbers and source metadata.
        Returns list of chunks:
        [
            {
                "chunk_id": "chunk_001",
                "text": str,
                "page": int,
                "source": str,
                "word_count": int,
                "char_count": int,
                "token_estimate": int
            }
        ]
        """
        all_chunks = []
        global_idx = 1

        for page_data in pages:
            page_num = page_data.get("page", 1)
            source = page_data.get("source", "document.pdf")
            text = page_data.get("text", "")

            # Split into sentences or paragraphs first
            sentences = self._split_into_sentences(text)
            if not sentences:
                continue

            current_chunk_words = []
            current_count = 0

            for sent in sentences:
                words = sent.split()
                if not words:
                    continue

                if current_count + len(words) > self.chunk_size and current_chunk_words:
                    # Emit current chunk
                    chunk_text = " ".join(current_chunk_words)
                    all_chunks.append({
                        "chunk_id": f"chunk_{global_idx:03d}",
                        "text": chunk_text,
                        "page": page_num,
                        "source": source,
                        "word_count": len(current_chunk_words),
                        "char_count": len(chunk_text),
                        "token_estimate": int(len(current_chunk_words) * 1.3)
                    })
                    global_idx += 1

                    # Retain overlap
                    overlap_words = current_chunk_words[-self.chunk_overlap:] if len(current_chunk_words) > self.chunk_overlap else current_chunk_words
                    current_chunk_words = list(overlap_words)
                    current_count = len(current_chunk_words)

                current_chunk_words.extend(words)
                current_count += len(words)

            if current_chunk_words:
                chunk_text = " ".join(current_chunk_words)
                all_chunks.append({
                    "chunk_id": f"chunk_{global_idx:03d}",
                    "text": chunk_text,
                    "page": page_num,
                    "source": source,
                    "word_count": len(current_chunk_words),
                    "char_count": len(chunk_text),
                    "token_estimate": int(len(current_chunk_words) * 1.3)
                })
                global_idx += 1

        return all_chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        # Clean and split on punctuation boundaries
        clean = re.sub(r"\n+", " ", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+", clean)
        return [s.strip() for s in sentences if s.strip()]
