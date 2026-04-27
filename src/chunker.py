"""
Document Chunker
Splits scraped content into embedding-friendly chunks for RAG pipeline.

Strategy:
- Split by paragraphs (blank line separated)
- Each paragraph becomes a chunk
- If paragraph > 500 tokens, split into multiple ~500 token chunks by words
- Preserve source URL for citations
"""

from pathlib import Path
import tiktoken
import json
import logging

logger = logging.getLogger(__name__)

TARGET_TOKENS = 1000  # Target chunk size


class DocumentChunker:
    """Splits documents into chunks suitable for embedding."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoder = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoder.encode(text))

    def split_into_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs based on blank lines."""
        paragraphs = []
        current = []

        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                if current:
                    paragraphs.append("\n".join(current))
                    current = []
            else:
                current.append(stripped)

        if current:
            paragraphs.append("\n".join(current))

        return [p for p in paragraphs if p.strip()]

    def create_chunk(self, text: str, source_url: str, chunk_id: int) -> dict:
        """Create a chunk dict with text and metadata."""
        return {
            "id": f"chunk_{chunk_id}",
            "text": text,
            "source": source_url,
            "token_count": self.count_tokens(text)
        }

    def chunk_document(self, url: str, content: str, start_chunk_id: int = 0) -> tuple[list[dict], int]:
        """
        Split a document into chunks.

        Simple algorithm:
        1. Split by paragraphs
        2. Each paragraph = one chunk (if < 500 tokens)
        3. Large paragraphs split by words into ~500 token chunks
        """
        chunks = []
        chunk_id = start_chunk_id
        paragraphs = self.split_into_paragraphs(content)

        for para in paragraphs:
            if not para.strip():
                continue

            para_tokens = self.count_tokens(para)

            if para_tokens <= TARGET_TOKENS:
                # Small paragraph - add as one chunk
                chunks.append(self.create_chunk(para, url, chunk_id))
                chunk_id += 1
            else:
                # Large paragraph - split by words into ~500 token chunks
                words = para.split()
                # Rule of thumb: 1 token ≈ 0.75 words in English
                # So 500 tokens ≈ 375-400 words, use 400 to be safe
                words_per_chunk = int(TARGET_TOKENS * 0.75)

                for i in range(0, len(words), words_per_chunk):
                    chunk_text = " ".join(words[i:i + words_per_chunk])
                    chunks.append(self.create_chunk(chunk_text, url, chunk_id))
                    chunk_id += 1

        return chunks, chunk_id

    def chunk_all(self, scraped_content: dict[str, str]) -> list[dict]:
        """Chunk all scraped documents."""
        all_chunks = []
        global_chunk_id = 0

        for url, content in scraped_content.items():
            logger.info(f"Chunking: {url}")
            chunks, global_chunk_id = self.chunk_document(url, content, global_chunk_id)
            logger.info(f"  {len(chunks)} chunks created")
            all_chunks.extend(chunks)

        logger.info(f"Total chunks: {len(all_chunks)}")
        return all_chunks


if __name__ == "__main__":
    from src.scraper import load_content

    logging.basicConfig(level=logging.INFO)

    content = load_content(Path("data/scraped_content_final.json"))
    if not content:
        print("No content found. Run scraper first.")
        exit(1)

    chunker = DocumentChunker()
    chunks = chunker.chunk_all(content)

    chunks_file = Path("data/chunks.json")
    with open(chunks_file, "w") as f:
        json.dump(chunks, f, indent=2)

    logger.info(f"Saved {len(chunks)} chunks to {chunks_file}")

    # Show token distribution
    token_counts = [c["token_count"] for c in chunks]
    print(f"\nToken distribution:")
    print(f"  Min: {min(token_counts)}, Max: {max(token_counts)}, Avg: {sum(token_counts) // len(token_counts)}")