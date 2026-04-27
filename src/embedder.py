"""
Embedder
Generates embeddings for chunks and stores in ChromaDB for semantic search.

Also handles:
- Query embedding and retrieval
- RAG pipeline: retrieve + generate answer
"""

import os
from pathlib import Path
from typing import Optional
import json
import logging

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

# Config
CHROMA_PATH = Path("data/chromadb")
COLLECTION_NAME = "gitlab_handbook"

# Gemini client
client = genai.Client(api_key=os.getenv("GOOGLE_STUDIO_API_KEY"))


class Embedder:
    """Handles embedding generation, storage, and retrieval."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Local embedding model (fast, good quality)
        self.embedding_model = SentenceTransformer(model_name)

        # ChromaDB client (persistent storage)
        self.chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "GitLab Handbook chunks"}
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()

    def add_chunks(self, chunks: list[dict]):
        """
        Add chunks to ChromaDB collection.

        Each chunk dict should have:
        - id: unique identifier (e.g., "chunk_0")
        - text: the text content
        - source: source URL
        - token_count: number of tokens
        """
        logger.info(f"Adding {len(chunks)} chunks to ChromaDB...")

        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [{"source": c["source"], "tokens": c["token_count"]} for c in chunks]

        # Generate embeddings
        embeddings = self.embed_texts(texts)

        # Add to collection
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        logger.info(f"Added {len(chunks)} chunks to collection")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search for most relevant chunks to a query.

        Returns list of dicts with text, source, distance, etc.
        """
        # Embed query
        query_embedding = self.embed_texts([query])[0]

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", ""),
                "distance": results["distances"][0][i],
            })

        return hits

    def load_chunks(self, chunks_file: Path = Path("data/chunks.json")) -> list[dict]:
        """Load chunks from JSON file."""
        with open(chunks_file) as f:
            chunks = json.load(f)
        logger.info(f"Loaded {len(chunks)} chunks from {chunks_file}")
        return chunks

    def index_chunks(self, chunks_file: Path = Path("data/chunks.json")):
        """Load chunks and add to ChromaDB."""
        chunks = self.load_chunks(chunks_file)
        self.add_chunks(chunks)

    def is_indexed(self) -> bool:
        """Check if collection already has data."""
        return self.collection.count() > 0


class RAGPipeline:
    """Combines retrieval with Gemini for question answering."""

    def __init__(self):
        self.embedder = Embedder()
        self.model_name = "gemini-1.5-flash"

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant context for a query."""
        hits = self.embedder.search(query, top_k=top_k)

        context_parts = []
        for hit in hits:
            context_parts.append(
                f"[Source: {hit['source']}]\n{hit['text']}"
            )

        return "\n\n".join(context_parts)

    def generate_prompt(self, query: str, context: str) -> str:
        """Generate the prompt for Gemini."""
        return f"""You are a helpful assistant answering questions about GitLab using the provided context.

Instructions:
- Only answer based on the provided context
- If the context doesn't contain the answer, say so
- Include the source URL in your response when referencing information
- Be helpful and concise

Context:
{context}

Question: {query}

Answer:"""

    def ask(self, query: str, top_k: int = 5) -> tuple[str, list[dict]]:
        """
        Answer a question using RAG.

        Returns (answer, sources) tuple.
        """
        # Retrieve relevant context
        context = self.retrieve_context(query, top_k=top_k)

        if not context:
            return "I couldn't find any relevant information in the GitLab Handbook to answer your question.", []

        # Generate answer
        prompt = self.generate_prompt(query, context)
        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )

        # Get sources
        hits = self.embedder.search(query, top_k=top_k)
        sources = [{"id": h["id"], "source": h["source"]} for h in hits]

        return response.text, sources


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    embedder = Embedder()

    # Check if already indexed
    if embedder.is_indexed():
        logger.info("Collection already indexed")
    else:
        logger.info("Indexing chunks...")
        embedder.index_chunks()
        logger.info("Indexing complete!")

    # Test queries
    test_queries = [
        "GitLab remote work policy",
        "how does GitLab handle expenses",
        "what are GitLab's core values"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        hits = embedder.search(query, top_k=5)

        for i, hit in enumerate(hits, 1):
            print(f"\n{i}. [{hit['source']}]")
            print(f"   Distance: {hit['distance']:.4f}")
            print(f"   Text: {hit['text'][:150]}...")