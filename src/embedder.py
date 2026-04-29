"""
Embedder
Generates embeddings for chunks and stores in ChromaDB for semantic search.

Also handles:
- Query embedding and retrieval (using sentence-transformers)
- RAG pipeline: retrieve + generate answer
"""

import os
from pathlib import Path
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
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
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

    def search(self, query: str, top_k: int = 10) -> list[dict]:
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

    def index_chunks(self, chunks_file: Path = Path("data/chunks.json")):
        """Load chunks from JSON and add to ChromaDB."""
        with open(chunks_file) as f:
            chunks = json.load(f)
        logger.info(f"Loaded {len(chunks)} chunks from {chunks_file}")
        self.add_chunks(chunks)

    def is_indexed(self) -> bool:
        """Check if collection already has data."""
        return self.collection.count() > 0


class RAGPipeline:
    """Combines retrieval with Gemini for question answering."""

    def __init__(self):
        self.embedder = Embedder()
        self.model_name = "gemini-3.1-flash-lite-preview"

    def retrieve_context(self, query: str, top_k: int = 10) -> str:
        """Retrieve relevant context for a query."""
        hits = self.embedder.search(query, top_k=top_k)

        context_parts = []
        for hit in hits:
            context_parts.append(
                f"[Source: {hit['source']}]\n{hit['text']}"
            )

        return "\n\n".join(context_parts)

    def generate_prompt(self, query: str, context: str) -> str:
        """Generate the prompt for Gemini with XML tags for clear demarcation."""
        return f"""<system>
You are a helpful, transparent AI assistant that answers questions about GitLab using ONLY the provided context from the GitLab Handbook and Direction pages.
</system>

<instructions>
- You are part of GitLab's "build in public" philosophy — be honest, clear, and transparent
- Answer based ONLY on the provided context below
- If the context does not contain information directly related to the question, say exactly: "I don't have that information in the provided context"
- Do NOT make up, guess, or add any information not explicitly stated in the context
- Be helpful, accurate, and concise in your response
- Cite sources inline using [Source 1], [Source 2], etc. when referencing specific information from a source
- Number sources in the order they appear in your answer (first reference = Source 1, second = Source 2, etc.)
- Do NOT invent or guess source numbers — only cite sources you actually use
- Sources are displayed separately below — do not list them again at the end
- Use a friendly, informative tone consistent with GitLab's open culture
</instructions>

<context>
{context}
</context>

<question>
{query}
</question>

<answer>
"""

    def ask(self, query: str, top_k: int = 10) -> tuple[str, list[dict]]:
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

        # Get all retrieved sources in order
        hits = self.embedder.search(query, top_k=top_k)
        all_sources = [{"id": h["id"], "source": h["source"]} for h in hits]

        # Extract cited source numbers from response
        import re
        cited_numbers = set(int(n) for n in re.findall(r'\[Source\s+(\d+)\]', response.text, re.IGNORECASE))

        # Only return sources that were actually cited
        cited_sources = [all_sources[i] for i in cited_numbers if i < len(all_sources)]

        return response.text, cited_sources


