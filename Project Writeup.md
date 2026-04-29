# Project Writeup: GitLab Handbook Chatbot

## Project Overview

Inspired by GitLab's "build in public" philosophy, this project is a RAG-powered chatbot that allows users to easily query GitLab's Handbook and Direction pages. The goal was to build something useful for employees and aspiring candidates who want to learn about GitLab's culture, policies, and product direction — without having to dig through hundreds of pages of documentation.

## Motivation

When I first looked at GitLab's handbook, I was overwhelmed by how much information exists — everything from remote work policies to engineering processes to product roadmaps. I thought: what if there was a conversational interface that could answer questions directly, with full transparency about where the information comes from? That aligned perfectly with GitLab's open culture.

## Technical Approach

### The Stack

- **Frontend**: Streamlit — chosen for its simplicity and fast iteration time
- **Vector Store**: ChromaDB — lightweight, easy to set up, good Python support
- **Embeddings**: `all-MiniLM-L6-v2` via sentence-transformers — fast, 384-dim embeddings that work well for semantic search
- **LLM**: Groq (Llama 3.3 70B) — Migrated from Gemini due to significant API outages with Google AI Studio in early 2026. Groq offers superior latency and reliability.
- **Scraping**: BeautifulSoup (requests) for static pages, Playwright for JS-rendered content

### How it Works

The pipeline has 5 stages:

1. **Scraping**: Fetches pages from handbook.gitlab.com and about.gitlab.com. Static pages use requests + BeautifulSoup. Pages requiring JavaScript rendering use Playwright.

2. **Chunking**: Content is split into ~1000-token chunks using tiktoken. Each chunk preserves its source URL for citations later.

3. **Embedding**: Chunks are embedded using `all-MiniLM-L6-v2` and stored in ChromaDB with metadata (source URL, token count).

4. **Retrieval**: When a user asks a question, it gets embedded and the top-k most similar chunks are retrieved from ChromaDB.

5. **Generation**: Retrieved chunks (with source URLs) are sent to Groq with a system prompt instructing it to answer only from the provided context and to not hallucinate.

### Key Decisions

**Groq over Gemini**: Initially, Gemini 3.1 Flash was used. However, due to significant stability issues and API outages with Google AI Studio in early 2026, the project was migrated to Groq. Groq provides significantly lower latency and higher reliability for the RAG pipeline.

**RAG over fine-tuning**: Instead of fine-tuning a model on GitLab's handbook (expensive, requires GPU, hard to update), RAG was chosen. It's more transparent — sources are always shown — and the knowledge base can be updated by re-scraping.

**Separate scraping vs. API**: I initially considered using a pre-built API but scraping gave me full control over what content to fetch and how to clean it. It also avoids API rate limits and costs.

**Sentence-transformers vs. Gemini embeddings**: I tried switching to Gemini's embedding API to reduce dependencies (torch was causing build issues on Streamlit Cloud), but the quota limits made indexing impractical. Sentence-transformers runs locally without API costs, so I kept it.

## Challenges

**Torch/torchvision on Streamlit Cloud**: The biggest hurdle was getting sentence-transformers' torch dependency to install on Streamlit Community Cloud. The full torch package is ~2GB and times out during build. The fix was using CPU-only wheels via `--index-url https://download.pytorch.org/whl/cpu`, which reduced the install size significantly.

**Embedding model compatibility**: Switching from google-generativeai to google-genai (the newer SDK) required updating the import from `google.genai` to the correct module path. The Google AI ecosystem has some confusing package naming — google-generativeai vs google-genai are different packages with different APIs.

**Strict similarity thresholds**: Initially, cosine distance filtering was too strict, causing general queries ("tell me about GitLab") to return no useful context. Increasing `top_k` from 3 to 10 helped significantly — even weakly relevant chunks provide enough context for Gemini to generate a good answer.

## Results

The chatbot is deployed at https://git-chat-joveo.streamlit.app and answers questions about GitLab's handbook with source citations. It's currently indexed with ~283 chunks from 90 scraped pages covering handbook.gitlab.com and about.gitlab.com.

## What I'd Do Differently

1. **Deeper scraping**: The current scraper only goes 2 levels deep. A recursive crawler that follows handbook sub-sections would provide richer context for nuanced queries.

2. **Query expansion/reranking**: Instead of just retrieving top-k by cosine similarity, I'd use a two-stage approach — retrieve 20 chunks, then use Gemini to pick the 5 most relevant. This handles vague queries better.

3. **Streaming responses**: Adding streaming would improve perceived latency for longer answers.

## Learnings

This project gave me hands-on experience with:
- Building a complete RAG pipeline from scratch
- Handling deployment challenges (dependency management, environment differences)
- Prompt engineering for grounded, hallucination-free generation
- Tradeoffs between API-based vs. local embedding approaches

I also learned that "build in public" isn't just a motto — it's a design philosophy that influenced every aspect of this project, from the transparent source citations to the honest "I don't know" responses when context doesn't contain the answer.
