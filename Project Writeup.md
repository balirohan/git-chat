# GitLab Handbook Chatbot

## Project Overview

Inspired by GitLab's "build in public" philosophy, this project is a RAG-powered chatbot that allows users to easily query GitLab's Handbook and Direction pages. The goal was to build a useful chatbot for employees and aspiring candidates who want to learn about GitLab's culture, policies, and product direction without having to dig through the documentation manually.

## Technical Approach

### The Stack

- **Frontend**: Streamlit — chosen for its simplicity and fast iteration time (Best for building POC's quickly)
- **Vector Store**: ChromaDB — lightweight, easy to set up, good Python support
- **Embeddings**: `all-MiniLM-L6-v2` via sentence-transformers — fast, 384-dim embeddings that work well for semantic search
- **LLM**: Groq (Llama 3.3 70B) — Migrated from Gemini due to significant API outages with Google AI Studio in the last few weeks (https://aistudio.google.com/status). Groq offers superior latency and reliability.
- **Scraping**: BeautifulSoup (requests) for static pages, Playwright for JS-rendered content, both with a depth of 2 (base domain + first-level URLs)
  > requests worked well for scraping the handbook but not the direction page, thus playwright + chromium 

### How it Works

The pipeline has 5 stages:

1. **Scraping**: Fetches pages from handbook.gitlab.com and about.gitlab.com. The Hanbook uses requests + BeautifulSoup and The Direction page requires JavaScript rendering using Playwright + Chromium.

2. **Chunking**: Content is split into ~1000-token chunks using tiktoken. Each chunk preserves its source URL for citations later.

3. **Embedding**: Chunks are embedded using `all-MiniLM-L6-v2` and stored in ChromaDB with metadata (source URL, token count).

4. **Retrieval**: When a user asks a question, it gets embedded and the top-k most similar chunks are retrieved from ChromaDB.

5. **Generation**: Retrieved chunks (with source URLs) are sent to Groq with a system prompt instructing it to answer only from the provided context and to not hallucinate.

### Key Decisions

**Groq over Gemini**: Initially, Gemini 3.1 Flash was used. However, due to significant stability issues and API outages with Google AI Studio, the project was migrated to Groq. Groq provides significantly lower latency and higher reliability for the RAG pipeline.

**RAG over fine-tuning**: Instead of fine-tuning a model on GitLab's handbook (expensive, requires GPU, hard to update), RAG was chosen. It's more transparent — sources are always shown — and the knowledge base can be updated by re-scraping.

**Separate scraping vs. API**: I initially considered using a pre-built API but scraping gave me full control over what content to fetch and how to clean it. It also avoids API rate limits and costs.

**Sentence-transformers vs. Gemini embeddings**: I tried switching to Gemini's embedding API to reduce dependencies (torch was causing build issues on Streamlit Cloud), but the quota limits made indexing impractical. Sentence-transformers adds initial model loading time and extra RAM usage but runs locally without API costs, so I kept it.

## Challenges

**Torch/torchvision on Streamlit Cloud**: The biggest hurdle was getting sentence-transformers' torch dependency to install on Streamlit Community Cloud. The full torch package is ~2GB and times out during build. The fix was using CPU-only wheels via `--index-url https://download.pytorch.org/whl/cpu`, which reduced the install size significantly.

**Strict similarity thresholds**: Initially, cosine distance filtering was too strict, causing general queries ("tell me about GitLab") to return no useful context. Increasing `top_k` from 3 to 10 helped significantly — even weakly relevant chunks provide enough context for Gemini to generate a good answer without hallucinating.

## Results

The chatbot is deployed at https://git-chat-joveo.streamlit.app and answers questions about GitLab's handbook with source citations. It's currently indexed with 283 chunks from 90 scraped pages covering handbook.gitlab.com and about.gitlab.com.

## What I'd improve if I had more time...

1. **Deeper scraping**: The current scraper only goes 2 levels deep. A recursive crawler that follows handbook sub-sections would provide richer context for nuanced queries, making this POC feel like a production ready application, minus the simple streamlit UI :)

2. **Query expansion/reranking**: I'd ask Groq to rewrite the user's query into 3 different versions and use all of them to make the semantic search more robust and instead of just retrieving top-k by cosine similarity, I'd use a two-stage approach — retrieve 20 chunks, then use Gemini to pick the 5 most relevant. This handles vague queries better.

3. **Streaming responses**: Adding streaming would improve perceived latency for longer answers. (Used only in real world scenarios)

4. **Follow-up Questions**: After the response, I'd have Groq suggest 2-3 logical follow-up questions based on the retrieved context, which is easily achievable by modifying the instructions and custom few shot examples in the system prompt

5. **Testing**: Though I have manually tested this application as much as I could, for a production project I would use RAGAS to evaluate the RAG pipeline methodically.

6. **Tracing**: I have used LangFuse in the past and debugging LLM pipelines with it just makes life so much easier.

## New Learnings

- I specifically chose to use ChromaDB, as I never had a chance to use it before.
- Learnt about Web Scraping, as I only knew about it theoretically.
- I had built a POC during an internship before but couldn't deploy it. Working through the deployment was nice.
- Never used Groq API before, it is noticeably faster than others.
