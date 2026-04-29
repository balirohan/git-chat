# GitLab Handbook Chatbot

A RAG-powered chatbot that answers questions about GitLab's Handbook and Direction pages using Generative AI.

**Live Demo**: https://git-chat-joveo.streamlit.app

>Note: You may encounter a brief delay during the first load while the environment initializes and dependencies are prepared.

## Features

- **RAG Pipeline**: Combines semantic search (ChromaDB + sentence-transformers) with Groq (Llama 3.3 70B) for accurate, context-grounded answers.

  > Switched from Gemini due to recent Google AI API outages - for more details check out: https://aistudio.google.com/status
- **Streamlit UI**: Clean chat interface with source citations.
- **Scraped Data**: Content from handbook.gitlab.com (Handbook) and about.gitlab.com (Direction).
  > scraped content with depth 2 from GitLab Handbook and Direction pages (Base domain + first-level URLs).

## Prerequisites

- Python 3.10+
- Groq API key

## Setup

### 1. Clone and virtual environment

```bash
git clone <repo-url>
cd git-chat
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_api_key_here
```

Get an API key from [Groq Console](https://console.groq.com).

### 4. Scrape and chunk content (first-time setup)

```bash
python -m src.scraper
python -m src.chunker
```

This scrapes GitLab's Handbook and Direction pages, then splits the content into embedding-friendly chunks saved to `data/chunks.json`.

### 5. Index chunks

On first run, the app automatically indexes `data/chunks.json` into ChromaDB. To manually re-index:

```bash
python -c "from src.embedder import Embedder; Embedder().index_chunks()"
```

## Running

```bash
streamlit run src/app.py
```

Open http://localhost:8501 in your browser.

## Project Structure

```
src/
  scraper.py      # Scrapes handbook.gitlab.com and about.gitlab.com
  chunker.py      # Splits content into ~1000-token chunks
  embedder.py     # Embedding + ChromaDB storage + RAG pipeline
  app.py          # Streamlit UI
data/
  scraped_content_final.json  # Raw scraped content
  chunks.json               # Chunked content
  chromadb/                 # Persistent vector store
```

## How it works

1. **Scraping**: Fetches pages from handbook.gitlab.com (requests) and about.gitlab.com (Playwright for JS-rendered content)
  > used playwright + chromium for scraping the direction page, as requests was returning nothing.
2. **Chunking**: Splits content into ~1000-token paragraphs using tiktoken
3. **Embedding**: Generates embeddings with `all-MiniLM-L6-v2` (sentence-transformers) and stores in ChromaDB
4. **Retrieval**: On query, embeds the question and retrieves top-k relevant chunks from ChromaDB
5. **Generation**: Sends retrieved context + question to Groq with a system prompt instructing it to answer only from the provided context

## Deployment

Deploy to Streamlit Community Cloud:

1. Push code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your repo and deploy

Note: Set `GROQ_API_KEY` as a secrets environment variable in Streamlit Cloud.
