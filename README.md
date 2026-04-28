# GitLab Handbook Chatbot

A RAG-powered chatbot that answers questions about GitLab's Handbook and Direction pages using Generative AI.

## Features

- **RAG Pipeline**: Combines semantic search (ChromaDB + sentence-transformers) with Gemini for accurate, context-grounded answers
- **Streamlit UI**: Clean chat interface with follow-up question support and source citations
- **Scraped Data**: Content from handbook.gitlab.com and about.gitlab.com

## Prerequisites

- Python 3.10+
- Google AI API key (Gemini)

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
GOOGLE_STUDIO_API_KEY=your_api_key_here
```

Get an API key from [Google AI Studio](https://aistudio.google.com).

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
2. **Chunking**: Splits content into ~1000-token paragraphs using tiktoken
3. **Embedding**: Generates embeddings with `all-MiniLM-L6-v2` (sentence-transformers) and stores in ChromaDB
4. **Retrieval**: On query, embeds the question and retrieves top-k relevant chunks from ChromaDB
5. **Generation**: Sends retrieved context + question to Gemini with a system prompt instructing it to answer only from the provided context

## Deployment

Deploy to Streamlit Community Cloud:

1. Push code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your repo and deploy

Note: Set `GOOGLE_STUDIO_API_KEY` as a secrets environment variable in Streamlit Cloud.
