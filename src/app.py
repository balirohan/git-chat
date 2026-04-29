"""
GitLab Handbook Chatbot - Streamlit UI
A RAG-powered chatbot for GitLab Handbook and Direction pages.
"""

import sys
from pathlib import Path

import os                                 
os.environ["STREAMLIT_WATCHER_IGNORE_GLOBS"] = "torchvision,*.vision*"

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from src.embedder import RAGPipeline, Embedder

# Page config
st.set_page_config(
    page_title="GitLab Handbook Chatbot",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def load_rag_pipeline():
    embedder = Embedder()
    if not embedder.is_indexed():
        embedder.index_chunks()
    return RAGPipeline()

rag = load_rag_pipeline()

# Header
st.title("📚 GitLab Handbook Chatbot")
st.markdown("Ask questions about GitLab's Handbook, Direction, and more!")

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This chatbot answers questions using:
    - **GitLab Handbook** - company policies, values, processes
    - **Direction pages** - product roadmap and strategy

    Answers include source citations for transparency.
    """)
    st.header("Tips")
    st.markdown("""
    - Be specific in your questions
    - Ask about policies, values, or processes
    - Click on source links to learn more
    """)

# Chat history (session state)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about GitLab..."):
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = rag.ask(prompt, top_k=10)
                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure the GROQ_API_KEY is set in your .env file.")