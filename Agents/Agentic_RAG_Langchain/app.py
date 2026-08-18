import streamlit as st
from typing import List, Dict, Any
from typing_extensions import TypedDict
from datetime import datetime, timedelta
from dateutil import parser as date_parser
import feedparser
import requests
from bs4 import BeautifulSoup
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from uuid import uuid4
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph, START
import json
import time
import os

# Debug logging helper (writes NDJSON lines to a fixed path)
def debug_log(session_id, run_id, hypothesis_id, location, message, data):
    log_path = "/uSers/tamanappainde/Desktop/Generative-AI-Projects/.cursor/debug.log"
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps({
               "sessionId" : session_id,
                "runId" :run_id,
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data,
                "timestamp" :   int(time.time() * 1000)
            }) + "\n")
    except Exception:
        pass

st.set_page_config(page_title="Newsletter Pipeline", page_icon=" ")
st.header(":blue[Multi-Agent Newsletter Pipeline] : green[with Langgraph]")


# Configuration
SOURCES_ALLOWLIST = [
    "https://www.news.aakashg.com/",
    "https://www.theunwindai.com/",
    "https://creatoreconomy.so/",
    "https://www.lennysnewsletter.com/",
    "https://ruben.substack.com/",
]

# Initialize the session state
if 'qdrant_host' not in st.session_state:
    st.session_state.qdrant_host = ""
if 'qdramt_api_key' not in st.session_state:
    st.session_state.qdrant_api_key = ""
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_open_key = ""
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""

def set_sidebar():
    """Setup sidebar for API keys and Configuration."""
    with st.sidebar:
        st.subheader("API Configuration")

        qdrant_host = st.text_input("Enter Your Qdrant URL: ", value=st.session_state.qdrant_host, type="default")
        qdrant_api_key = st.text_input("Enter Your Qdrant API Key: ", value=st.session_state.qdrant_api_key, type="password")
        openai_api_key = st.text_input("Enter Your OpenAI API Key (for embeddings & chat): ", value=st.session_state.openai_api_key, type="password")
        gemini_api_key = st.text_input("Enter Your Gemini API Key (legacy, optional): ", value=st.session_state.gemini_api_key, type="password")

        if st.button("Save Configuration"):
            if qdrant_host and qdrant_api_key and openai_api_key:
                st.session_state.qdrant_host = qdrant_host
                st.session_state.qdrant_api_key = qdrant_api_key
                st.session_state.openai_api_key = openai_api_key
                st.session_state.gemini_api_key = gemini_api_key
            else:
                st.warning("Pleae fill Qdrant Host, Qdrant API key, and OpenAI API key")


        st.subheader("Pipeline Configuration")
        time_window_days = st.number_input("Time Window (days): ", min_value=1, max_value=30, value=14)
        max_items_per_source = st.number_input("Max Items per source: ", min_value=1, max_value=50, value=10)
        language = st.selectbox("Language: ", ["en"], index=0)
        use_embeddings_config = st.checkbox("Enable Embeddings (for semantic clustering)", value=True, help="Disable if experincing timeout issues. App will work in LLM-mode only.")

        st.session_state.config = {
            "time_window_days": time_window_days,
            "max_items_per_source": max_items_per_source,
            "language": language,
            "use_embeddings": use_embeddings_config
        }

def initialize_components():
    """Initialize components that requires API key.
    Returns: (embedding_model, client, db) or (None, client, None) if embedding fail.
    Embeddings are optional - app can work without them.
    """
    if not all([st.session_state.qdrant_host,
               st.session_state.qdrant_api_key,
               st.session_state.openai_api_key]):
        return None, None, None
    embedding_model = None
    db = None

    # Initialize Qdrant Client first
    try:
        client = QdrantClient(
            url=st.session_state.qdrant_host,
            api_key=st.session_state.qdrant_api_key if st.session_state.qdrant_api_key else None,
            timeout=30
        )
    except Exception as client_error:
        st.error(f"Failed to Connect Qdrant: {str(client_error)}")
        return None, None, None

    # Try to run embeddings model (optional - app can work without it)
    # Check if embeddings are enabled in config
    use_embeddings_config = st.session_state.get("config", {}).get("use_embeddings", True)

    if not use_embeddings_config:
        st.info("! Embeddings are disabled in configuration. Running in LLM-only mode.")
        embedding_model = None
    else:
        try:
            embedding_model = OpenAIEmbeddings(
                model = "text-embedding-3-small",
                openai_api_key=st.session_state.openai_api_key
            )
            st.success("Emedding Model Initialized Successfully.")
        except Exception as embed_init_error:
            error_str = str(embed_init_error)
            if "504" in error_str or "Deadline" in error_str or "timeout" in error_str.lower():
                st.warning(" Embedding Model intialization timeout. The app will continue without embeddings (LLM-only-mode).")
            else:
                st.warning(f"Could not initialize embedding model: {error_str[:200]}. The app will continue without embeddings. ")
            embedding_model = None
            

    # Initialize Qdrant Collection (only if we have mebedding model)
    if embedding_model:
        collection_name = "newsletter_db"
        from qdrant_client.models import Distance, VectorParams
        embedding_dim = 1536 # OpenAI text-embedding-3-small dimension
        collection_needs_recreation = False

        # Check existing collection and dimension
        try:
            existing_collection = client.get_collection(collection_name)
            existing_dim = existing_collection.config.params.vectors.size
            if existing_dim != embedding_dim:
                st.warning(f" Existing Collectin has {existing_dim} dimensions, but embeddings require {embedding_dim}. Recreating Collection...")
                try:
                    client.delete_collection(collection_name)
                    st.info(f" Old Collection deleted.")
                    collection_needs_recreation = True
                except Exception as delete_error:
                    st.error(f" Failed to delete old collection: {str(delete_error)[:200]}")

                    
                    