"""
modules/embedder.py
-------------------
Purpose:
    Handles text embeddings and vector store management.
    Supports both Local (FAISS/Chroma) and Cloud (Pinecone) options.
"""

import os
from typing import List, Optional
# from langchain_community.vectorstores import FAISS, Chroma, Pinecone as LC_Pinecone

from langchain_community.vectorstores import FAISS, Chroma
from langchain_pinecone import PineconeVectorStore as LC_Pinecone

from langchain_core.documents import Document
from utils.utility import get_embedding_model

# --- New import for Pinecone SDK ---
from pinecone import Pinecone, ServerlessSpec


def build_or_update_vectorstores(
    docs: List[Document],
    config: dict,
    persist_dir: Optional[str] = None
):
    """
    Create or update a vectorstore based on the provider in config.

    Args:
        docs: List of LangChain Document chunks
        config: Dictionary loaded from config.json
        persist_dir: Optional local directory to store FAISS/Chroma data

    Returns:
        A LangChain VectorStore object (FAISS, Chroma, or Pinecone)
    """
    provider = config.get("embedding_provider", "openai").lower()
    store_config = config.get("vector_store", {})
    store_type = store_config.get("type", "faiss").lower()

    print(f"Embedding Provider: {provider}")
    print(f"Vector Store Type: {store_type}")

    # --- Load Embedding Model ---
    embedding_model = get_embedding_model(provider)
    print("✅ Embedding model loaded successfully.")

    #Local vecrordb
    if store_type in ["faiss"]:
        if not persist_dir:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            persist_dir = os.path.join(project_root, "vectorstores", f"{provider}_{store_type}_index")
        os.makedirs(persist_dir, exist_ok=True)

        if store_type == "faiss":
            index_path = os.path.join(persist_dir, "index.faiss")
            metadata_path = os.path.join(persist_dir, "index.pkl")

            if os.path.exists(index_path):
                print("♻️ Updating existing FAISS index...")
                vectorstore = FAISS.load_local(
                    persist_dir,
                    embeddings=embedding_model,
                    allow_dangerous_deserialization=True
                )
                if docs:
                    vectorstore.add_documents(docs)
                else:
                    print("loaded existing index only as no new documents provided.")
            else:
                if not docs:
                    raise ValueError("❌ No documents provided and FAISS index not found.")
                print("🆕 Creating new FAISS index...")
                vectorstore = FAISS.from_documents(docs, embedding_model)

            vectorstore.save_local(persist_dir)
            print(f"✅ FAISS vectorstore saved at: {persist_dir}")
            return vectorstore

        else:
            print("Store_type not supported")
    