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
                    print("⚪ No new documents provided — loaded existing index only.")
            else:
                if not docs:
                    raise ValueError("❌ No documents provided and FAISS index not found.")
                print("🆕 Creating new FAISS index...")
                vectorstore = FAISS.from_documents(docs, embedding_model)

            vectorstore.save_local(persist_dir)
            print(f"✅ FAISS vectorstore saved at: {persist_dir}")
            return vectorstore

        else:
            print("Store_type not supported ❌")

    #Cloud vector db
    elif store_type == "pinecone":
        print("☁️ Using Pinecone vector store...")

        # --- Load API credentials ---
        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENV") or store_config.get("cloud", {}).get("environment", "us-east-1")
        if not api_key:
            raise ValueError("❌ Missing PINECONE_API_KEY in environment variables.")

        pc = Pinecone(api_key=api_key)
        index_name = f"{store_config.get('cloud', {}).get('index_name', 'rag-assistant-index')}-{provider}"
        namespace = store_config.get("cloud", {}).get("namespace", "default")

        # --- Detect embedding model dimension dynamically ---
        print("🔍 Detecting embedding dimension...")
        test_text = ["Hello world!"]
        try:
            sample_vec = embedding_model.embed_documents(test_text)[0]
            embedding_dim = len(sample_vec)
        except Exception:
            embedding_dim = len(embedding_model.embed_query("Hello world!"))
        print(f"📏 Embedding dimension detected: {embedding_dim}")

        # --- List all existing indexes ---
        existing_indexes = [i["name"] for i in pc.list_indexes()]

        if index_name not in existing_indexes:
            print(f"🆕 Creating Pinecone index '{index_name}' with dim={embedding_dim}")

            if environment.endswith("-free") or environment.endswith("-starter"):
                from pinecone import PodSpec
                pc.create_index(
                    name=index_name,
                    dimension=embedding_dim,
                    metric="cosine",
                    spec=PodSpec(environment=environment)
                )
            else:
                pc.create_index(
                    name=index_name,
                    dimension=embedding_dim,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=environment)
                )
            print(f"✅ Pinecone index '{index_name}' created.")

        else:
            # ✅ Verify dimension matches the existing index
            desc = pc.describe_index(index_name)
            server_dim = desc["dimension"]
            if server_dim != embedding_dim:
                raise ValueError(
                    f"❌ Dimension mismatch: Embedding model = {embedding_dim}, "
                    f"Pinecone index = {server_dim}. "
                    f"Please use a different index name or matching embedding provider."
                )
            print(f"♻️ Using existing Pinecone index '{index_name}'.")

        # ✅ Always connect and upsert documents — runs for both new AND existing index
        vectorstore = LC_Pinecone.from_documents(
            docs,
            embedding=embedding_model,
            index_name=index_name,
            namespace=namespace,
            pinecone_api_key=api_key
        )

        print(f"✅ Pinecone index '{index_name}' ready (namespace: {namespace}, dim={embedding_dim})")
        return vectorstore