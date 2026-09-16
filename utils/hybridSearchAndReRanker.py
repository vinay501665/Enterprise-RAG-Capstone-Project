from langchain_community.retrievers import BM25Retriever

from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from typing import List
from langchain_core.documents import Document

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

def create_hybrid_retriever(retr: VectorStoreRetriever, docs: List[Document]):
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 5

    hybrid_retriever = EnsembleRetriever(
        retrievers = [bm25_retriever, retr],
        weights = [0.4, 0.6],
    )

    #Re-ranker
    rerank_model = HuggingFaceCrossEncoder(model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2", model_kwargs={"device": "cpu"})
    compressor = CrossEncoderReranker(model=rerank_model, top_n = 2)

    rerank_retriever = ContextualCompressionRetriever(base_compressor = compressor, base_retriever = hybrid_retriever)

    return rerank_retriever

