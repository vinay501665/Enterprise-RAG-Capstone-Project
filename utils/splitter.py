from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(docs: List[Document], chunk_size: int = 800, chunk_overlap:int = 100) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_doc = []

    for doc in docs:
        text = splitter.split_text(doc.page_content)
        for i, t in enumerate(text):
            meta = dict(doc.metadata) if doc.metadata else {}
            meta.update({'chunk_index': i})
            split_doc.append(Document(page_content=t, metadata = meta))

    return split_doc


def split_document(docs: List[Document], chunk_size: int = 800, chunk_overlap: int = 100) -> List[Document]:
    return split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

