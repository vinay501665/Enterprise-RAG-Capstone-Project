from langchain_openai import ChatOpenAI

def get_retriever(vectorstore, config):
    retriever_type = config.get("retriever_type", "base").lower() 

    if retriever_type == "base":
        print("USing base retriever(simaple similarity search)")
        return vectorstore.as_retriever(search_kwargs = {"k":3})
    else:
        raise ValueError(f'Unsupported retriever_type: {retriever_type}')