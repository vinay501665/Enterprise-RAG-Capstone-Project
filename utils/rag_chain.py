from langchain_classic.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.schema.runnable import RunnableLambda, RunnableParallel
from utils.memory import get_memory_summary, get_recent_context

def build_rag_chain(retriever, config):
    provider = config.get("llm_provider", "openai").lower()
    temperature = config.get("llm_temperature", 0.3)

    print(f"Building RAG chain using provider: {provider}")

    if provider == "openai":
        llm = ChatOpenAI(
            model=config.get("openai_model", "gpt-4o-mini"),
            temperature=temperature,
            streaming=True  # future-proof for streaming
        )
    elif provider == "gemini":
        llm = ChatGoogleGenerativeAI(
            model=config.get("gemini_model", "gemini-2.5-flash"),
            temperature=temperature,
            streaming=True  # same reason as above
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    # --- System prompt ---
    system_prompt = """
    You are a helpful AI assistant.
    You have access to two types of memory:
    1️⃣ A summary of past conversations.
    2️⃣ The most recent few question-answer turns for short-term context.
    Always use both to maintain context and continuity naturally.
    """

    # --- Prompt structure ---
    prompt = ChatPromptTemplate.from_template("""
System Prompt:
{system_prompt}

Conversation Summary Memory:
{memory_summary}

Recent Chat History (last few turns):
{recent_memory}

Retrieved Context:
{context}

User Question:
{question}

If the user asks to summarize or refer to earlier parts of the conversation,
rely primarily on the chat history and memory summary.
Otherwise, combine retrieved context and memories to answer effectively.
""")

    # --- Combine retrieved documents ---
    def combine_docs(docs):
        if isinstance(docs, dict):
            docs = docs.get("context", [])
        if not docs:
            return "No relevant context found."

        parts = []
        for i, r in enumerate(docs, start=1):
            src = r.metadata.get('source', 'unknown')
            page = r.metadata.get('page', '-')
            parts.append(f'(source: {src}, page: {page})\n{r.page_content}')

        return "\n\n".join(parts)

    # --- ✅ Build hybrid RAG + memory chain ---
    rag_chain = (
        RunnableParallel(
            {
                "context": RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(combine_docs),
                "question": RunnableLambda(lambda x: x["question"]),
                "system_prompt": RunnableLambda(lambda _: system_prompt),
                "memory_summary": RunnableLambda(
                    lambda x, config: get_memory_summary(config["configurable"]["session_id"])
                ),
                "recent_memory": RunnableLambda(
                    lambda x, config: get_recent_context(config["configurable"]["session_id"])
                ),
            }
        )
        | prompt
        | llm
    )

    print(f"✅ Hybrid RAG + Memory chain built successfully using {provider.upper()}.")
    return rag_chain
