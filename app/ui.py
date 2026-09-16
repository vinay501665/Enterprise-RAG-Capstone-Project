import sys, os
import streamlit as st
import json, uuid
from langchain_core.documents import Document

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from utils.loader import load_document
from utils.splitter import split_documents
from utils.embedder import build_or_update_vectorstores
from utils.retriever import get_retriever
from utils.rag_chain import build_rag_chain
from utils.memory import add_memory_to_chain
from utils.hybridSearchAndReRanker import create_hybrid_retriever

# define page config
st.set_page_config(page_title = "Enterprise Knowledge Assistant", layout = "wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #17233c;
        --muted: #65728a;
        --paper: #f7f9fc;
        --line: #dbe4ef;
        --teal: #00a6a6;
        --coral: #f26b5e;
        --yellow: #f8c95c;
    }

    .stApp {
        background: radial-gradient(circle at 85% 0%, #e5fbf8 0, transparent 27%),
                    linear-gradient(135deg, #f7f9fc 0%, #eef4f8 100%);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #17233c 0%, #243958 100%);
        border-right: 0;
    }

    [data-testid="stSidebar"] * { color: #f8c95c; }
    [data-testid="stSidebar"] label p { color: #d8e4f3 !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #b8c9df; }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: #243958 !important;
        border-color: #617895;
        color: #f8c95c !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"],
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] * {
        background-color: #243958 !important;
        color: #f8c95c !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] [role="button"],
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] [role="button"] > div {
        color: #f8c95c !important;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] span {
        color: #f8c95c !important;
    }
    [data-baseweb="popover"] [role="option"] {
        color: #17233c;
        background: #ffffff;
    }
    [data-baseweb="popover"] [role="option"][aria-selected="true"] {
        color: #17233c;
        background: #e5fbf8;
    }
    [data-testid="stSidebar"] .stButton button {
        background: var(--coral);
        border: 0;
        color: white;
        font-weight: 700;
        box-shadow: 0 8px 18px rgba(242, 107, 94, 0.25);
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: #ff8174;
        color: white;
    }

    .brand-lockup {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 4px 0 24px;
    }
    .brand-mark {
        display: grid;
        place-items: center;
        width: 40px;
        height: 40px;
        border-radius: 12px;
        background: var(--yellow);
        color: var(--ink);
        font-size: 21px;
        font-weight: 700;
    }
    .brand-name {
        color: white;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 18px;
        font-weight: 700;
        letter-spacing: 0;
    }
    .brand-caption {
        color: #a9bdd5;
        font-size: 12px;
        margin-top: 2px;
    }
    .hero-card {
        position: relative;
        overflow: hidden;
        margin: 12px 0 26px;
        padding: 28px 32px;
        border: 1px solid rgba(0, 166, 166, 0.16);
        border-radius: 20px;
        background: linear-gradient(110deg, #ffffff 0%, #effcf9 68%, #fff5df 100%);
        box-shadow: 0 16px 40px rgba(23, 35, 60, 0.08);
    }
    .hero-card::after {
        content: '';
        position: absolute;
        right: -35px;
        top: -42px;
        width: 150px;
        height: 150px;
        border: 22px solid rgba(242, 107, 94, 0.16);
        border-radius: 50%;
    }
    .eyebrow {
        color: var(--teal);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }
    .hero-title {
        margin: 8px 0 6px;
        color: var(--ink);
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(28px, 4vw, 44px);
        line-height: 1.05;
    }
    .hero-copy { margin: 0; color: var(--muted); font-size: 15px; }
    .session-chip {
        margin: 10px 0 22px;
        padding: 10px 12px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.07);
        color: #c4d4e9 !important;
        font-size: 11px;
        word-break: break-all;
    }
    .empty-state {
        margin-top: 28px;
        padding: 32px;
        border: 1px dashed #b8d6d5;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.68);
        text-align: center;
    }
    .empty-icon { color: var(--coral); font-size: 30px; }
    .empty-title { color: var(--ink); font-family: 'Space Grotesk', sans-serif; font-size: 20px; font-weight: 700; }
    .empty-copy { color: var(--muted); font-size: 14px; }
    [data-testid="stChatMessage"] {
        border: 1px solid rgba(219, 228, 239, 0.75);
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.66);
        margin-bottom: 10px;
    }
    [data-testid="stChatInput"] {
        border-color: var(--teal);
        box-shadow: 0 8px 24px rgba(0, 166, 166, 0.12);
    }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; }
    </style>
    """,
    unsafe_allow_html=True,
)

# load config
CONFIG_PATH = os.path.join(project_root, "config", "config.json")
with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)


# session & multichat management
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}

if "active_session" not in st.session_state:
    st.session_state.active_session = None


# sidebar: chat sessions
st.sidebar.markdown(
    """
    <div class="brand-lockup">
        <div class="brand-mark">✦</div>
        <div>
            <div class="brand-name">Enterprise Knowledge Assistant</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.markdown("### Chat Settings")

# new chat button
if st.sidebar.button("➕ New Chat"):
    new_id = str(uuid.uuid4())
    st.session_state.chat_sessions[new_id] = {
        "name": f"Chat {len(st.session_state.chat_sessions) + 1}",
        "llm_provider": "openai",
        "memory_enabled": True,
        "chat_history": []
    }
    st.session_state.active_session = new_id

# if no session exists, initialize first chat
if not st.session_state.chat_sessions:
    new_id = str(uuid.uuid4())
    st.session_state.chat_sessions[new_id] = {
        "name": f"Chat 1",
        "llm_provider": "openai",
        "memory_enabled": True,
        "chat_history": []        
    }
    st.session_state.active_session = new_id


# select active chat session
session_names = {
    sid: info["name"] for sid, info in st.session_state.chat_sessions.items()
}

selected_name = st.sidebar.selectbox(
    "Active Chat Session",
    options=list(session_names.values()),
    index=list(session_names.values()).index(
        st.session_state.chat_sessions[st.session_state.active_session]["name"]
    ),
)


for sid, info in st.session_state.chat_sessions.items():
    if info["name"] == selected_name:
        st.session_state.active_session = sid
        break

current_session = st.session_state.chat_sessions[st.session_state.active_session]

# show session ID for reference: optional
st.sidebar.markdown(
    f'<div class="session-chip">SESSION<br><strong>{st.session_state.active_session}</strong></div>',
    unsafe_allow_html=True,
)

# session-specific LLM provider
current_session["llm_provider"] = st.sidebar.selectbox(
    "LLM Provider",
    ["openai", "gemini"],
    index = 0 if current_session["llm_provider"] == "openai" else 1,
    key = f"provider_{st.session_state.active_session}",
)

# vectorstore path
os.path.join(project_root, "vectorstores", "faiss_index")

directory = '../policy_documents'
all_chunks: list[Document] = []

for file in os.listdir(directory):
    file_path = os.path.join(directory, file)

    if not os.path.isfile(file_path):
        continue

    docs = load_document(file_path)
    print('Loaded documents', len(docs))
    print('First 50 chars: ', docs[0].page_content[:100])

    print('=====Splitting into chunks ===========')
    chunk = split_documents(docs, 5000, 200)
    policy_type = os.path.splitext(file)[0]
    for ch in chunk:
        ch.metadata['source'] = file
        ch.metadata['path'] = file_path
        ch.metadata['type'] = policy_type
    all_chunks.extend(chunk)
    vectorstore = build_or_update_vectorstores(all_chunks, CONFIG)


# build RAG chain for Active-Session

try:
    vectorstore = build_or_update_vectorstores([], CONFIG)
    retriever = get_retriever(vectorstore, CONFIG)

    hybrid_retriever = create_hybrid_retriever(retriever, all_chunks)

    CONFIG["llm_provider"] = current_session["llm_provider"]

    rag_chain = build_rag_chain(hybrid_retriever, CONFIG)

    rag_chain_with_memory = add_memory_to_chain(
        rag_chain,
        session_id = st.session_state.active_session,
        enabled = True,
    )
except Exception as e:
    st.error(f"Error initializing backend: {e}")
    st.stop()


# -------------------------------------------------------------
# Chat UI Section
# -------------------------------------------------------------
st.markdown(
    """
    <section class="hero-card">
        <div class="eyebrow">Enterprise intelligence, made easy</div>
        <div class="hero-title">Ask. Discover. Decide.</div>
        <p class="hero-copy">Search your policy library and get grounded answers with useful references.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

if not current_session["chat_history"]:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">✦</div>
            <div class="empty-title">Your next answer is one question away</div>
            <div class="empty-copy">Try asking about leave, benefits, security, or any policy in your library.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --- Display chat messages for current session ---
for msg in current_session["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input ---
if prompt := st.chat_input("Ask your question..."):
    current_session["chat_history"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            retrieved_docs = hybrid_retriever.invoke(prompt)
            response = rag_chain_with_memory.invoke(
                {"question": prompt},
                config={
                    "configurable": {"session_id": st.session_state.active_session}
                },
            )
            full_response = getattr(response, "content", str(response))

            policy_terms = (
                "policy", "leave", "benefit", "security", "compliance", "employee",
                "vacation", "holiday", "payroll", "insurance", "privacy", "attendance",
                "reimbursement", "harassment", "conduct", "remote work", "work from home",
                "vpt", "verdant", "hr",
            )
            normalized_prompt = prompt.casefold()
            is_policy_question = any(term in normalized_prompt for term in policy_terms)

            if is_policy_question:
                references = []
                seen_references = set()
                for document in retrieved_docs:
                    source = document.metadata.get("source", "unknown")
                    page = document.metadata.get("page")
                    reference_key = (source, page)
                    if reference_key in seen_references:
                        continue
                    seen_references.add(reference_key)
                    page_text = f", page {page}" if page else ""
                    references.append(f"- {source}{page_text}")

                if references:
                    full_response += "\n\n**References**\n" + "\n".join(references)
        except Exception as e:
            full_response = f"⚠️ Error: {e}"

        response_placeholder.markdown(full_response)

    current_session["chat_history"].append(
        {"role": "assistant", "content": full_response}
    )