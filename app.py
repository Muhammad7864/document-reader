import os
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from utils import (
    extract_text_from_pdf,
    get_pdf_metadata,
    split_text_into_chunks,
    create_vectorstore,
    load_vectorstore,
    delete_vectorstore,
    retrieve_relevant_chunks,
    build_prompt,
)

load_dotenv()

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind – PDF Q&A",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ---- global ---- */
    [data-testid="stAppViewContainer"] { background: #0f1117; }
    [data-testid="stSidebar"] { background: #1a1d27; border-right: 1px solid #2d3045; }

    /* ---- header ---- */
    .hero-title {
        font-size: 2.4rem; font-weight: 800; letter-spacing: -0.5px;
        background: linear-gradient(135deg, #6c63ff 0%, #48b8f0 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-sub { color: #8892a4; font-size: 1rem; margin-top: 4px; }

    /* ---- chat bubbles ---- */
    .chat-user {
        background: linear-gradient(135deg, #6c63ff22, #6c63ff11);
        border: 1px solid #6c63ff44;
        border-radius: 16px 16px 4px 16px;
        padding: 14px 18px; margin: 8px 0;
        color: #e2e8f0;
    }
    .chat-bot {
        background: linear-gradient(135deg, #48b8f022, #48b8f011);
        border: 1px solid #48b8f044;
        border-radius: 16px 16px 16px 4px;
        padding: 14px 18px; margin: 8px 0;
        color: #e2e8f0; line-height: 1.7;
    }
    .chat-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 1px;
                  text-transform: uppercase; margin-bottom: 6px; }
    .label-user { color: #6c63ff; }
    .label-bot  { color: #48b8f0; }

    /* ---- source cards ---- */
    .source-card {
        background: #1e2130; border: 1px solid #2d3045;
        border-radius: 10px; padding: 10px 14px;
        margin: 6px 0; font-size: 0.82rem; color: #8892a4;
    }
    .source-card strong { color: #c8d0e0; }

    /* ---- stat boxes ---- */
    .stat-box {
        background: #1e2130; border: 1px solid #2d3045;
        border-radius: 12px; padding: 14px; text-align: center;
        margin: 4px 0;
    }
    .stat-num  { font-size: 1.6rem; font-weight: 800; color: #6c63ff; }
    .stat-label{ font-size: 0.75rem; color: #8892a4; margin-top: 2px; }

    /* ---- misc ---- */
    .section-divider { border-top: 1px solid #2d3045; margin: 20px 0; }
    .pill {
        display: inline-block; padding: 3px 10px;
        border-radius: 20px; font-size: 0.72rem; font-weight: 600;
        background: #6c63ff22; color: #9d97ff; border: 1px solid #6c63ff44;
        margin-right: 6px;
    }
    .ready-badge {
        display:inline-block; background:#0d7f4f22; color:#34d399;
        border:1px solid #34d39944; border-radius:20px;
        padding:4px 14px; font-size:0.8rem; font-weight:700;
    }
    .stTextInput > div > div > input {
        background: #1e2130 !important; color: #e2e8f0 !important;
        border: 1px solid #2d3045 !important; border-radius: 10px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6c63ff, #48b8f0);
        color: white; border: none; border-radius: 10px;
        font-weight: 700; transition: opacity .2s;
    }
    .stButton > button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialisation ───────────────────────────────────────────────
if "chat_history"    not in st.session_state: st.session_state.chat_history    = []
if "vectorstore"     not in st.session_state: st.session_state.vectorstore     = None
if "pdf_meta"        not in st.session_state: st.session_state.pdf_meta        = None
if "pdf_processed"   not in st.session_state: st.session_state.pdf_processed   = False
if "chunk_count"     not in st.session_state: st.session_state.chunk_count     = 0
if "current_pdf"     not in st.session_state: st.session_state.current_pdf     = None


# ── Helper: init LLM ───────────────────────────────────────────────────────────
@st.cache_resource
def get_llm(api_key: str):
    return ChatGroq(
        api_key=api_key,
        model_name="llama3-8b-8192",   # free & fast on Groq
        temperature=0.2,
        max_tokens=1024,
    )


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- API key ---
    groq_api_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get your free key at https://console.groq.com",
    )
    if groq_api_key:
        st.success("API key saved ✓", icon="✅")
    else:
        st.info("Enter your Groq API key to start.", icon="ℹ️")
        st.markdown("[Get a free key →](https://console.groq.com)", unsafe_allow_html=False)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- Advanced settings ---
    with st.expander("🛠 Advanced"):
        chunk_size    = st.slider("Chunk size (chars)",    300, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk overlap (chars)",  50,  500,  200,  50)
        top_k         = st.slider("Top-K chunks retrieved",  1,    8,    4,   1)
        model_name    = st.selectbox(
            "Groq model",
            ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # --- PDF status ---
    if st.session_state.pdf_processed and st.session_state.pdf_meta:
        meta = st.session_state.pdf_meta
        st.markdown("### 📄 Loaded Document")
        st.markdown(f'<span class="ready-badge">● READY</span>', unsafe_allow_html=True)
        st.markdown(f"**{meta.get('title','—')}**")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="stat-box"><div class="stat-num">{meta.get("pages","—")}</div><div class="stat-label">Pages</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="stat-box"><div class="stat-num">{st.session_state.chunk_count}</div><div class="stat-label">Chunks</div></div>', unsafe_allow_html=True)

        if st.button("🗑 Remove document"):
            delete_vectorstore()
            st.session_state.vectorstore   = None
            st.session_state.pdf_meta      = None
            st.session_state.pdf_processed = False
            st.session_state.chunk_count   = 0
            st.session_state.chat_history  = []
            st.session_state.current_pdf   = None
            st.rerun()

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🛠 Stack")
    for pill in ["LangChain", "Groq", "ChromaDB", "Streamlit", "HuggingFace"]:
        st.markdown(f'<span class="pill">{pill}</span>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════════════════════════
st.markdown('<h1 class="hero-title">DocuMind 📄</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Upload any PDF and ask questions — powered by Groq + LangChain</p>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── PDF Upload ─────────────────────────────────────────────────────
if not st.session_state.pdf_processed:
    st.markdown("### 📤 Upload your PDF")
    uploaded_file = st.file_uploader(
        "Drag & drop or browse", type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file and st.button("⚡ Process Document", use_container_width=True):
        if not groq_api_key:
            st.error("Please enter your Groq API key in the sidebar first.")
        else:
            with st.status("Processing your document…", expanded=True) as status:
                try:
                    # Save to temp file
                    st.write("📥 Saving uploaded file…")
                    os.makedirs("pdfs", exist_ok=True)
                    save_path = os.path.join("pdfs", uploaded_file.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Extract
                    st.write("📖 Extracting text from PDF…")
                    raw_text = extract_text_from_pdf(save_path)
                    meta      = get_pdf_metadata(save_path)
                    meta["title"] = meta["title"] if meta["title"] != "Unknown" else uploaded_file.name

                    # Split
                    st.write(f"✂️  Splitting into chunks (size={chunk_size}, overlap={chunk_overlap})…")
                    docs = split_text_into_chunks(raw_text, chunk_size, chunk_overlap)

                    # Embed + store
                    st.write(f"🧠 Embedding {len(docs)} chunks into ChromaDB…")
                    delete_vectorstore()   # clear any previous collection
                    vs = create_vectorstore(docs)

                    # Persist to session
                    st.session_state.vectorstore   = vs
                    st.session_state.pdf_meta      = meta
                    st.session_state.pdf_processed = True
                    st.session_state.chunk_count   = len(docs)
                    st.session_state.current_pdf   = uploaded_file.name
                    st.session_state.chat_history  = []

                    status.update(label="✅ Document ready!", state="complete")
                    time.sleep(0.8)
                    st.rerun()

                except Exception as e:
                    status.update(label="❌ Processing failed", state="error")
                    st.error(f"Error: {e}")

# ── Q&A Interface ──────────────────────────────────────────────────
else:
    # Greet
    if not st.session_state.chat_history:
        meta = st.session_state.pdf_meta
        st.markdown(
            f"> 📄 **{meta.get('title','Document')}** is loaded "
            f"({meta.get('pages','?')} pages · {st.session_state.chunk_count} chunks · "
            f"{meta.get('size_kb','?')} KB). Ask anything below!"
        )

    # Chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user">'
                f'<div class="chat-label label-user">👤 You</div>{msg["content"]}'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-bot">'
                f'<div class="chat-label label-bot">🤖 DocuMind</div>{msg["content"]}'
                f'</div>',
                unsafe_allow_html=True,
            )
            # Show source chunks in expander
            if msg.get("sources"):
                with st.expander(f"📚 View {len(msg['sources'])} source chunk(s)"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(
                            f'<div class="source-card"><strong>Chunk #{src["index"]+1}</strong><br>{src["text"][:400]}{"…" if len(src["text"])>400 else ""}</div>',
                            unsafe_allow_html=True,
                        )

    # Input
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        question = st.text_input(
            "Ask a question",
            placeholder="What is this document about? Summarise section 3…",
            label_visibility="collapsed",
            key="question_input",
        )
    with col_btn:
        ask_btn = st.button("Ask →", use_container_width=True)

    # Quick prompts
    st.markdown("**Quick prompts:**")
    qcols = st.columns(4)
    quick = ["Summarise this doc", "What are the key points?", "List all topics covered", "Who is the author?"]
    for i, q in enumerate(quick):
        if qcols[i].button(q, key=f"qp_{i}"):
            question = q
            ask_btn  = True

    # Process question
    if ask_btn and question.strip():
        if not groq_api_key:
            st.error("Please enter your Groq API key in the sidebar.")
        elif not st.session_state.vectorstore:
            st.error("Vector store not found. Please re-upload your PDF.")
        else:
            with st.spinner("🔍 Searching document & generating answer…"):
                try:
                    # Retrieve
                    chunks  = retrieve_relevant_chunks(st.session_state.vectorstore, question, k=top_k)
                    prompt  = build_prompt(chunks, question)

                    # LLM
                    llm     = get_llm(groq_api_key)
                    # Override model if changed in sidebar
                    llm.model_name = model_name
                    response = llm.invoke([HumanMessage(content=prompt)])
                    answer   = response.content

                    # Store
                    st.session_state.chat_history.append({"role": "user",      "content": question})
                    st.session_state.chat_history.append({
                        "role":    "assistant",
                        "content": answer,
                        "sources": [
                            {"index": doc.metadata.get("chunk_index", i), "text": doc.page_content}
                            for i, doc in enumerate(chunks)
                        ],
                    })
                    st.rerun()

                except Exception as e:
                    st.error(f"Error generating answer: {e}")

    # Clear chat
    if st.session_state.chat_history:
        if st.button("🗑 Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()
