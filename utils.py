import os
import pdfplumber
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ── Constants ──────────────────────────────────────────────────────────────────
VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"   # free, runs locally, fast


# ── PDF Extraction ─────────────────────────────────────────────────────────────
def extract_text_from_pdf(file_path: str) -> str:
    """
    Try pdfplumber first (better for complex layouts / tables).
    Fall back to pypdf if pdfplumber yields nothing.
    """
    text = ""

    # --- pdfplumber ---
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
    except Exception:
        pass

    # --- pypdf fallback ---
    if not text.strip():
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        except Exception as e:
            raise RuntimeError(f"Could not extract text from PDF: {e}")

    if not text.strip():
        raise ValueError(
            "No text could be extracted. The PDF may be scanned / image-based."
        )

    return text.strip()


def get_pdf_metadata(file_path: str) -> dict:
    """Return basic metadata (title, author, pages, file size)."""
    meta = {"pages": 0, "title": "Unknown", "author": "Unknown", "size_kb": 0}
    try:
        reader = PdfReader(file_path)
        meta["pages"] = len(reader.pages)
        if reader.metadata:
            meta["title"]  = reader.metadata.get("/Title",  "Unknown") or "Unknown"
            meta["author"] = reader.metadata.get("/Author", "Unknown") or "Unknown"
        meta["size_kb"] = round(os.path.getsize(file_path) / 1024, 1)
    except Exception:
        pass
    return meta


# ── Text Splitting ─────────────────────────────────────────────────────────────
def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """
    Split raw text into overlapping chunks and wrap each as a LangChain Document.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    documents = [
        Document(page_content=chunk, metadata={"chunk_index": i})
        for i, chunk in enumerate(chunks)
    ]
    return documents


# ── Embeddings & Vector Store ──────────────────────────────────────────────────
def get_embedding_function() -> HuggingFaceEmbeddings:
    """Load the local sentence-transformer embedding model (downloaded on first use)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vectorstore(documents: list[Document], collection_name: str = "pdf_docs") -> Chroma:
    """Embed documents and persist them to ChromaDB."""
    embeddings = get_embedding_function()
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR,
        collection_name=collection_name,
    )
    return vectorstore


def load_vectorstore(collection_name: str = "pdf_docs") -> Chroma | None:
    """Load an existing ChromaDB collection, or return None if it doesn't exist."""
    if not os.path.exists(VECTORSTORE_DIR):
        return None
    try:
        embeddings = get_embedding_function()
        vectorstore = Chroma(
            persist_directory=VECTORSTORE_DIR,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
        # Quick check: see if the collection has any documents
        if vectorstore._collection.count() == 0:
            return None
        return vectorstore
    except Exception:
        return None


def delete_vectorstore(collection_name: str = "pdf_docs") -> None:
    """Delete an existing ChromaDB collection (used when uploading a new PDF)."""
    try:
        embeddings = get_embedding_function()
        vs = Chroma(
            persist_directory=VECTORSTORE_DIR,
            embedding_function=embeddings,
            collection_name=collection_name,
        )
        vs.delete_collection()
    except Exception:
        pass


# ── Retrieval ──────────────────────────────────────────────────────────────────
def retrieve_relevant_chunks(vectorstore: Chroma, query: str, k: int = 4) -> list[Document]:
    """Return the top-k most relevant document chunks for a query."""
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever.invoke(query)


# ── Prompt Builder ─────────────────────────────────────────────────────────────
def build_prompt(context_docs: list[Document], question: str) -> str:
    """Combine retrieved chunks + user question into a single prompt string."""
    context = "\n\n---\n\n".join(doc.page_content for doc in context_docs)
    prompt = f"""You are a helpful assistant that answers questions based strictly on the provided document context.

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
- Answer only from the context above.
- If the answer is not in the context, say: "I couldn't find relevant information in the document."
- Be concise and accurate.
- Quote the document when helpful.

ANSWER:"""
    return prompt
