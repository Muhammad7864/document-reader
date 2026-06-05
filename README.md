# DocuMind 📄 — PDF Q&A with RAG

A Retrieval-Augmented Generation (RAG) app: upload any PDF and ask questions about it.  
Powered by **LangChain · Groq · ChromaDB · HuggingFace Embeddings · Streamlit**.

---

## Architecture

```
User uploads PDF
      ↓
Extract text  (pdfplumber → pypdf fallback)
      ↓
Split into chunks  (RecursiveCharacterTextSplitter)
      ↓
Embed chunks  (all-MiniLM-L6-v2, runs locally — free)
      ↓
Store in ChromaDB  (persistent on disk in ./vectorstore/)
      ↓
User asks a question
      ↓
Retrieve top-K relevant chunks  (cosine similarity)
      ↓
Build prompt  (context + question)
      ↓
Send to Groq LLM  (llama3-8b-8192 or your choice)
      ↓
Stream answer back to UI
```

---

## Project Structure

```
document-reader/
├── app.py            ← Streamlit UI + orchestration
├── utils.py          ← PDF extraction, chunking, embedding, retrieval
├── requirements.txt
├── .env.example      ← copy to .env and add your Groq key
├── pdfs/             ← uploaded PDFs saved here
└── vectorstore/      ← ChromaDB persisted here
```

---

## Quick Start

### 1. Clone / download the project

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key
```bash
cp .env.example .env
# Edit .env and paste your key
```
Or just enter the key directly in the sidebar when running the app.

Get a **free** Groq key at https://console.groq.com

### 5. Run the app
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Configuration (sidebar)

| Setting | Default | Description |
|---------|---------|-------------|
| Chunk size | 1000 | Characters per chunk |
| Chunk overlap | 200 | Overlap between consecutive chunks |
| Top-K | 4 | Chunks retrieved per question |
| Groq model | llama3-8b-8192 | Switch to 70b for better quality |

---

## Models

### Embedding (local, free)
`all-MiniLM-L6-v2` — downloaded once from HuggingFace Hub (~90 MB), runs on CPU.

### LLM (Groq — free tier)
| Model | Context | Notes |
|-------|---------|-------|
| `llama3-8b-8192` | 8 K | Fast, free |
| `llama3-70b-8192` | 8 K | Smarter, still free |
| `mixtral-8x7b-32768` | 32 K | Great for long docs |
| `gemma-7b-it` | 8 K | Lightweight |

---

## Notes

- **Scanned PDFs** (image-only) won't work without OCR.  
  Add `pytesseract` + `pdf2image` if you need OCR support.
- ChromaDB data persists in `./vectorstore/` between sessions.  
  Uploading a new PDF auto-clears the previous collection.
- The embedding model is downloaded on first run (~30 s on a good connection).
