# RAG Knowledge Assistant

A Retrieval-Augmented Generation pipeline built from scratch: upload a document, ask questions about it, and get answers grounded only in what the document actually says — with page-level citations, and an honest "I don't know" when the answer genuinely isn't there.

No LangChain. No LlamaIndex. Every stage of the pipeline — chunking, embedding orchestration, vector indexing, retrieval, prompt construction — is hand-written and inspectable in `backend/app/services/`.

---

## Overview

You give it a PDF, TXT, or MD file. It:

1. Extracts the text (with page numbers preserved for PDFs)
2. Splits it into overlapping chunks along paragraph/sentence/word boundaries
3. Embeds each chunk into a 384-dimensional vector using a local, open-source model
4. Indexes those vectors in FAISS
5. On a question: embeds the question, finds the most similar chunks, and asks an LLM (Groq) to answer *using only those chunks*
6. Returns the answer alongside the exact chunks it came from — page numbers, similarity scores, and all

If the retrieved chunks don't actually contain the answer, it says so, rather than guessing.

---

## Features

- Upload PDF / TXT / MD documents through a browser UI or the API directly
- Configurable chunk size, chunk overlap, and top-K at request time
- Real FAISS vector search (`IndexFlatIP`, exact cosine similarity via normalized inner product)
- Page-level source citations with similarity scores on every answer
- Honest refusal when a question isn't answerable from the document
- A from-scratch, professional-looking web UI (no framework, no build step)
- Full interactive API docs at `/docs` (FastAPI's built-in Swagger UI)
- 110+ automated tests covering chunking, embeddings, retrieval, generation, the API layer, and security hardening

---

## Architecture

```
                    User
                     |
                     v
              Frontend (browser)
        plain HTML / CSS / JS, no build step
                     |
                     v  HTTP
              FastAPI (backend/app/main.py)
                     |
     +---------------+----------------+
     |               |                |
     v               v                v
Document Loader   Chunker        Embedding Model
(PyMuPDF/txt)    (custom,       (sentence-transformers,
                 hand-written)   local, no API key)
     |               |                |
     +-------+-------+----------------+
             |
             v
        FAISS Index
   (IndexFlatIP, cosine via
    normalized inner product)
             |
             v
         Retriever
   (query embed + similarity search)
             |
             v
      LLM Generation (Groq)
   grounded prompt, low temperature
             |
             v
   Grounded Answer + Page Citations
```

---

## RAG Pipeline Explanation

**1. Document ingestion** (`services/document_loader.py`)
Opens a PDF (via PyMuPDF), TXT, or MD file and extracts plain text. PDFs are read page by page, so every extracted piece of text keeps its real page number attached — this is what makes accurate citations possible later. TXT/MD files are treated as a single logical "page 1," since those formats have no page concept, keeping the data shape consistent for every file type.

**2. Chunking** (`services/chunker.py`)
The extracted text is split into overlapping chunks. Rather than blindly cutting every N characters, the chunker prefers natural boundaries: it tries to keep paragraphs whole, falls back to splitting on sentences if a paragraph is too long, and falls back to word-level splitting only as a last resort. Chunking happens **per page**, not on the whole document as one blob — so a chunk can never span two PDF pages, which means every chunk has exactly one unambiguous page number.

**3. Embeddings** (`services/embeddings.py`)
Each chunk's text is converted into a 384-dimensional vector using a local HuggingFace model (`sentence-transformers/all-MiniLM-L6-v2`), entirely offline after the first download — no paid API, no per-request cost. The same model embeds the user's question later, so both live in the same vector space and can be meaningfully compared.

**4. Vector store** (`services/vector_store.py`)
Chunk vectors are indexed in FAISS (`IndexFlatIP`). Chunk metadata (text, filename, page number) is stored in a parallel JSON file, since FAISS itself only stores raw vectors and sequential integer IDs — it knows nothing about "page 12" or "handbook.pdf." The index and its metadata are saved to disk so a document only needs to be processed once.

**5. Retrieval** (`services/retriever.py`)
A user's question is embedded with the same model, normalized, and searched against the FAISS index. The top-K most similar chunks are returned, ranked by cosine similarity, each carrying its real page number and similarity score.

**6. Generation** (`services/generator.py`)
The question and the retrieved chunks are assembled into a strict prompt (see [Grounding](#grounding--hallucination-control) below) and sent to Groq. The model is instructed to answer only from the provided context and to say so explicitly if the answer isn't there.

**7. Grounded answer**
The API returns the generated answer plus the exact source chunks used — page numbers and similarity scores included — so every claim in the answer can be traced back to real, retrieved text.

---

## Why These Technologies?

**Why FastAPI?**
Async-native, generates interactive API documentation automatically (`/docs`), and Pydantic-based request/response validation catches malformed input before it reaches any business logic. For a project being evaluated on API design, that auto-generated documentation is a direct requirement satisfied for free.

**Why PyMuPDF?**
Fast, and — critically — gives per-page text access. That page-level granularity is what makes honest, accurate page citations possible; a PDF library that only returned one big blob of text for the whole document would make citations either impossible or fabricated.

**Why custom chunking, not a framework's text splitter?**
The task requires understanding and being able to modify every part of the pipeline in an interview. A framework's text splitter is a black box; a hand-written one is not. It also meant the chunking strategy (paragraph → sentence → word fallback, per-page boundaries) could be designed specifically around this project's citation requirements, rather than accepting whatever a general-purpose splitter happens to do.

**Why `sentence-transformers/all-MiniLM-L6-v2`?**
It's small (~80MB), runs fast on CPU with no GPU required, and is trained specifically for sentence/short-passage similarity — which matches this project's chunk size range well. It requires no API key and no per-request cost, which matters for a tool meant to run on a student's laptop. The trade-off, stated plainly: larger models (e.g. `bge-small-en-v1.5` at 768 dimensions, or bigger) generally retrieve somewhat more accurately, at the cost of more memory and slower embedding. For a single-document, student-scale RAG system, that accuracy gain rarely justifies the slower iteration loop during development — but it's a legitimate change to make later by editing one config value.

**Why FAISS?**
It's the standard, well-documented choice for exact and approximate vector similarity search, with a small, understandable API (`IndexFlatIP`, `.add()`, `.search()`) rather than the abstraction layers a full vector database would add. For a single document's worth of chunks (tens to a few hundred), exact brute-force search is fast enough that there's no need for an approximate index — simplicity wins.

**Why Groq?**
Free tier, and extremely fast inference — which matters directly for a live demo. It's used *only* for the final generation step; every other stage of the pipeline runs locally.

**Why cosine similarity (via normalized inner product)?**
Cosine similarity measures the *angle* between two vectors, not their magnitude — which is exactly what's wanted when comparing meaning: two chunks about the same topic should be "close" regardless of how long they are or how confidently-worded they are. FAISS has no dedicated cosine-similarity index type, so the standard technique is used: normalize every vector to unit length before indexing and before searching. Once every vector has length 1, a plain inner product between two of them is mathematically identical to their cosine similarity.

---

## Chunking Strategy

**Defaults: chunk size 800 characters, overlap 150 characters.**

- **800 characters** (roughly 150–180 tokens) is small enough that five retrieved chunks comfortably fit in the LLM's context window alongside the system prompt and question, and large enough to usually contain a complete idea — a full sentence or two, sometimes a short paragraph — rather than a meaningless fragment.
- **150 characters of overlap** (about 19% of chunk size) protects against the failure mode where the sentence that actually answers a question sits exactly on a chunk boundary and gets split across two chunks, neither of which alone contains the full fact. It's large enough to catch that case without so much overlap that the index balloons with near-duplicate content.

**Trade-offs**: smaller chunks retrieve more precisely but lose surrounding context; larger chunks keep more context but dilute similarity scores (a chunk about five different topics won't score highly for any single one) and cost more tokens per retrieved chunk. These are the values a student project should ship with by default — both are fully configurable per-request via the API and the UI's Settings panel.

**Boundary strategy**: rather than cutting text every N characters regardless of what's there, the chunker prefers paragraph boundaries first, falls back to sentence boundaries for paragraphs that are still too long, and falls back to word-level packing only as a last resort for a run of text with no real punctuation. Chunking is also done **per page**, never across a page boundary — trading a small amount of cross-page context for a citation guarantee that every chunk maps to exactly one real page number.

---

## Embedding Model

| | |
|---|---|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Dimensions | 384 |
| Size | ~80MB |
| Speed | Real-time on CPU; no GPU required |
| Cost | Free, runs entirely locally after the first download |

**Semantic quality**: strong for its size on general-purpose sentence/short-passage similarity — this is demonstrated live in `demo_pipeline.py`, where a query using none of the document's exact wording (*"What percentage of classes do I need to attend?"*) still correctly retrieves the chunk about *"75% attendance"* over unrelated chunks.

**Limitations**: it's a general-purpose model, not fine-tuned for any particular domain (legal, medical, code) — for a domain-specific deployment, a specialized embedding model would likely retrieve more accurately. It also has a fairly short effective input length; extremely long chunks may have their embedding quality degrade, which is part of why chunk size is capped at a modest 800 characters by default.

---

## Vector Store

FAISS `IndexFlatIP`, storing L2-normalized vectors so inner product search is equivalent to cosine similarity search (see [Why These Technologies](#why-these-technologies) above for the full reasoning). This is an **exact** search — every stored vector is compared, not approximated — which is the correct trade-off at the scale of a single document's chunks, and is simple enough to fully explain without needing to discuss approximate-nearest-neighbor algorithms like IVF or HNSW.

Chunk metadata (text, filename, page number, chunk ID) is stored separately, in a plain JSON file, position-aligned with FAISS's own sequential vector IDs — `metadata[i]` always describes exactly the vector FAISS knows as id `i`. Three files are saved per processed document, under `data/indexes/{document_id}/`:

- `index.faiss` — the actual vectors
- `metadata.json` — chunk text, filename, and page number, aligned by position
- `manifest.json` — dimension, embedding model name, vector count, timestamp

---

## Retrieval

A user's question is embedded with the same model used for the document's chunks, normalized the same way, and searched against the FAISS index for the top-K most similar chunks — K defaults to 5, and is configurable per request. If a document has fewer chunks than the requested K, the result is silently clamped to however many chunks actually exist, rather than erroring. Every result carries its real cosine similarity score, which is surfaced directly in the API response and the UI's source cards — nothing here is hidden or approximated after the fact.

---

## Generation

The retrieved chunks and the user's question are assembled into a prompt with three parts: a system message defining strict grounding rules, a `DOCUMENT CONTEXT` block listing every retrieved chunk (each labeled with its source filename, page number, and similarity score), and the `USER QUESTION` itself. This whole prompt is sent to Groq (`openai/gpt-oss-120b` by default) at a low temperature (0.1) — deliberately, since this is a fact-retrieval task, not a creative one, so the model's most consistent, least "creative" completion is what's wanted.

If retrieval returns **zero** chunks at all (nothing in the index was even remotely relevant), the app returns the "not found" message directly, without calling the LLM — there's nothing for it to work with, so that call would just be paying for a coin-flip about whether the model correctly declines. Deciding that deterministically in code is both cheaper and more reliable.

---

## Grounding / Hallucination Control

The system prompt (in `services/generator.py`) instructs the model, explicitly:

1. Use only information present in the given `DOCUMENT CONTEXT` — never outside knowledge, however confident.
2. Never invent facts, numbers, names, or dates not stated in the context.
3. If the context doesn't contain the answer, say exactly: *"I couldn't find enough information in the provided document to answer that question."*
4. Keep answers concise.
5. Refer to page numbers using the source labels given in the context.
6. Never claim to have read anything beyond what was actually retrieved.

This was verified with a real, adversarial test during development: asking a processed document's Q&A endpoint *"What is the capital of France?"* — a question the model obviously "knows" the answer to from its own training, but which the document has nothing to do with. Retrieval correctly returned its top-3 nearest chunks (retrieval always returns *something*, that's how nearest-neighbor search works), but with near-zero similarity scores (0.00–0.03, versus 0.5+ for genuinely relevant questions). The model correctly declined to answer rather than falling back on outside knowledge — proof the grounding prompt constrains behavior, not just a claim that it does.

No page number shown in a response is ever generated by the LLM directly — every citation traces back to real metadata retrieved from the FAISS index, so a fabricated citation isn't structurally possible.

---

## Project Structure

```
rag-knowledge-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app, CORS, error handlers, frontend mount
│   │   ├── config.py                # Settings, loaded from environment variables
│   │   ├── api/
│   │   │   ├── health.py            # GET /health
│   │   │   ├── upload.py            # POST /documents/upload, /documents/process
│   │   │   ├── query.py             # POST /query
│   │   │   └── documents.py         # GET /documents, DELETE /documents/{id}
│   │   ├── services/
│   │   │   ├── document_loader.py   # PDF/TXT/MD -> text + page metadata
│   │   │   ├── chunker.py           # text -> overlapping chunks
│   │   │   ├── embeddings.py        # chunks/query -> vectors
│   │   │   ├── vector_store.py      # FAISS build/search/save/load
│   │   │   ├── retriever.py         # query embed + FAISS search -> ranked chunks
│   │   │   ├── generator.py         # grounded prompt + Groq call
│   │   │   └── document_registry.py # JSON-backed document tracking
│   │   ├── models/
│   │   │   └── schemas.py           # Pydantic request/response models
│   │   └── utils/
│   │       └── helpers.py           # filename sanitization, id generation/validation
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js                       # plain JS, no framework, no build step
├── data/
│   ├── uploads/                     # gitignored, created at runtime
│   └── indexes/                     # gitignored, created at runtime
├── tests/
│   ├── test_document_loader.py
│   ├── test_chunker.py
│   ├── test_embeddings.py
│   ├── test_vector_store.py
│   ├── test_retrieval.py
│   ├── test_generator.py
│   ├── test_health.py
│   ├── test_api.py
│   └── test_security.py
├── demo_pipeline.py                 # standalone terminal demo of the full pipeline
├── README.md
├── .gitignore
└── LICENSE
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check; reports configured embedding model, vector store, and whether a Groq key is set (never the key itself) |
| `POST` | `/documents/upload` | Upload a PDF/TXT/MD file (multipart). Returns a `document_id`. Does not process it. |
| `POST` | `/documents/process` | Runs the full pipeline (extract, chunk, embed, index) for a previously uploaded document. Accepts optional `chunk_size` / `chunk_overlap` overrides. |
| `POST` | `/query` | Ask a question about a processed document. Returns `answer` plus `sources` (page number, similarity, chunk text). |
| `GET` | `/documents` | List all uploaded/processed documents and their stats. |
| `DELETE` | `/documents/{document_id}` | Remove a document's uploaded file, FAISS index, and registry entry. |

Full request/response schemas, with a "Try it out" button for every endpoint, are at `/docs`.

---

## Example Usage

**Question:** *"What is the attendance policy?"*

**Answer:**
> Students must maintain at least 75% attendance in each course to be eligible to sit for the end-semester examination. (Source: handbook.pdf, Page 1)

**Sources:**
- Page 1 — similarity 0.61
- Page 1 — similarity 0.60
- Page 1 — similarity 0.34

**Question (deliberately unrelated to the document):** *"What is the capital of France?"*

**Answer:**
> I couldn't find enough information in the provided document to answer that question.

---

## Limitations

Stated plainly, not glossed over:

- **The document registry is a plain JSON file**, not a real database — no transactions, and not safe under concurrent writes from multiple server processes. Fine for a single-user, single-process student deployment; would need to move to SQLite or Postgres for anything beyond that.
- **No OCR** — a scanned, image-only PDF with no real text layer will fail to extract any text.
- **Single-document retrieval only** — questions are answered from one document at a time, not across a whole library.
- **No conversation memory** — each question is answered independently; there's no follow-up-question context carried between turns.
- **The embedding model is general-purpose**, not fine-tuned for any specific domain (legal, medical, technical documentation) — retrieval accuracy on highly specialized text may be lower than a domain-specific model would achieve.
- **No rate limiting or authentication** — appropriate for a local/demo deployment, not for a public-facing production service without adding both.
- **Citations point to page numbers, not exact text spans within a page** — a citation says "this came from page 4," not "this came from these three specific sentences on page 4."

---

## Future Improvements

- **Reranking** — a second-stage cross-encoder pass over the top-K retrieved chunks, for higher precision than a single embedding-similarity pass alone.
- **Hybrid search** — combining this project's semantic (embedding) search with traditional keyword/BM25 search, which handles exact-term queries (product codes, names) that pure semantic search sometimes underperforms on.
- **OCR** for scanned PDFs, via something like Tesseract, as a fallback when a PDF's text layer is empty.
- **A persistent vector database** (e.g. Qdrant, Weaviate, or pgvector) in place of local FAISS files, for multi-user or multi-server deployments.
- **Multi-document / cross-document retrieval** — answering a question using chunks pulled from several documents at once, not just the currently active one.
- **Conversation memory** — letting a follow-up question ("what about the second point?") resolve against the previous turn's context.
- **Finer-grained citations** — highlighting the exact sentence(s) within a cited page that support a claim, not just the page number.

---
