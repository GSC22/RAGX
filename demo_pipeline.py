"""
Demo script: chains the ENTIRE RAG pipeline built across Phases 1-5 so
you can *see* it work, end to end, in a terminal -- before any FastAPI
server or frontend exists.

What this proves, in order:
  1. We can read a real document and pull out clean text + page numbers.
  2. We can slice that text into overlapping chunks.
  3. We can turn those chunks into meaning-vectors.
  4. We can index those vectors in FAISS, save/reload from disk, and
     search them instantly by meaning.
  5. We can take a real question, retrieve the right chunks, hand them
     to Groq, and get back a grounded answer -- one that refuses to
     answer when the document genuinely doesn't contain the answer,
     rather than making something up.

Run from the project root:
    python demo_pipeline.py
"""

import sys
import tempfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.services.chunker import chunk_document  # noqa: E402
from app.services.document_loader import DocumentLoader  # noqa: E402

SAMPLE_DOCUMENT_TEXT = """Attendance Policy

Students must maintain a minimum of 75% attendance in each course to be
eligible to sit for the end-semester examination. Attendance is calculated
based on the number of classes conducted versus the number of classes
attended by the student throughout the semester.

Grading Policy

The final grade for each course is calculated based on continuous
assessment (40%) and the end-semester examination (60%). Continuous
assessment includes quizzes, assignments, and a mid-semester test.
Students who score below 40% in the end-semester examination will be
awarded a fail grade regardless of their continuous assessment score.

Grievance Redressal

Any student with a grievance regarding their grade may submit a written
appeal to the department within 7 working days of the result being
published. The department committee will review the appeal and respond
within 14 working days.

Library Rules

Students may borrow up to 5 books at a time for a period of 14 days.
Late returns are subject to a fine of 5 rupees per day per book. Reference
books and periodicals cannot be borrowed and must be used within the
library premises.
"""


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    # --- Step 0: create a sample document on disk so the loader has a
    # real file to open, just like it would with a file you upload. ---
    tmp_dir = Path(tempfile.mkdtemp())
    sample_path = tmp_dir / "sample_handbook.txt"
    sample_path.write_text(SAMPLE_DOCUMENT_TEXT)

    # --- Step 1: Document Loader (Phase 1) --------------------------------
    section("STEP 1: DOCUMENT LOADER  -->  reading the raw file")
    loader = DocumentLoader()
    document = loader.load(sample_path)

    print(f"Filename       : {document.filename}")
    print(f"File type      : {document.file_type}")
    print(f"Pages          : {document.page_count}")
    print(f"Characters     : {document.char_count}")

    # --- Step 2: Chunker (Phase 2) -----------------------------------------
    section("STEP 2: CHUNKER  -->  slicing the text into overlapping pieces")
    chunk_size, overlap = 250, 40
    chunks = chunk_document(document, chunk_size=chunk_size, overlap=overlap)

    print(f"Chunk size     : {chunk_size} characters")
    print(f"Overlap        : {overlap} characters")
    print(f"Total chunks   : {len(chunks)}\n")

    for c in chunks:
        preview = c.text.replace("\n", " ")[:70]
        print(f"  [{c.chunk_index}] page={c.page_number} chars={c.char_count:>4}  \"{preview}...\"")

    # --- Step 3: Embeddings (Phase 3) --------------------------------------
    section("STEP 3: EMBEDDINGS  -->  turning each chunk into meaning-vectors")
    try:
        from app.services.embeddings import EmbeddingService
        import numpy as np

        embedder = EmbeddingService()
        print(f"Model          : {embedder.model_name}")
        print("Loading model (first run downloads ~80MB, then it's cached)...")

        vectors = embedder.embed_documents(chunks)
        print(f"Embedding matrix shape : {vectors.shape}  (chunks x dimensions)")
        print(f"Data type              : {vectors.dtype}")

        # --- Step 4: FAISS vector store (Phase 4) --------------------------
        section("STEP 4: FAISS VECTOR STORE  -->  indexing for instant search")
        from app.services.vector_store import VectorStore

        store = VectorStore(dimension=embedder.embedding_dim)
        store.build(chunks, vectors, embedding_model=embedder.model_name)
        print(f"Indexed {store.size} vectors into a FAISS IndexFlatIP index.")

        index_dir = tmp_dir / "demo_index"
        store.save(index_dir)
        print(f"Saved to disk at: {index_dir}")
        print("  - index.faiss    (the actual vectors)")
        print("  - metadata.json  (chunk text/filename/page, aligned by position)")
        print("  - manifest.json  (dimension, model name, timestamp)")

        reloaded = VectorStore.load(index_dir)
        print(f"Reloaded from disk successfully: {reloaded.size} vectors, dimension {reloaded.dimension}")

        # --- Step 5: Retriever (Phase 4) ------------------------------------
        section("STEP 5: RETRIEVER  -->  question in, ranked chunks out")
        from app.services.retriever import Retriever

        retriever = Retriever(embedding_service=embedder, vector_store=reloaded)

        sample_queries = [
            "What percentage of classes do I need to attend?",
            "How many books can I borrow from the library?",
        ]

        for query in sample_queries:
            print(f'\nQuestion: "{query}"')
            results = retriever.retrieve(query, top_k=2)
            for r in results:
                preview = r.text.replace("\n", " ")[:80]
                print(f"  rank={r.rank}  similarity={r.similarity:.3f}  page={r.page_number}  \"{preview}...\"")

        # --- Step 6: Generation (Phase 5) -----------------------------------
        section("STEP 6: GENERATION (Groq)  -->  turning chunks into a real answer")
        try:
            from app.config import settings
            from app.services.generator import Generator

            if not settings.groq_api_key:
                print("GROQ_API_KEY is not set in backend/.env -- skipping the real answer step.")
                print("Get a free key at https://console.groq.com/keys, add it to backend/.env,")
                print("then re-run this script to see a real generated answer.")
            else:
                generator = Generator()
                print(f"Model: {generator.model}\n")

                demo_questions = [
                    "What is the attendance policy?",
                    "What is the capital of France?",  # deliberately NOT in the document
                ]

                for question in demo_questions:
                    print(f'Question: "{question}"')
                    results = retriever.retrieve(question, top_k=3)
                    answer = generator.generate_answer(question, results)

                    print(f"Answer: {answer}")
                    print("Sources used:")
                    for r in results:
                        print(f"  - Page {r.page_number} (similarity {r.similarity:.2f})")
                    print()

                section("NOTE: why the France question matters")
                print(
                    "The document was never about France, yet the retriever still returned\n"
                    "its 3 numerically-closest chunks (retrieval always returns something --\n"
                    "that's just how nearest-neighbor search works). The important part is\n"
                    "what the LLM did with those irrelevant chunks: it should have refused\n"
                    "to answer and said it couldn't find the information, instead of using\n"
                    "outside knowledge to answer 'Paris' anyway. That refusal -- even with\n"
                    "SOME context handed to it -- is the grounding prompt from generator.py\n"
                    "doing its job. If it answered 'Paris', that would be a hallucination\n"
                    "bug worth fixing immediately."
                )

        except Exception as exc:
            print(f"Generation step unavailable: {exc}")
            print("Check that GROQ_API_KEY in backend/.env is a valid key and you have internet access.")

        section("PIPELINE COMPLETE")
        print(
            "That's the full RAG pipeline working end to end in a terminal:\n"
            "  Document -> Text Extraction -> Chunking -> Embeddings -> FAISS\n"
            "  -> Retrieval -> LLM Generation -> Grounded Answer\n\n"
            "What's still missing before this is a real app: FastAPI endpoints\n"
            "(Phase 6) so a website can call this pipeline over HTTP, and an\n"
            "actual frontend (Phase 7) to upload files and chat through a browser."
        )

    except Exception as exc:
        section("STEP 3 SKIPPED")
        print(f"Embedding model unavailable in this environment: {exc}")
        print("Run this same script on your machine (with internet) to see it.")


if __name__ == "__main__":
    main()
