"""
scripts/ingest_kb.py

One-time ingestion script: loads knowledge base text files, chunks them,
embeds with sentence-transformers, and stores in ChromaDB.
Run this once before starting the API server.
"""

import os
import sys
from pathlib import Path

# Add project root to path so imports work from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from sentence_transformers import SentenceTransformer

KB_DIR = Path(__file__).parent.parent / "data" / "knowledge_base"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "pediatric_kb"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Chunking parameters: 200 tokens ≈ 150 words; overlap = 20 tokens ≈ 15 words
CHUNK_SIZE_WORDS = 150
OVERLAP_WORDS = 15


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE_WORDS, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        # Move forward by chunk size minus overlap
        start += CHUNK_SIZE_WORDS - OVERLAP_WORDS
    return chunks


def load_kb_files() -> list[dict]:
    """Load all .txt files from the knowledge base directory."""
    documents = []
    for filepath in sorted(KB_DIR.glob("*.txt")):
        text = filepath.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            documents.append({
                "id": f"{filepath.stem}_chunk_{i}",
                "text": chunk,
                "source": filepath.name,
            })
    return documents


def get_existing_count(collection) -> int:
    """Return number of documents already in the collection."""
    return collection.count()


def ingest():
    """Main ingestion function — idempotent."""
    print("=== Mumzworld Pediatric KB Ingestion ===\n")

    # Load documents
    documents = load_kb_files()
    if not documents:
        print(f"ERROR: No .txt files found in {KB_DIR}")
        sys.exit(1)

    print(f"Files found: {len(set(d['source'] for d in documents))}")
    print(f"Total chunks to ingest: {len(documents)}")

    # Connect to ChromaDB (creates directory if needed)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Idempotency check: skip if collection already has same chunk count
    existing = get_existing_count(collection)
    if existing == len(documents):
        print(f"\nCollection already up-to-date ({existing} chunks). Skipping re-ingestion.")
        print("Done.")
        return

    if existing > 0:
        print(f"\nExisting collection has {existing} chunks (expected {len(documents)}). Re-ingesting...")
        # Delete and recreate for clean state
        client.delete_collection(COLLECTION_NAME)
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # Load embedding model (downloads on first run, cached after)
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    # Embed all chunks
    print("Embedding chunks...")
    texts = [d["text"] for d in documents]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    # Store in ChromaDB in batches
    print("\nStoring in ChromaDB...")
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        collection.add(
            ids=[d["id"] for d in batch],
            documents=[d["text"] for d in batch],
            embeddings=embeddings[i : i + batch_size].tolist(),
            metadatas=[{"source": d["source"]} for d in batch],
        )

    final_count = collection.count()
    print(f"\n=== Ingestion Complete ===")
    print(f"Files processed: {len(set(d['source'] for d in documents))}")
    print(f"Chunks created:  {len(documents)}")
    print(f"Collection size: {final_count}")
    print(f"ChromaDB path:   {CHROMA_DIR}")
    print("\nYou can now start the API server: uvicorn app.main:app --reload")


if __name__ == "__main__":
    ingest()
