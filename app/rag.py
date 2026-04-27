"""
app/rag.py

RAG retrieval module: queries ChromaDB for relevant pediatric knowledge base chunks.
"""

from dataclasses import dataclass
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "pediatric_kb"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Similarity threshold: if max score < this, consider nothing relevant retrieved
SIMILARITY_THRESHOLD = 0.3

# Singleton instances (loaded once at startup)
_client = None
_collection = None
_model = None


@dataclass
class RetrievalResult:
    """Result of RAG retrieval"""
    chunks: list[str]
    scores: list[float]
    retrieved: bool  # False if no relevant context found


def initialize_rag():
    """Initialize ChromaDB and embedding model (call once at startup)."""
    global _client, _collection, _model

    if _collection is not None:
        return  # Already initialized

    _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    _collection = _client.get_collection(name=COLLECTION_NAME)
    _model = SentenceTransformer(EMBED_MODEL)

    print(f"RAG initialized: {_collection.count()} chunks loaded from ChromaDB")


def retrieve_context(query: str, n_results: int = 4) -> RetrievalResult:
    """
    Retrieve relevant context chunks for a query.

    Args:
        query: The symptom description or question
        n_results: Number of top chunks to retrieve

    Returns:
        RetrievalResult with chunks, scores, and retrieved flag
    """
    if _collection is None or _model is None:
        raise RuntimeError("RAG not initialized. Call initialize_rag() first.")

    # Embed the query
    query_embedding = _model.encode(query).tolist()

    # Query ChromaDB (returns cosine distances; lower = more similar)
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    # Extract chunks and distances
    chunks = results["documents"][0] if results["documents"] else []
    distances = results["distances"][0] if results["distances"] else []

    # Convert cosine distance to similarity score (1 - distance)
    scores = [1.0 - d for d in distances]

    # Check if any result meets the similarity threshold
    max_score = max(scores) if scores else 0.0
    retrieved = max_score >= SIMILARITY_THRESHOLD

    # Log retrieval for debugging (not returned to user)
    print(f"RAG retrieval: query='{query[:50]}...', top_score={max_score:.3f}, retrieved={retrieved}")

    return RetrievalResult(
        chunks=chunks,
        scores=scores,
        retrieved=retrieved,
    )
