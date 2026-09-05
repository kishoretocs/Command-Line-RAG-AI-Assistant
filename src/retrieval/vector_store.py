from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer
from src.config import CHROMA_DIR, EMBEDDING_MODEL_NAME, COSINE_MIN_SCORE, RETRIEVAL_TOP_K

class VectorStore:
    _instance = None
    _embedder = None
    _collection = None

    def __init__(self):
        if VectorStore._embedder is None:
            VectorStore._embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        if VectorStore._collection is None:
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            VectorStore._collection = client.get_or_create_collection(
                name="cerulean_sections",
                metadata={"hnsw:space": "cosine"}
            )
        self.embedder = VectorStore._embedder
        self.collection = VectorStore._collection

    def search(self, query: str, top_k: int = RETRIEVAL_TOP_K, where: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Runs dense semantic search on query against section chunks.
        Returns list of dicts with: chunk_id, document_id, section_number, score, text, metadata.
        """
        query_embedding = self.embedder.encode([query], normalize_embeddings=True).tolist()
        
        kwargs = {
            "query_embeddings": query_embedding,
            "n_results": top_k
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        hits: List[Dict[str, Any]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return hits

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for idx in range(len(ids)):
            # Convert cosine distance to cosine similarity (1 - distance)
            sim_score = 1.0 - distances[idx]
            if sim_score < COSINE_MIN_SCORE:
                continue

            hits.append({
                "chunk_id": ids[idx],
                "document_id": metas[idx]["document_id"],
                "section_number": metas[idx].get("section_number", ""),
                "section_title": metas[idx].get("section_title", ""),
                "score": sim_score,
                "text": docs[idx],
                "metadata": metas[idx]
            })

        return hits
