# backend/app/core/vector_service.py
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

class VectorService:
    def __init__(self):
        self.chroma_client = None
        self.collection = None
        self.embedder = None

    def initialize(self):
        # Local persistent vector database stored under `backend/chroma_db`
        db_path = Path(__file__).resolve().parents[2] / "chroma_db"
        self.chroma_client = chromadb.PersistentClient(path=str(db_path))
        self.collection = self.chroma_client.get_or_create_collection(name="document_requirements")
        
        # Reuse MiniLM transformer
        self.embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        print("VectorService initialized with ChromaDB.")

    def ingest_document(self, doc_id: str, requirements: list[str]):
        """Index all extracted requirements of a PDF document into ChromaDB."""
        if not requirements:
            return

        embeddings = self.embedder.encode(requirements).tolist()
        ids = [f"{doc_id}_req_{i}" for i in range(len(requirements))]
        metadatas = [{"doc_id": doc_id, "text": req} for req in requirements]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=requirements,
            metadatas=metadatas
        )

    def retrieve_context(self, doc_id: str, query_req: str, top_k: int = 5) -> list[str]:
        """Fetch neighboring/similar clauses from the same document."""
        if not doc_id:
            return []

        query_emb = self.embedder.encode([query_req]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=top_k,
            where={"doc_id": doc_id}
        )

        retrieved_docs = results["documents"][0] if results and "documents" in results else []
        # Filter out exact self-matches
        return [d for d in retrieved_docs if d.strip() != query_req.strip()]

vector_service = VectorService()