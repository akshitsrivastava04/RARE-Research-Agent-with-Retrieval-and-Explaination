import os
import hashlib
import time
import psutil
from typing import List, Dict, Any
from dotenv import load_dotenv  
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import SentenceTransformer


class DocumentStore:
    """
    Efficient, production-ready document vector store using Qdrant + SentenceTransformer.
    Loads Qdrant credentials (URL and API key) securely from .env.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        threshold_gb: float = 3.5,
    ):
        # Load environment variables
        load_dotenv()

        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_key = os.getenv("QDRANT_API_KEY", None)

        self.collection_name = collection_name
        self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_key)
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.cache: Dict[str, List[float]] = {}
        self.threshold_gb = threshold_gb

        self._ensure_collection()

    # ---------- Core Setup ---------- #
    def _ensure_collection(self):
        """Ensure the collection exists with efficient settings."""
        if self.collection_name not in [c.name for c in self.qdrant.get_collections().collections]:
            self.qdrant.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(size=384, distance="Cosine"),
                optimizers_config=rest.OptimizersConfigDiff(memmap_threshold=20000),
            )
            # Enable compression (quantization)
            self.qdrant.update_collection(
                collection_name=self.collection_name,
                quantization_config=rest.ScalarQuantization(
                    scalar=rest.ScalarQuantizationConfig(type="int8", quantile=0.99, always_ram=False)
                ),
            )

    # ---------- Embeddings ---------- #
    def _get_embedding(self, text: str) -> List[float]:
        key = hashlib.md5(text.encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        embedding = self.embedder.encode(text, normalize_embeddings=True).tolist()
        self.cache[key] = embedding
        return embedding

    # ---------- Insert Documents ---------- #
    def add_documents(self, documents: List[Dict[str, Any]]):
        points = []
        for doc in documents:
            text = doc["text"]
            embedding = self._get_embedding(text)
            points.append(
                rest.PointStruct(
                    # NEW:
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={"text": text, "metadata": doc.get("metadata", {})},
                )
            )
        self.qdrant.upsert(collection_name=self.collection_name, points=points)
        self._check_and_cleanup()

    # ---------- Search ---------- #
    def search(self, query: str, limit: int = 3) -> List[str]:
        query_vector = self._get_embedding(query)
        results = self.qdrant.search(
            collection_name=self.collection_name, query_vector=query_vector, limit=limit
        )
        return [hit.payload["text"] for hit in results]

    # ---------- Memory Management ---------- #
    def _check_and_cleanup(self):
        mem_used_gb = psutil.virtual_memory().used / (1024 ** 3)
        if mem_used_gb > self.threshold_gb:
            print(f"[⚠️] Memory {mem_used_gb:.2f} GB > {self.threshold_gb} GB. Running cleanup...")
            self._delete_oldest_points()
            self._optimize_index()

    def _delete_oldest_points(self, keep_last: int = 10000):
        scroll = self.qdrant.scroll(collection_name=self.collection_name, limit=keep_last)
        all_ids = [p.id for p in scroll[0]]
        if len(all_ids) > keep_last:
            delete_ids = all_ids[:-keep_last]
            self.qdrant.delete(
                collection_name=self.collection_name,
                points_selector=rest.PointIdsList(points=delete_ids),
            )
            print(f"[🧹] Deleted {len(delete_ids)} old vectors to free space.")

    def _optimize_index(self):
        try:
            self.qdrant.optimize(collection_name=self.collection_name)
            print("[🧩] Index optimization complete.")
        except Exception as e:
            print(f"[!] Optimization failed: {e}")

    # ---------- Utilities ---------- #
    def count(self) -> int:
        info = self.qdrant.get_collection(self.collection_name)
        return info.points_count or 0


# ---------- Example Usage ---------- #
if __name__ == "__main__":
    store = DocumentStore()

    store.add_documents(
        [
            {"text": "The moon orbits the Earth every 27.3 days.", "metadata": {"source": "wiki"}},
            {"text": "The sun is the center of our solar system.", "metadata": {"source": "wiki"}},
        ]
    )

    print("\nTop Matches:")
    print(store.search("What orbits the Earth?"))
