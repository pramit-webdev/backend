import os
import faiss
import numpy as np
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class VectorStoreOpenAI:
    def __init__(self, dim: int = 1536, index_file: str = "faiss_index.idx", texts_file: str = "faiss_texts.npy"):
        self.dim = dim
        self.index_file = index_file
        self.texts_file = texts_file

        # Load existing index + texts if available
        if os.path.exists(index_file) and os.path.exists(texts_file):
            try:
                self.index = faiss.read_index(index_file)
                self.texts = np.load(texts_file, allow_pickle=True).tolist()
            except Exception as e:
                print(f"[VectorStoreOpenAI] Failed to load persisted index: {e}")
                self.index = faiss.IndexFlatL2(dim)
                self.texts = []
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.texts = []

    def _embed(self, texts):
        """Embed text(s) using OpenAI API."""
        response = client.embeddings.create(
            model="text-embedding-3-small",  # 1536-dim, cheaper than large
            input=texts
        )
        return [d.embedding for d in response.data]

    def add_texts(self, texts):
        """Embed and add texts to FAISS index."""
        vectors = self._embed(texts)
        self.index.add(np.array(vectors, dtype="float32"))
        self.texts.extend(texts)

        # persist to disk
        faiss.write_index(self.index, self.index_file)
        np.save(self.texts_file, np.array(self.texts, dtype=object))

    def search(self, query, k=3):
        """Semantic search over stored texts."""
        if len(self.texts) == 0:
            return []

        query_vec = self._embed([query])
        D, I = self.index.search(np.array(query_vec, dtype="float32"), k)
        return [self.texts[i] for i in I[0] if 0 <= i < len(self.texts)]
