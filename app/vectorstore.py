# vectorstore.py
import faiss
import numpy as np
import os
from groq import Groq

class VectorStore:
    def __init__(self, index_file="faiss_index.idx", texts_file="faiss_texts.npy", groq_api_key=None):
        self.client = Groq(api_key=groq_api_key or os.getenv("GROQ_API_KEY"))
        self.index_file = index_file
        self.texts_file = texts_file

        # Load existing index if exists
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            self.texts = list(np.load(self.texts_file, allow_pickle=True))
        else:
            # We'll use IndexFlatL2 (for MVP) or IndexIVFFlat for larger datasets
            self.index = None  # Will initialize after first embeddings
            self.texts = []

    def _get_embeddings(self, texts):
        embeddings = []
        for t in texts:
            response = self.client.embeddings.create(input=t, model="mixtral")  # or llama3-8b
            embeddings.append(response.data[0].embedding)
        return np.array(embeddings).astype('float32')

    def add_texts(self, texts):
        embeddings = self._get_embeddings(texts)

        # Initialize index if first time
        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)

        self.index.add(embeddings)
        self.texts.extend(texts)

        # Save for persistence
        faiss.write_index(self.index, self.index_file)
        np.save(self.texts_file, np.array(self.texts, dtype=object))

    def search(self, query, k=3):
        if len(self.texts) == 0 or self.index is None:
            return []

        q_emb = self._get_embeddings([query])
        scores, idxs = self.index.search(q_emb, k)
        return [self.texts[i] for i in idxs[0]]
