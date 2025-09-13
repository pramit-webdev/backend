# vectorstore_hf.py
import faiss
import numpy as np
import os
import csv
from io import StringIO
import requests

class VectorStoreHF:
    def __init__(self, index_file="faiss_index.idx", texts_file="faiss_texts.npy", hf_token=None, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.model_name = model_name
        self.index_file = index_file
        self.texts_file = texts_file
        self.index = None
        self.texts = []

        # Load existing index
        if os.path.exists(self.index_file) and os.path.getsize(self.index_file) > 0:
            try:
                self.index = faiss.read_index(self.index_file)
                if os.path.exists(self.texts_file) and os.path.getsize(self.texts_file) > 0:
                    self.texts = list(np.load(self.texts_file, allow_pickle=True))
            except Exception as e:
                print(f"[VectorStoreHF] Failed to load index: {e}")
                self.index = None
                self.texts = []

    def _get_embeddings(self, texts, batch_size=16):
        """Generate embeddings using HF Inference API."""
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = {"inputs": batch}
            url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_name}"
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            batch_embeddings = response.json()  # list of lists
            # Ensure numpy float32
            embeddings.extend([np.array(e, dtype=np.float32) for e in batch_embeddings])
        return np.array(embeddings).astype("float32")

    def add_texts(self, texts):
        embeddings = self._get_embeddings(texts)

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

    def add_csv(self, csv_content):
        """Convert CSV content to text chunks and add to vector store."""
        text_rows = []
        f = StringIO(csv_content.decode("utf-8"))
        reader = csv.reader(f)
        headers = next(reader, None)
        for row in reader:
            row_text = ", ".join(f"{h}: {v}" for h, v in zip(headers, row))
            text_rows.append(row_text)
        self.add_texts(text_rows)
