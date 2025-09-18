

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.text_chunks = []

    def add_document(self, chunks: list[str]):
        embeddings = self.model.encode(chunks, convert_to_numpy=True)
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        self.text_chunks.extend(chunks)

    def search(self, query: str, top_k: int = 3):
        if self.index is None:
            return []  # avoid crash if no docs yet
        query_emb = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_emb, top_k)
        return [self.text_chunks[i] for i in I[0]]
