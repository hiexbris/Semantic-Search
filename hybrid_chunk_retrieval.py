import json
import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from collections import defaultdict
import joblib  # type: ignore

class HybridChunkRetriever:
    def __init__(self, chunk_metadata, faiss_index, tfidf_matrix, n_dense=30, n_sparse=100, top_k=1, alpha=0.6, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.chunk_metadata = chunk_metadata
        self.faiss_index = faiss_index
        self.tfidf_matrix = tfidf_matrix
        self.vectorizer = joblib.load("dataset/tfidf_vectorizer.joblib")
        self.model = SentenceTransformer(model_name)
        self.index_to_chunk = {i: chunk for i, chunk in enumerate(chunk_metadata)}
        self.n_dense = n_dense
        self.n_sparse = n_sparse
        self.top_k = top_k
        self.alpha = alpha

    def retrieve(self, query):
        print("Encoding query...")
        dense_query = self.model.encode([query], normalize_embeddings=True)[0].astype('float32')
        sparse_query = self.vectorizer.transform([query])

        print("Running dense retrieval...")
        _, dense_indices = self.faiss_index.search(dense_query.reshape(1, -1), self.n_dense)

        print("Running sparse retrieval...")
        sparse_scores = cosine_similarity(sparse_query, self.tfidf_matrix).flatten()
        sparse_indices = np.argsort(sparse_scores)[-self.n_sparse:][::-1]

        print("Combining using Reciprocal Rank Fusion (RRF)...")
        rank_scores = defaultdict(float)
        for rank, idx in enumerate(dense_indices[0]):
            rank_scores[idx] += self.alpha / (60 + rank)
        for rank, idx in enumerate(sparse_indices):
            rank_scores[idx] += (1 - self.alpha) / (60 + rank)

        top_indices = sorted(rank_scores, key=rank_scores.get, reverse=True)[:self.top_k]

        print("Preparing final contexts...")
        merged_contexts = []
        for idx in top_indices:
            idx = int(idx)
            if idx not in self.index_to_chunk:
                print(f"[WARNING] Invalid index retrieved: {idx}")
                continue  # Skip bad indices instead of crashing
            chunk = self.index_to_chunk[idx]
            merged_contexts.append({
                "title": chunk["title"],
                "passage": chunk["text"].strip()
            })

        output = {
            "query": query,
            "merged_contexts": merged_contexts
        }

        with open("temp/final_contexts_for_llm.json", "w", encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"Saved final context(s) without merging")
        return output

    def cleanup(self):
        self.chunk_metadata = None
        self.faiss_index = None
        self.tfidf_matrix = None
        self.vectorizer = None
        self.model = None
        self.index_to_chunk = None
        self.n_dense = None
        self.n_sparse = None
        self.top_k = None
        self.alpha = None