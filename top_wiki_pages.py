import json
import numpy as np
import faiss  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
import joblib  # type: ignore
from scipy.sparse import csr_matrix  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

class HybridRetriever:
    def __init__(self, top_k, N_DENSE=30, N_SPARSE=100, k_rpf=60, alpha=0.6):
        self.top_k = top_k
        self.N_DENSE = N_DENSE
        self.N_SPARSE = N_SPARSE
        self.k_rpf = k_rpf
        self.alpha = alpha

        # Load precomputed assets
        self.faiss_index = faiss.read_index("dataset/wiki_dense_faiss.index")
        self.metadata = json.load(open("dataset/wiki_metadata.json", encoding="utf-8"))
        self.tfidf_vectorizer = joblib.load("dataset/tfidf_vectorizer.joblib")
        self.tfidf_matrix: csr_matrix = joblib.load("dataset/tfidf_matrix.joblib")
        self.dense_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    def retrieve(self, query):
        # === Dense Retrieval ===
        dense_query = self.dense_model.encode([query], normalize_embeddings=True)
        d_scores, d_indices = self.faiss_index.search(dense_query.astype("float32"), self.N_DENSE)
        d_indices = d_indices[0]

        # === Sparse Retrieval ===
        sparse_query = self.tfidf_vectorizer.transform([query])
        s_scores = cosine_similarity(sparse_query, self.tfidf_matrix).flatten()
        top_s_indices = np.argsort(s_scores)[::-1][:self.N_SPARSE]

        # === RRF Fusion ===
        rrf_scores = {}
        rank_info = {}

        for rank, idx in enumerate(d_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + (self.alpha) * (1 / (self.k_rpf + rank))
            if idx not in rank_info:
                rank_info[idx] = {}
            rank_info[idx]["dense_rank"] = rank + 1

        for rank, idx in enumerate(top_s_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + (1 - self.alpha) * (1 / (self.k_rpf + rank))
            if idx not in rank_info:
                rank_info[idx] = {}
            rank_info[idx]["sparse_rank"] = rank + 1

        # === Final Ranking ===
        top_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
        top_titles = [f"https://en.wikipedia.org/wiki/{self.metadata[idx]['title'].replace(' ', '_')}" for idx, _ in top_rrf] ##THIS LINES IS CHANGED FOR EXTRA WEBSITES

        # === Save JSON Outputs ===
        output = {"query": query, "top_results": top_titles}
        with open("temp/hybrid_top_results.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        detailed_output = {
            "query": query,
            "ranked_results": []
        }

        for final_rank, (idx, score) in enumerate(top_rrf, start=1):
            entry = {
                "rank": final_rank,
                "title": self.metadata[idx]["title"],
                "rrf_score": round(score, 5),
                "dense_rank": rank_info.get(idx, {}).get("dense_rank", None),
                "sparse_rank": rank_info.get(idx, {}).get("sparse_rank", None)
            }
            detailed_output["ranked_results"].append(entry)

        with open("temp/hybrid_top_results_with_ranks.json", "w", encoding="utf-8") as f:
            json.dump(detailed_output, f, indent=2, ensure_ascii=False)

        print("Top results saved to hybrid_top_results_with_ranks.json")
        return output

    def cleanup(self):
        self.top_k = None
        self.N_DENSE = None
        self.N_SPARSE = None
        self.k_rpf = None
        self.alpha = None
        self.faiss_index = None
        self.metadata = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix: csr_matrix = None
        self.dense_model = None

