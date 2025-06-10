import requests# type: ignore
import json
import faiss # type: ignore
import numpy as np
import joblib # type: ignore

from sentence_transformers import SentenceTransformer # type: ignore
from bs4 import BeautifulSoup # type: ignore

DENSE_INDEX_PATH = "temp/non_wiki_chunked_dense_faiss.index"
SPARSE_MATRIX_PATH = "temp/nonwiki_chunked_tfidf_matrix.joblib"
METADATA_PATH = "temp/nonwiki_chunked_metadata.json"
TFIDF_VECTORIZER_PATH = "dataset/tfidf_vectorizer.joblib"

class ChunkEmbedder:
    def __init__(self, chunk_size=150, chunk_overlap=30):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.dense_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        self.vectorizer = joblib.load("dataset/tfidf_vectorizer.joblib")

    def fetch_wiki_text(self, url):
        try:
            response = requests.get(url, timeout=10)  # 10 seconds timeout
            if response.status_code != 200:
                print(f"[WARN] Could not fetch page (status {response.status_code}): {url}")
                return ""
            soup = BeautifulSoup(response.text, "html.parser")

            for sup in soup.find_all('sup', {'class': 'reference'}):
                sup.decompose()

            paragraphs = soup.find_all("p")
            return " ".join(p.get_text() for p in paragraphs)
        
        except requests.exceptions.Timeout:
            print(f"[TIMEOUT] Page took too long to load: {url}")
            return ""
        
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to fetch page: {url} | Reason: {e}")
            return ""

    def chunk_text(self, text):
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = words[i:i+self.chunk_size]
            chunks.append(" ".join(chunk))
            i += self.chunk_size - self.chunk_overlap
        return chunks

    def create_dense_embeddings(self, texts):
        return self.dense_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

    def create_sparse_matrix(self, texts):
        return self.vectorizer.transform(texts)

    def process(self, top_results_data):
        urls = top_results_data["top_results"]
        all_chunks = []
        metadata = []

        for url in urls:
            print(f"[INFO] Processing: {url}")
            text = self.fetch_wiki_text(url)
            if not text:
                continue
            chunks = self.chunk_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                metadata.append({
                    "title": url,
                    "chunk_id": i,
                    "text": chunk
                })

        print("[INFO] Creating dense embeddings...")
        dense_embeds = self.create_dense_embeddings(all_chunks)

        print("[INFO] Creating FAISS index...")
        index = faiss.IndexFlatIP(dense_embeds.shape[1])
        index.add(dense_embeds)
        faiss.write_index(index, DENSE_INDEX_PATH)
        

        print("[INFO] Creating sparse matrix...")
        sparse_matrix = self.create_sparse_matrix(all_chunks)
        joblib.dump(sparse_matrix, SPARSE_MATRIX_PATH)

        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print("[DONE] Chunked embeddings and metadata saved.")

        return index, sparse_matrix, metadata
    
    def cleanup(self):
        self.chunk_size = None
        self.chunk_overlap = None
        self.dense_model = None
        self.vectorizer = None
