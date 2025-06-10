import json
import faiss # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
import joblib # type: ignore
import time

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

# Paths
json_file = "dataset/top_25000_wiki_summaries.json"  # Your JSON with title, views, summary
faiss_index_path = "dataset/wiki_dense_faiss.index"
tfidf_vectorizer_path = "dataset/tfidf_vectorizer.joblib"
tfidf_matrix_path = "dataset/tfidf_matrix.joblib"
metadata_path = "dataset/wiki_metadata.json"

def load_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_dense_embeddings(summaries, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    summaries = [s if s and s.strip() else " " for s in summaries]
    embeddings = model.encode(summaries, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype('float32')  # FAISS needs float32

def create_tfidf_matrix(summaries):
    try:
        summaries = [s if s else "" for s in summaries]
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(summaries)
        return vectorizer, tfidf_matrix
    except Exception as e:
        print(f"[ERROR] TF-IDF creation failed: {e}")
        return None, None

def save_faiss_index(embeddings, index_path):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity with normalized vectors
    index.add(embeddings)
    try:
        faiss.write_index(index, index_path)
        print(f"[INFO] FAISS index saved successfully to {index_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save FAISS index: {e}")

def save_metadata(data, metadata_path):
    # Save only title, views and an index id for retrieval mapping
    metadata = [{'id': i, 'title': title, 'views': data[title]["views"]} for i, title in enumerate(data)]
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def main():
    start = time.time()
    data = load_data(json_file)
    summaries = [data[title]["summary"] for title in data]

    print("Creating dense embeddings...")
    dense_embeddings = create_dense_embeddings(summaries)

    print("Creating TF-IDF matrix...")
    tfidf_vectorizer, tfidf_matrix = create_tfidf_matrix(summaries)

    print("Saving FAISS index...")
    save_faiss_index(dense_embeddings, faiss_index_path)

    print("Saving TF-IDF vectorizer and matrix...")
    joblib.dump(tfidf_vectorizer, tfidf_vectorizer_path)
    joblib.dump(tfidf_matrix, tfidf_matrix_path)

    print("Saving metadata...")
    save_metadata(data, metadata_path)

    print("Done! All data saved for retrieval.")
    end = time.time()
    print(end-start)

if __name__ == "__main__":
    main()
