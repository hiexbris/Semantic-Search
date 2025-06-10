from embedding_entire_page import ChunkEmbedder
from top_wiki_pages import HybridRetriever
from hybrid_chunk_retrieval import HybridChunkRetriever
from LLM_output import AnswerHighlighter
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

QUERY = "How does the James Webb Space Telescope work?"
print("=" * 60)
print(QUERY)
print("=" * 60)

# closest_websites = HybridRetriever(top_k=10, N_DENSE=30, N_SPARSE=50, k_rpf=60, alpha=0.6)
# websites = closest_websites.retrieve(QUERY)

# print("=" * 60)
# print(websites)
# print("=" * 60)

# chunks_websites = ChunkEmbedder(chunk_size=150, chunk_overlap=30)
# faiss, sparse_matrix, meta_data = chunks_websites.process(websites)

# print("=" * 60)
# print("Chunk Embdding for relevant websites completed, Making Passage now")
# print("=" * 60)

# relevant_chunk = HybridChunkRetriever(chunk_metadata=meta_data, faiss_index=faiss, tfidf_matrix=sparse_matrix, n_dense=30, n_sparse=30, top_k=3, alpha=0.6)
# passage = relevant_chunk.retrieve(QUERY)

# print("=" * 60)
# print(passage)
# print("=" * 60)

# # Save dictionary to JSON file
# with open("temp/passage.json", "w", encoding="utf-8") as f:
#     json.dump(passage, f, indent=2, ensure_ascii=False)

# Load it back
with open("temp/passage.json", "r", encoding="utf-8") as f:
    passage = json.load(f)
print("data loaded")

highlighter_LLM = AnswerHighlighter()
answer = highlighter_LLM.process_contexts(QUERY, passage)

print("=" * 60)
print(answer)
print("=" * 60)