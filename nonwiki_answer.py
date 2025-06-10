from top_non_wiki_pages import GoogleCSERetriever
from embedding_entire_page import ChunkEmbedder
from hybrid_chunk_retrieval import HybridChunkRetriever
from LLM_output import AnswerSummary
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)

QUERY = "How does the James Webb Space Telescope work?"

# print("=" * 60)
# print(QUERY)
# print("=" * 60)

# engine = GoogleCSERetriever(api_key="AIzaSyDLVoQR9QPYNupSs30w5jwOPJPEmRpdhbw", search_engine_id="b4978ed8b79894c9d", top_k=5)
# websites = engine.get_top_urls(QUERY)

# print("=" * 60)
# print(websites)
# print("=" * 60)

# chunks_websites = ChunkEmbedder(chunk_size=150, chunk_overlap=30)
# faiss, sparse_matrix, meta_data = chunks_websites.process(websites)

# print("=" * 60)
# print("Chunk Embdding for relevant websites completed, Making Passage now")
# print("=" * 60)

# relevant_chunk = HybridChunkRetriever(chunk_metadata=meta_data, faiss_index=faiss, tfidf_matrix=sparse_matrix, n_dense=20, n_sparse=20, top_k=3, alpha=0.6)
# passage = relevant_chunk.retrieve(QUERY)

# print("=" * 60)
# print(passage)
# print("=" * 60)

# Save dictionary to JSON file
# with open("temp/nonwiki_passage.json", "w", encoding="utf-8") as f:
#     json.dump(passage, f, indent=2, ensure_ascii=False)

# # Load it back
with open("temp/nonwiki_passage.json", "r", encoding="utf-8") as f:
    passage = json.load(f)
print("data loaded")

highlighter_LLM = AnswerSummary()
answer = highlighter_LLM.process_contexts(QUERY, passage)

print("=" * 60)
print(answer)
print("=" * 60)

output_path = "final_outputs.txt"
with open(output_path, "w", encoding="utf-8") as f:
    for entry in answer:
        query = entry.get('query', '').strip()
        link = entry.get('link', '').strip()
        summary = entry.get('summary', '')

        # Split by all [/INST] tags and take content after the last one
        if '[/INST]' in summary:
            parts = summary.split('[/INST]')
            output = parts[-1].strip()
        else:
            output = summary.strip()  # fallback if no [/INST]

        f.write(f"Query: {query}\n")
        f.write(f"Link: {link}\n")
        f.write(f"Model Output:\n{output}\n")
        f.write("---\n\n")
    
highlighter_LLM.cleanup()