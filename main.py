from embedding_entire_page import ChunkEmbedder
from top_wiki_pages import HybridRetriever
from hybrid_chunk_retrieval import HybridChunkRetriever
from top_non_wiki_pages import GoogleCSERetriever
from LLM_output import AnswerSummary
import sys, io, json, os
import time
import gc
import torch
from dotenv import load_dotenv #type:ignore
load_dotenv()
api_key = os.getenv("API_KEY")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
engine = GoogleCSERetriever(api_key, top_k=3)
closest_websites = HybridRetriever(top_k=3, N_DENSE=5, N_SPARSE=3, k_rpf=60, alpha=0.6)
chunks_websites = ChunkEmbedder(chunk_size=150, chunk_overlap=30)

def generate_wiki_answers(query):
    websites = closest_websites.retrieve(query) # FINDS THE BEST WIKIPEDIA WEBSITES FOR THE query
    faiss, sparse_matrix, meta_data = chunks_websites.process(websites) # EMBED THESE WEBSITES CONTENT
    relevant_chunk = HybridChunkRetriever(chunk_metadata=meta_data, faiss_index=faiss, tfidf_matrix=sparse_matrix, n_dense=5, n_sparse=3, top_k=3, alpha=0.6)
    passage = relevant_chunk.retrieve(query) # FIND THE MOST RELEVANT CHUNK OF THE EMBEDDED WEBSITES
    relevant_chunk.cleanup()
    print("WIKI PASSAGED DONE")
    faiss, sparse_matrix, meta_data, relevant_chunk, websites = None, None, None, None, None
    gc.collect()
    torch.cuda.empty_cache()
        
    return passage ## USE FOR TRUSTED SOURCES WHILE NON WIKI CANNOT BE TRUSTED

def generate_nonwiki_answers(query):
    websites = engine.get_top_urls(query) # FINDS THE BEST WEBSITES FOR THE query
    faiss, sparse_matrix, meta_data = chunks_websites.process(websites) # EMBED THESE WEBSITES CONTENT
    relevant_chunk = HybridChunkRetriever(chunk_metadata=meta_data, faiss_index=faiss, tfidf_matrix=sparse_matrix, n_dense=5, n_sparse=3, top_k=3, alpha=0.6)
    passage = relevant_chunk.retrieve(query) # FIND THE MOST RELEVANT CHUNK OF THE EMBEDDED WEBSITES
    relevant_chunk.cleanup()
    print("NONWIKI PASSAGED DONE")
    websites, faiss, sparse_matrix, meta_data, relevant_chunk = None, None, None, None, None
    gc.collect()
    torch.cuda.empty_cache()

    return passage

def process_query(query):
    wiki_passage = generate_wiki_answers(query)
    nonwiki_passage = generate_nonwiki_answers(query)
    merged_passage = {
        "query": wiki_passage["query"],
        "merged_contexts": wiki_passage["merged_contexts"] + nonwiki_passage["merged_contexts"]
    }
    return merged_passage   

def empty_cache():
    global closest_websites, engine, chunks_websites
    closest_websites.cleanup()
    closest_websites = None
    engine.cleanup()
    engine = None
    chunks_websites.cleanup()
    chunks_websites = None
    gc.collect()
    torch.cuda.empty_cache()
    print("CACHE EMPTIED")

def summarize(summary_LLM, context, query, i):
    answer = summary_LLM.process_contexts(query, context) # USES LLM TO SUMMARIZE THE ANSWER

    output_path = f"answers/{i}.txt"
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

            f.write(f"query: {query}\n")
            f.write(f"Link: {link}\n")
            f.write(f"Model Output:\n{output}\n")
            f.write("-----------------------------------------------------------------------------------------------\n\n")

    print(f"Saved to {output_path}")

def load_queries_from_file(file_path="queries.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]
    return queries


queries = load_queries_from_file()
contexts = []
start = time.time()
for query in queries:
    print(query)
    contexts.append(process_query(query))
end = time.time()
print("All Queries passages are made")
print("="*60)
print(f"TOTAL TIME TAKEN TO SEACRCH TEXTS: {start-end}")
print("="*60)


empty_cache()
summary_LLM = AnswerSummary()
print("Model is loaded")

for i, context in enumerate(contexts):
    summarize(summary_LLM, context, queries[i], i)
summary_LLM.cleanup()

print("Done")