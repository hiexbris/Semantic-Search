from embedding_entire_page import ChunkEmbedder
from top_wiki_pages import HybridRetriever
from hybrid_chunk_retrieval import HybridChunkRetriever
from top_non_wiki_pages import GoogleCSERetriever
from LLM_output import AnswerSummary
import sys, io, json
import time
import gc
import torch
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', line_buffering=True)
engine = GoogleCSERetriever(api_key="AIzaSyDLVoQR9QPYNupSs30w5jwOPJPEmRpdhbw", search_engine_id="b4978ed8b79894c9d", top_k=5)
closest_websites = HybridRetriever(top_k=4, N_DENSE=5, N_SPARSE=3, k_rpf=60, alpha=0.6)
chunks_websites = ChunkEmbedder(chunk_size=150, chunk_overlap=30)

### RATHER THAN OUTPUTTING MULTIPLE ANSWERS IT ONLY OUTPUTS SINGLE ANSWER FOR EVALUATION PRUPOSES WHILE THE MAIN FILE OUTPUTS MUTLIPLE ANSWER PER QUERY FROM DIFFRENT SOURCES FOR USER TO CHOOSE

def answers(query):
    websites1 = closest_websites.retrieve(query) # FINDS THE BEST WIKIPEDIA WEBSITES FOR THE query
    websites2 = engine.get_top_urls(query) # FINDS THE BEST WEBSITES FOR THE query

    websites = {"query": websites1["query"],
                "top_results": websites1["top_results"] + websites2["top_results"]}
    
    faiss, sparse_matrix, meta_data = chunks_websites.process(websites) # EMBED THESE WEBSITES CONTENT
    relevant_chunk = HybridChunkRetriever(chunk_metadata=meta_data, faiss_index=faiss, tfidf_matrix=sparse_matrix, n_dense=5, n_sparse=3, top_k=3, alpha=0.6) # ONLY RETURNS ONE PASSAGE WHICH IS AT THE TOP
    passage = relevant_chunk.retrieve(query) # FIND THE MOST RELEVANT CHUNKS OF THE EMBEDDED WEBSITES
    relevant_chunk.cleanup()
    print("PASSAGE DONE")
    websites, faiss, sparse_matrix, meta_data, relevant_chunk = None, None, None, None, None
    gc.collect()
    torch.cuda.empty_cache()

    if "merged_contexts" in passage:
        all_text = " ".join([ctx["passage"] for ctx in passage["merged_contexts"]])
        passage["merged_contexts"] = [{
            "title": passage["merged_contexts"][0]["title"] if passage["merged_contexts"] else "",
            "passage": all_text.strip()
        }]

    return passage

def process_query(query):
    passage = answers(query)
    return passage   

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

    output_path = f"semantic/{i}.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in answer:
            summary = entry.get('summary', '')

            # Split by all [/INST] tags and take content after the last one
            if '[/INST]' in summary:
                parts = summary.split('[/INST]')
                output = parts[-1].strip()
            else:
                output = summary.strip()  # fallback if no [/INST]

            f.write(output) # SAVES ONLY THE OUTPUT NOTHING ELSE AS IT IS ONLY NEEDED FOR EVALUATION

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