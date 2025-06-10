import os
from bert_score import score as bert_score #type:ignore
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction #type:ignore

# Paths to input directories
queries_file = "queries.txt"

# Read the queries to determine number of files
with open(queries_file, "r", encoding="utf-8") as f:
    queries = [line.strip() for line in f if line.strip()]

num_queries = len(queries)

# Prepare results
bleu_scores = []
bert_precision = []
bert_recall = []
bert_f1 = []

# Evaluation loop
for i in range(num_queries):
    with open(os.path.join(f"GPT/{i}.txt"), "r", encoding="utf-8") as f:
        reference = f.read().strip()

    with open(os.path.join(f"semantic/{i}.txt"), "r", encoding="utf-8") as f:
        semantic_output = f.read().strip()

    with open(os.path.join(f"LLM/{i}.txt"), "r", encoding="utf-8") as f:
        llm_output = f.read().strip()

    # BLEU score for semantic and LLM vs GPT reference
    ref_tokens = [reference.split()]
    sem_tokens = semantic_output.split()
    llm_tokens = llm_output.split()

    smooth_fn = SmoothingFunction().method1
    bleu_sem = sentence_bleu(ref_tokens, sem_tokens, smoothing_function=smooth_fn)
    bleu_llm = sentence_bleu(ref_tokens, llm_tokens, smoothing_function=smooth_fn)
    bleu_scores.append((bleu_sem, bleu_llm))

    # BERT score (semantic, llm) vs reference (GPT)
    cands = [semantic_output, llm_output]
    refs = [reference, reference]
    P, R, F1 = bert_score(cands, refs, lang="en", verbose=False)
    bert_precision.append((P[0].item(), P[1].item()))
    bert_recall.append((R[0].item(), R[1].item()))
    bert_f1.append((F1[0].item(), F1[1].item()))

# Average metrics
avg_bleu_sem = sum([b[0] for b in bleu_scores]) / num_queries
avg_bleu_llm = sum([b[1] for b in bleu_scores]) / num_queries

avg_bert_p_sem = sum([p[0] for p in bert_precision]) / num_queries
avg_bert_p_llm = sum([p[1] for p in bert_precision]) / num_queries

avg_bert_r_sem = sum([r[0] for r in bert_recall]) / num_queries
avg_bert_r_llm = sum([r[1] for r in bert_recall]) / num_queries

avg_bert_f1_sem = sum([f[0] for f in bert_f1]) / num_queries
avg_bert_f1_llm = sum([f[1] for f in bert_f1]) / num_queries

print({
    "BLEU": {
        "Semantic vs GPT": avg_bleu_sem,
        "LLM vs GPT": avg_bleu_llm
    },
    "BERTScore (Precision)": {
        "Semantic vs GPT": avg_bert_p_sem,
        "LLM vs GPT": avg_bert_p_llm
    },
    "BERTScore (Recall)": {
        "Semantic vs GPT": avg_bert_r_sem,
        "LLM vs GPT": avg_bert_r_llm
    },
    "BERTScore (F1)": {
        "Semantic vs GPT": avg_bert_f1_sem,
        "LLM vs GPT": avg_bert_f1_llm
    }
})
