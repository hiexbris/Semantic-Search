# 🔍 Semantic Search over Wikipedia and Web

A semantic question-answering system that retrieves the most relevant context from Wikipedia and Google search results, then uses a lightweight LLM to generate **fact-grounded answers with references**. This project emphasizes transparency, trust, and zero hallucination — keeping your LLMs up-to-date **without retraining**.

---

## 🌟 Features

- ✅ **Hybrid Retrieval** using both Sentence Transformers and TF-IDF by implementing Reciprocal Rank Fusion (RRF).
- ✅ **Per-Source Chunking** for transparent answers per page
- ✅ **No Hallucinations** — answers come strictly from retrieved data
- ✅ **Low Maintenance** — no retraining required to stay current
- ✅ **BLEU and BERTScore Evaluation**
- ✅ **Modular and Extendable Pipeline**

---

## 📂 Project Overview

### 🧠 Knowledge Construction

- `top_pages.py`  
  Fetches **top 25,000 Wikipedia pages** from the last 5 years based on page views.

- `summary_pages.py`  
  Downloads **summaries** of those Wikipedia pages.

- `embeddings_summary_pages.py`  
  Creates both **dense (Sentence Transformer)** and **sparse (TF-IDF)** embeddings for summaries, stores in a FAISS index.

---

### 🔍 Query-Time Retrieval

- `top_wiki_pages.py`  
  Retrieves **most relevant Wikipedia pages** for a given query using hybrid search. It combines dense (Sentence Transformers) and sparse (TF-IDF) retrieval using Reciprocal Rank Fusion (RRF).

- `top_non_wiki_pages.py`  
  Downloads **top web pages** from Google using SERPAPI.

- `embedding_entire_page.py`  
  Chunks each full page (~150 words/chunk), computes dense and sparse embeddings for each chunk.

- `hybrid_chunk_retrieval.py`  
  Retrieves top chunks for a given query from both Wikipedia and web sources.

---

### 🤖 Answer Generation

- `LLM_output.py`  
  Sends chunks to the LLM to generate **per-source answers**, ensuring traceable and trustworthy output.

- `main.py`  
  Complete pipeline: query → retrieval → chunking → per-site answer generation with source links.

---

### 📈 Evaluation Tools

- `eval_semantic.py`  
  Generates an answer using retrieved passages.

- `eval_LLM.py`  
  Lets the LLM generate an answer without any retrieved context (pure LLM knowledge).

- `eval.py`  
  Compares generated answers against references using **BLEU** and **BERTScore**.

---

## 🗂️ Input/Output Format

- `queries.txt`  
  Contains all queries to be processed. Each line is a separate question.

- `answers/`  
  Final answers generated for each query using the full hybrid pipeline.

- `LLM/`  
  Stores LLM-only answers (no retrieval used).

- `semantic/`  
  Stores answers generated from semantic-retrieved passages.

- `GPT/`  
  Reference answers or answers from another baseline (e.g., GPT-4 baseline).

---

## ⚙️ Performance Notes

> ⏱️ Expect slightly longer response times compared to a plain LLM.

This is due to:

- Real-time **webpage downloads**, requiring delay between API calls.
- On-the-fly **embedding computations** for chunked content.

✅ This can be improved by:

- **Batching embeddings** (require good VRAM)
- Using **premium API keys**
- Implementing caching for frequently queried sites

---

## 💡 Why Use This?

❌ Standard LLMs often hallucinate.  
❌ They can't cite sources.  
❌ Keeping them up-to-date requires **expensive** fine-tuning.

✅ This system keeps your answers grounded in **fresh, real-world data** — without any retraining.  
✅ Ideal for **news bots**, **research assistants**, **academic agents**, and **factual QA systems**.

> "Don’t hallucinate — retrieve, reason, and reference."

