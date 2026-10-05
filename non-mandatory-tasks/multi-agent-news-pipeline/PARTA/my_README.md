# Part A — Chunking & Embeddings

## Overview

In Part A, we built the basic text processing and embedding pipeline for the NewsQA dataset. The main goal was to convert news articles into smaller text chunks and generate embeddings for semantic retrieval.

## What We Did

- Loaded the **NewsQA dataset** from Hugging Face.
- Removed duplicate articles using the article ID.
- Cleaned the article text and removed empty articles.
- Split each article into **sentence-level chunks**.
- Removed very short sentences using a minimum length of **20 characters**.
- Generated embeddings for each chunk using **`all-MiniLM-L6-v2`**.
- Used **batch processing (batch size = 32)** for efficient embedding generation.
- Normalized the embeddings so that cosine similarity can be efficiently calculated using dot product.
- Saved the article ID, chunk ID, chunk text, and embedding into `embeddings.csv`.

## Why Sentence-Level Chunking?

I used sentence-level chunking because it is simple and keeps each chunk meaningful while making retrieval more fine-grained than using complete articles.

A lightweight regex-based sentence splitter was used to keep the implementation simple and dependency-light.

## Embedding Model

I used:

```text
sentence-transformers/all-MiniLM-L6-v2
```

It provides a lightweight way to convert text chunks into semantic vector representations. The embeddings were generated in batches and normalized before storage.

## Retrieval Preparation

A separate `retrieval.py` file was also created to load the saved embeddings and perform **cosine similarity search**. Since the embeddings are normalized, similarity can be calculated using a simple dot product.

## Main Design Choices

- **Dataset:** NewsQA
- **Chunking:** Sentence-level
- **Minimum chunk length:** 20 characters
- **Embedding model:** `all-MiniLM-L6-v2`
- **Batch size:** 32
- **Normalization:** Enabled
- **Storage:** CSV
- **Retrieval:** Cosine similarity

## Final Output

The final `embeddings.csv` contains:

```text
article_id
chunk_id
chunk_text
embedding
```

This output becomes the base for the next stage of the project, where the generated embeddings can be used for semantic search and retrieval.