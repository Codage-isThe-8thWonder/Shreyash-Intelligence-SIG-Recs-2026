# Part B — Multi-Agent NewsQA System

## Overview

In Part B, we built a **Separate Agents**  using the embeddings and retrieval pipeline from Part A.

They use **Gemini** to perform different tasks on the NewsQA dataset.

## What We Built

We created three separate agents:

- **Summarizer Agent** – Generates a short summary of a news article.
- **QA Agent** – Answers questions using only the retrieved NewsQA evidence.
- **Fact Checker Agent** – Checks whether a claim is supported, contradicted, or cannot be verified using the NewsQA corpus.

## Retrieval

The system reuses the embeddings generated in Part A.

For every question or claim, the retriever finds the **top 5 most relevant chunks** using cosine similarity and provides them as evidence to the agents.

## Gemini Integration

A common `GeminiClient` was created so all agents use the same Gemini API setup.

The API key is loaded from the `.env` file, and the Gemini model can be configured through the project settings.

## Important Design Choice

The QA and Fact Checker agents are designed to work **only with the retrieved NewsQA evidence**.

They are instructed not to use outside knowledge or guess when the evidence is insufficient. 

## Fact Checker

The Fact Checker supports three possible results:

- `corroborated`
- `contradicted`
- `unverifiable`

This also handles cases where the corpus does not contain enough information.

## Testing & Examples

Separate test scripts were created for:

- Gemini connection
- Retriever
- Summarizer
- QA Agent
- Fact Checker

Example outputs can't be generated due to unavailability error of gemini (error 503)

## Main Technologies

- Python
- Hugging Face NewsQA
- Sentence Transformers
- `all-MiniLM-L6-v2`
- Google Gemini API
- NumPy
- Pandas

## Final Result

Part B adds the **LLM agent layer** on top of Part A's retrieval system and provides summarization, question answering, and fact-checking capabilities using retrieved NewsQA evidence.