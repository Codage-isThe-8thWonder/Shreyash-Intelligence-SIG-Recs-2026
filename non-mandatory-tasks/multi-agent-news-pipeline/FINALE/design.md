# Finale Design — News Article Multi-Agent System

## 1. Objective

The Finale combines the three agents developed in Parts A, B, and C into a single multi-agent assistant for NewsQA.

The system supports:

- article summarization
- grounded question answering
- corpus-wide claim verification
- chained article-question-claim workflows

The system reuses the embedding corpus and standalone agents developed earlier rather than rebuilding them.

---

## 2. System Architecture

The system consists of four major layers:

1. Input Router
2. Agent Pipeline
3. Evidence Policy
4. Trace Manager

The controller receives structured input and determines which agents are required.

### Basic routes

| Input | Route |
|---|---|
| Article | Summarizer |
| Question | QA |
| Claim | Fact Checker |
| Article + Question | Summarizer → QA → Fact Checker |
| Article + Claim | Summarizer → Fact Checker |
| Article + Question + Claim | Summarizer → QA → Fact Checker |

---

## 3. Controller Routing

The controller prefers structured input rather than guessing the input type from raw text.

For example:

```python
{
    "type": "article_question",
    "article": "...",
    "question": "..."
}