# Part C — Multi-Agent NewsQA Pipeline

## Overview

In Part C, we combined the agents from Part B into a single **multi-agent pipeline**.

The main goal was to automatically understand the input, choose the correct agent or agent sequence, execute it, and keep a record of the complete execution.

## What We Built

- **Pipeline Router** to identify the input type and select the required agent.
- **Pipeline Controller** to manage the complete execution.
- Reused the **Summarizer, QA, and Fact Checker agents** from Part B.
- Added **execution tracing** to record each step of the pipeline.
- Added error handling and graceful failure responses.
- Added a retry mechanism for fact checking when the result is `unverifiable`.

## Input Types

The pipeline supports:

- `article` → Summarizer
- `question` → QA Agent
- `claim` → Fact Checker
- `article_question` → Summarizer + QA + Fact Checker

The router can also automatically detect the type of plain text input using simple rules.

## Full Pipeline

For an `article_question` input, the system runs the complete chain:

```text
Article → Summary
Question → Answer
Answer → Fact Check
```

If QA does not find sufficient evidence, fact checking is skipped instead of producing an unsupported result.

## Retry Mechanism

If the Fact Checker returns `unverifiable`, the system can reformulate the claim and retry once.

If it is still unverifiable, the final result remains `unverifiable`.

## Execution Tracing

A `TraceManager` was added to record:

- Input
- Selected route
- Agents executed
- Step status
- Outputs
- Errors
- Final result

Each execution is saved as a JSON trace inside the `traces/` directory.

## Testing

The pipeline was tested with **5 different examples**, including article summarization, question answering, corroborated/contradicted claims, and a full chained pipeline.

The traces of these executions are saved for inspection.

## Main Technologies

- Python
- NewsQA
- Sentence Transformers
- Gemini
- NumPy
- Pandas
- JSON

## Final Result

Part C turns the individual agents from Part B into a **single automated pipeline** that can route different inputs, run the required agents, handle failures/retries, and maintain complete execution traces.