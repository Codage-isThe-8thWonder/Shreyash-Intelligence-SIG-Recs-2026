# Finale

## Overview

Finale is the final part of the NewsQA project. It combines the retrieval system and agents developed in the previous parts into one complete multi-agent pipeline.

## Approach

The Finale uses a controller to receive and validate the input and a router to decide which agents need to be executed.

It supports articles, questions, claims, and combinations of article with a question or claim.

The pipeline reuses the Summarizer, QA, and Fact Checker agents from Part B. Depending on the input, these agents can run individually or sequentially.

For article and question inputs, the system first generates a summary, then answers the question using retrieved evidence. The generated answer can then be converted into a claim and fact-checked.

## Evidence Policy

A separate evidence policy checks the fact-checking result before returning it. It only allows three verdicts: `corroborated`, `contradicted`, and `unverifiable`.

If the retrieved evidence is missing or too weak, the result is marked as unverifiable instead of giving an unreliable verdict.

## Tracing and Error Handling

Every execution is recorded with the input, selected route, agents executed, outputs, status, and errors. Results are saved as JSON files, making the system easier to debug and evaluate.

The controller also handles failures without stopping the complete test run.

## Testing

The final system was tested with different cases including summarization, question answering, corroborated claims, contradicted claims, unverifiable claims, and complete multi-agent workflows.

The test results are saved in `Finale/test_results/`.

## Final Result

The Finale provides a complete evidence-grounded NewsQA system with multi-agent orchestration, fact checking, confidence handling, error handling, and execution tracing.