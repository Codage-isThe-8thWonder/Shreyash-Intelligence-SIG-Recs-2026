import json
from pathlib import Path

from datasets import load_dataset

from summarizer_agent import SummarizerAgent
from qa_agent import QAAgent
from fact_checker_agent import FactCheckerAgent


PART_B_DIR = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = PART_B_DIR / "examples"


def load_real_examples():

    dataset = load_dataset(
        "lucadiliello/newsqa",
        split="train",
    )

    examples = []

    seen_articles = set()

    for row in dataset:

        article_id = str(
            row["key"]
        )

        if article_id in seen_articles:
            continue

        if not row["context"]:
            continue

        seen_articles.add(
            article_id
        )

        examples.append(row)

        if len(examples) >= 3:
            break

    return examples


def main():

    EXAMPLES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    examples = load_real_examples()

    print(
        f"Selected {len(examples)} real NewsQA articles."
    )

    # ------------------------------------------------
    # Summarizer examples
    # ------------------------------------------------

    summarizer = SummarizerAgent()

    summary_results = []

    for index, row in enumerate(examples, start=1):

        summary = summarizer.run(
            row["context"]
        )

        summary_results.append(
            {
                "example_id": index,
                "article_id": str(row["key"]),
                "input_article": row["context"],
                "output_summary": summary,
            }
        )

        print(
            f"Generated summary example {index}"
        )

    with open(
        EXAMPLES_DIR / "summarizer_examples.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary_results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------
    # QA examples
    # ------------------------------------------------

    qa_agent = QAAgent()

    qa_results = []

    for index, row in enumerate(examples, start=1):

        result = qa_agent.run(
            row["question"],
            top_k=5,
        )

        qa_results.append(
            {
                "example_id": index,
                "article_id": str(row["key"]),
                "question": row["question"],
                "ground_truth_answer": row[
                    "answers"
                ],
                "agent_output": result,
            }
        )

        print(
            f"Generated QA example {index}"
        )

    with open(
        EXAMPLES_DIR / "qa_examples.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            qa_results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------
    # Fact Checker examples
    # ------------------------------------------------

    fact_checker = FactCheckerAgent()

    fact_results = []

    for index, row in enumerate(examples, start=1):

        # Use the actual answer-bearing text as a simple
        # corpus-grounded claim source.
        claim = (
            f"The article contains information answering "
            f"the question: {row['question']}"
        )

        result = fact_checker.run(
            claim,
            top_k=5,
        )

        fact_results.append(
            {
                "example_id": index,
                "article_id": str(row["key"]),
                "claim": claim,
                "agent_output": result,
            }
        )

        print(
            f"Generated fact-check example {index}"
        )

    with open(
        EXAMPLES_DIR / "fact_checker_examples.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            fact_results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\nExamples generated successfully.")


if __name__ == "__main__":
    main()