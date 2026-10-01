import json
from pathlib import Path

from datasets import load_dataset

from summarizer_agent import SummarizerAgent
from qa_agent import QAAgent
from fact_checker_agent import FactCheckerAgent


# ============================================================
# PATHS
# ============================================================

PART_B_DIR = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = PART_B_DIR / "examples"

EXAMPLES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# HELPER
# ============================================================

def save_json(data, filepath):
    """
    Save Python object as formatted JSON.
    """

    with open(
        filepath,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# LOAD DATASET
# ============================================================

def load_newsqa():

    print("\nLoading NewsQA dataset...")

    dataset = load_dataset(
        "lucadiliello/newsqa",
        split="train"
    )

    print(
        f"Dataset loaded: {len(dataset)} rows"
    )

    return dataset


# ============================================================
# SUMMARIZER EXAMPLES
# ============================================================

def generate_summarizer_examples(dataset):

    print("\n" + "=" * 70)
    print("GENERATING SUMMARIZER EXAMPLES")
    print("=" * 70)

    agent = SummarizerAgent()

    examples = []

    selected_indices = [
        0,
        1,
        2
    ]

    for example_number, index in enumerate(
        selected_indices,
        start=1
    ):

        row = dataset[index]

        article_id = str(
            row["key"]
        )

        article = str(
            row["context"]
        )

        print(
            f"\nRunning summarizer example "
            f"{example_number}/3..."
        )

        summary = agent.run(
            article
        )

        examples.append(
            {
                "example_id": example_number,
                "article_id": article_id,
                "input_article": article,
                "summary": summary
            }
        )

        print("Done.")

    output_file = (
        EXAMPLES_DIR /
        "summarizer_examples.json"
    )

    save_json(
        examples,
        output_file
    )

    print(
        f"\nSaved: {output_file}"
    )


# ============================================================
# QA EXAMPLES
# ============================================================

def generate_qa_examples(dataset):

    print("\n" + "=" * 70)
    print("GENERATING QA EXAMPLES")
    print("=" * 70)

    agent = QAAgent()

    examples = []

    # --------------------------------------------------------
    # Select rows containing valid questions and answers
    # --------------------------------------------------------

    selected_rows = []

    for row in dataset:

        question = row["question"]
        answers = row["answers"]

        if (
            question
            and answers
            and len(answers) > 0
            and row["context"]
        ):

            selected_rows.append(row)

        if len(selected_rows) == 3:
            break

    # --------------------------------------------------------
    # Run QA agent
    # --------------------------------------------------------

    for example_number, row in enumerate(
        selected_rows,
        start=1
    ):

        article_id = str(
            row["key"]
        )

        question = str(
            row["question"]
        )

        ground_truth = [
            str(answer)
            for answer in row["answers"]
        ]

        print(
            f"\nRunning QA example "
            f"{example_number}/3..."
        )

        result = agent.run(
            question,
            top_k=5
        )

        examples.append(
            {
                "example_id": example_number,
                "article_id": article_id,
                "question": question,
                "ground_truth_answers": ground_truth,
                "agent_answer": result["answer"],
                "evidence": result["evidence"]
            }
        )

        print("Done.")

    output_file = (
        EXAMPLES_DIR /
        "qa_examples.json"
    )

    save_json(
        examples,
        output_file
    )

    print(
        f"\nSaved: {output_file}"
    )


# ============================================================
# FACT CHECKER EXAMPLES
# ============================================================

def generate_fact_checker_examples():

    print("\n" + "=" * 70)
    print("GENERATING FACT CHECKER EXAMPLES")
    print("=" * 70)

    agent = FactCheckerAgent()

    # --------------------------------------------------------
    # Three deliberately different claim types
    # --------------------------------------------------------

    claims = [

        {
            "example_id": 1,
            "claim": (
                "The serial killings involved 19 victims."
            ),
            "expected_verdict": "corroborated"
        },

        {
            "example_id": 2,
            "claim": (
                "The high court upheld Pandher's "
                "death sentence."
            ),
            "expected_verdict": "contradicted"
        },

        {
            "example_id": 3,
            "claim": (
                "Pandher was born in 1965."
            ),
            "expected_verdict": "unverifiable"
        }
    ]

    examples = []

    for item in claims:

        print(
            f"\nRunning fact-checker "
            f"example {item['example_id']}/3..."
        )

        result = agent.run(
            item["claim"],
            top_k=5
        )

        examples.append(
            {
                "example_id": item["example_id"],
                "claim": item["claim"],
                "expected_verdict": item[
                    "expected_verdict"
                ],
                "agent_verdict": result[
                    "verdict"
                ],
                "explanation": result[
                    "explanation"
                ],
                "evidence": result[
                    "evidence"
                ]
            }
        )

        print("Done.")

    output_file = (
        EXAMPLES_DIR /
        "fact_checker_examples.json"
    )

    save_json(
        examples,
        output_file
    )

    print(
        f"\nSaved: {output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PART B EXAMPLE GENERATION")
    print("=" * 70)

    dataset = load_newsqa()

    # --------------------------------------------------------
    # 1. Summarizer
    # --------------------------------------------------------

    generate_summarizer_examples(
        dataset
    )

    # --------------------------------------------------------
    # 2. QA
    # --------------------------------------------------------

    generate_qa_examples(
        dataset
    )

    # --------------------------------------------------------
    # 3. Fact Checker
    # --------------------------------------------------------

    generate_fact_checker_examples()

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("ALL PART B EXAMPLES GENERATED")
    print("=" * 70)

    print(
        "\nFiles created:"
    )

    print(
        "1. PartB/examples/summarizer_examples.json"
    )

    print(
        "2. PartB/examples/qa_examples.json"
    )

    print(
        "3. PartB/examples/fact_checker_examples.json"
    )


if __name__ == "__main__":
    main()