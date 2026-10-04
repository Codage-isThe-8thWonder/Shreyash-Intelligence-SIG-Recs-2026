from pathlib import Path
import sys

from datasets import load_dataset


# ============================================================
# PATH
# ============================================================

PART_C_SRC = Path(
    __file__
).resolve().parent

if str(PART_C_SRC) not in sys.path:

    sys.path.insert(
        0,
        str(PART_C_SRC)
    )


from controller import (
    NewsQAPipelineController,
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_newsqa():

    print("\n" + "=" * 70)
    print("LOADING NEWSQA")
    print("=" * 70)

    dataset = load_dataset(
        "lucadiliello/newsqa",
        split="train",
    )

    print(
        f"Loaded {len(dataset)} rows."
    )

    return dataset


# ============================================================
# FIND SECOND ARTICLE
# ============================================================

def get_second_valid_row(dataset):

    first_key = str(
        dataset[0]["key"]
    )

    for row in dataset:

        if not row["context"]:
            continue

        if not row["question"]:
            continue

        if not row["answers"]:
            continue

        if str(row["key"]) == first_key:
            continue

        return row

    return dataset[1]


# ============================================================
# BUILD EXAMPLES
# ============================================================

def build_examples(dataset):

    first_row = dataset[0]

    second_row = (
        get_second_valid_row(
            dataset
        )
    )

    examples = [

        # ----------------------------------------------------
        # 1. ARTICLE
        # ----------------------------------------------------

        {
            "example_id": 1,

            "description": (
                "Article-only summarization"
            ),

            "input": {
                "type": "article",

                "text": first_row[
                    "context"
                ],
            },
        },

        # ----------------------------------------------------
        # 2. QUESTION
        # ----------------------------------------------------

        {
            "example_id": 2,

            "description": (
                "Question-only grounded QA"
            ),

            "input": {
                "type": "question",

                "text": first_row[
                    "question"
                ],
            },
        },

        # ----------------------------------------------------
        # 3. CORROBORATED CLAIM
        # ----------------------------------------------------

        {
            "example_id": 3,

            "description": (
                "Corroborated claim verification"
            ),

            "input": {
                "type": "claim",

                "text": (
                    "The serial killings "
                    "involved 19 victims."
                ),
            },
        },

        # ----------------------------------------------------
        # 4. FULL PIPELINE
        # ----------------------------------------------------

        {
            "example_id": 4,

            "description": (
                "Full chained pipeline"
            ),

            "input": {
                "type": "article_question",

                "article": second_row[
                    "context"
                ],

                "question": second_row[
                    "question"
                ],
            },
        },

        # ----------------------------------------------------
        # 5. CONTRADICTED CLAIM
        # ----------------------------------------------------

        {
            "example_id": 5,

            "description": (
                "Contradicted claim verification"
            ),

            "input": {
                "type": "claim",

                "text": (
                    "The high court upheld "
                    "Pandher's death sentence."
                ),
            },
        },
    ]

    return examples


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    example,
    trace,
):

    print("\n" + "-" * 70)

    print(
        f"Example {example['example_id']}: "
        f"{example['description']}"
    )

    print(
        f"\nStatus: {trace['status']}"
    )

    print(
        "\nRoute:"
    )

    print(
        " -> ".join(
            trace["route"]
        )
    )

    print(
        "\nExecution:"
    )

    for step in trace["steps"]:

        print(
            f"  Step {step['step']} | "
            f"{step['agent']} | "
            f"{step['status']}"
        )

    print(
        "\nFinal output:"
    )

    print(
        trace["final_output"]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "PART C — MULTI-AGENT NEWSQA PIPELINE"
    )
    print("=" * 70)

    dataset = load_newsqa()

    controller = (
        NewsQAPipelineController(
            top_k=5,
            max_fact_check_retries=1,
        )
    )

    examples = build_examples(
        dataset
    )

    print(
        f"\nRunning {len(examples)} examples..."
    )

    # --------------------------------------------------------
    # Run examples
    # --------------------------------------------------------

    for example in examples:

        trace = controller.run(
            example["input"]
        )

        trace["example_id"] = (
            example["example_id"]
        )

        trace["description"] = (
            example["description"]
        )

        # ----------------------------------------------------
        # Save trace
        # ----------------------------------------------------

        trace_file = (
            controller.trace_manager.save(
                trace
            )
        )

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print_summary(
            example,
            trace,
        )

        print(
            f"\nTrace saved to: {trace_file}"
        )

    print("\n" + "=" * 70)

    print(
        "ALL 5 PART C EXAMPLES COMPLETED"
    )

    print("=" * 70)

    print(
        "\nFull traces are available in:"
    )

    print(
        "PartC/traces/"
    )


if __name__ == "__main__":
    main()