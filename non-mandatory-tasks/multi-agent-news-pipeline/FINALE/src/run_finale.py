import json
import time
from pathlib import Path
import sys


# ============================================================
# PATH SETUP
# ============================================================

FINALE_SRC = Path(
    __file__
).resolve().parent

if str(FINALE_SRC) not in sys.path:
    sys.path.insert(
        0,
        str(FINALE_SRC)
    )


# ============================================================
# FINALE CONTROLLER
# ============================================================

from controller import FinaleController


# ============================================================
# NEWSQA LOADER
# ============================================================

def load_newsqa():

    from datasets import load_dataset

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
# VALID NEWSQA ROWS
# ============================================================

def get_valid_rows(dataset):

    rows = []

    for row in dataset:

        if not row["context"]:
            continue

        if not row["question"]:
            continue

        if not row["answers"]:
            continue

        rows.append(row)

        if len(rows) >= 4:
            break

    return rows


# ============================================================
# TEST CASES
# ============================================================

def build_test_cases(dataset):

    rows = get_valid_rows(
        dataset
    )

    if len(rows) < 2:

        raise RuntimeError(
            "Could not find enough valid "
            "NewsQA rows for testing."
        )

    first = rows[0]
    second = rows[1]

    test_cases = [

        # ====================================================
        # TEST 1
        # ====================================================

        {
            "id": 1,

            "name":
                "Article summarization",

            "input": {
                "type": "article",

                "article":
                    first["context"],
            },
        },

        # ====================================================
        # TEST 2
        # ====================================================

        {
            "id": 2,

            "name":
                "Grounded question answering",

            "input": {
                "type": "question",

                "question":
                    first["question"],
            },
        },

        # ====================================================
        # TEST 3
        # ====================================================

        {
            "id": 3,

            "name":
                "Corroborated claim",

            "input": {
                "type": "claim",

                "claim":
                    "The serial killings involved 19 victims.",
            },
        },

        # ====================================================
        # TEST 4
        # ====================================================

        {
            "id": 4,

            "name":
                "Contradicted claim",

            "input": {
                "type": "claim",

                "claim":
                    (
                        "The high court upheld "
                        "Pandher's death sentence."
                    ),
            },
        },

        # ====================================================
        # TEST 5
        # ====================================================

        {
            "id": 5,

            "name":
                "Unverifiable claim",

            "input": {
                "type": "claim",

                "claim":
                    (
                        "Pandher was born "
                        "in 1965."
                    ),
            },
        },

        # ====================================================
        # TEST 6
        # ====================================================

        {
            "id": 6,

            "name":
                "Article plus question",

            "input": {
                "type": "article_question",

                "article":
                    second["context"],

                "question":
                    second["question"],
            },
        },

        # ====================================================
        # TEST 7
        # ====================================================

        {
            "id": 7,

            "name":
                "Article plus claim",

            "input": {
                "type": "article_claim",

                "article":
                    first["context"],

                "claim":
                    (
                        "Pandher and Surinder Koli "
                        "were sentenced to death "
                        "in February."
                    ),
            },
        },

        # ====================================================
        # TEST 8
        # ====================================================

        {
            "id": 8,

            "name":
                "Full multi-agent pipeline",

            "input": {
                "type":
                    "article_question_claim",

                "article":
                    first["context"],

                "question":
                    (
                        "When was Pandher "
                        "sentenced to death?"
                    ),

                "claim":
                    (
                        "Pandher and Surinder Koli "
                        "were sentenced to death "
                        "in February."
                    ),
            },
        },
    ]

    return test_cases


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(
    test_case,
    trace,
):

    output_dir = (
        Path(__file__).resolve().parent.parent
        / "test_results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / f"test_{test_case['id']}.json"
    )

    trace["test_case_id"] = (
        test_case["id"]
    )

    trace["test_case_name"] = (
        test_case["name"]
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            trace,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    return output_file


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(
    test_case,
    trace,
):

    print("\n" + "-" * 70)

    print(
        f"TEST {test_case['id']}: "
        f"{test_case['name']}"
    )

    print(
        f"Status: {trace['status']}"
    )

    print(
        "Route: "
        + " -> ".join(
            trace["route"]
        )
    )

    print("\nAgents executed:")

    for step in trace["steps"]:

        print(
            f"  Step {step['step']}: "
            f"{step['agent']} "
            f"[{step['status']}]"
        )

    print("\nFinal output:")

    print(
        json.dumps(
            trace["final_output"],
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    dataset = load_newsqa()

    controller = FinaleController(
        top_k=5
    )

    test_cases = build_test_cases(
        dataset
    )

    print("\n" + "=" * 70)

    print(
        f"RUNNING {len(test_cases)} FINALE TEST CASES"
    )

    print("=" * 70)

    successful = 0
    failed = 0

    # ========================================================
    # TEST LOOP
    # ========================================================

    for index, test_case in enumerate(
        test_cases
    ):

        print(
            f"\nStarting test "
            f"{test_case['id']}/"
            f"{len(test_cases)}..."
        )

        # ----------------------------------------------------
        # Run pipeline
        # ----------------------------------------------------

        trace = controller.run(
            test_case["input"]
        )

        # ----------------------------------------------------
        # Save result
        # ----------------------------------------------------

        output_file = save_result(
            test_case,
            trace,
        )

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print_result(
            test_case,
            trace,
        )

        print(
            f"\nSaved: {output_file}"
        )

        # ----------------------------------------------------
        # Count result
        # ----------------------------------------------------

        if trace["status"] in {
            "success",
            "low_confidence",
        }:

            successful += 1

        else:

            failed += 1

        # ----------------------------------------------------
        # RATE LIMIT PROTECTION
        # ----------------------------------------------------

        if index < len(test_cases) - 1:

            wait_seconds = 15

            print(
                "\nWaiting "
                f"{wait_seconds} seconds "
                "before the next test "
                "to avoid Gemini rate limits..."
            )

            time.sleep(
                wait_seconds
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "FINALE TEST RUN COMPLETE"
    )

    print("=" * 70)

    print(
        f"Handled successfully: "
        f"{successful}"
    )

    print(
        f"Failed: "
        f"{failed}"
    )

    print(
        f"Total: "
        f"{len(test_cases)}"
    )

    print(
        "\nResults saved in:"
    )

    print(
        "Finale/test_results/"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()