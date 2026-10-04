import sys
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

FINALE_DIR = (
    Path(__file__).resolve().parent.parent
)

PROJECT_ROOT = FINALE_DIR.parent

PART_B_SRC = (
    PROJECT_ROOT / "PartB" / "src"
)

if str(PART_B_SRC) not in sys.path:
    sys.path.insert(
        0,
        str(PART_B_SRC)
    )


# ============================================================
# PART B AGENTS
# ============================================================

from summarizer_agent import (
    SummarizerAgent
)

from qa_agent import (
    QAAgent
)

from fact_checker_agent import (
    FactCheckerAgent
)


from evidence_policy import (
    EvidencePolicy
)


class FinalePipeline:

    def __init__(
        self,
        top_k=5,
    ):

        self.top_k = top_k

        self.summarizer = (
            SummarizerAgent()
        )

        self.qa_agent = (
            QAAgent()
        )

        self.fact_checker = (
            FactCheckerAgent()
        )

        self.evidence_policy = (
            EvidencePolicy()
        )

    # ========================================================
    # SUMMARIZER
    # ========================================================

    def summarize(
        self,
        article,
    ):

        return self.summarizer.run(
            article
        )

    # ========================================================
    # QA
    # ========================================================

    def answer_question(
        self,
        question,
    ):

        return self.qa_agent.run(
            question,
            top_k=self.top_k,
        )

    # ========================================================
    # FACT CHECK
    # ========================================================

    def fact_check(
        self,
        claim,
    ):

        result = (
            self.fact_checker.run(
                claim,
                top_k=self.top_k,
            )
        )

        return self.evidence_policy.finalize(
            result
        )

    # ========================================================
    # CLAIM FROM QA
    # ========================================================

    def create_claim_from_answer(
        self,
        question,
        answer,
    ):

        return (
            f"The answer to the question "
            f"'{question}' is '{answer}'."
        )