import sys
from pathlib import Path


# ============================================================
# PART B IMPORT PATH
# ============================================================

PART_C_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PART_C_DIR.parent

PART_B_SRC = PROJECT_ROOT / "PartB" / "src"

if str(PART_B_SRC) not in sys.path:
    sys.path.insert(
        0,
        str(PART_B_SRC)
    )


# ============================================================
# PART B AGENTS
# ============================================================

from summarizer_agent import SummarizerAgent
from qa_agent import QAAgent
from fact_checker_agent import FactCheckerAgent


class PipelineHandlers:
    """
    Executes the actual Part B agents.

    The controller decides WHAT route to use.
    This class handles HOW each route is executed.
    """

    def __init__(
        self,
        top_k=5,
        max_fact_check_retries=1,
    ):

        self.top_k = top_k

        self.max_fact_check_retries = (
            max_fact_check_retries
        )

        # ----------------------------------------------------
        # Reuse Part B agents
        # ----------------------------------------------------

        self.summarizer = (
            SummarizerAgent()
        )

        self.qa_agent = (
            QAAgent()
        )

        self.fact_checker = (
            FactCheckerAgent()
        )

    # ========================================================
    # ARTICLE
    # ========================================================

    def handle_article(
        self,
        article,
        trace,
        trace_manager,
    ):
        """
        Article → Summarizer
        """

        try:

            summary = self.summarizer.run(
                article
            )

            trace_manager.add_step(
                trace=trace,
                agent="SummarizerAgent",
                input_data=article,
                output={
                    "summary": summary
                },
                status="success",
            )

            return {
                "type": "article_summary",
                "status": "success",
                "summary": summary,
            }

        except Exception as error:

            trace_manager.add_step(
                trace=trace,
                agent="SummarizerAgent",
                input_data=article,
                status="failed",
                error=str(error),
            )

            raise

    # ========================================================
    # QUESTION
    # ========================================================

    def handle_question(
        self,
        question,
        trace,
        trace_manager,
    ):
        """
        Question → QA
        """

        try:

            result = self.qa_agent.run(
                question,
                top_k=self.top_k,
            )

            answer = str(
                result.get(
                    "answer",
                    "",
                )
            ).strip()

            # -----------------------------------------------
            # Detect insufficient evidence
            # -----------------------------------------------

            if (
                not answer
                or
                "insufficient evidence"
                in answer.lower()
            ):

                trace_manager.add_step(
                    trace=trace,
                    agent="QAAgent",
                    input_data=question,
                    output=result,
                    status="low_confidence",
                )

                return {
                    "type": "qa_result",
                    "status": "low_confidence",
                    "answer": answer,
                    "evidence": result.get(
                        "evidence",
                        [],
                    ),
                    "message": (
                        "QA did not find sufficient "
                        "grounded evidence."
                    ),
                }

            trace_manager.add_step(
                trace=trace,
                agent="QAAgent",
                input_data=question,
                output=result,
                status="success",
            )

            return {
                "type": "qa_result",
                "status": "success",
                "answer": answer,
                "evidence": result.get(
                    "evidence",
                    [],
                ),
            }

        except Exception as error:

            trace_manager.add_step(
                trace=trace,
                agent="QAAgent",
                input_data=question,
                status="failed",
                error=str(error),
            )

            return {
                "type": "qa_result",
                "status": "failed",
                "message": (
                    "QA failed. No unsupported "
                    "answer was generated."
                ),
                "error": str(error),
            }

    # ========================================================
    # CLAIM
    # ========================================================

    def handle_claim(
        self,
        claim,
        trace,
        trace_manager,
        starting_step=1,
    ):
        """
        Claim → Fact Checker
        """

        return self._fact_check(
            claim=claim,
            trace=trace,
            trace_manager=trace_manager,
            starting_step=starting_step,
        )

    # ========================================================
    # ARTICLE + QUESTION
    # ========================================================

    def handle_article_question(
        self,
        article,
        question,
        trace,
        trace_manager,
    ):
        """
        Full chain:

        Article
          ↓
        Summarizer

        Question
          ↓
        QA
          ↓
        Fact Checker
        """

        # ====================================================
        # STEP 1 — SUMMARIZER
        # ====================================================

        try:

            summary = self.summarizer.run(
                article
            )

            trace_manager.add_step(
                trace=trace,
                agent="SummarizerAgent",
                input_data=article,
                output={
                    "summary": summary
                },
                status="success",
            )

        except Exception as error:

            trace_manager.add_step(
                trace=trace,
                agent="SummarizerAgent",
                input_data=article,
                status="failed",
                error=str(error),
            )

            # -----------------------------------------------
            # QA is independent of the summary.
            # -----------------------------------------------

            summary = None

        # ====================================================
        # STEP 2 — QA
        # ====================================================

        try:

            qa_result = self.qa_agent.run(
                question,
                top_k=self.top_k,
            )

            answer = str(
                qa_result.get(
                    "answer",
                    "",
                )
            ).strip()

            # -----------------------------------------------
            # Low-confidence QA
            # -----------------------------------------------

            if (
                not answer
                or
                "insufficient evidence"
                in answer.lower()
            ):

                trace_manager.add_step(
                    trace=trace,
                    agent="QAAgent",
                    input_data=question,
                    output=qa_result,
                    status="low_confidence",
                )

                return {
                    "type": "article_question_result",
                    "status": "low_confidence",
                    "summary": summary,
                    "question": question,
                    "answer": answer,
                    "qa_evidence": qa_result.get(
                        "evidence",
                        [],
                    ),
                    "message": (
                        "QA did not find sufficient "
                        "grounded evidence. "
                        "Fact checking was skipped."
                    ),
                }

            trace_manager.add_step(
                trace=trace,
                agent="QAAgent",
                input_data=question,
                output=qa_result,
                status="success",
            )

        except Exception as error:

            trace_manager.add_step(
                trace=trace,
                agent="QAAgent",
                input_data=question,
                status="failed",
                error=str(error),
            )

            return {
                "type": "article_question_result",
                "status": "failed",
                "summary": summary,
                "question": question,
                "message": (
                    "QA failed, so the answer "
                    "was not fact-checked."
                ),
                "error": str(error),
            }

        # ====================================================
        # STEP 3 — FACT CHECK
        # ====================================================

        # Create a proper claim from question + answer.
        fact_check_claim = (
            f"The answer to the question "
            f"'{question}' is '{answer}'."
        )

        fact_result = self._fact_check(
            claim=fact_check_claim,
            trace=trace,
            trace_manager=trace_manager,
            starting_step=3,
        )

        # ====================================================
        # FINAL RESULT
        # ====================================================

        return {
            "type": "article_question_result",
            "status": "success",
            "summary": summary,
            "question": question,
            "answer": answer,
            "qa_evidence": qa_result.get(
                "evidence",
                [],
            ),
            "fact_check": fact_result,
        }

    # ========================================================
    # FACT CHECK + RETRY
    # ========================================================

    def _fact_check(
        self,
        claim,
        trace,
        trace_manager,
        starting_step=1,
    ):
        """
        Execute Fact Checker.

        If verdict == unverifiable, reformulate once
        and retry.
        """

        current_claim = claim

        for attempt in range(
            self.max_fact_check_retries + 1
        ):

            try:

                result = self.fact_checker.run(
                    current_claim,
                    top_k=self.top_k,
                )

                verdict = str(
                    result.get(
                        "verdict",
                        "",
                    )
                ).lower().strip()

                # -------------------------------------------
                # Valid positive/negative verdict
                # -------------------------------------------

                if verdict in {
                    "corroborated",
                    "contradicted",
                }:

                    trace_manager.add_step(
                        trace=trace,
                        agent="FactCheckerAgent",
                        input_data=current_claim,
                        output=result,
                        status="success",
                        attempt=attempt + 1,
                    )

                    return {
                        "status": "success",
                        "verdict": verdict,
                        "explanation": result.get(
                            "explanation",
                            "",
                        ),
                        "evidence": result.get(
                            "evidence",
                            [],
                        ),
                        "attempts": attempt + 1,
                    }

                # -------------------------------------------
                # Unverifiable
                # -------------------------------------------

                if verdict == "unverifiable":

                    trace_manager.add_step(
                        trace=trace,
                        agent="FactCheckerAgent",
                        input_data=current_claim,
                        output=result,
                        status="low_confidence",
                        attempt=attempt + 1,
                    )

                    # ---------------------------------------
                    # Retry
                    # ---------------------------------------

                    if (
                        attempt
                        <
                        self.max_fact_check_retries
                    ):

                        current_claim = (
                            "Check whether the NewsQA "
                            "corpus contains evidence "
                            "that supports or contradicts "
                            "the following statement:\n\n"
                            f"{claim}"
                        )

                        continue

                    # ---------------------------------------
                    # Final unverifiable
                    # ---------------------------------------

                    return {
                        "status": "unverifiable",
                        "verdict": "unverifiable",
                        "explanation": result.get(
                            "explanation",
                            (
                                "The available NewsQA "
                                "evidence was insufficient "
                                "to verify the claim."
                            ),
                        ),
                        "evidence": result.get(
                            "evidence",
                            [],
                        ),
                        "attempts": attempt + 1,
                    }

                # -------------------------------------------
                # Unexpected verdict
                # -------------------------------------------

                trace_manager.add_step(
                    trace=trace,
                    agent="FactCheckerAgent",
                    input_data=current_claim,
                    output=result,
                    status="failed",
                    attempt=attempt + 1,
                    error=(
                        "Unexpected verdict: "
                        f"{verdict}"
                    ),
                )

                return {
                    "status": "failed",
                    "message": (
                        "Fact Checker returned "
                        "an invalid verdict."
                    ),
                }

            except Exception as error:

                trace_manager.add_step(
                    trace=trace,
                    agent="FactCheckerAgent",
                    input_data=current_claim,
                    status="failed",
                    attempt=attempt + 1,
                    error=str(error),
                )

                return {
                    "status": "failed",
                    "message": (
                        "Fact Checker failed. "
                        "No verdict was invented."
                    ),
                    "error": str(error),
                }

        # Defensive fallback

        return {
            "status": "unverifiable",
            "verdict": "unverifiable",
            "explanation": (
                "The claim could not be verified "
                "from the available corpus."
            ),
            "evidence": [],
            "attempts": (
                self.max_fact_check_retries + 1
            ),
        }