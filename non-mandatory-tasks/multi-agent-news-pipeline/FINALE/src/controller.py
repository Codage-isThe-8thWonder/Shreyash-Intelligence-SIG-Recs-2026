from datetime import datetime

from router import FinaleRouter
from pipeline import FinalePipeline
from trace_manager import FinaleTraceManager


class FinaleController:
    """
    End-to-end controller for the Finale system.

    Responsibilities:
    - normalize input
    - route to appropriate agents
    - chain dependent agents
    - handle failures
    - maintain execution traces
    """

    def __init__(
        self,
        top_k=5,
    ):

        self.router = FinaleRouter()

        self.pipeline = FinalePipeline(
            top_k=top_k
        )

        self.trace_manager = (
            FinaleTraceManager()
        )

    # ========================================================
    # INPUT NORMALIZATION
    # ========================================================

    def normalize_input(
        self,
        raw_input,
    ):

        if not isinstance(
            raw_input,
            dict,
        ):

            raise ValueError(
                "Finale expects structured "
                "dictionary input."
            )

        input_type = raw_input.get(
            "type"
        )

        if input_type == "article":

            article = raw_input.get(
                "article",
                ""
            ).strip()

            if not article:
                raise ValueError(
                    "Article cannot be empty."
                )

            return {
                "type": "article",
                "article": article,
            }

        if input_type == "question":

            question = raw_input.get(
                "question",
                ""
            ).strip()

            if not question:
                raise ValueError(
                    "Question cannot be empty."
                )

            return {
                "type": "question",
                "question": question,
            }

        if input_type == "claim":

            claim = raw_input.get(
                "claim",
                ""
            ).strip()

            if not claim:
                raise ValueError(
                    "Claim cannot be empty."
                )

            return {
                "type": "claim",
                "claim": claim,
            }

        if input_type == "article_question":

            article = raw_input.get(
                "article",
                ""
            ).strip()

            question = raw_input.get(
                "question",
                ""
            ).strip()

            if not article:
                raise ValueError(
                    "Article cannot be empty."
                )

            if not question:
                raise ValueError(
                    "Question cannot be empty."
                )

            return {
                "type": "article_question",
                "article": article,
                "question": question,
            }

        if input_type == "article_claim":

            article = raw_input.get(
                "article",
                ""
            ).strip()

            claim = raw_input.get(
                "claim",
                ""
            ).strip()

            if not article:
                raise ValueError(
                    "Article cannot be empty."
                )

            if not claim:
                raise ValueError(
                    "Claim cannot be empty."
                )

            return {
                "type": "article_claim",
                "article": article,
                "claim": claim,
            }

        if input_type == "article_question_claim":

            article = raw_input.get(
                "article",
                ""
            ).strip()

            question = raw_input.get(
                "question",
                ""
            ).strip()

            claim = raw_input.get(
                "claim",
                ""
            ).strip()

            if not article:
                raise ValueError(
                    "Article cannot be empty."
                )

            if not question:
                raise ValueError(
                    "Question cannot be empty."
                )

            if not claim:
                raise ValueError(
                    "Claim cannot be empty."
                )

            return {
                "type": "article_question_claim",
                "article": article,
                "question": question,
                "claim": claim,
            }

        raise ValueError(
            f"Unsupported input type: {input_type}"
        )

    # ========================================================
    # MAIN RUN
    # ========================================================

    def run(
        self,
        raw_input,
    ):

        trace = None

        try:

            request = (
                self.normalize_input(
                    raw_input
                )
            )

            trace = (
                self.trace_manager.create_trace(
                    request
                )
            )

            input_type = request["type"]

            trace["route"] = (
                self.router.get_route(
                    input_type
                )
            )

            # =================================================
            # ARTICLE
            # =================================================

            if input_type == "article":

                return self._run_article(
                    request,
                    trace,
                )

            # =================================================
            # QUESTION
            # =================================================

            if input_type == "question":

                return self._run_question(
                    request,
                    trace,
                )

            # =================================================
            # CLAIM
            # =================================================

            if input_type == "claim":

                return self._run_claim(
                    request,
                    trace,
                )

            # =================================================
            # ARTICLE + QUESTION
            # =================================================

            if input_type == "article_question":

                return self._run_article_question(
                    request,
                    trace,
                )

            # =================================================
            # ARTICLE + CLAIM
            # =================================================

            if input_type == "article_claim":

                return self._run_article_claim(
                    request,
                    trace,
                )

            # =================================================
            # ARTICLE + QUESTION + CLAIM
            # =================================================

            if input_type == "article_question_claim":

                return self._run_full_input(
                    request,
                    trace,
                )

        except Exception as error:

            if trace is None:

                trace = (
                    self.trace_manager.create_trace(
                        raw_input
                    )
                )

            trace["status"] = "failed"

            trace["final_output"] = {
                "status": "failed",
                "error": str(error),
            }

            trace["finished_at"] = (
                datetime.now().isoformat()
            )

            return trace

    # ========================================================
    # ARTICLE
    # ========================================================

    def _run_article(
        self,
        request,
        trace,
    ):

        article = request["article"]

        try:

            summary = (
                self.pipeline.summarize(
                    article
                )
            )

            result = {
                "type": "article_summary",
                "status": "success",
                "summary": summary,
            }

            self.trace_manager.add_step(
                trace,
                "SummarizerAgent",
                article,
                result,
                "success",
            )

            trace["final_output"] = result
            trace["status"] = "success"

        except Exception as error:

            self.trace_manager.add_step(
                trace,
                "SummarizerAgent",
                article,
                status="failed",
                error=str(error),
            )

            trace["status"] = "failed"

            trace["final_output"] = {
                "status": "failed",
                "error": str(error),
            }

        trace["finished_at"] = (
            datetime.now().isoformat()
        )

        return trace

    # ========================================================
    # QUESTION
    # ========================================================

    def _run_question(
        self,
        request,
        trace,
    ):

        question = request["question"]

        try:

            result = (
                self.pipeline.answer_question(
                    question
                )
            )

            answer = str(
                result.get(
                    "answer",
                    ""
                )
            ).strip()

            if (
                not answer
                or
                "insufficient evidence"
                in answer.lower()
            ):

                status = "low_confidence"

            else:

                status = "success"

            self.trace_manager.add_step(
                trace,
                "QAAgent",
                question,
                result,
                status,
            )

            trace["final_output"] = {
                "type": "qa_result",
                "status": status,
                **result,
            }

            trace["status"] = status

        except Exception as error:

            self.trace_manager.add_step(
                trace,
                "QAAgent",
                question,
                status="failed",
                error=str(error),
            )

            trace["status"] = "failed"

            trace["final_output"] = {
                "status": "failed",
                "error": str(error),
            }

        trace["finished_at"] = (
            datetime.now().isoformat()
        )

        return trace

    # ========================================================
    # CLAIM
    # ========================================================

    def _run_claim(
        self,
        request,
        trace,
    ):

        claim = request["claim"]

        try:

            result = (
                self.pipeline.fact_check(
                    claim
                )
            )

            self.trace_manager.add_step(
                trace,
                "FactCheckerAgent",
                claim,
                result,
                "success",
            )

            trace["final_output"] = result
            trace["status"] = "success"

        except Exception as error:

            self.trace_manager.add_step(
                trace,
                "FactCheckerAgent",
                claim,
                status="failed",
                error=str(error),
            )

            trace["status"] = "failed"

            trace["final_output"] = {
                "status": "failed",
                "error": str(error),
            }

        trace["finished_at"] = (
            datetime.now().isoformat()
        )

        return trace

    # ========================================================
    # ARTICLE + QUESTION
    # ========================================================

    def _run_article_question(
        self,
        request,
        trace,
    ):

        article = request["article"]
        question = request["question"]

        summary = None

        # ----------------------------------------------------
        # STEP 1: SUMMARIZER
        # ----------------------------------------------------

        try:

            summary = (
                self.pipeline.summarize(
                    article
                )
            )

            self.trace_manager.add_step(
                trace,
                "SummarizerAgent",
                article,
                {
                    "summary": summary
                },
                "success",
            )

        except Exception as error:

            self.trace_manager.add_step(
                trace,
                "SummarizerAgent",
                article,
                status="failed",
                error=str(error),
            )

        # ----------------------------------------------------
        # STEP 2: QA
        # ----------------------------------------------------

        try:

            qa_result = (
                self.pipeline.answer_question(
                    question
                )
            )

            answer = str(
                qa_result.get(
                    "answer",
                    ""
                )
            ).strip()

            if (
                not answer
                or
                "insufficient evidence"
                in answer.lower()
            ):

                self.trace_manager.add_step(
                    trace,
                    "QAAgent",
                    question,
                    qa_result,
                    "low_confidence",
                )

                final_result = {
                    "type":
                        "article_question_result",

                    "status":
                        "low_confidence",

                    "summary":
                        summary,

                    "question":
                        question,

                    "qa":
                        qa_result,

                    "message":
                        (
                            "QA could not find "
                            "sufficient grounded "
                            "evidence, so fact "
                            "checking was skipped."
                        ),
                }

                trace["final_output"] = (
                    final_result
                )

                trace["status"] = (
                    "low_confidence"
                )

                trace["finished_at"] = (
                    datetime.now().isoformat()
                )

                return trace

            self.trace_manager.add_step(
                trace,
                "QAAgent",
                question,
                qa_result,
                "success",
            )

        except Exception as error:

            self.trace_manager.add_step(
                trace,
                "QAAgent",
                question,
                status="failed",
                error=str(error),
            )

            trace["status"] = "failed"

            trace["final_output"] = {
                "status": "failed",
                "summary": summary,
                "error": str(error),
            }

            trace["finished_at"] = (
                datetime.now().isoformat()
            )

            return trace

        # ----------------------------------------------------
        # STEP 3: DERIVED CLAIM
        # ----------------------------------------------------

        derived_claim = (
            self.pipeline.create_claim_from_answer(
                question,
                answer,
            )
        )

        # ----------------------------------------------------
        # STEP 4: FACT CHECK
        # ----------------------------------------------------

        try:

            fact_result = (
                self.pipeline.fact_check(
                    derived_claim
                )
            )

            self.trace_manager.add_step(
                trace,
                "FactCheckerAgent",
                derived_claim,
                fact_result,
                "success",
            )

        except Exception as error:

            self.trace_manager.add_step(
                trace,
                "FactCheckerAgent",
                derived_claim,
                status="failed",
                error=str(error),
            )

            fact_result = {
                "status": "failed",
                "error": str(error),
            }

        final_result = {
            "type":
                "article_question_result",

            "status":
                "success",

            "summary":
                summary,

            "question":
                question,

            "qa":
                qa_result,

            "derived_claim":
                derived_claim,

            "fact_check":
                fact_result,
        }

        trace["final_output"] = (
            final_result
        )

        trace["status"] = "success"

        trace["finished_at"] = (
            datetime.now().isoformat()
        )

        return trace

    # ========================================================
    # ARTICLE + CLAIM
    # ========================================================

    def _run_article_claim(
        self,
        request,
        trace,
    ):

        article = request["article"]
        claim = request["claim"]

        # ----------------------------------------------------
        # SUMMARIZER
        # ----------------------------------------------------

        try:

            summary = (
                self.pipeline.summarize(
                    article
                )
            )

            self.trace_manager.add_step(
                trace,
                "SummarizerAgent",
                article,
                {
                    "summary": summary
                },
                "success",
            )

        except Exception as error:

            summary = None

            self.trace_manager.add_step(
                trace,
                "SummarizerAgent",
                article,
                status="failed",
                error=str(error),
            )

        # ----------------------------------------------------
        # FACT CHECKER
        # ----------------------------------------------------

        try:

            fact_result = (
                self.pipeline.fact_check(
                    claim
                )
            )

            self.trace_manager.add_step(
                trace,
                "FactCheckerAgent",
                claim,
                fact_result,
                "success",
            )

            trace["final_output"] = {
                "type":
                    "article_claim_result",

                "status":
                    "success",

                "summary":
                    summary,

                "claim":
                    claim,

                "fact_check":
                    fact_result,
            }

            trace["status"] = "success"

        except Exception as error:

            self.trace_manager.add_step(
                trace,
                "FactCheckerAgent",
                claim,
                status="failed",
                error=str(error),
            )

            trace["status"] = "failed"

            trace["final_output"] = {
                "status": "failed",
                "summary": summary,
                "claim": claim,
                "error": str(error),
            }

        trace["finished_at"] = (
            datetime.now().isoformat()
        )

        return trace

    # ========================================================
    # ARTICLE + QUESTION + CLAIM
    # ========================================================

    def _run_full_input(
        self,
        request,
        trace,
    ):

        article = request["article"]
        question = request["question"]
        claim = request["claim"]

        # -----------------------------------------------
        # Summarize
        # -----------------------------------------------

        try:

            summary = (
                self.pipeline.summarize(
                    article
                )
            )

            self.trace_manager.add_step(
                trace,
                "SummarizerAgent",
                article,
                {
                    "summary": summary
                },
                "success",
            )

        except Exception as error:

            summary = None

            self.trace_manager.add_step(
                trace,
                "SummarizerAgent",
                article,
                status="failed",
                error=str(error),
            )

        # -----------------------------------------------
        # QA
        # -----------------------------------------------

        try:

            qa_result = (
                self.pipeline.answer_question(
                    question
                )
            )

            self.trace_manager.add_step(
                trace,
                "QAAgent",
                question,
                qa_result,
                "success",
            )

        except Exception as error:

            qa_result = {
                "status": "failed",
                "error": str(error),
            }

            self.trace_manager.add_step(
                trace,
                "QAAgent",
                question,
                status="failed",
                error=str(error),
            )

        # -----------------------------------------------
        # Fact Check explicit article claim
        # -----------------------------------------------

        try:

            fact_result = (
                self.pipeline.fact_check(
                    claim
                )
            )

            self.trace_manager.add_step(
                trace,
                "FactCheckerAgent",
                claim,
                fact_result,
                "success",
            )

        except Exception as error:

            fact_result = {
                "status": "failed",
                "error": str(error),
            }

            self.trace_manager.add_step(
                trace,
                "FactCheckerAgent",
                claim,
                status="failed",
                error=str(error),
            )

        trace["final_output"] = {
            "type":
                "full_pipeline_result",

            "summary":
                summary,

            "qa":
                qa_result,

            "claim":
                claim,

            "fact_check":
                fact_result,
        }

        trace["status"] = "success"

        trace["finished_at"] = (
            datetime.now().isoformat()
        )

        return trace