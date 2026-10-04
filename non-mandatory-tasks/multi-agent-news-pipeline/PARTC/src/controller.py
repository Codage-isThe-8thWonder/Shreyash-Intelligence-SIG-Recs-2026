from router import PipelineRouter
from trace_manager import TraceManager
from pipeline_handlers import PipelineHandlers


class NewsQAPipelineController:
    """
    High-level orchestration layer.

    Responsibilities:
    1. Normalize input.
    2. Ask router for route.
    3. Execute the selected pipeline.
    4. Maintain execution trace.
    """

    def __init__(
        self,
        top_k=5,
        max_fact_check_retries=1,
    ):

        self.router = PipelineRouter()

        self.trace_manager = (
            TraceManager()
        )

        self.handlers = PipelineHandlers(
            top_k=top_k,
            max_fact_check_retries=(
                max_fact_check_retries
            ),
        )

    # ========================================================
    # INPUT NORMALIZATION
    # ========================================================

    def _normalize_input(self, raw_input):

        # ----------------------------------------------------
        # Structured dictionary
        # ----------------------------------------------------

        if isinstance(raw_input, dict):

            input_type = raw_input.get(
                "type"
            )

            if input_type == "article":

                text = raw_input.get(
                    "text",
                    "",
                ).strip()

                if not text:
                    raise ValueError(
                        "Article cannot be empty."
                    )

                return {
                    "type": "article",
                    "text": text,
                }

            if input_type == "question":

                text = raw_input.get(
                    "text",
                    "",
                ).strip()

                if not text:
                    raise ValueError(
                        "Question cannot be empty."
                    )

                return {
                    "type": "question",
                    "text": text,
                }

            if input_type == "claim":

                text = raw_input.get(
                    "text",
                    "",
                ).strip()

                if not text:
                    raise ValueError(
                        "Claim cannot be empty."
                    )

                return {
                    "type": "claim",
                    "text": text,
                }

            if input_type == "article_question":

                article = raw_input.get(
                    "article",
                    "",
                ).strip()

                question = raw_input.get(
                    "question",
                    "",
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

            raise ValueError(
                "Invalid input type."
            )

        # ----------------------------------------------------
        # Plain text
        # ----------------------------------------------------

        if isinstance(raw_input, str):

            text = raw_input.strip()

            input_type = (
                self.router.detect_input_type(
                    text
                )
            )

            return {
                "type": input_type,
                "text": text,
            }

        raise ValueError(
            "Input must be a string or dictionary."
        )

    # ========================================================
    # RUN PIPELINE
    # ========================================================

    def run(self, raw_input):

        try:

            request = (
                self._normalize_input(
                    raw_input
                )
            )

            trace = (
                self.trace_manager.create_trace(
                    request
                )
            )

            input_type = request["type"]

            # ------------------------------------------------
            # Get route
            # ------------------------------------------------

            trace["route"] = (
                self.router.get_route(
                    input_type
                )
            )

            # ------------------------------------------------
            # Execute selected route
            # ------------------------------------------------

            if input_type == "article":

                result = (
                    self.handlers.handle_article(
                        article=request["text"],
                        trace=trace,
                        trace_manager=(
                            self.trace_manager
                        ),
                    )
                )

            elif input_type == "question":

                result = (
                    self.handlers.handle_question(
                        question=request["text"],
                        trace=trace,
                        trace_manager=(
                            self.trace_manager
                        ),
                    )
                )

            elif input_type == "claim":

                result = (
                    self.handlers.handle_claim(
                        claim=request["text"],
                        trace=trace,
                        trace_manager=(
                            self.trace_manager
                        ),
                    )
                )

            elif input_type == "article_question":

                result = (
                    self.handlers.handle_article_question(
                        article=request["article"],
                        question=request["question"],
                        trace=trace,
                        trace_manager=(
                            self.trace_manager
                        ),
                    )
                )

            else:

                raise ValueError(
                    f"Unsupported type: {input_type}"
                )

            trace["final_output"] = result

            # Even if an individual agent degraded
            # gracefully, the controller itself completed.
            trace["status"] = "success"

        except Exception as error:

            # ------------------------------------------------
            # Controller-level graceful failure
            # ------------------------------------------------

            if "trace" not in locals():

                trace = (
                    self.trace_manager.create_trace(
                        raw_input
                    )
                )

            trace["status"] = "failed"

            trace["final_output"] = {
                "message": (
                    "Pipeline execution failed."
                ),
                "error": str(error),
            }

        trace["finished_at"] = (
            __import__("datetime")
            .datetime.now()
            .isoformat()
        )

        return trace