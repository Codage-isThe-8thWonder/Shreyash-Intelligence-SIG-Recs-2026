class PipelineRouter:
    """
    Decides which pipeline route should be used
    based on the normalized input type.
    """

    VALID_TYPES = {
        "article",
        "question",
        "claim",
        "article_question",
    }

    def detect_input_type(self, text):
        """
        Simple heuristic for plain string input.

        Explicit structured input is preferred.
        """

        if not isinstance(text, str):
            raise ValueError(
                "Input must be a string."
            )

        text = text.strip()

        if not text:
            raise ValueError(
                "Input cannot be empty."
            )

        # Question
        if text.endswith("?"):
            return "question"

        # Long text → article
        if len(text.split()) >= 80:
            return "article"

        # Short declarative statement → claim
        return "claim"

    def get_route(self, input_type):
        """
        Return the ordered list of agents for an input type.
        """

        routes = {
            "article": [
                "SummarizerAgent"
            ],

            "question": [
                "QAAgent"
            ],

            "claim": [
                "FactCheckerAgent"
            ],

            "article_question": [
                "SummarizerAgent",
                "QAAgent",
                "FactCheckerAgent",
            ],
        }

        if input_type not in routes:
            raise ValueError(
                f"Unsupported input type: {input_type}"
            )

        return routes[input_type]