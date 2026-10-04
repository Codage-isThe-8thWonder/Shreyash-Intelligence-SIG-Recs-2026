class FinaleRouter:
    """
    Determines which agents should be executed
    for a given structured input.
    """

    ROUTES = {
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
            "FactCheckerAgent"
        ],

        "article_claim": [
            "SummarizerAgent",
            "FactCheckerAgent"
        ],

        "article_question_claim": [
            "SummarizerAgent",
            "QAAgent",
            "FactCheckerAgent"
        ],
    }

    def get_route(self, input_type):
        """
        Return ordered agent route.
        """

        if input_type not in self.ROUTES:
            raise ValueError(
                f"Unsupported input type: {input_type}"
            )

        return self.ROUTES[input_type]

    def detect_string_type(self, text):
        """
        Lightweight fallback for unstructured text.

        Structured dictionaries are preferred.
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

        if text.endswith("?"):
            return "question"

        if len(text.split()) >= 80:
            return "article"

        return "claim"