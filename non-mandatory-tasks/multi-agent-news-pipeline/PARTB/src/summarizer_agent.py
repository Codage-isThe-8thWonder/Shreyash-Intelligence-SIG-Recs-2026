from llm_client import GeminiClient


class SummarizerAgent:

    """
    Single-purpose agent responsible only for summarization.
    """

    SYSTEM_INSTRUCTION = """
You are a news article summarization agent.

Your only task is to summarize the provided NewsQA article.

Rules:
1. Preserve the main event, important people, organizations,
   locations, causes, and outcomes when present.
2. Do not invent facts.
3. Do not introduce information that is not present in the article.
4. Do not answer questions unless explicitly required by the summary.
5. Do not copy large portions of the article verbatim.
6. Keep the summary concise and coherent.
"""

    def __init__(
        self,
        llm: GeminiClient | None = None,
    ):

        self.llm = (
            llm
            if llm is not None
            else GeminiClient()
        )

    def run(
        self,
        article: str,
    ) -> str:

        if not article.strip():
            raise ValueError(
                "Article cannot be empty."
            )

        prompt = f"""
Summarize the following NewsQA article.

Target length:
Approximately 3-6 sentences.

ARTICLE:
--------------------
{article}
--------------------

Return only the summary.
"""

        return self.llm.generate(
            prompt=prompt,
            system_instruction=self.SYSTEM_INSTRUCTION,
            temperature=0.2,
            max_output_tokens=300,
        )