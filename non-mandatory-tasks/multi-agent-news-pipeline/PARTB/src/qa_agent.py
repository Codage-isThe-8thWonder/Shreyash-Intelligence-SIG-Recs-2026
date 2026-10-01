import json

from llm_client import GeminiClient
from retriever import NewsQARetriever


class QAAgent:

    """
    Single-purpose question answering agent.

    It retrieves relevant NewsQA chunks and then asks
    the LLM to answer strictly from those chunks.
    """

    SYSTEM_INSTRUCTION = """
You are a NewsQA question-answering agent.

Your task is to answer a user's question using ONLY the
provided evidence passages.

Rules:
1. Do not use outside knowledge.
2. Do not guess.
3. If the evidence does not support an answer, say:
   "Insufficient evidence in the provided corpus."
4. Keep the answer concise.
5. Return the answer and identify which evidence chunks support it.
"""

    def __init__(
        self,
        retriever: NewsQARetriever | None = None,
        llm: GeminiClient | None = None,
    ):

        self.retriever = (
            retriever
            if retriever is not None
            else NewsQARetriever()
        )

        self.llm = (
            llm
            if llm is not None
            else GeminiClient()
        )

    def run(
        self,
        question: str,
        top_k: int = 5,
    ) -> dict:

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        # --------------------------------------
        # 1. Retrieve evidence
        # --------------------------------------

        evidence = self.retriever.search(
            question,
            top_k=top_k,
        )

        if not evidence:

            return {
                "answer": (
                    "Insufficient evidence in "
                    "the provided corpus."
                ),
                "evidence": [],
            }

        # --------------------------------------
        # 2. Format evidence
        # --------------------------------------

        evidence_text = self._format_evidence(
            evidence
        )

        # --------------------------------------
        # 3. Ask LLM
        # --------------------------------------

        prompt = f"""
Answer the following question using ONLY
the evidence below.

QUESTION:
{question}

EVIDENCE:
{evidence_text}

Return valid JSON with exactly this structure:

{{
    "answer": "short grounded answer",
    "evidence_chunk_ids": [1, 2]
}}

If the evidence is insufficient, use:

{{
    "answer": "Insufficient evidence in the provided corpus.",
    "evidence_chunk_ids": []
}}
"""

        response = self.llm.generate(
            prompt=prompt,
            system_instruction=self.SYSTEM_INSTRUCTION,
            temperature=0.0,
            max_output_tokens=300,
        )

        parsed = self._parse_json(
            response
        )

        selected_ids = set(
            parsed.get(
                "evidence_chunk_ids",
                []
            )
        )

        selected_evidence = [
            item
            for item in evidence
            if item["chunk_id"] in selected_ids
        ]

        return {
            "answer": parsed.get(
                "answer",
                "Insufficient evidence in the provided corpus.",
            ),
            "evidence": selected_evidence,
        }

    @staticmethod
    def _format_evidence(
        evidence: list[dict]
    ) -> str:

        blocks = []

        for item in evidence:

            blocks.append(
                f"""
[Chunk ID: {item['chunk_id']}]
[Article ID: {item['article_id']}]
[Similarity: {item['score']}]

{item['chunk_text']}
"""
            )

        return "\n".join(blocks)

    @staticmethod
    def _parse_json(
        response: str
    ) -> dict:

        response = response.strip()

        # Handle accidental markdown fences.
        if response.startswith("```"):
            response = (
                response
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        try:
            return json.loads(response)

        except json.JSONDecodeError as exc:

            raise ValueError(
                f"QA Agent returned invalid JSON:\n"
                f"{response}"
            ) from exc