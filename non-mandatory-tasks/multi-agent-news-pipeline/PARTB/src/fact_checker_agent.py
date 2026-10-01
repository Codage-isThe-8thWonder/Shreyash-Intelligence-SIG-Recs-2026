import json

from llm_client import GeminiClient
from retriever import NewsQARetriever


class FactCheckerAgent:

    """
    Single-purpose fact-checking agent.

    It retrieves evidence from the Part A NewsQA corpus
    and determines whether the supplied claim is supported,
    contradicted, or unverifiable based only on that corpus.
    """

    SYSTEM_INSTRUCTION = """
You are a fact-checking agent operating over a fixed NewsQA corpus.

Your task is to evaluate a claim using ONLY the supplied evidence.

Allowed verdicts:

- corroborated:
  The evidence directly supports the claim.

- contradicted:
  The evidence directly conflicts with the claim.

- unverifiable:
  The available evidence is insufficient to establish
  whether the claim is true or false.

Rules:
1. Never use outside knowledge.
2. Never invent evidence.
3. Do not treat missing evidence as contradiction.
4. Prefer unverifiable when the corpus does not contain
   enough information.
5. Cite the specific evidence chunks used.
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
        claim: str,
        top_k: int = 5,
    ) -> dict:

        if not claim.strip():
            raise ValueError(
                "Claim cannot be empty."
            )

        # --------------------------------------
        # 1. Retrieve relevant evidence
        # --------------------------------------

        evidence = self.retriever.search(
            claim,
            top_k=top_k,
        )

        # --------------------------------------
        # 2. Guardrail
        # --------------------------------------

        if not evidence:

            return {
                "verdict": "unverifiable",
                "explanation": (
                    "No relevant evidence was found "
                    "in the NewsQA corpus."
                ),
                "evidence": [],
            }

        # --------------------------------------
        # 3. Format evidence
        # --------------------------------------

        evidence_text = self._format_evidence(
            evidence
        )

        # --------------------------------------
        # 4. Ask LLM
        # --------------------------------------

        prompt = f"""
Evaluate the following claim using ONLY
the provided NewsQA evidence.

CLAIM:
{claim}

EVIDENCE:
{evidence_text}

Return valid JSON with exactly this structure:

{{
    "verdict": "corroborated | contradicted | unverifiable",
    "explanation": "brief explanation",
    "evidence_chunk_ids": [1, 2]
}}

Important:
- Missing evidence is NOT contradiction.
- Use "unverifiable" when the evidence is insufficient.
- Do not use outside knowledge.
"""

        response = self.llm.generate(
            prompt=prompt,
            system_instruction=self.SYSTEM_INSTRUCTION,
            temperature=0.0,
            max_output_tokens=350,
        )

        parsed = self._parse_json(
            response
        )

        verdict = (
            parsed
            .get("verdict", "unverifiable")
            .lower()
            .strip()
        )

        allowed = {
            "corroborated",
            "contradicted",
            "unverifiable",
        }

        if verdict not in allowed:
            verdict = "unverifiable"

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
            "verdict": verdict,
            "explanation": parsed.get(
                "explanation",
                "The available evidence is insufficient.",
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
                f"Fact Checker returned invalid JSON:\n"
                f"{response}"
            ) from exc