class EvidencePolicy:
    """
    Applies deterministic safeguards to evidence returned
    by the Fact Checker.
    """

    VALID_VERDICTS = {
        "corroborated",
        "contradicted",
        "unverifiable",
    }

    def normalize_verdict(self, verdict):
        """
        Normalize model verdict.
        """

        if verdict is None:
            return "unverifiable"

        verdict = str(verdict).strip().lower()

        if verdict not in self.VALID_VERDICTS:
            return "unverifiable"

        return verdict

    def assess(self, result):
        """
        Assess the quality of a fact-check result.
        """

        verdict = self.normalize_verdict(
            result.get("verdict")
        )

        evidence = result.get(
            "evidence",
            []
        )

        if not evidence:
            return {
                "verdict": "unverifiable",
                "confidence": "low",
                "reason": (
                    "No supporting or contradicting "
                    "evidence was returned."
                ),
            }

        scores = []

        for item in evidence:

            score = item.get("score")

            if score is not None:

                try:
                    scores.append(
                        float(score)
                    )
                except (TypeError, ValueError):
                    pass

        # --------------------------------------------------
        # No usable scores
        # --------------------------------------------------

        if not scores:

            return {
                "verdict": verdict,
                "confidence": "medium",
                "reason": (
                    "Evidence was returned, but "
                    "retrieval scores were unavailable."
                ),
            }

        best_score = max(scores)

        # --------------------------------------------------
        # Weak retrieval evidence
        # --------------------------------------------------

        if best_score < 0.45:

            return {
                "verdict": "unverifiable",
                "confidence": "low",
                "reason": (
                    "Retrieved evidence was too weak "
                    "to support a reliable verdict."
                ),
            }

        # --------------------------------------------------
        # Strong evidence
        # --------------------------------------------------

        if best_score >= 0.70:

            confidence = "high"

        else:

            confidence = "medium"

        return {
            "verdict": verdict,
            "confidence": confidence,
            "best_retrieval_score": best_score,
            "reason": (
                "Evidence strength is sufficient "
                "for the model verdict."
            ),
        }

    def finalize(self, result):
        """
        Return a safe fact-check result.
        """

        assessment = self.assess(
            result
        )

        final_result = dict(result)

        final_result["verdict"] = (
            assessment["verdict"]
        )

        final_result["confidence"] = (
            assessment["confidence"]
        )

        final_result["evidence_assessment"] = (
            assessment
        )

        return final_result