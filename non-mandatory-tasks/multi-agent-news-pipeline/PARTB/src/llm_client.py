from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL


class GeminiClient:
    """
    Small wrapper around the Google Gemini API.

    All Part B agents use this class instead of directly
    calling the Gemini SDK.
    """

    def __init__(self, model_name: str = GEMINI_MODEL):

        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not configured."
            )

        self.model_name = model_name

        self.client = genai.Client(
            api_key=GEMINI_API_KEY
        )

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int = 512,
        thinking_level: str = "low",
    ) -> str:
        """
        Generate a text response from Gemini.

        For Gemini 3.8 Flash:
        - thinking_level controls reasoning effort
        - max_output_tokens includes thinking + visible output
        """

        if not prompt or not prompt.strip():
            raise ValueError(
                "Prompt cannot be empty."
            )

        # -------------------------------------------------
        # Generation configuration
        # -------------------------------------------------

        config_kwargs = {
            "max_output_tokens": max_output_tokens,

            "thinking_config": types.ThinkingConfig(
                thinking_level=thinking_level
            ),
        }

        # Gemini 3.x documentation recommends controlling
        # reasoning through thinking_level.
        #
        # Keep temperature optional rather than forcing it.
        if temperature is not None:
            config_kwargs["temperature"] = temperature

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        generation_config = types.GenerateContentConfig(
            **config_kwargs
        )

        # -------------------------------------------------
        # API call
        # -------------------------------------------------

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=generation_config,
        )

        # -------------------------------------------------
        # Robust response validation
        # -------------------------------------------------

        if response is None:
            raise RuntimeError(
                "Gemini returned no response object."
            )

        # Get finish reason safely
        finish_reason = None

        if response.candidates:
            finish_reason = response.candidates[0].finish_reason

        # -------------------------------------------------
        # Successful response
        # -------------------------------------------------

        if response.text:
            return response.text.strip()

        # -------------------------------------------------
        # Empty response diagnostics
        # -------------------------------------------------

        usage = getattr(response, "usage_metadata", None)

        prompt_tokens = getattr(
            usage,
            "prompt_token_count",
            None
        )

        output_tokens = getattr(
            usage,
            "candidates_token_count",
            None
        )

        thought_tokens = getattr(
            usage,
            "thoughts_token_count",
            None
        )

        # -------------------------------------------------
        # MAX TOKENS
        # -------------------------------------------------

        if str(finish_reason) == "MAX_TOKENS":
            raise RuntimeError(
                "\nGemini stopped because MAX_TOKENS was reached.\n"
                f"Model: {self.model_name}\n"
                f"Max output tokens: {max_output_tokens}\n"
                f"Thinking level: {thinking_level}\n"
                f"Prompt tokens: {prompt_tokens}\n"
                f"Thinking tokens: {thought_tokens}\n"
                f"Candidate output tokens: {output_tokens}\n\n"
                "Increase max_output_tokens or reduce thinking_level."
            )

        # -------------------------------------------------
        # SAFETY
        # -------------------------------------------------

        if str(finish_reason) == "SAFETY":
            raise RuntimeError(
                "\nGemini blocked the response for safety reasons.\n"
                f"Model: {self.model_name}\n"
                f"Finish reason: {finish_reason}"
            )

        # -------------------------------------------------
        # OTHER EMPTY RESPONSE
        # -------------------------------------------------

        raise RuntimeError(
            "\nGemini returned an empty response.\n"
            f"Model: {self.model_name}\n"
            f"Finish reason: {finish_reason}\n"
            f"Prompt tokens: {prompt_tokens}\n"
            f"Thinking tokens: {thought_tokens}\n"
            f"Candidate output tokens: {output_tokens}"
        )