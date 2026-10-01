from google import genai
from google.genai import types

from config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
)


class GeminiClient:

    def __init__(
        self,
        model_name: str = GEMINI_MODEL
    ):

        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not configured. "
                "Add it to the project .env file."
            )

        self.model_name = model_name

        self.client = genai.Client(
            api_key=GEMINI_API_KEY
        )

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 512,
    ) -> str:
        """
        Generate a text response using Gemini.
        """

        config_kwargs = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }

        if system_instruction:
            config_kwargs[
                "system_instruction"
            ] = system_instruction

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                **config_kwargs
            ),
        )

        if not response.text:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        return response.text.strip()