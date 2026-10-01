from llm_client import GeminiClient


def main():

    print("=" * 70)
    print("GEMINI CONNECTION TEST")
    print("=" * 70)

    client = GeminiClient()

    response = client.generate(
        prompt="Reply with exactly: Gemini connection successful.",
        max_output_tokens=256,
        thinking_level="low",
    )

    print("\nGemini response:")
    print(response)

    print("\n" + "=" * 70)
    print("TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()