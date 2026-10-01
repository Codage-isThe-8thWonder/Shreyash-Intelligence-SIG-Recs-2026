from fact_checker_agent import FactCheckerAgent


def main():

    agent = FactCheckerAgent()

    claim = (
        "The serial killings involved 19 victims."
    )

    result = agent.run(
        claim,
        top_k=5
    )

    print("\n" + "=" * 70)
    print("FACT-CHECKER AGENT")
    print("=" * 70)

    print("\nCLAIM:")
    print(claim)

    print("\nVERDICT:")
    print(result["verdict"])

    print("\nEXPLANATION:")
    print(result["explanation"])

    print("\nEVIDENCE:")

    for item in result["evidence"]:

        print("\n----------------------------")
        print("Article ID:", item["article_id"])
        print("Chunk ID:", item["chunk_id"])
        print("Score:", item["score"])
        print("Text:", item["chunk_text"])


if __name__ == "__main__":
    main()