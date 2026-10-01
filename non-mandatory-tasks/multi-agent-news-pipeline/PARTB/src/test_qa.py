from qa_agent import QAAgent


def main():

    agent = QAAgent()

    question = (
        "What was the amount of children murdered?"
    )

    result = agent.run(
        question,
        top_k=5
    )

    print("\n" + "=" * 70)
    print("QUERY-ANSWERING AGENT")
    print("=" * 70)

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(result["answer"])

    print("\nEVIDENCE:")

    for item in result["evidence"]:

        print("\n----------------------------")
        print("Article ID:", item["article_id"])
        print("Chunk ID:", item["chunk_id"])
        print("Score:", item["score"])
        print("Text:", item["chunk_text"])


if __name__ == "__main__":
    main()