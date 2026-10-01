from retriever import NewsQARetriever


def main():

    retriever = NewsQARetriever()

    query = "What was the amount of children murdered?"

    results = retriever.search(
        query,
        top_k=5
    )

    print("\n" + "=" * 70)
    print("RETRIEVAL RESULTS")
    print("=" * 70)

    for result in results:

        print("\n")
        print("Article ID:", result["article_id"])
        print("Chunk ID:", result["chunk_id"])
        print("Score:", result["score"])
        print("Text:", result["chunk_text"])


if __name__ == "__main__":
    main()