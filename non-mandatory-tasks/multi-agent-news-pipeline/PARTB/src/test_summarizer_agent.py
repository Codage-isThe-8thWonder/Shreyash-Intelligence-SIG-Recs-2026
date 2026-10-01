from datasets import load_dataset

from summarizer_agent import SummarizerAgent


def main():

    dataset = load_dataset(
        "lucadiliello/newsqa",
        split="train"
    )

    article = dataset[0]["context"]

    agent = SummarizerAgent()

    summary = agent.run(article)

    print("\n" + "=" * 70)
    print("SUMMARIZER AGENT")
    print("=" * 70)

    print("\nINPUT ARTICLE:\n")
    print(article)

    print("\nSUMMARY:\n")
    print(summary)


if __name__ == "__main__":
    main()