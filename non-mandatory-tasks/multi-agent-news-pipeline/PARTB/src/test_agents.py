from summarizer_agent import SummarizerAgent
from qa_agent import QAAgent
from fact_checker_agent import FactCheckerAgent


def test_summarizer():

    article = """
    The city council approved a new transportation plan
    on Tuesday. The plan includes additional bus routes,
    improved stations and expanded service during peak hours.
    Officials said the changes are intended to reduce
    congestion and improve public transportation.
    """

    agent = SummarizerAgent()

    result = agent.run(article)

    print("\n" + "=" * 70)
    print("SUMMARIZER AGENT")
    print("=" * 70)

    print(result)


def test_qa():

    question = (
        "What did the city council approve?"
    )

    agent = QAAgent()

    result = agent.run(
        question,
        top_k=5,
    )

    print("\n" + "=" * 70)
    print("QA AGENT")
    print("=" * 70)

    print(result)


def test_fact_checker():

    claim = (
        "The city council approved a new "
        "transportation plan."
    )

    agent = FactCheckerAgent()

    result = agent.run(
        claim,
        top_k=5,
    )

    print("\n" + "=" * 70)
    print("FACT CHECKER AGENT")
    print("=" * 70)

    print(result)


if __name__ == "__main__":

    test_summarizer()
    test_qa()
    test_fact_checker()