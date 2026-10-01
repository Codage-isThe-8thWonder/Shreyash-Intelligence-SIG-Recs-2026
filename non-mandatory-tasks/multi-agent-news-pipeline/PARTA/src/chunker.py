import re
import pandas as pd

from config import MIN_SENTENCE_LENGTH


def split_into_sentences(text: str) -> list[str]:
    """
    Split an article into sentence-level chunks.

    A lightweight regex-based splitter is used to keep the pipeline
    dependency-light and reproducible.
    """

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    return sentences


def create_chunks(articles: pd.DataFrame) -> pd.DataFrame:
    """
    Convert articles into sentence-level chunks.

    Output columns:
        article_id
        chunk_id
        chunk_text
    """

    records = []

    for _, row in articles.iterrows():

        article_id = str(row["article_id"])
        article_text = row["article_text"]

        sentences = split_into_sentences(article_text)

        chunk_counter = 0

        for sentence in sentences:

            # Avoid extremely tiny chunks such as isolated punctuation
            # or very short fragments.
            if len(sentence) < MIN_SENTENCE_LENGTH:
                continue

            records.append(
                {
                    "article_id": article_id,
                    "chunk_id": chunk_counter,
                    "chunk_text": sentence,
                }
            )

            chunk_counter += 1

    chunks = pd.DataFrame(records)

    if chunks.empty:
        raise ValueError("No chunks were created.")

    print(f"Total chunks: {len(chunks)}")
    print(f"Unique articles represented: {chunks['article_id'].nunique()}")

    return chunks