from datasets import load_dataset
import pandas as pd

from config import DATASET_NAME, DATASET_SPLIT


def load_newsqa() -> pd.DataFrame:
    """
    Load the required NewsQA dataset from Hugging Face.

    """

    print(f"Loading dataset: {DATASET_NAME}")

    dataset = load_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT
    )

    df = dataset.to_pandas()

    print(f"Loaded rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    return df


def prepare_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert question-level NewsQA records into unique article records.

    The NewsQA dataset can contain multiple questions for the same article.
    We therefore deduplicate using the dataset's article identifier.
    """

    required_columns = {"context", "key"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    articles = (
        df[["key", "context"]]
        .drop_duplicates(subset=["key"])
        .rename(
            columns={
                "key": "article_id",
                "context": "article_text"
            }
        )
        .reset_index(drop=True)
    )

    # Remove empty articles
    articles["article_text"] = (
        articles["article_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    articles = articles[
        articles["article_text"].str.len() > 0
    ].reset_index(drop=True)

    print(f"Unique articles: {len(articles)}")

    return articles