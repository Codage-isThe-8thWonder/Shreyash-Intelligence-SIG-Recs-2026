import numpy as np
import pandas as pd

from config import OUTPUT_FILE


def load_embeddings():

    df = pd.read_csv(OUTPUT_FILE)

    embeddings = np.vstack(
        df["embedding"]
        .apply(
            lambda x: np.array(
                [float(v) for v in x.split(",")]
            )
        )
    )

    return df, embeddings


def cosine_search(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    top_k: int = 5
):
    """
    Retrieve the top-k most similar chunks.

    Embeddings are already normalized, so cosine similarity
    is equivalent to the dot product.
    """

    scores = embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:top_k]

    return top_indices, scores[top_indices]