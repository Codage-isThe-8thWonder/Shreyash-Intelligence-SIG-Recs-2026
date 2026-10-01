from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDINGS_FILE,
    DEFAULT_TOP_K,
)


class NewsQARetriever:

    def __init__(
        self,
        embeddings_file: str | Path = EMBEDDINGS_FILE,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.embeddings_file = Path(
            embeddings_file
        )

        if not self.embeddings_file.exists():
            raise FileNotFoundError(
                f"Embeddings file not found: "
                f"{self.embeddings_file}"
            )

        print("Loading NewsQA embeddings...")

        self.data = pd.read_csv(
            self.embeddings_file
        )

        required_columns = {
            "article_id",
            "chunk_id",
            "chunk_text",
            "embedding",
        }

        missing = (
            required_columns
            - set(self.data.columns)
        )

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

        print(
            f"Loaded {len(self.data)} chunks."
        )

        print(
            f"Loading embedding model: {model_name}"
        )

        self.encoder = SentenceTransformer(
            model_name
        )

        self.embeddings = self._parse_embeddings()

        print(
            f"Embedding matrix shape: "
            f"{self.embeddings.shape}"
        )

    def _parse_embeddings(self) -> np.ndarray:
        """
        Convert CSV embedding strings into a NumPy matrix.
        """

        vectors = []

        for value in self.data["embedding"]:

            vector = np.array(
                [
                    float(x)
                    for x in str(value).split(",")
                ],
                dtype=np.float32,
            )

            vectors.append(vector)

        matrix = np.vstack(vectors)

        # Normalize for cosine similarity.
        norms = np.linalg.norm(
            matrix,
            axis=1,
            keepdims=True,
        )

        norms = np.maximum(
            norms,
            1e-12
        )

        matrix = matrix / norms

        return matrix

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = 0.0,
    ) -> list[dict]:

        if not query.strip():
            return []

        # Encode query.
        query_embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        # Cosine similarity because vectors are normalized.
        scores = (
            self.embeddings @ query_embedding
        )

        top_k = min(
            top_k,
            len(scores)
        )

        top_indices = np.argsort(
            scores
        )[::-1][:top_k]

        results = []

        for index in top_indices:

            score = float(
                scores[index]
            )

            if score < min_score:
                continue

            row = self.data.iloc[index]

            results.append(
                {
                    "article_id": str(
                        row["article_id"]
                    ),
                    "chunk_id": int(
                        row["chunk_id"]
                    ),
                    "chunk_text": str(
                        row["chunk_text"]
                    ),
                    "score": round(
                        score,
                        4
                    ),
                }
            )

        return results