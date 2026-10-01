import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import EMBEDDING_MODEL, BATCH_SIZE


class NewsEmbedder:

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

        print(
            f"Embedding dimension: "
            f"{self.model.get_sentence_embedding_dimension()}"
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a list of text chunks.
        """

        embeddings = self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings