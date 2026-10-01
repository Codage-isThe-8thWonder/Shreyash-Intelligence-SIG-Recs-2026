from pathlib import Path


# Paths

PART_A_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PART_A_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "embeddings.csv"



# Dataset

DATASET_NAME = "lucadiliello/newsqa"
DATASET_SPLIT = "train"


# Embedding model

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# Chunking

MIN_SENTENCE_LENGTH = 20


# Processing

BATCH_SIZE = 32