from pathlib import Path
import os

from dotenv import load_dotenv


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PART_B_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PART_B_DIR.parent

PART_A_DIR = PROJECT_ROOT / "PartA"

EMBEDDINGS_FILE = (
    PART_A_DIR / "output" / "embeddings.csv"
)


# --------------------------------------------------
# Environment
# --------------------------------------------------

load_dotenv(
    PROJECT_ROOT / ".env"
)

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)


# --------------------------------------------------
# Gemini configuration
# --------------------------------------------------

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)


# --------------------------------------------------
# Retrieval configuration
# --------------------------------------------------

DEFAULT_TOP_K = 5

MIN_RELEVANCE_SCORE = 0.30


# --------------------------------------------------
# Validation
# --------------------------------------------------

if not GEMINI_API_KEY:
    print(
        "WARNING: GEMINI_API_KEY is not set. "
        "LLM-based agents will not work until it is configured."
    )