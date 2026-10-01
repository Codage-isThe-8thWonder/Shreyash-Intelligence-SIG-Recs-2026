from pathlib import Path
import os

from dotenv import load_dotenv


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PART_B_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PART_B_DIR.parent

PART_A_DIR = PROJECT_ROOT / "PartA"

EMBEDDINGS_FILE = PART_A_DIR / "output" / "embeddings.csv"


# ---------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------

load_dotenv(PROJECT_ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)


# ---------------------------------------------------------
# RETRIEVER CONFIG
# ---------------------------------------------------------

DEFAULT_TOP_K = 5
MIN_RELEVANCE_SCORE = 0.30


# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not configured.\n"
        "Please add GEMINI_API_KEY to the project .env file."
    )