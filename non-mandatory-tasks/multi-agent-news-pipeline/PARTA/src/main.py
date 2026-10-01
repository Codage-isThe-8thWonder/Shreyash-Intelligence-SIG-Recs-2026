import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Allow imports from the same directory when running this file directly.
CURRENT_DIR = Path(__file__).resolve().parent

if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))


from config import OUTPUT_DIR, OUTPUT_FILE
from data_loader import load_newsqa, prepare_articles
from chunker import create_chunks
from embedder import NewsEmbedder


def main():

    print("=" * 70)
    print("PART A - NEWSQA EMBEDDING PIPELINE")
    print("=" * 70)

    # 1. Load NewsQA
    df = load_newsqa()

    
    # 2. Prepare unique articles
    articles = prepare_articles(df)

    # ---------------------------------
    # 3. Create sentence-level chunks

    chunks = create_chunks(articles)

    print("\nSample chunks:")
    print(chunks.head(5).to_string(index=False))

    # 4. Load embedding model
    # ---------------------------------
    embedder = NewsEmbedder()
    

    # ---------------------------------
    # 5. Generate embeddings
    # ---------------------------------

    texts = chunks["chunk_text"].tolist()

    embeddings = embedder.encode(texts)

    print(
        f"\nEmbedding matrix shape: {embeddings.shape}"
    )

    # ---------------------------------
    # 6. Convert embeddings to strings
    # ---------------------------------

    embedding_strings = [
        ",".join(map(str, vector))
        for vector in embeddings
    ]

    # ---------------------------------
    # 7. Build final output
    # ---------------------------------

    output_df = pd.DataFrame(
        {
            "article_id": chunks["article_id"],
            "chunk_id": chunks["chunk_id"],
            "chunk_text": chunks["chunk_text"],
            "embedding": embedding_strings,
        }
    )

    # ---------------------------------
    # 8. Create output directory
    # ---------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------
    # 9. Save CSV
    # ---------------------------------

    output_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED")
    print("=" * 70)

    print(f"Output file: {OUTPUT_FILE}")
    print(f"Rows: {len(output_df)}")
    print(f"Columns: {list(output_df.columns)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    print("\nOutput preview:")
    print(
        output_df.head(3).to_string(index=False)
    )


if __name__ == "__main__":
    main()