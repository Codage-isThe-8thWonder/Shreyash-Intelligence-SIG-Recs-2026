# Part B - Text-Based Product Matching

## Overview

This part focuses on matching Shopee product listings using their titles.

The main objective was to determine how well textual information alone can identify whether two listings belong to the same product.

## Approach

The dataset was split using `GroupShuffleSplit` based on `label_group` so that the same product group does not appear across training, validation, and test sets.

Positive pairs were created from listings belonging to the same product group.

Negative pairs were created using two strategies:

- Random negatives
- Hard negatives selected using similar product titles

This makes the task more realistic because hard negatives contain similar words but belong to different products.

## Text Representations

Several text representations were compared:

- Word-level TF-IDF
- Word 1-2 gram TF-IDF
- Character-level TF-IDF
- Multilingual MiniLM sentence embeddings
- Hybrid combinations of different text representations

Cosine similarity was used for TF-IDF and embedding-based comparisons.

Character-level TF-IDF performed better than simple word-level TF-IDF because it can handle spelling variations, abbreviations, and partial word similarities more effectively.

## Final Text Model

Pair-level features were created from different types of text similarity and additional information such as:

- Word-level cosine similarity
- Character-level cosine similarity
- Sentence-embedding similarity
- Title length differences
- Numeric information
- Token overlap

A Logistic Regression model was trained using these pair features.

The final model achieved a test F1 of approximately **0.832** on the pair-matching evaluation.

## Error Analysis

False positives and false negatives were examined to understand where text matching fails.

Important failure cases included:

- Similar titles belonging to different products
- Product variants with different numbers or specifications
- Very short titles
- Noisy or promotional text
- Hard negative pairs with strong lexical similarity

## Conclusion

Text provides a strong signal for product matching, but exact word overlap is not sufficient.

Character-level features and semantic information improve robustness, while pair-level modelling helps combine different textual signals.

However, some cases cannot be resolved reliably using titles alone, which motivates the image-based approach in Part C.