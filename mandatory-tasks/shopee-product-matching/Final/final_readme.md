# Finale - Multimodal Product Matching

## Overview

The Finale combines textual and visual information to build the final Shopee Product Matching system.

The main idea is that text and images provide complementary information. When one modality is weak or ambiguous, the other can provide additional evidence.

## Approach

The system uses:

- Character-level TF-IDF for product titles
- CLIP ViT-B/32 for image embeddings
- Optional multilingual MiniLM sentence embeddings
- Image pHash as an additional exact-match rule

The dataset was split by `label_group` into validation and test sets so that product groups do not overlap between the two sets.

Similarity matrices were calculated separately for text and images.

The text and image scores were then combined using score-level fusion:

`Final Score = w × Text Score + (1 - w) × Image Score`

The fusion weight was selected using validation performance instead of choosing it manually.

## Experiments

The following configurations were compared:

- TF-IDF text only
- CLIP image only
- Equal-weight text + image fusion
- Word TF-IDF + CLIP
- Character TF-IDF + CLIP
- MiniLM + CLIP
- Character TF-IDF + MiniLM + CLIP
- Character TF-IDF + CLIP with pHash matching

Character-level TF-IDF combined with CLIP performed better than the other multimodal configurations.

An additional pHash rule was tested because exact pHash matches had very high precision in the dataset.

## Final System

The final configuration uses:

- Character TF-IDF for text
- CLIP ViT-B/32 for images
- 50% text weight
- 50% image weight
- pHash exact-match rule
- Similarity threshold of 0.60

The final test performance was:

- Precision: **0.9194**
- Recall: **0.8471**
- F1: **0.8478**

The pHash rule slightly improved the validation performance and was therefore retained in the final system.

## Error Analysis

False positives and false negatives were analyzed separately.

The analysis checked whether errors were mainly caused by:

- Text being misleading
- Image similarity being misleading
- Both modalities being weak
- Short product titles
- Conflicting numeric/model information
- Visually similar product variants

This helped explain not only the final score but also why the system succeeds or fails on difficult examples.

## Conclusion

The experiments show that text and image information are complementary for product matching.

Character-level text similarity captures product names, specifications, and small textual variations, while CLIP provides visual product-level information.

Combining both modalities produced a more balanced matching system than relying on either text or image alone. The additional pHash rule was useful for highly reliable exact image matches.