# Part A - Dataset Exploration

## Overview

This part focuses on understanding the Shopee Product Matching dataset before building any matching model.

The main goal was to study how product listings vary in their titles and images, and to identify the difficulties involved in determining whether two listings represent the same product.

## Approach

The dataset was first inspected to understand its columns, missing values, duplicate listings, product groups, images, and titles.

The main analyses included:

- Number of listings and unique product groups
- Distribution of product-group sizes
- Number of unique images and image hashes
- Duplicate image and title analysis
- Product title length and word-count analysis
- Variation of titles and images within the same product group
- Text and image similarity analysis
- Examples of difficult matching cases

The dataset contains 34,250 training listings and 11,014 product groups. Each listing contains a `posting_id`, image filename, image perceptual hash, title, and `label_group`.

Product groups were treated as the ground truth for product identity.

## Similarity Analysis

TF-IDF cosine similarity was used to study textual similarity between product titles.

Perceptual image hashes were used to study visual similarity and identify duplicate or highly similar images.

Examples were also inspected manually to understand cases such as:

- Same product with different titles
- Same product with different images
- Similar titles belonging to different products
- Noisy or incomplete product information

The analysis showed that text and image information provide complementary signals.

## Key Findings

The dataset contains substantial variation between listings of the same product. At the same time, different products can have very similar titles or images.

For example, title-based nearest-neighbour matching achieved better performance than pHash-based matching, but neither modality was sufficient for every case.

The analysis also showed that many listings contain promotional text, uppercase text, very short titles, or long keyword-heavy titles.

## Conclusion

Product matching is not simple duplicate detection. The same product can appear in many different textual and visual forms, while different products can look or sound very similar.

These observations motivated the use of separate text-based and image-based approaches in Part B and Part C, followed by a multimodal approach in the Finale.