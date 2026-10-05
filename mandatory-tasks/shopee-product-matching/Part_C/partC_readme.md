# Part C - Image-Based Product Matching

## Overview

This part focuses on identifying matching Shopee products using product images.

The objective was to determine whether visual representations can identify the same product even when images have different backgrounds, viewpoints, or presentation styles.

## Approach

A subset of the dataset was selected while keeping complete product groups together.

The data was then divided into validation and test sets using `label_group` so that product groups did not overlap between the two splits.

Two pretrained image models were investigated:

- ResNet50 pretrained on ImageNet
- CLIP ViT-B/32

The models were used as feature extractors rather than being trained from scratch.

Each image was converted into an embedding vector. The embeddings were L2-normalized and cosine similarity was used to compare product images.

## Model Comparison

ResNet50 produced 2048-dimensional image embeddings, while CLIP produced 512-dimensional embeddings.

The experiments showed that both models were able to distinguish same-product images from random different-product images.

ResNet50 achieved the best validation/test F1 in the pair-matching experiment and was selected as the final image model.

The selected ResNet50 model achieved approximately:

- Test Precision: **0.834**
- Test Recall: **0.896**
- Test F1: **0.864**

A retrieval experiment was also performed to evaluate whether the correct product appears among the nearest visual neighbours.

## Threshold Selection

Instead of choosing an arbitrary similarity threshold, the threshold was tuned on the validation set by maximizing F1.

The selected threshold for the final ResNet50 configuration was approximately **0.454**.

The effect of changing the threshold was also examined to understand the precision-recall trade-off.

## Error Analysis

Nearest-neighbour examples and false positives/false negatives were inspected visually.

This helped identify cases where:

- Different products look visually similar
- The same product is shown from different viewpoints
- Backgrounds or image composition differ
- Fine-grained product differences are difficult to distinguish

## Conclusion

Image embeddings provide strong product-level information and can handle visual variation better than simple image hashing.

However, visual similarity alone can still confuse products that look alike. This motivates combining image information with textual information in the final multimodal system.