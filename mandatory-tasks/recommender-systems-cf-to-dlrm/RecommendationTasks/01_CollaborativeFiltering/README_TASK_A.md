# Task A - Collaborative Filtering

This task compares two classical collaborative filtering approaches on the MovieLens 100K dataset: Memory-Based User-User Collaborative Filtering and Matrix Factorization.

## Approach

First, the ratings were split user-wise into train, validation, and test sets so that every user had data in each split. A user-item rating matrix was then created from the training data.

For Memory-Based CF, cosine similarity was calculated between users. For a target rating, the ratings from the top-K similar users were combined using similarity-weighted averaging. Different K values were checked on the validation set and K=40 was selected.

For Matrix Factorization, each user and movie was represented by a 32-dimensional embedding. User bias, item bias, and a global bias were also included. The model was trained with MSE loss and Adam using mini-batches. Validation RMSE was monitored and the best checkpoint was restored before test evaluation.

## Why these methods

Memory-Based CF is simple and directly uses similarities between users. Matrix Factorization is a model-based approach that learns hidden user-item relationships. Using both gives a useful comparison between neighbourhood-based and latent-factor methods.

## Evaluation

MAE and RMSE were used because the task predicts numerical ratings. The final test results were:

Memory-Based CF:
MAE = 0.8285
RMSE = 1.0502

Matrix Factorization:
MAE = 0.8178
RMSE = 1.0494

Matrix Factorization performed slightly better on the final test set.

The notebook also includes K-selection and training/validation plots for analysis.

## Important limitations

The Memory-Based implementation uses cosine similarity on the filled user-item matrix, so missing ratings are represented as zero during similarity calculation. A mean-centered similarity approach could be explored as a further improvement.

Both approaches also have the usual cold-start limitation for completely new users or items.

## Final outcome

The task demonstrates the difference between neighbourhood-based and latent-factor collaborative filtering and evaluates both using the same held-out test data.
