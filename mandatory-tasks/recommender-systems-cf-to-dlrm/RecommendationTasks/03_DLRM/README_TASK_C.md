# Task C - DLRM

This task implements a Deep Learning Recommendation Model (DLRM) for CTR prediction using the same type of mixed numerical and categorical recommendation data.

## Approach

The data was split into training and validation sets using stratification. Numerical features were filled using training-set medians and standardized. Each categorical feature was mapped to integer IDs, with a separate unknown ID for unseen values.

The model uses an embedding table for every categorical field. Numerical features are passed through a bottom MLP to produce a dense representation. The dense representation and categorical embeddings are then used to compute pairwise feature interactions. These interactions are passed to a top MLP to produce the final click logit.

The model uses weighted binary cross-entropy because the positive class represents only about 3.2% of the training data.

## Why DLRM

CTR datasets usually contain many categorical fields with large cardinalities. Embeddings provide compact representations for these fields, while DLRM is designed specifically to learn interactions between dense and categorical features. This makes it a natural architecture for this recommendation problem.

## Training and evaluation

The DLRM used 16-dimensional embeddings, a bottom MLP of (64, 32), a top MLP of (128, 64), dropout, batch normalization, AdamW, and early stopping based on validation ROC-AUC.

The full interaction model produced:

Validation ROC-AUC = 0.66627
Validation PR-AUC = 0.06377

An ablation model without explicit feature interactions was also trained. It achieved validation ROC-AUC of about 0.70185 and PR-AUC of about 0.07980 in this experiment.

This shows that explicit DLRM interactions did not improve performance on this particular split and configuration. The ablation result is therefore important rather than being treated as a failure.

## Analysis

The notebook compares the full DLRM and the no-interaction ablation using ROC-AUC, PR-AUC, F1, log loss, training time, and parameter count. It also includes training curves, threshold tuning, a calibration curve, and a confusion matrix.

The full DLRM has about 1.48 million trainable parameters, with most parameters coming from the categorical embedding tables.

## Output

The trained model is saved as `saved_models/dlrm_model.pt`.

Test predictions are saved as `dlrm_test_predictions.csv`.

Additional comparison results are saved as `dlrm_ablation_results.csv` and `task2_task3_comparison.csv`.

## Important limitations

The dataset is highly imbalanced, so ROC-AUC and PR-AUC are more informative than accuracy alone. The model was trained on CPU and only a small training budget was used. More tuning of embedding size, learning rate, regularization, interaction design, and training epochs could be explored if more computation is available.
