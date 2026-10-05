# Task B - Neural CTR Prediction

This task builds a neural recommendation model for click-through-rate (CTR) prediction. The data contains numerical and categorical features, with the target indicating whether a click occurred.

## Approach

The data was split into training, validation, and test sets using a stratified split because the positive class is highly imbalanced.

Numerical features were imputed using training-set statistics and standardized. Categorical features were converted to integer IDs and represented using learned embeddings. Unknown/missing categorical values were handled through a reserved encoding.

A neural CTR model was then built using categorical embeddings followed by an MLP. Three architectures were tried: a small model, a medium model, and a more regularized model. AdamW, weighted binary cross-entropy, dropout, validation monitoring, and early stopping were used.

The architecture with the best validation ROC-AUC was selected. A separate validation threshold search was then performed to choose a threshold that maximized F1 instead of assuming 0.5.

## Why this approach

CTR data contains many high-cardinality categorical variables, so embeddings are more suitable than one-hot encoding for a neural model. The MLP can then learn nonlinear relationships between the numerical features and the embedded categorical features.

Weighted BCE was used because the click class is rare. ROC-AUC and PR-AUC were emphasized because accuracy alone can be misleading for an imbalanced CTR problem.

## Additional experiment

A DCN (Deep & Cross Network) was also implemented. It combines explicit feature-crossing layers with a deep neural network and was compared with the best vanilla neural model.

The DCN reached validation ROC-AUC of about 0.6780, while the best vanilla architecture reached about 0.6870 on validation. Therefore, the vanilla model remained the selected model in this experiment.

## Evaluation

The final model was evaluated using ROC-AUC, PR-AUC, accuracy, precision, recall, F1, log loss, and Brier score. ROC and Precision-Recall curves, a confusion matrix, and a calibration plot were also generated.

The final test predictions were saved as `ctr_predictions.csv`, and the trained model was saved for reuse.

## Important limitations

The dataset is strongly imbalanced, so threshold-dependent metrics can change significantly with the chosen threshold. The model was also trained on CPU, so the architecture and training budget were kept reasonably small.

Further improvements could include more systematic hyperparameter tuning, better calibration, or trying additional CTR architectures.
