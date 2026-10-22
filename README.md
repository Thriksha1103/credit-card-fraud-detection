# Credit Card Fraud Detection

Fraud detection on a highly imbalanced credit card dataset. The project has two parts: (A) a logistic regression and the imbalance-handling techniques implemented **from scratch** with NumPy, and (B) a scikit-learn benchmark with Logistic Regression and Random Forest.

## Dataset

[Kaggle: Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transactions, 31 columns, no missing values
- 492 frauds (0.173%), 284,315 normal transactions
- `V1`–`V28` are PCA components; `Time` and `Amount` are the original features
- Target: `Class` (1 = fraud, 0 = normal)

The dataset is not included in this repository. Download `creditcard.csv` and place it in Google Drive (default path: `/content/drive/MyDrive/creditcard.csv`).

## Project files

| File | Description |
|---|---|
| `credit_card_fraud_detection.ipynb` | Full pipeline with outputs (Google Colab) |
| `requirements.txt` | Python dependencies |

## Approach

### Preprocessing
- **Stratified 80/20 split** (seed 42): 227,845 train rows (393 fraud) and 56,962 test rows (99 fraud). The fraud ratio is the same in both sets (0.1725% vs 0.1738%).
- **Standardisation** (z-score) of `Amount` and `Time`. Mean and standard deviation are computed on the **training set only** and applied to the test set, to avoid data leakage.

### Part A: from scratch (NumPy)
Logistic regression trained with batch gradient descent (learning rate 0.3, 500 epochs, bias term, sigmoid with clipping). Four variants are compared:

| Variant | How imbalance is handled |
|---|---|
| Normal | No handling (baseline) |
| Undersample | Keep all 393 frauds + 3,930 random normal rows (10:1), 4,323 rows in total |
| SMOTE-style | Add 20,000 synthetic frauds by linear interpolation between two randomly chosen training frauds (simplified SMOTE, without k-nearest neighbours) |
| Class weight | Balanced weights: 0.50 for normal, 289.88 for fraud |

Precision, recall, F1 and the confusion matrix are computed by hand. Accuracy is printed only to show why it is misleading.

### Part B: scikit-learn benchmark
Logistic Regression (class weight, SMOTE, random undersampling via `imbalanced-learn`) and Random Forest (balanced subsample class weights). The training set is split again into train (75%) and validation (25%). The decision threshold is chosen on the **validation set** (max F1) and the results are reported on the untouched test set.

## Results

Test set: 56,962 transactions, 99 frauds.

### Part A: at the default threshold (0.5)

| Variant | TP | FP | FN | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Normal | 65 | 7 | 34 | 0.9993 | 0.903 | 0.657 | 0.760 |
| Undersample | 90 | 106 | 9 | 0.9980 | 0.459 | 0.909 | 0.610 |
| SMOTE-style | 88 | 33 | 11 | 0.9992 | 0.727 | 0.889 | 0.800 |
| Class weight | 96 | 1,351 | 3 | 0.9762 | 0.066 | 0.970 | 0.124 |

### Part A: at each variant's best threshold

The threshold was searched over a fixed grid (0.1 to 0.99) **using the test set**, so these numbers are optimistic.

| Variant | Best threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| Normal | 0.2 | 0.904 | 0.859 | 0.881 |
| Undersample | 0.9 | 0.761 | 0.869 | 0.811 |
| **SMOTE-style** | **0.99** | 0.896 | 0.869 | **0.882** |
| Class weight | 0.99 | 0.707 | 0.879 | 0.784 |

The saved final model is the SMOTE-style logistic regression at threshold 0.99: TP 86, FP 10, FN 13, TN 56,853.

### Part B: scikit-learn (threshold selected on validation)

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| **RF + class_weight** | 0.65 | 0.964 | 0.808 | **0.879** | 0.9737 | **0.9156** | 80 | 3 | 19 |
| LR + class_weight | ~1.0 | 0.894 | 0.848 | 0.870 | 0.9869 | 0.8010 | 84 | 10 | 15 |
| LR + SMOTE | ~1.0 | 0.894 | 0.848 | 0.870 | 0.9861 | 0.8093 | 84 | 10 | 15 |
| LR + Undersampling | ~1.0 | 0.734 | 0.697 | 0.715 | 0.9916 | 0.7451 | 69 | 25 | 30 |

## Key findings

- **Accuracy is meaningless here.** Predicting "normal" for every transaction already gives about 99.83% accuracy, so all models look excellent on accuracy. Precision, recall, F1 and PR-AUC are used instead.
- **The threshold matters a lot.** At 0.5, F1 ranges from 0.124 to 0.800. The class-weighted model flagged 1,447 transactions as fraud at 0.5 (precision 0.066) because heavy weights push probabilities up.
- **Resampling gave no clear gain over the baseline.** After threshold tuning, plain logistic regression (F1 0.881) and the SMOTE-style version (F1 0.882) are practically identical. In Part B, SMOTE and class weighting give identical confusion matrices.
- **Undersampling was consistently worse** (F1 0.811 in Part A, 0.715 in Part B), likely because it discards most normal transactions.
- **ROC-AUC can mislead.** In Part B, undersampling has the highest ROC-AUC (0.9916) but the lowest PR-AUC and F1.
- **Random Forest** achieved the best PR-AUC (0.916) and the fewest false alarms (3), while logistic regression caught slightly more frauds (84 vs 80).
- **The top results are within noise.** With only 99 test frauds, a single extra false positive changes F1 by about 0.005, so differences between 0.870 and 0.882 should not be over-interpreted.

## Limitations

- Part A selected thresholds on the test set, so its best-threshold numbers are optimistic. Part B avoids this by using a validation split.
- Single train/test split with 99 test frauds. Repeated stratified cross-validation would give a more reliable ranking.
- The from-scratch logistic regression has no regularisation and uses a fixed learning rate and epoch count.
- The SMOTE-style oversampler is simplified (random pair interpolation instead of k-nearest neighbours).
- Hyperparameters were not extensively tuned.

## How to run

1. Download `creditcard.csv` from Kaggle and upload it
