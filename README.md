# Credit Card Fraud Detection

Machine learning pipeline to detect fraudulent credit card transactions on a highly imbalanced dataset, comparing class weighting, SMOTE and undersampling with Logistic Regression and Random Forest.

## Dataset

[Kaggle: Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 transactions made by European cardholders over two days
- 492 frauds (about 0.17%), so the classes are extremely imbalanced
- Features `V1`–`V28` are PCA components; `Time` and `Amount` are the only original features
- Target: `Class` (1 = fraud, 0 = normal)

The dataset is not included in this repository. Download `creditcard.csv` from Kaggle and place it in your Google Drive (or update `DATA_PATH` in the notebook).

## Approach

1. **Setup:** Google Colab with Google Drive mounted
2. **Exploration:** class distribution, missing values, duplicates
3. **Split:** stratified 80/20 train/test split, so the fraud ratio is the same in both sets
4. **Scaling:** `RobustScaler` on `Amount` and `Time`, fitted on the training data only to avoid data leakage
5. **Imbalance handling:** class weighting, SMOTE, random undersampling (resampling applied to training data only)
6. **Models:** Logistic Regression and Random Forest
7. **Threshold selection:** the decision threshold that maximises F1 is chosen on a separate validation split, never on the test set
8. **Evaluation:** Precision, Recall, F1, ROC-AUC and PR-AUC on the untouched test set
9. **Export:** model, scaler, threshold and feature order saved together with `joblib`

### Why not accuracy?

About 99.8% of transactions are normal, so a model that predicts "normal" every time would score roughly 99.8% accuracy while catching zero fraud. Precision, recall, F1 and PR-AUC are used instead.

## Results

Test set: 99 fraud cases. Each model uses its own best-F1 threshold, selected on the validation split.

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| **RF + class_weight** | 0.65 | 0.964 | 0.808 | **0.879** | 0.9737 | **0.9156** | 80 | 3 | 19 |
| LR + class_weight | ~1.0 | 0.894 | 0.848 | 0.870 | 0.9869 | 0.8010 | 84 | 10 | 15 |
| LR + SMOTE | ~1.0 | 0.894 | 0.848 | 0.870 | 0.9861 | 0.8093 | 84 | 10 | 15 |
| LR + Undersampling | ~1.0 | 0.734 | 0.697 | 0.715 | 0.9916 | 0.7451 | 69 | 25 | 30 |

### Key findings

- **Random Forest** gave the best overall balance (highest F1 and PR-AUC) with only 3 false alarms.
- **SMOTE gave no measurable benefit** over simple class weighting for Logistic Regression: both produced identical confusion matrices.
- **Undersampling performed worst** (F1 0.715), likely because it throws away most of the normal transactions.
- **ROC-AUC can mislead on imbalanced data:** undersampling had the highest ROC-AUC (0.9916) but the lowest PR-AUC and F1.
- **Business trade-off:** Logistic Regression catches slightly more fraud (84 vs 80), while Random Forest raises far fewer false alarms (3 vs 10). The better choice depends on the relative cost of a missed fraud versus a manual review.
- Logistic Regression thresholds near 1.0 are expected with class weighting, because the predicted probabilities are shifted upward and are not calibrated.

### Limitations

- Results come from a single train/test split containing only 99 fraud cases. The F1 gap between Random Forest (0.879) and Logistic Regression (0.870) is small and may fall within random variation.
- Repeated stratified cross-validation would give a more reliable model ranking.
- Hyperparameters were not extensively tuned.

## How to run

1. Download `creditcard.csv` from Kaggle and upload it to Google Drive.
2. Open `credit_card_fraud_detection.ipynb` in Google Colab.
3. Update `DATA_PATH` to your file location.
4. Run all cells in order.

## Using the saved model

```python
import joblib

art = joblib.load("fraud_model.joblib")
new = new_df[art["feature_order"]].copy()
new[art["scaled_cols"]] = art["scaler"].transform(new[art["scaled_cols"]])
fraud_flag = (art["model"].predict_proba(new)[:, 1] >= art["threshold"]).astype(int)
```

## Tech stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn, matplotlib, joblib, Google Colab

## Future work

- Repeated stratified k-fold cross-validation
- Gradient boosting models (XGBoost / LightGBM)
- Probability calibration
- Cost-sensitive threshold selection using actual fraud and review costs
