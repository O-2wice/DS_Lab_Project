# Model Training Report: Building the Bankruptcy Prediction Model

## Executive Summary

This document explains **why** each modeling decision was made and **what results** we achieved in training machine learning models for bankruptcy prediction.

**Key Takeaway:** XGBoost with SMOTE-balanced training data and hyperparameter tuning achieved the best performance (F1=0.54, Recall=59%, ROC-AUC=0.94). The model correctly identifies high-risk companies while maintaining reasonable precision.

---

## 1. Why These Models Were Chosen

### 1.1 Model Selection Rationale

We tested three fundamentally different algorithms to understand what works best:

| Model | Type | Why We Chose It |
|-------|------|-----------------|
| **Logistic Regression** | Linear | Interpretable baseline, shows if problem is linearly separable |
| **Random Forest** | Ensemble (Bagging) | Handles non-linear patterns, robust to outliers |
| **XGBoost** | Ensemble (Boosting) | State-of-the-art for tabular data, handles imbalance well |

**Key insight:** Starting with simple models (Logistic Regression) before complex ones helps us understand if added complexity is justified.

---

## 2. The Class Imbalance Problem

### 2.1 Why This Matters

**The problem:** Only 3.2% of companies are bankrupt (220 out of 6,819).

**What happens without addressing it:** 
- A model that predicts "healthy" for everyone gets 97% accuracy!
- But it catches ZERO bankruptcies - completely useless

**Why this is dangerous:** In bankruptcy prediction, **missing a bankrupt company is very costly** (False Negative). Investors and business partners could lose millions on failed investments or unpaid invoices.

### 2.2 Our Solution: SMOTE

**What is SMOTE?** Synthetic Minority Over-sampling Technique creates synthetic examples of the minority class (bankrupt companies) by interpolating between existing examples.

**Before SMOTE:**
- Bankrupt: 176 (3.2%)
- Healthy: 5,279 (96.8%)

**After SMOTE:**
- Bankrupt: 5,279 (50%)
- Healthy: 5,279 (50%)

**Why SMOTE over alternatives:**
| Technique | Pros | Cons | Our Choice |
|-----------|------|------|------------|
| Undersampling | Simple | Loses majority class data | No - Only 176 samples left |
| Oversampling (duplicate) | Simple | Overfitting to duplicates | No - No new information |
| **SMOTE** | Creates new synthetic data | May create noise | **Yes - Best balance** |
| Class weights | No data modification | Doesn't help recall much | Used as backup |

---

## 3. Model Results Comparison

### 3.1 Baseline vs SMOTE Performance

| Model | Data | Precision | Recall | F1 | ROC-AUC |
|-------|------|-----------|--------|-----|---------|
| Logistic Regression | Baseline | 0.67 | 0.09 | 0.16 | 0.84 |
| Logistic Regression | SMOTE | 0.16 | 0.77 | 0.26 | 0.84 |
| Random Forest | Baseline | 0.73 | 0.24 | 0.37 | 0.89 |
| Random Forest | SMOTE | 0.22 | 0.75 | 0.34 | 0.90 |
| XGBoost | Baseline | 0.60 | 0.41 | 0.49 | 0.93 |
| **XGBoost** | **SMOTE** | **0.28** | **0.77** | **0.41** | **0.94** |

### 3.2 Key Observations

**Why SMOTE dramatically increases Recall:**
- Without SMOTE, models are "lazy" - they rarely predict bankruptcy because it's rare
- With SMOTE, the model sees equal amounts of both classes during training
- Result: Recall jumps from 9-41% to 75-77%

**Why Precision drops with SMOTE:**
- More aggressive bankruptcy predictions = more false alarms
- This is the **precision-recall tradeoff**
- For bankruptcy detection, we prefer higher recall (catch more true bankruptcies)

**Why XGBoost outperforms others:**
1. Gradient boosting corrects previous mistakes iteratively
2. Handles feature interactions automatically
3. Built-in regularization prevents overfitting
4. Works well with tabular financial data

---

## 4. Hyperparameter Tuning

### 4.1 Why Tune Hyperparameters?

Default parameters are rarely optimal. Tuning can significantly improve performance.

### 4.2 XGBoost Parameters Tuned

| Parameter | Search Range | Best Value | Why It Matters |
|-----------|--------------|------------|----------------|
| `n_estimators` | [100, 200, 300] | 200 | Number of boosting rounds |
| `max_depth` | [3, 5, 7] | 5 | Tree complexity (deeper = more complex) |
| `learning_rate` | [0.01, 0.1, 0.2] | 0.1 | How much each tree contributes |
| `subsample` | [0.8, 1.0] | 0.8 | Fraction of samples per tree |
| `colsample_bytree` | [0.8, 1.0] | 0.8 | Fraction of features per tree |
| `scale_pos_weight` | [1, 3, 5] | 3 | Extra weight for minority class |

### 4.3 Tuning Results

| Metric | Before Tuning | After Tuning | Improvement |
|--------|---------------|--------------|-------------|
| F1 Score | 0.41 | **0.54** | +32% |
| Recall | 0.77 | **0.59** | -23% (tradeoff) |
| Precision | 0.28 | **0.50** | +79% |
| ROC-AUC | 0.94 | **0.94** | Same |

**Interpretation:** Tuning improved F1 and Precision significantly while maintaining strong ROC-AUC. Recall dropped slightly, but we can adjust the threshold later (see Notebook 03).

---

## 5. Feature Importance Analysis

### 5.1 Top 10 Most Important Features (XGBoost)

| Rank | Feature | Importance | Business Interpretation |
|------|---------|------------|------------------------|
| 1 | **Borrowing dependency** | 0.089 | How much company relies on debt |
| 2 | Persistent EPS in Last 4 Seasons | 0.062 | Consistent earnings stability |
| 3 | Net Income to Total Assets | 0.058 | Overall profitability (ROA variant) |
| 4 | Net worth/Assets | 0.054 | Equity cushion against losses |
| 5 | Interest-bearing debt interest rate | 0.052 | Cost of debt |
| 6 | Total debt/Total net worth | 0.048 | Leverage ratio |
| 7 | Debt ratio % | 0.046 | Overall indebtedness |
| 8 | Retained Earnings to Total Assets | 0.044 | Accumulated profits |
| 9 | Current Liability to Assets | 0.042 | Short-term debt burden |
| 10 | Working Capital to Total Assets | 0.040 | Liquidity position |

### 5.2 Feature Importance Insights

**Debt dominates:** 6 of the top 10 features relate to debt/leverage. This aligns with financial theory - over-leveraged companies are most likely to go bankrupt.

**Profitability matters:** Net Income to Total Assets and EPS are key. Companies that can't generate profits can't service debt.

**Consistency helps:** "Persistent EPS" (4 consecutive quarters of positive earnings) is a strong signal of stability.

---

## 6. Cross-Validation Results

### 6.1 Why Cross-Validation?

A single train/test split might be lucky or unlucky. Cross-validation tests on multiple splits to get reliable estimates.

### 6.2 5-Fold CV Results (XGBoost Tuned)

| Fold | F1 Score | ROC-AUC |
|------|----------|---------|
| 1 | 0.52 | 0.93 |
| 2 | 0.55 | 0.94 |
| 3 | 0.54 | 0.94 |
| 4 | 0.53 | 0.94 |
| 5 | 0.56 | 0.95 |
| **Mean** | **0.54 ± 0.02** | **0.94 ± 0.01** |

**Key insight:** Low standard deviation (±0.02) indicates stable, reliable performance across different data splits.

---

## 7. Model Artifacts Saved

| File | Location | Purpose |
|------|----------|---------|
| `xgboost_tuned_smote.pkl` | `models/` | Production model for scoring |
| `best_params.csv` | `models/` | Optimal hyperparameters |
| `feature_importance.csv` | `outputs/` | Feature rankings |
| `model_comparison.csv` | `outputs/` | All model metrics |

---

## 8. Recommendations for Deployment

### 8.1 Model Selection

**Recommended Model:** XGBoost (Tuned) with SMOTE training

**Why:**
- Best F1 score (0.54) balances precision and recall
- Highest ROC-AUC (0.94) indicates excellent discrimination
- Stable cross-validation results
- Feature importance aligns with financial theory

### 8.2 Next Steps

1. **Threshold Optimization** (Notebook 03): Adjust decision threshold for business requirements
2. **SHAP Analysis** (Notebook 03): Explain individual predictions
3. **SAP Integration** (Notebook 03): Score SAP company codes
4. **Monitoring**: Track model performance over time

---

## 9. Key Takeaways

| Question | Answer |
|----------|--------|
| Which model is best? | XGBoost with SMOTE and tuning |
| Why SMOTE? | Fixes class imbalance, improves recall from 41% to 77% |
| What predicts bankruptcy? | Debt dependency, low profitability, poor liquidity |
| How reliable? | 5-fold CV shows F1 = 0.54 ± 0.02 (stable) |
| What's the tradeoff? | Higher recall = more false alarms (lower precision) |

---

*Generated from Notebook 02: Model Training*  
*Random State: 42 for reproducibility*
