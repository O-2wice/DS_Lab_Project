# Preprocessing Report: EDA & Data Preparation for Risk Forecasting

## Executive Summary

This document explains **why** each preprocessing decision was made and **what insights** we discovered during exploratory data analysis (EDA) for the integrated financial and operational risk forecasting project.

**Key Takeaway:** The external financial distress dataset is clean but heavily imbalanced (only 3% distressed companies). We applied SMOTE to balance classes for model training, identified 19 redundant feature pairs, and found that 45 principal components capture 90% of the information in 95 features. These preprocessing steps enable effective transfer learning to SAP S/4HANA companies.

---

## 1. Understanding Our Data Sources

### 1.1 Kaggle Bankruptcy Dataset (Training Data)

**What we have:** 6,819 Taiwanese companies with 95 financial ratios each, labeled as bankrupt or healthy.

**Why we're using it:** This is our **labeled training data**. Real bankruptcy labels are rare and expensive to obtain. The Kaggle dataset gives us ground truth to train models that we'll later apply to SAP data.

**Key insight:** Only 220 companies (3.2%) went bankrupt. This extreme imbalance will cause problems if not addressed.

### 1.2 SAP GBI Dataset (Application Data)

**What we have:** 6 tables of transactional data from Global Bike Inc's SAP system - accounting documents, sales orders, receivables, and payables.

**Why we're using it:** This is where we'll **apply** our trained model. SAP has no bankruptcy labels, so we need to:
1. Train on Kaggle (labeled) → 
2. Engineer matching features from SAP → 
3. Score SAP entities for risk

---

## 2. Why Each Preprocessing Step Was Necessary

### 2.1 Feature Scaling (StandardScaler)

**The problem:** Financial ratios have wildly different scales. "Total Assets" might be in millions while "Debt Ratio" is between 0-1.

**Why it matters:** Many ML algorithms (Logistic Regression, SVM, Neural Networks) are sensitive to feature scales. Without scaling, features with larger values would dominate.

**What we did:** Applied StandardScaler to normalize all features to mean=0, std=1.

**Result:** All 95 features now contribute equally to model training.

---

### 2.2 Correlation Analysis

**The problem:** With 95 features, we needed to understand which ones actually predict bankruptcy.

**What we found:**

| Rank | Feature | Correlation | Interpretation |
|------|---------|-------------|----------------|
| 1 | Net Income to Total Assets | -0.34 | Lower profitability → higher bankruptcy risk |
| 2 | ROA (A) | -0.32 | Lower return on assets → higher risk |
| 3 | ROA (B) | -0.31 | Same pattern |
| 4 | ROA (C) | -0.30 | Same pattern |
| 5 | Net worth/Assets | -0.29 | Lower equity ratio → higher risk |
| 6 | Debt ratio % | +0.29 | Higher debt → higher risk |

**Key insight:** All top predictors are **negatively correlated** - meaning bankrupt companies have LOWER profitability, ROA, and equity. The only positive correlation is Debt ratio - bankrupt companies have MORE debt. This matches financial theory!

---

### 2.3 Multicollinearity Check

**The problem:** Some features measure the same thing in different ways, which can confuse models and inflate feature importance.

**What we found:** 19 feature pairs with correlation > 0.8:

| Feature 1 | Feature 2 | Correlation | Issue |
|-----------|-----------|-------------|-------|
| Net worth/Assets | Debt ratio % | -1.00 | **Mathematically identical** (one is 1 minus the other) |
| Net Value Per Share (A) | Net Value Per Share (B) | 0.999 | Nearly identical calculations |
| Net Value Per Share (A) | Net Value Per Share (C) | 1.000 | Exactly the same |
| ROA(B) | ROA(C) | 0.987 | Very similar ROA definitions |

**Recommendation for modeling:** Remove one feature from each highly correlated pair to reduce redundancy and improve model interpretability.

---

### 2.4 Outlier Analysis

**The problem:** Financial data often has extreme values (e.g., a company with 10,000% debt ratio due to accounting quirks).

**What we found:**
- 30 features have >10% outliers (by IQR method)
- 5 features have >20% outliers:
  - Degree of Financial Leverage (DFL): 22% outliers
  - Interest Coverage Ratio: 21% outliers
  - Fixed Assets Turnover: 21% outliers

**Why these have outliers:** Financial ratios can explode when denominators approach zero. For example, Interest Coverage = EBIT / Interest Expense. If interest expense is tiny, the ratio becomes huge.

**What we did:** Created a "capped" version of the data using Winsorization (clipping values to Q1-1.5×IQR and Q3+1.5×IQR). This preserves the relative ordering while removing extreme outliers.

**Saved as:** `X_capped.csv` - can compare model performance with/without outlier treatment.

---

### 2.5 PCA Analysis

**The problem:** 95 features is a lot. Are they all adding unique information, or is there redundancy?

**What we found:**
- **45 components** capture 90% of variance (less than half the features!)
- **53 components** capture 95% of variance
- First 10 components alone explain 48.7%

**Interpretation:** There's significant redundancy in the 95 features. For faster training and potentially better generalization, we could reduce to ~45 dimensions.

**2D Visualization Result:** When we projected onto 2 principal components, bankrupt and healthy companies **overlap significantly**. This tells us:
- Linear methods (Logistic Regression) may struggle
- Non-linear methods (Random Forest, XGBoost) might perform better
- The problem is genuinely difficult - no simple line separates the classes

---

### 2.6 Train/Test Split (Stratified)

**The problem:** We need to hold out data for evaluation, but random splits could accidentally put most bankrupt companies in training or test.

**What we did:** Used stratified sampling to maintain the 97%/3% ratio in both sets:

| Set | Total | Healthy | Bankrupt | Ratio |
|-----|-------|---------|----------|-------|
| Training | 5,455 | 5,279 (96.8%) | 176 (3.2%) | Same as original |
| Test | 1,364 | 1,320 (96.8%) | 44 (3.2%) | Same as original |

**Why stratification matters:** Without it, the test set might have 0 bankrupt companies, making evaluation impossible.

---

### 2.7 SMOTE (Handling Class Imbalance)

**The problem:** With only 3% bankrupt companies, a model might learn to always predict "healthy" and be 97% accurate but useless.

**Why not just accuracy?** We care about catching bankruptcies. Missing a bankruptcy (false negative) could be catastrophic.

**What SMOTE does:** Creates synthetic minority samples by interpolating between existing bankrupt companies.

**Result:**
- Before: 5,279 healthy, 176 bankrupt (1:30 ratio)
- After: 5,279 healthy, 5,279 bankrupt (1:1 ratio)
- Training samples increased from 5,455 to 10,558

**Important:** We only applied SMOTE to training data. Test data remains imbalanced (real-world distribution) to get honest evaluation.

---

## 3. SAP Data Preparation

### 3.1 Loading Challenges

**Problem 1:** SAP exports have 3 header rows from SE16N transaction.  
**Solution:** `skiprows=3`

**Problem 2:** German number format (1.234,56 instead of 1,234.56)  
**Solution:** Custom `clean_currency()` function to swap comma/period

**Problem 3:** German special characters (umlauts)  
**Solution:** `encoding='latin-1'`

### 3.2 Feature Mapping Strategy

The key challenge is creating **comparable features** between Kaggle and SAP. Here's our mapping:

| What We Need | Kaggle Has | SAP Source | How to Calculate |
|--------------|------------|------------|------------------|
| Debt Ratio | `Debt ratio %` | BSEG | Sum liabilities / Sum assets (by G/L account type) |
| Profitability | `Net Income to Total Assets` | BSEG | P&L accounts / Balance sheet totals |
| Liquidity | `Current Ratio` | BSID, BSAK | AR / AP |
| Activity | `Total Asset Turnover` | VBAK, BSEG | Revenue / Assets |

**Current limitation:** SAP GBI is a teaching dataset with limited AR/AP data (only 3 AR items, 2 AP items). Real SAP systems would have thousands.

---

## 4. Summary of Insights

### What We Learned About Bankruptcy Prediction:

1. **Profitability is king** - Top predictors are all profitability ratios (ROA, Net Income/Assets)
2. **Debt matters** - Higher debt ratio correlates with bankruptcy
3. **Many features are redundant** - 45 PCA components capture 90% of information
4. **Classes are not linearly separable** - Non-linear models likely needed
5. **Extreme imbalance** - Must use SMOTE or class weights in modeling

### What We Prepared for Modeling:

| File | Rows | Description | Use Case |
|------|------|-------------|----------|
| `X_train.csv` | 5,455 | Original training features | Baseline training |
| `X_train_smote.csv` | 10,558 | SMOTE-balanced training | Compare with/without balancing |
| `X_test.csv` | 1,364 | Test features | Final evaluation |
| `X_capped.csv` | 6,819 | Outlier-treated features | Compare with/without outliers |
| `top_features.csv` | 20 | Most predictive features | Feature selection |
| `high_correlation_pairs.csv` | 19 | Redundant feature pairs | Feature removal candidates |

---

## 5. Next Steps

In **Notebook 02: Model Training**, we will:

1. **Train baseline models** - Logistic Regression, Random Forest, XGBoost
2. **Compare SMOTE vs no SMOTE** - Does balancing help?
3. **Compare all features vs top 20** - Does feature selection help?
4. **Tune hyperparameters** - Grid search with cross-validation
5. **Evaluate properly** - Focus on Recall (catching bankruptcies) and F1-score
6. **Apply to SAP** - Score Global Bike entities for risk

---

*Last Updated: December 2, 2025*
