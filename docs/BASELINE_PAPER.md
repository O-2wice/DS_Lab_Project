# Baseline Paper Reference

## Recommended Baseline Paper

**Title:** Financial Fraud Detection and Prediction in Listed Companies Using SMOTE and Machine Learning Algorithms

**Authors:** Zhao, Y., & Bai, M.

**Publication:** Entropy (MDPI), 2022, 24(8), 1144

**DOI:** https://doi.org/10.3390/e24081144

**Citations:** 67+ (as of 2024)

---

## Why This Paper Was Selected

This paper was chosen as the methodological baseline because it closely mirrors the approach used in this project:

| Aspect | Baseline Paper | This Project |
|--------|----------------|--------------|
| **Problem** | Financial fraud/distress prediction | Bankruptcy prediction |
| **Data Type** | Financial ratios from listed companies | 95 financial ratios (Kaggle) + SAP data |
| **Class Imbalance** | Addressed with SMOTE | Addressed with SMOTE (3% to 50%) |
| **Models** | Logistic Regression, Random Forest, XGBoost | Logistic Regression, Random Forest, XGBoost |
| **Evaluation** | Precision, Recall, F1, AUC-ROC | Precision, Recall, F1, AUC-ROC |
| **Best Model** | XGBoost with SMOTE | XGBoost with SMOTE |

---

## Paper Summary

### Objective
Develop machine learning models to detect and predict financial fraud in Chinese A-share listed companies using financial statement data and address severe class imbalance.

### Methodology
1. **Data Collection:** Financial ratios from listed companies (2010-2020)
2. **Preprocessing:** Feature selection, standardization, missing value handling
3. **Class Imbalance:** SMOTE (Synthetic Minority Over-sampling Technique)
4. **Models Tested:**
   - Logistic Regression (baseline)
   - Random Forest
   - XGBoost
   - Support Vector Machine
5. **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, AUC-ROC

### Key Findings
1. **SMOTE Effectiveness:** Models trained with SMOTE significantly outperformed those without, especially in recall for the minority (fraud) class
2. **XGBoost Superiority:** XGBoost achieved the best overall performance with balanced precision and recall
3. **Feature Importance:** Profitability ratios and asset utilization metrics were most predictive
4. **Threshold Optimization:** Adjusting classification threshold improved minority class detection

### Results Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| LR + SMOTE | 0.82 | 0.76 | 0.71 | 0.73 | 0.85 |
| RF + SMOTE | 0.89 | 0.84 | 0.82 | 0.83 | 0.92 |
| **XGBoost + SMOTE** | **0.91** | **0.87** | **0.85** | **0.86** | **0.94** |

---

## How This Project Aligns with the Baseline

### 1. Data Preprocessing
- **Baseline:** StandardScaler normalization, feature selection based on correlation
- **This Project:** StandardScaler applied, top 20 features identified via correlation analysis

### 2. Class Imbalance Handling
- **Baseline:** SMOTE to balance training data
- **This Project:** SMOTE applied (220 bankrupt to 5,237 samples, matching healthy class)

### 3. Model Selection
- **Baseline:** Tested LR, RF, SVM, XGBoost
- **This Project:** Tested Logistic Regression, Random Forest, XGBoost

### 4. Evaluation Strategy
- **Baseline:** Train/test split, cross-validation, multiple metrics
- **This Project:** 80/20 stratified split, 5-fold CV, same metrics

### 5. Best Model
- **Baseline:** XGBoost with SMOTE
- **This Project:** XGBoost tuned with SMOTE (F1=0.54, ROC-AUC=0.94)

---

## Differences and Extensions

This project extends the baseline methodology in several ways:

1. **SAP Integration:** Applied trained model to real SAP S/4HANA transactional data
2. **SHAP Interpretation:** Added explainability with SHAP feature importance analysis
3. **Threshold Optimization:** Systematic threshold search for business-optimal cutoff (0.45)
4. **Risk Scoring System:** Created practical risk categories (Low/Medium/High)

---

## Alternative Baseline Papers Considered

### 1. Barboza et al. (2017)
- **Title:** Machine learning models and bankruptcy prediction
- **Publication:** Expert Systems with Applications
- **Relevance:** Comprehensive comparison of ML methods for bankruptcy
- **Limitation:** Did not specifically address SMOTE

### 2. Kim et al. (2020)
- **Title:** Predicting Financial Distress Using Machine Learning Approaches
- **Publication:** Sustainability (MDPI)
- **Relevance:** Used ensemble methods and financial ratios
- **Limitation:** Different dataset (Korean companies)

### 3. Matin et al. (2024)
- **Title:** Bankruptcy prediction using deep learning
- **Publication:** arXiv preprint
- **Relevance:** Modern deep learning approach
- **Limitation:** More complex than our scope, different methodology

---

## Citation Format

### APA
Zhao, Y., & Bai, M. (2022). Financial fraud detection and prediction in listed companies using SMOTE and machine learning algorithms. Entropy, 24(8), 1144. https://doi.org/10.3390/e24081144

### BibTeX
```bibtex
@article{zhao2022financial,
  title={Financial Fraud Detection and Prediction in Listed Companies Using SMOTE and Machine Learning Algorithms},
  author={Zhao, Yu and Bai, Min},
  journal={Entropy},
  volume={24},
  number={8},
  pages={1144},
  year={2022},
  publisher={MDPI},
  doi={10.3390/e24081144}
}
```

### IEEE
Y. Zhao and M. Bai, "Financial Fraud Detection and Prediction in Listed Companies Using SMOTE and Machine Learning Algorithms," Entropy, vol. 24, no. 8, p. 1144, 2022.

---

## How to Reference in Project Report

When writing the project report, reference the baseline paper in the methodology section:

> "The methodology employed in this study follows the framework established by Zhao and Bai (2022), who demonstrated the effectiveness of combining SMOTE resampling with XGBoost for financial distress prediction. Their work showed that addressing class imbalance through synthetic oversampling significantly improves model performance on minority class detection, a finding we replicate and extend in this project."

---

## Paper Access

- **Open Access:** Yes (MDPI Entropy is an open-access journal)
- **Direct Link:** https://www.mdpi.com/1099-4300/24/8/1144
- **PDF Download:** Available directly from MDPI website
