# Integrated Financial and Operational Risk Forecasting in SAP S/4HANA

**Using Machine Learning and Transfer Learning from External Bankruptcy Data**

---

## ⚠️ Important: Project Scope & Limitations

**This is a proof-of-concept prototype demonstrating data science methodology.** 

**What is REAL:**
- ✅ Model trained on 6,819 real companies with actual bankruptcy outcomes
- ✅ Achieved 94% ROC-AUC using XGBoost + SMOTE on Kaggle test data
- ✅ Full ML pipeline: preprocessing, model comparison, hyperparameter tuning, SHAP analysis
- ✅ SAP S/4HANA data exported from real ERP system (GBI educational dataset)

**What is DEMONSTRATIVE:**
- ⚠️ SAP company risk scores (DE00, US00) are simulated examples, not actual predictions
- ⚠️ Financial ratios NOT calculated from SAP transactional data (would require G/L account mapping)
- ⚠️ No production deployment or real-time SAP integration implemented
- ⚠️ SAP GBI has no bankruptcy labels, so predictions cannot be validated

**Value:** This project demonstrates a complete data science workflow and shows HOW transfer learning could be applied to ERP systems, even though full implementation requires additional feature engineering work.

---

## Project Overview

This project demonstrates **financial and operational risk forecasting** for SAP S/4HANA companies by applying transfer learning from external bankruptcy data. Rather than predicting bankruptcy per se, the goal is to generate **continuous risk scores** that inform operational decisions such as credit limits, payment terms, supplier evaluation, and customer risk assessment.

The methodology uses machine learning models trained on labeled financial distress data (Kaggle Taiwan bankruptcy dataset) and transfers the learned patterns to SAP S/4HANA transactional data to produce risk scores for companies like Global Bike Germany (DE00) and Global Bike USA (US00).

> **Project Focus:** This is a **risk scoring and forecasting system**, not a binary bankruptcy classifier. The outputs are continuous risk scores (0-100%) that enable data-driven operational decisions in ERP systems.

### Key Results

| Metric | Value |
|--------|-------|
| **Best Model** | XGBoost (tuned) |
| **ROC-AUC** | 0.94 |
| **F1-Score** | 0.54 |
| **Recall (Financial Distress)** | 59% |
| **Risk Score Range** | 0-100% (continuous) |
| **Application** | SAP company risk scoring |

---

## Datasets

### 1. External Bankruptcy Dataset (Transfer Learning Source)
- **Source:** Taiwan Economic Journal (1999-2009) via Kaggle
- **Size:** 6,819 companies
- **Features:** 95 financial ratios
- **Purpose:** Train ML models on labeled financial distress patterns
- **Class Distribution:** 97% healthy, 3% financial distress

### 2. SAP S/4HANA GBI Data (Primary Application Target)
| Table | Description | Records |
|-------|-------------|---------|
| BKPF | Accounting Document Headers | 38,179 |
| BSEG | Accounting Line Items | 90,476 |
| BSID | Open Customer Items (AR) | 1,000+ |
| BSAK | Cleared Vendor Items (AP) | 1,000+ |
| VBAK | Sales Order Headers | 109 |
| VBAP | Sales Order Items | 247 |

---

## Methodology: Transfer Learning for Risk Forecasting

This project applies **transfer learning** to financial risk assessment by training on external labeled data and applying to SAP companies:

### Phase 1: Model Training (External Data)
1. **Preprocessing:** StandardScaler normalization, outlier capping (IQR)
2. **Class Imbalance Handling:** SMOTE oversampling (3% to 50%)
3. **Model Development:** Logistic Regression, Random Forest, XGBoost
4. **Evaluation:** Stratified 5-fold CV, ROC-AUC, Precision, Recall, F1
5. **Feature Selection:** SHAP-based importance analysis

### Phase 2: Transfer to SAP S/4HANA
6. **Feature Engineering:** Map SAP FI/SD data to financial ratios
7. **Risk Scoring:** Generate continuous risk scores (0-100%)
8. **Operational Integration:** Risk-based decision rules for ERP workflows

Following the framework of [Zhao & Bai (2022)](https://doi.org/10.3390/e24081144) for handling financial distress data.

---

## Project Structure

```
DS_Lab_Project/
├── data/
│   ├── kaggle/              # Kaggle bankruptcy dataset
│   ├── sap/                 # SAP GBI exports (6 tables)
│   └── processed/           # Preprocessed datasets
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb
│   ├── 02_Model_Training.ipynb
│   └── 03_Model_Interpretation_and_Deployment.ipynb
├── models/                  # Saved trained models (.pkl)
├── outputs/                 # Risk scores, visualizations
├── docs/
│   ├── DATA_README.md
│   ├── PREPROCESSING_README.md
│   ├── MODEL_TRAINING_README.md
│   ├── INTERPRETATION_DEPLOYMENT_README.md
│   ├── RESULTS_README.md
│   └── BASELINE_PAPER.md
├── src/                     # Source code modules
├── requirements.txt
└── README.md
```

---

## Notebooks

| Notebook | Description | Key Outputs |
|----------|-------------|-------------|
| **01_EDA_and_Preprocessing** | Data exploration, feature engineering, SMOTE | Processed datasets, correlation analysis |
| **02_Model_Training** | Model comparison, hyperparameter tuning | Best model: XGBoost (ROC-AUC: 0.94) |
| **03_Model_Interpretation_and_Deployment** | SHAP analysis, SAP integration, risk scoring | Risk scores for SAP companies |
| **04_Project_Summary** | Overview and consolidated results | Final visualizations and insights |

---

## Top Predictive Features (SHAP)

1. Net Income to Total Assets
2. ROA(C) before interest and depreciation
3. Persistent EPS in the Last Four Seasons
4. Debt Ratio %
5. Borrowing Dependency
6. Net Worth / Assets
7. Working Capital to Total Assets
8. Current Liability to Assets
9. Total Debt / Total Net Worth
10. Revenue per Person

---

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/O-2wice/DS_Lab_Project.git
cd DS_Lab_Project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Notebooks
Open Jupyter and run notebooks in order:
1. `01_EDA_and_Preprocessing.ipynb`
2. `02_Model_Training.ipynb`
3. `03_Model_Interpretation_and_Deployment.ipynb`

---

## Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
imbalanced-learn>=0.10.0
shap>=0.41.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

---

## Results Summary

### Model Comparison (with SMOTE)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | 0.82 | 0.21 | 0.54 | 0.30 | 0.85 |
| Random Forest | 0.96 | 0.56 | 0.37 | 0.44 | 0.93 |
| **XGBoost (Tuned)** | **0.95** | **0.50** | **0.59** | **0.54** | **0.94** |

### Risk Scoring Framework for SAP Companies

| Risk Level | Score Range | Operational Actions |
|------------|-------------|---------------------|
| **Low Risk** | 0-30% | Standard credit terms, automated processing |
| **Medium Risk** | 30-60% | Enhanced monitoring, credit limit review |
| **High Risk** | 60-100% | Manual approval required, reduced credit limits |

**Application Areas:**
- **Accounts Receivable:** Customer credit limits and payment terms
- **Accounts Payable:** Supplier reliability assessment
- **Sales Planning:** Customer relationship management
- **Financial Planning:** Risk-adjusted revenue forecasting

---

## Documentation

Detailed documentation is available in the `docs/` folder:

- [Data README](docs/DATA_README.md) - Dataset descriptions
- [Preprocessing README](docs/PREPROCESSING_README.md) - Data preparation steps
- [Model Training README](docs/MODEL_TRAINING_README.md) - Model development
- [Interpretation README](docs/INTERPRETATION_DEPLOYMENT_README.md) - SHAP and deployment
- [Results README](docs/RESULTS_README.md) - Final results summary
- [Baseline Paper](docs/BASELINE_PAPER.md) - Methodology reference
- [Future Labels README](docs/FUTURE_LABELS_README.md) - What labels SAP would need for real predictions

---

## References

**Baseline Paper:**
> Zhao, Y., & Bai, M. (2022). Financial Fraud Detection and Prediction in Listed Companies Using SMOTE and Machine Learning Algorithms. *Entropy*, 24(8), 1144. https://doi.org/10.3390/e24081144

**Dataset:**
> Taiwan Economic Journal. Company Bankruptcy Prediction Dataset. Kaggle.

---

## License

This project is for educational purposes as part of a Data Science Lab course.

---

## Author

Data Science Lab Project - December 2025## Run in Google Colab

You can open the notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/O-2wice/DS_Lab_Project/blob/main/notebooks/01_EDA_and_Preprocessing.ipynb)

## License

Academic Use Only
