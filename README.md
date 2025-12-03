# Bankruptcy Prediction Using Machine Learning

**Integrated Financial Risk Forecasting with SAP S/4HANA Data**

---

## Project Overview

This project develops a machine learning pipeline for predicting corporate bankruptcy using financial ratios. The trained model is then applied to SAP S/4HANA transactional data to generate real-time risk scores for business customers.

### Key Results

| Metric | Value |
|--------|-------|
| **Best Model** | XGBoost (tuned) |
| **ROC-AUC** | 0.94 |
| **F1-Score** | 0.54 |
| **Recall (Bankruptcy)** | 59% |
| **Optimal Threshold** | 0.45 |

---

## Datasets

### 1. Kaggle Bankruptcy Dataset (Training)
- **Source:** Taiwan Economic Journal (1999-2009)
- **Size:** 6,819 companies
- **Features:** 95 financial ratios
- **Target:** Binary (0 = Healthy, 1 = Bankrupt)
- **Class Imbalance:** 97% healthy, 3% bankrupt

### 2. SAP S/4HANA GBI Data (Application)
| Table | Description | Records |
|-------|-------------|---------|
| BKPF | Accounting Document Headers | 38,179 |
| BSEG | Accounting Line Items | 90,476 |
| BSID | Open Customer Items (AR) | 1,000+ |
| BSAK | Cleared Vendor Items (AP) | 1,000+ |
| VBAK | Sales Order Headers | 109 |
| VBAP | Sales Order Items | 247 |

---

## Methodology

Following the framework established by [Zhao & Bai (2022)](https://doi.org/10.3390/e24081144):

1. **Preprocessing:** StandardScaler normalization, outlier capping (IQR)
2. **Class Imbalance:** SMOTE oversampling (3% to 50%)
3. **Models Tested:** Logistic Regression, Random Forest, XGBoost
4. **Evaluation:** Stratified 5-fold CV, ROC-AUC, Precision, Recall, F1
5. **Interpretation:** SHAP feature importance analysis
6. **Deployment:** Risk scoring on SAP customer data

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
| **01_EDA_and_Preprocessing** | Data exploration, scaling, SMOTE | Train/test splits, correlation analysis |
| **02_Model_Training** | Model comparison, hyperparameter tuning | Best model: XGBoost (ROC-AUC: 0.94) |
| **03_Model_Interpretation** | SHAP analysis, threshold optimization, SAP scoring | Feature importance, risk categories |

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

### Risk Categories (SAP Customers)

| Risk Level | Probability Range | Recommended Action |
|------------|-------------------|-------------------|
| Low | < 30% | Standard terms |
| Medium | 30-60% | Enhanced monitoring |
| High | > 60% | Credit review required |

---

## Documentation

Detailed documentation is available in the `docs/` folder:

- [Data README](docs/DATA_README.md) - Dataset descriptions
- [Preprocessing README](docs/PREPROCESSING_README.md) - Data preparation steps
- [Model Training README](docs/MODEL_TRAINING_README.md) - Model development
- [Interpretation README](docs/INTERPRETATION_DEPLOYMENT_README.md) - SHAP and deployment
- [Results README](docs/RESULTS_README.md) - Final results summary
- [Baseline Paper](docs/BASELINE_PAPER.md) - Methodology reference

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
