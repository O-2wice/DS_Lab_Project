# Experiment Results Summary

## Project: Bankruptcy Prediction for SAP S/4HANA

This document summarizes the complete experimental pipeline and key results.

---

## 1. Experiment Overview

| Aspect | Details |
|--------|---------|
| **Objective** | Build a bankruptcy prediction model using Kaggle data, apply to SAP enterprise data |
| **Approach** | Transfer learning: Train on labeled data → Score unlabeled SAP data |
| **Models Tested** | Logistic Regression, Random Forest, XGBoost |
| **Best Model** | XGBoost with SMOTE + Hyperparameter Tuning |
| **Final Performance** | F1=0.54, Recall=59%, ROC-AUC=0.94 |

---

## 2. Pipeline Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                    BANKRUPTCY PREDICTION PIPELINE                    │
└─────────────────────────────────────────────────────────────────────┘
                                   │
     ┌─────────────────────────────┼─────────────────────────────┐
     │                             │                             │
     ▼                             ▼                             ▼
┌─────────────┐           ┌─────────────────┐           ┌─────────────────┐
│  NOTEBOOK 1 │           │   NOTEBOOK 2    │           │   NOTEBOOK 3    │
│  EDA & Pre- │    -->    │ Model Training  │    -->    │ Interpretation  │
│  processing │           │                 │           │ & Deployment    │
└─────────────┘           └─────────────────┘           └─────────────────┘
     │                             │                             │
     ▼                             ▼                             ▼
• Load Kaggle +            • Baseline models           • SHAP analysis
  SAP data                 • SMOTE balancing           • Threshold tuning
• Class imbalance          • Cross-validation          • SAP scoring demo
  analysis (3%)            • Hyperparameter            • Risk categories
• Correlation study          tuning (XGBoost)          • Export configs
• Outlier treatment        • Feature importance
• Train/test split         • Model comparison
• Feature scaling          • Save best model
```

---

## 3. Key Findings

### 3.1 Data Insights (Notebook 01)

| Finding | Value | Implication |
|---------|-------|-------------|
| Class imbalance | 3% bankrupt | SMOTE required |
| Missing values | 0 | Clean dataset |
| Multicollinear pairs | 19 pairs (r > 0.8) | Some redundancy |
| PCA components for 90% variance | 45 of 95 | High dimensionality |
| Top predictor | Net Income to Total Assets (r=-0.34) | Profitability key |

### 3.2 Model Performance (Notebook 02)

| Model | SMOTE | F1 | Recall | Precision | ROC-AUC |
|-------|-------|-----|--------|-----------|---------|
| Logistic Regression | No | 0.16 | 0.09 | 0.67 | 0.84 |
| Logistic Regression | Yes | 0.26 | 0.77 | 0.16 | 0.84 |
| Random Forest | No | 0.37 | 0.24 | 0.73 | 0.89 |
| Random Forest | Yes | 0.34 | 0.75 | 0.22 | 0.90 |
| XGBoost | No | 0.49 | 0.41 | 0.60 | 0.93 |
| XGBoost | Yes | 0.41 | 0.77 | 0.28 | 0.94 |
| **XGBoost (Tuned)** | **Yes** | **0.54** | **0.59** | **0.50** | **0.94** |

**Winner:** XGBoost with SMOTE and tuning

### 3.3 Feature Importance (Notebooks 02 & 03)

| Rank | Feature | Why It Matters |
|------|---------|----------------|
| 1 | **Borrowing dependency** | Companies reliant on debt are vulnerable |
| 2 | Persistent EPS (4 seasons) | Consistent earnings = stability |
| 3 | Net Income to Total Assets | Core profitability measure |
| 4 | Interest-bearing debt rate | High borrowing costs hurt |
| 5 | Net worth/Assets | Equity cushion against losses |

### 3.4 Threshold Analysis (Notebook 03)

| Threshold | Recall | Precision | F1 | Recommendation |
|-----------|--------|-----------|-----|----------------|
| 0.30 | 82% | 21% | 0.34 | Conservative (catch all) |
| **0.45** | **61%** | **47%** | **0.55** | **Balanced (recommended)** |
| 0.60 | 45% | 58% | 0.51 | Aggressive (few false alarms) |

---

## 4. Reproducibility

### 4.1 Random State

All notebooks use `RANDOM_STATE = 42` for:
- Train/test splits
- SMOTE sampling
- Model initialization
- Cross-validation folds

### 4.2 Data Versions

| Dataset | Rows | Columns | Location |
|---------|------|---------|----------|
| Kaggle Bankruptcy | 6,819 | 96 | `data/kaggle/` |
| SAP BKPF | ~15,000 | 15 | `data/sap/` |
| SAP BSEG | ~50,000 | 20 | `data/sap/` |

### 4.3 Package Versions

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
imbalanced-learn>=0.11
shap>=0.42
matplotlib>=3.7
seaborn>=0.12
```

---

## 5. Artifacts Generated

### 5.1 Processed Data (`data/processed/`)

| File | Description |
|------|-------------|
| `X_train.csv` | Training features (5,455 rows) |
| `X_test.csv` | Test features (1,364 rows) |
| `y_train.csv` | Training labels |
| `y_test.csv` | Test labels |
| `X_train_smote.csv` | SMOTE-balanced training (10,558 rows) |
| `y_train_smote.csv` | SMOTE labels |
| `feature_names.csv` | All 95 feature names |

### 5.2 Models (`models/`)

| File | Description |
|------|-------------|
| `xgboost_tuned_smote.pkl` | Production model (pickle) |
| `best_params.csv` | Optimal hyperparameters |
| `threshold_config.csv` | Threshold = 0.45 |

### 5.3 Outputs (`outputs/`)

| File | Description |
|------|-------------|
| `feature_importance.csv` | XGBoost feature rankings |
| `shap_feature_importance.csv` | SHAP-based rankings |
| `model_comparison.csv` | All model metrics |
| `threshold_analysis.csv` | Precision/Recall at each threshold |
| `risk_scores_demo.csv` | Sample predictions |

---

## 6. Documentation (`docs/`)

| Document | Purpose |
|----------|---------|
| `DATA_README.md` | Dataset descriptions and schemas |
| `PREPROCESSING_README.md` | EDA decisions and rationale |
| `MODEL_TRAINING_README.md` | Model selection and tuning |
| `INTERPRETATION_DEPLOYMENT_README.md` | SHAP, thresholds, SAP integration |
| `RESULTS_README.md` | This summary document |

---

## 7. How to Reproduce

### Step 1: Environment Setup
```bash
cd DS_Lab_Project
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

### Step 2: Run Notebooks in Order
1. `01_EDA_and_Preprocessing.ipynb` - Generates processed data
2. `02_Model_Training.ipynb` - Trains and saves best model
3. `03_Model_Interpretation_and_Deployment.ipynb` - SHAP + SAP scoring

### Step 3: Use the Model
```python
import pickle
import pandas as pd

# Load model
with open('models/xgboost_tuned_smote.pkl', 'rb') as f:
    model = pickle.load(f)

# Load threshold
THRESHOLD = 0.45

# Score new data
X_new = pd.read_csv('your_data.csv')
probabilities = model.predict_proba(X_new)[:, 1]
predictions = (probabilities >= THRESHOLD).astype(int)
```

---

## 8. Conclusions

### 8.1 What Worked

- **SMOTE** dramatically improved recall (9% to 77%)  
- **XGBoost** outperformed linear and random forest models  
- **Hyperparameter tuning** improved F1 by 32%  
- **SHAP analysis** confirmed debt as primary bankruptcy driver  
- **Threshold optimization** balanced precision/recall for business needs  

### 8.2 Limitations

- Kaggle data is from Taiwan (1999-2009) - may not generalize globally  
- SAP demo uses proxy features, not full financial ratios  
- Static threshold doesn't adapt to changing economic conditions  

### 8.3 Future Work

- Integrate with SAP Analytics Cloud for dashboards  
- Implement real-time scoring via SAP ABAP/CDS  
- Train on industry-specific subsets  
- Add temporal features (trends over time)  

---

## 9. Contact

**Project:** DS_Lab_Project  
**Repository:** https://github.com/O-2wice/DS_Lab_Project  
**Branch:** main  

---

*Last Updated: December 2025*  
*Random State: 42*
