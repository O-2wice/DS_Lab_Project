# Integrated Financial and Operational Risk Forecasting in SAP S/4HANA

**Using Machine Learning and Transfer Learning from External Bankruptcy Data**

## Overview

This project develops an intelligent financial and operational risk-forecasting prototype by combining:

1. **SAP S/4HANA Global Bike Dataset** (Primary - Unlabeled)
   - BKPF: Accounting Document Headers
   - BSEG: Accounting Document Line Items
   - BSID: Open Customer Items (Receivables)
   - BSAK: Cleared Vendor Items (Payables)
   - VBAK: Sales Order Headers
   - VBAP: Sales Order Items

2. **Kaggle Company Bankruptcy Prediction Dataset** (Secondary - Labeled)
   - 96 Financial Ratios
   - Binary Target: Bankrupt (0/1)

## Methodology

- Train ML models (Logistic Regression, GLM, Random Forest, XGBoost) on Kaggle data
- Engineer comparable features from SAP transactional data
- Apply trained models to generate risk scores for SAP company codes (DE00, US00)
- Visualize results in SAP Analytics Cloud (SAC)

## Project Structure

```
DS_Lab_Project/
├── data/                    # Raw and processed data (gitignored)
│   ├── sap/                 # SAP GBI exports
│   └── kaggle/              # Kaggle bankruptcy dataset
├── notebooks/               # Jupyter notebooks for exploration
├── src/                     # Source code
│   ├── data_loader.py       # Data ingestion utilities
│   ├── feature_engineering.py
│   ├── models.py            # ML model training
│   └── risk_scoring.py      # Apply models to SAP data
├── models/                  # Saved trained models
├── outputs/                 # Risk scores for SAC upload
├── docs/                    # Documentation
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run in Google Colab

You can open the notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/O-2wice/DS_Lab_Project/blob/main/notebooks/01_EDA_and_Preprocessing.ipynb)

## License

Academic Use Only
