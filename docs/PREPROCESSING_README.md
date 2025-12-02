# Preprocessing Documentation

## Overview
This document describes the preprocessing steps applied to both datasets.

---

## 1. Kaggle Bankruptcy Dataset

### Loading
```python
df_kaggle = pd.read_csv('data/kaggle/kaggle_company_bankruptcy.csv')
```

### Preprocessing Steps:
1. **No missing values** - Dataset is clean
2. **Feature scaling** - StandardScaler (mean=0, std=1)
3. **Class imbalance** - 97% healthy, 3% bankrupt (needs handling in modeling)

### Output:
- `data/processed/kaggle_scaled.csv` - Scaled features + target
- `data/processed/top_features.csv` - Top 20 predictive features

---

## 2. SAP GBI Dataset

### Loading
```python
# Use skiprows=3 for SE16N header, latin-1 for German characters
df = pd.read_csv('data/sap/TABLE_ALL.txt', sep='\t', encoding='latin-1', skiprows=3, low_memory=False)
```

### Tables:
| Table | Rows | Description |
|-------|------|-------------|
| BKPF | 38,179 | Accounting document headers |
| BSEG | 90,476 | Accounting line items |
| BSID | 3 | Open AR (customer receivables) |
| BSAK | 2 | Cleared AP (vendor payments) |
| VBAK | 109 | Sales order headers |
| VBAP | 247 | Sales order items |

### Currency Cleaning:
```python
# German format: 1.234,56 -> 1234.56
def clean_currency(series):
    return pd.to_numeric(
        series.str.replace('.', '', regex=False)
              .str.replace(',', '.', regex=False), 
        errors='coerce')
```

---

## 3. Feature Mapping: SAP -> Kaggle

| Kaggle Feature | SAP Source | Calculation |
|----------------|------------|-------------|
| Debt ratio % | BSEG | Liabilities / Total Assets |
| Current Liability to Assets | BSEG | Current Liab / Assets |
| Net Income to Total Assets | BSEG | Net Income / Assets |
| Total Asset Turnover | VBAK, BSEG | Revenue / Avg Assets |
| Accounts Receivable Turnover | VBAK, BSID | Revenue / Avg AR |
| Working Capital to Total Assets | BSID, BSAK | (AR - AP) / Assets |

---

## 4. Data Quality Notes

### Kaggle:
-  No missing values
-  All numeric features
-  Class imbalance (3% positive)

### SAP:
-  ~40-60% missing values in optional fields (expected)
-  Key fields (amounts, dates, IDs) are populated
-  German number format requires conversion

---

## 5. Files Generated

```
data/processed/
├── kaggle_scaled.csv      # Scaled Kaggle features (StandardScaler)
└── top_features.csv       # Top 20 features by correlation
```

---

*Generated: December 2025*
