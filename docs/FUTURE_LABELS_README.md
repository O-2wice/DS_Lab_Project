# Future Labels for Actual Bankruptcy Prediction in SAP

## Overview

This document explains **what labels SAP S/4HANA would need** to enable actual bankruptcy prediction rather than relying on external labeled datasets like Kaggle.

> **Current Project Status:** This project is a proof-of-concept prototype. We train on Kaggle's labeled bankruptcy data and apply the model to SAP GBI data, which has **no bankruptcy labels**. The "risk scores" generated are demonstrations, not validated predictions.

---

## Why SAP GBI Has No Bankruptcy Labels

SAP S/4HANA Global Bike Inc. (GBI) is an **educational ERP system** designed to teach students how SAP works. It contains:

✅ **What GBI Has:**
- Transactional data (orders, invoices, payments)
- Master data (customers, vendors, materials, G/L accounts)
- Financial documents (journal entries, AR/AP)
- Organizational structure (company codes, sales orgs, plants)

❌ **What GBI Does NOT Have:**
- Historical bankruptcy outcomes for business partners
- Credit rating scores from external agencies
- Default/write-off flags on receivables
- Customer/vendor financial distress indicators
- Time-series outcome data (who went bankrupt after date X)

---

## Option 1: Company-Level Bankruptcy Labels (For B2B Credit Risk)

### What You Would Need

If predicting bankruptcy risk for **customers/vendors** (business partners):

| Required Field | Description | Source |
|----------------|-------------|--------|
| `BP_BANKRUPT_FLAG` | Binary: Did this business partner go bankrupt? (1=Yes, 0=No) | External data / Credit bureau |
| `BANKRUPTCY_DATE` | When did bankruptcy occur? | Court filings |
| `OBSERVATION_DATE` | When was the prediction made? | System timestamp |

### Where This Would Live in SAP

In a real SAP S/4HANA system, you could create a custom table or extend the Business Partner (BP) master data:

```
Custom Table: ZBANKRUPTCY_OUTCOMES
---------------------------------
| PARTNER  | BUKRS | OUTCOME | OUTCOME_DATE | OBSERVATION_DATE |
|----------|-------|---------|--------------|------------------|
| 1000     | US00  | 1       | 2023-06-15   | 2022-01-01       |
| 1001     | US00  | 0       | NULL         | 2022-01-01       |
| 1002     | DE00  | 0       | NULL         | 2022-01-01       |
```

### How to Obtain This Data

| Method | Difficulty | Quality | Cost |
|--------|------------|---------|------|
| **Credit Bureau API** (D&B, Experian) | Medium | High | $$$ |
| **Public Court Records** | High | Medium | $ |
| **Internal Write-off History** | Low | High | Free |
| **News/Web Scraping** | High | Low | Free |

### Features You Could Calculate from SAP

With labeled business partners, you'd calculate features like:

```python
# For each customer/vendor, calculate:
features = {
    'avg_days_to_pay': "BSID/BSAD payment date - invoice date",
    'payment_variance': "Std dev of payment timing",
    'order_trend': "Revenue growth over last 4 quarters from VBAK",
    'credit_utilization': "Outstanding AR / Credit Limit",
    'return_rate': "Returns (RE orders) / Total orders",
    'dispute_rate': "Disputed invoices / Total invoices",
    'concentration_risk': "This customer's revenue / Total revenue"
}
```

---

## Option 2: Internal Company Distress Labels (For Your Own Company)

### What You Would Need

If predicting financial distress for **your own company** (like Global Bike):

| Required Field | Description | Source |
|----------------|-------------|--------|
| `DISTRESS_FLAG` | Binary: Was company in financial distress? | Management definition |
| `PERIOD` | Fiscal period (YYYYMM) | SAP fiscal calendar |
| `DISTRESS_TYPE` | Type: Liquidity crisis, solvency issue, covenant breach | Internal classification |

### Example Distress Definitions

| Distress Type | Definition | SAP Data Source |
|---------------|------------|-----------------|
| **Cash Flow Crisis** | Operating cash flow negative for 2+ quarters | BSEG cash account movements |
| **Debt Covenant Breach** | Debt/EBITDA > threshold | BSEG liability accounts, P&L |
| **Working Capital Stress** | Current ratio < 1.0 | BSID, BSAK, BSEG |
| **Revenue Decline** | Revenue down >20% YoY | VBAK/VBAP or P&L accounts |
| **Actual Bankruptcy** | Chapter 11/7 filing | External event |

### Creating Synthetic Labels (Research Approach)

For educational/research purposes, you could create **synthetic distress labels**:

```python
# Example: Create distress label based on financial ratios
def create_distress_label(df):
    """
    Define 'distress' as companies with:
    - Debt ratio > 0.8 OR
    - Current ratio < 1.0 OR
    - Negative net income for 2+ periods
    """
    distress = (
        (df['debt_ratio'] > 0.8) |
        (df['current_ratio'] < 1.0) |
        (df['net_income'] < 0)
    )
    return distress.astype(int)
```

> ⚠️ **Warning:** Synthetic labels are proxies, not actual outcomes. Models trained on synthetic labels predict the proxy definition, not true bankruptcy.

---

## Option 3: Financial Ratio Thresholds + Historical Outcomes

### The Altman Z-Score Approach

Instead of binary bankruptcy labels, use **established distress indicators**:

| Ratio | Formula | Healthy | Gray Zone | Distressed |
|-------|---------|---------|-----------|------------|
| **Altman Z-Score** | Complex 5-ratio formula | > 2.99 | 1.81-2.99 | < 1.81 |
| **Current Ratio** | Current Assets / Current Liabilities | > 2.0 | 1.0-2.0 | < 1.0 |
| **Debt-to-Equity** | Total Debt / Shareholder Equity | < 1.0 | 1.0-2.0 | > 2.0 |
| **Interest Coverage** | EBIT / Interest Expense | > 3.0 | 1.5-3.0 | < 1.5 |

### Calculating Z-Score from SAP Data

```python
def calculate_altman_z(bseg_data, vbak_data):
    """
    Altman Z-Score = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
    
    Where:
    A = Working Capital / Total Assets
    B = Retained Earnings / Total Assets
    C = EBIT / Total Assets
    D = Market Value Equity / Total Liabilities
    E = Sales / Total Assets
    """
    # Would require mapping SAP G/L accounts to these categories
    pass
```

---

## Option 4: External Label Integration via SAP BTP

### Architecture for Real-Time Label Enrichment

```
┌─────────────────────────────────────────────────────────────────┐
│                        SAP BTP                                  │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    │
│  │ SAP S/4HANA │───>│ Integration  │───>│ ML Model        │    │
│  │ (Features)  │    │ Suite        │    │ (Prediction)    │    │
│  └─────────────┘    └──────────────┘    └─────────────────┘    │
│         │                  │                     │              │
│         │                  ▼                     ▼              │
│         │          ┌──────────────┐    ┌─────────────────┐     │
│         │          │ External API │    │ Risk Dashboard  │     │
│         │          │ (D&B, Experian)   │ (SAC)           │     │
│         │          └──────────────┘    └─────────────────┘     │
│         │                  │                                    │
│         ▼                  ▼                                    │
│  ┌─────────────────────────────────────┐                       │
│  │        Training Dataset             │                       │
│  │  ┌─────────┬──────────┬──────────┐  │                       │
│  │  │Features │ Labels   │ Outcomes │  │                       │
│  │  │(SAP)    │(External)│(Validated)  │                       │
│  │  └─────────┴──────────┴──────────┘  │                       │
│  └─────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

### Required SAP BTP Components

| Component | Purpose | Cost |
|-----------|---------|------|
| **SAP Integration Suite** | Connect to external data sources | $$ |
| **SAP HANA Cloud** | Store labeled training data | $$ |
| **SAP AI Core** | Train and deploy ML models | $$$ |
| **SAP Analytics Cloud** | Visualize risk scores | $$ |

---

## Comparison: Current Project vs. Production System

| Aspect | Current Prototype | Production System |
|--------|-------------------|-------------------|
| **Training Data** | Kaggle (external) | SAP + External Labels |
| **Labels** | Taiwanese companies (1999-2009) | Current customers/vendors |
| **Features** | 95 pre-calculated ratios | Real-time SAP calculations |
| **Prediction Target** | Generic bankruptcy | Specific credit/distress risk |
| **Validation** | Test set (Kaggle) | Hold-out + Live monitoring |
| **Update Frequency** | One-time | Continuous retraining |
| **Regulatory Compliance** | N/A | GDPR, Fair Lending, etc. |

---

## Recommendations for a Real Implementation

### Step 1: Define Your Prediction Target

| Question | Answer Guides Implementation |
|----------|------------------------------|
| Predicting bankruptcy of **customers** or **your company**? | Determines data source |
| What is your **time horizon**? (1 year, 5 years?) | Affects feature engineering |
| What **actions** will you take based on predictions? | Determines threshold setting |
| What is the **cost of false negatives vs false positives**? | Guides model optimization |

### Step 2: Acquire or Create Labels

| If You Have... | Then... |
|----------------|---------|
| Historical write-offs in SAP | Use these as proxy bankruptcy labels |
| Credit bureau subscription | Integrate via API for external labels |
| Only transactional data | Create synthetic labels based on ratios |
| Customer/vendor master data | Check for "blocked" or "flagged" indicators |

### Step 3: Build Feature Pipeline

1. **Map SAP tables to financial ratios** (see DATA_README.md)
2. **Create time-series features** (trends, seasonality)
3. **Add external data** (macroeconomic indicators, industry data)
4. **Implement in ABAP or Python** (depending on architecture)

### Step 4: Train and Deploy

1. **Split data by time** (train on past, test on recent)
2. **Handle class imbalance** (SMOTE, class weights)
3. **Validate with business experts** (do predictions make sense?)
4. **Deploy to SAP BTP or embedded analytics**
5. **Monitor and retrain** (model drift detection)

---

## Summary

| Label Type | Difficulty | Data Quality | Use Case |
|------------|------------|--------------|----------|
| **External (Credit Bureau)** | Low | High | B2B credit risk |
| **Internal (Write-offs)** | Low | Medium | Existing customer analysis |
| **Synthetic (Ratio-based)** | Medium | Low | Research/prototyping |
| **Hybrid (External + SAP)** | High | Highest | Production systems |

**The key insight:** Without labels tied to actual bankruptcy outcomes for entities in your SAP system, any "predictions" are demonstrations of methodology rather than validated risk assessments.

---

## References

1. **Altman, E.I. (1968)** - "Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy" - *Journal of Finance*
2. **SAP Help Portal** - Business Partner Master Data
3. **Dun & Bradstreet** - Commercial Credit Risk API Documentation
4. **Zhao & Bai (2022)** - "Financial Fraud Detection Using SMOTE and Machine Learning" - *SHS Web of Conferences*

---

*Last Updated: December 3, 2025*
