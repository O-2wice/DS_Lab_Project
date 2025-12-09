# DS Lab 2 Roadmap: From Proof-of-Concept to Real Implementation

## Executive Summary

**Current State (DS Lab 1):**
- ✅ Strong ML foundation: XGBoost model trained on 6,819 real companies (94% ROC-AUC)
- ✅ Advanced techniques: SMOTE, SHAP interpretation, threshold optimization
- ⚠️ SAP risk scores are simulated (manually hardcoded), not actual model predictions
- ⚠️ Missing: Financial ratio calculation from SAP transactional data

**DS Lab 2 Goal:**
Transform the demonstrative SAP integration into a **real, validated implementation** by calculating actual financial ratios from SAP data and generating true model predictions.

---

## Data Availability Assessment

### What We Have ✅

**1. Kaggle Taiwan Bankruptcy Dataset** (Complete & Ready)
- 6,819 companies with bankruptcy labels
- 95 pre-calculated financial ratios
- Years: 1999-2009
- **Status:** Fully utilized for model training

**2. SAP S/4HANA GBI Tables** (Transactional Data)
```
Table       Rows      Description                       Key Columns
────────────────────────────────────────────────────────────────────
BKPF        38,179    Accounting Document Headers       CoCd, DocumentNo, Year
BSEG        90,476    Accounting Line Items            CoCd, G/L Acct, Amount LC
BSID        3         Open AR Items                    Customer, Amount
BSAK        2         Cleared AP Items                 Supplier, Amount
VBAK        109       Sales Order Headers              Sales Org., Net Value
VBAP        247       Sales Order Items                Sales Doc., Item
```

**Status:** Loaded but NOT processed into financial ratios

### What We Need to Extract from SAP 📊

From **BSEG** (90,476 line items):
- G/L Account numbers (e.g., 1200000, 4000000, 3800000)
- Amount in Local Currency
- Company Code (CoCd)
- Fiscal Year
- Debit/Credit indicator

**Critical Columns Confirmed Available:**
- `CoCd` - Company identifier (DE00, US00, etc.)
- `G/L Acct` - General Ledger account number ✅
- `Amount LC` - Transaction amount in local currency ✅
- `Year` - Fiscal year ✅
- `D/C` - Debit/Credit indicator ✅

**Validation:** ✅ All necessary columns for financial statement reconstruction are present in BSEG

---

## Implementation Roadmap

### Phase 1: Financial Ratio Calculation (Core Requirement)
**Effort:** 40-60 hours | **Priority:** Critical

#### Step 1.1: G/L Account Classification
Create mapping from SAP chart of accounts to financial statement categories.

**Input:** BSEG column `G/L Acct` (account numbers like 1200000, 4000000)

**Output:** `gl_account_mapping.csv`
```csv
gl_account,category,subcategory,statement
1200000,Cash,Cash and Equivalents,Balance Sheet
1300000,Accounts Receivable,Current Assets,Balance Sheet
1400000,Inventory,Current Assets,Balance Sheet
2100000,Accounts Payable,Current Liabilities,Balance Sheet
2600000,Long-term Debt,Non-Current Liabilities,Balance Sheet
3000000,Equity,Shareholders Equity,Balance Sheet
4000000,Revenue,Operating Revenue,Income Statement
4770000,Freight Revenue,Operating Revenue,Income Statement
5000000,COGS,Operating Expenses,Income Statement
```

**Code Module:** `src/sap_gl_mapper.py`
```python
def classify_gl_account(gl_account: str) -> dict:
    """Map SAP G/L account to financial statement category"""
    # 1xxx = Assets
    # 2xxx = Liabilities  
    # 3xxx = Equity
    # 4xxx = Revenue
    # 5xxx-9xxx = Expenses
    
    if gl_account.startswith('1'):
        return classify_asset(gl_account)
    elif gl_account.startswith('2'):
        return classify_liability(gl_account)
    # ... continue for all categories
```

**Effort:** 15-20 hours (requires SAP chart of accounts documentation)

---

#### Step 1.2: Balance Sheet Aggregation
Sum BSEG line items by company, account category, and fiscal period.

**Input:** 
- BSEG (90,476 rows)
- gl_account_mapping.csv

**Process:**
```python
# Aggregate line items to financial statement totals
balance_sheet = df_bseg.merge(gl_mapping, on='G/L Acct', how='left')

# Convert European format (1.234,56 → 1234.56)
balance_sheet['Amount'] = clean_currency(balance_sheet['Amount LC'])

# Sum by company and category
financial_totals = balance_sheet.groupby([
    'CoCd', 'Year', 'category', 'D/C'
]).agg({'Amount': 'sum'}).reset_index()

# Net debit - credit for each category
net_totals = financial_totals.pivot_table(
    index=['CoCd', 'Year', 'category'],
    columns='D/C',
    values='Amount',
    fill_value=0
)
net_totals['Net_Amount'] = net_totals['S'] - net_totals['H']  # S=Debit, H=Credit
```

**Output:** `data/processed/sap_financial_statements.csv`
```csv
CoCd,Year,Total_Assets,Current_Assets,Total_Liabilities,Equity,Revenue,COGS,Net_Income
DE00,2025,15234567,8923456,7123456,8111111,12000000,6500000,2340000
US00,2025,8923456,5234567,4123456,4800000,8500000,4200000,1560000
```

**Effort:** 10-15 hours

---

#### Step 1.3: Financial Ratio Calculation
Calculate the same 95 ratios that the Kaggle model expects.

**Critical Ratios (Top 20 from SHAP analysis):**

| Ratio | Formula | Category |
|-------|---------|----------|
| ROA(C) before interest and depreciation | Net Income / Total Assets | Profitability |
| ROA(A) before interest and % after tax | (Net Income + Interest) / Total Assets | Profitability |
| Net Value Per Share (B) | Equity / Outstanding Shares | Valuation |
| Persistent EPS in Last Four Seasons | Avg(EPS) over 4 quarters | Earnings |
| Operating Gross Margin | (Revenue - COGS) / Revenue | Margin |
| Current Ratio | Current Assets / Current Liabilities | Liquidity |
| Quick Ratio | (Current Assets - Inventory) / Current Liabilities | Liquidity |
| Debt Ratio % | Total Liabilities / Total Assets | Leverage |
| Cash Flow to Total Assets | Cash Flow / Total Assets | Cash Flow |
| Working Capital to Total Assets | (Current Assets - Current Liabilities) / Total Assets | Efficiency |

**Code Module:** `src/financial_ratios.py`
```python
def calculate_all_ratios(financial_statements: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 95 financial ratios matching Kaggle features.
    
    Parameters:
    -----------
    financial_statements : DataFrame with columns:
        - CoCd, Year, Total_Assets, Current_Assets, Equity, 
          Revenue, COGS, Net_Income, etc.
    
    Returns:
    --------
    DataFrame with 95 ratio columns matching Kaggle exactly
    """
    ratios = pd.DataFrame()
    
    # Profitability Ratios (14 ratios)
    ratios['ROA(C) before interest and depreciation before interest'] = \
        financial_statements['Net_Income'] / financial_statements['Total_Assets']
    
    ratios['Operating Gross Margin'] = \
        (financial_statements['Revenue'] - financial_statements['COGS']) / \
        financial_statements['Revenue']
    
    # Liquidity Ratios (8 ratios)
    ratios['Current Ratio'] = \
        financial_statements['Current_Assets'] / financial_statements['Current_Liabilities']
    
    # ... (calculate all 95 ratios)
    
    return ratios
```

**Challenge:** SAP data may not have all components needed for every ratio
- **Solution:** Calculate subset of ratios (50-70 out of 95) where data is available
- Document which ratios are calculated vs. imputed

**Effort:** 20-25 hours

---

### Phase 2: Real Model Predictions on SAP Companies
**Effort:** 15-20 hours | **Priority:** High

#### Step 2.1: Feature Alignment
Ensure SAP-calculated ratios exactly match Kaggle feature names and order.

```python
# Load Kaggle feature names from trained model
kaggle_features = list(X_train.columns)  # 95 features

# Align SAP ratios to same structure
sap_ratios_aligned = sap_ratios[kaggle_features]

# Handle missing ratios (if SAP doesn't have all components)
for col in kaggle_features:
    if col not in sap_ratios.columns:
        sap_ratios_aligned[col] = 0.0  # or industry median
        print(f"⚠️ Missing: {col} - imputed with 0")
```

**Validation:**
- Feature count: 95 exactly
- Feature names: exact string match
- Data types: all float64

---

#### Step 2.2: StandardScaler Transformation
Apply the SAME scaler used in training (loaded from pickle).

```python
# Load trained scaler
with open('../models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Transform SAP ratios
X_sap_scaled = scaler.transform(sap_ratios_aligned)

# Verify scaling (mean ≈ 0, std ≈ 1 for each feature)
print("Scaled mean:", X_sap_scaled.mean(axis=0)[:5])
print("Scaled std:", X_sap_scaled.std(axis=0)[:5])
```

---

#### Step 2.3: Generate REAL Predictions
Run XGBoost model on scaled SAP features.

```python
# Load trained model
with open('../models/best_xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict bankruptcy probabilities
sap_bankruptcy_prob = model.predict_proba(X_sap_scaled)[:, 1]

# Predict binary outcomes (using optimal threshold 0.45)
OPTIMAL_THRESHOLD = 0.45
sap_predictions = (sap_bankruptcy_prob >= OPTIMAL_THRESHOLD).astype(int)

# Build results DataFrame
sap_results = pd.DataFrame({
    'Company_Code': sap_ratios_aligned.index,
    'Bankruptcy_Probability': sap_bankruptcy_prob,
    'Risk_Category': pd.cut(sap_bankruptcy_prob, 
                            bins=[0, 0.2, 0.4, 0.7, 1.0],
                            labels=['Minimal', 'Low', 'Medium', 'High']),
    'Prediction': sap_predictions
})

print("\n🎯 REAL SAP PREDICTIONS:")
print(sap_results)
```

**Expected Output:**
```
Company_Code  Bankruptcy_Probability  Risk_Category  Prediction
DE00          0.23                    Low            0
US00          0.67                    Medium         1
```

**Effort:** 5-8 hours

---

### Phase 3: Validation & Reporting (If Possible)
**Effort:** 10-15 hours | **Priority:** Medium

#### Challenge: No Ground Truth
SAP GBI companies are fictional educational data with no real bankruptcy outcomes.

#### Solutions:

**Option A: Historical Validation** (Recommended)
- Use 2024 SAP data to predict 2025 bankruptcy
- Manually label companies based on financial deterioration
- Compare model predictions to expert judgment

**Option B: Cross-Validation on Real Data**
- Find Hungarian/EU bankruptcy database with labeled companies
- Calculate ratios from their financial statements
- Validate model predictions against actual outcomes

**Option C: Simulated Outcomes**
- Define bankruptcy thresholds (e.g., Debt Ratio > 80%, ROA < -5%)
- Label SAP companies based on these rules
- Compare model predictions to rule-based labels

---

## Deliverables for DS Lab 2

### Code Modules (New)
1. `src/sap_gl_mapper.py` - G/L account classification (300 lines)
2. `src/balance_sheet_aggregator.py` - Financial statement builder (200 lines)
3. `src/financial_ratios.py` - Ratio calculation engine (500 lines)
4. `src/sap_predictor.py` - End-to-end SAP scoring pipeline (150 lines)

### Notebooks (Enhanced)
5. `notebooks/05_SAP_Financial_Ratios.ipynb` - Ratio calculation documentation
6. `notebooks/03_Model_Interpretation_and_Deployment.ipynb` - Replace simulated scores with real predictions

### Data Files (New)
7. `data/processed/gl_account_mapping.csv` - Chart of accounts classification
8. `data/processed/sap_financial_statements.csv` - Aggregated balance sheets
9. `data/processed/sap_ratios.csv` - 95 calculated ratios per company
10. `data/sap_exports/real_sap_predictions.csv` - Model output on SAP data

### Documentation
11. `docs/SAP_RATIO_CALCULATION_GUIDE.md` - Detailed methodology
12. `docs/DS_LAB_2_REPORT.md` - Academic report addendum

---

## Feasibility Assessment

### ✅ Definitely Feasible

**Financial Statement Reconstruction:**
- BSEG has all required columns (G/L Acct, Amount LC, D/C, CoCd, Year)
- 90,476 transactions provide sufficient detail
- Standard SAP chart of accounts structure

**Ratio Calculation:**
- Can calculate 50-70 ratios with high confidence
- Missing components can be approximated or imputed
- Methodology is well-documented in accounting literature

**Model Prediction:**
- Infrastructure already exists (trained model, scaler)
- Just need properly formatted input features

### ⚠️ Challenging but Possible

**Complete 95 Ratio Coverage:**
- Some ratios require data not in SAP tables (e.g., stock prices, depreciation details)
- Solution: Calculate subset + document limitations

**Validation:**
- No ground truth for SAP GBI companies
- Solution: Use alternative validation approaches (Option A/B/C above)

### ❌ Not Feasible (Without External Data)

**Time-Series Analysis:**
- SAP data appears to be from 2025 only (single year)
- Need multi-year data for trend ratios (e.g., "Persistent EPS in Last Four Seasons")
- Solution: Focus on cross-sectional ratios, document time-series limitation

**Market-Based Ratios:**
- Net Value Per Share requires stock price data
- SAP GBI has no capital market data
- Solution: Exclude market ratios, calculate only 80-85 ratios

---

## Effort Estimate

| Phase | Tasks | Hours | Complexity |
|-------|-------|-------|------------|
| **Phase 1.1** | G/L Account Mapping | 15-20 | Medium |
| **Phase 1.2** | Balance Sheet Aggregation | 10-15 | Low |
| **Phase 1.3** | Ratio Calculation | 20-25 | High |
| **Phase 2.1-2.3** | Model Prediction Pipeline | 15-20 | Medium |
| **Phase 3** | Validation & Reporting | 10-15 | Medium |
| **Documentation** | Reports, README, Comments | 10-15 | Low |
| **Total** | | **80-110 hours** | |

**Estimated Timeline:** 10-14 weeks @ 8 hours/week

---

## Success Criteria

✅ **Minimum Viable Implementation (DS Lab 2 Pass):**
1. G/L account classification documented with 50+ accounts mapped
2. Balance sheet aggregation working for all SAP companies
3. Calculate 50+ financial ratios with clear methodology
4. Generate REAL model predictions on SAP data (not hardcoded)
5. Documentation explaining what works, what's approximated, what's missing

✅ **Excellent Implementation (High Grade):**
1. 70+ ratios calculated with proper formulas
2. Feature alignment validation (show Kaggle vs SAP distributions)
3. SHAP re-analysis on SAP predictions
4. Alternative validation method implemented (Option A, B, or C)
5. Interactive dashboard showing SAP company risk scores

🏆 **Outstanding Implementation (Publication Quality):**
1. 85+ ratios with full documentation
2. Time-series component added (multi-year SAP data acquired)
3. Validated against real bankruptcy outcomes (external database)
4. Model retraining on combined Kaggle + SAP data
5. Comparative analysis: Taiwan vs. European companies

---

## Recommended Approach

**Start with:** Phase 1.1 → Phase 1.2 → Phase 2 (simplified)
**Reason:** Get working predictions quickly, then improve ratio coverage

**Incremental Milestones:**
1. Week 1-2: Map 20 G/L accounts, aggregate 5 companies → **Proof of concept**
2. Week 3-4: Calculate 10 key ratios → **Generate first real prediction**
3. Week 5-8: Expand to 50 ratios, all companies → **Minimum viable**
4. Week 9-12: Refine ratios, add validation → **Excellent quality**
5. Week 13-14: Polish documentation, visualizations → **Final submission**

---

## Key Takeaway

**This is ABSOLUTELY doable for DS Lab 2!** 

The SAP data has everything needed for financial statement reconstruction. The main work is:
1. Creating the G/L account mapping (tedious but straightforward)
2. Writing the aggregation and ratio calculation code (mechanical)
3. Validating the output (creative problem-solving)

The end result will transform this from a "demonstrative proof-of-concept" into a **real, end-to-end bankruptcy prediction system** that shows you can:
- Work with raw ERP data
- Implement domain knowledge (accounting)
- Apply ML to real-world scenarios
- Document limitations transparently

This is **exactly** the kind of project that demonstrates data science maturity. 🎯
