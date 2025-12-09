# Data Documentation

This project demonstrates **transfer learning for financial risk forecasting** by training on external labeled data and applying to SAP S/4HANA companies.

---

## 1. External Training Dataset: Taiwan Financial Distress Data (Transfer Learning Source)

**Location:** `data/kaggle/kaggle_company_bankruptcy.csv`

**Source:** [Kaggle - Company Bankruptcy Prediction](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction)

**Description:** Taiwan Economic Journal company financial distress data (1999-2009)

**Purpose:** Train ML models on labeled financial distress patterns to transfer to SAP companies

| Property | Value |
|----------|-------|
| Rows | 6,819 companies |
| Columns | 96 features |
| Target | `Bankrupt?` (0 = No, 1 = Yes) |
| Class Distribution | ~3% bankrupt, ~97% healthy |

### Key Features (96 Financial Ratios):
- **Profitability:** ROA, ROE, Operating Profit Rate, Net Profit Margin
- **Liquidity:** Current Ratio, Quick Ratio, Cash Flow Rate
- **Leverage:** Debt Ratio, Equity to Liability, Interest Coverage
- **Efficiency:** Asset Turnover, Inventory Turnover, Receivables Turnover
- **Growth:** Revenue Growth, Net Income Growth

### Usage:
`python
import pandas as pd
df_kaggle = pd.read_csv('data/kaggle/kaggle_company_bankruptcy.csv')
`

---

## 2. SAP S/4HANA Global Bike Dataset (Primary Application Target - Unlabeled)

**Location:** `data/sap/`

**Source:** SAP S/4HANA Global Bike Inc. (GBI) Educational System via SAP GUI (UCC)

**Description:** Enterprise transactional data from Global Bike Inc., demonstrating how to apply trained risk models to SAP companies. Contains real ERP transactions across Financial Accounting (FI) and Sales & Distribution (SD) modules.

**Purpose:** Target dataset for risk scoring and operational decision-making

### Company Codes in GBI:
| Code | Description | Currency |
|------|-------------|----------|
| **US00** | Global Bike USA (Dallas) | USD |
| **DE00** | Global Bike Germany | EUR |

### Sales Organizations:
| Code | Description |
|------|-------------|
| **UW00** | US West |
| **UE00** | US East |
| **DS00** | Germany South |
| **DN00** | Germany North |
| **DB00** | Germany Berlin |

---

## SAP Table Definitions

### Financial Accounting (FI) Tables

#### BKPF - Accounting Document Header
**SAP Module:** FI (Financial Accounting)

**Purpose:** Stores header-level information for every accounting document posted in SAP. Each document represents a complete accounting transaction (invoice, payment, journal entry, etc.)

| Key Field | Description | Example |
|-----------|-------------|---------|
| BUKRS | Company Code | US00, DE00 |
| BELNR | Document Number | 1400000001 |
| GJAHR | Fiscal Year | 2025 |
| BLART | Document Type | RV (Billing), SA (G/L), KZ (Payment) |
| BLDAT | Document Date | 2025-09-24 |
| BUDAT | Posting Date | 2025-09-24 |
| WAERS | Currency | USD, EUR |
| USNAM | User Name | LEARN-076 |
| TCODE | Transaction Code | VF01 (Billing), FB01 (Post) |

**Document Types (BLART):**
| Code | Meaning |
|------|---------|
| RV | Billing Document |
| SA | G/L Account Document |
| KZ | Vendor Payment |
| DZ | Customer Payment |
| AB | Clearing Document |

---

#### BSEG - Accounting Document Line Item
**SAP Module:** FI (Financial Accounting)

**Purpose:** Stores individual line items within each accounting document. Every document has at least 2 lines (debit and credit) following double-entry bookkeeping.

| Key Field | Description | Example |
|-----------|-------------|---------|
| BUKRS | Company Code | US00 |
| BELNR | Document Number | 1400000001 |
| GJAHR | Fiscal Year | 2025 |
| BUZEI | Line Item Number | 001, 002 |
| KOART | Account Type | D (Customer), K (Vendor), S (G/L) |
| SHKZG | Debit/Credit | S (Debit), H (Credit) |
| DMBTR | Amount in Local Currency | 15000.00 |
| WRBTR | Amount in Document Currency | 15000.00 |
| HKONT | G/L Account | 1200000 (AR), 4000000 (Revenue) |
| KOSTL | Cost Center | 1000 |

**Account Types (KOART):**
| Code | Meaning |
|------|---------|
| D | Customer (Debtor) |
| K | Vendor (Kreditor) |
| S | G/L Account (Sachkonto) |
| A | Asset |
| M | Material |

---

#### BSID - Customer Open Items (Accounts Receivable)
**SAP Module:** FI-AR (Accounts Receivable)

**Purpose:** Contains all **open (unpaid)** customer invoices. When a customer pays, the item moves from BSID to BSAD (cleared items).

| Key Field | Description | Example |
|-----------|-------------|---------|
| BUKRS | Company Code | US00 |
| KUNNR | Customer Number | 1000 |
| BELNR | Invoice Document Number | 1400000001 |
| DMBTR | Amount in Local Currency | 24000.00 |
| ZFBDT | Baseline Date for Payment | 2025-09-24 |
| ZBD1T | Days for Net Payment | 30 |
| SHKZG | Debit/Credit Indicator | S (Debit = We are owed) |

**Use for Risk Analysis:**
- Aging analysis (days overdue)
- Customer payment behavior
- Working capital (AR turnover)

---

#### BSAK - Vendor Cleared Items (Accounts Payable)
**SAP Module:** FI-AP (Accounts Payable)

**Purpose:** Contains all **cleared (paid)** vendor invoices. Shows historical payment patterns to vendors.

| Key Field | Description | Example |
|-----------|-------------|---------|
| BUKRS | Company Code | US00 |
| LIFNR | Vendor Number | 100000 |
| BELNR | Document Number | 1500000001 |
| DMBTR | Amount Paid | 5000.00 |
| AUGDT | Clearing Date | 2025-10-01 |
| AUGBL | Clearing Document | 1500000050 |

**Use for Risk Analysis:**
- Payment timing patterns
- Cash outflow analysis
- AP turnover

---

### Sales & Distribution (SD) Tables

#### VBAK - Sales Order Header
**SAP Module:** SD (Sales & Distribution)

**Purpose:** Stores header-level information for each sales order. Represents customer orders before delivery and billing.

| Key Field | Description | Example |
|-----------|-------------|---------|
| VBELN | Sales Order Number | 1000 |
| ERDAT | Order Created Date | 2025-09-24 |
| VKORG | Sales Organization | UW00 (US West) |
| KUNNR | Sold-to Customer | 1000 |
| NETWR | Net Order Value | 20092.50 |
| WAERK | Currency | USD |
| AUART | Order Type | OR (Standard Order) |

**Order Types (AUART):**
| Code | Meaning |
|------|---------|
| OR | Standard Order |
| SO | Rush Order |
| RE | Returns |

---

#### VBAP - Sales Order Item
**SAP Module:** SD (Sales & Distribution)

**Purpose:** Stores line item details for each sales order. Each order can have multiple items (products).

| Key Field | Description | Example |
|-----------|-------------|---------|
| VBELN | Sales Order Number | 1000 |
| POSNR | Item Number | 10, 20, 30 |
| MATNR | Material Number | DXTR1000 (Bike model) |
| KWMENG | Order Quantity | 10 |
| NETPR | Net Price per Unit | 1500.00 |
| NETWR | Net Value | 15000.00 |
| WERKS | Plant | DL00 (Dallas) |

**Use for Risk Analysis:**
- Revenue volatility
- Product mix analysis
- Demand forecasting
- Customer behavior patterns

---

## Extraction Instructions

### Prerequisites:
- SAP GUI installed
- Access to SAP S/4HANA GBI system (UCC: e.g., DEVAAZ-201)
- Valid login credentials

### Step-by-Step Export:

1. **Log into SAP**
   - Open SAP Logon
   - Connect to your GBI system
   - Enter credentials

2. **For each table (BKPF, BSEG, BSID, BSAK, VBAK, VBAP):**

   a. Type in command bar: `/nSE16` then Press Enter
   
   b. Enter table name (e.g., `BKPF`) then Press Enter
   
   c. Set **Maximum No. of Hits** = `999999999`
   
   d. Press **F8** (Execute)
   
   e. Export: Menu - **System** - **List** - **Save** - **Local File**
   
   f. Choose **"Text with Tabs"**
   
   g. Save as `TABLENAME_ALL.txt` (e.g., `BKPF_ALL.txt`)

3. **Move files to:** `data/sap/`

### Expected Files:
`
data/sap/
+-- BKPF_ALL.txt    (Accounting Headers)
+-- BSEG_ALL.txt    (Accounting Line Items)
+-- BSID_ALL.txt    (Open AR)
+-- BSAK_ALL.txt    (Cleared AP)
+-- VBAK_ALL.txt    (Sales Order Headers)
+-- VBAP_ALL.txt    (Sales Order Items)
`

---

## Reading Data in Python

### Kaggle Dataset:
`python
import pandas as pd
df_kaggle = pd.read_csv('data/kaggle/kaggle_company_bankruptcy.csv')
`

### SAP Data (SE16N Text with Tabs Export):

**Important:** SAP exports include 3 header rows (date, title, blank line) that must be skipped.

```python
import pandas as pd

# Read any SAP table - use skiprows=3, encoding=latin-1
df_bkpf = pd.read_csv('data/sap/BKPF_ALL.txt', sep='	', encoding='latin-1', skiprows=3, low_memory=False)
df_bseg = pd.read_csv('data/sap/BSEG_ALL.txt', sep='	', encoding='latin-1', skiprows=3, low_memory=False)
df_bsid = pd.read_csv('data/sap/BSID_ALL.txt', sep='	', encoding='latin-1', skiprows=3, low_memory=False)
df_bsak = pd.read_csv('data/sap/BSAK_ALL.txt', sep='	', encoding='latin-1', skiprows=3, low_memory=False)
df_vbak = pd.read_csv('data/sap/VBAK_ALL.txt', sep='	', encoding='latin-1', skiprows=3, low_memory=False)
df_vbap = pd.read_csv('data/sap/VBAP_ALL.txt', sep='	', encoding='latin-1', skiprows=3, low_memory=False)
```

### Current Data Summary (as of Dec 2025):

| Table | Rows | Columns | Description |
|-------|------|---------|-------------|
| BKPF | 38,179 | 100 | Accounting Document Headers |
| BSEG | 90,476 | 100 | Accounting Line Items |
| BSID | 3 | 100 | Open Customer Items (AR) |
| BSAK | 2 | 100 | Cleared Vendor Items (AP) |
| VBAK | 109 | 100 | Sales Order Headers |
| VBAP | 247 | 100 | Sales Order Items |
`

---

## Data Integration Strategy

| Aspect | Kaggle Dataset | SAP Dataset |
|--------|---------------|-------------|
| **Role** | Training data (labeled) | Application data (unlabeled) |
| **Labels** | Bankruptcy outcome (0/1) | None |
| **Features** | 95 financial ratios | Raw transactions |
| **Purpose** | Train ML models | Generate risk scores |

### Workflow:
1. **Train** models on Kaggle data (learn bankruptcy patterns)
2. **Engineer features** from SAP data (create comparable ratios from BSEG, BSID, BSAK, VBAK, VBAP)
3. **Apply** trained model to SAP-derived features
4. **Generate** P(Distress) scores for GBI company codes (US00, DE00)

---

## Feature Mapping: SAP to Financial Ratios

| Kaggle Feature | SAP Source | Calculation |
|----------------|------------|-------------|
| Current Ratio | BSID, BSAK | AR Total / AP Total |
| Debt Ratio | BSEG | Liabilities / Assets |
| Revenue | VBAK, VBAP | Sum of Net Values |
| AR Turnover | BSID, VBAK | Revenue / Avg AR |
| Working Capital | BSID, BSAK | AR - AP |

---

## Notes:
- SAP data does **NOT** contain actual bankruptcy labels
- Risk scores are **predictions based on learned patterns**, not actual outcomes
- This is a **prototype** demonstrating transfer learning from external data to ERP systems
- GBI is a fictional company - results are for educational purposes only

