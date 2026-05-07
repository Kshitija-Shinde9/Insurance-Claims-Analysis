# Insurance Claims Analysis
### DATA 230 — Midterm + Final Group Project

This repository contains the complete DATA 230 insurance analytics project. It has two connected parts:

1. **Midterm Project:** insurance customer segmentation using synthetic customer CSV files, Plotly Dash, and Tableau.
2. **Final Group Project:** real motor insurance expected claim cost analysis using the freMTPL2 dataset, machine learning models, Power BI, Streamlit, and an IEEE-style final report.

The project starts with customer segmentation and ends with a larger real-world actuarial modeling workflow.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Part 1: Midterm Project](#part-1-midterm-project)
- [Part 2: Final Group Project](#part-2-final-group-project)
- [Datasets Used](#datasets-used)
- [Repository Structure](#repository-structure)
- [Setup and Installation](#setup-and-installation)
- [How to Run the Project](#how-to-run-the-project)
- [Machine Learning Models](#machine-learning-models)
- [Dashboards](#dashboards)
- [Reports and Outputs](#reports-and-outputs)
- [Key Findings](#key-findings)
- [Team](#team)

---

## Project Overview

This project analyzes insurance data to understand customer behavior, risk patterns, claim frequency, claim severity, and expected claim cost.

The midterm project focused on **customer segmentation** using synthetic insurance customer data. The final group project expanded the work into a stronger real-world insurance analytics problem using the **freMTPL2 French motor third-party liability dataset**.

The final goal is to answer a practical insurance question:

> Given driver, vehicle, policy, exposure, and geographic information, how can we estimate insurance risk and expected claim cost?

---

# Part 1: Midterm Project

## Midterm Title

**Insurance Customer Segmentation**

## Midterm Overview

The midterm project analyzes a synthetic insurance customer dataset to identify meaningful customer segments based on demographics, financial behavior, policy choices, claim history, and risk profiles.

The goal was to support:

- customer segmentation,
- targeted marketing,
- risk management,
- product personalization,
- dashboard-based storytelling.

The midterm included data cleaning, exploratory data analysis, an interactive Plotly Dash dashboard, and Tableau dashboards.

## Midterm Objectives

- Segment insurance customers into useful behavioral and demographic groups.
- Explore relationships between age, income, credit score, risk profile, and premium amount.
- Visualize customer segments through an interactive dashboard.
- Support business insights for underwriting, retention, and personalization.
- Communicate findings using dashboard screenshots and Tableau Public dashboards.

## Midterm Datasets

| File | Rows | Columns | Use |
|---|---:|---:|---|
| `data_synthetic_.csv` | 54,503 | 30 | Main synthetic customer segmentation dataset |
| `insurance_dataset.csv` | 13,000 | 7 | Supporting insurance claim amount dataset |

## Midterm Key Features

| Category | Example Features |
|---|---|
| Demographics | Age, Gender, Marital Status, Occupation, Education Level |
| Financial | Income Level, Premium Amount, Credit Score, Coverage Amount, Deductible |
| Policy | Policy Type, Policy Start Date, Policy Renewal Date, Risk Profile |
| Behavioral | Claim History, Previous Claims History, Driving Record, Life Events |
| Preferences | Preferred Communication Channel, Preferred Contact Time, Preferred Language |
| Segmentation | Segmentation Group |

## Midterm Dashboard

The midterm dashboard was built using **Plotly Dash**.

Dashboard features included:

- segmentation group filter,
- policy type filter,
- risk profile filter,
- year range filter,
- customer count KPI,
- average premium KPI,
- average credit score KPI,
- average policy tenure KPI,
- customer count by segment,
- policy type share,
- premium amount by segment,
- credit score distribution,
- risk profile vs segment heatmap,
- policies started over time.

## Tableau Dashboards

Two Tableau Public dashboards were created for the midterm:

| Dashboard | Link |
|---|---|
| Risk View & Occupation Analytics | [View on Tableau Public](https://public.tableau.com/app/profile/kshitija.dipakrao.shinde/viz/RiskViewOccupationAnalyticsDashboard/RiskViewOccupationAnalyticsDashboard?publish=yes) |
| Customer Risk & Revenue Intelligence | [View on Tableau Public](https://public.tableau.com/app/profile/kshitija.dipakrao.shinde/viz/CustomerRiskRevenueIntelligenceDashboard/CustomerRiskRevenueIntelligenceDashboard?publish=yes) |

---

# Part 2: Final Group Project

## Final Project Title

**Large-Scale Motor Insurance Risk Intelligence: Expected Claim Cost Prediction, Risk Classification, and Interactive Dashboards**

## Final Project Overview

The final group project extends the midterm work by using a real insurance dataset: **freMTPL2 French Motor Third-Party Liability Insurance**.

This dataset is much larger and more actuarially meaningful than the midterm dataset. It allows us to model:

- claim frequency,
- claim severity,
- expected claim cost,
- pure premium,
- driver and vehicle risk,
- geographic risk,
- dashboard-based decision support.

The final project uses both the midterm datasets and the freMTPL2 files, but each one has a different role:

- The **midterm CSV files** support the customer segmentation and dashboard foundation.
- The **freMTPL2 files** support the final expected claim cost modeling and final machine learning workflow.

---

## Datasets Used

The full project uses three main CSV inputs.

| Dataset | Rows | Columns | Purpose |
|---|---:|---:|---|
| `data_synthetic_.csv` | 54,503 | 30 | Midterm customer segmentation and premium/risk dashboarding |
| `insurance_dataset.csv` | 13,000 | 7 | Supporting claim amount exploration |
| `freMTPL2freq.csv` + `freMTPL2sev.csv` | 678,013 policies after processing | 23 final columns | Final expected claim cost / pure premium modeling |

## Final Processed Dataset

The final processed CSV file is:

```text
data/processed/fremtpl2_policy_claims.csv
```

This file is created by merging and cleaning:

```text
freMTPL2freq.csv + freMTPL2sev.csv
```

### Processing Steps

1. Load `freMTPL2freq.csv`, which contains policy-level information.
2. Load `freMTPL2sev.csv`, which contains claim amount information.
3. Aggregate claim amounts by `IDpol`.
4. Left join the aggregated claim amounts back to the frequency table.
5. Keep policies with no claims.
6. Fill missing claim amounts with zero.
7. Keep valid positive exposure records.
8. Create engineered columns for modeling and dashboarding.

### Important Engineered Columns

| Column | Meaning |
|---|---|
| `TotalClaimAmount` | Total claim amount per policy |
| `PurePremium` | `TotalClaimAmount / Exposure` |
| `Frequency` | `ClaimNb / Exposure` |
| `HasClaim` | Whether the policy has at least one claim |
| `AvgClaimAmount` | Average claim amount for claim policies |
| `DriverAgeBand` | Driver age grouped into bands |
| `VehicleAgeBand` | Vehicle age grouped into bands |
| `BonusMalusBand` | BonusMalus score grouped into bands |

### Final Processed Data Summary

| Metric | Value |
|---|---:|
| Final policies | 678,013 |
| Final columns | 23 |
| Total exposure | 358,499.45 |
| Total claims | 36,102 |
| Total claim amount | 59,909,216.50 |
| Policy claim incidence | 5.02% |
| Zero-claim policies | 94.98% |
| Exposure-weighted claim frequency | 0.100703 |
| Weighted pure premium | 167.1110 |

---

## Repository Structure

Recommended final repository structure:

```text
Insurance-Claims-Analysis/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── README.md
│   │
│   ├── midterm_reference/
│   │   ├── data_synthetic_.csv
│   │   ├── insurance_dataset.csv
│   │   └── midterm_dash_app.py
│   │
│   ├── external/
│   │   └── fremtpl2/
│   │       ├── freMTPL2freq.csv
│   │       └── freMTPL2sev.csv
│   │
│   └── processed/
│       └── fremtpl2_policy_claims.csv
│
├── src/
│   ├── __init__.py
│   └── fremtpl2_utils.py
│
├── scripts/
│   ├── download_fremtpl2_data.py
│   ├── build_fremtpl2_modeling_file.py
│   ├── generate_fremtpl2_eda.py
│   ├── model_expected_claim_cost.py
│   ├── generate_report_assets.py
│   └── build_final_ieee_report_with_images.py
│
├── dashboard_streamlit/
│   └── app.py
│
├── outputs/
│   └── fremtpl2/
│       ├── eda/
│       └── modeling/
│
└── report/
    └──Final_Report 
```

---

## Setup and Installation

### Requirements

- Python 3.8 or higher
- pip
- Jupyter Notebook, optional
- Streamlit, optional for final dashboard
- Power BI Desktop, optional for Power BI dashboard

### Clone Repository

```bash
git clone https://github.com/<your-username>/Insurance-Claims-Analysis.git
cd Insurance-Claims-Analysis
```

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run the Project

## 1. Midterm Dashboard

Place these files in the midterm data folder:

```text
data/midterm_reference/data_synthetic_.csv
data/midterm_reference/insurance_dataset.csv
```

Run the Dash app:

```bash
python data/midterm_reference/midterm_dash_app.py
```

Open:

```text
http://127.0.0.1:8050
```

## 2. Download freMTPL2 Data

```bash
python scripts/download_fremtpl2_data.py
```

If download is blocked, manually place the files here:

```text
data/external/fremtpl2/freMTPL2freq.csv
data/external/fremtpl2/freMTPL2sev.csv
```

## 3. Build the Final Processed Dataset

```bash
python scripts/build_fremtpl2_modeling_file.py
```

This creates:

```text
data/processed/fremtpl2_policy_claims.csv
```

## 4. Generate EDA Visuals

```bash
python scripts/generate_fremtpl2_eda.py
```

Outputs are saved in:

```text
outputs/fremtpl2/eda/
```

## 5. Train Machine Learning Models

```bash
python scripts/model_expected_claim_cost.py
```

Outputs are saved in:

```text
outputs/fremtpl2/modeling/
```

## 6. Run Streamlit Dashboard

```bash
streamlit run dashboard_streamlit/app.py
```

---

## Machine Learning Models

The final project uses two types of machine learning.

## 1. Expected Claim Cost / Pure Premium Prediction

These models are used for the final expected claim cost task:

| Model | Purpose |
|---|---|
| Mean Baseline | Simple reference model |
| Poisson Regression | Claim frequency modeling |
| Gamma Regression | Claim severity modeling |
| Poisson × Gamma Product | Frequency-severity expected cost model |
| Tweedie Regression | Direct pure premium modeling |
| Histogram Gradient Boosting | Nonlinear pure premium comparison model |

### Model Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean baseline | 767.46 | 30,402.78 | -0.000002 |
| Poisson × Gamma product | 641.13 | 30,402.15 | 0.000039 |
| Tweedie pure premium | 584.02 | 30,403.17 | -0.000028 |
| HistGradientBoosting sample | 674.69 | 30,373.58 | 0.001918 |

### Best Models

- **Best RMSE:** Histogram Gradient Boosting
- **Best MAE:** Tweedie Regression

The R² values are close to zero because individual policy claim cost is very difficult to predict exactly. This is normal for sparse and heavy-tailed insurance claim data.

## 2. Risk Profile Classification

A **Random Forest Classifier** is used in the dashboard for risk-tier classification.

It predicts:

- Low
- Medium
- High
- Very High

Important note: this classifier is based mainly on BonusMalus-derived risk tiers, so it is best understood as an interpretable dashboard tool, not as the main expected claim cost model.

---

## Dashboards

## Midterm Dashboards

| Tool | Purpose |
|---|---|
| Plotly Dash | Interactive customer segmentation dashboard |
| Tableau | Business dashboard storytelling |

## Final Dashboards

| Tool | Purpose |
|---|---|
| Power BI | Portfolio risk dashboard, claim-rate heatmaps, BonusMalus trends, region and vehicle-brand risk |[View Dashboard]https://app.powerbi.com/links/cpfOmhv2tK?ctid=e85c5307-76b1-4c48-bc5d-e88373dda261&pbi_source=linkShare |
| Streamlit | ML results dashboard, risk-decile calibration, feature importance, and policy risk predictor |

Across the midterm and final project, the work uses four dashboard/visualization tools:

```text
Tableau + Plotly Dash + Power BI + Streamlit
```

---

## Reports and Outputs

Important final report files:

```text
reports/ieee/DATA230_Final_Report_TwoColumn_v5_qa_fixed.docx
reports/ieee/DATA230_Final_Report_TwoColumn_v5_qa_fixed.pdf
```

Important output folders:

```text
outputs/fremtpl2/eda/
outputs/fremtpl2/modeling/
reports/ieee/figures/
```

---

## Key Findings

- The midterm synthetic dataset was useful for customer segmentation and dashboard practice.
- The final freMTPL2 dataset is stronger for real actuarial modeling.
- Most policies have no claims, so the target is highly imbalanced.
- 94.98% of policies have zero recorded claims.
- Young drivers aged 18-25 have the highest policy claim incidence and high pure premium.
- BonusMalus is the clearest risk signal.
- Exact policy-level claim cost is difficult to predict, so calibration, ranking, segmentation, and dashboards are more useful than R² alone.
- Histogram Gradient Boosting performs best by RMSE.
- Tweedie Regression performs best by MAE.
- Dashboards make the technical results easier to understand for pricing and portfolio decisions.

---

## Team

**Group 8**

- Prathmesh Mankar
- Kshitija Shinde
- Parth Patel
- Vrishin Dharmesh Kunnatham Parambath

---
*.pkl
```

If large data files are not uploaded, explain in `data/README.md` how to download or recreate them.
