# Insurance Customer Segmentation
### DATA 230 — Project

> Predictive customer segmentation for an insurance dataset using exploratory data analysis, clustering, and interactive dashboards.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [Running the Dashboard](#running-the-dashboard)
- [Key Results & Findings](#key-results--findings)
- [Tableau Dashboards](#tableau-dashboards)

---

## Project Overview

This project analyzes a synthetic insurance customer dataset of unique **53,503 customers** to identify meaningful customer segments based on demographics, financial behavior, policy choices, and risk profiles. The goal is to provide actionable insights for targeted marketing, risk management, and product personalization in the insurance industry.

The project spans the full data science pipeline — from raw data cleaning and EDA to clustering-based segmentation, interactive visualization via a Plotly Dash dashboard, and business-facing Tableau dashboards.

---

## Objectives

- **Segment** insurance customers into meaningful behavioral and demographic groups
- **Explore** relationships between age, income, credit score, risk profile, and premium amount
- **Visualize** segment distributions and trends through an interactive web dashboard
- **Derive** business insights to support underwriting, retention, and personalization strategies
- **Communicate** findings through a structured slide deck and Tableau dashboards

---

## Dataset

| Property | Details |
|---|---|
| File | `data_synthetic_.csv` and `insurance_dataset.csv`|
| Records | 53,503+ customers |
| Features | 30+ columns |
| Source | Synthetic insurance dataset |

**Key features include:**

| Category | Features |
|---|---|
| Demographics | Age, Gender, Marital Status, Occupation, Education Level |
| Financial | Income Level, Premium Amount, Credit Score, Coverage Amount, Deductible |
| Policy | Policy Type, Policy Start Date, Policy Renewal Date, Risk Profile |
| Behavioral | Claim History, Previous Claims History, Driving Record, Life Events |
| Preferences | Preferred Communication Channel, Preferred Contact Time, Preferred Language |
| Target | Segmentation Group (Segment1–Segment5) |

> **Note:** Place `data_synthetic_.csv` and `insurance_dataset.csv` in the `/data` folder before running any code.

---

## Repository Structure

```
Insurance-Customer-Segmentation/
│
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
│
├── data/                            # ← Place your CSV files here
│   ├── data_synthetic_.csv          # Main synthetic dataset 
│   └── insurance_dataset.csv        # Secondary dataset 
│
├── notebooks/
│   └── DATA_230_Project.ipynb       # Main EDA & analysis notebook
│
├── dashboard/
│   └── app.py                       # Plotly Dash interactive dashboard
│
├── presentation/
│   └── Predictive_Customer_Segmentation.pptx   # Project slide deck
│
├── tableau/
│   └── links.md                     # Links to published Tableau dashboards
│
└── docs/
    └── dashboard_screenshots/       # Screenshots of dashboard & Tableau
```

---

## Setup & Installation

### Prerequisites
- Python 3.8 or higher
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Insurance-Customer-Segmentation.git
cd Insurance-Customer-Segmentation
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the Data Files

Place the following files inside the `/data` folder:
- `data_synthetic_.csv`
- `insurance_dataset.csv`

### 5. Launch the Jupyter Notebook

```bash
jupyter notebook notebooks/DATA_230_Project.ipynb
```

---

## Running the Dashboard

The interactive Plotly Dash dashboard runs locally on your machine.

```bash
cd dashboard
python app.py
```

Then open your browser and go to:
```
http://127.0.0.1:8050
```

### Dashboard Features

| Panel | Description |
|---|---|
| **Filters** | Filter by Segmentation Group, Policy Type, Risk Profile, and Year Range |
| **KPI Cards** | Filtered customer count, average premium, average credit score, average tenure |
| **Customer Count by Segment** | Bar chart of segment sizes |
| **Policy Type Share** | Donut chart of policy distribution |
| **Premium Amount by Segment** | Box plot showing premium spread per segment |
| **Credit Score Distribution** | Overlaid histogram by segment |
| **Average Premium Heatmap** | Risk profile × Segment premium matrix |
| **Policies Over Time** | Line chart of policy starts by year |

---

## Key Results & Findings

### Segment Distribution
The dataset contains **5 customer segments (Segment1–Segment5)**. Segment5 is the largest group (~13,976 customers) and Segment1 the smallest (~8,683).

### Premium Behavior
- Premium amounts are **broadly similar across all segments** (median ~$3,000), suggesting segmentation was driven more by behavioral and demographic features than price alone.
- The heatmap of average premium by Risk Profile × Segment shows **Segment4 and Segment5 consistently command the highest premiums**, particularly at Risk Profile levels 0 and 3.

### Credit Score Profile
- Credit scores range from ~500 to 850 across all segments, with **overlapping distributions** — indicating credit score alone is not the primary segmentation driver.
- Segment5 and Segment3 appear slightly more concentrated in higher credit score bins (750+).

### Policy Type Mix
- **Group policies** are the most common (34.1%), followed by Business (26.1%), Family (23.2%), and Individual (16.5%).
- The distribution is relatively balanced across all four policy types.

### Age & Demographics (Tableau)
- The 30–35 age group holds the highest policy count.
- Marital status is distributed across Divorced, Married, Separated, Single, and Widowed with a fairly uniform share within each age bin.

### Claim History & Risk (Tableau)
- Higher claim histories correlate with higher risk profile scores, validating the risk segmentation logic.
- Driving record (Clean, Minor Violations, Major Violations, DUI, Accident) is a key filter for risk-based premium decisions.

### Temporal Trend
- Policy starts peaked in **2018 (~9,300)**, dipped in 2019–2020, recovered in 2021, then gradually declined through 2023 (~8,800).

---

## Tableau Dashboards

Two interactive Tableau Public dashboards complement the Python analysis:

| Dashboard | Link |
|---|---|
| **Risk View & Occupation Analytics** | [View on Tableau Public](https://public.tableau.com/app/profile/kshitija.dipakrao.shinde/viz/RiskViewOccupationAnalyticsDashboard/RiskViewOccupationAnalyticsDashboard?publish=yes) |
| **Customer Risk & Revenue Intelligence** | [View on Tableau Public](https://public.tableau.com/app/profile/kshitija.dipakrao.shinde/viz/CustomerRiskRevenueIntelligenceDashboard/CustomerRiskRevenueIntelligenceDashboard?publish=yes) |

---
