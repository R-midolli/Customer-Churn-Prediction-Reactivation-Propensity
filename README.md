# Customer Churn Prediction & Reactivation Propensity

![Python Version](https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-%23178CFF.svg?logo=XGBoost&logoColor=white)
![Marimo](https://img.shields.io/badge/Marimo-000000.svg)
![Pytest](https://img.shields.io/badge/pytest-%23ffffff.svg?logo=pytest&logoColor=2f9fe3)
![uv](https://img.shields.io/badge/uv-000000.svg)

[![Portfolio](https://img.shields.io/badge/Portfolio-Live_Page-007EC6?style=flat&logo=Web)](https://R-midolli.github.io/Customer-Churn-Prediction-Reactivation-Propensity/project-churn.html)

---

## 🌐 Live Demo & Portfolio

Visit the presentation page specially designed to illustrate this technical use case:
👉 **[Project Churn Analytics Portfolio Page](https://R-midolli.github.io/Customer-Churn-Prediction-Reactivation-Propensity/project-churn.html)**

---

## 🇬🇧 Project Description

### Problem
How do we accurately identify FMCG Retail customers who are about to churn and effectively allocate our reactivation marketing budget? Distributing broad discount coupons blindly erodes the campaign's Return on Investment (ROI) while targeting the wrong profiles.

### Solution
A Machine Learning pipeline using XGBoost on the Dunnhumby Complete Journey dataset. This platform optimizes its classification threshold not via standard F1 score, but by simulating the true business cost (the high cost of customer loss vs the lower cost of a wasted coupon). The output segmentations pair churning probability with Customer Lifetime Value (CLV) to recommend tailored discount values.

### Key Results
- **Zero Leakage Execution**: Walk-forward cross-validation ensuring pure generalization testing.
- **Business-Optimized Targets**: The tool prioritizes reactivation profiles that offer positive net ROI to maximize revenue retention.
- **Explainable AI**: Transparent modeling with global and local SHAP Values indicating why each predictive decision was driven.

---

## 📊 Model Performance & ROI

| Metric | Score |
|----------|-------|
| **AUC-ROC (Test Set)** | 0.84 |
| **Lift @ Decile 1** | 3.2x |
| **Simulated ROI** | +45% (vs Random Selection) |
| **Targeting Strategy** | CLV Segmentation (Low, Mid, High) |

---

## 💼 Business Context

In the FMCG sector, customer attrition (churn) is a major issue that directly impacts profitability. A classic approach is to distribute discount coupons on a large scale to reactivate inactive customers. However, this method often suffers from a negative ROI because the budget is wasted on customers who would have purchased anyway, or on customers who will never return.

Our strategy is based on a direct Return on Investment (ROI) calculation, with a personalized coupon recommendation (€5, €10, €20) based on the potential of each CLV segment (Low, Mid, High). For example, a High-Value customer with CLV > €200 generates a positive net ROI as soon as a €10 or €20 coupon is given—the model automatically prioritizes these profiles.

---

## 🗂️ Architecture

```text
.
├── notebooks/
│   ├── 01_eda.py        # Data Analysis & Purchasing Behavior
│   └── 02_model.py      # ML Training, Threshold Opt, SHAP
├── src/
│   ├── data.py          # Data ingestion and cleaning
│   ├── features.py      # RFM & interaction features
│   └── model.py         # XGBoost and walk-forward validation
└── README.md
```

---

## 🛠️ Tech Stack & Methodology

- **Data Processing:** `pandas`, `scikit-learn`
- **Modeling:** `xgboost` with walk-forward temporal validation
- **Interpretation:** `shap` for feature importance
- **Package Management:** `uv` for reproducible environments
- **Dashboarding:** Native HTML/JS simulation embedded seamlessly on the Portfolio (Mocking Python calculations instantly on the client-side).

## 🚀 Interactive Demo

The predictive results and financial optimization matrix are live on my portfolio. You can test the impact of varying risk thresholds and campaign budgets dynamically:

👉 **[View Interactive Dashboard Simulator](https://r-midolli.github.io/portfolio_rafael_midolli/project-churn.html)**

## 🏆 Key Takeaways

1.  **Targeted Optimization:** Discounting only High-CLV clients on the brink of churning yields much higher net ROI than carpet-bombing.
2.  **Business Alignment:** Tuning the decision threshold based on campaign cost transforms an ML probability score into an actionable financial lever.
3.  **Execution Speed:** Exporting the 'Hot Leads' matrix enables daily automated reactivation campaigns via CRM.

---

## 🚀 Quick Start

Ensure you have [uv](https://docs.astral.sh/uv/) installed.

```bash
# 1. Clone the repository
git clone https://github.com/R-midolli/Customer-Churn-Prediction-Reactivation-Propensity.git
cd Customer-Churn-Prediction-Reactivation-Propensity

# 2. Setup environment and download Dunnhumby data
uv sync
uv run python -c "from src.data import download_dunnhumby; download_dunnhumby()"

# 3. Train the XGBoost model
uv run python -m src.model
```
