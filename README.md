# Customer Churn Prediction & Reactivation Propensity

![Python Version](https://img.shields.io/badge/python-3.13-blue)
![CI Status](https://github.com/r-midolli/customer-churn-reactivation/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-green)
[![Portfolio](https://img.shields.io/badge/Portfolio-Live_Page-007EC6?style=flat&logo=Web)](https://R-midolli.github.io/Customer-Churn-Prediction-Reactivation-Propensity/project-churn.html)

---

## 🇫🇷 Description du Projet

### Problème
Comment identifier les clients du secteur FMCG (Fast-Moving Consumer Goods) sur le point d'abandonner (churner) et leur allouer efficacement un budget de réactivation marketing ? Actuellement, de nombreux coupons sont distribués à l'aveugle, ce qui diminue le Retour Sur Investissement (ROI) des campagnes.

### Solution
Un modèle de Machine Learning (XGBoost) entraîné sur les données transactionnelles Dunnhumby. Le système n'optimise pas seulement les probabilités de fidélisation, mais ajuste dynamiquement son seuil (threshold) selon le véritable coût métier (Le coût de perdre un client vs le coût d'un coupon inutile). Les clients sont ensuite segmentés via CLV (Customer Lifetime Value) pour personnaliser l'offre de réactivation.

### Résultats Clés
- **Validation Sans Fuite (Zero Leakage)** : Walk-forward Cross-Validation garantit des performances en conditions réelles.
- **Réduction des Coûts (ROI Targeting)** : Le système identifie prioritairement les profils dont l'Upside de réactivation excède l'investissement.
- **Transparence (Explainable AI)** : Calcul formel de l'importance des variables via les valeurs de SHAP globales et locales (Waterfall).

---

## 🇬🇧 Project Description

### Problem
How do we accurately identify FMCG Retail customers who are about to churn and effectively allocate our reactivation marketing budget? Distributing broad discount coupons blindly erodes the campaign's Return on Investment (ROI).

### Solution
A Machine Learning pipeline using XGBoost on the Dunnhumby Complete Journey dataset. This platform optimizes its classification threshold not via standard F1 score, but by simulating the true business cost (the high cost of customer loss vs the lower cost of a wasted coupon). The output segmentations pair churning probability with Customer Lifetime Value (CLV) to recommend tailored discount values.

### Key Results
- **Zero Leakage Execution**: Walk-forward cross-validation ensuring pure generalization testing.
- **Business-Optimized Targets**: The tool prioritizes reactivation profiles that offer positive net ROI to maximize revenue retention.
- **Explainable AI**: Transparent modeling with global and local SHAP Values indicating why each predictive decision was driven.

---

## 📊 Résultats du Modèle

| Métrique | Score |
|----------|-------|
| AUC-ROC | *(run pipeline to populate)* |
| Precision (threshold optimisé) | *(run pipeline to populate)* |
| Recall (threshold optimisé) | *(run pipeline to populate)* |
| Clients High-Value identifiés | *(run pipeline to populate)* |

---

## 💼 Contexte Métier

Dans le secteur FMCG (Grande Consommation), l'attrition des clients (churn) est un problème majeur qui impacte directement la rentabilité. Une approche classique consiste à distribuer des coupons de réduction à grande échelle pour réactiver les clients inactifs. Cependant, cette méthode souffre souvent d'un ROI négatif, car le budget est gaspillé sur des clients qui auraient acheté de toute façon ou sur des clients qui ne feront jamais leur retour.

Notre stratégie repose sur un calcul direct du Retour sur Investissement (ROI), avec une recommandation de coupon personnalisée (5€, 10€, 20€) basée sur le potentiel de chaque segment CLV (Low, Mid, High). Par exemple, pour le segment "Mid", nous avons un taux de réponse estimé à 16% pour un coupon de 10€, qui permet un ROI positif uniquement si la CLV est suffisante. Pour le segment "Low", tout coupon conduirait à un ROI prédictif négatif avec un taux de réponse de conversion très bas (5% à 14%), il est donc identifié comme tel et exclu des campagnes ("Pas de contact"). Le segment "High", avec un taux de réponse fort de 35% pour un coupon de 20€, est l'objectif prioritaire absolu afin de maximiser les marges et l'upside.

---

## 🗂️ Architecture

```text
.
├── notebooks/
│   ├── 01_eda.py        # Analyse Données, comportement d'achat
│   ├── 02_model.py      # Entraînement ML, threshold opt, SHAP
│   └── 03_demo.py       # App Marimo interactive ROI Simulator
└── src/
    ├── data.py          # Data ingestion Dunnhumby
    ├── features.py      # RFM, feature engineering
    ├── model.py         # XGBoost & optimisation
    └── roi.py           # Segmentation CLV & Recommandation
```

---

## 📎 Portfolio

Visitez notre page de présentation spécialement conçue pour illustrer ce cas d'usage technique :
👉 **[Page Portfolio du Projet Churn Analytics](https://R-midolli.github.io/Customer-Churn-Prediction-Reactivation-Propensity/project-churn.html)**

---

## 🚀 Quick Start

Ensure you have [uv](https://docs.astral.sh/uv/) installed.

```bash
# 1. Clone the repository
git clone https://github.com/r-midolli/customer-churn-reactivation.git
cd customer-churn-reactivation

# 2. Setup environment and download Dunnhumby data
uv sync
uv run python -c "from src.data import download_dunnhumby; download_dunnhumby()"

# 3. Launch interactive demo
uv run marimo run notebooks/03_demo.py
```
