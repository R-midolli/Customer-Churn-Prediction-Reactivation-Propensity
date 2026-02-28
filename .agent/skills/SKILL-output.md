---
name: output
description: >
  Use este skill sempre que precisar calcular ROI de reativação, segmentar clientes
  por CLV, gerar o CSV CRM, construir o notebook Marimo de demo ou integrar no portfólio.
---

# SKILL: Output de Negócio & Portfólio

## Segmentação CLV e lógica ROI

```python
import numpy as np
import pandas as pd

# Segmentar por gasto histórico total (2 anos)
def segment_clv(spend: pd.Series) -> pd.Series:
    p40 = spend.quantile(0.40)
    p75 = spend.quantile(0.75)
    return pd.cut(spend, bins=[-np.inf, p40, p75, np.inf],
                  labels=["Low", "Mid", "High"])

RESPONSE_RATES = {
    "Low":  {5: 0.05, 10: 0.09, 20: 0.14},
    "Mid":  {5: 0.10, 10: 0.16, 20: 0.24},
    "High": {5: 0.15, 10: 0.23, 20: 0.35},
}

def best_coupon(clv: float, segment: str) -> tuple:
    """Retorna (valor_cupom, roi_esperado) — None se ROI negativo."""
    best, best_roi = None, -np.inf
    for coupon in [5, 10, 20]:
        roi = clv * 0.5 * RESPONSE_RATES[segment][coupon] - coupon
        if roi > best_roi:
            best_roi, best = roi, coupon
    return (best, round(best_roi, 2)) if best_roi > 0 else (None, round(best_roi, 2))
```

## Schema do CSV de saída (obrigatório)

```
data/outputs/churn_crm_YYYYMMDD.csv

Colunas (nesta ordem):
household_key | churn_score | clv_eur | clv_segment |
recommended_coupon_eur | expected_roi_eur | action | export_date
```

Valores de `action`: `"Priority contact"` / `"Contact"` / `"Do not contact"`

## Notebook Marimo de demo (03_demo.py)

Células obrigatórias, nesta ordem:
1. **Markdown**: Business Challenge em FR/EN
2. **Controles**: `mo.ui.slider` janela (15/30/60d) + score mínimo + `mo.ui.number` budget
3. **KPIs**: 4 cards — AUC, Lift @D1, clientes identificados, ROI total
4. **Gráfico Plotly**: barras horizontais top-20 clientes, cor = segmento CLV
5. **Tabela**: clientes filtrados por threshold e budget
6. **Download**: `mo.download(df.to_csv().encode(), "churn_crm.csv")`

```python
import marimo as mo

churn_window = mo.ui.slider(15, 60, step=15, value=30,
    label="📅 Janela de predição (dias)")
min_score = mo.ui.slider(0.5, 0.9, step=0.05, value=0.70,
    label="🎯 Score mínimo")
budget = mo.ui.number(500, 10000, step=500, value=3000,
    label="💶 Budget de reativação (€)")
```

## Integração no portfólio

```bash
cd C:/Users/rafae/workspace/portfolio_rafael_midolli

# 1. Leia um projeto existente ANTES de criar o novo
cat project-retail.html | head -150

# 2. Crie project-churn.html com a mesma estrutura CSS/HTML
# 3. Adicione o card em index.html (mais novo primeiro)

git add project-churn.html index.html
git commit -m "feat: add Customer Churn Prediction project page"
git push
```

Links obrigatórios no `project-churn.html`:
- GitHub: `https://github.com/r-midolli/customer-churn-reactivation`
- Demo: HTML exportado do Marimo (`marimo export html notebooks/03_demo.py`)