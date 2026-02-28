---
name: model
description: >
  Use este skill sempre que precisar fazer feature engineering, treinar o XGBoost,
  validar sem data leakage, otimizar threshold ou gerar explicações SHAP.
---

# SKILL: Features & Modelo

## Janela temporal (regra anti-leakage)

```
|<-- 90 dias observação -->|<-- 30 dias label -->|
        features aqui       reference_date    t_end

Nenhuma transação APÓS reference_date pode entrar em qualquer feature.
```

## Features obrigatórias (por household_key)

**RFM:**
- `recency_days` → dias desde última compra até reference_date
- `frequency_90d` → número de visitas distintas
- `monetary_90d` → gasto total
- `avg_basket` → monetary / frequency

**Comportamentais:**
- `category_diversity` → nº de categorias distintas compradas
- `promo_sensitivity` → % compras com desconto (coupon_disc > 0)
- `trend_freq_4w` → (visitas últ. 4 sem − vis. 4 sem ant.) / (vis. ant. + 1)
- `inactive_weeks` → semanas sem compra nos últimos 30 dias

**Label:**
- `churned = 1` se zero compras nos 30 dias após reference_date
- `churned = 0` se ao menos 1 compra

## XGBoost — configuração

```python
import xgboost as xgb

# scale_pos_weight SEMPRE dinâmico, nunca hardcoded
scale_pos_weight = (1 - y_train.mean()) / y_train.mean()

model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    early_stopping_rounds=30,
    random_state=42,
)
```

## Validação — sempre walk-forward

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=4)
# X deve estar ordenado por data antes deste loop
for train_idx, val_idx in tscv.split(X):
    assert train_idx.max() < val_idx.min(), "DATA LEAKAGE DETECTADO"
```

## Threshold por custo de negócio (não por F1)

```python
import numpy as np

# Custo de perder um churner (~50€) >> enviar cupom inútil (~5€)
# Por isso o threshold ótimo fica abaixo de 0.5
costs = []
thresholds = np.arange(0.05, 0.95, 0.01)
for t in thresholds:
    fn = ((y_proba < t) & (y_true == 1)).sum()
    fp = ((y_proba >= t) & (y_true == 0)).sum()
    costs.append(fn * 50 + fp * 5)
optimal_threshold = thresholds[np.argmin(costs)]
```

## Métricas mínimas

| Métrica | Mínimo | Objetivo |
|---|---|---|
| AUC-ROC | > 0.75 | > 0.82 |
| Lift @Decil 1 | > 2x | > 3x |
| Recall @threshold | > 0.55 | > 0.65 |

Se AUC < 0.75 → revisar features antes de mudar o modelo.