---
name: data
description: >
  Use este skill sempre que precisar carregar dados, configurar credenciais,
  baixar o dataset Dunnhumby via kagglehub, converter datas ou validar arquivos.
---

# SKILL: Dados & Ambiente

## Variáveis de ambiente (.env já existe na raiz)

```python
from dotenv import load_dotenv
import os

load_dotenv()
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")  # rbmidolli
KAGGLE_KEY      = os.getenv("KAGGLE_KEY")
```

## Download do Dunnhumby com kagglehub

```python
import kagglehub, shutil
from pathlib import Path
from dotenv import load_dotenv

def download_dunnhumby(dest: Path = Path("data/raw")) -> Path:
    load_dotenv()
    path = kagglehub.dataset_download("frtgnn/dunnhumby-the-complete-journey")
    dest.mkdir(parents=True, exist_ok=True)
    for f in Path(path).glob("*.csv"):
        shutil.copy(f, dest / f.name)
    print(f"✅ Arquivos em: {dest}")
    return dest
```

## Conversão de datas — CRÍTICO

O campo `DAY` no Dunnhumby é um **inteiro**, não uma data.

```python
import pandas as pd

BASE_DATE = pd.Timestamp("2012-01-01")

def load_transactions(raw_path="data/raw") -> pd.DataFrame:
    df = pd.read_csv(f"{raw_path}/transaction_data.csv")
    df['date'] = BASE_DATE + pd.to_timedelta(df['DAY'] - 1, unit='D')
    return df

# ❌ NUNCA: pd.to_datetime(df['DAY']) — gera datas erradas
```

## Arquivos esperados após download

```
data/raw/
├── transaction_data.csv   ← principal (~275k linhas)
├── hh_demographic.csv
├── coupon.csv
├── coupon_redempt.csv
├── campaign_table.csv
├── campaign_desc.csv
└── product.csv
```

Se algum faltar: rode `download_dunnhumby()` novamente.