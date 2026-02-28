# Assets — Chart Exports

Place generated PNG exports here before pushing to GitHub Pages.

| Fichier | Généré par | Contenu |
|---------|-----------|---------|
| `shap_global_importance.png` | `notebooks/02_model.py` | SHAP feature importance bar chart |
| `churn_score_distribution.png` | `notebooks/01_eda.py` | Score distribution by CLV segment |
| `clv_segment_pie.png` | `notebooks/01_eda.py` | Proportion Low / Mid / High |

## Générer les assets

```bash
uv run marimo run notebooks/01_eda.py
uv run marimo run notebooks/02_model.py
```

Once generated, update the `<img src="docs/assets...">` placeholders in `docs/project-churn.html`.
