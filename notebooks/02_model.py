import marimo

__generated_with = "0.10.19"
app = marimo.App(width="full")

@app.cell
def __():
    import marimo as mo
    import shap
    import pandas as pd
    return mo, shap, pd

@app.cell
def __(mo):
    mo.md(
        \"\"\"
        # 🤖 Model Training & SHAP Explanations
        
        This notebook demonstrates the XGBoost model training process, including walk-forward validation to prevent data leakage, threshold optimization, and SHAP value generation for explainability.
        \"\"\"
    )
    return

if __name__ == "__main__":
    app.run()
