import marimo

__generated_with = "0.10.19"
app = marimo.App(width="full")

@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    return mo, pd, px

@app.cell
def __(mo):
    mo.md(
        \"\"\"
        # 📊 Exploratory Data Analysis (EDA)
        
        This notebook explores the Dunnhumby Complete Journey dataset to understand shopping behaviors, product categories, and baseline churn metrics.
        \"\"\"
    )
    return

@app.cell
def __(mo, pd, px):
    mo.md("## Data Loading / Chargement des Données")
    # For the portfolio, we can just show code or mocked data if volume makes the demo app slow
    # Here we will use a simple sample
    return

if __name__ == "__main__":
    app.run()
