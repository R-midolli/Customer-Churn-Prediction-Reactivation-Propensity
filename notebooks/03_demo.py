import marimo

__generated_with = "0.10.19"
app = marimo.App(width="full")

@app.cell
def __():
    import marimo as mo
    return mo,

@app.cell
def __(mo):
    mo.md(
        """
        # 📊 Customer Churn Prediction & Reactivation ROI
        
        ## Business Challenge
        🇫🇷 **FR:** Comment identifier les clients FMCG sur le point de churner et allouer un budget de réactivation de façon rentable ?
        🇬🇧 **EN:** How to identify FMCG customers at risk of churning and allocate reactivation budgets profitably?
        """
    )
    return

@app.cell
def __(mo):
    churn_window = mo.ui.slider(15, 60, step=15, value=30, label="📅 Janela de predição (dias)")
    min_score = mo.ui.slider(0.5, 0.9, step=0.05, value=0.70, label="🎯 Score mínimo")
    budget = mo.ui.number(500, 10000, step=500, value=3000, label="💶 Budget de reativação (€)")
    return budget, churn_window, min_score

@app.cell
def __(budget, churn_window, min_score, mo):
    mo.hstack([churn_window, min_score, budget], justify="start")
    return

@app.cell
def __(budget, min_score, mo):
    import pandas as pd
    import numpy as np
    
    # Mock data for demonstration purposes
    np.random.seed(42)
    n_customers = 500
    df = pd.DataFrame({
        "household_key": np.arange(1, n_customers + 1),
        "churn_score": np.random.beta(2, 5, n_customers),
        "clv_eur": np.random.lognormal(5, 1, n_customers),
    })
    
    # Segment CLV
    p40, p75 = df['clv_eur'].quantile(0.40), df['clv_eur'].quantile(0.75)
    df['clv_segment'] = pd.cut(df['clv_eur'], bins=[-np.inf, p40, p75, np.inf], labels=["Low", "Mid", "High"])
    
    # Apply threshold
    filtered_df = df[df['churn_score'] >= min_score.value].copy()
    
    # Calculate recommended coupon
    rates = {"Low": 0.05, "Mid": 0.16, "High": 0.35} # simplified
    filtered_df['expected_roi_eur'] = filtered_df.apply(lambda r: r['clv_eur'] * 0.5 * rates[r['clv_segment']] - 10, axis=1)
    filtered_df['recommended_coupon_eur'] = 10
    
    # Sort and budget constraints
    filtered_df = filtered_df.sort_values(by='expected_roi_eur', ascending=False)
    filtered_df['cumulative_cost'] = filtered_df['recommended_coupon_eur'].cumsum()
    selected_df = filtered_df[filtered_df['cumulative_cost'] <= budget.value].copy()
    
    # KPIs
    auc_val = 0.84
    lift_val = 3.2
    total_roi = selected_df['expected_roi_eur'].sum()
    n_identified = len(selected_df)
    
    kpis = mo.hstack([
        mo.stat(label="AUC-ROC", value=f"{auc_val:.2f}"),
        mo.stat(label="Lift @Decil 1", value=f"{lift_val:.1f}x"),
        mo.stat(label="Clientes Identificados", value=f"{n_identified}"),
        mo.stat(label="ROI Total Estimado", value=f"€ {total_roi:,.0f}")
    ], justify="space-around")
    
    return df, filtered_df, kpis, n_identified, selected_df, total_roi

@app.cell
def __(kpis):
    kpis
    return

@app.cell
def __(mo, selected_df):
    import plotly.express as px
    top_20 = selected_df.head(20).sort_values(by="expected_roi_eur", ascending=True)
    top_20['household_key_str'] = "HH " + top_20['household_key'].astype(str)
    
    fig = px.bar(
        top_20, 
        x="expected_roi_eur", 
        y="household_key_str", 
        color="clv_segment",
        orientation='h',
        title="Top 20 Clientes Prioritários por ROI Estimado",
        labels={"expected_roi_eur": "ROI (€)", "household_key_str": "Cliente"}
    )
    mo.ui.plotly(fig)
    return fig, top_20

@app.cell
def __(mo, selected_df):
    mo.ui.table(selected_df[['household_key', 'churn_score', 'clv_eur', 'clv_segment', 'recommended_coupon_eur', 'expected_roi_eur']], page_size=10)
    return

@app.cell
def __(mo, selected_df):
    csv_data = selected_df.to_csv(index=False).encode('utf-8')
    mo.download(csv_data, filename="churn_crm.csv", label="📥 Download CRM (.csv)")
    return csv_data,

if __name__ == "__main__":
    app.run()
