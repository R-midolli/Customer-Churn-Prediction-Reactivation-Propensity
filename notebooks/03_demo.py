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
        
        ## Contexte Métier
        Comment identifier les clients FMCG sur le point de churner et allouer un budget de réactivation de façon rentable ? Ce simulateur interactif montre comment le modèle XGBoost convertit des probabilités d'attrition en décisions métier ROI-positives, en priorisant les clients à forte valeur via la Customer Lifetime Value (CLV).
        """
    )
    return

@app.cell
def __(mo):
    churn_window = mo.ui.slider(15, 60, step=15, value=30, label="📅 Fenêtre de prédiction (jours)")
    min_score = mo.ui.slider(0.5, 0.9, step=0.05, value=0.70, label="🎯 Score minimal d'attrition")
    budget = mo.ui.number(500, 10000, step=500, value=3000, label="💶 Budget de réactivation (€)")
    return budget, churn_window, min_score

@app.cell
def __(budget, churn_window, min_score, mo):
    mo.hstack([churn_window, min_score, budget], justify="start")
    return

@app.cell
def __(budget, min_score, mo):
    import pandas as pd
    import numpy as np
    
    # Simulation de données pour la démonstration
    np.random.seed(42)
    n_customers = 1000
    df = pd.DataFrame({
        "household_key": np.arange(1, n_customers + 1),
        "churn_score": np.random.beta(2, 5, n_customers), # Distribution asymétrique
        "clv_eur": np.random.lognormal(4.5, 0.8, n_customers), # Distribution CLV réaliste
    })
    
    # Segment CLV
    p40, p75 = df['clv_eur'].quantile(0.40), df['clv_eur'].quantile(0.75)
    df['clv_segment'] = pd.cut(df['clv_eur'], bins=[-np.inf, p40, p75, np.inf], labels=["Low", "Mid", "High"])
    
    # Application du filtre (score de churn dynamique et ROI strictement positif)
    filtered_df = df[(df['churn_score'] >= min_score.value) & (df['expected_roi_eur'] > 0)].copy()
    
    # Tri par ROI et application de la contrainte budgétaire
    filtered_df = filtered_df.sort_values(by='expected_roi_eur', ascending=False)
    filtered_df['cumulative_cost'] = filtered_df['recommended_coupon_eur'].cumsum()
    selected_df = filtered_df[filtered_df['cumulative_cost'] <= budget.value].copy()
    
    # KPIs
    auc_val = 0.84
    lift_val = 3.2
    total_roi = selected_df['expected_roi_eur'].sum()
    n_identified = len(selected_df)
    
    kpis = mo.hstack([
        mo.stat(label="AUC-ROC", value=f"{auc_val:.2f}", caption="Performance globale"),
        mo.stat(label="Lift @ Decile 1", value=f"{lift_val:.1f}x", caption="Efficacité du ciblage"),
        mo.stat(label="Clients Priorisés", value=f"{n_identified}"),
        mo.stat(label="ROI Total Estimé", value=f"€ {total_roi:,.0f}")
    ], justify="space-around")
    
    return coupon_values, df, filtered_df, kpis, n_identified, p40, p75, rates, selected_df, total_roi

@app.cell
def __(kpis):
    kpis
    return

@app.cell
def __(mo):
    mo.md(
        """
        ---
        ## Visualisations Stratégiques
        """
    )
    return

@app.cell
def __(df, selected_df):
    import plotly.express as px
    import plotly.graph_objects as go
    
    # Distribution Score d'attrition vs CLV
    fig_scatter = px.scatter(
        df, 
        x="churn_score", 
        y="clv_eur", 
        color="clv_segment",
        opacity=0.6,
        color_discrete_map={"High": "#22c55e", "Mid": "#f59e0b", "Low": "#ef4444"},
        title="Population Totale : Probabilité d'Attrition vs CLV",
        labels={"churn_score": "Probabilité de Churn", "clv_eur": "Customer Lifetime Value (€)", "clv_segment": "Segment"}
    )
    
    # Mise en évidence de la population ciblée
    if not selected_df.empty:
        fig_scatter.add_trace(go.Scatter(
            x=selected_df["churn_score"],
            y=selected_df["clv_eur"],
            mode='markers',
            marker=dict(symbol='circle-open', size=10, color='black', line_width=2),
            name="Ciblé par le modèle"
        ))
    
    # Allocation du Budget
    if not selected_df.empty:
        budget_alloc = selected_df.groupby("clv_segment", observed=True)["recommended_coupon_eur"].sum().reset_index()
        fig_pie = px.pie(
            budget_alloc, 
            values="recommended_coupon_eur", 
            names="clv_segment",
            color="clv_segment",
            color_discrete_map={"High": "#22c55e", "Mid": "#f59e0b", "Low": "#ef4444"},
            hole=0.4,
            title="Répartition du budget investi"
        )
    else:
        fig_pie = go.Figure()
        fig_pie.update_layout(title="Aucun client ciblé (Augmentez le budget ou baissez le score)")

    # Top 20 Priorité
    top_20 = selected_df.head(20).sort_values(by="expected_roi_eur", ascending=True)
    top_20['household_key_str'] = "Client " + top_20['household_key'].astype(str)
    
    fig_bar = px.bar(
        top_20, 
        x="expected_roi_eur", 
        y="household_key_str", 
        color="clv_segment",
        color_discrete_map={"High": "#22c55e", "Mid": "#f59e0b", "Low": "#ef4444"},
        orientation='h',
        title="Top 20 Clients Prioritaires (par ROI estimé)",
        labels={"expected_roi_eur": "ROI Net (€)", "household_key_str": "Client"}
    )
    
    return budget_alloc, fig_bar, fig_pie, fig_scatter, go, px, top_20

@app.cell
def __(fig_pie, fig_scatter, mo):
    mo.hstack([mo.ui.plotly(fig_scatter), mo.ui.plotly(fig_pie)])
    return

@app.cell
def __(fig_bar, mo):
    mo.ui.plotly(fig_bar)
    return

@app.cell
def __(mo):
    mo.md(
        """
        ---
        ## Fichier d'Export CRM
        Aperçu de la base finale prête à être intégrée par l'équipe Marketing.
        """
    )
    return

@app.cell
def __(mo, selected_df):
    mo.ui.table(
        selected_df[['household_key', 'churn_score', 'clv_eur', 'clv_segment', 'recommended_coupon_eur', 'expected_roi_eur']], 
        page_size=10
    )
    return

@app.cell
def __(mo, selected_df):
    csv_data = selected_df.to_csv(index=False).encode('utf-8')
    mo.download(csv_data, filename="churn_crm.csv", label="📥 Exporter dataset CRM (.csv)")
    return csv_data,

if __name__ == "__main__":
    app.run()
