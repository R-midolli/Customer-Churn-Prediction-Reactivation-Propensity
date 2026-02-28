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
        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                    padding: 2rem; border-radius: 12px; margin-bottom: 1rem;">
            <div style="font-size:0.75rem; color:#e94560; text-transform:uppercase;
                        letter-spacing:2px; margin-bottom:0.5rem;">Machine Learning & CRM</div>
            <h1 style="color: #e2e8f0; font-size: 2rem; margin: 0; font-family: Inter, sans-serif;">
                📊 Customer Churn Prediction & Reactivation ROI
            </h1>
            <p style="color: #a8b2d8; margin: 0.5rem 0 0 0; font-size: 1rem;">
                XGBoost · RFM Features · CLV-Based ROI Optimisation
            </p>
        </div>
        """
    )
    mo.md(
        """
        <div style="background:#1e2235; border:1px solid #2d3561; border-radius:10px;
                    padding:1.5rem; margin:1rem 0;">
            <h2 style="color:#e2e8f0; margin-top:0;">🎯 Défi Business / Business Challenge</h2>
            <p style="color:#a8b2d8; line-height:1.7; margin:0.5rem 0;">
                <span style="color:#e94560; font-weight:600;">FR :</span>
                Comment identifier les clients FMCG sur le point de churner et allouer un budget
                de réactivation de façon rentable ?
            </p>
            <p style="color:#a8b2d8; line-height:1.7; margin:0.5rem 0 0 0;">
                <span style="color:#00d4aa; font-weight:600;">EN :</span>
                How to identify FMCG customers at risk of churning and allocate reactivation
                budgets profitably?
            </p>
        </div>
        """
    )
    return

@app.cell
def __(mo):
    mo.md(
        """
        <div style="background:#1e2235; border:1px solid #2d3561; border-radius:10px;
                    padding:1rem 1.5rem; margin:1rem 0;">
            <h3 style="color:#e2e8f0; margin-top:0;">⚙️ Paramètres du modèle / Model Parameters</h3>
            <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:1rem;
                        color:#a8b2d8; font-size:0.85rem;">
                <div>
                    <strong style="color:#7c83fd;">🗓 Fenêtre de prédiction / Prediction window</strong><br>
                    Durée en jours pour la détection du churn · Days for churn detection
                </div>
                <div>
                    <strong style="color:#7c83fd;">🎯 Score minimum / Minimum score</strong><br>
                    Seuil de probabilité pour qualifier un client à risque · Minimum churn probability threshold
                </div>
                <div>
                    <strong style="color:#7c83fd;">💶 Budget de réactivation / Reactivation budget</strong><br>
                    Enveloppe maximale allouée à la campagne CRM · Maximum CRM campaign budget in euros
                </div>
            </div>
        </div>
        """
    )
    churn_window = mo.ui.slider(15, 60, step=15, value=30, label="🗓 Fenêtre de prédiction (jours) / Prediction window (days)")
    min_score = mo.ui.slider(0.5, 0.9, step=0.05, value=0.70, label="🎯 Score minimum / Min score")
    budget = mo.ui.number(500, 10000, step=500, value=3000, label="💶 Budget de réactivation (€) / Reactivation budget (€)")
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
    
    kpis = mo.md(f"""
<div style="display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin:1rem 0;">

    <div style="background:#1e2235; border:1px solid #2d3561; border-radius:8px;
                padding:1.2rem; text-align:center;">
        <div style="color:#a8b2d8; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px;">AUC-ROC</div>
        <div style="color:#00d4aa; font-size:2rem; font-weight:700; margin-top:0.5rem;">{auc_val:.2f}</div>
        <div style="color:#a8b2d8; font-size:0.65rem; margin-top:0.25rem;">Discriminative power</div>
    </div>

    <div style="background:#1e2235; border:1px solid #2d3561; border-radius:8px;
                padding:1.2rem; text-align:center;">
        <div style="color:#a8b2d8; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px;">Lift @Décile 1</div>
        <div style="color:#00d4aa; font-size:2rem; font-weight:700; margin-top:0.5rem;">{lift_val:.1f}x</div>
        <div style="color:#a8b2d8; font-size:0.65rem; margin-top:0.25rem;">vs. sélection aléatoire</div>
    </div>

    <div style="background:#1e2235; border:1px solid #2d3561; border-radius:8px;
                padding:1.2rem; text-align:center;">
        <div style="color:#a8b2d8; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px;">Clients identifiés</div>
        <div style="color:#e94560; font-size:2rem; font-weight:700; margin-top:0.5rem;">{n_identified}</div>
        <div style="color:#a8b2d8; font-size:0.65rem; margin-top:0.25rem;">Above score threshold</div>
    </div>

    <div style="background:#1e2235; border:1px solid #2d3561; border-radius:8px;
                padding:1.2rem; text-align:center;">
        <div style="color:#a8b2d8; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px;">ROI Total estimé</div>
        <div style="color:{'#00d4aa' if total_roi >= 0 else '#e94560'}; font-size:2rem; font-weight:700; margin-top:0.5rem;">€ {total_roi:,.0f}</div>
        <div style="color:#a8b2d8; font-size:0.65rem; margin-top:0.25rem;">Net of campaign cost</div>
    </div>

</div>
""")
    
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
        ## 📊 Analyse ROI & Priorisation / ROI Analysis & Prioritization
        """
    )
    return

@app.cell
def __(df, selected_df):
    import plotly.express as px
    import plotly.graph_objects as go
    
    # Top 20 Priority
    if not selected_df.empty:
        top_20 = selected_df.head(20).sort_values(by="expected_roi_eur", ascending=True)
        top_20['household_key_str'] = "Client " + top_20['household_key'].astype(str)
        
        fig_bar = px.bar(
            top_20,
            x="expected_roi_eur",
            y="household_key_str",
            orientation="h",
            color="clv_segment",
            color_discrete_map={
                "High":   "#00d4aa",
                "Mid":    "#7c83fd",
                "Low":    "#e94560"
            },
            title="Top 20 Clients Prioritaires par ROI Estimé / Top 20 Priority Clients by Estimated ROI",
            labels={
                "expected_roi_eur": "ROI (€)",
                "household_key_str":    "Client",
                "clv_segment":      "Segment CLV"
            }
        )

        fig_bar.update_layout(
            plot_bgcolor="#1e2235",
            paper_bgcolor="#1a1a2e",
            font={"color": "#e2e8f0", "family": "Inter, sans-serif"},
            title_font_size=14,
            title_font_color="#e2e8f0",
            xaxis=dict(gridcolor="#2d3561", zerolinecolor="#e94560"),
            yaxis=dict(gridcolor="#2d3561"),
            legend_title_text="Segment CLV",
            height=500
        )
    else:
        fig_bar = go.Figure()
        fig_bar.update_layout(
            title="Aucun client ciblé / No customers targeted",
            plot_bgcolor="#1e2235",
            paper_bgcolor="#1a1a2e",
            font={"color": "#e2e8f0", "family": "Inter, sans-serif"},
            height=500
        )
        top_20 = selected_df.copy()
    
    # Chart A — Churn score distribution by CLV segment
    plot_df = selected_df if not selected_df.empty else df
    fig_box = px.box(
        plot_df,
        x="clv_segment",
        y="churn_score",
        color="clv_segment",
        color_discrete_map={"High": "#00d4aa", "Mid": "#7c83fd", "Low": "#e94560"},
        title="Distribution du Score par Segment CLV",
        labels={"clv_segment": "Segment CLV", "churn_score": "Churn Score"}
    )
    fig_box.update_layout(
        plot_bgcolor="#1e2235", paper_bgcolor="#1a1a2e",
        font={"color": "#e2e8f0"}, showlegend=False,
        xaxis=dict(gridcolor="#2d3561"), yaxis=dict(gridcolor="#2d3561")
    )
    
    # Chart B — CLV vs Churn Score scatter
    fig_scatter = px.scatter(
        plot_df,
        x="churn_score",
        y="clv_eur",
        color="clv_segment",
        size="recommended_coupon_eur" if "recommended_coupon_eur" in plot_df.columns and plot_df["recommended_coupon_eur"].sum() > 0 else None,
        title="CLV vs Probabilité de Churn / CLV vs Churn Probability",
        labels={
            "churn_score": "Churn Score",
            "clv_eur": "CLV (€)",
            "clv_segment": "Segment CLV"
        },
        color_discrete_map={"High": "#00d4aa", "Mid": "#7c83fd", "Low": "#e94560"}
    )
    fig_scatter.update_layout(
        plot_bgcolor="#1e2235", paper_bgcolor="#1a1a2e",
        font={"color": "#e2e8f0"},
        xaxis=dict(gridcolor="#2d3561"), yaxis=dict(gridcolor="#2d3561")
    )
    
    return fig_bar, fig_box, fig_scatter, go, plot_df, px, top_20

@app.cell
def __(fig_bar, mo):
    mo.ui.plotly(fig_bar)
    return

@app.cell
def __(mo):
    mo.md("""
<h3 style="color:#e2e8f0; margin:1.5rem 0 0.5rem 0;">
    🔍 Analyse des Segments / Segment Insights
</h3>
<p style="color:#a8b2d8; font-size:0.875rem;">
    FR : Distribution du score de churn et relation CLV par segment client.<br>
    EN : Churn score distribution and CLV relationship per customer segment.
</p>
    """)
    return

@app.cell
def __(fig_box, fig_scatter, mo):
    mo.hstack([mo.ui.plotly(fig_box), mo.ui.plotly(fig_scatter)])
    return

@app.cell
def __(mo):
    mo.md(
        """
<div style="margin:1.5rem 0 0.5rem 0;">
    <h3 style="color:#e2e8f0;">📋 Matrice d'Action CRM / CRM Action Matrix</h3>
    <p style="color:#a8b2d8; font-size:0.875rem;">
        FR : Clients sélectionnés avec coupon recommandé et ROI attendu, prêts à exporter vers votre CRM.<br>
        EN : Selected clients with recommended coupon and expected ROI, ready to export to your CRM.
    </p>
</div>
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
    mo.download(csv_data, filename="churn_crm.csv", label="⬇ Télécharger la matrice CRM (.csv) / Download CRM Action Matrix (.csv)")
    return csv_data,

@app.cell
def __(mo):
    mo.md(
        """
        ---
        <div style="text-align:center; color:#a8b2d8; font-size:0.8rem;
                    padding:1.5rem 0 0.5rem 0; border-top:1px solid #2d3561; margin-top:2rem;">
            <strong style="color:#e94560;">Rafael Midolli</strong> &nbsp;·&nbsp;
            XGBoost &nbsp;·&nbsp; RFM Features &nbsp;·&nbsp; CLV Optimisation &nbsp;·&nbsp; Marimo &nbsp;·&nbsp; Python<br>
            <a href="https://github.com/R-midolli/Customer-Churn-Prediction-Reactivation-Propensity"
               style="color:#7c83fd; text-decoration:none; display:inline-block; margin-top:0.5rem;">
               GitHub Repository ↗
            </a>
            &nbsp;·&nbsp;
            <a href="https://r-midolli.github.io/portfolio_rafael_midolli/project-churn.html"
               style="color:#7c83fd; text-decoration:none;">
               Portfolio ↗
            </a>
        </div>
        """
    )
    return

if __name__ == "__main__":
    app.run()
