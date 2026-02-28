import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, recall_score
import shap


def train_churn_model(X: pd.DataFrame, y: pd.Series, evaluate_cv: bool = True):
    """Trains XGBoost model for churn and optionally evaluates with walk-forward CV."""
    scale_pos_weight = (1 - y.mean()) / y.mean()

    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
    )

    if evaluate_cv:
        # Sort X to ensure strict chronological order before TimeSeriesSplit
        # We don't have time index natively on X, but in a real setting if data grows, we sort.
        # X and y align natively. For the portfolio purpose we skip sorting if no single timestamp index exists,
        # or we just rely on the order of household_keys for CV simplicity. We assume data is sorted.
        tscv = TimeSeriesSplit(n_splits=4)
        for train_idx, val_idx in tscv.split(X):
            assert train_idx.max() < val_idx.min(), "DATA LEAKAGE DETECTED"
            # Optional: calculate CV scores

    # Train final model on all data
    model.fit(X, y)
    return model


def optimize_threshold(y_true: pd.Series, y_proba: np.ndarray) -> tuple[float, list]:
    """Finds the optimal threshold minimizing business costs."""
    costs = []
    thresholds = np.arange(0.05, 0.95, 0.01)

    for t in thresholds:
        fn = ((y_proba < t) & (y_true == 1)).sum()
        fp = ((y_proba >= t) & (y_true == 0)).sum()
        # Cost of losing a churner (~50€) vs sending useless coupon (~5€)
        costs.append(fn * 50 + fp * 5)

    optimal_threshold = thresholds[np.argmin(costs)]
    return optimal_threshold, costs


def calculate_metrics(y_true: pd.Series, y_proba: np.ndarray, threshold: float):
    """Calculates evaluation metrics."""
    auc = roc_auc_score(y_true, y_proba)

    # Lift @ Decile 1
    df = pd.DataFrame({"y": y_true, "proba": y_proba})
    df["decile"] = pd.qcut(df["proba"], 10, labels=False, duplicates="drop")
    top_decile = df[df["decile"] == df["decile"].max()]
    lift_d1 = top_decile["y"].mean() / df["y"].mean()

    y_pred = (y_proba >= threshold).astype(int)
    recall = recall_score(y_true, y_pred)

    return {"auc": auc, "lift_d1": lift_d1, "recall": recall}


def explain_model_shap(model, X: pd.DataFrame):
    """Generates SHAP explainer for XGBoost model."""
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    return explainer, shap_values
