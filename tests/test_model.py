import pandas as pd
import numpy as np

from src.model import train_churn_model, optimize_threshold, calculate_metrics


def test_train_churn_model():
    X = pd.DataFrame(
        {
            "feat1": [1, 2, 3, 4, 5, 6, 7, 8],
            "feat2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    # Very small data usually fails TimeSeriesSplit or XGBoost without adjusting,
    # but we just want to test if it runs without crashing for mocking purposes.
    # evaluate_cv=False to avoid Split error on 8 rows.
    model = train_churn_model(X, y, evaluate_cv=False)
    assert model is not None


def test_optimize_threshold():
    y_true = pd.Series([0, 0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.6, 0.4])

    threshold, costs = optimize_threshold(y_true, y_proba)
    assert 0.05 <= threshold <= 0.95
    assert len(costs) > 0


def test_calculate_metrics():
    y_true = pd.Series([0, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.6, 0.4, 0.1, 0.1, 0.8, 0.7])

    metrics = calculate_metrics(y_true, y_proba, threshold=0.5)

    assert "auc" in metrics
    assert "lift_d1" in metrics
    assert "recall" in metrics
    assert metrics["auc"] > 0
