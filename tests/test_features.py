import pandas as pd
from src.features import (
    compute_rfm,
    compute_behavioral,
    compute_churn_label,
    build_feature_matrix,
)


def test_features():
    # build a small mock transactions df
    df = pd.DataFrame(
        {
            "household_key": [1, 1, 2, 2, 3],
            "BASKET_ID": [101, 102, 103, 104, 105],
            "DAY": [1, 10, 20, 30, 40],
            "SALES_VALUE": [10.0, 20.0, 30.0, 40.0, 50.0],
            "COUPON_DISC": [0, -2, 0, 0, -5],
            "COUPON_MATCH_DISC": [0, 0, 0, 0, 0],
            "RETAIL_DISC": [0, 0, 0, 0, 0],
            "COMMODITY_DESC": ["A", "B", "A", "C", "A"],
            "PRODUCT_ID": [10, 20, 10, 30, 10],
        }
    )
    # Date logic in tests
    BASE = pd.Timestamp("2012-01-01")
    df["date"] = BASE + pd.to_timedelta(df["DAY"] - 1, unit="D")

    reference_date = pd.Timestamp("2012-03-01")

    # 1. Test RFM
    rfm = compute_rfm(df, reference_date, window_days=90)
    assert "recency_days" in rfm.columns
    assert "avg_basket" in rfm.columns

    # 2. Test Behavioral
    behav = compute_behavioral(df, reference_date, window_days=60)
    assert "category_diversity" in behav.columns
    assert "promo_sensitivity" in behav.columns
    assert "trend_freq_4w" in behav.columns
    assert "inactive_weeks" in behav.columns

    # 3. Test Churn Label
    churn = compute_churn_label(
        df, reference_date=pd.Timestamp("2012-01-15"), label_days=30
    )
    assert churn.name == "churned"

    # 4. Test feature matrix assertions
    # In order to test the assertions, we need build_feature_matrix to load our mock.
    # So we'll run it in the actual environment later, or patch load_transactions.
    pass


def test_build_feature_matrix(monkeypatch):
    import src.features

    # Mock data
    df = pd.DataFrame(
        {
            "household_key": [1, 2],
            "BASKET_ID": [101, 103],
            "DAY": [10, 20],
            "SALES_VALUE": [10.0, 30.0],
            "COUPON_DISC": [0, 0],
            "COUPON_MATCH_DISC": [0, 0],
            "RETAIL_DISC": [0, 0],
            "PRODUCT_ID": [10, 10],
        }
    )
    df["date"] = pd.Timestamp("2012-01-01") + pd.to_timedelta(df["DAY"] - 1, unit="D")

    def mock_read_csv(filepath, **kwargs):
        if "product.csv" in filepath:
            return pd.DataFrame({"PRODUCT_ID": [10, 30], "COMMODITY_DESC": ["A", "A"]})
        return df
        
    monkeypatch.setattr(src.features.pd, "read_csv", mock_read_csv)

    # also mock load_transactions inside build_feature_matrix
    def mock_load():
        return df

    monkeypatch.setattr("src.data.load_transactions", mock_load)

    X, y = build_feature_matrix(pd.Timestamp("2012-02-01"))

    # Since y only computes for the given users, we expect exact alignment and no NaNs
    assert X.isnull().sum().sum() == 0
    assert set(y.unique()) <= {0, 1}
    assert X.index.equals(y.index)
