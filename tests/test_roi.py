import pandas as pd

from src.roi import segment_clv, best_coupon, generate_crm_output


def test_segment_clv():
    spends = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    segments = segment_clv(spends)
    assert len(segments) == 10
    assert set(segments) == {"Low", "Mid", "High"}


def test_best_coupon():
    # Example logic test based on RESPONSE_RATES
    # Low clv = 200: rate 5=0.05, 10=0.09, 20=0.14
    # roi 5 = 200 * 0.5 * 0.05 - 5 = 0
    # roi 10 = 200 * 0.5 * 0.09 - 10 = -1
    coupon, roi = best_coupon(300, "Low")
    assert coupon is not None
    assert roi > 0

    coupon, roi = best_coupon(10, "Low")
    assert coupon is None
    assert roi < 0


def test_generate_crm_output():
    household_ids = pd.Series([1, 2, 3])
    churn_scores = pd.Series([0.9, 0.4, 0.8])
    clv_euros = pd.Series([1000, 200, 500])
    clv_segments = pd.Series(["High", "Low", "Mid"])
    threshold = 0.5

    df = generate_crm_output(
        household_ids, churn_scores, clv_euros, clv_segments, threshold
    )

    assert list(df.columns) == [
        "household_key",
        "churn_score",
        "clv_eur",
        "clv_segment",
        "recommended_coupon_eur",
        "expected_roi_eur",
        "action",
        "export_date",
    ]

    assert df.loc[1, "action"] == "Do not contact"  # row index 1 has score 0.4 < 0.5
