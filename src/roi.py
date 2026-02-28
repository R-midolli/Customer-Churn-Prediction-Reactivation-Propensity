import pandas as pd
import numpy as np
from datetime import datetime


def segment_clv(spend: pd.Series) -> pd.Series:
    """Segments customers based on historical spend percentiles."""
    p40 = spend.quantile(0.40)
    p75 = spend.quantile(0.75)

    return pd.cut(
        spend, bins=[-np.inf, p40, p75, np.inf], labels=["Low", "Mid", "High"]
    )


def best_coupon(clv: float, segment: str) -> tuple[int | None, float]:
    """Returns the optimal coupon value and its expected ROI. Returns (None, ROI) if negative."""
    RESPONSE_RATES = {
        "Low": {5: 0.05, 10: 0.09, 20: 0.14},
        "Mid": {5: 0.10, 10: 0.16, 20: 0.24},
        "High": {5: 0.15, 10: 0.23, 20: 0.35},
    }

    best, best_roi = None, -np.inf

    if segment not in RESPONSE_RATES:
        return None, best_roi

    for coupon in [5, 10, 20]:
        # ROI formula given: clv * 0.5 * response_rate - coupon cost
        roi = clv * 0.5 * RESPONSE_RATES[segment][coupon] - coupon
        if roi > best_roi:
            best_roi, best = roi, coupon

    if best_roi > 0:
        return best, round(float(best_roi), 2)
    return None, round(float(best_roi), 2)


def generate_crm_output(
    household_ids: pd.Series,
    churn_scores: pd.Series,
    clv_euros: pd.Series,
    clv_segments: pd.Series,
    threshold: float,
) -> pd.DataFrame:
    """Builds the final CRM CSV conforming to the required schema."""

    df = pd.DataFrame(
        {
            "household_key": household_ids,
            "churn_score": churn_scores,
            "clv_eur": clv_euros,
            "clv_segment": clv_segments,
        }
    )

    df["recommended_coupon_eur"] = df.apply(
        lambda row: best_coupon(row["clv_eur"], row["clv_segment"])[0], axis=1
    )
    df["expected_roi_eur"] = df.apply(
        lambda row: best_coupon(row["clv_eur"], row["clv_segment"])[1], axis=1
    )

    def determine_action(row):
        if row["churn_score"] < threshold:
            return "Do not contact"
        if row["expected_roi_eur"] > 50:  # Example logic for priority
            return "Priority contact"
        return "Contact"

    df["action"] = df.apply(determine_action, axis=1)

    today_str = datetime.now().strftime("%Y%m%d")
    df["export_date"] = today_str

    # Required schema
    # household_key | churn_score | clv_eur | clv_segment | recommended_coupon_eur | expected_roi_eur | action | export_date
    df = df[
        [
            "household_key",
            "churn_score",
            "clv_eur",
            "clv_segment",
            "recommended_coupon_eur",
            "expected_roi_eur",
            "action",
            "export_date",
        ]
    ]

    return df
