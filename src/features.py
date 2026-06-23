from __future__ import annotations

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()

    customer_map = {"regular": 0, "new": 1, "professional": 2}
    payment_map = {"card": 0, "mobile": 1, "gift_card": 2}
    store_map = {"city": 0, "mall": 1, "retail_park": 2, "hyper": 3}

    features["customer_type_encoded"] = features["customer_type"].map(customer_map)
    features["payment_method_encoded"] = features["payment_method"].map(payment_map)
    features["store_format_encoded"] = features["store_format"].map(store_map)

    checkout_freq = features["checkout_id"].value_counts().to_dict()
    store_freq = features["store_id"].value_counts().to_dict()

    features["checkout_tx_count"] = features["checkout_id"].map(checkout_freq)
    features["store_tx_count"] = features["store_id"].map(store_freq)

    features["friction_score_raw"] = (
        features["nb_scans_retries"] * 2
        + features["nb_help_requests"] * 4
        + (1 - features["has_loyalty_id"]) * 1
    )

    return features