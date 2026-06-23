from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class CheckoutConfig:
    n_samples: int = 8000
    random_state: int = 42
    output_dir: str = "outputs"
    output_file: str = "outputs/checkout_predictions.csv"


def generate_checkout_data(cfg: CheckoutConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_state)

    stores = [
        {"store_id": "STORE_01", "store_format": "city"},
        {"store_id": "STORE_02", "store_format": "city"},
        {"store_id": "STORE_03", "store_format": "mall"},
        {"store_id": "STORE_04", "store_format": "mall"},
        {"store_id": "STORE_05", "store_format": "retail_park"},
        {"store_id": "STORE_06", "store_format": "retail_park"},
        {"store_id": "STORE_07", "store_format": "hyper"},
        {"store_id": "STORE_08", "store_format": "hyper"},
    ]
    checkout_ids = [f"CHK_{i:03d}" for i in range(1, 41)]
    customer_types = ["regular", "new", "professional"]
    payment_methods = ["card", "mobile", "gift_card"]

    rows = []

    for i in range(cfg.n_samples):
        store = stores[rng.integers(0, len(stores))]
        store_id = store["store_id"]
        store_format = store["store_format"]
        checkout_id = rng.choice(checkout_ids)

        transaction_hour = int(rng.integers(8, 22))
        day_of_week = int(rng.integers(0, 7))
        is_weekend = 1 if day_of_week >= 5 else 0

        customer_type = rng.choice(customer_types, p=[0.65, 0.25, 0.10])
        payment_method = rng.choice(payment_methods, p=[0.75, 0.20, 0.05])

        has_loyalty_id = int(rng.random() < 0.55)
        nb_items = max(1, int(rng.normal(9, 4)))

        if store_format == "hyper":
            nb_items += rng.integers(1, 5)
        elif store_format == "city":
            nb_items -= rng.integers(0, 2)
        nb_items = max(1, nb_items)

        nb_scans_retries = int(rng.poisson(0.8))
        nb_help_requests = int(rng.poisson(0.25))

        peak_factor = 1.0
        if 12 <= transaction_hour <= 14:
            peak_factor = 1.15
        elif 18 <= transaction_hour <= 20:
            peak_factor = 1.25

        customer_factor = 1.0
        if customer_type == "new":
            customer_factor = 1.20
        elif customer_type == "professional":
            customer_factor = 1.10

        loyalty_factor = 0.92 if has_loyalty_id == 1 else 1.0

        payment_factor = 1.0
        if payment_method == "mobile":
            payment_factor = 0.95
        elif payment_method == "gift_card":
            payment_factor = 1.10

        retry_penalty = nb_scans_retries * rng.uniform(4, 8)
        help_penalty = nb_help_requests * rng.uniform(25, 60)

        base_duration = 18 + nb_items * rng.uniform(6.0, 9.5)
        transaction_duration_sec = (
            base_duration * peak_factor * customer_factor * loyalty_factor * payment_factor
            + retry_penalty
            + help_penalty
        )

        transaction_duration_sec = max(12, int(transaction_duration_sec))

        # label abandon/friction simulé
        abandon_score = 0
        if nb_scans_retries >= 3:
            abandon_score += 1
        if nb_help_requests >= 1:
            abandon_score += 1
        if transaction_duration_sec >= 160:
            abandon_score += 1
        if customer_type == "new" and has_loyalty_id == 0:
            abandon_score += 1

        abandon_risk = 1 if abandon_score >= 2 else 0

        rows.append(
            {
                "transaction_id": f"TX_{i+1:06d}",
                "store_id": store_id,
                "store_format": store_format,
                "checkout_id": checkout_id,
                "transaction_hour": transaction_hour,
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "customer_type": customer_type,
                "payment_method": payment_method,
                "has_loyalty_id": has_loyalty_id,
                "nb_items": nb_items,
                "nb_scans_retries": nb_scans_retries,
                "nb_help_requests": nb_help_requests,
                "transaction_duration_sec": transaction_duration_sec,
                "abandon_risk": abandon_risk,
            }
        )

    return pd.DataFrame(rows)