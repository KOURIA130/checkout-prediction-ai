from __future__ import annotations

import os
import numpy as np
import pandas as pd


def add_congestion_scoring(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["congestion_score"] = (
        0.5 * result["pred_transaction_duration_sec"]
        + 8 * result["nb_help_requests"]
        + 4 * result["nb_scans_retries"]
        + 0.6 * result["nb_items"]
    )

    # normalisation 0-100
    min_s = result["congestion_score"].min()
    max_s = result["congestion_score"].max()
    if max_s - min_s == 0:
        result["congestion_score_norm"] = 0.0
    else:
        result["congestion_score_norm"] = (
            (result["congestion_score"] - min_s) / (max_s - min_s) * 100
        ).round(2)

    def congestion_level(score: float) -> str:
        if score >= 75:
            return "critical"
        if score >= 55:
            return "high"
        if score >= 35:
            return "medium"
        return "low"

    def recommendation(score: float, abandon_proba: float) -> str:
        if score >= 75 or abandon_proba >= 70:
            return "OPEN_SUPPORT"
        if score >= 55 or abandon_proba >= 50:
            return "MONITOR_QUEUE"
        if score >= 35:
            return "WATCH"
        return "OK"

    result["congestion_level"] = result["congestion_score_norm"].apply(congestion_level)
    result["recommendation"] = result.apply(
        lambda row: recommendation(row["congestion_score_norm"], row["pred_abandon_proba"]),
        axis=1,
    )

    return result


def evaluate_and_save(df: pd.DataFrame, metrics: dict, output_dir: str, output_file: str) -> None:
    print("\n====== CHECKOUT PREDICTION RESULTS ======")
    print(f"Transactions: {len(df)}")
    print(f"Duration MAE: {metrics['duration_mae']} sec")
    print("\nAbandon classification report:")
    print(metrics["classification_report"])

    print("\nCongestion levels:")
    print(df["congestion_level"].value_counts())

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")