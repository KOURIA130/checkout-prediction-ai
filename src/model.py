from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, classification_report
from sklearn.model_selection import train_test_split


FEATURE_COLS = [
    "transaction_hour",
    "day_of_week",
    "is_weekend",
    "customer_type_encoded",
    "payment_method_encoded",
    "store_format_encoded",
    "has_loyalty_id",
    "nb_items",
    "nb_scans_retries",
    "nb_help_requests",
    "checkout_tx_count",
    "store_tx_count",
    "friction_score_raw",
]


def train_models(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    X = df[FEATURE_COLS].copy()
    y_duration = df["transaction_duration_sec"].copy()
    y_abandon = df["abandon_risk"].copy()

    X_train, X_test, y_d_train, y_d_test, y_a_train, y_a_test = train_test_split(
        X, y_duration, y_abandon, test_size=0.25, random_state=42
    )

    reg = RandomForestRegressor(
        n_estimators=250,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    reg.fit(X_train, y_d_train)
    clf.fit(X_train, y_a_train)

    pred_duration = reg.predict(X)
    pred_abandon_proba = clf.predict_proba(X)[:, 1]
    pred_abandon = clf.predict(X)

    result = df.copy()
    result["pred_transaction_duration_sec"] = pred_duration.round(1)
    result["pred_abandon_proba"] = (pred_abandon_proba * 100).round(2)
    result["pred_abandon_flag"] = pred_abandon

    metrics = {
        "duration_mae": round(mean_absolute_error(y_d_test, reg.predict(X_test)), 2),
        "classification_report": classification_report(
            y_a_test, clf.predict(X_test), output_dict=False
        ),
    }

    return result, metrics