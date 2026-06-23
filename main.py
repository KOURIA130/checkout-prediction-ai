from __future__ import annotations

from src.data_generation import CheckoutConfig, generate_checkout_data
from src.features import build_features
from src.model import train_models
from src.scoring import add_congestion_scoring, evaluate_and_save


def main() -> None:
    cfg = CheckoutConfig()

    df = generate_checkout_data(cfg)
    df = build_features(df)
    df, metrics = train_models(df)
    df = add_congestion_scoring(df)

    evaluate_and_save(
        df=df,
        metrics=metrics,
        output_dir=cfg.output_dir,
        output_file=cfg.output_file,
    )


if __name__ == "__main__":
    main()