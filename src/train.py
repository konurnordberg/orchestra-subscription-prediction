"""Evaluate candidate models and generate a Kaggle-ready submission.

From the repository root, first run `python -m src.features`, then:
    python -m src.train --models logistic catboost

All generated reports and predictions go to the ignored data/processed folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import cross_validate_model


DEFAULT_PROCESSED = Path("data/processed")
DEFAULT_SAMPLE = Path("data/raw/sample_submission.csv")


def load_feature_tables(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read matching account-level feature datasets."""

    dtypes = {"account.id": "string", "billing_zip": "string"}
    train = pd.read_csv(processed_dir / "train_features.csv", dtype=dtypes)
    test = pd.read_csv(processed_dir / "test_features.csv", dtype=dtypes)
    return train, test


def make_submission(
    sample_path: Path,
    test_ids: pd.Series,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Match probabilities by account ID and preserve Kaggle's exact row order."""

    sample = pd.read_csv(sample_path, dtype={"ID": "string"})
    if sample.columns.tolist() != ["ID", "Predicted"]:
        raise ValueError(
            "Unexpected sample submission format; expected ID,Predicted."
        )

    predictions = pd.DataFrame({
        "ID": test_ids.astype("string").to_numpy(),
        "Predicted": probabilities,
    })

    if predictions["ID"].isna().any() or predictions["ID"].duplicated().any():
        raise ValueError("Test prediction IDs are missing or duplicated.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Submission probabilities must be finite.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Submission probabilities must lie between zero and one.")

    submission = sample[["ID"]].merge(
        predictions, on="ID", how="left", validate="one_to_one", sort=False
    )
    if submission["Predicted"].isna().any():
        raise ValueError("Some sample-submission accounts lack predictions.")
    if len(submission) != len(test_ids) or set(submission["ID"]) != set(test_ids):
        raise ValueError("Submission accounts do not exactly match test accounts.")

    return submission[sample.columns]


def run_training(
    processed_dir: Path,
    sample_path: Path,
    models: list[str],
    folds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Score candidate models, save diagnostics, and select by OOF AUROC."""

    processed_dir.mkdir(parents=True, exist_ok=True)
    train, test = load_feature_tables(processed_dir)

    if not models or len(set(models)) != len(models):
        raise ValueError("Supply one or more distinct model names.")

    oof = train[["account.id", "label"]].copy()
    test_preds = pd.DataFrame({"ID": test["account.id"].astype("string")})
    summaries = []
    fold_frames = []
    model_results = {}

    for name in models:
        print(f"Evaluating {name} with {folds}-fold stratified CV...")
        result = cross_validate_model(
            train, test, model_name=name, n_splits=folds, seed=seed
        )
        model_results[name] = result
        fold_frames.append(result.fold_metrics)
        oof[f"pred_{name}"] = result.oof_probabilities
        test_preds[f"pred_{name}"] = result.test_probabilities

        fold_aucs = result.fold_metrics["roc_auc"]
        summaries.append({
            "model": name,
            "oof_auc": result.oof_auc,
            "mean_fold_auc": float(fold_aucs.mean()),
            "std_fold_auc": float(fold_aucs.std(ddof=1)),
            "oof_average_precision": result.oof_average_precision,
        })

        if result.feature_importance is not None:
            result.feature_importance.to_csv(
                processed_dir / f"{name}_feature_importance.csv", index=False
            )

        print(
            f"  OOF AUROC: {result.oof_auc:.5f} | "
            f"Mean fold AUROC: {fold_aucs.mean():.5f}"
        )

    summary = pd.DataFrame(summaries).sort_values(
        "oof_auc", ascending=False
    ).reset_index(drop=True)
    best_name = str(summary.loc[0, "model"])
    best_predictions = model_results[best_name].test_probabilities

    submission = make_submission(
        sample_path, test["account.id"], best_predictions
    )

    # Files under data/ are ignored by Git, including account-level predictions.
    summary.to_csv(processed_dir / "model_summary.csv", index=False)
    pd.concat(fold_frames, ignore_index=True).to_csv(
        processed_dir / "fold_metrics.csv", index=False
    )
    oof.to_csv(processed_dir / "oof_predictions.csv", index=False)
    test_preds.to_csv(processed_dir / "test_predictions.csv", index=False)
    submission.to_csv(processed_dir / "submission.csv", index=False)

    config = {
        "candidate_models": models,
        "selected_model": best_name,
        "selection_metric": "out-of-fold ROC AUC",
        "folds": folds,
        "random_seed": seed,
        "training_rows": len(train),
        "test_rows": len(test),
        "positive_training_labels": int(train["label"].sum()),
    }
    (processed_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print(f"\nSelected by CV: {best_name}")
    print(f"Kaggle file: {processed_dir / 'submission.csv'}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir", type=Path, default=DEFAULT_PROCESSED
    )
    parser.add_argument(
        "--sample-submission", type=Path, default=DEFAULT_SAMPLE
    )
    parser.add_argument(
        "--models", nargs="+", choices=["logistic", "catboost"],
        default=["logistic", "catboost"],
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = run_training(
        processed_dir=args.processed_dir,
        sample_path=args.sample_submission,
        models=args.models,
        folds=args.folds,
        seed=args.seed,
    )
    print("\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
