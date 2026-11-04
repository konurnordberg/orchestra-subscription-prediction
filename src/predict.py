"""Generate the final orchestra Kaggle submission with a CatBoost CV ensemble.

From the repository root::

    python -m src.predict

This uses the original 39-feature table, the previously selected CatBoost
hyperparameters, and the same five-fold / three-seed validation protocol used
for model selection. Test probabilities are averaged across 15 fitted models.
No Kaggle API calls or uploads occur here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .evaluate import IDENTIFIER, TARGET, cross_validate_model, prepare_features
from .train import load_feature_tables, make_submission


DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_SAMPLE_PATH = Path("data/raw/sample_submission.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed/final")
DEFAULT_SEEDS = (42, 123, 2026)
DEFAULT_FOLDS = 5


def _validate_probabilities(values: np.ndarray, expected: int, name: str) -> None:
    if values.shape != (expected,):
        raise ValueError(f"{name}: expected {expected} probabilities, got {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: contains NaN or infinite values.")
    if ((values < 0) | (values > 1)).any():
        raise ValueError(f"{name}: values must lie in [0, 1].")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_final_submission(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    sample_path: Path = DEFAULT_SAMPLE_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seeds: tuple[int, ...] | list[int] = DEFAULT_SEEDS,
    folds: int = DEFAULT_FOLDS,
) -> dict:
    """Train the 15-model CV ensemble and save diagnostics plus a Kaggle CSV.

    Each training account receives an out-of-fold prediction from a model that
    excluded that account. Test accounts receive the average of the fold-model
    predictions. No train-label information enters a test feature or prediction.
    """

    processed_dir = Path(processed_dir)
    sample_path = Path(sample_path)
    output_dir = Path(output_dir)
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Provide at least one distinct random seed.")
    if folds < 2:
        raise ValueError("At least two folds are required.")

    train_path = processed_dir / "train_features.csv"
    test_path = processed_dir / "test_features.csv"
    train, test = load_feature_tables(processed_dir)
    X, _, y, categoricals, numerics = prepare_features(train, test)
    if len(np.unique(y)) != 2 or np.bincount(y, minlength=2).min() < folds:
        raise ValueError("Both classes must have at least one account per fold.")

    # Fail before any model training if the supplied Kaggle template is wrong.
    make_submission(sample_path, test[IDENTIFIER], np.zeros(len(test), dtype=float))

    oof_table = train[[IDENTIFIER, TARGET]].copy()
    test_table = pd.DataFrame({"ID": test[IDENTIFIER].astype("string")})
    seed_summary: list[dict] = []
    fold_frames: list[pd.DataFrame] = []
    seed_importances: list[pd.DataFrame] = []
    all_oof: list[np.ndarray] = []
    all_test: list[np.ndarray] = []

    for seed in seeds:
        print(f"Training CatBoost: seed={seed}, folds={folds}", flush=True)
        result = cross_validate_model(
            train, test, model_name="catboost", n_splits=folds, seed=seed
        )
        oof_probs = np.asarray(result.oof_probabilities, dtype=float)
        test_probs = np.asarray(result.test_probabilities, dtype=float)
        _validate_probabilities(oof_probs, len(train), f"OOF seed={seed}")
        _validate_probabilities(test_probs, len(test), f"test seed={seed}")

        all_oof.append(oof_probs)
        all_test.append(test_probs)
        oof_table[f"pred_seed_{seed}"] = oof_probs
        test_table[f"pred_seed_{seed}"] = test_probs

        fold_table = result.fold_metrics.copy()
        fold_table.insert(0, "seed", seed)
        fold_frames.append(fold_table)
        seed_summary.append({
            "candidate": f"catboost_seed_{seed}",
            "seed": seed,
            "oof_auc": float(roc_auc_score(y, oof_probs)),
            "oof_average_precision": float(average_precision_score(y, oof_probs)),
        })
        if result.feature_importance is not None:
            values = result.feature_importance[["feature", "importance_mean"]].copy()
            values = values.rename(columns={"importance_mean": "importance"})
            values["seed"] = seed
            seed_importances.append(values)
        print(f"  OOF AUROC: {seed_summary[-1]['oof_auc']:.5f}")

    final_oof = np.mean(np.stack(all_oof), axis=0)
    final_test = np.mean(np.stack(all_test), axis=0)
    _validate_probabilities(final_oof, len(train), "averaged OOF")
    _validate_probabilities(final_test, len(test), "averaged test")
    oof_table["pred_mean"] = final_oof
    test_table["pred_mean"] = final_test

    submission = make_submission(sample_path, test[IDENTIFIER], final_test)
    summary = pd.DataFrame(seed_summary)
    ensemble_metrics = {
        "oof_auc": float(roc_auc_score(y, final_oof)),
        "oof_average_precision": float(average_precision_score(y, final_oof)),
    }
    summary = pd.concat([
        summary,
        pd.DataFrame([{
            "candidate": "catboost_15_model_mean",
            "seed": "ensemble",
            **ensemble_metrics,
        }]),
    ], ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / "catboost_final.csv"
    submission.to_csv(submission_path, index=False, float_format="%.12g")
    oof_table.to_csv(output_dir / "oof_predictions.csv", index=False)
    test_table.to_csv(output_dir / "test_predictions.csv", index=False)
    summary.to_csv(output_dir / "cross_validation_summary.csv", index=False)
    pd.concat(fold_frames, ignore_index=True).to_csv(
        output_dir / "fold_metrics.csv", index=False
    )

    if seed_importances:
        feature_importance = (
            pd.concat(seed_importances, ignore_index=True)
            .groupby("feature", as_index=False)
            .agg(
                importance_mean=("importance", "mean"),
                importance_std=("importance", "std"),
            )
            .sort_values("importance_mean", ascending=False)
        )
        feature_importance.to_csv(output_dir / "feature_importance.csv", index=False)

    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": "catboost_default",
        "model_factory": "src.evaluate.make_catboost_model",
        "dataset": "baseline",
        "folds_per_seed": folds,
        "seeds": list(seeds),
        "trained_models": folds * len(seeds),
        "train_accounts": len(train),
        "positive_train_labels": int(y.sum()),
        "test_accounts": len(test),
        "predictive_features": len(X.columns),
        "categorical_features": categoricals,
        "numeric_feature_count": len(numerics),
        "mean_seed_oof_auc": float(np.mean([r["oof_auc"] for r in seed_summary])),
        "std_seed_oof_auc": float(np.std([r["oof_auc"] for r in seed_summary], ddof=1))
        if len(seed_summary) > 1 else 0.0,
        "averaged_oof_metrics": ensemble_metrics,
        "input_file_sha256": {
            "train_features.csv": _file_sha256(train_path),
            "test_features.csv": _file_sha256(test_path),
            "sample_submission.csv": _file_sha256(sample_path),
        },
        "submission_filename": submission_path.name,
        "submission_columns": submission.columns.tolist(),
        "notes": (
            "Final selection was based on labeled CV; hidden Kaggle labels never accessed. "
            "Each model fits 80% of training rows; test probabilities are averaged "
            "over folds and seeds. No public-leaderboard feedback used."
        ),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print(f"\nMean seed OOF AUROC: {config['mean_seed_oof_auc']:.5f}")
    print(f"Averaged OOF AUROC: {ensemble_metrics['oof_auc']:.5f}")
    print(f"Kaggle submission saved: {submission_path}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--sample-submission", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    args = parser.parse_args()

    generate_final_submission(
        processed_dir=args.processed_dir,
        sample_path=args.sample_submission,
        output_dir=args.output_dir,
        seeds=args.seeds,
        folds=args.folds,
    )


if __name__ == "__main__":
    main()
