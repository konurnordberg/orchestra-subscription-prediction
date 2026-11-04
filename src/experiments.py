"""Controlled CatBoost feature-ablation experiments.

Run from the repository root after generating data/processed/*_features.csv:

    python -m src.experiments
    python -m src.experiments --seeds 42 123 2026

Every candidate uses identical CatBoost hyperparameters and the same
StratifiedKFold partitions for a given seed. Only input features vary.
Kaggle test labels and leaderboard feedback are never used in experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .evaluate import IDENTIFIER, TARGET, cross_validate_model
from .train import load_feature_tables


DEFAULT_PROCESSED = Path("data/processed")
DEFAULT_OUTPUT = DEFAULT_PROCESSED / "ablation"
BASELINE = "all_features"

# Residence-related columns only. sub_last_location is deliberately grouped
# with the customer's subscription behavior (venue/package preference).
GEOGRAPHY_COLUMNS = frozenset({
    "billing_city", "billing_zip", "zip_state", "zip_lat", "zip_long",
})

EXPERIMENT_DESCRIPTIONS = {
    "all_features": "Every engineered predictive feature (reference)",
    "no_geography": "Remove residential ZIP, city, state, and coordinates",
    "no_subscriptions": "Remove all subscription-history features",
    "no_donations": "Remove historical donation features",
    "no_tickets": "Remove all ticket-purchase history features",
    "subscriptions_only": "Subscription-history features alone",
    "geography_only": "Residential geography features alone",
    "subscriptions_plus_geography": "Subscription history plus geography",
}


def feature_group(column: str) -> str:
    """Assign each input column to one mutually exclusive feature group."""

    if column in GEOGRAPHY_COLUMNS:
        return "geography"
    if column.startswith("sub_") or column == "has_subscription_history":
        return "subscriptions"
    if column.startswith("ticket_") or column == "has_ticket_history":
        return "tickets"
    if column.startswith("donation_") or column.startswith("donated_"):
        return "donations"
    return "other"


def build_feature_sets(columns: list[str]) -> dict[str, list[str]]:
    """Specify each experiment by the features retained, in original order."""

    if len(columns) != len(set(columns)):
        raise ValueError("Duplicate feature column names are not supported.")
    if not columns or IDENTIFIER in columns or TARGET in columns:
        raise ValueError("Pass predictive columns without account ID or label.")

    groups = {column: feature_group(column) for column in columns}
    predicates = {
        "all_features": lambda group: True,
        "no_geography": lambda group: group != "geography",
        "no_subscriptions": lambda group: group != "subscriptions",
        "no_donations": lambda group: group != "donations",
        "no_tickets": lambda group: group != "tickets",
        "subscriptions_only": lambda group: group == "subscriptions",
        "geography_only": lambda group: group == "geography",
        "subscriptions_plus_geography": lambda group: group in {
            "subscriptions", "geography"
        },
    }
    return {
        name: [col for col in columns if keep(groups[col])]
        for name, keep in predicates.items()
    }


def run_ablations(
    processed_dir: Path = DEFAULT_PROCESSED,
    output_dir: Path = DEFAULT_OUTPUT,
    experiments: list[str] | None = None,
    seeds: list[int] | None = None,
    folds: int = 5,
) -> pd.DataFrame:
    """Run paired CV comparisons and save reports without altering submissions.

    Returns one summary row per ablation. A positive delta_auc_vs_full means
    the experiment outperformed all_features; negative means a score loss.
    """

    train, test = load_feature_tables(processed_dir)
    if TARGET in test.columns:
        raise ValueError("Test features must not contain the target column.")

    features = [c for c in train.columns if c not in (IDENTIFIER, TARGET)]
    feature_sets = build_feature_sets(features)
    if [c for c in test.columns if c != IDENTIFIER] != features:
        raise ValueError("Training/test feature columns are not aligned.")

    experiments = list(EXPERIMENT_DESCRIPTIONS) if experiments is None else list(experiments)
    seeds = [42] if seeds is None else list(seeds)
    if not experiments or len(set(experiments)) != len(experiments):
        raise ValueError("Provide one or more distinct experiment names.")
    if unknown := set(experiments) - set(EXPERIMENT_DESCRIPTIONS):
        raise ValueError(f"Unknown experiments: {sorted(unknown)}")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Provide one or more distinct seeds.")
    if folds < 2:
        raise ValueError("At least two folds are required.")

    # Always include the full-feature reference even for a subset request.
    if BASELINE not in experiments:
        experiments.insert(0, BASELINE)
    else:
        experiments.remove(BASELINE)
        experiments.insert(0, BASELINE)

    for name in experiments:
        if not feature_sets[name]:
            raise ValueError(f"{name} has no usable features in these tables.")

    output_dir.mkdir(parents=True, exist_ok=True)
    all_runs: list[dict] = []
    all_folds: list[pd.DataFrame] = []
    all_oof = train[[IDENTIFIER, TARGET]].copy()

    for seed in seeds:
        baseline_auc: float | None = None
        for name in experiments:
            cols = feature_sets[name]
            print(f"Seed {seed} | {name} | {len(cols)} features | {folds} folds", flush=True)
            reduced_train = train[[IDENTIFIER, TARGET, *cols]]
            reduced_test = test[[IDENTIFIER, *cols]]

            # The same evaluator also powers src/train.py. For this seed,
            # validation accounts and model settings match across experiments.
            result = cross_validate_model(
                reduced_train, reduced_test,
                model_name="catboost", n_splits=folds, seed=seed,
            )
            aucs = result.fold_metrics["roc_auc"]
            if name == BASELINE:
                baseline_auc = result.oof_auc
            assert baseline_auc is not None

            all_runs.append({
                "experiment": name,
                "seed": seed,
                "n_features": len(cols),
                "oof_auc": result.oof_auc,
                "oof_average_precision": result.oof_average_precision,
                "mean_fold_auc": float(aucs.mean()),
                "std_fold_auc": float(aucs.std(ddof=1)),
                "delta_auc_vs_full": result.oof_auc - baseline_auc,
            })

            fold_table = result.fold_metrics.copy()
            fold_table.insert(0, "seed", seed)
            fold_table.insert(0, "experiment", name)
            all_folds.append(fold_table)
            all_oof[f"pred_{name}_seed_{seed}"] = result.oof_probabilities
            print(f"  OOF AUC {result.oof_auc:.5f} | delta {result.oof_auc - baseline_auc:+.5f}")

    runs = pd.DataFrame(all_runs)
    summary = (
        runs.groupby("experiment", sort=False)
        .agg(
            n_features=("n_features", "first"),
            n_seeds=("seed", "nunique"),
            mean_oof_auc=("oof_auc", "mean"),
            std_oof_auc=("oof_auc", "std"),
            mean_oof_average_precision=("oof_average_precision", "mean"),
            delta_auc_vs_full=("delta_auc_vs_full", "mean"),
        )
        .reset_index()
    )
    summary["std_oof_auc"] = summary["std_oof_auc"].fillna(0.0)
    summary.insert(
        1, "description",
        summary["experiment"].map(EXPERIMENT_DESCRIPTIONS),
    )
    summary = summary.sort_values("mean_oof_auc", ascending=False).reset_index(drop=True)

    groups = pd.DataFrame({
        "feature": features,
        "group": [feature_group(col) for col in features],
    })

    runs.to_csv(output_dir / "ablation_runs.csv", index=False)
    summary.to_csv(output_dir / "ablation_summary.csv", index=False)
    pd.concat(all_folds, ignore_index=True).to_csv(
        output_dir / "ablation_folds.csv", index=False
    )
    all_oof.to_csv(output_dir / "ablation_oof_predictions.csv", index=False)
    groups.to_csv(output_dir / "ablation_feature_groups.csv", index=False)
    config = {
        "model": "catboost",
        "reference_experiment": BASELINE,
        "validation_method": "StratifiedKFold, shuffled, paired by seed",
        "folds": folds,
        "seeds": seeds,
        "experiments": {
            name: {
                "description": EXPERIMENT_DESCRIPTIONS[name],
                "features": feature_sets[name],
            }
            for name in experiments
        },
        "training_rows": len(train),
        "positive_training_labels": int(train[TARGET].sum()),
        "note": "No Kaggle submission is generated or overwritten.",
    }
    (output_dir / "ablation_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print(f"\nReports saved in: {output_dir}")
    print(summary.to_string(index=False, float_format=lambda n: f"{n:.5f}"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument(
        "--experiments", nargs="+", choices=list(EXPERIMENT_DESCRIPTIONS),
        default=None,
        help="Optional subset; all_features reference is always included.",
    )
    args = parser.parse_args()
    run_ablations(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        experiments=args.experiments,
        seeds=args.seeds,
        folds=args.folds,
    )


if __name__ == "__main__":
    main()
