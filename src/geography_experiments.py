"""Compare geography extensions against the unchanged 39-feature baseline.

Example from the repository root:

    python -m src.geography_experiments
    python -m src.geography_experiments --seeds 42 123 2026

This writes separate variant feature tables and CV reports under
data/processed/geography/. It never overwrites the existing baseline files,
models, or Kaggle submission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .evaluate import cross_validate_model
from .features import build_master_datasets, load_data
from .geography import GEO_MODES, VENUE_REFERENCE_ZIPS


BASELINE = "baseline"
DEFAULT_OUTPUT_DIR = Path("data/processed/geography")
EXPERIMENT_DESCRIPTIONS = {
    "baseline": "Original engineered features, no new geography",
    "demographics": "Baseline plus ZIP tax/population/wages features",
    "distance": "Baseline plus distance to historical venue ZIP areas",
    "density": "Baseline plus ZIP-level pre-2014 patron counts and rates",
    "all": "Baseline plus every new geography feature group",
}


def run_geography_experiments(
    raw_dir: Path = Path("data/raw"),
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    modes: list[str] | None = None,
    seeds: list[int] | None = None,
    folds: int = 5,
) -> pd.DataFrame:
    """Build feature variants and evaluate them on paired CV partitions."""

    modes = list(EXPERIMENT_DESCRIPTIONS) if modes is None else list(modes)
    seeds = [42] if seeds is None else list(seeds)
    if len(set(modes)) != len(modes) or not modes:
        raise ValueError("Provide distinct, nonempty experiment modes.")
    if set(modes) - set(GEO_MODES):
        raise ValueError(f"Unknown geography mode(s): {set(modes)-set(GEO_MODES)}")
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Provide distinct, nonempty random seeds.")
    if folds < 2:
        raise ValueError("At least two validation folds are required.")

    if BASELINE in modes:
        modes.remove(BASELINE)
    modes.insert(0, BASELINE)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(raw_dir)
    variants = {}
    feature_manifest = []
    print("Building account-level feature variants...", flush=True)
    for mode in modes:
        train, test = build_master_datasets(data, geo_mode=mode)
        variants[mode] = (train, test)
        variant_dir = output_dir / "variants" / mode
        variant_dir.mkdir(parents=True, exist_ok=True)
        train.to_csv(variant_dir / "train_features.csv", index=False)
        test.to_csv(variant_dir / "test_features.csv", index=False)

        for col in test.columns:
            if col == "account.id":
                continue
            feature_manifest.append({
                "mode": mode,
                "feature": col,
                "is_new_geo_feature": col.startswith("geo_"),
            })
        print(f"  {mode}: {len(test.columns) - 1} features")

    reference_train = variants[BASELINE][0]
    oof = reference_train[["account.id", "label"]].copy()
    runs, fold_frames, importance_frames = [], [], []

    for seed in seeds:
        reference_auc = None
        for mode in modes:
            print(f"CV seed={seed}, mode={mode}", flush=True)
            train, test = variants[mode]
            result = cross_validate_model(
                train, test, model_name="catboost", n_splits=folds, seed=seed
            )
            if mode == BASELINE:
                reference_auc = result.oof_auc
            assert reference_auc is not None
            aucs = result.fold_metrics["roc_auc"]
            runs.append({
                "mode": mode,
                "seed": seed,
                "n_features": len(test.columns) - 1,
                "oof_auc": result.oof_auc,
                "oof_average_precision": result.oof_average_precision,
                "mean_fold_auc": float(aucs.mean()),
                "std_fold_auc": float(aucs.std(ddof=1)),
                "delta_auc_vs_baseline": result.oof_auc - reference_auc,
            })
            folds_table = result.fold_metrics.copy()
            folds_table.insert(0, "mode", mode)
            folds_table.insert(1, "seed", seed)
            fold_frames.append(folds_table)
            oof[f"pred_{mode}_seed_{seed}"] = result.oof_probabilities

            if result.feature_importance is not None:
                importance_table = result.feature_importance.copy()
                importance_table.insert(0, "mode", mode)
                importance_table.insert(1, "seed", seed)
                importance_frames.append(importance_table)
            print(f"  AUROC {result.oof_auc:.5f} ({result.oof_auc-reference_auc:+.5f})")

    runs_df = pd.DataFrame(runs)
    summary = (
        runs_df.groupby("mode", sort=False)
        .agg(
            n_features=("n_features", "first"),
            n_seeds=("seed", "nunique"),
            mean_oof_auc=("oof_auc", "mean"),
            std_oof_auc=("oof_auc", "std"),
            mean_oof_average_precision=("oof_average_precision", "mean"),
            delta_auc_vs_baseline=("delta_auc_vs_baseline", "mean"),
        )
        .reset_index()
    )
    summary["std_oof_auc"] = summary["std_oof_auc"].fillna(0.0)
    summary.insert(1, "description", summary["mode"].map(EXPERIMENT_DESCRIPTIONS))
    summary = summary.sort_values("mean_oof_auc", ascending=False).reset_index(drop=True)

    summary.to_csv(output_dir / "experiment_summary.csv", index=False)
    runs_df.to_csv(output_dir / "experiment_runs.csv", index=False)
    pd.concat(fold_frames, ignore_index=True).to_csv(
        output_dir / "experiment_folds.csv", index=False
    )
    oof.to_csv(output_dir / "experiment_oof_predictions.csv", index=False)
    pd.DataFrame(feature_manifest).to_csv(
        output_dir / "feature_manifest.csv", index=False
    )
    if importance_frames:
        pd.concat(importance_frames, ignore_index=True).to_csv(
            output_dir / "catboost_feature_importance.csv", index=False
        )

    config = {
        "model": "catboost",
        "baseline_feature_count": len(variants[BASELINE][1].columns) - 1,
        "folds": folds,
        "seeds": seeds,
        "modes": modes,
        "venue_reference_zip_centroids": VENUE_REFERENCE_ZIPS,
        "note": "ZIP centroids approximate venue areas, not street locations. No Kaggle submission is written.",
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print("\n", summary.round(5).to_string(index=False))
    print(f"Reports and master feature tables saved in: {output_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--modes", choices=GEO_MODES, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    run_geography_experiments(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        modes=args.modes,
        seeds=args.seeds,
        folds=args.folds,
    )


if __name__ == "__main__":
    main()
