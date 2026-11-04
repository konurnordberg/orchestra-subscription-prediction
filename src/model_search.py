"""Benchmark, tune, and compare models for the orchestra competition.

From the repository root:

    python -m src.model_search --mode benchmark
    python -m src.model_search --mode tune --models catboost xgboost lightgbm --trials 15
    python -m src.model_search --mode compare --seeds 42 123 2026

The script reads the same labeled training and UNLABELED test feature tables
used earlier. No Kaggle submission is produced or overwritten.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .ensemble import compute_blend, predefined_blends
from .evaluate import IDENTIFIER, TARGET, prepare_features
from .models import BOOSTING_MODELS, MODEL_NAMES, make_model


DATA_ROOT = Path("data/processed")
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "model_search"


@dataclass
class ModelCVResult:
    oof: np.ndarray
    test: np.ndarray
    fold_metrics: pd.DataFrame
    auc: float
    ap: float


def load_feature_dataset(
    dataset: str,
    data_root: Path = DATA_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the original 39-feature or distance-augmented 46-feature tables."""

    if dataset == "baseline":
        path = data_root
    elif dataset == "distance":
        path = data_root / "geography" / "variants" / "distance"
    else:
        raise ValueError("Dataset must be 'baseline' or 'distance'.")

    if not (path / "train_features.csv").exists():
        raise FileNotFoundError(
            f"Missing {path}/train_features.csv. Run src.features first; "
            "for distance run src.geography_experiments first."
        )
    dtypes = {IDENTIFIER: "string", "billing_zip": "string"}
    return (
        pd.read_csv(path / "train_features.csv", dtype=dtypes),
        pd.read_csv(path / "test_features.csv", dtype=dtypes),
    )


def _validate_probabilities(prob: np.ndarray, rows: int) -> None:
    if prob.shape != (rows,):
        raise ValueError(f"Expected {rows} predicted probabilities; got {prob.shape}.")
    if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
        raise ValueError("Model predictions must be finite probabilities in [0,1].")


def cross_validate_candidate(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    categoricals: list[str],
    numerics: list[str],
    model_name: str,
    parameters: dict | None,
    folds: int,
    seed: int,
) -> ModelCVResult:
    """Score every labeled account using a model that never trained on it."""

    if len(np.unique(y)) != 2 or np.bincount(y, minlength=2).min() < folds:
        raise ValueError("Each target class needs at least one example per fold.")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan, dtype=float)
    test_folds = []
    rows = []
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(X, y), 1):
        model = make_model(
            model_name, categoricals, numerics,
            seed=seed + fold - 1, parameters=parameters,
        )
        model.fit(X.iloc[fit_idx], y[fit_idx])
        val_prob = np.asarray(
            model.predict_proba(X.iloc[val_idx])[:, 1], dtype=float
        )
        test_prob = np.asarray(model.predict_proba(X_test)[:, 1], dtype=float)
        _validate_probabilities(val_prob, len(val_idx))
        _validate_probabilities(test_prob, len(X_test))
        oof[val_idx] = val_prob
        test_folds.append(test_prob)
        rows.append({
            "fold": fold,
            "validation_accounts": len(val_idx),
            "validation_buyers": int(y[val_idx].sum()),
            "roc_auc": roc_auc_score(y[val_idx], val_prob),
            "average_precision": average_precision_score(y[val_idx], val_prob),
        })
    _validate_probabilities(oof, len(y))
    return ModelCVResult(
        oof=oof,
        test=np.mean(test_folds, axis=0),
        fold_metrics=pd.DataFrame(rows),
        auc=float(roc_auc_score(y, oof)),
        ap=float(average_precision_score(y, oof)),
    )


def tune_parameters(model_name: str, trial) -> dict:
    """Compact Optuna search spaces to keep CPU experiments manageable."""

    if model_name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 400, 1100, step=100),
            "depth": trial.suggest_int("depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.095, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 400, 1100, step=100),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 20.0, log=True),
        }
    if model_name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 400, 1100, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 31),
            "min_child_samples": trial.suggest_int("min_child_samples", 15, 80),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 20.0, log=True),
        }
    raise ValueError("Hyperparameter tuning is supported for CatBoost, XGBoost and LightGBM.")


def run_tuning(
    X: pd.DataFrame,
    y: np.ndarray,
    cats: list[str],
    nums: list[str],
    model_names: list[str],
    trials: int,
    cv_folds: int,
    output_dir: Path,
    dataset: str,
) -> None:
    """Tune with inner CV on known training labels; never inspect test labels."""

    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Optuna missing; pip install optuna") from exc
    if trials < 1 or cv_folds < 2:
        raise ValueError("Tuning requires positive trials and >=2 folds.")

    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_seed = 1701
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=fixed_seed)
    indices = list(splitter.split(X, y))

    for name in model_names:
        if name not in BOOSTING_MODELS:
            raise ValueError(f"Optuna tuning is only configured for {BOOSTING_MODELS}.")

        print(f"Tuning {name}: {trials} trials, {cv_folds} folds per trial", flush=True)

        def objective(trial):
            params = tune_parameters(name, trial)
            probabilities = np.full(len(y), np.nan)
            for fold_idx, (fit_idx, val_idx) in enumerate(indices):
                model = make_model(
                    name, cats, nums, seed=fixed_seed + fold_idx,
                    parameters=params,
                )
                model.fit(X.iloc[fit_idx], y[fit_idx])
                probabilities[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
            _validate_probabilities(probabilities, len(y))
            return roc_auc_score(y, probabilities)

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=2026)
        )
        study.optimize(objective, n_trials=trials, show_progress_bar=False)
        record = {
            "dataset": dataset,
            "model": name,
            "parameters": study.best_params,
            "best_tuning_oof_auc": float(study.best_value),
            "tuning_folds": cv_folds,
            "tuning_seed": fixed_seed,
            "trials": trials,
            "note": "Tuning AUC is selected-on-CV and is not an unbiased final score.",
        }
        (output_dir / f"{name}_best.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        study.trials_dataframe().to_csv(output_dir / f"{name}_trials.csv", index=False)
        print(f"  Best tuning AUROC: {study.best_value:.5f} | {study.best_params}")


def get_candidates(
    model_names: list[str],
    *,
    mode: str,
    tuning_dir: Path,
) -> dict[str, tuple[str, dict | None]]:
    """Include default candidates, plus tuned candidates if files exist."""

    specs = {}
    for model in model_names:
        specs[f"{model}_default"] = (model, None)
        if mode == "compare" and model in BOOSTING_MODELS:
            path = tuning_dir / f"{model}_best.json"
            if path.exists():
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("model") != model:
                    raise ValueError(f"Mismatched tuning config in {path}")
                specs[f"{model}_tuned"] = (model, record["parameters"])
    return specs


def run_benchmark_or_compare(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_names: list[str],
    seeds: list[int],
    folds: int,
    mode: str,
    dataset: str,
    output_root: Path,
) -> pd.DataFrame:
    """Evaluate candidates with matched splits; save all inspectable results."""

    X, X_test, y, cats, nums = prepare_features(train, test)
    tuning_dir = output_root / "tuning" / dataset
    specs = get_candidates(model_names, mode=mode, tuning_dir=tuning_dir)
    output_dir = output_root / mode / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_table = train[[IDENTIFIER, TARGET]].copy()
    test_table = pd.DataFrame({"ID": test[IDENTIFIER].astype("string")})
    run_rows: list[dict] = []
    fold_rows: list[pd.DataFrame] = []
    blend_definitions = predefined_blends(list(specs)) if mode == "compare" else {}

    for seed in seeds:
        fold_assignment = list(
            StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(X, y)
        )
        by_name_oof: dict[str, np.ndarray] = {}
        by_name_test: dict[str, np.ndarray] = {}
        for candidate, (family, parameters) in specs.items():
            print(f"{mode}: {candidate}, seed={seed}, folds={folds}", flush=True)
            result = cross_validate_candidate(
                X, y, X_test, cats, nums, family,
                parameters=parameters, folds=folds, seed=seed,
            )
            by_name_oof[candidate] = result.oof
            by_name_test[candidate] = result.test
            oof_table[f"pred_{candidate}_seed_{seed}"] = result.oof
            test_table[f"pred_{candidate}_seed_{seed}"] = result.test
            run_rows.append({
                "candidate": candidate,
                "family": family,
                "kind": "single_model",
                "seed": seed,
                "oof_auc": result.auc,
                "oof_average_precision": result.ap,
                "mean_fold_auc": result.fold_metrics["roc_auc"].mean(),
                "std_fold_auc": result.fold_metrics["roc_auc"].std(ddof=1),
            })
            one_folds = result.fold_metrics.copy()
            one_folds.insert(0, "candidate", candidate)
            one_folds.insert(1, "seed", seed)
            fold_rows.append(one_folds)
            print(f"  OOF AUROC={result.auc:.5f} AP={result.ap:.5f}")

        # Fixed blends require no model fitting or label-dependent weights.
        for candidate, (method, sources) in blend_definitions.items():
            oof_prob = compute_blend(method, sources, by_name_oof)
            test_prob = compute_blend(method, sources, by_name_test)
            _validate_probabilities(oof_prob, len(y))
            _validate_probabilities(test_prob, len(test))
            oof_table[f"pred_{candidate}_seed_{seed}"] = oof_prob
            test_table[f"pred_{candidate}_seed_{seed}"] = test_prob
            metrics = pd.DataFrame([
                {
                    "fold": fold_idx,
                    "validation_accounts": len(val_idx),
                    "validation_buyers": int(y[val_idx].sum()),
                    "roc_auc": roc_auc_score(y[val_idx], oof_prob[val_idx]),
                    "average_precision": average_precision_score(
                        y[val_idx], oof_prob[val_idx]
                    ),
                }
                for fold_idx, (_, val_idx) in enumerate(fold_assignment, 1)
            ])
            run_rows.append({
                "candidate": candidate,
                "family": "blend",
                "kind": method,
                "seed": seed,
                "oof_auc": roc_auc_score(y, oof_prob),
                "oof_average_precision": average_precision_score(y, oof_prob),
                "mean_fold_auc": metrics["roc_auc"].mean(),
                "std_fold_auc": metrics["roc_auc"].std(ddof=1),
            })
            metrics.insert(0, "candidate", candidate)
            metrics.insert(1, "seed", seed)
            fold_rows.append(metrics)

    runs = pd.DataFrame(run_rows)
    summary = (
        runs.groupby("candidate", sort=False)
        .agg(
            family=("family", "first"), kind=("kind", "first"),
            n_seeds=("seed", "nunique"), mean_oof_auc=("oof_auc", "mean"),
            std_oof_auc=("oof_auc", "std"),
            mean_oof_average_precision=("oof_average_precision", "mean"),
        ).reset_index()
    )
    summary["std_oof_auc"] = summary["std_oof_auc"].fillna(0.0)
    reference = summary.loc[summary["candidate"] == "catboost_default", "mean_oof_auc"]
    if not reference.empty:
        summary["delta_vs_original_catboost"] = summary["mean_oof_auc"] - reference.iloc[0]
    else:
        summary["delta_vs_original_catboost"] = np.nan
    summary = summary.sort_values("mean_oof_auc", ascending=False).reset_index(drop=True)

    summary.to_csv(output_dir / "summary.csv", index=False)
    runs.to_csv(output_dir / "runs.csv", index=False)
    pd.concat(fold_rows, ignore_index=True).to_csv(output_dir / "folds.csv", index=False)
    oof_table.to_csv(output_dir / "oof_predictions.csv", index=False)
    test_table.to_csv(output_dir / "test_predictions.csv", index=False)
    config = {
        "mode": mode,
        "dataset": dataset,
        "models": model_names,
        "folds": folds,
        "seeds": seeds,
        "candidate_parameters": {
            name: {"family": family, "tuned_overrides": params}
            for name, (family, params) in specs.items()
        },
        "predeclared_blends": {
            name: {"method": method, "source_candidates": sources}
            for name, (method, sources) in blend_definitions.items()
        },
        "training_accounts": len(train),
        "positive_training_labels": int(y.sum()),
        "test_accounts": len(test),
        "note": "No Kaggle submission written; tuned configs were selected on labeled CV data.",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("\n", summary.round(5).to_string(index=False))
    print("Results saved to:", output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["benchmark", "tune", "compare"], default="benchmark")
    parser.add_argument("--dataset", choices=["baseline", "distance"], default="baseline")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--tuning-folds", type=int, default=3)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    models = list(BOOSTING_MODELS) if args.mode == "tune" and args.models is None else (
        list(MODEL_NAMES) if args.models is None else args.models
    )
    if len(models) != len(set(models)):
        raise ValueError("Provide distinct model names only.")
    if args.folds < 2:
        raise ValueError("At least two folds are required.")
    train, test = load_feature_dataset(args.dataset, args.data_root)

    if args.mode == "tune":
        X, _, y, cats, nums = prepare_features(train, test)
        run_tuning(
            X, y, cats, nums, models, args.trials, args.tuning_folds,
            args.output_root / "tuning" / args.dataset,
            dataset=args.dataset,
        )
    else:
        seeds = args.seeds if args.seeds is not None else (
            [42] if args.mode == "benchmark" else [42, 123, 2026]
        )
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError("Provide one or more distinct seeds.")
        run_benchmark_or_compare(
            train, test, models, seeds, args.folds,
            mode=args.mode, dataset=args.dataset,
            output_root=args.output_root,
        )


if __name__ == "__main__":
    main()
