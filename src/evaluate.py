"""Leakage-aware cross-validation for orchestra subscription models.

Every preprocessing step learned from the data is fitted inside the training
portion of each fold. The hidden Kaggle test labels are never accessed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


IDENTIFIER = "account.id"
TARGET = "label"
MISSING_CATEGORY = "__MISSING__"


@dataclass
class EvaluationResult:
    """Predictions and diagnostics from one model's cross-validation run."""

    model_name: str
    fold_metrics: pd.DataFrame
    oof_probabilities: np.ndarray
    test_probabilities: np.ndarray
    oof_auc: float
    oof_average_precision: float
    feature_importance: pd.DataFrame | None = None


def prepare_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str], list[str]]:
    """Validate and align features; never use identifiers or labels as inputs."""

    for required in (IDENTIFIER, TARGET):
        if required not in train:
            raise ValueError(f"Training data is missing {required!r}.")
    if IDENTIFIER not in test:
        raise ValueError(f"Test data is missing {IDENTIFIER!r}.")

    if train[IDENTIFIER].isna().any() or train[IDENTIFIER].duplicated().any():
        raise ValueError("Training account identifiers must be unique and nonmissing.")
    if test[IDENTIFIER].isna().any() or test[IDENTIFIER].duplicated().any():
        raise ValueError("Test account identifiers must be unique and nonmissing.")

    columns = [c for c in train if c not in (IDENTIFIER, TARGET)]
    test_columns = [c for c in test if c != IDENTIFIER]
    if columns != test_columns:
        raise ValueError("Train/test feature columns differ or are differently ordered.")
    if not columns:
        raise ValueError("No predictive features are available.")

    if train[TARGET].isna().any() or not set(train[TARGET].unique()) <= {0, 1}:
        raise ValueError("The target must contain only nonmissing binary labels.")
    y = train[TARGET].to_numpy(dtype=int)

    X = train[columns].copy().reset_index(drop=True)
    X_test = test[columns].copy().reset_index(drop=True)

    categoricals = X.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()
    numerics = [col for col in columns if col not in categoricals]

    for col in categoricals:
        X[col] = X[col].astype("string").fillna(MISSING_CATEGORY).astype(str)
        X_test[col] = X_test[col].astype("string").fillna(MISSING_CATEGORY).astype(str)
    for col in numerics:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce").astype(float)

    return X, X_test, y, categoricals, numerics


def make_logistic_model(categoricals: list[str], numerics: list[str]) -> Pipeline:
    """Linear baseline with fold-local imputation, scaling and encoding."""

    transformers = []
    if numerics:
        numeric_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ])
        transformers.append(("numerical", numeric_pipeline, numerics))
    if categoricals:
        categorical_pipeline = OneHotEncoder(
            handle_unknown="ignore", min_frequency=5
        )
        transformers.append(("categorical", categorical_pipeline, categoricals))

    return Pipeline([
        ("preprocess", ColumnTransformer(transformers)),
        ("classifier", LogisticRegression(
            max_iter=3000, C=0.5, class_weight="balanced"
        )),
    ])


def make_catboost_model(categoricals: list[str], seed: int):
    """Nonlinear baseline with fixed hyperparameters for honest outer-fold AUC."""

    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError(
            "CatBoost is not installed. In your WSL environment, run "
            "`python -m pip install catboost` and try again."
        ) from exc

    return CatBoostClassifier(
        iterations=650,
        learning_rate=0.035,
        depth=5,
        l2_leaf_reg=5.0,
        loss_function="Logloss",
        random_seed=seed,
        cat_features=categoricals,
        verbose=False,
        allow_writing_files=False,
        thread_count=4,
    )


def cross_validate_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_name: str,
    n_splits: int = 5,
    seed: int = 42,
) -> EvaluationResult:
    """Return unbiased outer-fold predictions and a fold-averaged test estimate.

    Model hyperparameters are fixed before cross-validation. In particular, the
    outer validation fold is not used for CatBoost early stopping.
    """

    if model_name not in {"logistic", "catboost"}:
        raise ValueError("model_name must be 'logistic' or 'catboost'.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least two.")

    X, X_test, y, categoricals, numerics = prepare_features(train, test)
    if np.bincount(y, minlength=2).min() < n_splits:
        raise ValueError("Each target class needs at least n_splits examples.")

    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    oof = np.full(len(y), np.nan, dtype=float)
    test_per_fold = np.zeros((n_splits, len(test)), dtype=float)
    rows = []
    importances = []

    for fold_idx, (fit_idx, valid_idx) in enumerate(splitter.split(X, y)):
        X_fit, X_valid = X.iloc[fit_idx], X.iloc[valid_idx]
        y_fit, y_valid = y[fit_idx], y[valid_idx]

        if model_name == "logistic":
            model = make_logistic_model(categoricals, numerics)
        else:
            model = make_catboost_model(categoricals, seed + fold_idx)

        model.fit(X_fit, y_fit)
        valid_prob = model.predict_proba(X_valid)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]

        oof[valid_idx] = valid_prob
        test_per_fold[fold_idx] = test_prob

        rows.append({
            "model": model_name,
            "fold": fold_idx + 1,
            "validation_accounts": len(valid_idx),
            "validation_buyers": int(y_valid.sum()),
            "roc_auc": roc_auc_score(y_valid, valid_prob),
            "average_precision": average_precision_score(y_valid, valid_prob),
        })

        if model_name == "catboost":
            importances.append(model.get_feature_importance())

    if not np.isfinite(oof).all():
        raise RuntimeError("Out-of-fold predictions contain missing values.")

    importance_table = None
    if importances:
        values = np.stack(importances)
        importance_table = pd.DataFrame({
            "feature": X.columns,
            "importance_mean": values.mean(axis=0),
            "importance_std": values.std(axis=0),
        }).sort_values("importance_mean", ascending=False)

    return EvaluationResult(
        model_name=model_name,
        fold_metrics=pd.DataFrame(rows),
        oof_probabilities=oof,
        test_probabilities=test_per_fold.mean(axis=0),
        oof_auc=float(roc_auc_score(y, oof)),
        oof_average_precision=float(average_precision_score(y, oof)),
        feature_importance=importance_table,
    )
