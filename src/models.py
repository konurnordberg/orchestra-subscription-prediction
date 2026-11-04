"""Reusable model factories for the orchestra subscription competition.

Model preprocessing is *always* learned inside a CV training fold. Categorical
variables are encoded only for models that cannot use CatBoost's native
categorical support. No account ID or target label is passed into these models.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MODEL_NAMES = (
    "catboost", "xgboost", "lightgbm", "random_forest", "extra_trees", "mlp"
)
BOOSTING_MODELS = ("catboost", "xgboost", "lightgbm")

DEFAULT_PARAMETERS = {
    "catboost": {
        "iterations": 650, "learning_rate": 0.035,
        "depth": 5, "l2_leaf_reg": 5.0,
    },
    "xgboost": {
        "n_estimators": 650, "max_depth": 3, "learning_rate": 0.04,
        "min_child_weight": 4.0, "subsample": 0.85,
        "colsample_bytree": 0.85, "reg_lambda": 5.0,
    },
    "lightgbm": {
        "n_estimators": 650, "learning_rate": 0.035, "num_leaves": 15,
        "min_child_samples": 35, "subsample": 0.85,
        "subsample_freq": 1, "colsample_bytree": 0.85, "reg_lambda": 5.0,
    },
    "random_forest": {"n_estimators": 400, "min_samples_leaf": 2},
    "extra_trees": {"n_estimators": 500, "min_samples_leaf": 2},
    "mlp": {
        "hidden_sizes": [128, 64, 32], "dropout": 0.25,
        "learning_rate": 0.001, "weight_decay": 0.001,
        "max_epochs": 65, "patience": 9, "batch_size": 256,
    },
}


def make_preprocessor(
    categoricals: list[str], numerics: list[str], *, dense: bool = False
) -> ColumnTransformer:
    """Fold-local imputation and one-hot encoding, dense for PyTorch."""

    transformers = []
    if numerics:
        transformers.append((
            "numerical",
            Pipeline([
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]),
            numerics,
        ))
    if categoricals:
        transformers.append((
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore", min_frequency=10,
                sparse_output=not dense,
            ),
            categoricals,
        ))
    return ColumnTransformer(transformers, sparse_threshold=0 if dense else 0.3)


def make_model(
    name: str,
    categoricals: list[str],
    numerics: list[str],
    seed: int,
    parameters: dict | None = None,
):
    """Return a fresh unfitted classifier with consistent predict_proba API."""

    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}.")
    config = deepcopy(DEFAULT_PARAMETERS[name])
    if parameters:
        if unknown := set(parameters) - set(config):
            raise ValueError(f"Unknown {name} parameters: {sorted(unknown)}")
        config.update(parameters)

    if name == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:
            raise ImportError("CatBoost missing; pip install catboost") from exc
        return CatBoostClassifier(
            **config,
            loss_function="Logloss",
            cat_features=categoricals,
            random_seed=seed,
            verbose=False,
            thread_count=4,
            allow_writing_files=False,
        )

    if name == "mlp":
        return TorchTabularMLP(
            categoricals=categoricals, numerics=numerics, seed=seed, **config
        )

    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("XGBoost missing; pip install xgboost") from exc
        estimator = XGBClassifier(
            **config, objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", n_jobs=4, random_state=seed,
        )
    elif name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("LightGBM missing; pip install lightgbm") from exc
        estimator = LGBMClassifier(
            **config, objective="binary", random_state=seed,
            verbosity=-1, n_jobs=4,
        )
    elif name == "random_forest":
        estimator = RandomForestClassifier(
            **config, class_weight="balanced_subsample", random_state=seed,
            n_jobs=4,
        )
    else:
        estimator = ExtraTreesClassifier(
            **config, class_weight="balanced", random_state=seed,
            n_jobs=4,
        )

    return Pipeline([
        ("preprocess", make_preprocessor(categoricals, numerics)),
        ("classifier", estimator),
    ])


class TorchTabularMLP:
    """Small feed-forward neural network with an *inner* early-stop split.

    Numeric scaling, categorical encoding, and early stopping are all trained
    on the outer fold's training accounts only. The outer validation fold is
    never used to select epochs or network weights.
    """

    def __init__(
        self,
        categoricals: list[str],
        numerics: list[str],
        seed: int,
        hidden_sizes: list[int],
        dropout: float = 0.25,
        learning_rate: float = 0.001,
        weight_decay: float = 0.001,
        max_epochs: int = 65,
        patience: int = 9,
        batch_size: int = 256,
    ):
        self.categoricals = categoricals
        self.numerics = numerics
        self.seed = int(seed)
        self.hidden_sizes = list(hidden_sizes)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)

    def fit(self, X, y):
        try:
            import torch
            from torch import nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError as exc:
            raise ImportError("PyTorch missing; pip install torch") from exc

        torch.manual_seed(self.seed)
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.preprocess_ = make_preprocessor(
            self.categoricals, self.numerics, dense=True
        )
        array = np.asarray(self.preprocess_.fit_transform(X), dtype=np.float32)
        labels = np.asarray(y, dtype=np.float32)
        fit_idx, stop_idx = train_test_split(
            np.arange(len(labels)), test_size=0.15,
            stratify=labels.astype(int), random_state=self.seed,
        )

        layers = []
        width = array.shape[1]
        for next_width in self.hidden_sizes:
            layers.extend([nn.Linear(width, next_width), nn.ReLU(), nn.Dropout(self.dropout)])
            width = next_width
        layers.append(nn.Linear(width, 1))
        self.network_ = nn.Sequential(*layers)

        positive_count = float(labels[fit_idx].sum())
        negative_count = float(len(fit_idx) - positive_count)
        positive_weight = min(10.0, (negative_count / max(positive_count, 1.0)) ** 0.5)
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([positive_weight], dtype=torch.float32)
        )
        optimizer = torch.optim.AdamW(
            self.network_.parameters(), lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        data = TensorDataset(
            torch.from_numpy(array[fit_idx]),
            torch.from_numpy(labels[fit_idx, None]),
        )
        loader = DataLoader(
            data, batch_size=self.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        stop_X = torch.from_numpy(array[stop_idx])
        stop_y = torch.from_numpy(labels[stop_idx, None])

        best_loss, patience_left, best_weights = np.inf, self.patience, None
        self.epochs_ = 0
        for epoch in range(self.max_epochs):
            self.network_.train()
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.network_(batch_X), batch_y)
                loss.backward()
                optimizer.step()

            self.network_.eval()
            with torch.no_grad():
                stop_loss = float(loss_fn(self.network_(stop_X), stop_y).item())
            if stop_loss < best_loss - 1e-5:
                best_loss = stop_loss
                best_weights = deepcopy(self.network_.state_dict())
                patience_left = self.patience
            else:
                patience_left -= 1
            self.epochs_ = epoch + 1
            if patience_left <= 0:
                break

        if best_weights is not None:
            self.network_.load_state_dict(best_weights)
        self.network_.eval()
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        import torch

        matrix = np.asarray(self.preprocess_.transform(X), dtype=np.float32)
        with torch.no_grad():
            logits = self.network_(torch.from_numpy(matrix))[:, 0]
            p1 = torch.sigmoid(logits).numpy().astype(float)
        return np.column_stack([1.0 - p1, p1])
