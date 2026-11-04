"""Predeclared equal-weight probability and rank ensembles.

No weights are fitted against validation labels. Learning stacking weights on
these same OOF predictions and then reporting the same OOF AUC would be an
optimistic evaluation, so stacking is deliberately left for a later nested
validation stage if the fixed blends justify it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def blend_probabilities(predictions: list[np.ndarray]) -> np.ndarray:
    """Equal-weight average of model class-1 probabilities."""

    array = np.column_stack([np.asarray(p, dtype=float) for p in predictions])
    if not np.isfinite(array).all() or ((array < 0) | (array > 1)).any():
        raise ValueError("All blended predictions must be probabilities in [0, 1].")
    return array.mean(axis=1)


def blend_ranks(predictions: list[np.ndarray]) -> np.ndarray:
    """Average percentile ranks, normalized to the range [0, 1]."""

    array = np.column_stack([np.asarray(p, dtype=float) for p in predictions])
    if not np.isfinite(array).all():
        raise ValueError("Cannot rank-blend nonfinite predictions.")
    if array.shape[0] < 2:
        return np.full(array.shape[0], 0.5)
    ranks = np.column_stack([
        (pd.Series(array[:, i]).rank(method="average").to_numpy() - 1)
        / (array.shape[0] - 1)
        for i in range(array.shape[1])
    ])
    return ranks.mean(axis=1)


def predefined_blends(candidate_names: list[str]) -> dict[str, tuple[str, list[str]]]:
    """Fixed combinations, chosen without inspecting their validation scores."""

    candidate_names = list(candidate_names)
    picks = {}
    for family in ("catboost", "xgboost", "lightgbm"):
        preferred = f"{family}_tuned" if f"{family}_tuned" in candidate_names else f"{family}_default"
        if preferred in candidate_names:
            picks[family] = preferred

    blends = {}
    boosting = list(picks.values())
    if len(boosting) >= 2:
        blends["boosting_prob_mean"] = ("probability", boosting)
        blends["boosting_rank_mean"] = ("rank", boosting)
    if "catboost" in picks and "xgboost" in picks:
        pair = [picks["catboost"], picks["xgboost"]]
        blends["catboost_xgboost_prob_mean"] = ("probability", pair)
    if "catboost" in picks and "lightgbm" in picks:
        pair = [picks["catboost"], picks["lightgbm"]]
        blends["catboost_lightgbm_prob_mean"] = ("probability", pair)
    if boosting and "mlp_default" in candidate_names:
        blends["boosting_plus_mlp_rank_mean"] = (
            "rank", [*boosting, "mlp_default"]
        )
    return blends


def compute_blend(
    method: str,
    sources: list[str],
    predictions: dict[str, np.ndarray],
) -> np.ndarray:
    """Apply a named fixed ensemble to aligned account-level predictions."""

    arrays = [predictions[name] for name in sources]
    if method == "probability":
        return blend_probabilities(arrays)
    if method == "rank":
        return blend_ranks(arrays)
    raise ValueError(f"Unknown blend method {method!r}.")
