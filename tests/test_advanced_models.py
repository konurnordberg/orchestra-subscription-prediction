"""Small CPU tests; no CatBoost/XGBoost/LightGBM/Torch installs required."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.ensemble import blend_probabilities, blend_ranks, predefined_blends
from src.evaluate import prepare_features
from src.model_search import (
    ModelCVResult, cross_validate_candidate, get_candidates,
    run_benchmark_or_compare,
)


def sample_tables():
    count = 48
    labels = np.array(([0] * 5 + [1]) * 8)
    train = pd.DataFrame({
        "account.id": pd.Series([f"acct{i:03d}" for i in range(count)], dtype="string"),
        "label": labels,
        "sub_recency_seasons": np.where(labels == 1, 1, 5).astype(float),
        "zip_lat": np.linspace(37.4, 38.0, count),
        "billing_zip": pd.Series([f"{94100 + i % 3:05d}" for i in range(count)], dtype="string"),
        "sub_last_package": pd.Series(["Full" if i % 2 else "Quartet" for i in range(count)]),
    })
    train.loc[0, "zip_lat"] = np.nan
    test = train.drop(columns="label").iloc[:8].copy()
    test["account.id"] = [f"hidden{i}" for i in range(8)]
    return train, test


class ModelsTests(unittest.TestCase):
    def test_preprocessing_never_uses_id_or_target(self):
        train, test = sample_tables()
        X, X_test, y, categoricals, _ = prepare_features(train, test)
        self.assertNotIn("account.id", X)
        self.assertNotIn("label", X)
        self.assertEqual(len(y), 48)
        self.assertEqual(X.columns.tolist(), X_test.columns.tolist())
        self.assertIn("billing_zip", categoricals)

    def test_extra_trees_produces_valid_oof_predictions(self):
        train, test = sample_tables()
        X, X_test, y, cat, num = prepare_features(train, test)
        result = cross_validate_candidate(
            X, y, X_test, cat, num, "extra_trees",
            parameters={"n_estimators": 16}, folds=3, seed=42,
        )
        self.assertEqual(len(result.oof), len(train))
        self.assertEqual(len(result.test), len(test))
        self.assertTrue(np.isfinite(result.oof).all())
        self.assertTrue(np.all((result.oof >= 0) & (result.oof <= 1)))
        self.assertGreaterEqual(result.auc, 0.5)

    def test_blends_are_fixed_and_valid(self):
        a = np.array([0.1, 0.8, 0.4, 0.9])
        b = np.array([0.2, 0.7, 0.5, 0.6])
        np.testing.assert_allclose(blend_probabilities([a, b]), (a+b)/2)
        ranks = blend_ranks([a, b])
        self.assertTrue(np.all((ranks >= 0) & (ranks <= 1)))
        definitions = predefined_blends([
            "catboost_tuned", "xgboost_default", "lightgbm_tuned", "mlp_default"
        ])
        self.assertIn("boosting_rank_mean", definitions)
        self.assertEqual(definitions["boosting_prob_mean"][1][0], "catboost_tuned")

    def test_tuned_configs_loaded_only_in_compare_mode(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            (path / "catboost_best.json").write_text(json.dumps({
                "model": "catboost", "parameters": {"depth": 4}
            }))
            benchmark = get_candidates(["catboost"], mode="benchmark", tuning_dir=path)
            compare = get_candidates(["catboost"], mode="compare", tuning_dir=path)
            self.assertEqual(list(benchmark), ["catboost_default"])
            self.assertIn("catboost_tuned", compare)
            self.assertEqual(compare["catboost_tuned"][1]["depth"], 4)

    def test_report_generation_and_ensemble_alignment(self):
        train, test = sample_tables()
        fake_result = None

        def stub_cv(X, y, X_test, categoricals, numerics, model_name, parameters, folds, seed):
            self.assertEqual(model_name in {"catboost", "xgboost", "lightgbm"}, True)
            # Deliberately distinct but aligned prediction rankings.
            p = 0.1 + 0.7 * y.astype(float)
            if model_name == "xgboost":
                p = np.clip(p + np.linspace(0, 0.03, len(p)), 0, 1)
            if model_name == "lightgbm":
                p = np.clip(p + np.linspace(0.02, 0, len(p)), 0, 1)
            rows = pd.DataFrame({
                "fold": list(range(1, folds+1)),
                "roc_auc": [0.9] * folds,
                "average_precision": [0.7] * folds,
            })
            return ModelCVResult(
                oof=p, test=np.linspace(0.2, 0.8, len(X_test)),
                fold_metrics=rows,
                auc=roc_auc_score(y, p),
                ap=average_precision_score(y, p),
            )

        with tempfile.TemporaryDirectory() as d, \
             patch("src.model_search.cross_validate_candidate", side_effect=stub_cv):
            out = Path(d)
            summary = run_benchmark_or_compare(
                train, test, ["catboost", "xgboost", "lightgbm"],
                seeds=[42, 123], folds=2, mode="compare",
                dataset="baseline", output_root=out,
            )
            folder = out / "compare" / "baseline"
            self.assertTrue((folder / "summary.csv").exists())
            self.assertTrue((folder / "oof_predictions.csv").exists())
            self.assertTrue((folder / "test_predictions.csv").exists())
            self.assertTrue((folder / "config.json").exists())
            self.assertIn("boosting_rank_mean", summary["candidate"].tolist())
            oof = pd.read_csv(folder / "oof_predictions.csv")
            test_predictions = pd.read_csv(folder / "test_predictions.csv")
            self.assertEqual(len(oof), len(train))
            self.assertEqual(len(test_predictions), len(test))
            self.assertEqual(test_predictions["ID"].tolist(), test["account.id"].tolist())


if __name__ == "__main__":
    unittest.main()
