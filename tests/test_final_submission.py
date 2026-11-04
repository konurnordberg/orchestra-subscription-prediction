"""Tests for final CatBoost prediction aggregation and Kaggle CSV formatting."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.predict import generate_final_submission


class FinalSubmissionTests(TestCase):
    def _fixtures(self, root: Path) -> tuple[Path, Path, Path]:
        processed = root / "processed"
        processed.mkdir()
        output = root / "final"
        train = pd.DataFrame({
            "account.id": [f"train-{i}" for i in range(8)],
            "label": [0, 0, 1, 0, 0, 1, 0, 0],
            "sub_count": [0, 0, 2, 0, 0, 1, 0, 0],
            "billing_zip": ["00120", "00121", "00120", "00122", "00123", "00120", "00124", "00125"],
        })
        test = pd.DataFrame({
            "account.id": ["patron-A", "patron-B", "patron-C"],
            "sub_count": [1, 0, 3],
            "billing_zip": ["00120", "00122", "00125"],
        })
        sample = root / "sample_submission.csv"
        # Kaggle's sample order is *intentionally different* from feature-table order.
        pd.DataFrame({
            "ID": ["patron-C", "patron-A", "patron-B"],
            "Predicted": [0.5, 0.5, 0.5],
        }).to_csv(sample, index=False)
        train.to_csv(processed / "train_features.csv", index=False)
        test.to_csv(processed / "test_features.csv", index=False)
        return processed, sample, output

    def test_multi_seed_predictions_align_by_id_and_average(self):
        with TemporaryDirectory() as tmp:
            processed, sample, output = self._fixtures(Path(tmp))

            def fake_cross_validate(train, test, model_name, n_splits, seed):
                self.assertEqual(model_name, "catboost")
                self.assertEqual(n_splits, 2)
                shift = 0.1 if seed == 22 else 0.0
                oof = np.array([0.1, 0.2, 0.9, 0.2, 0.1, 0.8, 0.2, 0.1]) + shift
                test_pred = np.array([0.15, 0.25, 0.35]) + shift
                return SimpleNamespace(
                    oof_probabilities=oof,
                    test_probabilities=test_pred,
                    fold_metrics=pd.DataFrame({
                        "fold": [1, 2], "roc_auc": [1.0, 1.0],
                        "validation_accounts": [4, 4], "validation_buyers": [1, 1],
                        "average_precision": [1.0, 1.0],
                    }),
                    feature_importance=pd.DataFrame({
                        "feature": ["sub_count", "billing_zip"],
                        "importance_mean": [90.0, 10.0],
                        "importance_std": [1.0, 1.0],
                    }),
                )

            with patch("src.predict.cross_validate_model", side_effect=fake_cross_validate) as mocked:
                config = generate_final_submission(
                    processed, sample, output, seeds=[11, 22], folds=2
                )

            self.assertEqual(mocked.call_count, 2)
            self.assertEqual(config["trained_models"], 4)
            self.assertEqual(config["predictive_features"], 2)
            submission = pd.read_csv(output / "catboost_final.csv", dtype={"ID": "string"})
            self.assertEqual(submission.columns.tolist(), ["ID", "Predicted"])
            self.assertEqual(submission["ID"].tolist(), ["patron-C", "patron-A", "patron-B"])
            np.testing.assert_allclose(submission["Predicted"], [0.40, 0.20, 0.30])
            self.assertEqual(len(pd.read_csv(output / "fold_metrics.csv")), 4)
            oof = pd.read_csv(output / "oof_predictions.csv")
            self.assertTrue({"pred_seed_11", "pred_seed_22", "pred_mean"} <= set(oof))
            self.assertEqual(len(pd.read_csv(output / "feature_importance.csv")), 2)

    def test_rejects_invalid_seeds_before_training(self):
        with TemporaryDirectory() as tmp:
            processed, sample, output = self._fixtures(Path(tmp))
            for seeds in ([], [42, 42]):
                with self.assertRaises(ValueError):
                    generate_final_submission(processed, sample, output, seeds=seeds, folds=2)

    def test_rejects_missing_sample_accounts_before_training(self):
        with TemporaryDirectory() as tmp:
            processed, sample, output = self._fixtures(Path(tmp))
            bad_sample = pd.DataFrame({"ID": ["patron-X", "patron-A", "patron-B"], "Predicted": 0.5})
            bad_sample.to_csv(sample, index=False)
            with patch("src.predict.cross_validate_model") as mocked:
                with self.assertRaises(ValueError):
                    generate_final_submission(processed, sample, output, seeds=[42], folds=2)
                mocked.assert_not_called()

    def test_rejects_nonfinite_predictions(self):
        with TemporaryDirectory() as tmp:
            processed, sample, output = self._fixtures(Path(tmp))
            bad_result = SimpleNamespace(
                oof_probabilities=np.array([0.5] * 8),
                test_probabilities=np.array([0.5, float("nan"), 0.5]),
            )
            with patch("src.predict.cross_validate_model", return_value=bad_result):
                with self.assertRaisesRegex(ValueError, "NaN"):
                    generate_final_submission(processed, sample, output, seeds=[42], folds=2)
            self.assertFalse((output / "catboost_final.csv").exists())
