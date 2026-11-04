"""Fast tests for ablation feature selection and report generation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.evaluate import EvaluationResult
from src.experiments import build_feature_sets, feature_group, run_ablations


class FeatureAblationTests(unittest.TestCase):
    def test_feature_groups_keep_location_and_subscriptions_separate(self):
        self.assertEqual(feature_group("billing_zip"), "geography")
        self.assertEqual(feature_group("sub_last_location"), "subscriptions")
        self.assertEqual(feature_group("donated_2013"), "donations")
        self.assertEqual(feature_group("ticket_records"), "tickets")

        features = [
            "billing_zip", "zip_lat", "sub_last_year", "sub_last_location",
            "donation_2013", "ticket_records",
        ]
        sets = build_feature_sets(features)
        self.assertEqual(
            sets["no_geography"],
            ["sub_last_year", "sub_last_location", "donation_2013", "ticket_records"],
        )
        self.assertEqual(sets["subscriptions_only"], ["sub_last_year", "sub_last_location"])
        self.assertEqual(
            sets["subscriptions_plus_geography"],
            ["billing_zip", "zip_lat", "sub_last_year", "sub_last_location"],
        )

    def test_experiment_outputs_and_identical_seed_schedule(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            processed = root / "processed"
            processed.mkdir()
            output = processed / "ablation"

            train = pd.DataFrame({
                "account.id": ["a", "b", "c", "d", "e", "f"],
                "label": [0, 1, 0, 1, 0, 1],
                "billing_zip": ["00001", "00002", "00001", "00002", "00001", "00002"],
                "sub_last_year": [2013, 2012, 2013, 2011, 2013, 2012],
                "donation_2013": [0, 25, 0, 20, 0, 30],
                "ticket_records": [1, 2, 1, 2, 1, 2],
            })
            test = train.drop(columns=["label"]).iloc[:2].copy()
            test["account.id"] = ["g", "h"]
            train.to_csv(processed / "train_features.csv", index=False)
            test.to_csv(processed / "test_features.csv", index=False)

            calls = []

            def evaluate_stub(reduced_train, reduced_test, model_name, n_splits, seed):
                self.assertEqual(model_name, "catboost")
                self.assertEqual(n_splits, 2)
                self.assertEqual(
                    [col for col in reduced_train if col != "label"],
                    reduced_test.columns.tolist(),
                )
                calls.append((seed, tuple(reduced_test.columns)))
                probs = np.where(reduced_train["label"].to_numpy() == 1, 0.8, 0.2)
                return EvaluationResult(
                    model_name="catboost",
                    fold_metrics=pd.DataFrame({"fold": [1, 2], "roc_auc": [1.0, 1.0]}),
                    oof_probabilities=probs,
                    test_probabilities=np.array([0.3, 0.4]),
                    oof_auc=1.0,
                    oof_average_precision=1.0,
                )

            with patch("src.experiments.cross_validate_model", side_effect=evaluate_stub):
                summary = run_ablations(
                    processed_dir=processed,
                    output_dir=output,
                    experiments=["no_geography", "subscriptions_only"],
                    seeds=[42, 123],
                    folds=2,
                )

            self.assertEqual(len(summary), 3)  # includes all_features
            self.assertEqual(len(calls), 6)  # three experiments × two seeds
            self.assertEqual(calls[0][0], 42)
            self.assertEqual(calls[3][0], 123)
            self.assertEqual(calls[0][1][0], "account.id")
            for filename in [
                "ablation_runs.csv", "ablation_summary.csv", "ablation_folds.csv",
                "ablation_oof_predictions.csv", "ablation_feature_groups.csv",
                "ablation_config.json",
            ]:
                self.assertTrue((output / filename).exists(), filename)
            report = pd.read_csv(output / "ablation_oof_predictions.csv")
            self.assertEqual(len(report), 6)
            self.assertIn("pred_no_geography_seed_42", report)
            self.assertIn("pred_all_features_seed_123", report)
            config = json.loads((output / "ablation_config.json").read_text())
            self.assertEqual(config["seeds"], [42, 123])


if __name__ == "__main__":
    unittest.main()
