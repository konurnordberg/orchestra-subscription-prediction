"""Tests for venue distances, geography features, and experiment reports."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.evaluate import EvaluationResult
from src.geography import augment_geography, haversine_km
from src.geography_experiments import run_geography_experiments


class GeographyTests(unittest.TestCase):
    def setUp(self):
        self.zips = pd.DataFrame({
            "Zipcode": pd.Series(["94102", "94704", "94301", "94305", "94611"], dtype="string"),
            "Lat": [37.77, 37.86, 37.44, 37.41, 37.81],
            "Long": [-122.41, -122.25, -122.15, -122.17, -122.23],
            "TaxReturnsFiled": [120, 80, 70, 30, 100],
            "EstimatedPopulation": [300, 210, 190, 50, 260],
            "TotalWages": [6_000_000, 4_000_000, 5_000_000, 2_000_000, 6_000_000],
        })
        self.accounts = pd.DataFrame({
            "account.id": pd.Series(["a", "b", "c"], dtype="string"),
            "billing.zip.code": pd.Series(["94102", "94102", "94611"], dtype="string"),
        })
        self.features = pd.DataFrame({
            "account.id": pd.Series(["a", "b", "c"], dtype="string"),
            "billing_zip": pd.Series(["94102", "94102", "94611"], dtype="string"),
            "zip_lat": [37.77, 37.77, 37.81],
            "zip_long": [-122.41, -122.41, -122.23],
        })
        self.subscriptions = pd.DataFrame({
            "account.id": pd.Series(["a", "b", "c"], dtype="string"),
            "season": ["2013-2014", "2012-2013", "2014-2015"],
        })

    def test_distance_and_missing_coordinates(self):
        distance = haversine_km(
            pd.Series([37.77, 37.77, None]),
            pd.Series([-122.41, -122.41, -122.41]),
            37.77, -122.41,
        )
        self.assertAlmostEqual(distance[0], 0.0, places=6)
        self.assertTrue(np.isnan(distance[2]))

    def test_baseline_unchanged_and_distance_zero_at_venue_zip(self):
        result = augment_geography(
            self.features, self.accounts, self.zips, self.subscriptions, mode="baseline"
        )
        pd.testing.assert_frame_equal(self.features, result)

        result = augment_geography(
            self.features, self.accounts, self.zips, self.subscriptions, mode="all"
        )
        self.assertAlmostEqual(result.loc[0, "geo_distance_san_francisco_km"], 0.0)
        self.assertEqual(result.loc[0, "geo_nearest_venue_area"], "san_francisco")
        self.assertAlmostEqual(result.loc[0, "geo_zip_wages_per_return"], 50_000.0)
        self.assertGreater(result.loc[2, "geo_distance_nearest_km"], 0)

    def test_density_leaves_out_focal_account_and_excludes_future_seasons(self):
        result = augment_geography(
            self.features, self.accounts, self.zips, self.subscriptions, mode="density"
        )
        self.assertEqual(result.loc[0, "geo_zip_other_patron_count"], 1)
        self.assertEqual(result.loc[0, "geo_zip_other_active_subscribers"], 0)
        self.assertEqual(result.loc[1, "geo_zip_other_active_subscribers"], 1)
        # Account c has a future-season subscription: it must NOT be counted.
        self.assertEqual(result.loc[2, "geo_zip_other_active_subscribers"], 0)
        self.assertEqual(result.loc[2, "geo_zip_other_patron_count"], 0)


class GeographyExperimentTests(unittest.TestCase):
    def test_reports_do_not_touch_baseline_submission(self):
        with tempfile.TemporaryDirectory() as dirname:
            parent = Path(dirname)
            output = parent / "geography"
            calls = []
            original_submission = parent / "submission.csv"
            original_submission.write_text("ID,Predicted\ntest1,0.2\n")

            def make_variant(_data, geo_mode):
                train = pd.DataFrame({
                    "account.id": pd.Series(["a", "b", "c", "d"], dtype="string"),
                    "label": [0, 1, 0, 1],
                    "sub_last_year": [2012, 2013, 2012, 2013],
                })
                test = train.drop(columns="label").iloc[:1].copy()
                test["account.id"] = "test1"
                if geo_mode != "baseline":
                    train["geo_new"] = [0, 1, 0, 1]
                    test["geo_new"] = [1]
                return train, test

            def cv_stub(train, test, model_name, n_splits, seed):
                calls.append((len(train.columns), seed))
                return EvaluationResult(
                    model_name="catboost",
                    fold_metrics=pd.DataFrame({"fold": [1, 2], "roc_auc": [0.7, 0.8]}),
                    oof_probabilities=np.array([0.1, 0.8, 0.2, 0.9]),
                    test_probabilities=np.array([0.6]),
                    oof_auc=0.75,
                    oof_average_precision=0.7,
                )

            with patch("src.geography_experiments.load_data", return_value={}), \
                 patch("src.geography_experiments.build_master_datasets", side_effect=make_variant), \
                 patch("src.geography_experiments.cross_validate_model", side_effect=cv_stub):
                summary = run_geography_experiments(
                    raw_dir=parent, output_dir=output,
                    modes=["distance", "demographics"], seeds=[42, 123], folds=2,
                )

            self.assertEqual(len(summary), 3)  # baseline automatically added
            self.assertEqual(len(calls), 6)
            self.assertEqual(original_submission.read_text(), "ID,Predicted\ntest1,0.2\n")
            self.assertTrue((output / "variants" / "baseline" / "train_features.csv").exists())
            self.assertTrue((output / "variants" / "distance" / "test_features.csv").exists())
            self.assertTrue((output / "experiment_summary.csv").exists())
            self.assertTrue((output / "experiment_oof_predictions.csv").exists())
            config = json.loads((output / "experiment_config.json").read_text())
            self.assertEqual(config["seeds"], [42, 123])


if __name__ == "__main__":
    unittest.main()
