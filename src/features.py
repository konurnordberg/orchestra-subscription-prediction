
"""Build account-level features for orchestra subscription prediction.

Run from the repository root:
    python -m src.features

Optional geography experiments (leave original processed tables unchanged):
    python -m src.features --geo-mode all --output-dir data/processed/geography/manual

Outputs:
    data/processed/train_features.csv
    data/processed/test_features.csv

The default geography mode is "baseline" and preserves all original features.
"""

from pathlib import Path
import argparse
import warnings

import pandas as pd

from .geography import GEO_MODES, augment_geography


TARGET_YEAR = 2014
MISSING_THRESHOLD = 0.75

FILES = {
    "train": "train.csv",
    "test": "test.csv",
    "account": "account.csv",
    "tickets": "tickets_all.csv",
    "subscriptions": "subscriptions.csv",
    "zipcodes": "zipcodes.csv",
}

ID_COLUMNS = {
    "train": ["account.id"],
    "test": ["ID"],
    "account": ["account.id", "billing.zip.code"],
    "tickets": ["account.id"],
    "subscriptions": ["account.id"],
    "zipcodes": ["Zipcode"],
}


def load_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Read source files, preserving identifiers as strings."""
    data = {}

    for name, filename in FILES.items():
        dtype = {col: "string" for col in ID_COLUMNS[name]}
        path = Path(data_dir) / filename

        try:
            df = pd.read_csv(
                path, dtype=dtype, encoding="utf-8-sig"
            )
        except UnicodeDecodeError:
            df = pd.read_csv(
                path, dtype=dtype, encoding="cp1252"
            )

        data[name] = df

    return data


def normalize_zip(values: pd.Series) -> pd.Series:
    """Extract five-digit ZIP codes while preserving leading zeros."""
    return (
        values.astype("string")
        .str.strip()
        .str.extract(r"(\d{5})", expand=False)
    )


def validate_unique(df: pd.DataFrame, column: str, name: str):
    """Reject duplicate or missing record identifiers."""
    if df[column].isna().any():
        raise ValueError(f"{name} contains missing {column} values.")

    if df[column].duplicated().any():
        raise ValueError(f"{name} contains duplicate {column} values.")


def historical_records(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Retain records from seasons beginning before 2014."""
    result = df.copy()

    result["season_start"] = pd.to_numeric(
        result["season"].astype("string").str[:4],
        errors="coerce",
    )

    future_count = (result["season_start"] >= TARGET_YEAR).sum()
    invalid_count = result["season_start"].isna().sum()

    if future_count or invalid_count:
        warnings.warn(
            f"{name}: excluding {future_count} future-season "
            f"and {invalid_count} unparseable-season records."
        )

    result = result[
        result["season_start"].notna()
        & (result["season_start"] < TARGET_YEAR)
    ].copy()

    result["season_start"] = result["season_start"].astype(int)
    return result


def subscription_streak(years: set[int]) -> int:
    """Count consecutive subscribed seasons ending in 2013-14."""
    streak = 0

    while (TARGET_YEAR - 1 - streak) in years:
        streak += 1

    return streak


def build_subscription_features(
    subscriptions: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate previous subscriptions to one row per account."""
    df = historical_records(subscriptions, "subscriptions")

    df["no.seats"] = pd.to_numeric(
        df["no.seats"], errors="coerce"
    )
    df["subscription_tier"] = pd.to_numeric(
        df["subscription_tier"], errors="coerce"
    )

    grouped = df.groupby("account.id")

    features = grouped.agg(
        sub_records=("season", "size"),
        sub_unique_seasons=("season", "nunique"),
        sub_total_seats=("no.seats", "sum"),
        sub_mean_seats=("no.seats", "mean"),
        sub_mean_tier=("subscription_tier", "mean"),
        sub_first_year=("season_start", "min"),
        sub_last_year=("season_start", "max"),
    )

    features["sub_recency_seasons"] = (
        TARGET_YEAR - features["sub_last_year"]
    )

    features["sub_tenure_span"] = (
        features["sub_last_year"]
        - features["sub_first_year"]
        + 1
    )

    features["sub_consistency"] = (
        features["sub_unique_seasons"]
        / features["sub_tenure_span"]
    )

    features["sub_active_2013"] = (
        features["sub_last_year"] == 2013
    ).astype(int)

    # Count distinct subscribed seasons within recent windows.
    unique_seasons = df[
        ["account.id", "season_start"]
    ].drop_duplicates()

    for window in [2, 3, 5, 10]:
        recent = unique_seasons[
            unique_seasons["season_start"]
            >= TARGET_YEAR - window
        ]

        counts = recent.groupby("account.id").size()

        features[f"sub_seasons_last_{window}"] = (
            counts.reindex(features.index)
            .fillna(0)
            .astype(int)
        )

    # Consecutive renewal streak.
    years_by_account = unique_seasons.groupby(
        "account.id"
    )["season_start"].agg(set)

    features["sub_streak_to_2013"] = (
        years_by_account.apply(subscription_streak)
    )

    # Characteristics of the most recent subscription record.
    latest = (
        df.sort_values(
            ["account.id", "season_start"],
            kind="stable",
        )
        .drop_duplicates("account.id", keep="last")
        .set_index("account.id")
    )

    latest_columns = {
        "package": "sub_last_package",
        "location": "sub_last_location",
        "price.level": "sub_last_price_level",
        "subscription_tier": "sub_last_tier",
    }

    features = features.join(
        latest[list(latest_columns)].rename(
            columns=latest_columns
        )
    )

    return features.reset_index()


def build_ticket_features(
    tickets: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate previous individual ticket purchases."""
    df = historical_records(tickets, "tickets")

    df["no.seats"] = pd.to_numeric(
        df["no.seats"], errors="coerce"
    )

    grouped = df.groupby("account.id")

    features = grouped.agg(
        ticket_records=("season", "size"),
        ticket_unique_seasons=("season", "nunique"),
        ticket_total_seats=("no.seats", "sum"),
        ticket_mean_seats=("no.seats", "mean"),
        ticket_unique_sets=("set", "nunique"),
        ticket_last_year=("season_start", "max"),
    )

    features["ticket_recency_seasons"] = (
        TARGET_YEAR - features["ticket_last_year"]
    )

    for window in [1, 2, 3]:
        recent = df[
            df["season_start"] >= TARGET_YEAR - window
        ]

        counts = recent.groupby("account.id").size()

        features[f"ticket_records_last_{window}"] = (
            counts.reindex(features.index)
            .fillna(0)
            .astype(int)
        )

    return features.reset_index()


def build_account_features(
    accounts: pd.DataFrame,
    zipcodes: pd.DataFrame,
    include_lifetime_donations: bool = False,
    *,
    subscriptions: pd.DataFrame | None = None,
    geo_mode: str = "baseline",
) -> pd.DataFrame:
    """Build account features, optionally adding geographic feature blocks."""
    df = accounts.copy()
    validate_unique(df, "account.id", "account")

    # Use raw columns passing the missingness threshold.
    candidate_columns = [
        "billing.city",
        "shipping.city",
        "shipping.zip.code",
        "relationship",
    ]

    retained = [
        col for col in candidate_columns
        if df[col].isna().mean() <= MISSING_THRESHOLD
    ]

    features = df[["account.id"] + retained].copy()

    features = features.rename(columns={
        "billing.city": "billing_city",
        "shipping.city": "shipping_city",
        "shipping.zip.code": "shipping_zip",
        "relationship": "relationship",
    })

    features["billing_zip"] = normalize_zip(
        df["billing.zip.code"]
    )

    features["donation_2013"] = pd.to_numeric(
        df["amount.donated.2013"], errors="coerce"
    )

    features["donated_2013"] = (
        features["donation_2013"].fillna(0) > 0
    ).astype(int)

    # Lifetime totals may contain post-cutoff activity.
    # Enable only after verifying the source's snapshot date.
    if include_lifetime_donations:
        for original, new in [
            ("amount.donated.lifetime", "donation_lifetime"),
            ("no.donations.lifetime", "donation_count_lifetime"),
        ]:
            features[new] = pd.to_numeric(
                df[original], errors="coerce"
            )

    # Join static geographic information.
    zips = zipcodes.copy()
    zips["Zipcode"] = normalize_zip(zips["Zipcode"])

    zip_columns = {
        "Zipcode": "billing_zip",
        "State": "zip_state",
        "Lat": "zip_lat",
        "Long": "zip_long",
    }

    zips = zips[list(zip_columns)].rename(
        columns=zip_columns
    )

    zips = zips.dropna(subset=["billing_zip"])
    validate_unique(zips, "billing_zip", "zipcodes")

    features = features.merge(
        zips,
        on="billing_zip",
        how="left",
        validate="many_to_one",
    )

    return augment_geography(
        features, accounts, zipcodes, subscriptions, mode=geo_mode
    )


def build_master_datasets(
    data: dict[str, pd.DataFrame],
    include_lifetime_donations: bool = False,
    geo_mode: str = "baseline",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate matching training and test feature tables."""
    train = data["train"].copy()
    test = data["test"].copy()

    validate_unique(train, "account.id", "train")
    validate_unique(test, "ID", "test")

    # Normalize the differing ID column names.
    test = test.rename(columns={"ID": "account.id"})

    train_ids = set(train["account.id"])
    test_ids = set(test["account.id"])

    if train_ids.intersection(test_ids):
        raise ValueError("Train and test account IDs overlap.")

    # Build one row per account across both splits.
    all_ids = pd.concat(
        [train["account.id"], test["account.id"]],
        ignore_index=True,
    )

    master = pd.DataFrame({
        "account.id": all_ids
    })

    feature_tables = [
        build_account_features(
            data["account"],
            data["zipcodes"],
            include_lifetime_donations,
            subscriptions=data["subscriptions"],
            geo_mode=geo_mode,
        ),
        build_subscription_features(data["subscriptions"]),
        build_ticket_features(data["tickets"]),
    ]

    for table in feature_tables:
        validate_unique(table, "account.id", "feature table")

        master = master.merge(
            table,
            on="account.id",
            how="left",
            validate="one_to_one",
        )

    # Zero means no recorded activity for count-based features.
    count_columns = [
        "sub_records",
        "sub_unique_seasons",
        "sub_total_seats",
        "sub_active_2013",
        "sub_streak_to_2013",
        "ticket_records",
        "ticket_unique_seasons",
        "ticket_total_seats",
        "ticket_unique_sets",
        "ticket_records_last_1",
        "ticket_records_last_2",
        "ticket_records_last_3",
        "sub_seasons_last_2",
        "sub_seasons_last_3",
        "sub_seasons_last_5",
        "sub_seasons_last_10",
    ]

    for col in count_columns:
        master[col] = master[col].fillna(0)

    master["has_subscription_history"] = (
        master["sub_records"] > 0
    ).astype(int)

    master["has_ticket_history"] = (
        master["ticket_records"] > 0
    ).astype(int)

    # Keep recency missing when there is no corresponding history.
    # The history indicators distinguish this case.

    train_master = train.merge(
        master,
        on="account.id",
        how="left",
        validate="one_to_one",
    )

    test_master = test.merge(
        master,
        on="account.id",
        how="left",
        validate="one_to_one",
    )

    if len(train_master) != len(train):
        raise ValueError("Training row count changed.")

    if len(test_master) != len(test):
        raise ValueError("Test row count changed.")

    train_features = [
        col for col in train_master
        if col not in ["account.id", "label"]
    ]

    test_features = [
        col for col in test_master
        if col != "account.id"
    ]

    if train_features != test_features:
        raise ValueError("Train and test feature columns differ.")

    return train_master, test_master


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
    )
    parser.add_argument(
        "--include-lifetime-donations",
        action="store_true",
        help="Use only if lifetime donation totals are verified as pre-cutoff.",
    )
    parser.add_argument(
        "--geo-mode", choices=GEO_MODES, default="baseline",
        help="Optional geography feature block (default preserves baseline).",
    )

    args = parser.parse_args()

    data = load_data(args.data_dir)

    train_master, test_master = build_master_datasets(
        data,
        include_lifetime_donations=args.include_lifetime_donations,
        geo_mode=args.geo_mode,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_master.to_csv(
        args.output_dir / "train_features.csv",
        index=False,
    )

    test_master.to_csv(
        args.output_dir / "test_features.csv",
        index=False,
    )

    print(f"Training features: {train_master.shape}")
    print(f"Test features:     {test_master.shape}")
    print(f"Feature count:     {test_master.shape[1] - 1}")
    print(f"Saved to:          {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
