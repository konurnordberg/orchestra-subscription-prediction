"""Optional ZIP demographics, venue-area distance, and patron-density features.

All inputs come from the supplied historical source tables. No 2014-15 labels
or Kaggle results are used to construct the features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


GEO_MODES = ("baseline", "demographics", "distance", "density", "all")

# Historical Philharmonia Baroque Orchestra performance areas, represented by
# the centroid of the venue ZIP in the PROVIDED zipcodes.csv. These are NOT
# precise street-address coordinates or proof of the 2014-15 event schedule.
# The old source has recurring series in San Francisco, Berkeley and Peninsula.
VENUE_REFERENCE_ZIPS = {
    "san_francisco": "94102",  # Herbst Theatre area
    "berkeley": "94704",       # First Congregational Church area
    "palo_alto": "94301",      # First United Methodist Church area
    "stanford": "94305",       # Bing Concert Hall area
}

SMOOTHING_ACCOUNTS = 20.0


def normalize_zip(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.extract(r"(\d{5})", expand=False)


def haversine_km(lat1, lon1, lat2: float, lon2: float) -> np.ndarray:
    """Great-circle distance in kilometers; invalid source coordinates -> NaN."""

    latitude = pd.to_numeric(lat1, errors="coerce").to_numpy(dtype=float)
    longitude = pd.to_numeric(lon1, errors="coerce").to_numpy(dtype=float)
    valid = (
        np.isfinite(latitude) & np.isfinite(longitude)
        & (np.abs(latitude) <= 90) & (np.abs(longitude) <= 180)
    )
    lat_a, lon_a = np.radians(latitude), np.radians(longitude)
    lat_b, lon_b = np.radians(lat2), np.radians(lon2)
    dlat, dlon = lat_b - lat_a, lon_b - lon_a
    term = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2) ** 2
    )
    term = np.clip(term, 0.0, 1.0)
    distances = 6371.0088 * 2 * np.arcsin(np.sqrt(term))
    distances[~valid] = np.nan
    return distances


def _clean_zip_table(zipcodes: pd.DataFrame) -> pd.DataFrame:
    zips = zipcodes.copy()
    zips["billing_zip"] = normalize_zip(zips["Zipcode"])
    zips = zips.dropna(subset=["billing_zip"])
    if zips["billing_zip"].duplicated().any():
        raise ValueError("ZIP reference table contains duplicate five-digit codes.")
    return zips


def add_demographic_features(features: pd.DataFrame, zipcodes: pd.DataFrame) -> pd.DataFrame:
    """Use only the census/tax-style fields actually present in zipcodes.csv."""

    zips = _clean_zip_table(zipcodes)
    columns = {
        "TaxReturnsFiled": "geo_zip_tax_returns",
        "EstimatedPopulation": "geo_zip_est_population",
        "TotalWages": "geo_zip_total_wages",
    }
    demographic = zips[["billing_zip", *columns]].rename(columns=columns).copy()
    for col in columns.values():
        demographic[col] = pd.to_numeric(demographic[col], errors="coerce")
        demographic.loc[demographic[col] < 0, col] = np.nan

    valid_returns = demographic["geo_zip_tax_returns"].where(
        demographic["geo_zip_tax_returns"] > 0
    )
    demographic["geo_zip_wages_per_return"] = (
        demographic["geo_zip_total_wages"] / valid_returns
    )
    demographic["geo_zip_people_per_return"] = (
        demographic["geo_zip_est_population"] / valid_returns
    )
    return features.merge(demographic, on="billing_zip", how="left", validate="many_to_one")


def add_distance_features(features: pd.DataFrame, zipcodes: pd.DataFrame) -> pd.DataFrame:
    """Approximate travel proximity using provided venue-ZIP centroids."""

    zips = _clean_zip_table(zipcodes).set_index("billing_zip")
    result = features.copy()
    coords = []
    names = []
    for area, venue_zip in VENUE_REFERENCE_ZIPS.items():
        if venue_zip not in zips.index:
            raise ValueError(f"Venue ZIP {venue_zip} missing from reference table.")
        row = zips.loc[venue_zip]
        lat, lon = float(row["Lat"]), float(row["Long"])
        if not np.isfinite(lat) or not np.isfinite(lon):
            raise ValueError(f"Venue ZIP {venue_zip} has no usable coordinates.")
        result[f"geo_distance_{area}_km"] = haversine_km(
            result["zip_lat"], result["zip_long"], lat, lon
        )
        coords.append(result[f"geo_distance_{area}_km"].to_numpy(dtype=float))
        names.append(area)

    distances = np.stack(coords, axis=1)
    masked = np.where(np.isfinite(distances), distances, np.inf)
    nearest_idx = np.argmin(masked, axis=1)
    min_dist = masked[np.arange(len(result)), nearest_idx]
    present = np.isfinite(min_dist)
    result["geo_distance_nearest_km"] = np.where(present, min_dist, np.nan)
    result["geo_nearest_venue_area"] = pd.Series(
        np.where(present, np.asarray(names)[nearest_idx], pd.NA),
        index=result.index, dtype="string",
    )
    result["geo_distance_peninsula_km"] = np.minimum(
        masked[:, names.index("palo_alto")], masked[:, names.index("stanford")]
    )
    result.loc[~np.isfinite(result["geo_distance_peninsula_km"]),
               "geo_distance_peninsula_km"] = np.nan
    return result


def add_density_features(
    features: pd.DataFrame,
    accounts: pd.DataFrame,
    subscriptions: pd.DataFrame,
) -> pd.DataFrame:
    """Leave-one-account-out local audience counts from pre-2014 activity.

    These aggregates use historically observable account/subscription data,
    NEVER purchase labels. Excluding the focal account prevents a single
    record's own history from determining its regional propensity.
    """

    account_zips = accounts[["account.id", "billing.zip.code"]].copy()
    account_zips["billing_zip"] = normalize_zip(account_zips["billing.zip.code"])
    if account_zips["account.id"].duplicated().any():
        raise ValueError("Account IDs must be unique for local density.")

    season_year = pd.to_numeric(
        subscriptions["season"].astype("string").str[:4], errors="coerce"
    )
    active_last_season = set(
        subscriptions.loc[season_year == 2013, "account.id"].dropna().astype("string")
    )
    account_zips["active_2013"] = (
        account_zips["account.id"].astype("string").isin(active_last_season).astype(int)
    )
    counts = (
        account_zips.dropna(subset=["billing_zip"])
        .groupby("billing_zip", observed=True)
        .agg(geo_zip_patron_count=("account.id", "size"),
             geo_zip_active_2013_count=("active_2013", "sum"))
        .reset_index()
    )
    result = features.merge(counts, on="billing_zip", how="left", validate="many_to_one")

    # The focal customer's own pre-2014 activity is known, but its own count
    # should not be treated as evidence of OTHER neighborhood subscribers.
    own_active = result["account.id"].astype("string").isin(active_last_season).astype(int)
    others = result["geo_zip_patron_count"] - 1
    other_active = result["geo_zip_active_2013_count"] - own_active
    global_rate = float(account_zips["active_2013"].mean())
    result["geo_zip_other_patron_count"] = others
    result["geo_zip_other_active_subscribers"] = other_active
    result["geo_zip_other_active_share_smoothed"] = (
        (other_active + SMOOTHING_ACCOUNTS * global_rate)
        / (others + SMOOTHING_ACCOUNTS)
    )
    result["geo_zip_log_other_patrons"] = np.log1p(others.clip(lower=0))
    return result.drop(columns=["geo_zip_patron_count", "geo_zip_active_2013_count"])


def augment_geography(
    features: pd.DataFrame,
    accounts: pd.DataFrame,
    zipcodes: pd.DataFrame,
    subscriptions: pd.DataFrame | None,
    mode: str = "all",
) -> pd.DataFrame:
    """Return original features plus the requested geographic feature block."""

    if mode not in GEO_MODES:
        raise ValueError(f"Unknown geography mode {mode!r}. Expected one of {GEO_MODES}.")
    if mode == "baseline":
        return features

    result = features.copy()
    if mode in ("demographics", "all"):
        result = add_demographic_features(result, zipcodes)
    if mode in ("distance", "all"):
        result = add_distance_features(result, zipcodes)
    if mode in ("density", "all"):
        if subscriptions is None:
            raise ValueError("Historical subscriptions are required for density features.")
        result = add_density_features(result, accounts, subscriptions)
    return result
