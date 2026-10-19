"""Loading and cleaning the freight data.

Two kinds of cleaning happen here:
  1. Input fixes (weight, market_index) -> applied to train AND validation.
  2. Label fixes (corrupted prices)     -> applied to TRAIN ONLY.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Outlier rule: a load is corrupted if its rate per mile is below 0.5x or above
# 1.8x the typical rate for loads with the same equipment and a similar distance.
# Normal loads sit within ~0.8x-1.3x and nothing lies in between, so any cut-off
# inside that empty gap flags exactly the same rows.
OUTLIER_LOW, OUTLIER_HIGH = 0.5, 1.8
N_DISTANCE_BINS = 20


def load_raw():
    """Read the three input files exactly as provided."""
    train = pd.read_csv(DATA_DIR / "train_test.csv", parse_dates=["date"])
    val = pd.read_csv(DATA_DIR / "validation.csv", parse_dates=["date"])
    dec = pd.read_csv(DATA_DIR / "december_chart_inputs.csv", parse_dates=["date"])
    return train, val, dec


def daily_market_index(*frames):
    """One market level per day: the average market_index of all loads that day.

    Uses train and validation together. This is safe: market_index is an input
    we are given for every day, not a price.
    """
    stacked = pd.concat([f[["date", "market_index"]] for f in frames])
    return stacked.groupby("date")["market_index"].mean()


def clean_inputs(df, weight_fill, daily_mi):
    """Fix input columns. Safe to use on train, validation and December rows."""
    df = df.copy()
    # Negative weights are sign errors: their absolute values match normal weights.
    df["weight"] = df["weight"].abs()
    # Missing weights -> median weight from the training data.
    df["weight"] = df["weight"].fillna(weight_fill)
    # Missing market_index -> that day's market level.
    day_level = df["date"].map(daily_mi)
    if "market_index" in df:
        df["market_index"] = df["market_index"].fillna(day_level)
    else:  # December chart rows have no market_index column at all
        df["market_index"] = day_level
    return df


def find_rate_outliers(df, reference=None):
    """Return True for rows of `df` whose price looks corrupted.

    The "typical rate" for each (equipment, distance group) is learned from
    `reference` (default: df itself). In validation we pass the training rows
    as reference, so test-period prices are never used to build the rule.
    """
    reference = df if reference is None else reference
    ref_rpm = reference["posted_rate"] / reference["distance"]
    edges = np.quantile(reference["distance"], np.linspace(0, 1, N_DISTANCE_BINS + 1))
    edges[0], edges[-1] = 0, np.inf  # make the outer groups catch any distance
    typical = ref_rpm.groupby(
        [reference["equipment"], pd.cut(reference["distance"], edges)], observed=True
    ).median()

    keys = pd.MultiIndex.from_arrays([df["equipment"], pd.cut(df["distance"], edges)])
    ratio = (df["posted_rate"] / df["distance"]).to_numpy() / typical.reindex(keys).to_numpy()
    return pd.Series((ratio < OUTLIER_LOW) | (ratio > OUTLIER_HIGH), index=df.index)


def prepare_data(train, val, dec=None, drop_outliers=True, verbose=True):
    """Clean everything. Statistics (median weight) come from `train` only."""
    weight_fill = train["weight"].abs().median()
    daily_mi = daily_market_index(train, val)

    train_clean = clean_inputs(train, weight_fill, daily_mi)
    val_clean = clean_inputs(val, weight_fill, daily_mi)
    dec_clean = clean_inputs(dec, weight_fill, daily_mi) if dec is not None else None

    outliers = find_rate_outliers(train_clean)
    if drop_outliers:
        train_clean = train_clean[~outliers].reset_index(drop=True)

    if verbose:
        print(f"Weight fill value (train median): {weight_fill:,.0f} lb")
        print(f"Rate outliers found: {outliers.sum()} ({outliers.mean():.2%}) "
              f"-> {'removed' if drop_outliers else 'kept'}")
        print(f"Train rows: {len(train)} -> {len(train_clean)} | Validation rows: {len(val_clean)}")
    return train_clean, val_clean, dec_clean


if __name__ == "__main__":
    # Quick self-check: python src/data.py
    train, val, dec = load_raw()
    tr, va, de = prepare_data(train, val, dec)
    for name, d in [("train", tr), ("validation", va), ("december", de)]:
        print(f"{name}: missing weight={d.weight.isna().sum()}, "
              f"missing market_index={d.market_index.isna().sum()}, "
              f"negative weight={(d.weight < 0).sum()}")
    rpm = tr.posted_rate / tr.distance
    print(f"Clean train rate per mile: min {rpm.min():.2f}, max {rpm.max():.2f}")
    print(de[["date", "market_index"]].head(7).to_string(index=False))