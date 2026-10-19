"""Feature engineering for the two-part model.

Part 1 (linear "price level"): explains the general price level of a day,
    including the trend and the quarter-end ramp, so it can extrapolate in time.
Part 2 (LightGBM "load details"): explains what is special about one load
    (route, direction, quote). It gets NO time features, so it never extrapolates.

Both parts predict log(rate per mile).
"""
import numpy as np
import pandas as pd

REFERENCE_DATE = pd.Timestamp("2025-01-01")  # day 0 of the time trend
QUARTER_RAMP_DAYS = 28  # prices climb over the last ~4 weeks of each quarter
EQUIPMENT_CODES = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}


# ----------------------------- Time helpers -----------------------------
def days_to_quarter_end(dates):
    """How many days until the end of the quarter (Mar 31, Jun 30, Sep 30, Dec 31)."""
    dates = pd.to_datetime(pd.Series(dates))
    quarter_end = dates.dt.to_period("Q").dt.end_time.dt.normalize()
    return (quarter_end - dates).dt.days


def quarter_end_ramp(dates):
    """0 for most of the quarter, rising linearly to 1 on the last day of the quarter."""
    days_left = days_to_quarter_end(dates)
    return ((QUARTER_RAMP_DAYS - days_left) / QUARTER_RAMP_DAYS).clip(lower=0)


def months_since_start(dates):
    """Time trend: months since Jan 1, 2025 (a 30-day month)."""
    return (pd.to_datetime(pd.Series(dates)) - REFERENCE_DATE).dt.days / 30


# ----------------------------- Target -----------------------------
def make_target(df):
    """What the model learns: log of the rate per mile."""
    return np.log(df["posted_rate"] / df["distance"])


def to_price(log_rpm, distance):
    """Turn a predicted log(rate per mile) back into a dollar price."""
    return np.exp(np.asarray(log_rpm)) * np.asarray(distance)


# ----------------------------- Part 1 features -----------------------------
def level_features(df):
    """Inputs for the linear price-level model."""
    return pd.DataFrame({
        "log_distance": np.log(df["distance"]),
        "is_reefer": (df["equipment"] == "Reefer").astype(int),
        "is_flatbed": (df["equipment"] == "Flatbed").astype(int),
        "weight_10k": df["weight"] / 10_000,
        "market_index": df["market_index"],
        "months_since_start": months_since_start(df["date"]).to_numpy(),
        "quarter_end_ramp": quarter_end_ramp(df["date"]).to_numpy(),
    }, index=df.index)


# ----------------------------- Part 2 features -----------------------------
def detail_features(df):
    """Inputs for the LightGBM correction model (no time features on purpose)."""
    return pd.DataFrame({
        "distance": df["distance"],
        "weight": df["weight"],
        "equipment": df["equipment"].map(EQUIPMENT_CODES),
        "pickup_lat": df["pickup_lat"],
        "pickup_lon": df["pickup_lon"],
        "delivery_lat": df["delivery_lat"],
        "delivery_lon": df["delivery_lon"],
        "delta_lat": df["delivery_lat"] - df["pickup_lat"],   # + = heading north
        "delta_lon": df["delivery_lon"] - df["pickup_lon"],   # + = heading east
        "quote_signal": df["quote_signal"],
    }, index=df.index)


if __name__ == "__main__":
    # Quick look: python src/features.py
    from data import load_raw, prepare_data

    train, val, dec = load_raw()
    tr, _, _ = prepare_data(train, val, verbose=False)
    sample = tr.iloc[[0, 1, 2]]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    print("Raw rows:\n", sample[["pickup", "delivery", "distance", "equipment", "weight",
                                 "date", "market_index", "quote_signal", "posted_rate"]], "\n")
    print("Part 1 (level) features:\n", level_features(sample).round(3), "\n")
    print("Part 2 (detail) features:\n", detail_features(sample).round(3), "\n")
    print("Target log(rate per mile):", make_target(sample).round(4).tolist(), "\n")

    days = pd.Series(pd.to_datetime(["2025-11-15", "2025-12-01", "2025-12-04",
                                     "2025-12-15", "2025-12-25", "2025-12-31", "2026-01-01"]))
    print(pd.DataFrame({"date": days.dt.date,
                        "days_to_quarter_end": days_to_quarter_end(days),
                        "quarter_end_ramp": quarter_end_ramp(days).round(3),
                        "months_since_start": months_since_start(days).round(2)}).to_string(index=False))