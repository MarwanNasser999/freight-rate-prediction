"""Sanity checks on the final predictions (run after predict.py).

    python src/check_predictions.py

We cannot score Nov-Dec ourselves (Spotter holds those prices), so we check that the
predictions are plausible: right format, realistic $/mile, consistent with the recent
past, sensible for new cities, and a price-level curve that continues Jan-Oct smoothly.
Saves reports/price_level_forecast.png for the report.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import load_raw, prepare_data
from evaluate import baseline_equipment_distance

ROOT = Path(__file__).resolve().parents[1]


def check(name, passed, detail):
    print(f"[{'OK ' if passed else 'BAD'}] {name}: {detail}")
    return passed


def relative_level(df, price_col, reference):
    """Price per mile divided by the typical $/mile for that equipment & distance range."""
    bins = [0, 250, 500, 1000, 1500, 2500, np.inf]
    ref_rpm = reference["posted_rate"] / reference["distance"]
    typical = ref_rpm.groupby([reference["equipment"], pd.cut(reference["distance"], bins)],
                              observed=True).median()
    keys = pd.MultiIndex.from_arrays([df["equipment"], pd.cut(df["distance"], bins)])
    return (df[price_col] / df["distance"]).to_numpy() / typical.reindex(keys).to_numpy()


def main():
    raw_train, raw_val, _ = load_raw()
    train, val, _ = prepare_data(raw_train, raw_val, verbose=False)
    preds = pd.read_csv(ROOT / "validation_predictions.csv")
    val = val.merge(preds, on="load_id", how="left")
    rpm_pred = val["predicted_rate"] / val["distance"]
    rpm_train = train["posted_rate"] / train["distance"]
    results = []

    # 1) Format (the same things score.py checks)
    expected = {f"TE-{i:06d}" for i in range(1, 12_001)}
    results.append(check("Format", list(preds.columns) == ["load_id", "predicted_rate"]
                         and len(preds) == 12_000 and set(preds["load_id"]) == expected
                         and (preds["predicted_rate"] > 0).all(),
                         "2 columns, 12,000 unique IDs, all prices positive"))

    # 2) Realistic $/mile: inside the range seen in clean training data
    results.append(check("Rate per mile is realistic",
                         rpm_pred.between(rpm_train.min(), rpm_train.max() * 1.05).all(),
                         f"predicted ${rpm_pred.min():.2f}-${rpm_pred.max():.2f}/mile "
                         f"(clean training: ${rpm_train.min():.2f}-${rpm_train.max():.2f})"))

    # 3) Consistent with the most recent months, per equipment
    recent = train[train["date"] >= "2025-09-01"]
    recent_rpm = (recent["posted_rate"] / recent["distance"]).groupby(recent["equipment"]).median()
    pred_rpm = rpm_pred.groupby(val["equipment"]).median()
    change = (pred_rpm / recent_rpm - 1) * 100
    results.append(check("Close to Sep-Oct prices", change.abs().max() < 5,
                         " | ".join(f"{eq}: ${recent_rpm[eq]:.2f} -> ${pred_rpm[eq]:.2f} ({change[eq]:+.1f}%)"
                                    for eq in recent_rpm.index)))

    # 4) Agrees with the simple baseline, slightly higher because of the trend + Q4 ramp
    base = baseline_equipment_distance(train, val)
    ratio = (val["predicted_rate"] / base).mean()
    corr = np.corrcoef(val["predicted_rate"], base)[0, 1]
    results.append(check("Agrees with simple baseline", corr > 0.99 and 0.98 < ratio < 1.06,
                         f"correlation {corr:.4f}, model / baseline = {ratio:.3f}"))

    # 5) Loads involving cities never seen in training get sensible prices too
    new_cities = (set(val["pickup"]) | set(val["delivery"])) - (set(train["pickup"]) | set(train["delivery"]))
    is_new = val["pickup"].isin(new_cities) | val["delivery"].isin(new_cities)
    r_new, r_seen = (val["predicted_rate"] / base)[is_new].mean(), (val["predicted_rate"] / base)[~is_new].mean()
    results.append(check("New-city loads behave like the rest", abs(r_new - r_seen) < 0.02,
                         f"{is_new.sum():,} loads with {len(new_cities)} new cities: model/baseline "
                         f"{r_new:.3f} vs {r_seen:.3f} for seen cities"))

    # 6) Inputs inside the training range (no wild extrapolation)
    inside = all(val[c].between(train[c].min(), train[c].max()).all()
                 for c in ["distance", "weight", "market_index", "quote_signal"])
    results.append(check("Validation inputs inside training range", inside,
                         "distance, weight, market_index and quote_signal all within Jan-Oct ranges"))

    print(f"\n{sum(results)}/{len(results)} checks OK")

    # Chart: daily price level, actual Jan-Oct and predicted Nov-Dec
    actual = pd.Series(relative_level(train, "posted_rate", train), index=train["date"]).groupby(level=0).mean()
    forecast = pd.Series(relative_level(val, "predicted_rate", train), index=val["date"]).groupby(level=0).mean()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(actual.index, actual.rolling(7, center=True, min_periods=1).mean(), color="black", lw=2,
            label="Actual price level, Jan-Oct (7-day avg)")
    ax.plot(forecast.index, forecast.rolling(7, center=True, min_periods=1).mean(), color="#2E86AB", lw=2,
            label="Predicted price level, Nov-Dec validation loads (7-day avg)")
    ax.axvline(pd.Timestamp("2025-11-01"), color="grey", ls="--", lw=1)
    for q in ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]:
        ax.axvline(pd.Timestamp(q), color="#D1495B", ls=":", lw=1)
    ax.set_title("Price level: actual history and model forecast (red dotted lines = quarter ends)")
    ax.set_ylabel("Price level\n(1.0 = typical for distance & equipment)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    (ROOT / "reports").mkdir(exist_ok=True)
    plt.savefig(ROOT / "reports" / "price_level_forecast.png", dpi=130)
    print("Saved reports/price_level_forecast.png")


if __name__ == "__main__":
    main()