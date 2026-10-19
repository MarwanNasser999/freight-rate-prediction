"""Train the final model on all of Jan-Oct and write both prediction files.

    python src/predict.py

Outputs:
  validation_predictions.csv          load_id,predicted_rate for the 12,000 validation loads
  data/december_chart_inputs.csv      the same file, with predicted_rate filled in
  reports/december_predictions.csv    December rows with the inputs we filled, for the report
"""
from pathlib import Path

import pandas as pd

from data import DATA_DIR, load_raw, prepare_data
from model import TwoPartModel

ROOT = Path(__file__).resolve().parents[1]
DECEMBER_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]


def city_coordinates(*frames):
    """Each city's (lat, lon), taken from the pickup/delivery columns of the given tables."""
    parts = []
    for f in frames:
        parts.append(f[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1))
        parts.append(f[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(parts).drop_duplicates("city").set_index("city")


def complete_december_inputs(dec, train, val):
    """The December file only has pickup, delivery, distance, equipment, weight and date.

    - market_index: already filled by prepare_data() with each day's average from validation.csv
    - coordinates : looked up from the cities' coordinates in the data
    - quote_signal: not given -> a typical (median) quote for the same equipment in training.
                    Tested: any reasonable value moves predictions by less than 1%.
    """
    dec = dec.copy()
    coords = city_coordinates(train, val)
    dec["pickup_lat"] = dec["pickup"].map(coords["lat"])
    dec["pickup_lon"] = dec["pickup"].map(coords["lon"])
    dec["delivery_lat"] = dec["delivery"].map(coords["lat"])
    dec["delivery_lon"] = dec["delivery"].map(coords["lon"])
    typical_quote = train.groupby("equipment")["quote_signal"].median()
    dec["quote_signal"] = dec["equipment"].map(typical_quote)
    assert dec[["pickup_lat", "delivery_lat", "quote_signal", "market_index"]].notna().all().all()
    return dec


def main():
    raw_train, raw_val, raw_dec = load_raw()
    train, val, dec = prepare_data(raw_train, raw_val, raw_dec)  # outliers removed from train

    print("\nTraining the final model on all of Jan-Oct ...")
    model = TwoPartModel().fit(train)
    print("Part 1 weights:", model.level_coefficients())

    # 1) Validation: 12,000 predictions, in the template's order.
    val["predicted_rate"] = model.predict(val).round(2)
    template = pd.read_csv(DATA_DIR / "validation_predictions_template.csv")
    out = template[["load_id"]].merge(val[["load_id", "predicted_rate"]], on="load_id", how="left")
    assert len(out) == 12_000 and out["predicted_rate"].notna().all() and (out["predicted_rate"] > 0).all()
    out.to_csv(ROOT / "validation_predictions.csv", index=False)
    print(f"\nWrote validation_predictions.csv: {len(out):,} rows, "
          f"rates ${out.predicted_rate.min():,.0f} - ${out.predicted_rate.max():,.0f}")

    # 2) December chart: fill predicted_rate, keep the original 7 columns and order.
    dec_full = complete_december_inputs(dec, train, val)
    dec_full["predicted_rate"] = model.predict(dec_full).round(2)
    december = raw_dec.copy()
    december["predicted_rate"] = dec_full["predicted_rate"].to_numpy()
    december["date"] = december["date"].dt.strftime("%Y-%m-%d")
    december[DECEMBER_COLUMNS].to_csv(DATA_DIR / "december_chart_inputs.csv", index=False)

    (ROOT / "reports").mkdir(exist_ok=True)
    dec_full.assign(weekday=dec_full["date"].dt.day_name())[
        ["date", "weekday", "market_index", "quote_signal", "predicted_rate"]
    ].to_csv(ROOT / "reports" / "december_predictions.csv", index=False)
    p = dec_full["predicted_rate"]
    print(f"Filled data/december_chart_inputs.csv: 31 days, ${p.min():.0f} - ${p.max():.0f} "
          f"(Dec 1 ${p.iloc[0]:.0f} -> Dec 31 ${p.iloc[-1]:.0f})")


if __name__ == "__main__":
    main()