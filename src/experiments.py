"""Reproduce every model comparison from Step 6.

    python src/experiments.py           # comparisons + new-city check (~2 min)
    python src/experiments.py --tune    # also the 24-setting tuning grid (~5 more min)

Every result is scored with evaluate(): train on the past, test on Sep-Oct and Jul-Aug.
Results are also saved to reports/experiments.csv for the report.
"""
import argparse
import itertools
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from data import find_rate_outliers, load_raw, prepare_data
from evaluate import (EXTRA_FOLDS, MAIN_FOLD, baseline_equipment_distance, evaluate,
                      metrics, print_result, time_split)
from features import detail_features, level_features, make_target, to_price
from model import FINAL_LGB_PARAMS, TwoPartModel

FOLDS = [MAIN_FOLD] + EXTRA_FOLDS
REPORTS = Path(__file__).resolve().parents[1] / "reports"
results = []


def run(label, predict_fn, **kwargs):
    """Score one approach on both folds, print it and remember it."""
    scores = [evaluate(predict_fn, fold, **kwargs) for fold in FOLDS]
    for s in scores:
        print_result(label, s)
    avg = np.mean([s["clean"]["MAE"] for s in scores])
    print(f"{'':<34} -> average MAE ${avg:.1f}\n")
    results.append({"approach": label,
                    **{f"MAE {s['fold']}": round(s["clean"]["MAE"], 1) for s in scores},
                    "average MAE": round(avg, 1)})


# ----------------------------- Approaches -----------------------------
def two_part(**kwargs):
    return lambda train, test: TwoPartModel(**kwargs).fit(train).predict(test)


def lightgbm_only(train, test):
    """One LightGBM with ALL features, time included (it cannot extrapolate time)."""
    def X(d):
        both = pd.concat([level_features(d), detail_features(d)], axis=1)
        return both.loc[:, ~both.columns.duplicated()]
    model = lgb.LGBMRegressor(**FINAL_LGB_PARAMS).fit(X(train), make_target(train))
    return to_price(model.predict(X(test)), test["distance"])


def two_part_with(make_learner):
    """Same two-part idea, but a different tree model for Part 2."""
    def predict(train, test):
        y = make_target(train)
        part1 = Ridge(alpha=1e-3).fit(level_features(train), y)
        residual = y - part1.predict(level_features(train))
        part2 = make_learner().fit(detail_features(train), residual)
        log_rpm = part1.predict(level_features(test)) + part2.predict(detail_features(test))
        return to_price(log_rpm, test["distance"])
    return predict


def recency_weighted(half_life_days):
    """Recent loads count more: weight halves every `half_life_days`."""
    def predict(train, test):
        age = (train["date"].max() - train["date"]).dt.days
        weights = 0.5 ** (age / half_life_days)
        return TwoPartModel().fit(train, sample_weight=weights.to_numpy()).predict(test)
    return predict


# ----------------------------- Checks -----------------------------
def new_city_check(n_cities=8, repeats=3):
    """Hide some cities from training entirely, then score loads that involve them."""
    raw_train, _, _ = load_raw()
    raw_tr, raw_te = time_split(raw_train, *MAIN_FOLD[1:])
    train, test, _ = prepare_data(raw_tr, raw_te, drop_outliers=False, verbose=False)
    test = test[~find_rate_outliers(test, reference=train)]
    train = train[~find_rate_outliers(train)]

    rows = []
    for seed in range(repeats):
        cities = pd.Series(sorted(train["pickup"].unique())).sample(n_cities, random_state=seed)
        touches = lambda d: d["pickup"].isin(cities) | d["delivery"].isin(cities)
        tr = train[~touches(train)]
        new, seen = test[touches(test)], test[~touches(test)]
        model = TwoPartModel().fit(tr)
        rows.append({
            "repeat": seed + 1,
            "test loads with new cities": len(new),
            "MAE new cities": metrics(new["posted_rate"], model.predict(new))["MAE"],
            "MAE seen cities": metrics(seen["posted_rate"], model.predict(seen))["MAE"],
            "baseline MAE new cities": metrics(new["posted_rate"],
                                               baseline_equipment_distance(tr, new))["MAE"],
        })
    print(pd.DataFrame(rows).round(1).to_string(index=False), "\n")


def tuning_grid():
    """24 LightGBM settings, each scored on both folds."""
    grid = itertools.product([15, 31, 63], [20, 50, 150], [(0.05, 600), (0.03, 1200)], [1.0, 0.8])
    rows = []
    for leaves, min_leaf, (lr, trees), cols in grid:
        params = dict(num_leaves=leaves, min_child_samples=min_leaf, learning_rate=lr,
                      n_estimators=trees, colsample_bytree=cols)
        maes = [evaluate(two_part(lgb_params=params), f)["clean"]["MAE"] for f in FOLDS]
        rows.append({**params, "MAE Sep-Oct": maes[0], "MAE Jul-Aug": maes[1],
                     "average MAE": np.mean(maes)})
    table = pd.DataFrame(rows).sort_values("average MAE").round(2)
    print(table.head(8).to_string(index=False))
    print(f"... worst average MAE: ${table['average MAE'].max():.2f}\n")
    table.to_csv(REPORTS / "tuning.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="also run the tuning grid")
    args = parser.parse_args()
    REPORTS.mkdir(exist_ok=True)

    print("=== Round 1: which kind of model? ===")
    run("B2 baseline", baseline_equipment_distance)
    run("Linear only (Part 1)", two_part(use_part2=False))
    run("LightGBM only (all features)", lightgbm_only)
    run("Two-part: Linear + RandomForest", two_part_with(
        lambda: RandomForestRegressor(300, min_samples_leaf=20, n_jobs=-1, random_state=0)))
    run("Two-part: Linear + HistGB", two_part_with(
        lambda: HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, random_state=0)))
    run("FINAL: Two-part Linear + LightGBM", two_part())

    print("=== Round 2: open questions ===")
    run("Keep outliers in training", two_part(), drop_train_outliers=False)
    run("Keep outliers + Huber loss", two_part(lgb_params=dict(objective="huber", alpha=0.05)),
        drop_train_outliers=False)
    run("Recency weight, half-life 60 days", recency_weighted(60))
    run("Recency weight, half-life 120 days", recency_weighted(120))
    run("L1 loss", two_part(lgb_params=dict(objective="l1")))

    print("=== New-city check (Sep-Oct, cities hidden from training) ===")
    new_city_check()

    if args.tune:
        print("=== Tuning grid ===")
        tuning_grid()

    pd.DataFrame(results).to_csv(REPORTS / "experiments.csv", index=False)
    print(f"Saved results to {REPORTS / 'experiments.csv'}")

    final = TwoPartModel().fit(prepare_data(*load_raw()[:2], verbose=False)[0])
    print("\nFinal model, Part 1 weights (trained on all Jan-Oct):", final.level_coefficients())