"""How we split, score and compare models.

Main split (mirrors Spotter's setup): train on Jan-Aug, test on Sep-Oct.
Extra fold for a stability check:    train on Jan-Jun, test on Jul-Aug.
"""
import numpy as np
import pandas as pd

from data import find_rate_outliers, load_raw, prepare_data

# Each fold: (name, first test day, last test day). Training = everything before the test.
MAIN_FOLD = ("Sep-Oct", "2025-09-01", "2025-10-31")
EXTRA_FOLDS = [("Jul-Aug", "2025-07-01", "2025-08-31")]

DISTANCE_RANGES = [0, 250, 500, 1000, 1500, 2500, np.inf]


def metrics(y_true, y_pred):
    """MAE ($), MAPE (%) and RMSE ($)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    err = y_pred - y_true
    return {
        "MAE": np.mean(np.abs(err)),
        "MAPE": np.mean(np.abs(err) / y_true) * 100,
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "Bias%": (y_pred.sum() / y_true.sum() - 1) * 100,  # + = over-predicting overall
    }


def time_split(df, test_start, test_end):
    """Train = all loads before test_start. Test = loads from test_start to test_end."""
    test_start, test_end = pd.Timestamp(test_start), pd.Timestamp(test_end)
    train = df[df["date"] < test_start]
    test = df[(df["date"] >= test_start) & (df["date"] <= test_end)]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def evaluate(predict_fn, fold=MAIN_FOLD, drop_train_outliers=True):
    """Train on the past, score on the fold's test period.

    predict_fn(train_df, test_df) -> predictions for test_df.
    Training uses clean rows (unless drop_train_outliers=False, for comparison).
    We score on clean test rows (main number) and on all test rows including
    corrupted prices (what Spotter may see).
    """
    raw_train, _, _ = load_raw()
    name, start, end = fold

    # 1) Split FIRST, so every statistic below is learned from the past only.
    raw_tr, raw_te = time_split(raw_train, start, end)
    # 2) Clean inputs: median weight comes from raw_tr; the daily market level uses
    #    market_index of both periods (an input we are given, never a price).
    train, test, _ = prepare_data(raw_tr, raw_te, drop_outliers=False, verbose=False)
    # 3) Outlier rule built from training prices only, then applied to both.
    train["is_outlier"] = find_rate_outliers(train)
    test["is_outlier"] = find_rate_outliers(test, reference=train)

    if drop_train_outliers:
        train = train[~train["is_outlier"]].reset_index(drop=True)

    preds = np.asarray(predict_fn(train, test))
    clean = ~test["is_outlier"].to_numpy()
    return {
        "fold": name,
        "clean": metrics(test["posted_rate"][clean], preds[clean]),
        "with_outliers": metrics(test["posted_rate"], preds),
        "n_train": len(train),
        "n_test": int(clean.sum()),
    }


def print_result(label, result):
    c, w = result["clean"], result["with_outliers"]
    print(f"{label:<34} [{result['fold']}] "
          f"MAE ${c['MAE']:6.1f} | MAPE {c['MAPE']:4.2f}% | RMSE ${c['RMSE']:6.1f} | "
          f"Bias {c['Bias%']:+5.2f}%  || with outliers: MAE ${w['MAE']:6.1f}")


# ----------------------------- Baselines -----------------------------
def baseline_global(train, test):
    """B1: one median rate per mile for every load."""
    rpm = (train["posted_rate"] / train["distance"]).median()
    return test["distance"] * rpm


def baseline_equipment_distance(train, test):
    """B2: median rate per mile per (equipment, distance range)."""
    rpm = train["posted_rate"] / train["distance"]
    groups_train = [train["equipment"], pd.cut(train["distance"], DISTANCE_RANGES)]
    table = rpm.groupby(groups_train, observed=True).median()

    keys = pd.MultiIndex.from_arrays([test["equipment"], pd.cut(test["distance"], DISTANCE_RANGES)])
    test_rpm = pd.Series(table.reindex(keys).to_numpy(), index=test.index)
    test_rpm = test_rpm.fillna(rpm.median())  # unseen group -> overall median
    return test["distance"] * test_rpm


if __name__ == "__main__":
    for fold in [MAIN_FOLD] + EXTRA_FOLDS:
        r1 = evaluate(baseline_global, fold)
        r2 = evaluate(baseline_equipment_distance, fold)
        print(f"\nTrain rows: {r2['n_train']:,} | Test rows (clean): {r2['n_test']:,}")
        print_result("B1 global median $/mile", r1)
        print_result("B2 equipment x distance $/mile", r2)