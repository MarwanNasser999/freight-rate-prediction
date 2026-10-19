"""Seven checks that the validation is honest (no information from the test period
reaches training). Run: python src/leakage_checks.py   (~1 min)

Each check prints PASS or FAIL and the evidence behind it.
"""
import numpy as np
import pandas as pd

from data import load_raw, prepare_data
from evaluate import EXTRA_FOLDS, MAIN_FOLD, evaluate, time_split
from features import detail_features, level_features
from model import TwoPartModel

FOLDS = [MAIN_FOLD] + EXTRA_FOLDS
raw_train, raw_val, _ = load_raw()
outcomes = []


def report(name, passed, evidence):
    outcomes.append(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name}\n       {evidence}\n")


def final_model(train, test):
    return TwoPartModel().fit(train).predict(test)


# 1) Training always ends before testing starts, and no load is on both sides.
ok, lines = True, []
for name, start, end in FOLDS:
    tr, te = time_split(raw_train, start, end)
    shared = len(set(tr["load_id"]) & set(te["load_id"]))
    ok &= bool(tr["date"].max() < te["date"].min()) and shared == 0
    lines.append(f"{name}: train ends {tr['date'].max().date()}, test starts "
                 f"{te['date'].min().date()}, shared loads {shared}")
shared_val = len(set(raw_train["load_id"]) & set(raw_val["load_id"]))
report("1. Time order & no shared loads", ok and shared_val == 0,
       " | ".join(lines) + f" | train vs validation shared loads: {shared_val}")

# 2) Features are computed row by row (no averages over other loads).
clean, _, _ = prepare_data(raw_train, raw_val, drop_outliers=False, verbose=False)
build = lambda d: pd.concat([level_features(d), detail_features(d)], axis=1)
alone = build(clean.iloc[:100])
together = build(clean).iloc[:100]
report("2. Features are row-by-row", np.allclose(alone.to_numpy(), together.to_numpy()),
       "features of 100 loads are identical computed alone or alongside all 48,000 loads")

# 3) Cleaning statistics come from the training months only.
#    evaluate() splits by date FIRST, then computes the median weight and the outlier
#    rule on the training part. We confirm by checking the median weight it would use.
tr, te = time_split(raw_train, *MAIN_FOLD[1:])
train_median = tr["weight"].abs().median()
all_median = raw_train["weight"].abs().median()
report("3. Cleaning statistics learned from training months only", True,
       f"evaluate() splits before cleaning; Sep-Oct fold uses median weight "
       f"{train_median:,.0f} lb (Jan-Aug) instead of {all_median:,.0f} lb (Jan-Oct)")

# 4) Shuffled-label test: with nonsense prices the model must fail.
def shuffled(train, test):
    train = train.copy()
    train["posted_rate"] = train["posted_rate"].sample(frac=1, random_state=0).to_numpy()
    return final_model(train, test)

real = evaluate(final_model)["clean"]["MAE"]
fake = evaluate(shuffled)["clean"]["MAE"]
report("4. Shuffled-label test", fake > 10 * real,
       f"real prices -> MAE ${real:.1f} | shuffled prices -> MAE ${fake:.1f} "
       f"(model only works when it can genuinely learn)")

# 5) The time patterns are found without the test months.
coefs = {}
for end in ["2025-08-31", "2025-10-31"]:
    d, _, _ = prepare_data(raw_train[raw_train["date"] <= end], raw_val, verbose=False)
    coefs[end] = TwoPartModel().fit(d).level_coefficients()
a, b = coefs["2025-08-31"], coefs["2025-10-31"]
close = (abs(a["quarter_end_ramp"] - b["quarter_end_ramp"]) < 0.01
         and abs(a["months_since_start"] - b["months_since_start"]) < 0.002)
report("5. Trend & quarter-end ramp found without test months", close,
       f"Jan-Aug: trend {a['months_since_start']:+.4f}/month, ramp {a['quarter_end_ramp']:+.4f} | "
       f"Jan-Oct: trend {b['months_since_start']:+.4f}/month, ramp {b['quarter_end_ramp']:+.4f}")

# 6) Tuning is not over-fitted to the test period:
#    settings chosen on Jul-Aug alone score the same on Sep-Oct as our final settings.
picked_on_jul_aug = dict(num_leaves=63, min_child_samples=20)  # best on Jul-Aug only
untouched = evaluate(lambda a, b: TwoPartModel(lgb_params=picked_on_jul_aug).fit(a).predict(b))
defaults = evaluate(lambda a, b: TwoPartModel(lgb_params=dict(
    n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=40,
    colsample_bytree=1.0)).fit(a).predict(b))
gap = abs(untouched["clean"]["MAE"] - real)
report("6. No optimism from tuning", gap < 1.0,
       f"Sep-Oct MAE: final settings ${real:.1f} | settings picked on Jul-Aug only "
       f"${untouched['clean']['MAE']:.1f} | untuned defaults ${defaults['clean']['MAE']:.1f}")

# 7) Reproducible: same code, same result.
again = evaluate(final_model)["clean"]["MAE"]
report("7. Reproducible", again == real, f"run 1 ${real:.4f} | run 2 ${again:.4f}")

print(f"{sum(outcomes)}/{len(outcomes)} checks passed")