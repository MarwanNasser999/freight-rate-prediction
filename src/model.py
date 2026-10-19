"""The final two-part freight rate model.

Part 1: a linear model on level_features()  -> the general price level of a day
        (market_index, time trend, quarter-end ramp). Linear = it can extrapolate.
Part 2: LightGBM on detail_features()       -> a correction for each load
        (route, direction, quote). It learns what Part 1 got wrong.

prediction = distance * exp(part1 + part2)      (both parts work in log $/mile)
"""
import lightgbm as lgb
from sklearn.linear_model import Ridge

from features import detail_features, level_features, make_target, to_price

# Chosen in Step 6: 24 combinations tested on both validation folds.
# The spread between best and worst was only ~$1 MAE, so the model is not fragile.
FINAL_LGB_PARAMS = dict(
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=50,
    colsample_bytree=0.8,
    random_state=42,  # fixed seed -> identical results on every run
    verbose=-1,
)


class TwoPartModel:
    def __init__(self, lgb_params=None, use_part2=True):
        self.lgb_params = {**FINAL_LGB_PARAMS, **(lgb_params or {})}
        self.use_part2 = use_part2  # False = "linear only", used for comparison
        self.part1 = Ridge(alpha=1e-3)
        self.part2 = lgb.LGBMRegressor(**self.lgb_params)

    def fit(self, train, sample_weight=None):
        y = make_target(train)
        X1 = level_features(train)
        self.part1.fit(X1, y, sample_weight=sample_weight)
        if self.use_part2:
            residual = y - self.part1.predict(X1)  # what Part 1 could not explain
            self.part2.fit(detail_features(train), residual, sample_weight=sample_weight)
        return self

    def predict_log_rpm(self, df):
        log_rpm = self.part1.predict(level_features(df))
        if self.use_part2:
            log_rpm = log_rpm + self.part2.predict(detail_features(df))
        return log_rpm

    def predict(self, df):
        """Predicted price in dollars."""
        return to_price(self.predict_log_rpm(df), df["distance"])

    def level_coefficients(self):
        """Part 1 weights in log units; e.g. 0.12 for is_reefer = about +13% price."""
        return {name: round(float(w), 4) for name, w in zip(self.part1.feature_names_in_, self.part1.coef_)}