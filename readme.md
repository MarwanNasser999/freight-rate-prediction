# Freight Rate Prediction

Predicting the posted rate (USD) of truck loads for Spotter's ML take-home assessment.

**Result (time-based validation, trained on the past and tested on later months):**

| Model | MAE Sep–Oct | MAE Jul–Aug |
|---|---|---|
| Baseline: median $/mile by equipment × distance | $83.0 (3.8%) | $87.6 (4.0%) |
| **Final: two-part model (linear price level + LightGBM)** | **$33.0 (1.4%)** | **$44.3 (1.8%)** |

Scores are on test loads with valid prices; ~1.4% of prices in the data are corrupted (see below), and including them in the test raises MAE to ~$89–96 for any model.

---

## Quick start

Requires Python 3.10+ (tested on 3.13 and 3.14).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows (cmd)   | PowerShell: .venv\Scripts\Activate.ps1
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Train the final model and write both prediction files (~10–30 s)
python src/predict.py

# 4. Validate the outputs and draw the December chart with Spotter's scorer
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

Outputs:
- `validation_predictions.csv` — `load_id,predicted_rate` for the 12,000 validation loads
- `data/december_chart_inputs.csv` — December rows with `predicted_rate` filled
- `scorer_results/candidate_december.png` — the December chart

### Optional: reproduce the analysis

```bash
python src/evaluate.py            # baselines on both validation folds
python src/experiments.py         # every model comparison (~2 min); add --tune for the tuning grid
python src/leakage_checks.py      # 7 checks that the validation is honest (~1 min)
python src/check_predictions.py   # sanity checks on the final predictions + forecast chart
```

---

## Project structure

```
├── data/                     input files (as provided)
├── notebooks/01_eda.ipynb    exploratory analysis
├── src/
│   ├── data.py               loading and cleaning
│   ├── features.py           feature engineering
│   ├── model.py              the final two-part model
│   ├── evaluate.py           time-based split, metrics, baselines
│   ├── experiments.py        model comparisons and tuning
│   ├── leakage_checks.py     leakage audit
│   ├── predict.py            final training + prediction files
│   └── check_predictions.py  sanity checks on the predictions
├── reports/                  result tables and charts
├── scorer_results/           chart produced by score.py
├── score.py                  Spotter's scorer (unchanged)
└── validation_predictions.csv
```

---

## Approach

### Key findings from the data
- **Validation is the future.** Training covers Jan–Oct 2025; validation covers Nov–Dec 2025. The task is forecasting, so validation must be time-based.
- **Price is mostly distance.** Rate per mile (median ~$2.15) is the stable quantity: short trips cost more per mile, and Reefer and Flatbed cost more than Dry Van.
- **`market_index` is a daily market level.** It's shared by all loads on the same day, with a weekly cycle (peak Thursday, low Sunday) and market regimes.
- **Prices have three time patterns.** They follow market_index, drift upward by ~0.6% per month, and **climb ~4% over the last 4 weeks of every quarter**, resetting on the 1st. Together these explain 97% of the daily price level.
- **8 cities appear only in validation** (12% of validation loads), so the model must generalise beyond city names.

### Data-quality issues and fixes
| Issue | Fix |
|---|---|
| 677 corrupted prices (1.4%): 0.15–0.5× or 2–5.5× the typical $/mile, with an empty gap separating them from normal loads | Removed from training only (keeping them doubled the error) |
| Negative weights (sign errors; same distribution as normal weights) | Absolute value |
| Missing weights | Training median |
| Missing market_index | That day's average (one value per day) |

### Model
Both parts predict log(rate per mile):
1. **Part 1 — linear price level:** log distance, equipment, weight, market_index, months since Jan 1 (trend), quarter-end ramp. Being linear, it can carry the trend into Nov–Dec.
2. **Part 2 — LightGBM correction** on Part 1's residuals: distance, weight, equipment, pickup/delivery coordinates, direction, quote_signal. It has no time features, so it never needs to extrapolate in time.

Why: tree models alone can't extrapolate the upward trend (a single LightGBM scored $46 / $76), and a linear model alone misses route-level detail ($58 / $60). The combination is best on both test periods.

### Validation
- **Main split:** train Jan–Aug, test Sep–Oct (same 2-month horizon as the real task). **Second fold:** train Jan–Jun, test Jul–Aug.
- A random split would be misleading: the same model scores ~2× better on a random split than on a time split.
- Every cleaning statistic (median weight, outlier rule) is learned from the training months only.
- `leakage_checks.py` verifies time order, row-wise features, a shuffled-label test, that the time patterns are found without test months, that tuning adds no optimism, and reproducibility.
- The final model is retrained on all of Jan–Oct before predicting Nov–Dec.

### December chart
The fixed load (Lexington → Fort Wayne, 360 mi, Dry Van, 32,000 lb) is missing three model inputs: market_index (each day's average from validation.csv), coordinates (from the cities in the data) and quote_signal (median for Dry Van; any reasonable value moves predictions < 1%). Predictions rise from $823 to $886, with a weekly wave from market_index and a year-end climb from the trend and the quarter-end ramp.

### Limitations
- The upward trend is assumed to continue for two more months; a market reversal would make predictions slightly high.
- Training contains no December, so holiday-specific effects can only come through market_index.
- Corrupted prices in the scored data (if present, as in training) cannot be predicted by any model.