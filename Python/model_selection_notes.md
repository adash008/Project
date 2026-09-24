# Model Selection Notes

## Initial Untuned Model Comparison

Dataset:

- Source: EPA Table E.2 reproduction of Rai et al. (1988) chromium adsorption data
- Rows: 76 total
- Modeling rows: 68 positive-Kd rows for `log10_kd`
- Excluded from log target: 8 zero-Kd Ocala rows

Validation:

- Current strategy: 5-fold shuffled KFold on positive-Kd rows
- Important caveat: records are clustered by soil, so this likely overestimates generalization to unseen soils
- Next validation improvement: add leave-one-soil-out or group-aware validation

Initial ranking by shuffled cross-validated RMSE on `log10_kd`:

| Rank | Model | RMSE | MAE | R2 |
|---:|---|---:|---:|---:|
| 1 | Random Forest | 0.060991 | 0.039304 | 0.996212 |
| 2 | Linear Regression | 0.070351 | 0.043356 | 0.995263 |
| 3 | Gradient Boosting | 0.071093 | 0.041358 | 0.994030 |
| 4 | Ridge | 0.080121 | 0.049347 | 0.993440 |
| 5 | XGBoost | 0.094145 | 0.060282 | 0.987172 |
| 6 | MLP | 0.161749 | 0.117317 | 0.974736 |
| 7 | Dummy Mean | 1.117313 | 0.985337 | -0.133267 |

Soil-aware validation:

- A leave-one-soil-out validation was added after the initial run.
- This is the more conservative validation result because each fold tests on a soil type absent from training.
- Under this validation, Linear Regression performed best.

Leave-one-soil-out ranking:

| Rank | Model | RMSE | MAE | R2 |
|---:|---|---:|---:|---:|
| 1 | Linear Regression | 0.070454 | 0.047265 | 0.743709 |
| 2 | Ridge | 0.083821 | 0.053165 | 0.741191 |
| 3 | Gradient Boosting | 0.133532 | 0.099987 | 0.711870 |
| 4 | XGBoost | 0.136169 | 0.100826 | 0.707472 |
| 5 | Random Forest | 0.156876 | 0.121029 | 0.700772 |
| 6 | MLP | 0.572049 | 0.508574 | -0.287946 |
| 7 | Dummy Mean | 1.245711 | 1.149333 | -0.946301 |

Current best preliminary model for generalization:

- Linear Regression

Do not treat this as the final model until duplicate checks, outlier review, and interpretability review are complete.
