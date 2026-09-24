# Phase 5 Model Development Summary

## Scope

This phase compared baseline and modern regression models for predicting `log10_kd` on positive chromium Kd records.

The primary model selection criterion is leave-one-soil-out RMSE because records are clustered by soil type.

## Best Model Under Soil-Aware Validation

- Best model: `linear_regression`
- Leave-one-soil-out RMSE: 0.070454
- Leave-one-soil-out MAE: 0.047265
- Leave-one-soil-out R2: 0.743709

## Leave-One-Soil-Out Ranking

| Rank | Model | RMSE | MAE | R2 |
|---:|---|---:|---:|---:|
| 1 | linear_regression | 0.070454 | 0.047265 | 0.743709 |
| 2 | elastic_net | 0.073505 | 0.048135 | 0.741634 |
| 3 | ridge | 0.083821 | 0.053165 | 0.741191 |
| 4 | gradient_boosting | 0.133532 | 0.099987 | 0.711870 |
| 5 | xgboost | 0.136169 | 0.100826 | 0.707472 |
| 6 | random_forest | 0.156876 | 0.121029 | 0.700772 |
| 7 | extra_trees | 0.168650 | 0.129265 | 0.707798 |
| 8 | hist_gradient_boosting | 0.561259 | 0.486865 | 0.348127 |
| 9 | mlp | 0.572049 | 0.508574 | -0.287946 |
| 10 | dummy_mean | 1.245711 | 1.149333 | -0.946301 |

## Shuffled 5-Fold Ranking

| Rank | Model | RMSE | MAE | R2 |
|---:|---|---:|---:|---:|
| 1 | extra_trees | 0.046734 | 0.030639 | 0.997335 |
| 2 | random_forest | 0.060991 | 0.039304 | 0.996212 |
| 3 | linear_regression | 0.070351 | 0.043356 | 0.995263 |
| 4 | gradient_boosting | 0.071093 | 0.041358 | 0.994030 |
| 5 | elastic_net | 0.072394 | 0.043660 | 0.994887 |
| 6 | ridge | 0.080121 | 0.049347 | 0.993440 |
| 7 | xgboost | 0.094145 | 0.060282 | 0.987172 |
| 8 | mlp | 0.161749 | 0.117317 | 0.974736 |
| 9 | hist_gradient_boosting | 0.252772 | 0.214530 | 0.934214 |
| 10 | dummy_mean | 1.117313 | 0.985337 | -0.133267 |

## Held-Out Soil Diagnostics for Best Model

| Held-out soil | Rows | RMSE | MAE | R2 |
|---|---:|---:|---:|---:|
| Cecil/Pacolet | 20 | 0.069580 | 0.033187 | 0.991709 |
| Holton/Cloudland | 23 | 0.070117 | 0.046014 | 0.996173 |
| Kenoma | 15 | 0.054435 | 0.031105 | 0.986953 |
| Ocala | 10 | 0.087686 | 0.078756 | 0.000000 |

## Permutation Importance for Best Model

| Feature | Mean importance | Std importance |
|---|---:|---:|
| neg_log_c_mol_m3 | 0.846647 | 0.051340 |
| neg_log_s_mol_kg | 0.634693 | 0.038103 |
| experiment_pH | 0.008095 | 0.002543 |
| soil_dcb_extractable_fe_mmol_g | 0.000268 | 0.000542 |
| soil_native_pH | 0.000158 | 0.000454 |
| soil_clay_wt_pct | 0.000071 | 0.000236 |
| soil | 0.000050 | 0.000184 |
| soil_iron_oxide_wt_pct | 0.000024 | 0.000070 |
| soil_cec_meq_100g | 0.000024 | 0.000236 |
| soil_organic_carbon_wt_pct | 0.000008 | 0.000077 |

## Technical Interpretation

- Linear and regularized models generalize best to unseen soil groups, which is consistent with the dataset being small and strongly structured by experimental chemistry.
- Tree ensembles perform very well under shuffled cross-validation but degrade under leave-one-soil-out validation, indicating some soil-specific memorization.
- MLP is not recommended at this dataset size because it performs poorly in soil-aware validation.
- The final report should present both validation views, but model selection should emphasize leave-one-soil-out results.

## Recommendation

Use the best soil-aware model as the current candidate final model. Proceed to Phase 6 with light tuning of linear/regularized models and selected tree ensembles, then Phase 7 interpretability.
