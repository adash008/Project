# Phase 6 Optimization Summary

## Optimization Strategy

The tuning pass used leave-one-soil-out validation as the primary objective. This avoids selecting models that perform well only because records from the same soil appear in both train and validation folds.

Models tuned:

- Linear Regression baseline
- Ridge
- ElasticNet
- Random Forest
- ExtraTrees
- Gradient Boosting
- XGBoost
- A small stacked ensemble

## Best Tuned Model

- Model: `ridge`
- Parameters: `{'alpha': 0.1}`
- Leave-one-soil-out RMSE: 0.068077
- Leave-one-soil-out MAE: 0.043648
- Leave-one-soil-out R2: 0.743739

## Top 10 Configurations

| Rank | Model | RMSE | MAE | R2 | Params |
|---:|---|---:|---:|---:|---|
| 1 | ridge | 0.068077 | 0.043648 | 0.743739 | `{'alpha': 0.1}` |
| 2 | elastic_net | 0.070120 | 0.044722 | 0.742490 | `{'alpha': 0.01, 'l1_ratio': 0.5}` |
| 3 | ridge | 0.070133 | 0.046852 | 0.743715 | `{'alpha': 0.01}` |
| 4 | ridge | 0.070421 | 0.047224 | 0.743709 | `{'alpha': 0.001}` |
| 5 | linear_regression | 0.070454 | 0.047265 | 0.743709 | `{}` |
| 6 | elastic_net | 0.072138 | 0.047875 | 0.742498 | `{'alpha': 0.01, 'l1_ratio': 0.8}` |
| 7 | elastic_net | 0.072579 | 0.048781 | 0.742283 | `{'alpha': 0.003, 'l1_ratio': 0.2}` |
| 8 | elastic_net | 0.073290 | 0.050907 | 0.742541 | `{'alpha': 0.001, 'l1_ratio': 0.5}` |
| 9 | elastic_net | 0.073505 | 0.048135 | 0.741634 | `{'alpha': 0.01, 'l1_ratio': 0.2}` |
| 10 | elastic_net | 0.073866 | 0.047361 | 0.742132 | `{'alpha': 0.01, 'l1_ratio': 0.1}` |

## Best Model Held-Out Soil Diagnostics

| Held-out soil | Rows | RMSE | MAE | R2 |
|---|---:|---:|---:|---:|
| Cecil/Pacolet | 20 | 0.070054 | 0.033125 | 0.991595 |
| Holton/Cloudland | 23 | 0.066343 | 0.041506 | 0.996574 |
| Kenoma | 15 | 0.054779 | 0.031038 | 0.986788 |
| Ocala | 10 | 0.081134 | 0.068924 | 0.000000 |

## Optimization Decision

The tuning pass should be accepted only if it improves leave-one-soil-out RMSE without making the model unnecessarily complex. If the best tuned model is only marginally better than Linear Regression, the final report should prefer the simpler model or present both as candidate models.
