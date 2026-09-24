# Phase 7 Interpretability Summary

## Model Interpreted

- Optimized model: Ridge Regression
- Alpha: 0.1
- Target: `log10_kd`
- Rows interpreted: positive-Kd records only

## Main Scientific Findings

- The strongest experimental predictors are `-log C`, `-log S`, and experiment pH.
- Higher `-log C` is associated with higher predicted `log10_kd`. Because `-log C` increases as dissolved concentration decreases, this is consistent with stronger apparent sorption at lower aqueous chromium concentrations.
- Higher `-log S` is associated with lower predicted `log10_kd` in this dataset. This should be interpreted cautiously because `-log S` is tied to experimental sorbed/solid-phase quantities and covaries strongly with other experimental variables.
- Experiment pH has a negative coefficient, matching chromium(VI) sorption chemistry: higher pH generally reduces Cr(VI) adsorption and lowers Kd.
- Soil-level features have smaller direct coefficients after accounting for experimental variables, but DCB-extractable iron remains scientifically important because it reflects Cr(VI)-reducing capacity discussed in the literature.
- Zero-Kd Ocala records remain outside the log-regression target and should be documented as a limitation/future two-stage modeling opportunity.

## Top Positive Coefficients

| Feature | Coefficient |
|---|---:|
| neg_log_c_mol_m3 | 0.632890 |
| soil_dcb_extractable_fe_mmol_g | 0.005429 |
| soil_Holton/Cloudland | 0.002534 |
| soil_clay_wt_pct | 0.001748 |
| soil_iron_oxide_wt_pct | 0.000184 |

## Top Negative Coefficients

| Feature | Coefficient |
|---|---:|
| neg_log_s_mol_kg | -0.488960 |
| experiment_pH | -0.030114 |
| soil_native_pH | -0.004899 |
| soil_cec_meq_100g | -0.003010 |
| soil_Ocala | -0.001465 |

## Top Permutation Importance Features

| Feature | Mean RMSE increase | Std |
|---|---:|---:|
| neg_log_c_mol_m3 | 0.840078 | 0.051836 |
| neg_log_s_mol_kg | 0.634944 | 0.037251 |
| experiment_pH | 0.014011 | 0.003263 |
| soil_dcb_extractable_fe_mmol_g | 0.000522 | 0.000708 |
| soil_native_pH | 0.000398 | 0.000629 |
| soil_cec_meq_100g | 0.000137 | 0.000387 |
| soil | 0.000068 | 0.000218 |
| soil_clay_wt_pct | 0.000059 | 0.000217 |

## Residual Pattern

| Soil | Rows | Mean residual | Mean absolute residual |
|---|---:|---:|---:|
| Cecil/Pacolet | 20 | -0.000006 | 0.032491 |
| Holton/Cloudland | 23 | 0.000011 | 0.026360 |
| Kenoma | 15 | 0.000000 | 0.030191 |
| Ocala | 10 | -0.000015 | 0.063283 |

## Interpretation Caveats

- This is an observational/experimental regression analysis, not causal proof.
- The dataset is small and includes repeated measurements within a few soil groups.
- Some soil-level variables are constant within soil, so their coefficients are less robust than row-level experimental variables.
- Strong correlations among pH, `-log C`, and `-log S` mean coefficient signs should be interpreted alongside domain chemistry and permutation importance.

## Generated Outputs

- `reports/tables/phase7_ridge_coefficients.csv`
- `reports/tables/phase7_permutation_importance.csv`
- `reports/tables/phase7_partial_dependence.csv`
- `reports/tables/phase7_residual_summary_by_soil.csv`
- `reports/figures/phase7_ridge_coefficients.png`
- `reports/figures/phase7_permutation_importance.png`
- `reports/figures/phase7_partial_dependence_key_features.png`
- `reports/figures/phase7_actual_vs_predicted.png`
- `reports/figures/phase7_residuals.png`
