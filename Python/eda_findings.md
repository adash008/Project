# EDA Findings Summary

## Dataset Audit

- Total rows: 76
- Positive-Kd rows usable for log10 modeling: 68
- Zero/non-positive Kd rows: 8
- Exact duplicate rows across soil, condition, pH, -log C, -log S, and Kd: 0
- Near-duplicate rows across soil, pH, -log C, and -log S: 0
- Soil groups: Cecil/Pacolet, Holton/Cloudland, Kenoma, Ocala

## Key Findings

- The dataset is small but usable for a focused proof-of-concept regression project.
- Kd spans several orders of magnitude, so `log10_kd` is the more stable modeling target.
- Experiment pH is strongly and negatively associated with `log10_kd`; lower pH generally corresponds to higher Kd.
- Soil identity and soil chemistry matter: Holton/Cloudland has the highest Kd range, while Ocala is mostly zero or one.
- The 8 zero-Kd rows all belong to Ocala and cannot be used directly for `log10_kd`; they should be either modeled separately as a low/zero mobility class or excluded from log-regression with clear documentation.
- Exact duplicate checks found no duplicate rows under the project audit definition.
- Because records are clustered by soil, leave-one-soil-out validation should remain the main generalization check.

## Correlations With log10(Kd)

- `neg_log_c_mol_m3`: 0.956
- `neg_log_s_mol_kg`: -0.925
- `experiment_pH`: -0.831
- `soil_cec_meq_100g`: -0.658
- `soil_native_pH`: -0.650
- `soil_dcb_extractable_fe_mmol_g`: 0.613
- `soil_iron_oxide_wt_pct`: 0.507
- `soil_organic_carbon_wt_pct`: -0.432
- `soil_clay_wt_pct`: -0.243

## Recommended Modeling Decision

Use `log10_kd` as the primary regression target for positive-Kd records. It is recommended to preserve zero-Kd rows in the dataset and report them as excluded from log-regression, with a note that a future two-stage model could first classify zero/near-zero Kd and then regress positive Kd values.

## Generated Figures

- `reports/figures/kd_distribution.png`
- `reports/figures/log10_kd_distribution.png`
- `reports/figures/kd_vs_ph_by_soil.png`
- `reports/figures/log10_kd_vs_ph_by_soil.png`
- `reports/figures/log10_kd_vs_neg_log_c.png`
- `reports/figures/log10_kd_vs_neg_log_s.png`
- `reports/figures/soil_wise_kd_ranges.png`
- `reports/figures/correlation_matrix_positive_kd.png`
