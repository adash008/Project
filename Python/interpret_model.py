from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "chromium_kd_modeling_dataset.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"

NUMERIC_FEATURES = [
    "experiment_pH",
    "neg_log_c_mol_m3",
    "neg_log_s_mol_kg",
    "soil_clay_wt_pct",
    "soil_organic_carbon_wt_pct",
    "soil_iron_oxide_wt_pct",
    "soil_dcb_extractable_fe_mmol_g",
    "soil_native_pH",
    "soil_cec_meq_100g",
]
CATEGORICAL_FEATURES = ["soil"]


def build_model():
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", Ridge(alpha=0.1)),
        ]
    )


def savefig(name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=220)
    plt.close()


def transformed_feature_names(model):
    preprocessor = model.named_steps["preprocess"]
    numeric_names = NUMERIC_FEATURES
    cat_encoder = preprocessor.named_transformers_["categorical"]
    categorical_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return numeric_names + categorical_names


def coefficient_table(model):
    names = transformed_feature_names(model)
    coefs = model.named_steps["model"].coef_
    table = (
        pd.DataFrame({"feature": names, "coefficient": coefs})
        .assign(abs_coefficient=lambda d: d["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )
    return table


def partial_dependence_grid(model, df):
    rows = []
    feature_ranges = {
        "experiment_pH": np.linspace(df["experiment_pH"].min(), df["experiment_pH"].max(), 80),
        "neg_log_c_mol_m3": np.linspace(df["neg_log_c_mol_m3"].min(), df["neg_log_c_mol_m3"].max(), 80),
        "neg_log_s_mol_kg": np.linspace(df["neg_log_s_mol_kg"].min(), df["neg_log_s_mol_kg"].max(), 80),
    }
    base = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for feature, values in feature_ranges.items():
        for value in values:
            temp = base.copy()
            temp[feature] = value
            pred = model.predict(temp)
            rows.append(
                {
                    "feature": feature,
                    "value": value,
                    "mean_predicted_log10_kd": float(np.mean(pred)),
                }
            )
    return pd.DataFrame(rows)


def make_plots(coefs, perm, partial, predictions):
    sns.set_theme(style="whitegrid", context="notebook")

    top_coefs = coefs.head(12).sort_values("coefficient")
    plt.figure(figsize=(9, 6))
    colors = ["#b34d4d" if v < 0 else "#2f6f6d" for v in top_coefs["coefficient"]]
    plt.barh(top_coefs["feature"], top_coefs["coefficient"], color=colors)
    plt.axvline(0, color="#222222", linewidth=1)
    plt.xlabel("Standardized Ridge coefficient")
    plt.title("Optimized Ridge Coefficients")
    savefig("phase7_ridge_coefficients.png")

    top_perm = perm.head(10).sort_values("importance_mean")
    plt.figure(figsize=(9, 6))
    plt.barh(top_perm["feature"], top_perm["importance_mean"], xerr=top_perm["importance_std"], color="#5c6f9c")
    plt.xlabel("Permutation importance, RMSE increase")
    plt.title("Permutation Importance for Optimized Ridge")
    savefig("phase7_permutation_importance.png")

    plt.figure(figsize=(9, 6))
    sns.lineplot(
        data=partial,
        x="value",
        y="mean_predicted_log10_kd",
        hue="feature",
        linewidth=2,
    )
    plt.xlabel("Feature value")
    plt.ylabel("Mean predicted log10(Kd)")
    plt.title("One-Way Partial Dependence for Key Experimental Variables")
    savefig("phase7_partial_dependence_key_features.png")

    plt.figure(figsize=(7, 7))
    sns.scatterplot(
        data=predictions,
        x="actual_log10_kd",
        y="predicted_log10_kd",
        hue="soil",
        s=70,
    )
    lower = min(predictions["actual_log10_kd"].min(), predictions["predicted_log10_kd"].min())
    upper = max(predictions["actual_log10_kd"].max(), predictions["predicted_log10_kd"].max())
    plt.plot([lower, upper], [lower, upper], color="#222222", linestyle="--", linewidth=1)
    plt.xlabel("Actual log10(Kd)")
    plt.ylabel("Predicted log10(Kd)")
    plt.title("Optimized Ridge Fitted Predictions")
    savefig("phase7_actual_vs_predicted.png")

    plt.figure(figsize=(9, 5))
    sns.scatterplot(
        data=predictions,
        x="predicted_log10_kd",
        y="residual_log10_kd",
        hue="soil",
        s=70,
    )
    plt.axhline(0, color="#222222", linestyle="--", linewidth=1)
    plt.xlabel("Predicted log10(Kd)")
    plt.ylabel("Residual log10(Kd)")
    plt.title("Optimized Ridge Residuals")
    savefig("phase7_residuals.png")


def write_summary(coefs, perm, partial, predictions):
    top_positive = coefs[coefs["coefficient"] > 0].head(5)
    top_negative = coefs[coefs["coefficient"] < 0].head(5)

    lines = [
        "# Phase 7 Interpretability Summary",
        "",
        "## Model Interpreted",
        "",
        "- Optimized model: Ridge Regression",
        "- Alpha: 0.1",
        "- Target: `log10_kd`",
        "- Rows interpreted: positive-Kd records only",
        "",
        "## Main Scientific Findings",
        "",
        "- The strongest experimental predictors are `-log C`, `-log S`, and experiment pH.",
        "- Higher `-log C` is associated with higher predicted `log10_kd`. Because `-log C` increases as dissolved concentration decreases, this is consistent with stronger apparent sorption at lower aqueous chromium concentrations.",
        "- Higher `-log S` is associated with lower predicted `log10_kd` in this dataset. This should be interpreted cautiously because `-log S` is tied to experimental sorbed/solid-phase quantities and covaries strongly with other experimental variables.",
        "- Experiment pH has a negative coefficient, matching chromium(VI) sorption chemistry: higher pH generally reduces Cr(VI) adsorption and lowers Kd.",
        "- Soil-level features have smaller direct coefficients after accounting for experimental variables, but DCB-extractable iron remains scientifically important because it reflects Cr(VI)-reducing capacity discussed in the literature.",
        "- Zero-Kd Ocala records remain outside the log-regression target and should be documented as a limitation/future two-stage modeling opportunity.",
        "",
        "## Top Positive Coefficients",
        "",
        "| Feature | Coefficient |",
        "|---|---:|",
    ]
    for row in top_positive.itertuples(index=False):
        lines.append(f"| {row.feature} | {row.coefficient:.6f} |")

    lines += [
        "",
        "## Top Negative Coefficients",
        "",
        "| Feature | Coefficient |",
        "|---|---:|",
    ]
    for row in top_negative.itertuples(index=False):
        lines.append(f"| {row.feature} | {row.coefficient:.6f} |")

    lines += [
        "",
        "## Top Permutation Importance Features",
        "",
        "| Feature | Mean RMSE increase | Std |",
        "|---|---:|---:|",
    ]
    for row in perm.head(8).itertuples(index=False):
        lines.append(f"| {row.feature} | {row.importance_mean:.6f} | {row.importance_std:.6f} |")

    residual_by_soil = (
        predictions.groupby("soil")
        .agg(
            rows=("residual_log10_kd", "size"),
            mean_residual=("residual_log10_kd", "mean"),
            mean_abs_residual=("residual_log10_kd", lambda s: s.abs().mean()),
        )
        .reset_index()
    )
    residual_by_soil.to_csv(TABLES / "phase7_residual_summary_by_soil.csv", index=False)

    lines += [
        "",
        "## Residual Pattern",
        "",
        "| Soil | Rows | Mean residual | Mean absolute residual |",
        "|---|---:|---:|---:|",
    ]
    for row in residual_by_soil.itertuples(index=False):
        lines.append(
            f"| {row.soil} | {row.rows} | {row.mean_residual:.6f} | {row.mean_abs_residual:.6f} |"
        )

    lines += [
        "",
        "## Interpretation Caveats",
        "",
        "- This is an observational/experimental regression analysis, not causal proof.",
        "- The dataset is small and includes repeated measurements within a few soil groups.",
        "- Some soil-level variables are constant within soil, so their coefficients are less robust than row-level experimental variables.",
        "- Strong correlations among pH, `-log C`, and `-log S` mean coefficient signs should be interpreted alongside domain chemistry and permutation importance.",
        "",
        "## Generated Outputs",
        "",
        "- `reports/tables/phase7_ridge_coefficients.csv`",
        "- `reports/tables/phase7_permutation_importance.csv`",
        "- `reports/tables/phase7_partial_dependence.csv`",
        "- `reports/tables/phase7_residual_summary_by_soil.csv`",
        "- `reports/figures/phase7_ridge_coefficients.png`",
        "- `reports/figures/phase7_permutation_importance.png`",
        "- `reports/figures/phase7_partial_dependence_key_features.png`",
        "- `reports/figures/phase7_actual_vs_predicted.png`",
        "- `reports/figures/phase7_residuals.png`",
    ]

    (RESULTS / "phase7_interpretability_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET)
    positive = df[df["kd_positive"]].copy()
    x = positive[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = positive["log10_kd"]

    model = build_model()
    model.fit(x, y)

    predictions = positive[["soil", "experiment_pH", "neg_log_c_mol_m3", "neg_log_s_mol_kg", "kd_ml_g", "log10_kd"]].copy()
    predictions["predicted_log10_kd"] = model.predict(x)
    predictions["residual_log10_kd"] = predictions["log10_kd"] - predictions["predicted_log10_kd"]
    predictions = predictions.rename(columns={"log10_kd": "actual_log10_kd"})
    predictions.to_csv(RESULTS / "phase7_ridge_predictions.csv", index=False)

    coefs = coefficient_table(model)
    coefs.to_csv(TABLES / "phase7_ridge_coefficients.csv", index=False)

    perm_raw = permutation_importance(
        model,
        x,
        y,
        scoring="neg_root_mean_squared_error",
        n_repeats=100,
        random_state=42,
    )
    perm = (
        pd.DataFrame(
            {
                "feature": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
                "importance_mean": perm_raw.importances_mean,
                "importance_std": perm_raw.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    perm.to_csv(TABLES / "phase7_permutation_importance.csv", index=False)

    partial = partial_dependence_grid(model, positive)
    partial.to_csv(TABLES / "phase7_partial_dependence.csv", index=False)

    make_plots(coefs, perm, partial, predictions)
    write_summary(coefs, perm, partial, predictions)

    print("Top coefficients:")
    print(coefs.head(12).to_string(index=False))
    print("\nTop permutation importance:")
    print(perm.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
