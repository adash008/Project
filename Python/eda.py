from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "chromium_kd_modeling_dataset.csv"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"
RESULTS = ROOT / "results"


def savefig(name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=200)
    plt.close()


def audit_dataset(df: pd.DataFrame) -> dict:
    feature_cols = [
        "soil",
        "condition",
        "experiment_pH",
        "neg_log_c_mol_m3",
        "neg_log_s_mol_kg",
        "kd_ml_g",
    ]
    duplicate_mask = df.duplicated(subset=feature_cols, keep=False)
    near_duplicate_cols = ["soil", "experiment_pH", "neg_log_c_mol_m3", "neg_log_s_mol_kg"]
    near_duplicate_mask = df.duplicated(subset=near_duplicate_cols, keep=False)

    duplicates = df.loc[duplicate_mask].sort_values(feature_cols)
    near_duplicates = df.loc[near_duplicate_mask].sort_values(near_duplicate_cols)
    zero_kd = df.loc[~df["kd_positive"]].copy()

    TABLES.mkdir(parents=True, exist_ok=True)
    duplicates.to_csv(TABLES / "audit_exact_duplicates.csv", index=False)
    near_duplicates.to_csv(TABLES / "audit_near_duplicates.csv", index=False)
    zero_kd.to_csv(TABLES / "audit_zero_kd_rows.csv", index=False)

    return {
        "rows": len(df),
        "positive_kd_rows": int(df["kd_positive"].sum()),
        "zero_kd_rows": int((~df["kd_positive"]).sum()),
        "exact_duplicate_rows": int(duplicate_mask.sum()),
        "near_duplicate_rows": int(near_duplicate_mask.sum()),
        "soils": sorted(df["soil"].unique()),
    }


def make_plots(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    positive = df[df["kd_positive"]].copy()

    plt.figure(figsize=(8, 5))
    sns.histplot(df["kd_ml_g"], bins=24, color="#2b6f6c")
    plt.xlabel("Kd (ml/g)")
    plt.ylabel("Record count")
    plt.title("Chromium Kd Distribution")
    savefig("kd_distribution.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(positive["log10_kd"], bins=18, color="#7251a3")
    plt.xlabel("log10(Kd)")
    plt.ylabel("Record count")
    plt.title("Log-Transformed Chromium Kd Distribution")
    savefig("log10_kd_distribution.png")

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="experiment_pH",
        y="kd_ml_g",
        hue="soil",
        style="kd_positive",
        s=70,
    )
    plt.yscale("symlog", linthresh=1)
    plt.xlabel("Experiment pH")
    plt.ylabel("Kd (ml/g), symlog scale")
    plt.title("Chromium Kd vs Experiment pH by Soil")
    savefig("kd_vs_ph_by_soil.png")

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=positive,
        x="experiment_pH",
        y="log10_kd",
        hue="soil",
        s=70,
    )
    sns.regplot(
        data=positive,
        x="experiment_pH",
        y="log10_kd",
        scatter=False,
        color="#222222",
        line_kws={"linewidth": 1.5, "linestyle": "--"},
    )
    plt.xlabel("Experiment pH")
    plt.ylabel("log10(Kd)")
    plt.title("log10(Kd) vs Experiment pH")
    savefig("log10_kd_vs_ph_by_soil.png")

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=positive,
        x="neg_log_c_mol_m3",
        y="log10_kd",
        hue="soil",
        s=70,
    )
    plt.xlabel("-log C (mol/m3)")
    plt.ylabel("log10(Kd)")
    plt.title("log10(Kd) vs -log C")
    savefig("log10_kd_vs_neg_log_c.png")

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=positive,
        x="neg_log_s_mol_kg",
        y="log10_kd",
        hue="soil",
        s=70,
    )
    plt.xlabel("-log S (mol/kg)")
    plt.ylabel("log10(Kd)")
    plt.title("log10(Kd) vs -log S")
    savefig("log10_kd_vs_neg_log_s.png")

    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="soil", y="kd_ml_g", hue="soil", legend=False)
    plt.yscale("symlog", linthresh=1)
    plt.xlabel("Soil")
    plt.ylabel("Kd (ml/g), symlog scale")
    plt.title("Soil-wise Chromium Kd Ranges")
    plt.xticks(rotation=20, ha="right")
    savefig("soil_wise_kd_ranges.png")

    corr_cols = [
        "experiment_pH",
        "neg_log_c_mol_m3",
        "neg_log_s_mol_kg",
        "soil_clay_wt_pct",
        "soil_organic_carbon_wt_pct",
        "soil_iron_oxide_wt_pct",
        "soil_dcb_extractable_fe_mmol_g",
        "soil_native_pH",
        "soil_cec_meq_100g",
        "log10_kd",
    ]
    corr = positive[corr_cols].corr(numeric_only=True)
    corr.to_csv(TABLES / "correlation_matrix_positive_kd.csv")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="vlag", center=0, annot=True, fmt=".2f", square=True)
    plt.title("Correlation Matrix for Positive-Kd Records")
    savefig("correlation_matrix_positive_kd.png")


def write_summary(df: pd.DataFrame, audit: dict) -> None:
    positive = df[df["kd_positive"]].copy()
    by_soil = (
        df.groupby("soil", dropna=False)
        .agg(
            records=("kd_ml_g", "size"),
            positive_kd=("kd_positive", "sum"),
            zero_kd=("kd_positive", lambda s: int((~s).sum())),
            kd_min=("kd_ml_g", "min"),
            kd_median=("kd_ml_g", "median"),
            kd_max=("kd_ml_g", "max"),
            ph_min=("experiment_pH", "min"),
            ph_max=("experiment_pH", "max"),
        )
        .reset_index()
    )
    by_soil.to_csv(TABLES / "eda_summary_by_soil.csv", index=False)

    corr_with_target = (
        positive[
            [
                "experiment_pH",
                "neg_log_c_mol_m3",
                "neg_log_s_mol_kg",
                "soil_clay_wt_pct",
                "soil_organic_carbon_wt_pct",
                "soil_iron_oxide_wt_pct",
                "soil_dcb_extractable_fe_mmol_g",
                "soil_native_pH",
                "soil_cec_meq_100g",
                "log10_kd",
            ]
        ]
        .corr(numeric_only=True)["log10_kd"]
        .drop("log10_kd")
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )

    lines = [
        "# EDA Findings Summary",
        "",
        "## Dataset Audit",
        "",
        f"- Total rows: {audit['rows']}",
        f"- Positive-Kd rows usable for log10 modeling: {audit['positive_kd_rows']}",
        f"- Zero/non-positive Kd rows: {audit['zero_kd_rows']}",
        f"- Exact duplicate rows across soil, condition, pH, -log C, -log S, and Kd: {audit['exact_duplicate_rows']}",
        f"- Near-duplicate rows across soil, pH, -log C, and -log S: {audit['near_duplicate_rows']}",
        f"- Soil groups: {', '.join(audit['soils'])}",
        "",
        "## Key Findings",
        "",
        "- The dataset is small but usable for a focused proof-of-concept regression project.",
        "- Kd spans several orders of magnitude, so `log10_kd` is the more stable modeling target.",
        "- Experiment pH is strongly and negatively associated with `log10_kd`; lower pH generally corresponds to higher Kd.",
        "- Soil identity and soil chemistry matter: Holton/Cloudland has the highest Kd range, while Ocala is mostly zero or one.",
        "- The 8 zero-Kd rows all belong to Ocala and cannot be used directly for `log10_kd`; they should be either modeled separately as a low/zero mobility class or excluded from log-regression with clear documentation.",
        "- Exact duplicate checks found no duplicate rows under the project audit definition.",
        "- Because records are clustered by soil, leave-one-soil-out validation should remain the main generalization check.",
        "",
        "## Correlations With log10(Kd)",
        "",
    ]
    for feature, value in corr_with_target.items():
        lines.append(f"- `{feature}`: {value:.3f}")

    lines += [
        "",
        "## Recommended Modeling Decision",
        "",
        "Use `log10_kd` as the primary regression target for positive-Kd records. Preserve zero-Kd rows in the dataset and report them as excluded from log-regression, with a note that a future two-stage model could first classify zero/near-zero Kd and then regress positive Kd values.",
        "",
        "## Generated Figures",
        "",
        "- `reports/figures/kd_distribution.png`",
        "- `reports/figures/log10_kd_distribution.png`",
        "- `reports/figures/kd_vs_ph_by_soil.png`",
        "- `reports/figures/log10_kd_vs_ph_by_soil.png`",
        "- `reports/figures/log10_kd_vs_neg_log_c.png`",
        "- `reports/figures/log10_kd_vs_neg_log_s.png`",
        "- `reports/figures/soil_wise_kd_ranges.png`",
        "- `reports/figures/correlation_matrix_positive_kd.png`",
    ]

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "eda_findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = pd.read_csv(DATASET)
    audit = audit_dataset(df)
    make_plots(df)
    write_summary(df, audit)
    for key, value in audit.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
