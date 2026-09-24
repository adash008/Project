from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "chromium_kd_modeling_dataset.csv"
RESULTS = ROOT / "results"


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


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def build_preprocessor(scale_numeric=False):
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_step, NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_models():
    models = {
        "dummy_mean": Pipeline(
            [("preprocess", build_preprocessor()), ("model", DummyRegressor(strategy="mean"))]
        ),
        "linear_regression": Pipeline(
            [("preprocess", build_preprocessor(scale_numeric=True)), ("model", LinearRegression())]
        ),
        "ridge": Pipeline(
            [("preprocess", build_preprocessor(scale_numeric=True)), ("model", Ridge(alpha=1.0))]
        ),
        "elastic_net": Pipeline(
            [
                ("preprocess", build_preprocessor(scale_numeric=True)),
                ("model", ElasticNet(alpha=0.01, l1_ratio=0.2, random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=2,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=400,
                        min_samples_leaf=2,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                ("model", GradientBoostingRegressor(random_state=42)),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=200,
                        learning_rate=0.04,
                        l2_regularization=0.1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "mlp": Pipeline(
            [
                ("preprocess", build_preprocessor(scale_numeric=True)),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(16, 8),
                        activation="relu",
                        alpha=0.01,
                        max_iter=5000,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }
    if XGBRegressor is not None:
        models["xgboost"] = Pipeline(
            [
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=200,
                        max_depth=2,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="reg:squarederror",
                        random_state=42,
                    ),
                ),
            ]
        )
    return models


def cross_validated_predictions(model, x, y, groups):
    rows = []
    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(x, y, groups):
        fold_model = clone(model)
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        test_groups = groups.iloc[test_idx]
        fold_model.fit(x_train, y_train)
        pred = fold_model.predict(x_test)
        rows.append(
            pd.DataFrame(
                {
                    "held_out_soil": test_groups.values,
                    "actual_log10_kd": y_test.values,
                    "predicted_log10_kd": pred,
                    "residual_log10_kd": y_test.values - pred,
                },
                index=y_test.index,
            )
        )
    return pd.concat(rows).sort_index()


def leave_one_soil_fold_metrics(model, x, y, groups):
    rows = []
    logo = LeaveOneGroupOut()
    for train_idx, test_idx in logo.split(x, y, groups):
        fold_model = clone(model)
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        held_out_soil = groups.iloc[test_idx].iloc[0]
        fold_model.fit(x_train, y_train)
        pred = fold_model.predict(x_test)
        rows.append(
            {
                "held_out_soil": held_out_soil,
                "rows": len(test_idx),
                "rmse": rmse(y_test, pred),
                "mae": mean_absolute_error(y_test, pred),
                "r2": r2_score(y_test, pred) if len(test_idx) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_phase5_summary(metrics, group_metrics, fold_metrics, best_name, importance):
    lines = [
        "# Phase 5 Model Development Summary",
        "",
        "## Scope",
        "",
        "This phase compared baseline and modern regression models for predicting `log10_kd` on positive chromium Kd records.",
        "",
        "The primary model selection criterion is leave-one-soil-out RMSE because records are clustered by soil type.",
        "",
        "## Best Model Under Soil-Aware Validation",
        "",
        f"- Best model: `{best_name}`",
        f"- Leave-one-soil-out RMSE: {group_metrics.iloc[0]['rmse']:.6f}",
        f"- Leave-one-soil-out MAE: {group_metrics.iloc[0]['mae']:.6f}",
        f"- Leave-one-soil-out R2: {group_metrics.iloc[0]['r2']:.6f}",
        "",
        "## Leave-One-Soil-Out Ranking",
        "",
        "| Rank | Model | RMSE | MAE | R2 |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(group_metrics.itertuples(index=False), start=1):
        lines.append(f"| {rank} | {row.model} | {row.rmse:.6f} | {row.mae:.6f} | {row.r2:.6f} |")

    lines += [
        "",
        "## Shuffled 5-Fold Ranking",
        "",
        "| Rank | Model | RMSE | MAE | R2 |",
        "|---:|---|---:|---:|---:|",
    ]
    for rank, row in enumerate(metrics.itertuples(index=False), start=1):
        lines.append(f"| {rank} | {row.model} | {row.rmse:.6f} | {row.mae:.6f} | {row.r2:.6f} |")

    lines += [
        "",
        "## Held-Out Soil Diagnostics for Best Model",
        "",
        "| Held-out soil | Rows | RMSE | MAE | R2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in fold_metrics.itertuples(index=False):
        lines.append(f"| {row.held_out_soil} | {row.rows} | {row.rmse:.6f} | {row.mae:.6f} | {row.r2:.6f} |")

    lines += [
        "",
        "## Permutation Importance for Best Model",
        "",
        "| Feature | Mean importance | Std importance |",
        "|---|---:|---:|",
    ]
    for row in importance.itertuples(index=False):
        lines.append(f"| {row.feature} | {row.importance_mean:.6f} | {row.importance_std:.6f} |")

    lines += [
        "",
        "## Technical Interpretation",
        "",
        "- Linear and regularized models generalize best to unseen soil groups, which is consistent with the dataset being small and strongly structured by experimental chemistry.",
        "- Tree ensembles perform very well under shuffled cross-validation but degrade under leave-one-soil-out validation, indicating some soil-specific memorization.",
        "- MLP is not recommended at this dataset size because it performs poorly in soil-aware validation.",
        "- The final report should present both validation views, but model selection should emphasize leave-one-soil-out results.",
        "",
        "## Recommendation",
        "",
        "Use the best soil-aware model as the current candidate final model. Proceed to Phase 6 with light tuning of linear/regularized models and selected tree ensembles, then Phase 7 interpretability.",
    ]
    (RESULTS / "phase5_model_development_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def evaluate_models(df):
    positive = df[df["kd_positive"]].copy()
    x = positive[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = positive["log10_kd"]
    groups = positive["soil"]

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "neg_rmse": "neg_root_mean_squared_error",
        "neg_mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    rows = []
    group_rows = []
    for name, model in build_models().items():
        scores = cross_validate(model, x, y, cv=cv, scoring=scoring, error_score="raise")
        rows.append(
            {
                "model": name,
                "target": "log10_kd",
                "split_strategy": "5-fold shuffled KFold on positive Kd rows",
                "rmse": -float(np.mean(scores["test_neg_rmse"])),
                "mae": -float(np.mean(scores["test_neg_mae"])),
                "r2": float(np.mean(scores["test_r2"])),
                "notes": "Initial untuned model comparison; soil records are small and clustered.",
            }
        )

        logo = LeaveOneGroupOut()
        group_scores = cross_validate(
            model,
            x,
            y,
            cv=logo,
            groups=groups,
            scoring=scoring,
            error_score="raise",
        )
        group_rows.append(
            {
                "model": name,
                "target": "log10_kd",
                "split_strategy": "leave-one-soil-out on positive Kd rows",
                "rmse": -float(np.mean(group_scores["test_neg_rmse"])),
                "mae": -float(np.mean(group_scores["test_neg_mae"])),
                "r2": float(np.mean(group_scores["test_r2"])),
                "notes": "Harder validation: each fold holds out an entire soil type.",
            }
        )

    metrics = pd.DataFrame(rows).sort_values(["rmse", "mae"])
    metrics.to_csv(RESULTS / "metrics.csv", index=False)
    group_metrics = pd.DataFrame(group_rows).sort_values(["rmse", "mae"])
    group_metrics.to_csv(RESULTS / "metrics_leave_one_soil_out.csv", index=False)

    best_name = group_metrics.iloc[0]["model"]
    best_model = build_models()[best_name]
    best_model.fit(x, y)
    preds = best_model.predict(x)
    fitted = pd.DataFrame(
        {
            "soil": positive["soil"],
            "experiment_pH": positive["experiment_pH"],
            "kd_ml_g": positive["kd_ml_g"],
            "actual_log10_kd": y,
            "predicted_log10_kd": preds,
            "residual_log10_kd": y - preds,
            "model": best_name,
        }
    )
    fitted.to_csv(RESULTS / "best_model_fitted_values.csv", index=False)

    logo_predictions = cross_validated_predictions(best_model, x, y, groups)
    prediction_context = positive.loc[
        logo_predictions.index,
        ["soil", "experiment_pH", "neg_log_c_mol_m3", "neg_log_s_mol_kg", "kd_ml_g"],
    ].copy()
    logo_predictions = pd.concat([prediction_context, logo_predictions], axis=1)
    logo_predictions.to_csv(RESULTS / "best_model_leave_one_soil_predictions.csv", index=False)

    fold_metrics = leave_one_soil_fold_metrics(best_model, x, y, groups)
    fold_metrics.to_csv(RESULTS / "best_model_leave_one_soil_fold_metrics.csv", index=False)

    perm = permutation_importance(
        best_model,
        x,
        y,
        scoring="neg_root_mean_squared_error",
        n_repeats=50,
        random_state=42,
    )
    importance = (
        pd.DataFrame(
            {
                "feature": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(RESULTS / "best_model_permutation_importance.csv", index=False)

    summary = {
        "best_model_by_leave_one_soil_out": best_name,
        "training_rmse_log10": rmse(y, preds),
        "training_mae_log10": mean_absolute_error(y, preds),
        "training_r2_log10": r2_score(y, preds),
        "rows_used": len(positive),
        "rows_excluded_nonpositive_kd": int((~df["kd_positive"]).sum()),
    }
    write_phase5_summary(metrics, group_metrics, fold_metrics, best_name, importance)
    return metrics, group_metrics, fold_metrics, importance, summary


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET)
    metrics, group_metrics, fold_metrics, importance, summary = evaluate_models(df)

    print("Shuffled 5-fold CV:")
    print(metrics.to_string(index=False))
    print("\nLeave-one-soil-out CV:")
    print(group_metrics.to_string(index=False))
    print("\nBest model leave-one-soil-out fold diagnostics:")
    print(fold_metrics.to_string(index=False))
    print("\nBest model permutation importance:")
    print(importance.to_string(index=False))
    print("\nBest model summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
