from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import StackingRegressor

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


def model_specs():
    specs = []

    specs.append(
        (
            "linear_regression",
            lambda _: Pipeline(
                [
                    ("preprocess", build_preprocessor(scale_numeric=True)),
                    ("model", LinearRegression()),
                ]
            ),
            [{}],
        )
    )

    specs.append(
        (
            "ridge",
            lambda p: Pipeline(
                [
                    ("preprocess", build_preprocessor(scale_numeric=True)),
                    ("model", Ridge(alpha=p["alpha"])),
                ]
            ),
            [{"alpha": alpha} for alpha in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]],
        )
    )

    specs.append(
        (
            "elastic_net",
            lambda p: Pipeline(
                [
                    ("preprocess", build_preprocessor(scale_numeric=True)),
                    (
                        "model",
                        ElasticNet(
                            alpha=p["alpha"],
                            l1_ratio=p["l1_ratio"],
                            random_state=42,
                            max_iter=10000,
                        ),
                    ),
                ]
            ),
            [
                {"alpha": alpha, "l1_ratio": l1_ratio}
                for alpha, l1_ratio in product([0.001, 0.003, 0.01, 0.03, 0.1], [0.1, 0.2, 0.5, 0.8])
            ],
        )
    )

    specs.append(
        (
            "random_forest",
            lambda p: Pipeline(
                [
                    ("preprocess", build_preprocessor()),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=p["n_estimators"],
                            max_depth=p["max_depth"],
                            min_samples_leaf=p["min_samples_leaf"],
                            max_features=p["max_features"],
                            random_state=42,
                        ),
                    ),
                ]
            ),
            [
                {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "max_features": max_features,
                }
                for n_estimators, max_depth, min_samples_leaf, max_features in product(
                    [200, 500],
                    [2, 3, None],
                    [1, 2, 4],
                    [0.6, 1.0],
                )
            ],
        )
    )

    specs.append(
        (
            "extra_trees",
            lambda p: Pipeline(
                [
                    ("preprocess", build_preprocessor()),
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_estimators=p["n_estimators"],
                            max_depth=p["max_depth"],
                            min_samples_leaf=p["min_samples_leaf"],
                            max_features=p["max_features"],
                            random_state=42,
                        ),
                    ),
                ]
            ),
            [
                {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "max_features": max_features,
                }
                for n_estimators, max_depth, min_samples_leaf, max_features in product(
                    [200, 500],
                    [2, 3, None],
                    [1, 2, 4],
                    [0.6, 1.0],
                )
            ],
        )
    )

    specs.append(
        (
            "gradient_boosting",
            lambda p: Pipeline(
                [
                    ("preprocess", build_preprocessor()),
                    (
                        "model",
                        GradientBoostingRegressor(
                            n_estimators=p["n_estimators"],
                            learning_rate=p["learning_rate"],
                            max_depth=p["max_depth"],
                            min_samples_leaf=p["min_samples_leaf"],
                            random_state=42,
                        ),
                    ),
                ]
            ),
            [
                {
                    "n_estimators": n_estimators,
                    "learning_rate": learning_rate,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                }
                for n_estimators, learning_rate, max_depth, min_samples_leaf in product(
                    [50, 100, 200],
                    [0.03, 0.05, 0.1],
                    [1, 2, 3],
                    [1, 2, 4],
                )
            ],
        )
    )

    if XGBRegressor is not None:
        specs.append(
            (
                "xgboost",
                lambda p: Pipeline(
                    [
                        ("preprocess", build_preprocessor()),
                        (
                            "model",
                            XGBRegressor(
                                n_estimators=p["n_estimators"],
                                learning_rate=p["learning_rate"],
                                max_depth=p["max_depth"],
                                subsample=p["subsample"],
                                colsample_bytree=p["colsample_bytree"],
                                reg_lambda=p["reg_lambda"],
                                objective="reg:squarederror",
                                random_state=42,
                            ),
                        ),
                    ]
                ),
                [
                    {
                        "n_estimators": n_estimators,
                        "learning_rate": learning_rate,
                        "max_depth": max_depth,
                        "subsample": subsample,
                        "colsample_bytree": colsample_bytree,
                        "reg_lambda": reg_lambda,
                    }
                    for n_estimators, learning_rate, max_depth, subsample, colsample_bytree, reg_lambda in product(
                        [50, 100, 200],
                        [0.03, 0.05, 0.1],
                        [1, 2, 3],
                        [0.8, 1.0],
                        [0.8, 1.0],
                        [1.0, 5.0],
                    )
                ],
            )
        )

    return specs


def evaluate_logo(model, x, y, groups):
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
    fold_metrics = pd.DataFrame(rows)
    return {
        "rmse": float(fold_metrics["rmse"].mean()),
        "mae": float(fold_metrics["mae"].mean()),
        "r2": float(fold_metrics["r2"].mean()),
        "fold_metrics": fold_metrics,
    }


def build_stack():
    estimators = [
        (
            "linear",
            Pipeline(
                [
                    ("preprocess", build_preprocessor(scale_numeric=True)),
                    ("model", LinearRegression()),
                ]
            ),
        ),
        (
            "ridge",
            Pipeline(
                [
                    ("preprocess", build_preprocessor(scale_numeric=True)),
                    ("model", Ridge(alpha=0.01)),
                ]
            ),
        ),
        (
            "extra_trees",
            Pipeline(
                [
                    ("preprocess", build_preprocessor()),
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_estimators=300,
                            max_depth=2,
                            min_samples_leaf=2,
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
    ]
    return StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=3,
        passthrough=False,
    )


def write_summary(results, best_row, best_fold_metrics):
    lines = [
        "# Phase 6 Optimization Summary",
        "",
        "## Optimization Strategy",
        "",
        "The tuning pass used leave-one-soil-out validation as the primary objective. This avoids selecting models that perform well only because records from the same soil appear in both train and validation folds.",
        "",
        "Models tuned:",
        "",
        "- Linear Regression baseline",
        "- Ridge",
        "- ElasticNet",
        "- Random Forest",
        "- ExtraTrees",
        "- Gradient Boosting",
        "- XGBoost",
        "- A small stacked ensemble",
        "",
        "## Best Tuned Model",
        "",
        f"- Model: `{best_row['model']}`",
        f"- Parameters: `{best_row['params']}`",
        f"- Leave-one-soil-out RMSE: {best_row['rmse']:.6f}",
        f"- Leave-one-soil-out MAE: {best_row['mae']:.6f}",
        f"- Leave-one-soil-out R2: {best_row['r2']:.6f}",
        "",
        "## Top 10 Configurations",
        "",
        "| Rank | Model | RMSE | MAE | R2 | Params |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for rank, row in enumerate(results.head(10).itertuples(index=False), start=1):
        lines.append(
            f"| {rank} | {row.model} | {row.rmse:.6f} | {row.mae:.6f} | {row.r2:.6f} | `{row.params}` |"
        )

    lines += [
        "",
        "## Best Model Held-Out Soil Diagnostics",
        "",
        "| Held-out soil | Rows | RMSE | MAE | R2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in best_fold_metrics.itertuples(index=False):
        lines.append(
            f"| {row.held_out_soil} | {row.rows} | {row.rmse:.6f} | {row.mae:.6f} | {row.r2:.6f} |"
        )

    lines += [
        "",
        "## Optimization Decision",
        "",
        "The tuning pass should be accepted only if it improves leave-one-soil-out RMSE without making the model unnecessarily complex. If the best tuned model is only marginally better than Linear Regression, the final report should prefer the simpler model or present both as candidate models.",
    ]
    (RESULTS / "phase6_optimization_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATASET)
    positive = df[df["kd_positive"]].copy()
    x = positive[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = positive["log10_kd"]
    groups = positive["soil"]

    rows = []
    fold_metric_by_key = {}
    for model_name, factory, grid in model_specs():
        for params in grid:
            model = factory(params)
            result = evaluate_logo(model, x, y, groups)
            key = (model_name, str(params))
            fold_metric_by_key[key] = result["fold_metrics"]
            rows.append(
                {
                    "model": model_name,
                    "params": str(params),
                    "rmse": result["rmse"],
                    "mae": result["mae"],
                    "r2": result["r2"],
                }
            )

    stack = build_stack()
    stack_result = evaluate_logo(stack, x, y, groups)
    rows.append(
        {
            "model": "stacking_linear_ridge_extratrees",
            "params": "{}",
            "rmse": stack_result["rmse"],
            "mae": stack_result["mae"],
            "r2": stack_result["r2"],
        }
    )
    fold_metric_by_key[("stacking_linear_ridge_extratrees", "{}")] = stack_result[
        "fold_metrics"
    ]

    results = pd.DataFrame(rows).sort_values(["rmse", "mae"]).reset_index(drop=True)
    results.to_csv(RESULTS / "phase6_tuning_results.csv", index=False)

    best_row = results.iloc[0]
    best_key = (best_row["model"], best_row["params"])
    best_fold_metrics = fold_metric_by_key[best_key]
    best_fold_metrics.to_csv(RESULTS / "phase6_best_model_fold_metrics.csv", index=False)
    write_summary(results, best_row, best_fold_metrics)

    print(results.head(20).to_string(index=False))
    print("\nBest held-out soil diagnostics:")
    print(best_fold_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
