"""Train expected claim cost models and export final ML visuals."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import GammaRegressor, PoissonRegressor, TweedieRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_gamma_deviance,
    mean_poisson_deviance,
    mean_squared_error,
    mean_tweedie_deviance,
    root_mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from fremtpl2_utils import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    OUTPUT_DIR,
    ensure_directories,
    load_modeling_data,
    write_text,
)


MODEL_DIR = OUTPUT_DIR / "modeling"


def rmse(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    return float(root_mean_squared_error(y_true, y_pred))


def positive(values: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=float), floor)


def linear_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def tree_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def metrics_row(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float | str]:
    pred = positive(y_pred, floor=0.0)
    y_nonneg = np.maximum(np.asarray(y_true, dtype=float), 0.0)
    row: dict[str, float | str] = {
        "Model": name,
        "MAE": mean_absolute_error(y_nonneg, pred),
        "RMSE": rmse(y_nonneg, pred),
        "R2": r2_score(y_nonneg, pred),
    }
    try:
        row["TweedieDeviance_p1_5"] = mean_tweedie_deviance(
            y_nonneg,
            positive(pred),
            power=1.5,
        )
    except ValueError:
        row["TweedieDeviance_p1_5"] = np.nan
    return row


def savefig(name: str) -> None:
    path = MODEL_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_actual_predicted(y_true: pd.Series, y_pred: np.ndarray, name: str) -> None:
    sample_n = min(len(y_true), 25_000)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_true), size=sample_n, replace=False)
    actual = np.asarray(y_true)[idx]
    pred = np.asarray(y_pred)[idx]

    plt.figure(figsize=(7, 6))
    plt.scatter(np.log1p(actual), np.log1p(positive(pred, 0.0)), s=8, alpha=0.22)
    limit = max(np.log1p(actual).max(), np.log1p(positive(pred, 0.0)).max())
    plt.plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=1)
    plt.title("Actual vs Predicted Pure Premium")
    plt.xlabel("Actual log(1 + Pure Premium)")
    plt.ylabel("Predicted log(1 + Pure Premium)")
    savefig(name)


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, name: str) -> None:
    sample_n = min(len(y_true), 25_000)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_true), size=sample_n, replace=False)
    actual = np.asarray(y_true)[idx]
    pred = positive(np.asarray(y_pred)[idx], 0.0)
    residual = np.log1p(actual) - np.log1p(pred)

    plt.figure(figsize=(7, 5))
    plt.scatter(np.log1p(pred), residual, s=8, alpha=0.22)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Residuals vs Predicted Pure Premium")
    plt.xlabel("Predicted log(1 + Pure Premium)")
    plt.ylabel("Log residual")
    savefig(name)


def model_comparison_plot(metrics: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    order = metrics.sort_values("RMSE")["Model"]
    sns.barplot(data=metrics, y="Model", x="RMSE", order=order, color="#2563eb")
    plt.title("Model Comparison by RMSE")
    plt.xlabel("RMSE")
    plt.ylabel("")
    savefig("01_model_comparison_rmse.png")


def error_distribution(y_true: pd.Series, y_pred: np.ndarray) -> None:
    error = np.log1p(np.asarray(y_true)) - np.log1p(positive(y_pred, 0.0))
    plt.figure(figsize=(8, 5))
    sns.histplot(error, bins=70, color="#7c3aed")
    plt.title("Prediction Error Distribution")
    plt.xlabel("log(1 + actual) - log(1 + predicted)")
    plt.ylabel("Policy Count")
    savefig("04_prediction_error_distribution.png")


def risk_decile_plot(y_true: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    deciles = pd.DataFrame(
        {
            "ActualPurePremium": np.asarray(y_true),
            "PredictedPurePremium": positive(y_pred, 0.0),
        }
    )
    deciles["RiskDecile"] = pd.qcut(
        deciles["PredictedPurePremium"].rank(method="first"),
        q=10,
        labels=[str(i) for i in range(1, 11)],
    )
    grouped = (
        deciles.groupby("RiskDecile", observed=True)
        .agg(
            ActualPurePremium=("ActualPurePremium", "mean"),
            PredictedPurePremium=("PredictedPurePremium", "mean"),
            Policies=("ActualPurePremium", "size"),
        )
        .reset_index()
    )
    long = grouped.melt(
        id_vars=["RiskDecile"],
        value_vars=["ActualPurePremium", "PredictedPurePremium"],
        var_name="Series",
        value_name="PurePremium",
    )

    plt.figure(figsize=(9, 5))
    sns.lineplot(data=long, x="RiskDecile", y="PurePremium", hue="Series", marker="o")
    plt.title("Actual vs Predicted Pure Premium by Predicted Risk Decile")
    plt.xlabel("Predicted Risk Decile")
    plt.ylabel("Average Pure Premium")
    savefig("05_risk_decile_calibration.png")
    return grouped


def permutation_importance_plot(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    sample_n = min(len(X_test), 15_000)
    sample = X_test.sample(sample_n, random_state=42)
    y_sample = y_test.loc[sample.index]
    result = permutation_importance(
        model,
        sample,
        y_sample,
        n_repeats=5,
        random_state=42,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    imp = pd.DataFrame(
        {
            "Feature": X_test.columns,
            "Importance": result.importances_mean,
            "Std": result.importances_std,
        }
    ).sort_values("Importance", ascending=False)
    imp.to_csv(MODEL_DIR / "permutation_importance.csv", index=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=imp.head(12), y="Feature", x="Importance", color="#dc2626")
    plt.title("Permutation Feature Importance")
    plt.xlabel("Decrease in negative MAE when shuffled")
    plt.ylabel("")
    savefig("06_permutation_feature_importance.png")


def train() -> None:
    ensure_directories()
    sns.set_theme(style="whitegrid")
    df = load_modeling_data()

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].copy()
    y_pure = df["PurePremium"].astype(float)
    exposure = df["Exposure"].astype(float)

    X_train, X_test, y_train, y_test, exposure_train, exposure_test = train_test_split(
        X,
        y_pure,
        exposure,
        test_size=0.2,
        random_state=42,
    )

    rows: list[dict[str, float | str]] = []

    baseline_pred = np.full(len(y_test), y_train.mean())
    rows.append(metrics_row("Mean baseline", y_test, baseline_pred))

    frequency_target = (df["ClaimNb"] / df["Exposure"]).astype(float)
    Xf_train, Xf_test, yf_train, yf_test, wf_train, wf_test = train_test_split(
        X,
        frequency_target,
        exposure,
        test_size=0.2,
        random_state=42,
    )
    freq_model = Pipeline(
        [
            ("preprocess", linear_preprocessor()),
            ("model", PoissonRegressor(alpha=1e-4, max_iter=300)),
        ]
    )
    freq_model.fit(Xf_train, yf_train, model__sample_weight=wf_train)
    freq_pred = positive(freq_model.predict(X_test))

    severity_df = df[df["TotalClaimAmount"] > 0].copy()
    X_sev = severity_df[feature_cols]
    y_sev = (severity_df["TotalClaimAmount"] / severity_df["ClaimNb"].clip(lower=1)).astype(float)
    w_sev = severity_df["ClaimNb"].clip(lower=1).astype(float)
    Xsev_train, Xsev_test, ysev_train, ysev_test, wsev_train, wsev_test = train_test_split(
        X_sev,
        y_sev,
        w_sev,
        test_size=0.2,
        random_state=42,
    )
    sev_model = Pipeline(
        [
            ("preprocess", linear_preprocessor()),
            ("model", GammaRegressor(alpha=1e-4, max_iter=500)),
        ]
    )
    sev_model.fit(Xsev_train, ysev_train, model__sample_weight=wsev_train)
    sev_pred_test_positive = positive(sev_model.predict(Xsev_test))
    sev_pred_all = positive(sev_model.predict(X_test))
    product_pred = freq_pred * sev_pred_all
    rows.append(metrics_row("Poisson x Gamma product", y_test, product_pred))

    tweedie = Pipeline(
        [
            ("preprocess", linear_preprocessor()),
            ("model", TweedieRegressor(power=1.5, alpha=1e-4, link="log", max_iter=500)),
        ]
    )
    tweedie.fit(X_train, y_train, model__sample_weight=exposure_train)
    tweedie_pred = positive(tweedie.predict(X_test))
    rows.append(metrics_row("Tweedie pure premium", y_test, tweedie_pred))

    # Tree model is trained on a sample to keep runtime practical on student laptops.
    tree_train_n = min(len(X_train), 180_000)
    tree_idx = X_train.sample(tree_train_n, random_state=42).index
    tree = Pipeline(
        [
            ("preprocess", tree_preprocessor()),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.06,
                    max_iter=220,
                    l2_regularization=0.05,
                    random_state=42,
                    loss="squared_error",
                ),
            ),
        ]
    )
    tree.fit(X_train.loc[tree_idx], y_train.loc[tree_idx])
    tree_pred = positive(tree.predict(X_test), 0.0)
    rows.append(metrics_row("HistGradientBoosting sample", y_test, tree_pred))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(MODEL_DIR / "metrics_summary.csv", index=False)
    metrics.to_string(MODEL_DIR / "metrics_summary.txt", index=False)

    best_name = metrics.sort_values("RMSE").iloc[0]["Model"]
    best_pred = {
        "Mean baseline": baseline_pred,
        "Poisson x Gamma product": product_pred,
        "Tweedie pure premium": tweedie_pred,
        "HistGradientBoosting sample": tree_pred,
    }[str(best_name)]
    best_model = {
        "Poisson x Gamma product": freq_model,
        "Tweedie pure premium": tweedie,
        "HistGradientBoosting sample": tree,
    }.get(str(best_name), tree)

    model_comparison_plot(metrics)
    plot_actual_predicted(y_test, best_pred, "02_actual_vs_predicted_pure_premium.png")
    plot_residuals(y_test, best_pred, "03_residuals_vs_predicted.png")
    error_distribution(y_test, best_pred)
    deciles = risk_decile_plot(y_test, best_pred)
    deciles.to_csv(MODEL_DIR / "risk_decile_calibration.csv", index=False)
    permutation_importance_plot(best_model, X_test, y_test)

    plt.figure(figsize=(8, 5))
    plt.scatter(ysev_test, sev_pred_test_positive, s=10, alpha=0.25)
    limit = max(ysev_test.max(), sev_pred_test_positive.max())
    plt.plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=1)
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Severity Model: Actual vs Predicted Claim Severity")
    plt.xlabel("Actual average claim amount")
    plt.ylabel("Predicted average claim amount")
    savefig("07_severity_actual_vs_predicted.png")

    freq_eval = pd.DataFrame({"ActualFrequency": yf_test, "PredictedFrequency": positive(freq_model.predict(Xf_test))})
    freq_eval["Decile"] = pd.qcut(
        freq_eval["PredictedFrequency"].rank(method="first"),
        q=10,
        labels=[str(i) for i in range(1, 11)],
    )
    freq_group = freq_eval.groupby("Decile", observed=True).mean(numeric_only=True).reset_index()
    freq_long = freq_group.melt(id_vars="Decile", var_name="Series", value_name="Frequency")
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=freq_long, x="Decile", y="Frequency", hue="Series", marker="o")
    plt.title("Frequency Model Calibration by Predicted Decile")
    plt.xlabel("Predicted Frequency Decile")
    plt.ylabel("Claim Frequency")
    savefig("08_frequency_calibration.png")

    joblib.dump(tweedie, MODEL_DIR / "tweedie_pure_premium_model.joblib")
    joblib.dump(tree, MODEL_DIR / "hgb_pure_premium_model.joblib")

    run_summary = f"""Expected Claim Cost modeling run summary

Rows: {len(df):,}
Train rows: {len(X_train):,}
Test rows: {len(X_test):,}
Features: {", ".join(feature_cols)}
Primary target: PurePremium = TotalClaimAmount / Exposure
Best model by RMSE: {best_name}

Metrics:
{metrics.to_string(index=False)}

Generated ML visuals:
01_model_comparison_rmse.png
02_actual_vs_predicted_pure_premium.png
03_residuals_vs_predicted.png
04_prediction_error_distribution.png
05_risk_decile_calibration.png
06_permutation_feature_importance.png
07_severity_actual_vs_predicted.png
08_frequency_calibration.png
"""
    write_text(MODEL_DIR / "run_summary.txt", run_summary)
    print(run_summary)


if __name__ == "__main__":
    train()
