#!/usr/bin/env python3
"""
Comprehensive out-of-sample validation for WAR forecasting and price models.

Performs:
1. Rolling forward validation across multiple years (not just 2024)
2. Price model OOS validation (currently missing)
3. Calibration assessment
4. Comparison to baselines
5. Year-by-year validation tables with Spearman rank correlation
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from shared.validation import rolling_forward_validation, compute_calibration, compare_to_baseline
from shared.io import load_dataset

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODEL_DATA_DIR = DATA_DIR / "model"
TABS_DIR = BASE_DIR / "tabs"


def load_data():
    """Load feature data and trained model."""
    print("Loading data...")

    df = load_dataset(MODEL_DATA_DIR / "auction_features_train")
    print(f"  Features: {len(df)} records")

    model_path = MODELS_DIR / "war_forecast_xgb.joblib"
    if model_path.exists():
        model_data = joblib.load(model_path)
        model = model_data["model"]
        imputer = model_data["imputer"]
        feature_cols = model_data["feature_cols"]
        print(f"  Model loaded from {model_path}")
    else:
        model, imputer, feature_cols = None, None, None
        print("  No trained model found")

    return df, model, imputer, feature_cols


def validate_war_model(df, model, imputer, feature_cols):
    """Comprehensive validation of WAR forecasting model."""
    print("\n" + "=" * 60)
    print("WAR MODEL VALIDATION")
    print("=" * 60)

    if model is None:
        print("  No model to validate")
        return None

    df = df.copy()
    df["has_ipl_history"] = df["ipl_war_lag1"].notna().astype(int)
    df["has_t20i_history"] = df["t20i_war_12m"].notna().astype(int)

    valid_df = df[df["next_season_war"].notna()].copy()

    def train_and_predict(X_train, y_train):
        imp = SimpleImputer(strategy="median")
        X_train_imp = imp.fit_transform(X_train)

        import xgboost as xgb
        m = xgb.XGBRegressor(
            objective="reg:squarederror",
            max_depth=3,
            learning_rate=0.05,
            n_estimators=200,
            min_child_weight=10,
            random_state=42
        )
        m.fit(X_train_imp, y_train, verbose=False)

        class ModelWrapper:
            def __init__(self, model, imputer):
                self._model = model
                self._imputer = imputer

            def predict(self, X):
                X_imp = self._imputer.transform(X)
                return self._model.predict(X_imp)

        return ModelWrapper(m, imp)

    print("\n1. Rolling Forward Validation (2015-2024)")
    print("-" * 40)

    cv_results = rolling_forward_validation(
        model_fn=train_and_predict,
        df=valid_df,
        feature_cols=feature_cols,
        target_col="next_season_war",
        year_col="year",
        test_years=list(range(2015, 2025)),
        min_train_size=100,
        verbose=True,
    )

    print("\n2. Calibration Assessment")
    print("-" * 40)

    X_all = valid_df[feature_cols]
    y_all = valid_df["next_season_war"]
    X_all_imp = imputer.transform(X_all)
    y_pred_all = model.predict(X_all_imp)

    calibration = compute_calibration(y_all, y_pred_all, n_bins=5)
    print("\nPrediction vs Actual by quintile:")
    print(f"{'Quintile':<10} {'N':<8} {'Predicted':<12} {'Actual':<12} {'Error':<10}")
    for _, row in calibration.iterrows():
        print(f"{int(row['decile'])+1:<10} {int(row['n']):<8} {row['predicted_mean']:.2f}{'':<5} {row['actual_mean']:.2f}{'':<5} {row['error']:+.2f}")

    print("\n3. Baseline Comparison")
    print("-" * 40)

    valid_with_lag = valid_df[valid_df["ipl_war_lag1"].notna()].copy()
    if len(valid_with_lag) > 50:
        X_sub = valid_with_lag[feature_cols]
        y_sub = valid_with_lag["next_season_war"]
        X_sub_imp = imputer.transform(X_sub)
        y_pred_sub = model.predict(X_sub_imp)
        y_baseline = valid_with_lag["ipl_war_lag1"].values

        comparison = compare_to_baseline(y_sub.values, y_pred_sub, y_baseline)

        print(f"Baseline (lag-1 WAR) R²: {comparison['baseline_r2']:.3f}")
        print(f"Model R²:               {comparison['model_r2']:.3f}")
        print(f"R² improvement:         {comparison['r2_improvement']:.3f} ({comparison['pct_r2_improvement']:.1f}%)")
        print(f"\nBaseline RMSE: {comparison['baseline_rmse']:.2f}")
        print(f"Model RMSE:    {comparison['model_rmse']:.2f}")
        print(f"RMSE improvement: {comparison['rmse_improvement']:.2f}")

    print("\n4. Year-by-Year Stability")
    print("-" * 40)

    if len(cv_results) > 0:
        print(f"R² range: {cv_results['r2'].min():.3f} to {cv_results['r2'].max():.3f}")
        print(f"R² std:   {cv_results['r2'].std():.3f}")

        low_years = cv_results[cv_results["r2"] < 0.1]
        if len(low_years) > 0:
            print(f"\nYears with R² < 0.1 (poor predictions):")
            for _, row in low_years.iterrows():
                print(f"  {int(row['test_year'])}: R²={row['r2']:.3f}, n={int(row['test_size'])}")

    return cv_results


def validate_price_model(df):
    """Out-of-sample validation for price prediction model."""
    print("\n" + "=" * 60)
    print("PRICE MODEL VALIDATION")
    print("=" * 60)

    df = df.copy()
    df["final_price_lakh"] = pd.to_numeric(df["final_price_lakh"], errors="coerce")
    df = df[df["final_price_lakh"] > 0]
    df["log_price"] = np.log(df["final_price_lakh"])

    valid_df = df[
        df["ipl_war_lag1"].notna() &
        df["log_price"].notna()
    ].copy()

    print(f"\nValid observations: {len(valid_df)}")

    import statsmodels.api as sm

    results = []
    for test_year in range(2015, 2025):
        train_mask = valid_df["year"] < test_year
        test_mask = valid_df["year"] == test_year

        if train_mask.sum() < 50 or test_mask.sum() < 10:
            continue

        train = valid_df[train_mask]
        test = valid_df[test_mask]

        X_train = train[["ipl_war_lag1"]].fillna(0)
        X_train = sm.add_constant(X_train)
        y_train = train["log_price"]

        model = sm.OLS(y_train, X_train).fit()

        X_test = test[["ipl_war_lag1"]].fillna(0)
        X_test = sm.add_constant(X_test, has_constant="add")
        y_test = test["log_price"]

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append({
            "test_year": test_year,
            "r2": r2,
            "rmse": rmse,
            "n": len(test),
        })

        print(f"  {test_year}: R²={r2:.3f}, RMSE={rmse:.2f} (n={len(test)})")

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        print(f"\n  Mean R²: {results_df['r2'].mean():.3f} ± {results_df['r2'].std():.3f}")

    return results_df


def create_year_by_year_validation(df, model, imputer, feature_cols):
    """
    Create detailed year-by-year validation table with Spearman rank correlation.

    Returns DataFrame with columns: Year, N, R², Rank_rho, RMSE
    """
    df = df.copy()
    df["has_ipl_history"] = df["ipl_war_lag1"].notna().astype(int)
    df["has_t20i_history"] = df["t20i_war_12m"].notna().astype(int)

    available_cols = [c for c in feature_cols if c in df.columns]

    results = []

    for year in range(2015, 2025):
        train_mask = df["year"] < year
        test_mask = df["year"] == year

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(train_df) < 50 or len(test_df) < 10:
            continue

        X_train = train_df[available_cols]
        y_train = train_df["next_season_war"]
        X_test = test_df[available_cols]
        y_test = test_df["next_season_war"]

        imp = SimpleImputer(strategy="median")
        X_train_imp = imp.fit_transform(X_train)
        X_test_imp = imp.transform(X_test)

        import xgboost as xgb
        m = xgb.XGBRegressor(
            objective="reg:squarederror",
            max_depth=3,
            learning_rate=0.05,
            n_estimators=200,
            min_child_weight=10,
            random_state=42
        )
        m.fit(X_train_imp, y_train, verbose=False)
        y_pred = m.predict(X_test_imp)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        rho, _ = spearmanr(y_test, y_pred)

        lag1_available = test_df["ipl_war_lag1"].notna()
        if lag1_available.sum() >= 5:
            y_naive = test_df.loc[lag1_available, "ipl_war_lag1"]
            y_actual_naive = test_df.loc[lag1_available, "next_season_war"]
            naive_r2 = r2_score(y_actual_naive, y_naive)
            naive_rmse = np.sqrt(mean_squared_error(y_actual_naive, y_naive))
        else:
            naive_r2 = np.nan
            naive_rmse = np.nan

        results.append({
            "Year": year,
            "N": len(test_df),
            "R2": r2,
            "Rank_rho": rho,
            "RMSE": rmse,
            "Naive_R2": naive_r2,
            "Naive_RMSE": naive_rmse,
        })

    return pd.DataFrame(results)


def generate_validation_report(war_cv, price_cv, year_by_year=None):
    """Generate comprehensive validation report."""
    TABS_DIR.mkdir(parents=True, exist_ok=True)

    report = []
    report.append("=" * 60)
    report.append("COMPREHENSIVE MODEL VALIDATION REPORT")
    report.append("=" * 60)
    report.append("")

    report.append("1. WAR FORECASTING MODEL")
    report.append("-" * 40)
    if war_cv is not None and len(war_cv) > 0:
        report.append(f"Rolling CV years: {int(war_cv['test_year'].min())}-{int(war_cv['test_year'].max())}")
        report.append(f"Mean R²: {war_cv['r2'].mean():.3f} ± {war_cv['r2'].std():.3f}")
        report.append(f"Mean RMSE: {war_cv['rmse'].mean():.2f} ± {war_cv['rmse'].std():.2f}")
        report.append("")
        report.append("Year-by-year:")
        for _, row in war_cv.iterrows():
            report.append(f"  {int(row['test_year'])}: R²={row['r2']:.3f}")
    else:
        report.append("No WAR model validation results")
    report.append("")

    report.append("2. PRICE PREDICTION MODEL")
    report.append("-" * 40)
    if price_cv is not None and len(price_cv) > 0:
        report.append(f"Rolling CV years: {int(price_cv['test_year'].min())}-{int(price_cv['test_year'].max())}")
        report.append(f"Mean R²: {price_cv['r2'].mean():.3f} ± {price_cv['r2'].std():.3f}")
        report.append("")
        report.append("Year-by-year:")
        for _, row in price_cv.iterrows():
            report.append(f"  {int(row['test_year'])}: R²={row['r2']:.3f}")
    else:
        report.append("No price model validation results")
    report.append("")

    if year_by_year is not None and len(year_by_year) > 0:
        report.append("3. DETAILED YEAR-BY-YEAR VALIDATION")
        report.append("-" * 40)
        report.append(f"{'Year':<6} {'N':>4} {'R²':>8} {'Rank ρ':>8} {'RMSE':>8} {'Naive R²':>10}")
        for _, row in year_by_year.iterrows():
            naive_str = f"{row['Naive_R2']:.3f}" if pd.notna(row["Naive_R2"]) else "N/A"
            report.append(
                f"{int(row['Year']):<6} {int(row['N']):>4} "
                f"{row['R2']:>8.3f} {row['Rank_rho']:>8.3f} "
                f"{row['RMSE']:>8.2f} {naive_str:>10}"
            )
        report.append("")
        report.append(f"Mean Model R²: {year_by_year['R2'].mean():.3f}")
        report.append(f"Mean Rank ρ:   {year_by_year['Rank_rho'].mean():.3f}")
        valid_naive = year_by_year["Naive_R2"].dropna()
        if len(valid_naive) > 0:
            report.append(f"Mean Naive R²: {valid_naive.mean():.3f}")
        report.append("")

    report.append("4. KEY INSIGHTS")
    report.append("-" * 40)
    if war_cv is not None and len(war_cv) > 0:
        stable_years = war_cv[war_cv["r2"] > 0.15]
        report.append(f"Stable prediction years (R² > 15%): {len(stable_years)}/{len(war_cv)}")

    if year_by_year is not None and len(year_by_year) > 0:
        total_n = year_by_year["N"].sum()
        report.append(f"Total validation observations: {total_n}")

    report_text = "\n".join(report)

    report_path = TABS_DIR / "validation_report.txt"
    report_path.write_text(report_text)
    print(f"\nSaved validation report to {report_path}")

    if year_by_year is not None:
        csv_path = TABS_DIR / "validation_by_year.csv"
        year_by_year.to_csv(csv_path, index=False)
        print(f"Saved year-by-year validation to {csv_path}")

    return report_text


def main():
    print("=" * 60)
    print("Comprehensive Model Validation")
    print("=" * 60)

    df, model, imputer, feature_cols = load_data()

    war_cv = validate_war_model(df, model, imputer, feature_cols)

    price_cv = validate_price_model(df)

    year_by_year = None
    if model is not None and feature_cols is not None:
        print("\n" + "=" * 60)
        print("YEAR-BY-YEAR DETAILED VALIDATION")
        print("=" * 60)
        year_by_year = create_year_by_year_validation(df, model, imputer, feature_cols)

    report = generate_validation_report(war_cv, price_cv, year_by_year)

    print("\n" + report)
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
