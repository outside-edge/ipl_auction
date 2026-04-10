#!/usr/bin/env python3
"""
Multi-year forward validation utilities.

Provides rolling forward validation framework for out-of-sample
evaluation of predictive models.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def rolling_forward_validation(
    model_fn,
    df,
    feature_cols,
    target_col,
    year_col="year",
    test_years=None,
    min_train_size=100,
    verbose=True,
):
    """
    Rolling window forward validation.

    For each test year:
    - Train on all data from years < test_year
    - Evaluate on test_year
    - Track stability across years

    Parameters
    ----------
    model_fn : callable
        Function that takes (X_train, y_train) and returns fitted model
    df : pd.DataFrame
        Full dataset with features, target, and year columns
    feature_cols : list
        Column names to use as features
    target_col : str
        Column name for target variable
    year_col : str
        Column name for year/season
    test_years : list, optional
        Years to use as test sets. If None, uses 2015-2024
    min_train_size : int
        Minimum training set size required
    verbose : bool
        Print progress

    Returns
    -------
    pd.DataFrame
        Results with columns: test_year, train_size, test_size, r2, rmse, mae
    """
    if test_years is None:
        test_years = list(range(2015, 2025))

    results = []

    for test_year in test_years:
        train_mask = df[year_col] < test_year
        test_mask = df[year_col] == test_year

        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, target_col]
        X_test = df.loc[test_mask, feature_cols]
        y_test = df.loc[test_mask, target_col]

        if len(X_train) < min_train_size or len(X_test) < 10:
            continue

        try:
            model = model_fn(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)

            results.append({
                "test_year": test_year,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
            })

            if verbose:
                print(f"  {test_year}: R²={r2:.3f}, RMSE={rmse:.2f} (n={len(X_test)})")

        except Exception as e:
            if verbose:
                print(f"  {test_year}: ERROR - {e}")
            continue

    cv_df = pd.DataFrame(results)

    if verbose and len(cv_df) > 0:
        print(f"\n  Mean CV R²: {cv_df['r2'].mean():.3f} ± {cv_df['r2'].std():.3f}")
        print(f"  Mean RMSE:  {cv_df['rmse'].mean():.2f} ± {cv_df['rmse'].std():.2f}")

    return cv_df


def compute_calibration(y_true, y_pred, n_bins=10):
    """
    Assess prediction calibration across deciles.

    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    n_bins : int
        Number of bins for calibration assessment

    Returns
    -------
    pd.DataFrame
        Calibration table with actual vs predicted means by bin
    """
    df = pd.DataFrame({
        "actual": y_true,
        "predicted": y_pred,
    })

    df["decile"] = pd.qcut(df["predicted"], q=n_bins, labels=False, duplicates="drop")

    calibration = df.groupby("decile").agg(
        n=("actual", "count"),
        actual_mean=("actual", "mean"),
        predicted_mean=("predicted", "mean"),
    ).reset_index()

    calibration["error"] = calibration["predicted_mean"] - calibration["actual_mean"]

    return calibration


def compare_to_baseline(y_true, y_pred, y_baseline):
    """
    Compare model predictions to a baseline (e.g., lag-1 WAR).

    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Model predictions
    y_baseline : array-like
        Baseline predictions

    Returns
    -------
    dict
        Comparison metrics
    """
    model_r2 = r2_score(y_true, y_pred)
    model_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    baseline_r2 = r2_score(y_true, y_baseline)
    baseline_rmse = np.sqrt(mean_squared_error(y_true, y_baseline))

    return {
        "model_r2": model_r2,
        "model_rmse": model_rmse,
        "baseline_r2": baseline_r2,
        "baseline_rmse": baseline_rmse,
        "r2_improvement": model_r2 - baseline_r2,
        "rmse_improvement": baseline_rmse - model_rmse,
        "pct_r2_improvement": (model_r2 - baseline_r2) / max(abs(baseline_r2), 0.001) * 100,
    }
