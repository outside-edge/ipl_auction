#!/usr/bin/env python3
"""
Comprehensive retroactive predictions using expanding window training.

For each IPL season, uses ALL data available at auction time to:
1. Build features using only prior data (lag1/2/3 WAR, T20I WAR, etc.)
2. Train XGBoost on all prior auction+outcome data (expanding window)
3. Predict WAR for current year's auction players
4. Compare to actual WAR and calculate overpaid/underpaid

The key questions:
- Can we predict next-season WAR from prior performance?
- Were players overpaid/underpaid relative to predicted value?
- How well do predictions correlate with actual outcomes?

Output:
    tabs/retroactive_predictions.csv - Per-player predictions
    tabs/retroactive_summary.csv - Per-year metrics
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from shared.names import normalize_name, names_compatible, convert_full_to_initial_format
from shared.io import load_dataset, dataset_exists

DATA_DIR = BASE_DIR / "data"
ACQUISITIONS_DIR = DATA_DIR / "acquisitions"
PERF_DIR = DATA_DIR / "perf" / "ipl"
T20I_DIR = DATA_DIR / "perf" / "t20i"
JOINED_DIR = DATA_DIR / "analysis" / "joined"
TABS_DIR = BASE_DIR / "tabs"


def load_data():
    """Load all required data files."""
    print("Loading data...")

    auction = load_dataset(ACQUISITIONS_DIR / "auction_all_years")
    print(f"  Auction records: {len(auction)}")

    ipl_war = load_dataset(PERF_DIR / "player_season_war")
    print(f"  IPL WAR records: {len(ipl_war)}")

    if dataset_exists(T20I_DIR / "player_year_war"):
        t20i_war = load_dataset(T20I_DIR / "player_year_war")
        print(f"  T20I WAR records: {len(t20i_war)}")
    else:
        t20i_war = pd.DataFrame(columns=["year", "player", "total_war"])
        print("  T20I WAR: not found")

    if dataset_exists(JOINED_DIR / "player_master"):
        player_master = load_dataset(JOINED_DIR / "player_master")
        print(f"  Player master: {len(player_master)}")
    else:
        player_master = pd.DataFrame()
        print("  Player master: not found")

    return auction, ipl_war, t20i_war, player_master


def build_name_index(ipl_war):
    """Build index mapping normalized names to player records."""
    ipl_war = ipl_war.copy()
    ipl_war["player_norm"] = ipl_war["player"].apply(normalize_name)
    ipl_war["player_initial"] = ipl_war["player_norm"].apply(convert_full_to_initial_format)
    return ipl_war


def find_player_history(player_name, ipl_war_indexed, max_season):
    """Find player's IPL history using fuzzy name matching."""
    player_norm = normalize_name(player_name)
    player_initial = convert_full_to_initial_format(player_norm)

    prior_war = ipl_war_indexed[ipl_war_indexed["season"] < max_season]

    exact_match = prior_war[prior_war["player_norm"] == player_norm]
    if len(exact_match) > 0:
        return exact_match.sort_values("season", ascending=False)

    initial_match = prior_war[prior_war["player_initial"] == player_initial]
    if len(initial_match) > 0:
        return initial_match.sort_values("season", ascending=False)

    compatible_matches = []
    for idx, row in prior_war.iterrows():
        if names_compatible(player_norm, row["player_norm"]):
            compatible_matches.append(row)

    if compatible_matches:
        result = pd.DataFrame(compatible_matches)
        return result.sort_values("season", ascending=False)

    return pd.DataFrame()


def find_player_war_for_season(player_name, ipl_war_indexed, season):
    """Find player's WAR for a specific season."""
    player_norm = normalize_name(player_name)
    player_initial = convert_full_to_initial_format(player_norm)

    season_war = ipl_war_indexed[ipl_war_indexed["season"] == season]

    exact_match = season_war[season_war["player_norm"] == player_norm]
    if len(exact_match) > 0:
        return exact_match.iloc[0]["total_war"]

    initial_match = season_war[season_war["player_initial"] == player_initial]
    if len(initial_match) > 0:
        return initial_match.iloc[0]["total_war"]

    for _, row in season_war.iterrows():
        if names_compatible(player_norm, row["player_norm"]):
            return row["total_war"]

    return np.nan


def build_features_for_year(auction_df, ipl_war_indexed, t20i_war, year):
    """Build features for a specific auction year using only prior data."""
    auction_year = auction_df[auction_df["year"] == year].copy()

    if len(auction_year) == 0:
        return auction_year

    auction_year["player_norm"] = auction_year["player_name"].apply(normalize_name)

    auction_year["ipl_war_lag1"] = np.nan
    auction_year["ipl_war_lag2"] = np.nan
    auction_year["ipl_war_lag3"] = np.nan
    auction_year["ipl_career_war"] = np.nan
    auction_year["ipl_seasons_played"] = np.nan
    auction_year["ipl_matches_lag1"] = np.nan

    for idx, row in auction_year.iterrows():
        player_history = find_player_history(
            row["player_name"], ipl_war_indexed, year
        )

        if len(player_history) >= 1:
            auction_year.at[idx, "ipl_war_lag1"] = player_history.iloc[0]["total_war"]
            bf = player_history.iloc[0].get("balls_faced", 0) or 0
            bb = player_history.iloc[0].get("balls_bowled", 0) or 0
            auction_year.at[idx, "ipl_matches_lag1"] = (bf + bb) / 30

        if len(player_history) >= 2:
            auction_year.at[idx, "ipl_war_lag2"] = player_history.iloc[1]["total_war"]

        if len(player_history) >= 3:
            auction_year.at[idx, "ipl_war_lag3"] = player_history.iloc[2]["total_war"]

        if len(player_history) > 0:
            auction_year.at[idx, "ipl_career_war"] = player_history["total_war"].sum()
            auction_year.at[idx, "ipl_seasons_played"] = len(player_history)

    if len(t20i_war) > 0:
        t20i_war_norm = t20i_war.copy()
        t20i_war_norm["player_norm"] = t20i_war_norm["player"].apply(normalize_name)

        auction_year["t20i_war_12m"] = np.nan
        auction_year["t20i_war_24m"] = np.nan
        auction_year["t20i_career_war"] = np.nan

        for idx, row in auction_year.iterrows():
            player_norm = row["player_norm"]

            player_t20i = t20i_war_norm[
                t20i_war_norm["player_norm"] == player_norm
            ].copy()

            if len(player_t20i) == 0:
                continue

            prior_t20i = player_t20i[player_t20i["year"] < year].sort_values(
                "year", ascending=False
            )

            if len(prior_t20i) >= 1:
                auction_year.at[idx, "t20i_war_12m"] = prior_t20i.iloc[0]["total_war"]

            if len(prior_t20i) >= 2:
                auction_year.at[idx, "t20i_war_24m"] = prior_t20i.iloc[:2]["total_war"].sum()

            if len(prior_t20i) > 0:
                auction_year.at[idx, "t20i_career_war"] = prior_t20i["total_war"].sum()
    else:
        auction_year["t20i_war_12m"] = np.nan
        auction_year["t20i_war_24m"] = np.nan
        auction_year["t20i_career_war"] = np.nan

    def compute_trend(r):
        if pd.notna(r["ipl_war_lag1"]) and pd.notna(r["ipl_war_lag2"]):
            return r["ipl_war_lag1"] - r["ipl_war_lag2"]
        return np.nan

    auction_year["ipl_war_trend"] = auction_year.apply(compute_trend, axis=1)

    def compute_avg(r):
        vals = [r.get("ipl_war_lag1"), r.get("ipl_war_lag2"), r.get("ipl_war_lag3")]
        vals = [v for v in vals if pd.notna(v)]
        if vals:
            return np.mean(vals)
        return np.nan

    auction_year["ipl_war_avg_3y"] = auction_year.apply(compute_avg, axis=1)

    auction_year["is_indian"] = (auction_year["nationality"] == "Indian").astype(int)

    mega_auction_years = [2008, 2011, 2014, 2018, 2022, 2025]
    auction_year["is_mega_auction"] = auction_year["year"].isin(mega_auction_years).astype(int)

    auction_year["combined_war_12m"] = (
        auction_year["ipl_war_lag1"].fillna(0) +
        auction_year["t20i_war_12m"].fillna(0) * 0.5
    )
    auction_year["combined_war_24m"] = (
        auction_year["ipl_war_lag1"].fillna(0) +
        auction_year["ipl_war_lag2"].fillna(0) +
        auction_year["t20i_war_12m"].fillna(0) * 0.3 +
        auction_year["t20i_war_24m"].fillna(0) * 0.2
    )

    auction_year["has_ipl_history"] = auction_year["ipl_war_lag1"].notna().astype(int)
    auction_year["has_t20i_history"] = auction_year["t20i_war_12m"].notna().astype(int)

    return auction_year


def get_actual_war(auction_year, ipl_war_indexed, year):
    """Get actual WAR for auction year players."""
    auction_year = auction_year.copy()
    auction_year["actual_war"] = np.nan

    for idx, row in auction_year.iterrows():
        actual = find_player_war_for_season(row["player_name"], ipl_war_indexed, year)
        auction_year.at[idx, "actual_war"] = actual

    return auction_year


def prepare_training_data(auction, ipl_war_indexed, t20i_war, max_year):
    """
    Build features for all auctions in years < max_year.
    Get actual outcomes (WAR in auction year) as target.
    """
    all_years = auction[auction["year"] < max_year]["year"].unique()
    all_years = sorted([y for y in all_years if y >= 2008])

    all_features = []

    for year in all_years:
        year_features = build_features_for_year(auction, ipl_war_indexed, t20i_war, year)
        year_features = get_actual_war(year_features, ipl_war_indexed, year)
        all_features.append(year_features)

    if not all_features:
        return pd.DataFrame(), pd.Series(dtype=float)

    train_df = pd.concat(all_features, ignore_index=True)

    feature_cols = get_feature_cols()
    available_cols = [c for c in feature_cols if c in train_df.columns]

    valid = train_df[train_df["actual_war"].notna()].copy()
    valid = valid[valid["ipl_war_lag1"].notna() | valid["t20i_war_12m"].notna()]

    if len(valid) == 0:
        return pd.DataFrame(), pd.Series(dtype=float)

    X = valid[available_cols].copy()
    y = valid["actual_war"].copy()

    return X, y


def get_feature_cols():
    """Return list of feature columns used for modeling."""
    return [
        "ipl_war_lag1", "ipl_war_lag2", "ipl_war_lag3",
        "ipl_career_war", "ipl_seasons_played",
        "t20i_war_12m", "t20i_career_war",
        "ipl_war_trend", "ipl_war_avg_3y",
        "combined_war_12m", "combined_war_24m",
        "has_ipl_history", "has_t20i_history",
        "is_mega_auction",
    ]


def train_model(X_train, y_train):
    """Train XGBoost model with same hyperparameters as 02_train_war_forecast.py."""
    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)

    params = {
        "objective": "reg:squarederror",
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "min_child_weight": 10,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "random_state": 42,
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train_imp, y_train, verbose=False)

    return model, imputer


def evaluate_year(predictions_df):
    """Calculate metrics for a year's predictions."""
    valid = predictions_df[
        predictions_df["actual_war"].notna() &
        predictions_df["predicted_war"].notna()
    ].copy()

    if len(valid) < 5:
        return None

    r2 = r2_score(valid["actual_war"], valid["predicted_war"])
    rmse = np.sqrt(mean_squared_error(valid["actual_war"], valid["predicted_war"]))
    mae = mean_absolute_error(valid["actual_war"], valid["predicted_war"])

    rho, p_value = spearmanr(valid["predicted_war"], valid["actual_war"])

    result = {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "n_players": len(valid),
        "rank_corr": rho,
        "rank_corr_pvalue": p_value,
    }

    valid["final_price_lakh"] = pd.to_numeric(valid.get("final_price_lakh", np.nan), errors="coerce")
    valid_with_price = valid[valid["final_price_lakh"].notna() & (valid["final_price_lakh"] > 0)]

    if len(valid_with_price) >= 5:
        price_pred_corr, _ = spearmanr(valid_with_price["final_price_lakh"], valid_with_price["predicted_war"])
        price_actual_corr, _ = spearmanr(valid_with_price["final_price_lakh"], valid_with_price["actual_war"])
        result["price_pred_corr"] = price_pred_corr
        result["price_actual_corr"] = price_actual_corr
    else:
        result["price_pred_corr"] = np.nan
        result["price_actual_corr"] = np.nan

    return result


def calculate_overpaid_underpaid(df):
    """
    Calculate overpaid/underpaid flags.

    Overpaid: actual_war < predicted_war AND price > median_price
    Underpaid: actual_war > predicted_war AND price < median_price
    """
    df = df.copy()

    df["final_price_lakh"] = pd.to_numeric(df.get("final_price_lakh", np.nan), errors="coerce")

    valid = df[
        df["actual_war"].notna() &
        df["predicted_war"].notna() &
        df["final_price_lakh"].notna() &
        (df["final_price_lakh"] > 0)
    ].copy()

    if len(valid) == 0:
        df["overpaid"] = False
        df["underpaid"] = False
        return df

    median_price = valid["final_price_lakh"].median()

    df["prediction_error"] = df["actual_war"] - df["predicted_war"]

    df["overpaid"] = (
        (df["actual_war"] < df["predicted_war"]) &
        (df["final_price_lakh"] > median_price)
    )

    df["underpaid"] = (
        (df["actual_war"] > df["predicted_war"]) &
        (df["final_price_lakh"] < median_price)
    )

    return df


def main():
    print("=" * 60)
    print("Comprehensive Retroactive Predictions")
    print("=" * 60)

    auction, ipl_war, t20i_war, _ = load_data()

    ipl_war_indexed = build_name_index(ipl_war)

    test_years = [2009, 2010, 2011, 2012, 2014, 2018, 2021, 2022, 2023, 2025]

    all_predictions = []
    year_summaries = []

    for year in test_years:
        print(f"\n{'='*40}")
        print(f"Year {year}")
        print("=" * 40)

        auction_year = build_features_for_year(auction, ipl_war_indexed, t20i_war, year)
        auction_year = get_actual_war(auction_year, ipl_war_indexed, year)

        with_features = auction_year[
            auction_year["ipl_war_lag1"].notna() |
            auction_year["t20i_war_12m"].notna()
        ]
        print(f"  Players in auction: {len(auction_year)}")
        print(f"  Players with features: {len(with_features)}")
        print(f"  Players with actual WAR: {auction_year['actual_war'].notna().sum()}")

        X_train, y_train = prepare_training_data(auction, ipl_war_indexed, t20i_war, year)

        if len(X_train) < 20:
            print(f"  Skipping: insufficient training data ({len(X_train)} samples)")
            continue

        print(f"  Training data: {len(X_train)} samples from prior years")

        model, imputer = train_model(X_train, y_train)

        feature_cols = get_feature_cols()
        available_cols = [c for c in feature_cols if c in auction_year.columns]

        predict_mask = (
            auction_year["ipl_war_lag1"].notna() |
            auction_year["t20i_war_12m"].notna()
        )
        auction_predict = auction_year[predict_mask].copy()

        if len(auction_predict) == 0:
            print("  No players to predict")
            continue

        X_test = auction_predict[available_cols].copy()
        X_test_imp = imputer.transform(X_test)
        auction_predict["predicted_war"] = model.predict(X_test_imp)

        auction_predict = calculate_overpaid_underpaid(auction_predict)

        metrics = evaluate_year(auction_predict)
        if metrics:
            print(f"  R²: {metrics['r2']:.3f}, RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}")
            print(f"  Rank corr (pred vs actual): {metrics['rank_corr']:.3f} (p={metrics['rank_corr_pvalue']:.3f})")
            if not np.isnan(metrics.get("price_pred_corr", np.nan)):
                print(f"  Price corr with pred WAR: {metrics['price_pred_corr']:.3f}, with actual WAR: {metrics['price_actual_corr']:.3f}")
            print(f"  Players evaluated: {metrics['n_players']}")

            overpaid_count = auction_predict["overpaid"].sum()
            underpaid_count = auction_predict["underpaid"].sum()
            print(f"  Overpaid: {overpaid_count}, Underpaid: {underpaid_count}")

            year_summaries.append({
                "year": year,
                "n_players": metrics["n_players"],
                "r2": metrics["r2"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "rank_corr": metrics["rank_corr"],
                "price_pred_corr": metrics.get("price_pred_corr", np.nan),
                "price_actual_corr": metrics.get("price_actual_corr", np.nan),
                "pct_overpaid": 100 * overpaid_count / len(auction_predict) if len(auction_predict) > 0 else np.nan,
                "pct_underpaid": 100 * underpaid_count / len(auction_predict) if len(auction_predict) > 0 else np.nan,
            })
        else:
            print("  Insufficient data for evaluation")

        all_predictions.append(auction_predict)

    if not all_predictions:
        print("\nNo predictions generated")
        return

    predictions_df = pd.concat(all_predictions, ignore_index=True)

    output_cols = [
        "year", "player_name", "team", "final_price_lakh",
        "ipl_war_lag1", "ipl_career_war", "t20i_war_12m",
        "predicted_war", "actual_war", "prediction_error",
        "overpaid", "underpaid",
    ]
    output_cols = [c for c in output_cols if c in predictions_df.columns]
    predictions_output = predictions_df[output_cols].copy()

    predictions_output["final_price_cr"] = predictions_output["final_price_lakh"] / 100
    predictions_output = predictions_output.drop(columns=["final_price_lakh"], errors="ignore")

    TABS_DIR.mkdir(parents=True, exist_ok=True)
    predictions_path = TABS_DIR / "retroactive_predictions.csv"
    predictions_output.to_csv(predictions_path, index=False)
    print(f"\nSaved predictions to {predictions_path}")

    summary_df = pd.DataFrame(year_summaries)
    summary_path = TABS_DIR / "retroactive_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)

    print(f"\nYears evaluated: {len(summary_df)}")
    print(f"Total players: {summary_df['n_players'].sum()}")
    print(f"\nPer-year results:")
    for _, row in summary_df.iterrows():
        print(f"  {int(row['year'])}: R²={row['r2']:.3f}, ρ={row['rank_corr']:.3f} (n={int(row['n_players'])})")

    print(f"\nMean R²: {summary_df['r2'].mean():.3f} ± {summary_df['r2'].std():.3f}")
    print(f"Mean rank correlation (pred vs actual): {summary_df['rank_corr'].mean():.3f}")
    print(f"Mean price corr with pred WAR: {summary_df['price_pred_corr'].mean():.3f}")
    print(f"Mean price corr with actual WAR: {summary_df['price_actual_corr'].mean():.3f}")
    print(f"Mean RMSE: {summary_df['rmse'].mean():.2f}")
    print(f"Mean % Overpaid: {summary_df['pct_overpaid'].mean():.1f}%")

    print("\n" + "=" * 60)
    print("TOP OVERPAID PLAYERS (by prediction error)")
    print("=" * 60)

    valid_predictions = predictions_output[
        predictions_output["actual_war"].notna() &
        predictions_output["predicted_war"].notna()
    ].copy()

    if len(valid_predictions) > 0:
        overpaid_most = valid_predictions.nsmallest(10, "prediction_error")
        print("\nMost overpaid (actual << predicted):")
        for _, row in overpaid_most.iterrows():
            print(
                f"  {int(row['year'])} {row['player_name']:25} | "
                f"Pred: {row['predicted_war']:5.2f} | "
                f"Actual: {row['actual_war']:5.2f} | "
                f"Error: {row['prediction_error']:+5.2f} | "
                f"Price: {row['final_price_cr']:.1f}Cr"
            )

        underpaid_most = valid_predictions.nlargest(10, "prediction_error")
        print("\nMost underpaid (actual >> predicted):")
        for _, row in underpaid_most.iterrows():
            print(
                f"  {int(row['year'])} {row['player_name']:25} | "
                f"Pred: {row['predicted_war']:5.2f} | "
                f"Actual: {row['actual_war']:5.2f} | "
                f"Error: {row['prediction_error']:+5.2f} | "
                f"Price: {row['final_price_cr']:.1f}Cr"
            )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
