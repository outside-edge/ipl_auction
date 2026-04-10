#!/usr/bin/env python3
"""
Historical backtest of dud predictions.

For each past auction year, generates dud predictions using only
data available at that time, then evaluates against actual outcomes.

This validates whether our "predicted duds" methodology actually
identifies players who underperformed.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from shared.names import normalize_name
from shared.io import load_dataset

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ACQUISITIONS_DIR = DATA_DIR / "acquisitions"
PERF_DIR = DATA_DIR / "perf" / "ipl"
TABS_DIR = BASE_DIR / "tabs"


def load_data():
    """Load auction and WAR data."""
    print("Loading data...")

    auction = load_dataset(ACQUISITIONS_DIR / "auction_all_years")
    print(f"  Auction records: {len(auction)}")

    ipl_war = load_dataset(PERF_DIR / "player_season_war")
    print(f"  IPL WAR records: {len(ipl_war)}")

    return auction, ipl_war


def create_lag_features(auction, ipl_war, year):
    """Create lagged WAR features for a specific auction year."""
    auction_year = auction[auction["year"] == year].copy()
    ipl_war = ipl_war.copy()

    auction_year["player_norm"] = auction_year["player_name"].apply(normalize_name)
    ipl_war["player_norm"] = ipl_war["player"].apply(normalize_name)

    auction_year["war_lag1"] = np.nan
    auction_year["war_actual"] = np.nan

    for idx, row in auction_year.iterrows():
        player_norm = row["player_norm"]

        prior = ipl_war[
            (ipl_war["player_norm"] == player_norm) &
            (ipl_war["season"] == year - 1)
        ]
        if len(prior) > 0:
            auction_year.at[idx, "war_lag1"] = prior.iloc[0]["total_war"]

        actual = ipl_war[
            (ipl_war["player_norm"] == player_norm) &
            (ipl_war["season"] == year)
        ]
        if len(actual) > 0:
            auction_year.at[idx, "war_actual"] = actual.iloc[0]["total_war"]

    return auction_year


def estimate_price_model(auction, ipl_war, max_year):
    """Estimate price model using only data up to max_year."""
    auction = auction.copy()
    ipl_war = ipl_war.copy()

    auction_hist = auction[(auction["year"] > 2008) & (auction["year"] < max_year)].copy()

    ipl_war["player_norm"] = ipl_war["player"].apply(normalize_name)
    auction_hist["player_norm"] = auction_hist["player_name"].apply(normalize_name)

    auction_hist["war_prior"] = np.nan

    for idx, row in auction_hist.iterrows():
        player_norm = row["player_norm"]
        year = row["year"]

        prior_war = ipl_war[
            (ipl_war["player_norm"] == player_norm) &
            (ipl_war["season"] == year - 1)
        ]

        if len(prior_war) > 0:
            auction_hist.at[idx, "war_prior"] = prior_war.iloc[0]["total_war"]

    valid = auction_hist[
        (auction_hist["war_prior"].notna()) &
        (auction_hist["final_price_lakh"].notna())
    ].copy()

    valid["final_price_lakh"] = pd.to_numeric(valid["final_price_lakh"], errors="coerce")
    valid = valid[valid["final_price_lakh"] > 0]

    if len(valid) < 30:
        return None

    valid["log_price"] = np.log(valid["final_price_lakh"])
    valid["is_indian"] = (valid["nationality"] == "Indian").astype(int)

    X = valid[["war_prior", "is_indian"]]
    X = sm.add_constant(X)
    y = valid["log_price"]

    model = sm.OLS(y, X).fit()
    return model


def calculate_predicted_premium(df, price_model):
    """Calculate premium for each player."""
    df = df.copy()

    df["final_price_lakh"] = pd.to_numeric(df["final_price_lakh"], errors="coerce")
    df = df[df["final_price_lakh"] > 0]

    df["is_indian"] = (df["nationality"] == "Indian").astype(int)

    valid = df[df["war_lag1"].notna()].copy()

    if len(valid) == 0:
        return df

    X_pred = valid[["war_lag1", "is_indian"]].copy()
    X_pred.columns = ["war_prior", "is_indian"]
    X_pred = sm.add_constant(X_pred, has_constant="add")

    valid["log_price_predicted"] = price_model.predict(X_pred)
    valid["price_predicted_lakh"] = np.exp(valid["log_price_predicted"])

    valid["premium_pct"] = (
        (valid["final_price_lakh"] - valid["price_predicted_lakh"])
        / valid["price_predicted_lakh"]
    ) * 100

    return valid


def evaluate_dud_predictions(df):
    """Evaluate whether high-premium players actually underperformed."""
    if "premium_pct" not in df.columns:
        return None

    valid = df[
        df["war_lag1"].notna() &
        df["war_actual"].notna() &
        df["premium_pct"].notna()
    ].copy()

    if len(valid) < 10:
        return None

    valid["underperformance"] = valid["war_lag1"] - valid["war_actual"]

    corr = valid["premium_pct"].corr(valid["underperformance"])

    top_quintile = valid.nlargest(int(len(valid) * 0.2), "premium_pct")
    bottom_quintile = valid.nsmallest(int(len(valid) * 0.2), "premium_pct")

    top_underperf = top_quintile["underperformance"].mean()
    bottom_underperf = bottom_quintile["underperformance"].mean()

    return {
        "n": len(valid),
        "corr_premium_underperf": corr,
        "top_quintile_underperf": top_underperf,
        "bottom_quintile_underperf": bottom_underperf,
        "diff": top_underperf - bottom_underperf,
    }


def main():
    print("=" * 60)
    print("Historical Backtest of Dud Predictions")
    print("=" * 60)

    auction, ipl_war = load_data()

    test_years = list(range(2015, 2025))
    results = []

    for year in test_years:
        print(f"\n--- {year} ---")

        auction_year = create_lag_features(auction, ipl_war, year)

        with_lag = auction_year[auction_year["war_lag1"].notna()]
        with_actual = auction_year[auction_year["war_actual"].notna()]
        print(f"  Players with prior WAR: {len(with_lag)}")
        print(f"  Players with actual WAR: {len(with_actual)}")

        price_model = estimate_price_model(auction, ipl_war, year)
        if price_model is None:
            print("  Insufficient data for price model")
            continue

        with_premium = calculate_predicted_premium(auction_year, price_model)

        evaluation = evaluate_dud_predictions(with_premium)
        if evaluation is None:
            print("  Insufficient data for evaluation")
            continue

        results.append({
            "year": year,
            **evaluation
        })

        print(f"  Correlation (premium vs underperformance): {evaluation['corr_premium_underperf']:.3f}")
        print(f"  Top quintile underperformance: {evaluation['top_quintile_underperf']:.2f}")
        print(f"  Bottom quintile underperformance: {evaluation['bottom_quintile_underperf']:.2f}")
        print(f"  Difference: {evaluation['diff']:.2f}")

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("BACKTEST SUMMARY")
    print("=" * 60)

    if len(results_df) > 0:
        print(f"\nYears tested: {len(results_df)}")
        print(f"Mean correlation (premium vs underperformance): {results_df['corr_premium_underperf'].mean():.3f}")
        print(f"Mean difference (top vs bottom quintile): {results_df['diff'].mean():.2f}")

        positive_corr = (results_df["corr_premium_underperf"] > 0).sum()
        print(f"Years with positive correlation: {positive_corr}/{len(results_df)}")

        positive_diff = (results_df["diff"] > 0).sum()
        print(f"Years where high-premium players underperformed more: {positive_diff}/{len(results_df)}")

        print("\nYear-by-year results:")
        print(results_df.to_string(index=False))

        TABS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = TABS_DIR / "backtest_results.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\nSaved to {output_path}")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("""
If our dud prediction methodology works:
- Correlation should be positive (higher premium -> more underperformance)
- Top quintile (predicted duds) should underperform more than bottom quintile

A positive mean difference indicates that players we identify as "duds"
do tend to underperform relative to their price premium.

Limitations:
- Small sample sizes per year
- Only evaluates players with prior IPL history
- Does not account for all factors teams consider
""")


if __name__ == "__main__":
    main()
