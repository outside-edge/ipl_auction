#!/usr/bin/env python3
"""
Identify IPL Auction "Big Duds" - players who were overpaid relative to
what teams knew (lagged performance) AND then underperformed.

DUD SCORE = f(Premium Paid, Future Underperformance)
Where:
- Premium Paid = (Actual Price - Predicted Price) / Predicted Price
- Future Underperformance = Expected WAR - Actual Next-Season WAR
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import percentileofscore


def load_data():
    """Load auction and WAR data."""
    base_path = Path(__file__).parent.parent / "data"

    auction = pd.read_csv(base_path / "analysis" / "auction_inflation_adjusted.csv")
    war = pd.read_csv(base_path / "player_season_war.csv")

    return auction, war


def create_lagged_and_future_war(war_df):
    """
    Create lagged WAR (prior season) and future WAR (next season) for each player.
    """
    war_df = war_df.copy()
    war_df = war_df.sort_values(["player", "season"])

    war_df["war_lag"] = war_df.groupby("player")["total_war"].shift(1)
    war_df["war_future"] = war_df.groupby("player")["total_war"].shift(-1)

    war_df["balls_faced_lag"] = war_df.groupby("player")["balls_faced"].shift(1)
    war_df["balls_bowled_lag"] = war_df.groupby("player")["balls_bowled"].shift(1)
    war_df["balls_faced_future"] = war_df.groupby("player")["balls_faced"].shift(-1)
    war_df["balls_bowled_future"] = war_df.groupby("player")["balls_bowled"].shift(-1)

    return war_df


def merge_auction_with_war(auction_df, war_df):
    """Merge auction data with lagged and future WAR."""
    merged = auction_df.merge(
        war_df[
            [
                "season",
                "player",
                "war_lag",
                "war_future",
                "balls_faced_lag",
                "balls_bowled_lag",
                "balls_faced_future",
                "balls_bowled_future",
            ]
        ],
        left_on=["year", "player"],
        right_on=["season", "player"],
        how="left",
    )

    return merged


def filter_valid_observations(df, min_matches=5):
    """
    Filter to valid observations:
    - Exclude 2008 auctions (no prior season)
    - Exclude 2025 auctions (no future season yet)
    - Require minimum activity in prior and future seasons
    """
    df = df.copy()

    df = df[df["year"] > 2008]
    df = df[df["year"] < 2025]

    df = df[df["war_lag"].notna()]
    df = df[df["war_future"].notna()]

    df["prior_activity"] = df["balls_faced_lag"].fillna(0) + df[
        "balls_bowled_lag"
    ].fillna(0)
    df["future_activity"] = df["balls_faced_future"].fillna(0) + df[
        "balls_bowled_future"
    ].fillna(0)

    min_balls = min_matches * 20
    df = df[df["prior_activity"] >= min_balls]
    df = df[df["future_activity"] >= min_balls]

    return df


def estimate_price_model(df):
    """
    Estimate log(price) = β₀ + β₁(total_war_lag) + β₂(is_indian) + β₃(is_mega_auction)

    Returns model and predictions.
    """
    df = df.copy()

    df["log_price"] = np.log(df["price_2024_cr"])

    df["is_indian"] = (df["nationality"] == "Indian").astype(int)

    mega_years = [2008, 2011, 2014, 2018, 2022, 2025]
    df["is_mega_auction"] = df["year"].isin(mega_years).astype(int)

    valid = df[df["log_price"].notna() & df["war_lag"].notna()].copy()

    X = valid[["war_lag", "is_indian", "is_mega_auction"]]
    X = sm.add_constant(X)
    y = valid["log_price"]

    model = sm.OLS(y, X).fit()

    print("=" * 60)
    print("PRICE PREDICTION MODEL (Lagged WAR)")
    print("=" * 60)
    print(model.summary())
    print()

    df.loc[valid.index, "log_price_predicted"] = model.predict(X)
    df["price_predicted_cr"] = np.exp(df["log_price_predicted"])

    return df, model


def compute_dud_score(df):
    """
    Compute dud score combining:
    1. Premium paid (how much overpaid relative to predicted)
    2. Future underperformance (expected WAR - actual future WAR)
    """
    df = df.copy()

    df["premium_pct"] = (
        (df["price_2024_cr"] - df["price_predicted_cr"]) / df["price_predicted_cr"]
    ) * 100

    df["expected_future_war"] = df["war_lag"]
    df["underperformance"] = df["expected_future_war"] - df["war_future"]

    valid = df[df["premium_pct"].notna() & df["underperformance"].notna()].copy()

    valid["premium_rank"] = valid["premium_pct"].apply(
        lambda x: percentileofscore(valid["premium_pct"].values, x)
    )
    valid["underperf_rank"] = valid["underperformance"].apply(
        lambda x: percentileofscore(valid["underperformance"].values, x)
    )

    valid["dud_score"] = (valid["premium_rank"] + valid["underperf_rank"]) / 2

    return valid


def get_team_abbreviation(team):
    """Convert team name to standard abbreviation."""
    team_map = {
        "Chennai Super Kings": "CSK",
        "Mumbai Indians": "MI",
        "Royal Challengers Bangalore": "RCB",
        "Royal Challengers Bengaluru": "RCB",
        "Kolkata Knight Riders": "KKR",
        "Delhi Capitals": "DC",
        "Delhi Daredevils": "DC",
        "Punjab Kings": "PBKS",
        "Kings XI Punjab": "PBKS",
        "Rajasthan Royals": "RR",
        "Sunrisers Hyderabad": "SRH",
        "Deccan Chargers": "SRH",
        "Gujarat Titans": "GT",
        "Lucknow Super Giants": "LSG",
        "Rising Pune Supergiant": "RPS",
        "Rising Pune Supergiants": "RPS",
        "Gujarat Lions": "GL",
        "Pune Warriors India": "PWI",
        "Kochi Tuskers Kerala": "KTK",
    }
    if pd.isna(team):
        return team
    for full_name, abbr in team_map.items():
        if full_name.lower() in str(team).lower():
            return abbr
    return team[:3].upper() if isinstance(team, str) else team


def format_output(df, top_n=20):
    """Format and print top duds."""
    df = df.copy()
    df = df.sort_values("dud_score", ascending=False).head(top_n)

    output_cols = [
        "player_name",
        "year",
        "team_x",
        "price_2024_cr",
        "war_lag",
        "price_predicted_cr",
        "premium_pct",
        "war_future",
        "underperformance",
        "dud_score",
    ]

    result = df[output_cols].copy()
    result.columns = [
        "Player",
        "Year",
        "Team",
        "Price (Cr)",
        "Prior WAR",
        "Predicted (Cr)",
        "Premium %",
        "Next WAR",
        "Shortfall",
        "Dud Score",
    ]

    result["Rank"] = range(1, len(result) + 1)
    result["Team"] = result["Team"].apply(get_team_abbreviation)

    result = result[
        [
            "Rank",
            "Player",
            "Year",
            "Team",
            "Price (Cr)",
            "Prior WAR",
            "Predicted (Cr)",
            "Premium %",
            "Next WAR",
            "Shortfall",
            "Dud Score",
        ]
    ]

    return result


def main():
    print("Loading data...")
    auction, war = load_data()

    print(f"Auction records: {len(auction)}")
    print(f"WAR records: {len(war)}")

    print("\nCreating lagged and future WAR...")
    war_with_lags = create_lagged_and_future_war(war)

    print("Merging auction with WAR data...")
    merged = merge_auction_with_war(auction, war_with_lags)

    print("Filtering valid observations...")
    filtered = filter_valid_observations(merged, min_matches=5)
    print(f"Valid observations for analysis: {len(filtered)}")

    print("\nEstimating price prediction model...\n")
    with_predictions, model = estimate_price_model(filtered)

    print("Computing dud scores...")
    duds = compute_dud_score(with_predictions)

    top_duds = format_output(duds, top_n=20)

    print("\n" + "=" * 80)
    print("THE DUMBEST IPL BUYS EVER")
    print("=" * 80)
    print(
        "\nRanking based on: Premium paid over fair value + Future underperformance\n"
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")

    print(top_duds.to_string(index=False))

    output_path = Path(__file__).parent.parent / "data" / "analysis" / "worst_bets.csv"
    full_output = format_output(duds, top_n=len(duds))
    full_output.to_csv(output_path, index=False)
    print(f"\nFull results saved to: {output_path}")

    print("\n" + "=" * 80)
    print("SANITY CHECKS")
    print("=" * 80)

    corr = duds["premium_pct"].corr(duds["underperformance"])
    print(f"\nCorrelation between premium paid and underperformance: {corr:.3f}")

    print("\nTop 5 highest premiums paid:")
    top_premium = duds.nlargest(5, "premium_pct")[
        ["player_name", "year", "premium_pct", "war_lag"]
    ].to_string(index=False)
    print(top_premium)

    print("\nTop 5 biggest underperformers:")
    top_underperf = duds.nlargest(5, "underperformance")[
        ["player_name", "year", "underperformance", "war_lag", "war_future"]
    ].to_string(index=False)
    print(top_underperf)


if __name__ == "__main__":
    main()
