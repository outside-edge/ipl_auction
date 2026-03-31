#!/usr/bin/env python3
"""
Predict IPL 2026 Auction "Duds" - players who are most likely to disappoint
based on overpayment relative to performance-predicted fair value.

Since we don't have 2026 season WAR data yet, predictions are based solely on
the PREMIUM PAID metric (actual price vs. model-predicted price based on prior
season WAR). Historical data shows high premiums correlate with underperformance.

Output: data/analysis/predicted_duds_2026.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


def load_data():
    """Load auction and WAR data."""
    auction = pd.read_csv(DATA_DIR / "auction_all_years.csv")
    war = pd.read_csv(DATA_DIR / "player_season_war.csv")

    return auction, war


def create_lagged_war(war_df):
    """Create lagged WAR (prior season) for each player."""
    war_df = war_df.copy()
    war_df = war_df.sort_values(["player", "season"])

    war_df["war_lag"] = war_df.groupby("player")["total_war"].shift(1)
    war_df["balls_faced_lag"] = war_df.groupby("player")["balls_faced"].shift(1)
    war_df["balls_bowled_lag"] = war_df.groupby("player")["balls_bowled"].shift(1)

    return war_df


def get_prior_season_war(war_df, year):
    """Get WAR from prior season (year - 1) for each player."""
    prior_year = year - 1
    prior_war = war_df[war_df["season"] == prior_year][
        ["player", "total_war", "balls_faced", "balls_bowled"]
    ].copy()
    prior_war.columns = ["player", "war_prior", "balls_faced_prior", "balls_bowled_prior"]
    return prior_war


def standardize_player_name(name):
    """Standardize player names for matching."""
    if pd.isna(name):
        return ""
    name = str(name).strip()
    name = name.replace("'", "'").replace("'", "'")
    name = " ".join(name.split())
    return name.lower()


def get_player_name_mapping():
    """Manual mapping from auction names (full) to WAR names (abbreviated)."""
    return {
        "cameron green": "c green",
        "matheesha pathirana": "m pathirana",
        "liam livingstone": "ls livingstone",
        "venkatesh iyer": "vr iyer",
        "ravi bishnoi": "r bishnoi",
        "rahul chahar": "rd chahar",
        "quinton de kock": "qj de kock",
        "josh inglis": "jp inglis",
        "david miller": "da miller",
        "wanindu hasaranga": "pwadd hasaranga",
        "anrich nortje": "a nortje",
        "jason holder": "jr holder",
        "mustafizur rahman": "mustafizur rahman",
        "rachin ravindra": "rachin ravindra",
        "ben duckett": "bm duckett",
        "sarfaraz khan": "sn khan",
        "prithvi shaw": "pp shaw",
        "finn allen": "fg allen",
        "shivam mavi": "s mavi",
        "akash deep": "akash deep",
        "rahul tripathi": "rahul tripathi",
        "kyle jamieson": "kj jamieson",
        "lungi ngidi": "l ngidi",
        "kuldeep sen": "kuldeep sen",
        "cooper connolly": "c connolly",
        "ben dwarshuis": "bj dwarshuis",
        "pathum nissanka": "pbd nissanka",
        "adam milne": "af milne",
        "tim seifert": "kd seifert",
        "akeal hosein": "as hosein",
        "kartik tyagi": "kartik tyagi",
        "matt henry": "mj henry",
        "jacob duffy": "jd duffy",
        "luke wood": "l wood",
        "mangesh yadav": "m yadav",
        "vicky ostwal": "r ostwal",
        "zak foulkes": "za foulkes",
        "tom banton": "t banton",
        "jordan cox": "jm cox",
        "matthew short": "mw short",
    }


def match_player_names(auction_df, war_df):
    """Match player names between auction and WAR data."""
    auction_df = auction_df.copy()
    war_df = war_df.copy()

    auction_df["player_std"] = auction_df["player_name"].apply(standardize_player_name)
    war_df["player_std"] = war_df["player"].apply(standardize_player_name)

    name_map = get_player_name_mapping()

    def map_name(name):
        name_lower = name.lower() if isinstance(name, str) else ""
        return name_map.get(name_lower, name_lower)

    auction_df["player_mapped"] = auction_df["player_name"].apply(map_name)

    return auction_df, war_df


def filter_valid_training_data(df, min_matches=5):
    """Filter to valid observations for training the price model."""
    df = df.copy()

    df = df[df["year"] > 2008]
    df = df[df["year"] < 2026]
    df = df[df["war_lag"].notna()]

    df["prior_activity"] = df["balls_faced_lag"].fillna(0) + df["balls_bowled_lag"].fillna(0)
    min_balls = min_matches * 20
    df = df[df["prior_activity"] >= min_balls]

    return df


def estimate_price_model(df):
    """
    Estimate log(price) = β₀ + β₁(total_war_lag) + β₂(is_indian) + β₃(is_mega_auction)

    Returns model and training data with predictions.
    """
    df = df.copy()

    df["final_price_lakh"] = pd.to_numeric(df["final_price_lakh"], errors="coerce")
    df["log_price"] = np.log(df["final_price_lakh"])

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

    return model


def predict_2026_prices(auction_2026, war_all, model):
    """Predict fair prices for 2026 auction players based on most recent WAR."""
    auction_2026 = auction_2026.copy()
    war_all = war_all.copy()

    auction_2026["final_price_lakh"] = pd.to_numeric(
        auction_2026["final_price_lakh"], errors="coerce"
    )

    auction_2026["player_std"] = auction_2026["player_name"].apply(standardize_player_name)
    war_all["player_std"] = war_all["player"].apply(standardize_player_name)

    name_map = get_player_name_mapping()

    war_recent = war_all.sort_values("season", ascending=False).groupby("player_std").first()

    war_dict = {}
    season_dict = {}
    for player_std, row in war_recent.iterrows():
        war_dict[player_std] = row["total_war"]
        season_dict[player_std] = row["season"]

    for auction_name, war_name in name_map.items():
        if war_name in war_dict:
            war_dict[auction_name] = war_dict[war_name]
            season_dict[auction_name] = season_dict[war_name]

    auction_2026["war_prior"] = auction_2026["player_std"].map(war_dict)
    auction_2026["war_season"] = auction_2026["player_std"].map(season_dict)

    matched = auction_2026[auction_2026["war_prior"].notna()].copy()
    unmatched = auction_2026[auction_2026["war_prior"].isna()].copy()

    print(f"\nMatched {len(matched)} of {len(auction_2026)} players with prior WAR")
    print(f"Unmatched: {len(unmatched)} players (no IPL history)")
    if len(matched) > 0:
        print(f"WAR seasons used: {matched['war_season'].value_counts().to_dict()}")

    if len(matched) == 0:
        print("No players matched with WAR data!")
        return pd.DataFrame()

    matched["is_indian"] = (matched["nationality"] == "Indian").astype(int)
    matched["is_mega_auction"] = 0

    X_pred = matched[["war_prior", "is_indian", "is_mega_auction"]].copy()
    X_pred.columns = ["war_lag", "is_indian", "is_mega_auction"]
    X_pred = sm.add_constant(X_pred, has_constant="add")

    matched["log_price_predicted"] = model.predict(X_pred)
    matched["price_predicted_lakh"] = np.exp(matched["log_price_predicted"])

    matched["premium_pct"] = (
        (matched["final_price_lakh"] - matched["price_predicted_lakh"])
        / matched["price_predicted_lakh"]
    ) * 100

    return matched


def get_team_abbreviation(team):
    """Convert team name to standard abbreviation."""
    if pd.isna(team):
        return team
    team_map = {
        "CSK": "CSK",
        "DC": "DC",
        "MI": "MI",
        "KKR": "KKR",
        "RCB": "RCB",
        "RR": "RR",
        "SRH": "SRH",
        "PBKS": "PBKS",
        "LSG": "LSG",
        "GT": "GT",
    }
    return team_map.get(str(team).upper(), team)


def format_output(df, top_n=20):
    """Format predicted duds for output."""
    df = df.copy()
    df = df.sort_values("premium_pct", ascending=False).head(top_n)

    result = pd.DataFrame(
        {
            "Rank": range(1, len(df) + 1),
            "Player": df["player_name"].values,
            "Team": df["team"].apply(get_team_abbreviation).values,
            "Price (Cr)": (df["final_price_lakh"] / 100).round(2).values,
            "Prior WAR": df["war_prior"].round(2).values,
            "Predicted (Cr)": (df["price_predicted_lakh"] / 100).round(2).values,
            "Premium %": df["premium_pct"].round(0).astype(int).astype(str).values + "%",
        }
    )

    return result


def main():
    print("Loading data...")
    auction, war = load_data()

    print(f"Auction records: {len(auction)}")
    print(f"WAR records: {len(war)}")

    print("\nCreating lagged WAR...")
    war_with_lags = create_lagged_war(war)

    auction_hist = auction[auction["year"] < 2026].copy()
    merged = auction_hist.merge(
        war_with_lags[["season", "player", "war_lag", "balls_faced_lag", "balls_bowled_lag"]],
        left_on=["year", "player_name"],
        right_on=["season", "player"],
        how="left",
    )

    name_map = get_player_name_mapping()

    for idx, row in merged[merged["war_lag"].isna()].iterrows():
        player_std = standardize_player_name(row["player_name"])
        if player_std in name_map:
            war_name = name_map[player_std]
            match = war_with_lags[
                (war_with_lags["player"].apply(standardize_player_name) == war_name)
                & (war_with_lags["season"] == row["year"])
            ]
            if len(match) > 0:
                merged.loc[idx, "war_lag"] = match.iloc[0]["war_lag"]
                merged.loc[idx, "balls_faced_lag"] = match.iloc[0]["balls_faced_lag"]
                merged.loc[idx, "balls_bowled_lag"] = match.iloc[0]["balls_bowled_lag"]

    print("\nFiltering valid observations for training...")
    training_data = filter_valid_training_data(merged, min_matches=5)
    print(f"Valid training observations: {len(training_data)}")

    print("\nEstimating price prediction model...\n")
    model = estimate_price_model(training_data)

    auction_2026 = auction[auction["year"] == 2026].copy()

    print(f"\n2026 auction players: {len(auction_2026)}")

    predictions = predict_2026_prices(auction_2026, war, model)

    if len(predictions) == 0:
        print("\nNo predictions possible - no players matched with WAR data")
        return

    output = format_output(predictions, top_n=len(predictions))

    print("\n" + "=" * 80)
    print("PREDICTED DUDS FOR IPL 2026")
    print("=" * 80)
    print("\nRanked by Premium % (how much teams overpaid vs. model prediction)\n")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_rows", 25)

    print(output.head(15).to_string(index=False))

    output_path = DATA_DIR / "analysis" / "predicted_duds_2026.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"\nFull results saved to: {output_path}")

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("""
Premium % shows how much teams paid above the model-predicted fair value.
Higher premium = higher risk of disappointing performance.

The model uses:
- Prior season WAR (2025) as the primary predictor
- Indian player premium
- Auction type (mega vs. mini - 2026 was a mini auction)

Caveats:
- Predictions exclude new IPL entrants (no prior WAR data)
- Teams may have private information justifying higher prices
- Model explains ~40% of price variance historically
""")


if __name__ == "__main__":
    main()
