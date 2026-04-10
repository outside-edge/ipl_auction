#!/usr/bin/env python3
"""
Build feature matrix for WAR forecasting model.

Combines IPL and T20I performance history to create features for predicting
next-season IPL WAR at auction time.

Features:
- IPL WAR: lag 1, 2, 3 seasons, career total
- T20I WAR: last 12 months, career total
- Form: trend (improving/declining)
- Context: age, experience, role, nationality, auction type

Output: data/model/auction_features.parquet
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from shared.names import normalize_name
from shared.io import load_dataset, save_dataset, dataset_exists

DATA_DIR = BASE_DIR / "data"
ACQUISITIONS_DIR = DATA_DIR / "acquisitions"
PERF_DIR = DATA_DIR / "perf" / "ipl"
T20I_DIR = DATA_DIR / "perf" / "t20i"
JOINED_DIR = DATA_DIR / "analysis" / "joined"
MODEL_DIR = DATA_DIR / "model"


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


def create_ipl_lagged_features(auction, ipl_war):
    """Create lagged IPL WAR features for each auction entry."""
    print("\nCreating IPL lagged WAR features...")

    auction = auction.copy()
    ipl_war = ipl_war.copy()

    auction["player_norm"] = auction["player_name"].apply(normalize_name)
    ipl_war["player_norm"] = ipl_war["player"].apply(normalize_name)

    auction["ipl_war_lag1"] = np.nan
    auction["ipl_war_lag2"] = np.nan
    auction["ipl_war_lag3"] = np.nan
    auction["ipl_career_war"] = np.nan
    auction["ipl_seasons_played"] = np.nan
    auction["ipl_matches_lag1"] = np.nan

    for idx, row in auction.iterrows():
        player_norm = row["player_norm"]
        year = row["year"]

        player_history = ipl_war[
            (ipl_war["player_norm"] == player_norm) &
            (ipl_war["season"] < year)
        ].sort_values("season", ascending=False)

        if len(player_history) >= 1:
            auction.at[idx, "ipl_war_lag1"] = player_history.iloc[0]["total_war"]
            bf = player_history.iloc[0]["balls_faced"]
            bb = player_history.iloc[0]["balls_bowled"]
            auction.at[idx, "ipl_matches_lag1"] = (bf + bb) / 30

        if len(player_history) >= 2:
            auction.at[idx, "ipl_war_lag2"] = player_history.iloc[1]["total_war"]

        if len(player_history) >= 3:
            auction.at[idx, "ipl_war_lag3"] = player_history.iloc[2]["total_war"]

        if len(player_history) > 0:
            auction.at[idx, "ipl_career_war"] = player_history["total_war"].sum()
            auction.at[idx, "ipl_seasons_played"] = len(player_history)

    matched = auction["ipl_war_lag1"].notna().sum()
    print(f"  Matched lag1 WAR: {matched}/{len(auction)} ({100*matched/len(auction):.1f}%)")

    return auction


def create_t20i_features(auction, t20i_war, player_master):
    """Create T20I WAR features using player_master for cross-reference."""
    print("\nCreating T20I WAR features...")

    auction = auction.copy()

    auction["t20i_war_12m"] = np.nan
    auction["t20i_war_24m"] = np.nan
    auction["t20i_career_war"] = np.nan

    if player_master.empty or len(t20i_war) == 0:
        print("  Skipping T20I features (no data)")
        return auction

    t20i_war = t20i_war.copy()
    t20i_war["player_norm"] = t20i_war["player"].apply(normalize_name)

    for idx, row in auction.iterrows():
        player_norm = normalize_name(row["player_name"])
        year = row["year"]

        player_t20i = t20i_war[
            (t20i_war["player_norm"] == player_norm)
        ].copy()

        if len(player_t20i) == 0:
            continue

        prior_t20i = player_t20i[player_t20i["year"] < year].sort_values("year", ascending=False)

        if len(prior_t20i) >= 1:
            auction.at[idx, "t20i_war_12m"] = prior_t20i.iloc[0]["total_war"]

        if len(prior_t20i) >= 2:
            auction.at[idx, "t20i_war_24m"] = prior_t20i.iloc[:2]["total_war"].sum()

        if len(prior_t20i) > 0:
            auction.at[idx, "t20i_career_war"] = prior_t20i["total_war"].sum()

    matched = auction["t20i_war_12m"].notna().sum()
    print(f"  Matched T20I 12m WAR: {matched}/{len(auction)} ({100*matched/len(auction):.1f}%)")

    return auction


def create_next_season_target(auction, ipl_war):
    """Create target variable: next season IPL WAR."""
    print("\nCreating target variable (next_season_war)...")

    auction = auction.copy()
    ipl_war = ipl_war.copy()

    auction["player_norm"] = auction["player_name"].apply(normalize_name)
    ipl_war["player_norm"] = ipl_war["player"].apply(normalize_name)

    auction["next_season_war"] = np.nan

    war_dict = ipl_war.set_index(["player_norm", "season"])["total_war"].to_dict()

    for idx, row in auction.iterrows():
        player_norm = row["player_norm"]
        year = row["year"]

        key = (player_norm, year)
        if key in war_dict:
            auction.at[idx, "next_season_war"] = war_dict[key]

    matched = auction["next_season_war"].notna().sum()
    print(f"  Matched next_season_war: {matched}/{len(auction)} ({100*matched/len(auction):.1f}%)")

    return auction


def create_context_features(auction):
    """Create context features: nationality, role, auction type."""
    print("\nCreating context features...")

    auction = auction.copy()

    auction["is_indian"] = (auction["nationality"] == "Indian").astype(int)

    mega_auction_years = [2008, 2011, 2014, 2018, 2022, 2025]
    auction["is_mega_auction"] = auction["year"].isin(mega_auction_years).astype(int)

    auction["is_retained"] = (auction["status"] == "RETAINED").astype(int)

    auction["role_batsman"] = (auction["role"] == "Batsman").astype(int)
    auction["role_bowler"] = (auction["role"] == "Bowler").astype(int)
    auction["role_allrounder"] = (auction["role"] == "All-Rounder").astype(int)
    auction["role_wk"] = (auction["role"] == "Wicket-Keeper").astype(int)

    auction["final_price_lakh"] = pd.to_numeric(auction["final_price_lakh"], errors="coerce")
    auction["log_price"] = np.log(auction["final_price_lakh"].replace(0, np.nan))

    print(f"  Mega auction years: {mega_auction_years}")
    print(f"  Indian players: {auction['is_indian'].sum()}")
    print(f"  Retained players: {auction['is_retained'].sum()}")

    return auction


def create_form_features(auction):
    """Create form trend features (improving vs declining)."""
    print("\nCreating form features...")

    auction = auction.copy()

    def compute_trend(row):
        if pd.notna(row["ipl_war_lag1"]) and pd.notna(row["ipl_war_lag2"]):
            return row["ipl_war_lag1"] - row["ipl_war_lag2"]
        return np.nan

    auction["ipl_war_trend"] = auction.apply(compute_trend, axis=1)

    def compute_avg(row):
        vals = [row.get("ipl_war_lag1"), row.get("ipl_war_lag2"), row.get("ipl_war_lag3")]
        vals = [v for v in vals if pd.notna(v)]
        if vals:
            return np.mean(vals)
        return np.nan

    auction["ipl_war_avg_3y"] = auction.apply(compute_avg, axis=1)

    print(f"  Players with trend data: {auction['ipl_war_trend'].notna().sum()}")

    return auction


def filter_for_training(auction):
    """Filter to valid training observations."""
    print("\nFiltering for training data...")

    train_years = auction[(auction["year"] >= 2009) & (auction["year"] <= 2024)]

    with_target = train_years[train_years["next_season_war"].notna()]

    with_features = with_target[
        with_target["ipl_war_lag1"].notna() |
        with_target["t20i_war_12m"].notna()
    ]

    print(f"  Total auctions 2009-2024: {len(train_years)}")
    print(f"  With next_season_war: {len(with_target)}")
    print(f"  With at least one feature: {len(with_features)}")

    return with_features


def main():
    print("=" * 60)
    print("Building Auction Feature Matrix")
    print("=" * 60)

    auction, ipl_war, t20i_war, player_master = load_data()

    auction = create_ipl_lagged_features(auction, ipl_war)
    auction = create_t20i_features(auction, t20i_war, player_master)
    auction = create_next_season_target(auction, ipl_war)
    auction = create_context_features(auction)
    auction = create_form_features(auction)

    feature_cols = [
        "year", "player_id", "player_name", "team",
        "final_price_lakh", "log_price",
        "ipl_war_lag1", "ipl_war_lag2", "ipl_war_lag3",
        "ipl_career_war", "ipl_seasons_played", "ipl_matches_lag1",
        "t20i_war_12m", "t20i_war_24m", "t20i_career_war",
        "ipl_war_trend", "ipl_war_avg_3y",
        "is_indian", "is_mega_auction", "is_retained",
        "role_batsman", "role_bowler", "role_allrounder", "role_wk",
        "nationality", "role", "status",
        "next_season_war"
    ]

    output = auction[[c for c in feature_cols if c in auction.columns]]

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = save_dataset(output, MODEL_DIR / "auction_features", format="parquet")
    print(f"\nSaved full features to {output_path}")

    train_data = filter_for_training(auction)
    train_cols = [c for c in feature_cols if c in train_data.columns]
    train_output = train_data[train_cols]

    train_path = save_dataset(train_output, MODEL_DIR / "auction_features_train", format="parquet")
    print(f"Saved training features to {train_path}")

    print("\n" + "=" * 60)
    print("FEATURE SUMMARY")
    print("=" * 60)

    print("\nFeature coverage (training set):")
    for col in ["ipl_war_lag1", "ipl_war_lag2", "ipl_war_lag3",
                "t20i_war_12m", "ipl_career_war", "next_season_war"]:
        if col in train_output.columns:
            coverage = train_output[col].notna().mean() * 100
            print(f"  {col:20s}: {coverage:.1f}%")

    print("\nCorrelation with next_season_war:")
    numeric_cols = ["ipl_war_lag1", "ipl_war_lag2", "ipl_war_lag3",
                   "ipl_career_war", "t20i_war_12m", "ipl_war_trend"]
    for col in numeric_cols:
        if col in train_output.columns:
            valid = train_output[[col, "next_season_war"]].dropna()
            if len(valid) > 10:
                corr = valid[col].corr(valid["next_season_war"])
                print(f"  {col:20s}: {corr:.3f}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
