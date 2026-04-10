#!/usr/bin/env python3
"""
Compute T20I WAR (Wins Above Replacement) from ball-by-ball data.

Same methodology as IPL WAR (compute_war.py) but applied to T20 Internationals.
Uses T20I-specific replacement levels (15th percentile SR batting, 80th percentile economy bowling).

Input: data/t20i/deliveries.csv, data/t20i/matches.csv
Output: data/t20i/player_season_war.csv, data/t20i/player_match_stats.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
T20I_DIR = DATA_DIR / "perf" / "t20i"

RUNS_PER_WIN = 10
RUNS_PER_DISMISSAL = 6.0
RUNS_PER_WICKET = 6.0


def load_t20i_data():
    """Load T20I deliveries and matches."""
    print("Loading T20I data...")

    deliveries_path = T20I_DIR / "deliveries.csv"
    matches_path = T20I_DIR / "matches.csv"

    if not deliveries_path.exists():
        raise FileNotFoundError(f"Run process_t20i_deliveries.py first. Missing: {deliveries_path}")

    deliveries = pd.read_csv(deliveries_path)
    matches = pd.read_csv(matches_path)

    matches["year"] = pd.to_datetime(matches["match_date"]).dt.year

    deliveries = deliveries.merge(
        matches[["match_number", "year", "match_date"]],
        left_on="ID",
        right_on="match_number",
        how="left"
    )

    print(f"  Loaded {len(deliveries):,} deliveries")
    print(f"  Loaded {len(matches):,} matches")
    print(f"  Years: {sorted(deliveries['year'].dropna().unique().astype(int))}")

    return deliveries, matches


def compute_match_batting_stats(deliveries):
    """Compute match-level batting statistics."""
    print("\nComputing match-level batting stats...")

    bat = deliveries.groupby(["ID", "year", "match_date", "Batter", "BattingTeam"]).agg(
        runs=("BatsmanRun", "sum"),
        balls=("BatsmanRun", "count"),
        fours=("BatsmanRun", lambda x: (x == 4).sum()),
        sixes=("BatsmanRun", lambda x: (x == 6).sum()),
    ).reset_index()

    dismissals = deliveries[deliveries["IsWicketDelivery"] == 1].groupby(
        ["ID", "PlayerOut"]
    ).size().reset_index(name="times_out")

    bat = bat.merge(
        dismissals,
        left_on=["ID", "Batter"],
        right_on=["ID", "PlayerOut"],
        how="left"
    )
    bat["times_out"] = bat["times_out"].fillna(0).astype(int)
    bat["is_out"] = bat["times_out"] > 0
    bat["strike_rate"] = np.where(bat["balls"] > 0, bat["runs"] / bat["balls"] * 100, 0)

    bat = bat.rename(columns={"Batter": "player", "BattingTeam": "team"})

    print(f"  {len(bat):,} batting innings")
    return bat


def compute_match_bowling_stats(deliveries):
    """Compute match-level bowling statistics."""
    print("Computing match-level bowling stats...")

    bowl = deliveries.groupby(["ID", "year", "match_date", "Bowler"]).agg(
        balls=("TotalRun", "count"),
        runs_conceded=("TotalRun", "sum"),
        wides=("ExtraType", lambda x: (x == "wides").sum()),
        noballs=("ExtraType", lambda x: (x == "noballs").sum()),
    ).reset_index()

    wickets = deliveries[deliveries["IsWicketDelivery"] == 1].copy()
    wickets = wickets[~wickets["Kind"].isin(["run out", "retired hurt", "retired out", "obstructing the field"])]
    wicket_counts = wickets.groupby(["ID", "Bowler"]).size().reset_index(name="wickets")

    bowl = bowl.merge(wicket_counts, on=["ID", "Bowler"], how="left")
    bowl["wickets"] = bowl["wickets"].fillna(0).astype(int)

    bowl["legal_balls"] = bowl["balls"] - bowl["wides"] - bowl["noballs"]
    bowl["overs"] = bowl["legal_balls"] / 6
    bowl["economy"] = np.where(bowl["overs"] > 0, bowl["runs_conceded"] / bowl["overs"], 0)

    bowl = bowl.rename(columns={"Bowler": "player"})

    print(f"  {len(bowl):,} bowling innings")
    return bowl


def aggregate_to_year(bat_match, bowl_match):
    """Aggregate match stats to year level."""
    print("\nAggregating to year level...")

    bat_year = bat_match.groupby(["year", "player"]).agg(
        matches_batting=("ID", "nunique"),
        runs=("runs", "sum"),
        balls_faced=("balls", "sum"),
        fours=("fours", "sum"),
        sixes=("sixes", "sum"),
        dismissals=("is_out", "sum"),
        highest_score=("runs", "max"),
        team=("team", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]),
    ).reset_index()

    bat_year["strike_rate"] = np.where(
        bat_year["balls_faced"] > 0,
        bat_year["runs"] / bat_year["balls_faced"] * 100,
        0
    )
    bat_year["batting_avg"] = np.where(
        bat_year["dismissals"] > 0,
        bat_year["runs"] / bat_year["dismissals"],
        bat_year["runs"]
    )

    bowl_year = bowl_match.groupby(["year", "player"]).agg(
        matches_bowling=("ID", "nunique"),
        balls_bowled=("legal_balls", "sum"),
        runs_conceded=("runs_conceded", "sum"),
        wickets=("wickets", "sum"),
        wides=("wides", "sum"),
        noballs=("noballs", "sum"),
    ).reset_index()

    bowl_year["overs"] = bowl_year["balls_bowled"] / 6
    bowl_year["economy"] = np.where(
        bowl_year["overs"] > 0,
        bowl_year["runs_conceded"] / bowl_year["overs"],
        0
    )
    bowl_year["bowling_avg"] = np.where(
        bowl_year["wickets"] > 0,
        bowl_year["runs_conceded"] / bowl_year["wickets"],
        np.nan
    )

    print(f"  {len(bat_year):,} player-years (batting)")
    print(f"  {len(bowl_year):,} player-years (bowling)")

    return bat_year, bowl_year


def compute_batting_war(bat_year):
    """
    Compute batting WAR using 15th percentile replacement level.

    Includes dismissal penalty to account for opportunity cost of losing wickets.
    """
    print("\nComputing batting WAR...")

    qualified = bat_year[bat_year["balls_faced"] >= 30].copy()

    season_replacement = qualified.groupby("year")["strike_rate"].quantile(0.15)
    overall_replacement = qualified["strike_rate"].quantile(0.15)

    print(f"  Overall replacement SR: {overall_replacement:.1f}")
    print("  Replacement SR by year (sample):")
    for year in sorted(season_replacement.index)[-5:]:
        print(f"    {year}: {season_replacement[year]:.1f}")

    bat_year = bat_year.merge(
        season_replacement.rename("replacement_sr").reset_index(),
        on="year",
        how="left"
    )
    bat_year["replacement_sr"] = bat_year["replacement_sr"].fillna(overall_replacement)

    bat_year["expected_runs"] = bat_year["balls_faced"] * bat_year["replacement_sr"] / 100
    bat_year["dismissal_penalty"] = bat_year["dismissals"].fillna(0) * RUNS_PER_DISMISSAL
    bat_year["runs_above_replacement"] = bat_year["runs"] - bat_year["expected_runs"] - bat_year["dismissal_penalty"]
    bat_year["batting_war"] = bat_year["runs_above_replacement"] / RUNS_PER_WIN
    print(f"  Including dismissal penalty: {RUNS_PER_DISMISSAL:.1f} runs per dismissal")

    return bat_year


def compute_bowling_war(bowl_year):
    """
    Compute bowling WAR using 80th percentile replacement level.

    Includes wicket bonus to reward bowlers for taking wickets (not just economy).
    """
    print("Computing bowling WAR...")

    qualified = bowl_year[bowl_year["balls_bowled"] >= 60].copy()

    season_replacement = qualified.groupby("year")["economy"].quantile(0.80)
    overall_replacement = qualified["economy"].quantile(0.80)

    print(f"  Overall replacement economy: {overall_replacement:.2f}")
    print("  Replacement economy by year (sample):")
    for year in sorted(season_replacement.index)[-5:]:
        print(f"    {year}: {season_replacement[year]:.2f}")

    bowl_year = bowl_year.merge(
        season_replacement.rename("replacement_econ").reset_index(),
        on="year",
        how="left"
    )
    bowl_year["replacement_econ"] = bowl_year["replacement_econ"].fillna(overall_replacement)

    bowl_year["expected_runs_conceded"] = bowl_year["overs"] * bowl_year["replacement_econ"]
    bowl_year["wicket_bonus"] = bowl_year["wickets"].fillna(0) * RUNS_PER_WICKET
    bowl_year["runs_saved"] = (
        bowl_year["expected_runs_conceded"] - bowl_year["runs_conceded"] + bowl_year["wicket_bonus"]
    )
    bowl_year["bowling_war"] = bowl_year["runs_saved"] / RUNS_PER_WIN
    print(f"  Including wicket bonus: {RUNS_PER_WICKET:.1f} runs per wicket")

    return bowl_year


def compute_total_war(bat_war, bowl_war):
    """Combine batting and bowling WAR."""
    print("\nCombining batting and bowling WAR...")

    total = bat_war.merge(
        bowl_war[["year", "player", "balls_bowled", "overs", "runs_conceded", "wickets", "economy", "bowling_war"]],
        on=["year", "player"],
        how="outer"
    )

    total["batting_war"] = total["batting_war"].fillna(0)
    total["bowling_war"] = total["bowling_war"].fillna(0)
    total["total_war"] = total["batting_war"] + total["bowling_war"]

    for col in ["balls_faced", "runs", "balls_bowled"]:
        if col in total.columns:
            total[col] = total[col].fillna(0)

    total = total.sort_values(["year", "total_war"], ascending=[True, False])

    print(f"  Total: {len(total):,} player-year records")
    return total


def save_match_stats(bat_match, bowl_match):
    """Save match-level stats for potential later use."""
    print("\nSaving match-level stats...")

    combined = bat_match.merge(
        bowl_match[["ID", "year", "player", "legal_balls", "runs_conceded", "wickets", "economy"]],
        on=["ID", "year", "player"],
        how="outer"
    )

    output_path = T20I_DIR / "player_match_stats.csv"
    combined.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")


def validate_war(war_df):
    """Validate WAR computations."""
    print("\n" + "=" * 60)
    print("T20I WAR VALIDATION")
    print("=" * 60)

    print("\nTop 20 Total WAR (career):")
    career_war = war_df.groupby("player").agg(
        total_war=("total_war", "sum"),
        batting_war=("batting_war", "sum"),
        bowling_war=("bowling_war", "sum"),
        seasons=("year", "count"),
    ).sort_values("total_war", ascending=False).head(20)
    print(career_war.round(2).to_string())

    print("\nTop players 2023:")
    if 2023 in war_df["year"].values:
        top_2023 = war_df[war_df["year"] == 2023].nlargest(10, "total_war")[
            ["player", "total_war", "batting_war", "bowling_war"]
        ]
        for _, row in top_2023.iterrows():
            print(f"  {row['player']:25s} WAR={row['total_war']:6.2f} "
                  f"(bat={row['batting_war']:5.2f}, bowl={row['bowling_war']:5.2f})")

    print("\nWAR distribution:")
    print(f"  Mean:   {war_df['total_war'].mean():.2f}")
    print(f"  Median: {war_df['total_war'].median():.2f}")
    print(f"  Std:    {war_df['total_war'].std():.2f}")
    print(f"  Min:    {war_df['total_war'].min():.2f}")
    print(f"  Max:    {war_df['total_war'].max():.2f}")


def main():
    print("=" * 60)
    print("T20I WAR Computation")
    print("=" * 60)

    deliveries, _ = load_t20i_data()

    bat_match = compute_match_batting_stats(deliveries)
    bowl_match = compute_match_bowling_stats(deliveries)

    save_match_stats(bat_match, bowl_match)

    bat_year, bowl_year = aggregate_to_year(bat_match, bowl_match)

    bat_war = compute_batting_war(bat_year)
    bowl_war = compute_bowling_war(bowl_year)
    war_df = compute_total_war(bat_war, bowl_war)

    validate_war(war_df)

    output_cols = [
        "year", "player", "team",
        "batting_war", "bowling_war", "total_war",
        "runs", "balls_faced", "strike_rate",
        "balls_bowled", "overs", "runs_conceded", "wickets", "economy"
    ]
    output_df = war_df[[c for c in output_cols if c in war_df.columns]]

    output_path = T20I_DIR / "player_year_war.csv"
    output_df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    return war_df


if __name__ == "__main__":
    main()
