#!/usr/bin/env python3
"""
Compute IPL WAR (Wins Above Replacement) from ball-by-ball data.

WAR provides a context-adjusted, replacement-level-normalized performance metric.
Based on baseball economics methodology (Scully 1974, Zimbalist).

Includes both:
- Naive WAR: quantile-based replacement level
- GAM WAR: context-adjusted using Generalized Additive Models
"""

import sys
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.shared.war_gam import (
    compute_batting_war_gam,
    compute_bowling_war_gam,
    validate_gam_war,
)
DATA_DIR = BASE_DIR / "data"
PERF_DIR = DATA_DIR / "perf" / "ipl"
SOURCES_DIR = DATA_DIR / "perf" / "sources"
KAGGLE_DIR = SOURCES_DIR / "kaggle"
CONSTANTS_PATH = DATA_DIR / "perf" / "ipl_constants_by_year.csv"


def load_ipl_constants():
    """Load year-specific IPL WAR constants."""
    if not CONSTANTS_PATH.exists():
        print(f"  Warning: {CONSTANTS_PATH} not found, using defaults")
        return None
    constants = pd.read_csv(CONSTANTS_PATH)
    print(f"  Loaded year-specific constants for {len(constants)} years")
    return constants


def load_ball_by_ball():
    """Load ball-by-ball data with year extracted from date."""
    print("Loading ball-by-ball data...")
    bbb = pd.read_csv(KAGGLE_DIR / "ball_by_ball_ipl.csv")
    print(f"  Loaded {len(bbb):,} deliveries")

    bbb = bbb.rename(columns={
        "Match ID": "ID",
        "Over": "Overs",
        "Ball": "Ball",
        "Batter": "Batter",
        "Bowler": "Bowler",
        "Batter Runs": "BatsmanRun",
        "Runs From Ball": "TotalRun",
        "Extra Type": "ExtraType",
        "Wicket": "IsWicketDelivery",
        "Innings": "Innings",
    })

    bbb["year"] = pd.to_datetime(bbb["Date"]).dt.year

    print(f"  Years covered: {sorted(bbb['year'].dropna().unique().astype(int))}")
    return bbb


def get_phase(over):
    """Determine T20 phase from over number (1-indexed)."""
    if over <= 6:
        return "powerplay"
    elif over <= 15:
        return "middle"
    else:
        return "death"


DEFAULT_RUNS_PER_WICKET = 5.72
DEFAULT_RUNS_PER_WIN = 16.04


def compute_batting_war(bbb, constants=None):
    """
    Compute batting WAR for each player-season.

    Method:
    1. Compute mean runs-per-ball by phase across all seasons
    2. Replacement level = 15th percentile of player strike rates (per season)
    3. batting_war = (actual_runs - expected_runs - dismissal_penalty) / runs_per_win

    Dismissal penalty accounts for opportunity cost of losing a wicket.
    Uses year-specific IPL constants when available.
    """
    print("\nComputing batting WAR...")

    bbb["phase"] = bbb["Overs"].apply(get_phase)

    phase_baseline = (
        bbb.groupby("phase")
        .agg(total_runs=("BatsmanRun", "sum"), total_balls=("BatsmanRun", "count"))
        .assign(baseline_sr=lambda x: x["total_runs"] / x["total_balls"] * 100)
    )
    print("  Phase baselines (strike rate):")
    print(f"    Powerplay: {phase_baseline.loc['powerplay', 'baseline_sr']:.1f}")
    print(f"    Middle:    {phase_baseline.loc['middle', 'baseline_sr']:.1f}")
    print(f"    Death:     {phase_baseline.loc['death', 'baseline_sr']:.1f}")

    batter_stats = (
        bbb.groupby(["year", "Batter"])
        .agg(
            runs=("BatsmanRun", "sum"),
            balls_faced=("BatsmanRun", "count"),
            fours=("BatsmanRun", lambda x: (x == 4).sum()),
            sixes=("BatsmanRun", lambda x: (x == 6).sum()),
            dismissals=("IsWicketDelivery", "sum"),
        )
        .reset_index()
    )

    batter_stats["strike_rate"] = (
        batter_stats["runs"] / batter_stats["balls_faced"] * 100
    )

    qualified = batter_stats[batter_stats["balls_faced"] >= 30].copy()

    season_replacement = qualified.groupby("year")["strike_rate"].quantile(0.15)
    print("\n  Replacement level SR by season (15th percentile):")
    for year, sr in season_replacement.items():
        print(f"    {year}: {sr:.1f}")

    overall_replacement_sr = qualified["strike_rate"].quantile(0.15)
    print(f"  Overall replacement SR: {overall_replacement_sr:.1f}")

    batter_stats = batter_stats.merge(
        season_replacement.rename("replacement_sr").reset_index(),
        on="year",
        how="left",
    )
    batter_stats["replacement_sr"] = batter_stats["replacement_sr"].fillna(
        overall_replacement_sr
    )

    if constants is not None:
        batter_stats = batter_stats.merge(
            constants[["year", "runs_per_wicket", "runs_per_win"]],
            on="year",
            how="left",
        )
        batter_stats["runs_per_wicket"] = batter_stats["runs_per_wicket"].fillna(
            DEFAULT_RUNS_PER_WICKET
        )
        batter_stats["runs_per_win"] = batter_stats["runs_per_win"].fillna(
            DEFAULT_RUNS_PER_WIN
        )
        print("  Using year-specific IPL constants")
    else:
        batter_stats["runs_per_wicket"] = DEFAULT_RUNS_PER_WICKET
        batter_stats["runs_per_win"] = DEFAULT_RUNS_PER_WIN
        print("  Using default constants")

    batter_stats["expected_runs"] = (
        batter_stats["balls_faced"] * batter_stats["replacement_sr"] / 100
    )
    batter_stats["dismissal_penalty"] = (
        batter_stats["dismissals"] * batter_stats["runs_per_wicket"]
    )
    batter_stats["runs_above_replacement"] = (
        batter_stats["runs"]
        - batter_stats["expected_runs"]
        - batter_stats["dismissal_penalty"]
    )
    batter_stats["batting_war"] = (
        batter_stats["runs_above_replacement"] / batter_stats["runs_per_win"]
    )

    batter_stats = batter_stats.rename(columns={"Batter": "player", "year": "season"})
    return batter_stats[
        ["season", "player", "runs", "balls_faced", "strike_rate", "dismissals", "batting_war"]
    ]


def compute_bowling_war(bbb, constants=None):
    """
    Compute bowling WAR for each player-season.

    Method:
    1. Replacement level = 80th percentile of economy rate among qualified bowlers
    2. bowling_war = (replacement_runs - actual_conceded + wicket_bonus) / runs_per_win

    Wicket bonus rewards bowlers for taking wickets (not just economy).
    Uses year-specific IPL constants when available.
    """
    print("\nComputing bowling WAR...")

    bowler_stats = (
        bbb.groupby(["year", "Bowler"])
        .agg(
            balls_bowled=("TotalRun", "count"),
            runs_conceded=("TotalRun", "sum"),
            wickets=("IsWicketDelivery", "sum"),
            wides=("ExtraType", lambda x: (x == "wides").sum()),
            noballs=("ExtraType", lambda x: (x == "noballs").sum()),
        )
        .reset_index()
    )

    bowler_stats["overs"] = bowler_stats["balls_bowled"] / 6
    bowler_stats["economy"] = bowler_stats["runs_conceded"] / bowler_stats["overs"]

    qualified = bowler_stats[bowler_stats["balls_bowled"] >= 60].copy()

    season_replacement = qualified.groupby("year")["economy"].quantile(0.80)
    print("  Replacement level economy by season (80th percentile):")
    for year, econ in season_replacement.items():
        print(f"    {year}: {econ:.2f}")

    overall_replacement_econ = qualified["economy"].quantile(0.80)
    print(f"  Overall replacement economy: {overall_replacement_econ:.2f}")

    bowler_stats = bowler_stats.merge(
        season_replacement.rename("replacement_econ").reset_index(),
        on="year",
        how="left",
    )
    bowler_stats["replacement_econ"] = bowler_stats["replacement_econ"].fillna(
        overall_replacement_econ
    )

    if constants is not None:
        bowler_stats = bowler_stats.merge(
            constants[["year", "runs_per_wicket", "runs_per_win"]],
            on="year",
            how="left",
        )
        bowler_stats["runs_per_wicket"] = bowler_stats["runs_per_wicket"].fillna(
            DEFAULT_RUNS_PER_WICKET
        )
        bowler_stats["runs_per_win"] = bowler_stats["runs_per_win"].fillna(
            DEFAULT_RUNS_PER_WIN
        )
        print("  Using year-specific IPL constants")
    else:
        bowler_stats["runs_per_wicket"] = DEFAULT_RUNS_PER_WICKET
        bowler_stats["runs_per_win"] = DEFAULT_RUNS_PER_WIN
        print("  Using default constants")

    bowler_stats["replacement_runs"] = (
        bowler_stats["overs"] * bowler_stats["replacement_econ"]
    )
    bowler_stats["wicket_bonus"] = (
        bowler_stats["wickets"] * bowler_stats["runs_per_wicket"]
    )
    bowler_stats["runs_saved"] = (
        bowler_stats["replacement_runs"]
        - bowler_stats["runs_conceded"]
        + bowler_stats["wicket_bonus"]
    )
    bowler_stats["bowling_war"] = (
        bowler_stats["runs_saved"] / bowler_stats["runs_per_win"]
    )

    bowler_stats = bowler_stats.rename(columns={"Bowler": "player", "year": "season"})
    return bowler_stats[
        [
            "season",
            "player",
            "balls_bowled",
            "overs",
            "runs_conceded",
            "wickets",
            "economy",
            "bowling_war",
        ]
    ]


def compute_total_war(batting_war, bowling_war):
    """Combine batting and bowling WAR into total WAR."""
    print("\nCombining batting and bowling WAR...")

    total = batting_war.merge(
        bowling_war, on=["season", "player"], how="outer", suffixes=("_bat", "_bowl")
    )

    total["batting_war"] = total["batting_war"].fillna(0)
    total["bowling_war"] = total["bowling_war"].fillna(0)
    total["total_war"] = total["batting_war"] + total["bowling_war"]

    total["balls_faced"] = total["balls_faced"].fillna(0)
    total["balls_bowled"] = total["balls_bowled"].fillna(0)

    total = total.sort_values(["season", "total_war"], ascending=[True, False])

    return total


def validate_war(war_df):
    """Validate WAR computations with sanity checks."""
    print("\n" + "=" * 60)
    print("WAR VALIDATION")
    print("=" * 60)

    print("\nTop 20 Total WAR (all seasons):")
    top_war = (
        war_df.groupby("player")
        .agg(
            total_war=("total_war", "sum"),
            batting_war=("batting_war", "sum"),
            bowling_war=("bowling_war", "sum"),
            seasons=("season", "count"),
        )
        .sort_values("total_war", ascending=False)
        .head(20)
    )
    print(top_war.round(2).to_string())

    print("\nTop 10 by season (sample years):")
    for year in [2016, 2019, 2023]:
        if year in war_df["season"].values:
            print(f"\n  {year}:")
            top_year = war_df[war_df["season"] == year].nlargest(5, "total_war")[
                ["player", "total_war", "batting_war", "bowling_war"]
            ]
            for _, row in top_year.iterrows():
                print(
                    f"    {row['player']:25s} WAR={row['total_war']:6.2f} "
                    f"(bat={row['batting_war']:5.2f}, bowl={row['bowling_war']:5.2f})"
                )

    print("\nWAR distribution:")
    print(f"  Mean:   {war_df['total_war'].mean():.2f}")
    print(f"  Median: {war_df['total_war'].median():.2f}")
    print(f"  Std:    {war_df['total_war'].std():.2f}")
    print(f"  Min:    {war_df['total_war'].min():.2f}")
    print(f"  Max:    {war_df['total_war'].max():.2f}")

    positive_war = (war_df["total_war"] > 0).mean() * 100
    print(f"  % positive WAR: {positive_war:.1f}%")


def main():
    print("=" * 60)
    print("IPL WAR (Wins Above Replacement) Computation")
    print("=" * 60)

    bbb = load_ball_by_ball()
    constants = load_ipl_constants()

    batting_war = compute_batting_war(bbb, constants)
    print(f"\n  Computed batting WAR for {len(batting_war):,} player-seasons")

    bowling_war = compute_bowling_war(bbb, constants)
    print(f"  Computed bowling WAR for {len(bowling_war):,} player-seasons")

    war_df = compute_total_war(batting_war, bowling_war)
    print(f"  Total: {len(war_df):,} player-seasons with WAR")

    validate_war(war_df)

    batting_war_gam = compute_batting_war_gam(bbb, year_col="year", player_col="Batter")
    bowling_war_gam = compute_bowling_war_gam(bbb, year_col="year", player_col="Bowler")

    war_df = war_df.merge(
        batting_war_gam[["season", "player", "batting_war_gam"]],
        on=["season", "player"],
        how="left",
    )
    war_df = war_df.merge(
        bowling_war_gam[["season", "player", "bowling_war_gam"]],
        on=["season", "player"],
        how="left",
    )
    war_df["batting_war_gam"] = war_df["batting_war_gam"].fillna(0)
    war_df["bowling_war_gam"] = war_df["bowling_war_gam"].fillna(0)
    war_df["total_war_gam"] = war_df["batting_war_gam"] + war_df["bowling_war_gam"]

    validate_gam_war(war_df)

    output_cols = [
        "season",
        "player",
        "batting_war",
        "bowling_war",
        "total_war",
        "batting_war_gam",
        "bowling_war_gam",
        "total_war_gam",
        "balls_faced",
        "runs",
        "strike_rate",
        "balls_bowled",
        "overs",
        "runs_conceded",
        "wickets",
        "economy",
    ]
    output_df = war_df[[c for c in output_cols if c in war_df.columns]]

    output_path = PERF_DIR / "player_season_war.csv"
    output_df.to_csv(output_path, index=False)
    print(f"\n  Saved to {output_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    return war_df


if __name__ == "__main__":
    main()
