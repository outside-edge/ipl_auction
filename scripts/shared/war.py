#!/usr/bin/env python3
"""
Shared WAR (Wins Above Replacement) computation utilities.

Provides common functions for computing batting and bowling WAR
using quantile-based replacement levels. Used by both IPL and T20I
WAR computation scripts.

Based on baseball economics methodology (Scully 1974, Zimbalist).

WAR formulas:
- Batting: WAR = (runs - expected_runs - dismissal_penalty) / RUNS_PER_WIN
  - Dismissal penalty accounts for opportunity cost of losing a wicket
- Bowling: WAR = (replacement_runs - runs_conceded + wicket_bonus) / RUNS_PER_WIN
  - Wicket bonus rewards bowlers for taking wickets (not just economy)
"""

import pandas as pd  # noqa: F401
import numpy as np

RUNS_PER_WIN = 8
RUNS_PER_DISMISSAL = 6.0
RUNS_PER_WICKET = 6.0


def compute_batting_war(
    stats_df,
    balls_col="balls_faced",
    runs_col="runs",
    dismissals_col="dismissals",
    year_col="season",
    player_col="player",
    min_balls=30,
    replacement_pct=0.15,
    include_dismissal_penalty=True,
    verbose=True,
):
    """
    Compute batting WAR for each player-season/year.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Player-season batting statistics
    balls_col : str
        Column name for balls faced
    runs_col : str
        Column name for runs scored
    dismissals_col : str
        Column name for dismissals (defaults to "dismissals")
    year_col : str
        Column name for season/year
    player_col : str
        Column name for player identifier
    min_balls : int
        Minimum balls faced to qualify for replacement calculation
    replacement_pct : float
        Quantile for replacement level (default 0.15 = 15th percentile)
    include_dismissal_penalty : bool
        Whether to include penalty for dismissals (default True)
    verbose : bool
        Print progress information

    Returns
    -------
    pd.DataFrame
        Input dataframe with batting_war column added
    """
    if verbose:
        print("\nComputing batting WAR...")

    df = stats_df.copy()

    df["strike_rate"] = df[runs_col] / df[balls_col] * 100
    df["strike_rate"] = df["strike_rate"].replace([np.inf, -np.inf], np.nan)

    qualified = df[df[balls_col] >= min_balls].copy()

    season_replacement = qualified.groupby(year_col)["strike_rate"].quantile(replacement_pct)
    overall_replacement = qualified["strike_rate"].quantile(replacement_pct)

    if verbose:
        print(f"  Overall replacement SR: {overall_replacement:.1f}")
        print(f"  Replacement SR by {year_col} (sample):")
        for year in sorted(season_replacement.index)[-5:]:
            print(f"    {year}: {season_replacement[year]:.1f}")

    df = df.merge(
        season_replacement.rename("replacement_sr").reset_index(),
        on=year_col,
        how="left",
    )
    df["replacement_sr"] = df["replacement_sr"].fillna(overall_replacement)

    df["expected_runs"] = df[balls_col] * df["replacement_sr"] / 100
    df["runs_above_replacement"] = df[runs_col] - df["expected_runs"]

    if include_dismissal_penalty and dismissals_col in df.columns:
        df["dismissal_penalty"] = df[dismissals_col].fillna(0) * RUNS_PER_DISMISSAL
        df["runs_above_replacement"] = df["runs_above_replacement"] - df["dismissal_penalty"]
        if verbose:
            print(f"  Including dismissal penalty: {RUNS_PER_DISMISSAL:.1f} runs per dismissal")
        df = df.drop(columns=["dismissal_penalty"])

    df["batting_war"] = df["runs_above_replacement"] / RUNS_PER_WIN

    df = df.drop(columns=["replacement_sr", "expected_runs", "runs_above_replacement"])

    return df


def compute_bowling_war(
    stats_df,
    balls_col="balls_bowled",
    runs_col="runs_conceded",
    wickets_col="wickets",
    year_col="season",
    player_col="player",
    min_balls=60,
    replacement_pct=0.80,
    include_wicket_bonus=True,
    verbose=True,
):
    """
    Compute bowling WAR for each player-season/year.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Player-season bowling statistics
    balls_col : str
        Column name for balls bowled
    runs_col : str
        Column name for runs conceded
    wickets_col : str
        Column name for wickets taken (defaults to "wickets")
    year_col : str
        Column name for season/year
    player_col : str
        Column name for player identifier
    min_balls : int
        Minimum balls bowled to qualify for replacement calculation
    replacement_pct : float
        Quantile for replacement level (default 0.80 = 80th percentile economy)
    include_wicket_bonus : bool
        Whether to include bonus for wickets taken (default True)
    verbose : bool
        Print progress information

    Returns
    -------
    pd.DataFrame
        Input dataframe with bowling_war column added
    """
    if verbose:
        print("\nComputing bowling WAR...")

    df = stats_df.copy()

    if "overs" not in df.columns:
        df["overs"] = df[balls_col] / 6

    df["economy"] = df[runs_col] / df["overs"]
    df["economy"] = df["economy"].replace([np.inf, -np.inf], np.nan)

    qualified = df[df[balls_col] >= min_balls].copy()

    season_replacement = qualified.groupby(year_col)["economy"].quantile(replacement_pct)
    overall_replacement = qualified["economy"].quantile(replacement_pct)

    if verbose:
        print(f"  Overall replacement economy: {overall_replacement:.2f}")
        print(f"  Replacement economy by {year_col} (sample):")
        for year in sorted(season_replacement.index)[-5:]:
            print(f"    {year}: {season_replacement[year]:.2f}")

    df = df.merge(
        season_replacement.rename("replacement_econ").reset_index(),
        on=year_col,
        how="left",
    )
    df["replacement_econ"] = df["replacement_econ"].fillna(overall_replacement)

    df["replacement_runs"] = df["overs"] * df["replacement_econ"]
    df["runs_saved"] = df["replacement_runs"] - df[runs_col]

    if include_wicket_bonus and wickets_col in df.columns:
        df["wicket_bonus"] = df[wickets_col].fillna(0) * RUNS_PER_WICKET
        df["runs_saved"] = df["runs_saved"] + df["wicket_bonus"]
        if verbose:
            print(f"  Including wicket bonus: {RUNS_PER_WICKET:.1f} runs per wicket")
        df = df.drop(columns=["wicket_bonus"])

    df["bowling_war"] = df["runs_saved"] / RUNS_PER_WIN

    df = df.drop(columns=["replacement_econ", "replacement_runs", "runs_saved"])

    return df


def combine_war(
    batting_war_df,
    bowling_war_df,
    year_col="season",
    player_col="player",
    verbose=True,
):
    """
    Combine batting and bowling WAR into total WAR.

    Parameters
    ----------
    batting_war_df : pd.DataFrame
        Batting statistics with batting_war column
    bowling_war_df : pd.DataFrame
        Bowling statistics with bowling_war column
    year_col : str
        Column name for season/year
    player_col : str
        Column name for player identifier
    verbose : bool
        Print progress information

    Returns
    -------
    pd.DataFrame
        Combined dataframe with total_war
    """
    if verbose:
        print("\nCombining batting and bowling WAR...")

    total = batting_war_df.merge(
        bowling_war_df,
        on=[year_col, player_col],
        how="outer",
        suffixes=("_bat", "_bowl"),
    )

    total["batting_war"] = total["batting_war"].fillna(0)
    total["bowling_war"] = total["bowling_war"].fillna(0)
    total["total_war"] = total["batting_war"] + total["bowling_war"]

    for col in ["balls_faced", "balls_bowled"]:
        if col in total.columns:
            total[col] = total[col].fillna(0)
        elif f"{col}_bat" in total.columns:
            total[col] = total[f"{col}_bat"].fillna(0)

    total = total.sort_values([year_col, "total_war"], ascending=[True, False])

    if verbose:
        print(f"  Total: {len(total):,} player-{year_col} records")

    return total


def validate_war(war_df, year_col="season", player_col="player", verbose=True):
    """
    Validate WAR computations with sanity checks.

    Parameters
    ----------
    war_df : pd.DataFrame
        Combined WAR dataframe
    year_col : str
        Column name for season/year
    player_col : str
        Column name for player identifier
    verbose : bool
        Print validation results

    Returns
    -------
    dict
        Validation statistics
    """
    if not verbose:
        return {}

    print("\n" + "=" * 60)
    print("WAR VALIDATION")
    print("=" * 60)

    print("\nTop 20 Total WAR (all seasons):")
    top_war = (
        war_df.groupby(player_col)
        .agg(
            total_war=("total_war", "sum"),
            batting_war=("batting_war", "sum"),
            bowling_war=("bowling_war", "sum"),
            seasons=(year_col, "count"),
        )
        .sort_values("total_war", ascending=False)
        .head(20)
    )
    print(top_war.round(2).to_string())

    print(f"\nTop players by {year_col} (sample):")
    years = sorted(war_df[year_col].unique())
    sample_years = [y for y in [2016, 2019, 2023] if y in years]
    if not sample_years and years:
        sample_years = years[-3:] if len(years) >= 3 else years

    for year in sample_years:
        print(f"\n  {year}:")
        top_year = war_df[war_df[year_col] == year].nlargest(5, "total_war")[
            [player_col, "total_war", "batting_war", "bowling_war"]
        ]
        for _, row in top_year.iterrows():
            print(
                f"    {row[player_col]:25s} WAR={row['total_war']:6.2f} "
                f"(bat={row['batting_war']:5.2f}, bowl={row['bowling_war']:5.2f})"
            )

    print("\nWAR distribution:")
    stats = {
        "mean": war_df["total_war"].mean(),
        "median": war_df["total_war"].median(),
        "std": war_df["total_war"].std(),
        "min": war_df["total_war"].min(),
        "max": war_df["total_war"].max(),
        "pct_positive": (war_df["total_war"] > 0).mean() * 100,
    }

    print(f"  Mean:   {stats['mean']:.2f}")
    print(f"  Median: {stats['median']:.2f}")
    print(f"  Std:    {stats['std']:.2f}")
    print(f"  Min:    {stats['min']:.2f}")
    print(f"  Max:    {stats['max']:.2f}")
    print(f"  % positive WAR: {stats['pct_positive']:.1f}%")

    return stats
