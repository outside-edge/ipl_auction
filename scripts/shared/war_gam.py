#!/usr/bin/env python3
"""
GAM-based WAR (Wins Above Replacement) computation.

Uses Generalized Additive Models to compute context-adjusted WAR.
The GAM approach accounts for:
- Over number (smooth spline capturing powerplay/middle/death phases)
- Batting position (inferred from order of appearance)
- Year/era (scoring pattern changes over seasons)

This provides more nuanced expected values compared to simple phase-based
or quantile-based methods.
"""

import pandas as pd
from pygam import LinearGAM, s, f

REPLACEMENT_ALPHA = 0.85
RUNS_PER_WIN = 10
WICKET_BONUS = 6.0
RUNS_PER_DISMISSAL = 6.0


def infer_batting_position(bbb):
    """
    Infer batting position from order of first appearance in each innings.

    Parameters
    ----------
    bbb : pd.DataFrame
        Ball-by-ball data with columns: ID, Innings, Batter, Overs, Ball

    Returns
    -------
    pd.DataFrame
        Dataframe with batting_position column added (1-11)
    """
    df = bbb.copy()

    df["ball_number"] = df["Overs"] * 6 + df["Ball"]

    first_appearance = (
        df.groupby(["ID", "Innings", "Batter"])["ball_number"]
        .min()
        .reset_index()
        .rename(columns={"ball_number": "first_ball"})
    )

    first_appearance["batting_position"] = (
        first_appearance.groupby(["ID", "Innings"])["first_ball"]
        .rank(method="dense")
        .astype(int)
    )

    first_appearance["batting_position"] = first_appearance["batting_position"].clip(
        upper=11
    )

    df = df.merge(
        first_appearance[["ID", "Innings", "Batter", "batting_position"]],
        on=["ID", "Innings", "Batter"],
        how="left",
    )

    df["batting_position"] = df["batting_position"].fillna(11).astype(int)

    return df


def train_batting_gam(bbb, year_col="year"):
    """
    Train GAM model to predict expected runs per ball.

    Model: E[runs] = s(over) + f(batting_position) + f(year)
    - s(over): Smooth spline on over number (0-19)
    - f(batting_position): Factor for batting position (1-11)
    - f(year): Factor for year/era

    Parameters
    ----------
    bbb : pd.DataFrame
        Ball-by-ball data with batting_position column
    year_col : str
        Column name for year

    Returns
    -------
    tuple
        (fitted_gam, year_codes) where year_codes maps years to numeric codes
    """
    df = bbb.copy()

    year_codes = {y: i for i, y in enumerate(sorted(df[year_col].dropna().unique()))}
    df["year_code"] = df[year_col].map(year_codes)

    df = df.dropna(subset=["Overs", "batting_position", "year_code", "BatsmanRun"])

    X = df[["Overs", "batting_position", "year_code"]].values
    y = df["BatsmanRun"].values

    gam = LinearGAM(
        s(0, n_splines=10, spline_order=3)
        + f(1, coding="dummy")
        + f(2, coding="dummy")
    )
    gam.fit(X, y)

    print(f"  Batting GAM pseudo R-squared: {gam.statistics_['pseudo_r2']['explained_deviance']:.4f}")

    return gam, year_codes


def compute_batting_war_gam(bbb, year_col="year", player_col="Batter", verbose=True):
    """
    Compute context-adjusted batting WAR using GAM.

    WAR = (actual_runs - REPLACEMENT_ALPHA * expected_runs) / RUNS_PER_WIN

    Parameters
    ----------
    bbb : pd.DataFrame
        Ball-by-ball data
    year_col : str
        Column name for year
    player_col : str
        Column name for player
    verbose : bool
        Print progress

    Returns
    -------
    pd.DataFrame
        Player-season batting WAR with columns: season, player, batting_war_gam
    """
    if verbose:
        print("\nComputing GAM-based batting WAR...")

    df = infer_batting_position(bbb)

    gam, year_codes = train_batting_gam(df, year_col=year_col)

    df["year_code"] = df[year_col].map(year_codes)
    df = df.dropna(subset=["Overs", "batting_position", "year_code"])

    X = df[["Overs", "batting_position", "year_code"]].values
    df["expected_runs"] = gam.predict(X)

    player_season = (
        df.groupby([year_col, player_col])
        .agg(
            actual_runs=("BatsmanRun", "sum"),
            expected_runs=("expected_runs", "sum"),
            balls_faced=("BatsmanRun", "count"),
            dismissals=("IsWicketDelivery", "sum"),
        )
        .reset_index()
    )

    player_season["dismissal_penalty"] = player_season["dismissals"] * RUNS_PER_DISMISSAL
    player_season["batting_war_gam"] = (
        player_season["actual_runs"]
        - REPLACEMENT_ALPHA * player_season["expected_runs"]
        - player_season["dismissal_penalty"]
    ) / RUNS_PER_WIN

    if verbose:
        print(f"  Computed GAM batting WAR for {len(player_season):,} player-seasons")
        top5 = player_season.nlargest(5, "batting_war_gam")
        print("  Top 5 by batting_war_gam:")
        for _, row in top5.iterrows():
            print(f"    {row[player_col]}: {row['batting_war_gam']:.2f}")

    player_season = player_season.rename(
        columns={year_col: "season", player_col: "player"}
    )
    return player_season[["season", "player", "batting_war_gam"]]


def train_bowling_gam(bbb, year_col="year"):
    """
    Train GAM model to predict expected runs conceded per ball.

    Model: E[runs_conceded] = s(over) + f(year)
    - s(over): Smooth spline on over number (0-19)
    - f(year): Factor for year/era

    Parameters
    ----------
    bbb : pd.DataFrame
        Ball-by-ball data
    year_col : str
        Column name for year

    Returns
    -------
    tuple
        (fitted_gam, year_codes)
    """
    df = bbb.copy()

    year_codes = {y: i for i, y in enumerate(sorted(df[year_col].dropna().unique()))}
    df["year_code"] = df[year_col].map(year_codes)

    df = df.dropna(subset=["Overs", "year_code", "TotalRun"])

    X = df[["Overs", "year_code"]].values
    y = df["TotalRun"].values

    gam = LinearGAM(s(0, n_splines=10, spline_order=3) + f(1, coding="dummy"))
    gam.fit(X, y)

    print(f"  Bowling GAM pseudo R-squared: {gam.statistics_['pseudo_r2']['explained_deviance']:.4f}")

    return gam, year_codes


def compute_bowling_war_gam(bbb, year_col="year", player_col="Bowler", verbose=True):
    """
    Compute context-adjusted bowling WAR using GAM.

    WAR = (replacement_runs - actual_runs + wicket_bonus) / RUNS_PER_WIN
    where replacement_runs = expected_runs from GAM

    Parameters
    ----------
    bbb : pd.DataFrame
        Ball-by-ball data
    year_col : str
        Column name for year
    player_col : str
        Column name for player
    verbose : bool
        Print progress

    Returns
    -------
    pd.DataFrame
        Player-season bowling WAR with columns: season, player, bowling_war_gam
    """
    if verbose:
        print("\nComputing GAM-based bowling WAR...")

    df = bbb.copy()

    gam, year_codes = train_bowling_gam(df, year_col=year_col)

    df["year_code"] = df[year_col].map(year_codes)
    df = df.dropna(subset=["Overs", "year_code"])

    X = df[["Overs", "year_code"]].values
    df["expected_runs"] = gam.predict(X)

    player_season = (
        df.groupby([year_col, player_col])
        .agg(
            actual_runs=("TotalRun", "sum"),
            expected_runs=("expected_runs", "sum"),
            balls_bowled=("TotalRun", "count"),
            wickets=("IsWicketDelivery", "sum"),
        )
        .reset_index()
    )

    player_season["wicket_bonus"] = player_season["wickets"] * WICKET_BONUS
    player_season["bowling_war_gam"] = (
        player_season["expected_runs"]
        - player_season["actual_runs"]
        + player_season["wicket_bonus"]
    ) / RUNS_PER_WIN

    if verbose:
        print(f"  Computed GAM bowling WAR for {len(player_season):,} player-seasons")
        top5 = player_season.nlargest(5, "bowling_war_gam")
        print("  Top 5 by bowling_war_gam:")
        for _, row in top5.iterrows():
            print(f"    {row[player_col]}: {row['bowling_war_gam']:.2f}")

    player_season = player_season.rename(
        columns={year_col: "season", player_col: "player"}
    )
    return player_season[["season", "player", "bowling_war_gam"]]


def validate_gam_war(war_df, verbose=True):
    """
    Validate GAM WAR by comparing with naive WAR.

    Parameters
    ----------
    war_df : pd.DataFrame
        WAR dataframe with both naive and GAM WAR columns
    verbose : bool
        Print validation results

    Returns
    -------
    dict
        Validation statistics including correlations
    """
    if not verbose:
        return {}

    print("\n" + "=" * 60)
    print("GAM WAR VALIDATION")
    print("=" * 60)

    stats = {}

    if "batting_war" in war_df.columns and "batting_war_gam" in war_df.columns:
        valid = war_df[["batting_war", "batting_war_gam"]].dropna()
        corr = valid["batting_war"].corr(valid["batting_war_gam"])
        stats["batting_correlation"] = corr
        print(f"\nBatting WAR correlation (naive vs GAM): {corr:.3f}")

    if "bowling_war" in war_df.columns and "bowling_war_gam" in war_df.columns:
        valid = war_df[["bowling_war", "bowling_war_gam"]].dropna()
        corr = valid["bowling_war"].corr(valid["bowling_war_gam"])
        stats["bowling_correlation"] = corr
        print(f"Bowling WAR correlation (naive vs GAM): {corr:.3f}")

    if "total_war" in war_df.columns and "total_war_gam" in war_df.columns:
        valid = war_df[["total_war", "total_war_gam"]].dropna()
        corr = valid["total_war"].corr(valid["total_war_gam"])
        stats["total_correlation"] = corr
        print(f"Total WAR correlation (naive vs GAM): {corr:.3f}")

        top10_naive = set(war_df.nlargest(10, "total_war")["player"])
        top10_gam = set(war_df.nlargest(10, "total_war_gam")["player"])
        overlap = len(top10_naive & top10_gam)
        stats["top10_overlap"] = overlap
        print(f"\nTop 10 overlap (naive vs GAM): {overlap}/10 players")

    print("\nGAM WAR distribution:")
    if "total_war_gam" in war_df.columns:
        print(f"  Mean:   {war_df['total_war_gam'].mean():.2f}")
        print(f"  Median: {war_df['total_war_gam'].median():.2f}")
        print(f"  Std:    {war_df['total_war_gam'].std():.2f}")
        print(f"  Min:    {war_df['total_war_gam'].min():.2f}")
        print(f"  Max:    {war_df['total_war_gam'].max():.2f}")

    return stats
