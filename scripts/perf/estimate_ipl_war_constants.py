#!/usr/bin/env python3
"""
Empirically estimate WAR constants from IPL ball-by-ball data.

Estimates year-specific constants (contemporaneous):
- RUNS_PER_WICKET: From innings regression
- RUNS_PER_WIN: From match outcomes (mean victory margin)

Uses same methodology as T20I estimation script but with IPL data.
"""

import warnings
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
PERF_DIR = DATA_DIR / "perf"
KAGGLE_DIR = DATA_DIR / "perf" / "sources" / "kaggle"


def load_ipl_deliveries():
    """Load IPL ball-by-ball data."""
    print("Loading IPL deliveries data...")
    deliveries = pd.read_csv(KAGGLE_DIR / "ball_by_ball_ipl.csv")
    deliveries["year"] = pd.to_datetime(deliveries["Date"]).dt.year
    print(f"  Loaded {len(deliveries):,} deliveries")
    print(f"  Years: {sorted(deliveries['year'].unique())}")
    return deliveries


def aggregate_to_innings(deliveries):
    """Aggregate ball-by-ball data to innings level."""
    innings = (
        deliveries.groupby(["Match ID", "Innings", "year"])
        .agg(
            total_runs=("Runs From Ball", "sum"),
            balls_faced=("Runs From Ball", "count"),
            wickets_lost=("Wicket", "sum"),
            batting_team=("Bat First", "first"),
        )
        .reset_index()
    )
    innings["overs_batted"] = innings["balls_faced"] / 6
    return innings


def estimate_runs_per_wicket_by_year(deliveries):
    """
    Estimate RUNS_PER_WICKET for each year using OLS regression.

    Model: total_runs = alpha + beta * wickets_lost + gamma * overs_batted + epsilon

    Returns dataframe with year and runs_per_wicket columns.
    """
    print("\nEstimating RUNS_PER_WICKET by year...")

    innings = aggregate_to_innings(deliveries)
    results = []

    for year in sorted(innings["year"].unique()):
        year_innings = innings[innings["year"] == year].copy()

        if len(year_innings) < 20:
            print(f"  {year}: insufficient data ({len(year_innings)} innings)")
            continue

        X = year_innings[["wickets_lost", "overs_batted"]]
        X = sm.add_constant(X)
        y = year_innings["total_runs"]

        model = sm.OLS(y, X).fit()
        wicket_coef = model.params["wickets_lost"]
        runs_per_wicket = abs(wicket_coef)

        results.append({
            "year": year,
            "runs_per_wicket": runs_per_wicket,
            "wicket_coef": wicket_coef,
            "wicket_se": model.bse["wickets_lost"],
            "n_innings": len(year_innings),
        })
        print(f"  {year}: {runs_per_wicket:.2f} runs/wicket (n={len(year_innings)})")

    return pd.DataFrame(results)


def build_match_outcomes(deliveries):
    """Build match outcome dataset with team run totals."""
    match_team_runs = (
        deliveries.groupby(["Match ID", "Innings", "year"])
        .agg(total_runs=("Runs From Ball", "sum"), batting_team=("Bat First", "first"))
        .reset_index()
    )

    first_innings = match_team_runs[match_team_runs["Innings"] == 1][
        ["Match ID", "year", "batting_team", "total_runs"]
    ].rename(columns={"batting_team": "team1", "total_runs": "team1_runs"})

    second_innings = match_team_runs[match_team_runs["Innings"] == 2][
        ["Match ID", "total_runs"]
    ].rename(columns={"total_runs": "team2_runs"})

    match_runs = first_innings.merge(second_innings, on="Match ID", how="inner")

    winners = deliveries.groupby("Match ID")["Winner"].first().reset_index()
    match_runs = match_runs.merge(winners, on="Match ID", how="inner")
    match_runs = match_runs[match_runs["Winner"].notna()]

    match_runs["team1_won"] = (match_runs["Winner"] == match_runs["team1"]).astype(int)
    match_runs["run_differential"] = match_runs["team1_runs"] - match_runs["team2_runs"]

    return match_runs


def estimate_runs_per_win_by_year(deliveries):
    """
    Estimate RUNS_PER_WIN for each year using mean victory margin.

    This is more stable than logistic regression when sample sizes are small.
    The mean margin of victory approximates the runs that separate winner from loser.

    Returns dataframe with year and runs_per_win columns.
    """
    print("\nEstimating RUNS_PER_WIN by year (mean victory margin)...")

    match_outcomes = build_match_outcomes(deliveries)
    results = []

    for year in sorted(match_outcomes["year"].unique()):
        year_matches = match_outcomes[match_outcomes["year"] == year].copy()

        if len(year_matches) < 20:
            print(f"  {year}: insufficient data ({len(year_matches)} matches)")
            continue

        year_matches["win_margin"] = abs(year_matches["run_differential"])
        mean_margin = year_matches["win_margin"].mean()
        median_margin = year_matches["win_margin"].median()

        results.append({
            "year": year,
            "runs_per_win": mean_margin,
            "median_margin": median_margin,
            "n_matches": len(year_matches),
        })
        print(f"  {year}: {mean_margin:.1f} runs/win (median={median_margin:.1f}, n={len(year_matches)})")

    return pd.DataFrame(results)


def estimate_overall_constants(deliveries):
    """Estimate overall constants (pooled across all years) for reference."""
    print("\n" + "=" * 60)
    print("OVERALL IPL CONSTANTS (pooled across all years)")
    print("=" * 60)

    innings = aggregate_to_innings(deliveries)
    X = innings[["wickets_lost", "overs_batted"]]
    X = sm.add_constant(X)
    y = innings["total_runs"]
    model = sm.OLS(y, X).fit()
    runs_per_wicket = abs(model.params["wickets_lost"])
    print(f"RUNS_PER_WICKET: {runs_per_wicket:.2f}")

    match_outcomes = build_match_outcomes(deliveries)
    match_outcomes["win_margin"] = abs(match_outcomes["run_differential"])
    runs_per_win = match_outcomes["win_margin"].mean()
    print(f"RUNS_PER_WIN: {runs_per_win:.2f} (mean victory margin)")

    return {"runs_per_wicket": runs_per_wicket, "runs_per_win": runs_per_win}


def main():
    print("=" * 60)
    print("IPL WAR CONSTANTS ESTIMATION (Contemporaneous)")
    print("=" * 60)

    deliveries = load_ipl_deliveries()

    rpw_df = estimate_runs_per_wicket_by_year(deliveries)
    rpwin_df = estimate_runs_per_win_by_year(deliveries)

    constants_df = rpw_df.merge(rpwin_df, on="year", how="outer")

    overall = estimate_overall_constants(deliveries)

    print("\n" + "=" * 60)
    print("YEAR-SPECIFIC CONSTANTS SUMMARY")
    print("=" * 60)
    print(constants_df[["year", "runs_per_wicket", "runs_per_win", "n_innings", "n_matches"]].to_string(index=False))

    print("\n  Statistics:")
    print(f"    runs_per_wicket: mean={constants_df['runs_per_wicket'].mean():.2f}, std={constants_df['runs_per_wicket'].std():.2f}")
    print(f"    runs_per_win: mean={constants_df['runs_per_win'].mean():.2f}, std={constants_df['runs_per_win'].std():.2f}")

    output_path = PERF_DIR / "ipl_constants_by_year.csv"
    constants_df.to_csv(output_path, index=False)
    print(f"\n  Saved to {output_path}")

    return constants_df, overall


if __name__ == "__main__":
    main()
