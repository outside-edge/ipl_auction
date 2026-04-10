#!/usr/bin/env python3
"""
Empirically estimate WAR constants from ball-by-ball data.

Uses innings-level regression to estimate runs per wicket:
    Innings_Total_Runs = α + β × Wickets_Lost + γ × Overs_Batted + ε

The coefficient β represents the average runs lost per wicket (negative),
so |β| = RUNS_PER_DISMISSAL = RUNS_PER_WICKET.

Uses T20I data which has the same format and game length as IPL.
"""

import pandas as pd
import statsmodels.api as sm
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
T20I_DIR = DATA_DIR / "perf" / "t20i"


def load_deliveries():
    """Load T20I ball-by-ball data."""
    print("Loading T20I deliveries data...")
    deliveries = pd.read_csv(T20I_DIR / "deliveries.csv")
    print(f"  Loaded {len(deliveries):,} deliveries")
    return deliveries


def aggregate_to_innings(deliveries):
    """Aggregate ball-by-ball data to innings level."""
    print("\nAggregating to innings level...")

    innings = (
        deliveries.groupby(["ID", "Innings"])
        .agg(
            total_runs=("TotalRun", "sum"),
            balls_faced=("TotalRun", "count"),
            wickets_lost=("IsWicketDelivery", "sum"),
            batting_team=("BattingTeam", "first"),
        )
        .reset_index()
    )

    innings["overs_batted"] = innings["balls_faced"] / 6

    completed = innings[innings["overs_batted"] >= 15].copy()
    print(f"  Total innings: {len(innings):,}")
    print(f"  Innings with 15+ overs: {len(completed):,}")

    return innings, completed


def estimate_runs_per_wicket(innings_df):
    """
    Estimate runs per wicket using OLS regression.

    Model: total_runs = α + β × wickets_lost + γ × overs_batted + ε
    """
    print("\n" + "=" * 60)
    print("REGRESSION: Runs ~ Wickets + Overs")
    print("=" * 60)

    X = innings_df[["wickets_lost", "overs_batted"]]
    X = sm.add_constant(X)
    y = innings_df["total_runs"]

    model = sm.OLS(y, X).fit()
    print(model.summary())

    wicket_coef = model.params["wickets_lost"]
    wicket_se = model.bse["wickets_lost"]
    wicket_pval = model.pvalues["wickets_lost"]

    print("\n" + "=" * 60)
    print("KEY RESULTS")
    print("=" * 60)
    print(f"Coefficient on wickets_lost: {wicket_coef:.4f}")
    print(f"Standard error: {wicket_se:.4f}")
    print(f"p-value: {wicket_pval:.6f}")
    print(f"95% CI: [{wicket_coef - 1.96*wicket_se:.4f}, {wicket_coef + 1.96*wicket_se:.4f}]")

    runs_per_wicket = abs(wicket_coef)
    print(f"\nESTIMATED RUNS_PER_WICKET = {runs_per_wicket:.2f}")

    return runs_per_wicket, model


def phase_analysis(deliveries):
    """Analyze runs per wicket by game phase (optional)."""
    print("\n" + "=" * 60)
    print("PHASE-SPECIFIC ANALYSIS")
    print("=" * 60)

    def get_phase(over):
        if over <= 5:
            return "powerplay"
        elif over <= 14:
            return "middle"
        else:
            return "death"

    deliveries = deliveries.copy()
    deliveries["phase"] = deliveries["Overs"].apply(get_phase)

    phase_innings = (
        deliveries.groupby(["ID", "Innings", "phase"])
        .agg(
            total_runs=("TotalRun", "sum"),
            balls=("TotalRun", "count"),
            wickets=("IsWicketDelivery", "sum"),
        )
        .reset_index()
    )
    phase_innings["overs"] = phase_innings["balls"] / 6

    results = {}
    for phase in ["powerplay", "middle", "death"]:
        df = phase_innings[phase_innings["phase"] == phase].copy()
        df = df[df["overs"] >= 2]

        X = df[["wickets", "overs"]]
        X = sm.add_constant(X)
        y = df["total_runs"]

        model = sm.OLS(y, X).fit()
        coef = model.params["wickets"]
        se = model.bse["wickets"]

        results[phase] = {"coef": coef, "se": se, "runs_per_wicket": abs(coef)}
        print(f"{phase:12s}: coef = {coef:7.3f} (SE={se:.3f}), runs/wicket = {abs(coef):.2f}")

    return results


def sanity_checks(innings_df, runs_per_wicket):
    """Perform sanity checks on the estimated value."""
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    print("\n1. Value range check:")
    if 5 <= runs_per_wicket <= 15:
        print(f"   ✓ {runs_per_wicket:.2f} is within expected T20 range (5-15)")
    else:
        print(f"   ✗ {runs_per_wicket:.2f} is outside expected T20 range (5-15)")

    print("\n2. Innings statistics:")
    print(f"   Mean runs: {innings_df['total_runs'].mean():.1f}")
    print(f"   Mean wickets: {innings_df['wickets_lost'].mean():.1f}")
    print(f"   Mean overs: {innings_df['overs_batted'].mean():.1f}")

    print("\n3. Correlation check:")
    corr = innings_df["total_runs"].corr(innings_df["wickets_lost"])
    print(f"   Correlation(runs, wickets) = {corr:.3f}")
    if corr < 0:
        print("   ✓ Negative correlation (more wickets = fewer runs)")
    else:
        print("   ✗ Positive correlation - unexpected!")

    print("\n4. All-out innings check (10 wickets):")
    all_out = innings_df[innings_df["wickets_lost"] == 10]
    if len(all_out) > 0:
        print(f"   All-out innings: {len(all_out)}")
        print(f"   Mean runs when all out: {all_out['total_runs'].mean():.1f}")


def main():
    print("=" * 60)
    print("EMPIRICAL ESTIMATION OF WAR CONSTANTS")
    print("=" * 60)

    deliveries = load_deliveries()

    innings_all, _ = aggregate_to_innings(deliveries)

    print("\nUsing all innings (including incomplete):")
    runs_per_wicket_all, _ = estimate_runs_per_wicket(innings_all)

    sanity_checks(innings_all, runs_per_wicket_all)

    phase_results = phase_analysis(deliveries)

    print("\n" + "=" * 60)
    print("FINAL RECOMMENDATION")
    print("=" * 60)
    print(f"RUNS_PER_DISMISSAL = {runs_per_wicket_all:.1f}")
    print(f"RUNS_PER_WICKET = {runs_per_wicket_all:.1f}")
    print("\nPhase-specific values (for reference):")
    for phase, result in phase_results.items():
        print(f"  {phase}: {result['runs_per_wicket']:.1f}")

    return runs_per_wicket_all


if __name__ == "__main__":
    main()
