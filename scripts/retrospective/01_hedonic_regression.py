#!/usr/bin/env python3
"""
Hedonic Wage Regression for IPL Auction Prices.

Applies classic labor economics approaches to cricket player valuation:
- Hedonic wage model (Rosen 1974, Rastogi & Deodhar 2009)
- WAR-based specifications using context-adjusted performance (Scully 1974)
- Panel data methods: Player FE, Year FE, Two-Way FE, First Difference
- Variance decomposition (Between vs Within)
- Quantile regression for heterogeneous effects
- Superstar premium analysis
- Market efficiency test: Do prices predict future performance?

IMPORTANT: Retrospective vs Prospective Analysis
================================================
This script performs RETROSPECTIVE (oracle) analysis where same-season
performance is used as a regressor. This answers the question:
"Given what players actually achieved, were prices fair?"

This is NOT predictive - prices are set BEFORE the season, so teams cannot
know actual performance at auction time. Models using same-season data
(baseline, full_performance, war_current, war_components, war_mega, etc.)
have intentional "lookahead" bias.

For PREDICTIVE analysis, use the lagged specifications (lagged, full_lagged,
war_lagged) which only use prior-season data available at auction time.

The scripts in scripts/prediction/ provide proper out-of-sample prediction
with temporal train/test splits.

Output: Regression tables saved to tabs/regression_results.txt
"""

import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg
from linearmodels.panel import PanelOLS, PooledOLS, BetweenOLS, FirstDifferenceOLS

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from shared.inflation import adjust_prices_for_inflation
from shared.io import load_dataset

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = BASE_DIR / "data"
JOINED_DIR = DATA_DIR / "analysis" / "joined"
TABS_DIR = BASE_DIR / "tabs"

MEGA_AUCTION_YEARS = [2008, 2011, 2014, 2018, 2022, 2025]


def load_analysis_data():
    """Load auction data with performance metrics and apply inflation adjustment."""
    df = load_dataset(JOINED_DIR / "auction_with_performance")

    if "price_2024_cr" not in df.columns:
        df = adjust_prices_for_inflation(df)

    df["price_2024_cr"] = pd.to_numeric(df["price_2024_cr"], errors="coerce")
    df = df[df["price_2024_cr"] > 0].copy()

    df["log_price"] = np.log(df["price_2024_cr"])

    df["is_indian"] = (df["nationality"] == "Indian").astype(int)
    df["is_overseas"] = (df["nationality"] == "Overseas").astype(int)

    df["is_batsman"] = (df["role"] == "Batsman").astype(int)
    df["is_bowler"] = (df["role"] == "Bowler").astype(int)
    df["is_allrounder"] = (df["role"] == "All-Rounder").astype(int)
    df["is_wicketkeeper"] = (df["role"] == "Wicket-Keeper").astype(int)

    df["is_mega_auction"] = df["year"].isin(MEGA_AUCTION_YEARS).astype(int)

    numeric_cols = [
        "runs", "batting_avg", "batting_sr", "wickets",
        "bowling_avg", "economy", "catches", "matches_played",
        "batting_war", "bowling_war", "total_war"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df.loc[:, col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def create_lagged_performance(df):
    """Create lagged performance variables (performance from prior season)."""
    df = df.sort_values(["player_name", "year"]).copy()

    perf_cols = [
        "runs", "batting_avg", "batting_sr", "wickets",
        "bowling_avg", "economy", "catches", "matches_played",
        "batting_war", "bowling_war", "total_war"
    ]

    for col in perf_cols:
        if col in df.columns:
            df[f"{col}_lag"] = df.groupby("player_name")[col].shift(1)

    return df


def create_future_performance(df):
    """Create future performance variables for market efficiency test."""
    df = df.sort_values(["player_name", "year"]).copy()

    perf_cols = ["total_war", "runs", "wickets", "batting_war", "bowling_war"]

    for col in perf_cols:
        if col in df.columns:
            df[f"{col}_future"] = df.groupby("player_name")[col].shift(-1)

    return df


def create_panel_data(df):
    """Create panel dataset with players appearing multiple times."""
    player_counts = df.groupby("player_name").size()
    repeat_players = player_counts[player_counts > 1].index
    df_panel = df[df["player_name"].isin(repeat_players)].copy()
    df_panel = df_panel.set_index(["player_name", "year"])
    df_panel = df_panel.sort_index()
    return df_panel


def estimate_pooled_ols(df, spec_name="baseline"):
    """
    Estimate pooled OLS with various specifications.

    Specifications are categorized as:
    - PREDICTIVE (lagged): Uses only prior-season data (no lookahead)
    - RETROSPECTIVE (oracle): Uses same-season data (intentional lookahead)

    Retrospective models answer: "Given actual performance, was price fair?"
    Predictive models answer: "Can we forecast value from available data?"
    """
    df_reg = df.dropna(subset=["log_price"]).copy()

    # RETROSPECTIVE (oracle) models - use same-season performance
    if spec_name == "baseline":
        # Oracle: uses same-season runs/wickets
        X_vars = ["runs", "wickets", "is_indian"]
        df_subset = df_reg[df_reg["runs"].notna() & df_reg["wickets"].notna()]

    elif spec_name == "full_performance":
        # Oracle: uses same-season performance metrics
        X_vars = [
            "runs", "batting_avg", "batting_sr",
            "wickets", "economy", "catches",
            "is_indian"
        ]
        df_subset = df_reg.dropna(subset=X_vars)

    # PREDICTIVE models - use only prior-season data (available at auction time)
    elif spec_name == "lagged":
        # Predictive: uses prior season only
        X_vars = ["runs_lag", "wickets_lag", "is_indian"]
        df_subset = df_reg.dropna(subset=X_vars)

    elif spec_name == "full_lagged":
        # Predictive: uses prior season only
        X_vars = [
            "runs_lag", "batting_avg_lag", "batting_sr_lag",
            "wickets_lag", "economy_lag", "catches_lag",
            "is_indian"
        ]
        df_subset = df_reg.dropna(subset=X_vars)

    # RETROSPECTIVE WAR models
    elif spec_name == "war_current":
        # Oracle: uses same-season WAR
        X_vars = ["total_war", "is_indian"]
        df_subset = df_reg.dropna(subset=X_vars)
        df_subset = df_subset[df_subset["total_war"] != 0]

    elif spec_name == "war_components":
        # Oracle: uses same-season WAR components
        X_vars = ["batting_war", "bowling_war", "is_indian"]
        df_subset = df_reg.dropna(subset=X_vars)
        df_subset = df_subset[
            (df_subset["batting_war"] != 0) | (df_subset["bowling_war"] != 0)
        ]

    # PREDICTIVE WAR model
    elif spec_name == "war_lagged":
        # Predictive: uses prior season WAR
        X_vars = ["total_war_lag", "is_indian"]
        df_subset = df_reg.dropna(subset=X_vars)
        df_subset = df_subset[df_subset["total_war_lag"] != 0]

    # RETROSPECTIVE with controls
    elif spec_name == "war_mega":
        # Oracle: uses same-season WAR + mega auction
        X_vars = ["total_war", "is_indian", "is_mega_auction"]
        df_subset = df_reg.dropna(subset=X_vars)
        df_subset = df_subset[df_subset["total_war"] != 0]

    elif spec_name == "lagged_current":
        # Mixed: uses both lagged and current (for decomposition)
        X_vars = ["runs", "wickets", "runs_lag", "wickets_lag", "is_indian"]
        df_subset = df_reg.dropna(subset=X_vars)

    elif spec_name == "with_roles":
        # Oracle: uses same-season performance + roles
        X_vars = [
            "runs", "wickets", "is_indian",
            "is_batsman", "is_bowler", "is_allrounder"
        ]
        df_subset = df_reg.dropna(subset=["runs", "wickets"])

    else:
        raise ValueError(f"Unknown specification: {spec_name}")

    if len(df_subset) < 30:
        print(f"Warning: Only {len(df_subset)} observations for {spec_name}")
        return None

    X = df_subset[X_vars].copy()
    X = sm.add_constant(X)
    y = df_subset["log_price"]

    model = sm.OLS(y, X)
    results = model.fit(cov_type="HC1")

    return results


def estimate_market_efficiency(df):
    """
    Test market efficiency: Does price predict future performance?

    Model: future_war_{t+1} = alpha + beta * log(price_t) + gamma * X_t + eps

    If beta > 0 and significant, markets are informationally efficient.
    """
    df_reg = df.dropna(subset=["log_price", "total_war_future", "is_indian"]).copy()
    df_reg = df_reg[df_reg["total_war_future"] != 0]

    if len(df_reg) < 30:
        print(f"Warning: Only {len(df_reg)} observations for efficiency test")
        return None

    X_vars = ["log_price", "is_indian"]
    X = df_reg[X_vars].copy()
    X = sm.add_constant(X)
    y = df_reg["total_war_future"]

    model = sm.OLS(y, X)
    results = model.fit(cov_type="HC1")

    return results


def estimate_by_role(df, role):
    """Estimate separate models for batsmen/bowlers."""
    if role == "batsman":
        df_subset = df[df["is_batsman"] == 1].copy()
        X_vars = ["runs", "batting_avg", "batting_sr", "is_indian"]
    elif role == "bowler":
        df_subset = df[df["is_bowler"] == 1].copy()
        X_vars = ["wickets", "bowling_avg", "economy", "is_indian"]
    elif role == "allrounder":
        df_subset = df[df["is_allrounder"] == 1].copy()
        X_vars = ["runs", "wickets", "batting_avg", "economy", "is_indian"]
    else:
        raise ValueError(f"Unknown role: {role}")

    df_subset = df_subset.dropna(subset=["log_price"] + X_vars)

    if len(df_subset) < 20:
        print(f"Warning: Only {len(df_subset)} observations for {role}")
        return None

    X = df_subset[X_vars].copy()
    X = sm.add_constant(X)
    y = df_subset["log_price"]

    model = sm.OLS(y, X)
    results = model.fit(cov_type="HC1")

    return results


def estimate_panel_models(df_panel):
    """
    Estimate panel data models: Player FE, Year FE, Two-Way FE, First Difference.
    Returns dict of results.
    """
    results = {}
    exog_vars = ["runs", "wickets", "is_indian"]
    time_varying_vars = ["runs", "wickets"]

    df_reg = df_panel.dropna(subset=["log_price"] + exog_vars).copy()
    if len(df_reg) < 50:
        print(f"Warning: Only {len(df_reg)} observations for panel models")
        return results

    y = df_reg["log_price"]
    X = sm.add_constant(df_reg[exog_vars])
    X_tv = df_reg[time_varying_vars]

    pooled = PooledOLS(y, X)
    results["Pooled OLS"] = pooled.fit(cov_type="clustered", cluster_entity=True)

    player_fe = PanelOLS(y, X_tv, entity_effects=True)
    results["Player FE"] = player_fe.fit(cov_type="clustered", cluster_entity=True)

    time_fe = PanelOLS(y, X, time_effects=True)
    results["Year FE"] = time_fe.fit(cov_type="clustered", cluster_entity=True)

    two_way_fe = PanelOLS(y, X_tv, entity_effects=True, time_effects=True)
    results["Two-Way FE"] = two_way_fe.fit(cov_type="clustered", cluster_entity=True)

    fd_model = FirstDifferenceOLS(y, X_tv)
    results["First Diff"] = fd_model.fit(cov_type="robust")

    between = BetweenOLS(y, X_tv)
    results["Between"] = between.fit()

    return results


def estimate_war_panel_models(df_panel, has_war):
    """Estimate WAR-based panel models."""
    results = {}
    if not has_war:
        return results

    df_war = df_panel[df_panel["total_war"] != 0].copy()
    if len(df_war) < 50:
        print("Warning: Insufficient WAR data for panel models")
        return results

    df_war_reg = df_war.dropna(subset=["log_price", "total_war"]).copy()

    y = df_war_reg["log_price"]
    X_pooled = sm.add_constant(df_war_reg[["total_war", "is_indian"]])
    X_fe = df_war_reg[["total_war"]]

    pooled = PooledOLS(y, X_pooled)
    results["WAR Pooled"] = pooled.fit(cov_type="clustered", cluster_entity=True)

    player_fe = PanelOLS(y, X_fe, entity_effects=True)
    results["WAR Player FE"] = player_fe.fit(cov_type="clustered", cluster_entity=True)

    two_way = PanelOLS(y, X_fe, entity_effects=True, time_effects=True)
    results["WAR Two-Way FE"] = two_way.fit(cov_type="clustered", cluster_entity=True)

    return results


def estimate_superstar_premium(df):
    """
    Test whether top performers receive disproportionately higher prices.
    Superstar = top 10% of runs OR wickets in a given year.
    """
    df_star = df.copy()
    df_star["runs_percentile"] = df_star.groupby("year")["runs"].rank(pct=True)
    df_star["wickets_percentile"] = df_star.groupby("year")["wickets"].rank(pct=True)

    df_star["top_10pct_runs"] = (df_star["runs_percentile"] >= 0.90).astype(int)
    df_star["top_10pct_wickets"] = (df_star["wickets_percentile"] >= 0.90).astype(int)
    df_star["superstar"] = (
        (df_star["top_10pct_runs"] == 1) | (df_star["top_10pct_wickets"] == 1)
    ).astype(int)

    df_reg = df_star.dropna(subset=["log_price", "runs", "wickets", "is_indian"])

    if len(df_reg) < 30:
        return None

    X = sm.add_constant(df_reg[["runs", "wickets", "is_indian", "superstar"]])
    y = df_reg["log_price"]

    model = sm.OLS(y, X)
    results = model.fit(cov_type="HC1")

    return results


def estimate_quantile_regression(df):
    """
    Estimate quantile regression at 10th, 25th, 50th, 75th, 90th percentiles.
    Returns dict of results keyed by quantile.
    """
    df_qr = df.dropna(subset=["log_price", "runs", "wickets", "is_indian"]).copy()

    if len(df_qr) < 50:
        return {}

    X = sm.add_constant(df_qr[["runs", "wickets", "is_indian"]])
    y = df_qr["log_price"]

    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
    results = {}

    for q in quantiles:
        qr_model = QuantReg(y, X)
        results[q] = qr_model.fit(q=q)

    return results


def compute_variance_decomposition(panel_results, war_panel_results=None):
    """Compute variance decomposition from panel models."""
    decomp = {}

    if "Pooled OLS" in panel_results and "Player FE" in panel_results:
        pooled_r2 = panel_results["Pooled OLS"].rsquared
        fe_r2 = panel_results["Player FE"].rsquared_overall
        within_r2 = panel_results["Player FE"].rsquared_within

        pooled_unexplained = 1 - pooled_r2
        fe_unexplained = 1 - fe_r2
        if pooled_unexplained > 0:
            variance_absorbed = (pooled_unexplained - fe_unexplained) / pooled_unexplained * 100
        else:
            variance_absorbed = 0

        decomp["raw_stats"] = {
            "pooled_r2": pooled_r2,
            "fe_r2": fe_r2,
            "within_r2": within_r2,
            "variance_absorbed_pct": variance_absorbed,
        }

        if "Between" in panel_results:
            decomp["raw_stats"]["between_r2"] = panel_results["Between"].rsquared

    if war_panel_results and "WAR Pooled" in war_panel_results and "WAR Player FE" in war_panel_results:
        war_pooled_r2 = war_panel_results["WAR Pooled"].rsquared
        war_fe_r2 = war_panel_results["WAR Player FE"].rsquared_overall
        war_within_r2 = war_panel_results["WAR Player FE"].rsquared_within

        war_pooled_unexplained = 1 - war_pooled_r2
        war_fe_unexplained = 1 - war_fe_r2
        if war_pooled_unexplained > 0:
            war_variance_absorbed = (war_pooled_unexplained - war_fe_unexplained) / war_pooled_unexplained * 100
        else:
            war_variance_absorbed = 0

        decomp["war"] = {
            "pooled_r2": war_pooled_r2,
            "fe_r2": war_fe_r2,
            "within_r2": war_within_r2,
            "variance_absorbed_pct": war_variance_absorbed,
        }

    return decomp


def print_interpretation(results, spec_name):
    """Print economic interpretation of regression coefficients."""
    print(f"\n=== Interpretation: {spec_name} ===")

    params = results.params
    pvals = results.pvalues

    for var in params.index:
        if var == "const":
            continue

        coef = params[var]
        pval = pvals[var]
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""

        if var == "runs":
            pct = (np.exp(coef * 100) - 1) * 100
            print(f"  Runs: 100 additional runs -> {pct:.1f}% price increase{sig}")
        elif var == "runs_lag":
            pct = (np.exp(coef * 100) - 1) * 100
            print(f"  Runs (prior season): 100 additional runs -> {pct:.1f}% price increase{sig}")
        elif var == "wickets":
            pct = (np.exp(coef * 10) - 1) * 100
            print(f"  Wickets: 10 additional wickets -> {pct:.1f}% price increase{sig}")
        elif var == "wickets_lag":
            pct = (np.exp(coef * 10) - 1) * 100
            print(f"  Wickets (prior season): 10 additional wickets -> {pct:.1f}% price increase{sig}")
        elif var == "total_war":
            pct = (np.exp(coef) - 1) * 100
            print(f"  Total WAR: 1 WAR -> {pct:.1f}% price increase{sig}")
        elif var == "total_war_lag":
            pct = (np.exp(coef) - 1) * 100
            print(f"  Total WAR (prior season): 1 WAR -> {pct:.1f}% price increase{sig}")
        elif var == "batting_war":
            pct = (np.exp(coef) - 1) * 100
            print(f"  Batting WAR: 1 WAR -> {pct:.1f}% price increase{sig}")
        elif var == "bowling_war":
            pct = (np.exp(coef) - 1) * 100
            print(f"  Bowling WAR: 1 WAR -> {pct:.1f}% price increase{sig}")
        elif var == "is_indian":
            pct = (np.exp(coef) - 1) * 100
            print(f"  Indian nationality: {pct:.1f}% price premium{sig}")
        elif var == "is_mega_auction":
            pct = (np.exp(coef) - 1) * 100
            print(f"  Mega auction effect: {pct:.1f}% price change{sig}")
        elif var == "batting_avg":
            pct = (np.exp(coef * 10) - 1) * 100
            print(f"  Batting avg: 10 point increase -> {pct:.1f}% price change{sig}")
        elif var == "economy":
            pct = (np.exp(coef) - 1) * 100
            print(f"  Economy: 1 run/over increase -> {pct:.1f}% price change{sig}")
        elif var == "superstar":
            pct = (np.exp(coef) - 1) * 100
            print(f"  Superstar premium (top 10%): {pct:.1f}% price increase{sig}")


def write_results_to_file(
    output_path, ols_results, efficiency_result, panel_results,
    war_panel_results, superstar_result, qr_results, variance_decomp
):
    """Write all results to output file."""
    with open(output_path, "w") as f:
        f.write("IPL Auction Hedonic Wage Regression Results\n")
        f.write("=" * 70 + "\n\n")

        f.write("POOLED OLS MODELS\n")
        f.write("-" * 70 + "\n")
        for name, result in ols_results.items():
            if result:
                f.write(f"\n{name} Model:\n")
                f.write(str(result.summary()) + "\n")

        if efficiency_result:
            f.write("\n" + "=" * 70 + "\n")
            f.write("MARKET EFFICIENCY TEST\n")
            f.write("-" * 70 + "\n")
            f.write(str(efficiency_result.summary()) + "\n")

        if panel_results:
            f.write("\n" + "=" * 70 + "\n")
            f.write("PANEL DATA MODELS\n")
            f.write("-" * 70 + "\n")
            for name, result in panel_results.items():
                f.write(f"\n{name}:\n")
                f.write(str(result.summary) + "\n")

            f.write("\n--- Coefficient Comparison (runs, wickets) ---\n")
            coef_data = []
            for name in ["Pooled OLS", "Player FE", "Year FE", "Two-Way FE", "First Diff"]:
                if name in panel_results:
                    res = panel_results[name]
                    row = {"Model": name}
                    if "runs" in res.params.index:
                        row["runs"] = f"{res.params['runs']:.5f}"
                    if "wickets" in res.params.index:
                        row["wickets"] = f"{res.params['wickets']:.5f}"
                    coef_data.append(row)
            if coef_data:
                f.write(pd.DataFrame(coef_data).to_string(index=False) + "\n")

        if war_panel_results:
            f.write("\n" + "=" * 70 + "\n")
            f.write("WAR-BASED PANEL MODELS\n")
            f.write("-" * 70 + "\n")
            for name, result in war_panel_results.items():
                f.write(f"\n{name}:\n")
                f.write(str(result.summary) + "\n")

        if variance_decomp:
            f.write("\n" + "=" * 70 + "\n")
            f.write("VARIANCE DECOMPOSITION\n")
            f.write("-" * 70 + "\n")
            if "raw_stats" in variance_decomp:
                d = variance_decomp["raw_stats"]
                f.write("\nRaw Stats Models:\n")
                f.write(f"  Pooled OLS R²:     {d['pooled_r2']:.3f}\n")
                f.write(f"  Player FE R²:      {d['fe_r2']:.3f}\n")
                f.write(f"  Within R²:         {d['within_r2']:.3f}\n")
                if "between_r2" in d:
                    f.write(f"  Between R²:        {d['between_r2']:.3f}\n")
                f.write(f"\n  Player FE absorbs {d['variance_absorbed_pct']:.1f}% of pooled residual variance\n")
            if "war" in variance_decomp:
                d = variance_decomp["war"]
                f.write("\nWAR Models:\n")
                f.write(f"  WAR Pooled R²:     {d['pooled_r2']:.3f}\n")
                f.write(f"  WAR Player FE R²:  {d['fe_r2']:.3f}\n")
                f.write(f"  WAR Within R²:     {d['within_r2']:.3f}\n")
                f.write(f"\n  Player FE absorbs {d['variance_absorbed_pct']:.1f}% of WAR model residual variance\n")

        if superstar_result:
            f.write("\n" + "=" * 70 + "\n")
            f.write("SUPERSTAR PREMIUM MODEL\n")
            f.write("-" * 70 + "\n")
            f.write(str(superstar_result.summary()) + "\n")
            pct = (np.exp(superstar_result.params["superstar"]) - 1) * 100
            f.write(f"\nSuperstar premium (top 10% runs OR wickets): {pct:.1f}%\n")

        if qr_results:
            f.write("\n" + "=" * 70 + "\n")
            f.write("QUANTILE REGRESSION\n")
            f.write("-" * 70 + "\n")
            qr_coefs = pd.DataFrame({q: res.params for q, res in qr_results.items()})
            f.write("\nCoefficients by Quantile:\n")
            f.write(qr_coefs.round(5).to_string() + "\n")

            f.write("\n--- Interpretation ---\n")
            f.write("Q10 = bottom 10% of prices (budget players)\n")
            f.write("Q50 = median price\n")
            f.write("Q90 = top 10% of prices (premium players)\n")


def main():
    print("Loading data...")
    df = load_analysis_data()
    print(f"  Loaded {len(df)} auction records")

    has_war = "total_war" in df.columns and df["total_war"].notna().any()
    if has_war:
        war_count = (df["total_war"] != 0).sum()
        print(f"  WAR data available for {war_count} records")
    else:
        print("  WAR data not available")

    print("\nCreating lagged performance variables...")
    df = create_lagged_performance(df)

    print("Creating future performance variables...")
    df = create_future_performance(df)

    print("\n" + "=" * 60)
    print("HEDONIC WAGE REGRESSION ANALYSIS")
    print("=" * 60)

    ols_results = {}

    print("\n" + "=" * 60)
    print("PRIMARY MODELS: LAGGED PERFORMANCE (No Selection Bias)")
    print("=" * 60)

    print("\n1. Lagged Performance Model (prior season stats)")
    ols_results["Lagged"] = estimate_pooled_ols(df, "lagged")
    if ols_results["Lagged"]:
        print(ols_results["Lagged"].summary().tables[1])
        print_interpretation(ols_results["Lagged"], "Lagged")

    print("\n2. Full Lagged Model (all prior season metrics)")
    ols_results["Full Lag"] = estimate_pooled_ols(df, "full_lagged")
    if ols_results["Full Lag"]:
        print(ols_results["Full Lag"].summary().tables[1])

    if has_war:
        print("\n3. WAR Lagged (prior season WAR)")
        ols_results["WAR Lag"] = estimate_pooled_ols(df, "war_lagged")
        if ols_results["WAR Lag"]:
            print(ols_results["WAR Lag"].summary().tables[1])
            print_interpretation(ols_results["WAR Lag"], "WAR Lagged")

    print("\n" + "=" * 60)
    print("SECONDARY MODELS: SAME-SEASON (Selection Bias Caveat)")
    print("=" * 60)

    print("\n4. Baseline Model (runs + wickets + nationality)")
    ols_results["Baseline"] = estimate_pooled_ols(df, "baseline")
    if ols_results["Baseline"]:
        print(ols_results["Baseline"].summary().tables[1])
        print_interpretation(ols_results["Baseline"], "Baseline")

    print("\n5. Full Performance Model")
    ols_results["Full"] = estimate_pooled_ols(df, "full_performance")
    if ols_results["Full"]:
        print(ols_results["Full"].summary().tables[1])

    if has_war:
        print("\n" + "=" * 60)
        print("WAR-BASED MODELS (Same-Season)")
        print("=" * 60)

        print("\n6. WAR Current Season")
        ols_results["WAR"] = estimate_pooled_ols(df, "war_current")
        if ols_results["WAR"]:
            print(ols_results["WAR"].summary().tables[1])
            print_interpretation(ols_results["WAR"], "WAR Current")

        print("\n7. WAR Components (Batting + Bowling)")
        ols_results["WAR Comp"] = estimate_pooled_ols(df, "war_components")
        if ols_results["WAR Comp"]:
            print(ols_results["WAR Comp"].summary().tables[1])

    print("\n" + "=" * 60)
    print("MARKET EFFICIENCY TEST")
    print("=" * 60)

    print("\nTesting: Does price predict FUTURE performance?")
    efficiency_result = estimate_market_efficiency(df)
    if efficiency_result:
        print(efficiency_result.summary().tables[1])
        price_coef = efficiency_result.params.get("log_price", 0)
        price_pval = efficiency_result.pvalues.get("log_price", 1)
        sig = "***" if price_pval < 0.01 else "**" if price_pval < 0.05 else "*" if price_pval < 0.1 else ""
        print(f"\n  Price coefficient: {price_coef:.4f} (p={price_pval:.3f}){sig}")
        print(f"  R-squared: {efficiency_result.rsquared:.3f}")

    print("\n" + "=" * 60)
    print("ROLE-SPECIFIC MODELS")
    print("=" * 60)

    print("\n8. Batsmen Only")
    ols_results["Batsmen"] = estimate_by_role(df, "batsman")
    if ols_results["Batsmen"]:
        print(ols_results["Batsmen"].summary().tables[1])

    print("\n9. Bowlers Only")
    ols_results["Bowlers"] = estimate_by_role(df, "bowler")
    if ols_results["Bowlers"]:
        print(ols_results["Bowlers"].summary().tables[1])

    print("\n" + "=" * 60)
    print("PANEL DATA MODELS")
    print("=" * 60)

    print("\nCreating panel data (players with multiple appearances)...")
    df_panel = create_panel_data(df)
    n_players = df_panel.index.get_level_values(0).nunique()
    n_years = df_panel.index.get_level_values(1).nunique()
    print(f"  Panel: {len(df_panel)} obs from {n_players} players over {n_years} years")

    print("\n10. Panel Estimators (Player FE, Year FE, Two-Way FE, First Diff)")
    panel_results = estimate_panel_models(df_panel)

    if panel_results:
        print("\nCoefficient Comparison:")
        coef_data = []
        for name in ["Pooled OLS", "Player FE", "Year FE", "Two-Way FE", "First Diff"]:
            if name in panel_results:
                res = panel_results[name]
                row = {"Model": name}
                if "runs" in res.params.index:
                    row["runs"] = f"{res.params['runs']:.5f}"
                if "wickets" in res.params.index:
                    row["wickets"] = f"{res.params['wickets']:.5f}"
                coef_data.append(row)
        if coef_data:
            print(pd.DataFrame(coef_data).to_string(index=False))

    war_panel_results = {}
    if has_war:
        print("\n11. WAR-Based Panel Models")
        war_panel_results = estimate_war_panel_models(df_panel, has_war)
        if war_panel_results:
            for name, res in war_panel_results.items():
                print(f"\n{name}: total_war coef = {res.params.get('total_war', 'N/A'):.4f}")

    print("\n" + "=" * 60)
    print("VARIANCE DECOMPOSITION")
    print("=" * 60)

    variance_decomp = compute_variance_decomposition(panel_results, war_panel_results)
    if "raw_stats" in variance_decomp:
        d = variance_decomp["raw_stats"]
        print(f"\nPooled OLS R²:     {d['pooled_r2']:.3f}")
        print(f"Player FE R²:      {d['fe_r2']:.3f}")
        print(f"Within R²:         {d['within_r2']:.3f}")
        if "between_r2" in d:
            print(f"Between R²:        {d['between_r2']:.3f}")
        print(f"\nPlayer FE absorbs {d['variance_absorbed_pct']:.1f}% of pooled residual variance")
        print("(This represents time-invariant player characteristics: reputation, brand, ability)")

    print("\n" + "=" * 60)
    print("SUPERSTAR PREMIUM ANALYSIS")
    print("=" * 60)

    print("\n12. Superstar Premium (top 10% performers)")
    superstar_result = estimate_superstar_premium(df)
    if superstar_result:
        print(superstar_result.summary().tables[1])
        pct = (np.exp(superstar_result.params["superstar"]) - 1) * 100
        print(f"\nSuperstar premium (top 10% runs OR wickets): {pct:.1f}%")

    print("\n" + "=" * 60)
    print("QUANTILE REGRESSION")
    print("=" * 60)

    print("\n13. Quantile Regression (effects at different price levels)")
    qr_results = estimate_quantile_regression(df)
    if qr_results:
        qr_coefs = pd.DataFrame({f"Q{int(q*100)}": res.params for q, res in qr_results.items()})
        print("\nCoefficients by Quantile:")
        print(qr_coefs.round(5).to_string())

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print("\nOLS Model Comparison:")
    comparison_models = ["Lagged", "Full Lag"]
    if has_war:
        comparison_models.append("WAR Lag")
    comparison_models.extend(["Baseline", "Full"])
    if has_war:
        comparison_models.append("WAR")

    comparison_data = []
    for name in comparison_models:
        if name in ols_results and ols_results[name] is not None:
            comparison_data.append({
                "Model": name,
                "R-squared": f"{ols_results[name].rsquared:.3f}",
                "Adj R-sq": f"{ols_results[name].rsquared_adj:.3f}",
                "N": int(ols_results[name].nobs)
            })
    if comparison_data:
        print(pd.DataFrame(comparison_data).to_string(index=False))

    TABS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TABS_DIR / "regression_results.txt"
    write_results_to_file(
        output_path, ols_results, efficiency_result, panel_results,
        war_panel_results, superstar_result, qr_results, variance_decomp
    )
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
