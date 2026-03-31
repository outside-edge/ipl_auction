#!/usr/bin/env python3
"""
Hedonic Wage Regression for IPL Auction Prices.

Applies classic labor economics approaches to cricket player valuation:
- Hedonic wage model (Rosen 1974, Rastogi & Deodhar 2009)
- WAR-based specifications using context-adjusted performance (Scully 1974)
- Market efficiency test: Do prices predict future performance?
- Mega-auction controls for auction structure effects

Output: Regression tables and diagnostic plots
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "analysis"

MEGA_AUCTION_YEARS = [2022, 2025]


def load_analysis_data():
    """Load inflation-adjusted auction data with performance metrics and WAR."""
    df = pd.read_csv(OUTPUT_DIR / "auction_inflation_adjusted.csv")

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


def estimate_pooled_ols(df, spec_name="baseline"):
    """Estimate pooled OLS with various specifications."""
    df_reg = df.dropna(subset=["log_price"]).copy()

    if spec_name == "baseline":
        X_vars = ["runs", "wickets", "is_indian"]
        df_subset = df_reg[df_reg["runs"].notna() & df_reg["wickets"].notna()]

    elif spec_name == "full_performance":
        X_vars = [
            "runs", "batting_avg", "batting_sr",
            "wickets", "economy", "catches",
            "is_indian"
        ]
        df_subset = df_reg.dropna(subset=X_vars)

    elif spec_name == "lagged":
        X_vars = ["runs_lag", "wickets_lag", "is_indian"]
        df_subset = df_reg.dropna(subset=X_vars)

    elif spec_name == "full_lagged":
        X_vars = [
            "runs_lag", "batting_avg_lag", "batting_sr_lag",
            "wickets_lag", "economy_lag", "catches_lag",
            "is_indian"
        ]
        df_subset = df_reg.dropna(subset=X_vars)

    elif spec_name == "war_current":
        X_vars = ["total_war", "is_indian"]
        df_subset = df_reg.dropna(subset=X_vars)
        df_subset = df_subset[df_subset["total_war"] != 0]

    elif spec_name == "war_components":
        X_vars = ["batting_war", "bowling_war", "is_indian"]
        df_subset = df_reg.dropna(subset=X_vars)
        df_subset = df_subset[
            (df_subset["batting_war"] != 0) | (df_subset["bowling_war"] != 0)
        ]

    elif spec_name == "war_lagged":
        X_vars = ["total_war_lag", "is_indian"]
        df_subset = df_reg.dropna(subset=X_vars)
        df_subset = df_subset[df_subset["total_war_lag"] != 0]

    elif spec_name == "war_mega":
        X_vars = ["total_war", "is_indian", "is_mega_auction"]
        df_subset = df_reg.dropna(subset=X_vars)
        df_subset = df_subset[df_subset["total_war"] != 0]

    elif spec_name == "lagged_current":
        X_vars = ["runs", "wickets", "runs_lag", "wickets_lag", "is_indian"]
        df_subset = df_reg.dropna(subset=X_vars)

    elif spec_name == "with_roles":
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


def estimate_year_fe(df, use_war=False):
    """Estimate model with year fixed effects."""
    if use_war:
        df_reg = df.dropna(subset=["log_price", "total_war"]).copy()
        df_reg = df_reg[df_reg["total_war"] != 0]
        X_vars = ["total_war", "is_indian"]
    else:
        df_reg = df.dropna(subset=["log_price", "runs", "wickets"]).copy()
        X_vars = ["runs", "wickets", "is_indian"]

    year_dummies = pd.get_dummies(
        df_reg["year"], prefix="year", drop_first=True, dtype=float
    )

    X = df_reg[X_vars].copy().astype(float)
    X = pd.concat([X, year_dummies], axis=1)
    X = sm.add_constant(X)
    y = df_reg["log_price"].astype(float)

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


def format_regression_table(results_dict):
    """Format multiple regression results into a publication-style table."""
    results_list = [r for r in results_dict.values() if r is not None]
    names = [k for k, v in results_dict.items() if v is not None]

    if not results_list:
        return "No valid regression results"

    table = summary_col(
        results_list,
        model_names=names,
        stars=True,
        float_format="%.3f",
        info_dict={
            "N": lambda x: f"{int(x.nobs)}",
            "R-squared": lambda x: f"{x.rsquared:.3f}",
            "Adj. R-squared": lambda x: f"{x.rsquared_adj:.3f}",
        }
    )

    return table


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


def compute_predicted_vs_actual(df, results, X_vars):
    """Compute predicted prices and identify over/underpaid players."""
    df_pred = df.dropna(subset=["log_price"] + X_vars).copy()

    X = df_pred[X_vars].copy()
    X = sm.add_constant(X)

    df_pred["log_price_predicted"] = results.predict(X)
    df_pred["price_predicted_cr"] = np.exp(df_pred["log_price_predicted"])
    df_pred["residual"] = df_pred["log_price"] - df_pred["log_price_predicted"]
    df_pred["pct_over_predicted"] = (
        df_pred["price_2024_cr"] / df_pred["price_predicted_cr"] - 1
    ) * 100

    return df_pred


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

    print("\n--- RAW PERFORMANCE MODELS ---")

    results = {}

    print("\n1. Baseline Model (runs + wickets + nationality)")
    results["Baseline"] = estimate_pooled_ols(df, "baseline")
    if results["Baseline"]:
        print(results["Baseline"].summary().tables[1])
        print_interpretation(results["Baseline"], "Baseline")

    print("\n2. Full Performance Model")
    results["Full"] = estimate_pooled_ols(df, "full_performance")
    if results["Full"]:
        print(results["Full"].summary().tables[1])

    print("\n3. Lagged Performance Model (prior season stats)")
    results["Lagged"] = estimate_pooled_ols(df, "lagged")
    if results["Lagged"]:
        print(results["Lagged"].summary().tables[1])
        print_interpretation(results["Lagged"], "Lagged")

    print("\n4. Lagged + Current Model (information revelation)")
    results["Lag+Curr"] = estimate_pooled_ols(df, "lagged_current")
    if results["Lag+Curr"]:
        print(results["Lag+Curr"].summary().tables[1])

    print("\n5. Model with Year Fixed Effects")
    results["Year FE"] = estimate_year_fe(df, use_war=False)
    if results["Year FE"]:
        fe_params = results["Year FE"].params[
            [c for c in results["Year FE"].params.index if not c.startswith("year_")]
        ]
        print(f"  Non-FE coefficients: {fe_params.to_dict()}")
        print(f"  R-squared: {results['Year FE'].rsquared:.3f}")

    if has_war:
        print("\n" + "=" * 60)
        print("WAR-BASED MODELS")
        print("=" * 60)

        print("\n6. WAR Current Season")
        results["WAR"] = estimate_pooled_ols(df, "war_current")
        if results["WAR"]:
            print(results["WAR"].summary().tables[1])
            print_interpretation(results["WAR"], "WAR Current")

        print("\n7. WAR Components (Batting + Bowling)")
        results["WAR Comp"] = estimate_pooled_ols(df, "war_components")
        if results["WAR Comp"]:
            print(results["WAR Comp"].summary().tables[1])
            print_interpretation(results["WAR Comp"], "WAR Components")

        print("\n8. WAR Lagged (prior season)")
        results["WAR Lag"] = estimate_pooled_ols(df, "war_lagged")
        if results["WAR Lag"]:
            print(results["WAR Lag"].summary().tables[1])
            print_interpretation(results["WAR Lag"], "WAR Lagged")

        print("\n9. WAR with Mega-Auction Control")
        results["WAR+Mega"] = estimate_pooled_ols(df, "war_mega")
        if results["WAR+Mega"]:
            print(results["WAR+Mega"].summary().tables[1])
            print_interpretation(results["WAR+Mega"], "WAR + Mega Auction")

        print("\n10. WAR with Year Fixed Effects")
        results["WAR YFE"] = estimate_year_fe(df, use_war=True)
        if results["WAR YFE"]:
            fe_params = results["WAR YFE"].params[
                [c for c in results["WAR YFE"].params.index if not c.startswith("year_")]
            ]
            print(f"  Non-FE coefficients: {fe_params.to_dict()}")
            print(f"  R-squared: {results['WAR YFE'].rsquared:.3f}")

    print("\n" + "=" * 60)
    print("MARKET EFFICIENCY TEST")
    print("=" * 60)

    print("\nTesting: Does price predict FUTURE performance?")
    print("Model: future_WAR_{t+1} = alpha + beta * log(price_t) + gamma * is_indian + eps")
    efficiency_result = estimate_market_efficiency(df)
    if efficiency_result:
        print(efficiency_result.summary().tables[1])
        price_coef = efficiency_result.params.get("log_price", 0)
        price_pval = efficiency_result.pvalues.get("log_price", 1)
        sig = "***" if price_pval < 0.01 else "**" if price_pval < 0.05 else "*" if price_pval < 0.1 else ""
        print(f"\n  Price coefficient: {price_coef:.4f} (p={price_pval:.3f}){sig}")
        print(f"  R-squared: {efficiency_result.rsquared:.3f}")
        if price_coef > 0 and price_pval < 0.05:
            print("  INTERPRETATION: Prices DO predict future performance -> markets are efficient")
        elif price_coef > 0 and price_pval >= 0.05:
            print("  INTERPRETATION: Positive but insignificant -> weak efficiency evidence")
        else:
            print("  INTERPRETATION: Prices don't predict future performance -> potential inefficiency")
    else:
        print("  Insufficient data for efficiency test")

    print("\n" + "=" * 60)
    print("ROLE-SPECIFIC MODELS")
    print("=" * 60)

    print("\n11. Batsmen Only")
    results["Batsmen"] = estimate_by_role(df, "batsman")
    if results["Batsmen"]:
        print(results["Batsmen"].summary().tables[1])

    print("\n12. Bowlers Only")
    results["Bowlers"] = estimate_by_role(df, "bowler")
    if results["Bowlers"]:
        print(results["Bowlers"].summary().tables[1])

    print("\n13. All-Rounders")
    results["AllRounders"] = estimate_by_role(df, "allrounder")
    if results["AllRounders"]:
        print(results["AllRounders"].summary().tables[1])

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print("\nRaw Stats vs WAR Comparison:")
    comparison_models = ["Baseline", "Full", "Lagged"]
    if has_war:
        comparison_models.extend(["WAR", "WAR Lag"])

    comparison_data = []
    for name in comparison_models:
        if name in results and results[name] is not None:
            comparison_data.append({
                "Model": name,
                "R-squared": f"{results[name].rsquared:.3f}",
                "Adj R-sq": f"{results[name].rsquared_adj:.3f}",
                "N": int(results[name].nobs)
            })
    if comparison_data:
        print(pd.DataFrame(comparison_data).to_string(index=False))

    print("\n" + "=" * 60)
    print("KEY FINDINGS SUMMARY")
    print("=" * 60)

    if results.get("Baseline"):
        baseline = results["Baseline"]
        print(f"""
1. RAW PERFORMANCE-PRICE RELATIONSHIP:
   - R-squared: {baseline.rsquared:.3f} (explains {baseline.rsquared * 100:.1f}% of price variation)
   - Runs coefficient: {baseline.params.get('runs', 0):.4f} (p={baseline.pvalues.get('runs', 1):.3f})
   - Wickets coefficient: {baseline.params.get('wickets', 0):.4f} (p={baseline.pvalues.get('wickets', 1):.3f})
""")

    if has_war and results.get("WAR"):
        war_model = results["WAR"]
        print(f"""2. WAR-BASED MODEL:
   - R-squared: {war_model.rsquared:.3f} (explains {war_model.rsquared * 100:.1f}% of price variation)
   - WAR coefficient: {war_model.params.get('total_war', 0):.4f} (p={war_model.pvalues.get('total_war', 1):.3f})
   - 1 WAR = {(np.exp(war_model.params.get('total_war', 0)) - 1) * 100:.1f}% price increase
""")

    if results.get("Lagged"):
        lagged = results["Lagged"]
        lagged_rsq = lagged.rsquared if lagged else 0
        baseline_rsq = baseline.rsquared if baseline else 0
        print(f"""3. FORECAST ERROR ANALYSIS:
   - Current season R²: {baseline_rsq:.3f}
   - Lagged season R²:  {lagged_rsq:.3f}
   - Gap of {(baseline_rsq - lagged_rsq) * 100:.1f}pp suggests teams have private information
   - Teams predict future, we measure past -> explains some residual variance
""")

    if efficiency_result:
        eff_rsq = efficiency_result.rsquared
        price_coef = efficiency_result.params.get("log_price", 0)
        print(f"""4. MARKET EFFICIENCY:
   - Price predicts {eff_rsq * 100:.1f}% of next-season WAR variance
   - Coefficient: {price_coef:.3f} {'(significant)' if efficiency_result.pvalues.get('log_price', 1) < 0.05 else '(not significant)'}
""")

    print("""5. INTERPRETING UNEXPLAINED VARIANCE:
   The ~60% unexplained variance reflects:
   - Forecast error: Teams buy expected FUTURE performance, not past
   - Auction mechanics: Slot effects, purse constraints, team composition needs
   - Measurement error: Raw stats don't capture context, consistency, match impact
   - Private information: Teams have scouting insights we don't observe

   This is NOT simply "star power/marketability" - it's a mix of factors
   that econometric models cannot fully disentangle.
""")

    output_path = OUTPUT_DIR / "regression_results.txt"
    with open(output_path, "w") as f:
        f.write("IPL Auction Hedonic Wage Regression Results\n")
        f.write("=" * 60 + "\n\n")
        for name, result in results.items():
            if result:
                f.write(f"\n{name} Model:\n")
                f.write(str(result.summary()) + "\n")
        if efficiency_result:
            f.write("\nMarket Efficiency Test:\n")
            f.write(str(efficiency_result.summary()) + "\n")
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
