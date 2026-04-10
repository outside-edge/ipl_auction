#!/usr/bin/env python3
"""
Economic analysis for newspaper story.

Generates compelling statistics and insights about IPL auction efficiency,
team performance, and player value.

Output:
    tabs/economic_summary.txt - Key statistics for reporting
    tabs/team_efficiency.csv - Team-level efficiency rankings
    tabs/lucky_unlucky.csv - Players who exceeded/missed expectations
"""

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

BASE_DIR = Path(__file__).parent.parent.parent
TABS_DIR = BASE_DIR / "tabs"


def load_predictions():
    """Load retroactive predictions."""
    df = pd.read_csv(TABS_DIR / "retroactive_predictions.csv")
    valid = df[
        df["actual_war"].notna() &
        df["predicted_war"].notna() &
        df["final_price_cr"].notna() &
        (df["final_price_cr"] > 0)
    ].copy()
    valid["prediction_error"] = valid["actual_war"] - valid["predicted_war"]
    return valid


def compute_fair_prices(df):
    """Compute fair prices based on predicted WAR."""
    total_pred_war = df["predicted_war"].clip(lower=0).sum()
    total_spent = df["final_price_cr"].sum()
    price_per_war = total_spent / total_pred_war

    df = df.copy()
    df["fair_price_cr"] = df["predicted_war"].clip(lower=0) * price_per_war
    df["overpay_cr"] = df["final_price_cr"] - df["fair_price_cr"]
    df["value_per_cr"] = df["prediction_error"] / (df["final_price_cr"] + 0.5)
    return df, price_per_war


def headline_stats(df, price_per_war):
    """Generate headline statistics."""
    stats = []

    stats.append("=" * 70)
    stats.append("IPL AUCTION ECONOMIC ANALYSIS")
    stats.append("=" * 70)

    stats.append(f"\nData: {len(df)} player-seasons from 2012-2025")
    stats.append(f"Total spending analyzed: ₹{df['final_price_cr'].sum():.0f} Cr")
    stats.append(f"Average price: ₹{df['final_price_cr'].mean():.2f} Cr")
    stats.append(f"Implied price per WAR: ₹{price_per_war:.2f} Cr")

    stats.append("\n" + "-" * 70)
    stats.append("THE WASTE: Money Spent on Underperformers")
    stats.append("-" * 70)

    underperf_5 = df[df["prediction_error"] < -5]
    underperf_10 = df[df["prediction_error"] < -10]
    stats.append(f"\nPlayers who underperformed by 5+ WAR: {len(underperf_5)}")
    stats.append(f"  Total spent: ₹{underperf_5['final_price_cr'].sum():.0f} Cr")
    stats.append(f"  Average price: ₹{underperf_5['final_price_cr'].mean():.2f} Cr")

    stats.append(f"\nPlayers who underperformed by 10+ WAR: {len(underperf_10)}")
    stats.append(f"  Total spent: ₹{underperf_10['final_price_cr'].sum():.0f} Cr")
    stats.append(f"  Examples: {', '.join(underperf_10.nsmallest(5, 'prediction_error')['player_name'].tolist())}")

    stats.append("\n" + "-" * 70)
    stats.append("THE FINDS: Breakout Performances")
    stats.append("-" * 70)

    overperf_10 = df[df["prediction_error"] > 10]
    overperf_15 = df[df["prediction_error"] > 15]
    stats.append(f"\nPlayers who exceeded expectations by 10+ WAR: {len(overperf_10)}")
    stats.append(f"  Total spent: ₹{overperf_10['final_price_cr'].sum():.0f} Cr")
    stats.append(f"  Average price: ₹{overperf_10['final_price_cr'].mean():.2f} Cr")

    stats.append(f"\nPlayers who exceeded expectations by 15+ WAR: {len(overperf_15)}")
    stats.append(f"  Examples: {', '.join(overperf_15.nlargest(5, 'prediction_error')['player_name'].tolist())}")

    stats.append("\n" + "-" * 70)
    stats.append("THE GAMBLE: Expensive Bets Are Riskier")
    stats.append("-" * 70)

    for tier, label in [((0, 2), "<2 Cr"), ((2, 5), "2-5 Cr"),
                        ((5, 10), "5-10 Cr"), ((10, 100), "10+ Cr")]:
        subset = df[(df["final_price_cr"] > tier[0]) & (df["final_price_cr"] <= tier[1])]
        if len(subset) > 0:
            stats.append(f"\n{label}: n={len(subset)}")
            stats.append(f"  Mean error: {subset['prediction_error'].mean():+.1f} WAR")
            stats.append(f"  Std error: {subset['prediction_error'].std():.1f} WAR")
            underperf_rate = (subset["prediction_error"] < -5).mean() * 100
            stats.append(f"  Underperformance rate (>5 WAR miss): {underperf_rate:.0f}%")

    stats.append("\n" + "-" * 70)
    stats.append("THE BIGGEST MISSES: Costliest Disappointments")
    stats.append("-" * 70)

    df["wasted_cr"] = df["final_price_cr"] * (df["prediction_error"] < 0).astype(int) * abs(df["prediction_error"]) / 10
    biggest_misses = df.nsmallest(10, "prediction_error")
    stats.append("\n")
    for _, row in biggest_misses.iterrows():
        stats.append(
            f"  {row['year']} {row['player_name']:25} | "
            f"Paid: ₹{row['final_price_cr']:.1f}Cr | "
            f"Expected: {row['predicted_war']:.1f} WAR | "
            f"Got: {row['actual_war']:.1f} WAR"
        )

    stats.append("\n" + "-" * 70)
    stats.append("THE STEALS: Best Value Finds")
    stats.append("-" * 70)

    best_value = df[df["prediction_error"] > 5].nlargest(10, "value_per_cr")
    stats.append("\n")
    for _, row in best_value.iterrows():
        stats.append(
            f"  {row['year']} {row['player_name']:25} | "
            f"Paid: ₹{row['final_price_cr']:.1f}Cr | "
            f"Expected: {row['predicted_war']:.1f} WAR | "
            f"Got: {row['actual_war']:.1f} WAR"
        )

    return "\n".join(stats)


def team_efficiency(df):
    """Compute team-level efficiency metrics."""
    team_stats = df.groupby("team").agg({
        "final_price_cr": ["sum", "mean", "count"],
        "overpay_cr": "sum",
        "prediction_error": ["mean", "std"],
        "actual_war": "sum",
        "predicted_war": "sum",
    })
    team_stats.columns = [
        "total_spent_cr", "avg_price_cr", "n_players",
        "total_overpay_cr", "mean_error", "std_error",
        "total_actual_war", "total_predicted_war"
    ]
    team_stats["war_per_cr"] = team_stats["total_actual_war"] / team_stats["total_spent_cr"]
    team_stats["efficiency_rank"] = team_stats["war_per_cr"].rank(ascending=False)

    team_stats = team_stats.sort_values("war_per_cr", ascending=False)
    return team_stats.reset_index()


def lucky_unlucky_players(df):
    """Identify lucky and unlucky players."""
    lucky = df.nlargest(20, "prediction_error")[
        ["year", "player_name", "team", "final_price_cr",
         "predicted_war", "actual_war", "prediction_error"]
    ].copy()
    lucky["category"] = "breakout"

    unlucky = df.nsmallest(20, "prediction_error")[
        ["year", "player_name", "team", "final_price_cr",
         "predicted_war", "actual_war", "prediction_error"]
    ].copy()
    unlucky["category"] = "disappointment"

    best_value = df[df["prediction_error"] > 3].nlargest(20, "value_per_cr")[
        ["year", "player_name", "team", "final_price_cr",
         "predicted_war", "actual_war", "prediction_error"]
    ].copy()
    best_value["category"] = "bargain"

    combined = pd.concat([lucky, unlucky, best_value], ignore_index=True)
    return combined


def mega_vs_mini_analysis(df):
    """Compare mega vs mini auction dynamics."""
    mega_years = [2014, 2018, 2022, 2025]
    df = df.copy()
    df["is_mega"] = df["year"].isin(mega_years)

    stats = []
    stats.append("\n" + "-" * 70)
    stats.append("MEGA vs MINI AUCTION DYNAMICS")
    stats.append("-" * 70)

    for is_mega, label in [(True, "Mega Auctions"), (False, "Mini Auctions")]:
        subset = df[df["is_mega"] == is_mega]
        if len(subset) > 0:
            stats.append(f"\n{label}: n={len(subset)}")
            stats.append(f"  Mean prediction error: {subset['prediction_error'].mean():+.2f} WAR")
            stats.append(f"  Std prediction error: {subset['prediction_error'].std():.2f} WAR")
            stats.append(f"  Total spent: ₹{subset['final_price_cr'].sum():.0f} Cr")

            rho, p = spearmanr(subset["final_price_cr"], subset["actual_war"])
            stats.append(f"  Price-performance correlation: ρ={rho:.2f} (p={p:.3f})")

    return "\n".join(stats)


def main():
    print("=" * 60)
    print("Economic Analysis for IPL Auction Study")
    print("=" * 60)

    df = load_predictions()
    print(f"\nLoaded {len(df)} valid predictions")

    df, price_per_war = compute_fair_prices(df)
    print(f"Implied price per WAR: ₹{price_per_war:.2f} Cr")

    headline = headline_stats(df, price_per_war)
    mega_mini = mega_vs_mini_analysis(df)

    full_report = headline + mega_mini

    TABS_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = TABS_DIR / "economic_summary.txt"
    with open(summary_path, "w") as f:
        f.write(full_report)
    print(f"\nSaved summary to {summary_path}")

    team_df = team_efficiency(df)
    team_path = TABS_DIR / "team_efficiency.csv"
    team_df.to_csv(team_path, index=False)
    print(f"Saved team efficiency to {team_path}")

    lucky_df = lucky_unlucky_players(df)
    lucky_path = TABS_DIR / "lucky_unlucky.csv"
    lucky_df.to_csv(lucky_path, index=False)
    print(f"Saved lucky/unlucky players to {lucky_path}")

    print("\n" + full_report)


if __name__ == "__main__":
    main()
