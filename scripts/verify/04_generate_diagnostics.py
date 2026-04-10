#!/usr/bin/env python3
"""
Generate diagnostics for unmatched players between auction and performance data.

Outputs:
- data/analysis/diagnostics/unmatched_auction_by_year.csv
- data/analysis/diagnostics/unmatched_perf_by_year.csv
- data/analysis/diagnostics/match_rates_summary.txt
"""

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from shared.names import normalize_name
from shared.io import load_dataset

DATA_DIR = BASE_DIR / "data"
AUCTION_DIR = DATA_DIR / "auction"
PERF_DIR = DATA_DIR / "perf" / "ipl"
JOINED_DIR = DATA_DIR / "analysis" / "joined"
DIAGNOSTICS_DIR = DATA_DIR / "analysis" / "diagnostics"


def load_data():
    """Load auction and performance data."""
    print("Loading data...")

    auction = load_dataset(AUCTION_DIR / "auction_all_years")
    print(f"  Auction records: {len(auction)}")

    perf = load_dataset(PERF_DIR / "player_season_stats")
    print(f"  Performance records: {len(perf)}")

    try:
        merged = load_dataset(JOINED_DIR / "auction_with_performance")
        print(f"  Merged records: {len(merged)}")
    except FileNotFoundError:
        merged = None
        print("  Merged data not found")

    return auction, perf, merged


def compute_unmatched_auction_by_year(auction, merged):
    """Find auction players with no performance match, grouped by year."""
    if merged is None:
        return pd.DataFrame(), pd.DataFrame()

    unmatched = merged[merged["runs"].isna()].copy()

    by_year = unmatched.groupby("year").agg(
        count=("player_name", "count"),
        players=("player_name", lambda x: ", ".join(sorted(x.head(10)))),
        total_value_lakh=("final_price_lakh", "sum"),
    ).reset_index()

    by_year["total_value_cr"] = by_year["total_value_lakh"] / 100

    return by_year, unmatched


def compute_unmatched_perf_by_year(auction, perf, merged):
    """Find performance players with no auction match, grouped by year."""
    if merged is None:
        return pd.DataFrame(), pd.DataFrame()

    auction["player_norm"] = auction["player_name"].apply(normalize_name)
    perf["player_norm"] = perf["player"].apply(normalize_name)

    auction_by_year = {}
    for year in auction["year"].unique():
        auction_by_year[year] = set(auction[auction["year"] == year]["player_norm"])

    unmatched_perf = []
    for _, row in perf.iterrows():
        year = row["season"]
        player_norm = row["player_norm"]

        if year in auction_by_year and player_norm not in auction_by_year[year]:
            unmatched_perf.append({
                "season": year,
                "player": row["player"],
                "runs": row.get("runs", 0),
                "wickets": row.get("wickets", 0),
            })

    unmatched_df = pd.DataFrame(unmatched_perf)

    if unmatched_df.empty:
        return pd.DataFrame(), unmatched_df

    by_year = unmatched_df.groupby("season").agg(
        count=("player", "count"),
        players=("player", lambda x: ", ".join(sorted(x.head(10)))),
    ).reset_index()

    return by_year, unmatched_df


def compute_match_rates(auction, merged):
    """Compute match rates by year and acquisition type."""
    if merged is None:
        return {}

    rates = {}

    year_stats = merged.groupby("year").apply(
        lambda x: pd.Series({
            "total": len(x),
            "matched": x["runs"].notna().sum(),
            "rate": x["runs"].notna().sum() / len(x) * 100 if len(x) > 0 else 0
        }),
        include_groups=False
    ).reset_index()
    rates["by_year"] = year_stats

    if "acquisition_type" in merged.columns:
        acq_stats = merged.groupby("acquisition_type").apply(
            lambda x: pd.Series({
                "total": len(x),
                "matched": x["runs"].notna().sum(),
                "rate": x["runs"].notna().sum() / len(x) * 100 if len(x) > 0 else 0
            }),
            include_groups=False
        ).reset_index()
        rates["by_acquisition"] = acq_stats

    overall = {
        "total": len(merged),
        "matched": merged["runs"].notna().sum(),
        "rate": merged["runs"].notna().sum() / len(merged) * 100,
    }
    rates["overall"] = overall

    return rates


def write_summary(rates, auction_unmatched_by_year, perf_unmatched_by_year, output_path):
    """Write summary report."""
    with open(output_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("IPL AUCTION-PERFORMANCE MATCH DIAGNOSTICS\n")
        f.write("=" * 60 + "\n\n")

        if "overall" in rates:
            f.write("OVERALL MATCH RATE\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total auction records: {rates['overall']['total']}\n")
            f.write(f"Matched to performance: {rates['overall']['matched']}\n")
            f.write(f"Match rate: {rates['overall']['rate']:.1f}%\n\n")

        if "by_year" in rates:
            f.write("MATCH RATE BY YEAR\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Year':<6} {'Total':<8} {'Matched':<10} {'Rate':<10}\n")
            for _, row in rates["by_year"].iterrows():
                f.write(f"{int(row['year']):<6} {int(row['total']):<8} {int(row['matched']):<10} {row['rate']:.1f}%\n")
            f.write("\n")

        if "by_acquisition" in rates:
            f.write("MATCH RATE BY ACQUISITION TYPE\n")
            f.write("-" * 40 + "\n")
            for _, row in rates["by_acquisition"].iterrows():
                f.write(f"{row['acquisition_type']:<15} {int(row['total']):<8} {int(row['matched']):<10} {row['rate']:.1f}%\n")
            f.write("\n")

        f.write("UNMATCHED AUCTION PLAYERS BY YEAR\n")
        f.write("-" * 40 + "\n")
        if not auction_unmatched_by_year.empty:
            for _, row in auction_unmatched_by_year.iterrows():
                f.write(f"{int(row['year'])}: {int(row['count'])} players, {row['total_value_cr']:.1f} Cr\n")
        else:
            f.write("No unmatched auction players.\n")
        f.write("\n")

        f.write("UNMATCHED PERFORMANCE PLAYERS BY YEAR\n")
        f.write("-" * 40 + "\n")
        if not perf_unmatched_by_year.empty:
            for _, row in perf_unmatched_by_year.iterrows():
                f.write(f"{int(row['season'])}: {int(row['count'])} players\n")
        else:
            f.write("No unmatched performance players.\n")


def main():
    print("=" * 60)
    print("Generating Match Diagnostics")
    print("=" * 60)

    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    auction, perf, merged = load_data()

    print("\nComputing unmatched auction players...")
    auction_by_year, auction_unmatched = compute_unmatched_auction_by_year(auction, merged)
    if not auction_unmatched.empty:
        auction_unmatched_path = DIAGNOSTICS_DIR / "unmatched_auction_by_year.csv"
        team_col = "team_auction" if "team_auction" in auction_unmatched.columns else "team"
        cols = ["year", "player_name", team_col, "final_price_lakh"]
        cols = [c for c in cols if c in auction_unmatched.columns]
        auction_unmatched[cols].to_csv(auction_unmatched_path, index=False)
        print(f"  Saved to {auction_unmatched_path}")

    print("Computing unmatched performance players...")
    perf_by_year, perf_unmatched = compute_unmatched_perf_by_year(auction, perf, merged)
    if not perf_unmatched.empty:
        perf_unmatched_path = DIAGNOSTICS_DIR / "unmatched_perf_by_year.csv"
        perf_unmatched.to_csv(perf_unmatched_path, index=False)
        print(f"  Saved to {perf_unmatched_path}")

    print("Computing match rates...")
    rates = compute_match_rates(auction, merged)

    print("Writing summary report...")
    summary_path = DIAGNOSTICS_DIR / "match_rates_summary.txt"
    write_summary(rates, auction_by_year, perf_by_year, summary_path)
    print(f"  Saved to {summary_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if "overall" in rates:
        print(f"Overall match rate: {rates['overall']['rate']:.1f}%")
        print(f"Unmatched auction players: {len(auction_unmatched)}")
        print(f"Unmatched perf players: {len(perf_unmatched)}")


if __name__ == "__main__":
    main()
