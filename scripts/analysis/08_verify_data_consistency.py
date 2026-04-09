#!/usr/bin/env python3
"""
Verify data consistency across multiple IPL auction data sources.

Performs:
1. Cross-source verification for 2022 (Sold vs FullList files)
2. Year-by-year data completeness checks
3. Price consistency validation
4. Generate verification report
"""

import re

import numpy as np
import pandas as pd
from pathlib import Path
from rapidfuzz import fuzz
from rapidfuzz.process import cdist

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
SOURCES_DIR = DATA_DIR / "sources"
AUCTION_DIR = DATA_DIR / "auction"
DIAGNOSTICS_DIR = DATA_DIR / "analysis" / "diagnostics"


def parse_indian_price(price_str):
    """Parse Indian price format to lakhs."""
    if pd.isna(price_str):
        return np.nan
    price_str = str(price_str).strip()
    price_str = price_str.replace("₹", "").replace(",", "").replace(" ", "")
    try:
        value = float(price_str)
        if value > 1_000_000:
            return value / 100_000
        elif value > 100:
            return value
        else:
            return value * 100
    except ValueError:
        return np.nan


def verify_2022_sources():
    """Compare 2022 auction data from multiple sources."""
    print("=" * 60)
    print("2022 AUCTION DATA CROSS-SOURCE VERIFICATION")
    print("=" * 60)

    sold_path = SOURCES_DIR / "2022_auction/IPL_2022_Sold_Players.csv"
    full_path = SOURCES_DIR / "2022_auction/IPL_Auction_2022_FullList.csv"

    issues = []

    if not sold_path.exists():
        print(f"ERROR: {sold_path} not found")
        return ["2022 Sold Players file not found"]
    if not full_path.exists():
        print(f"ERROR: {full_path} not found")
        return ["2022 Full List file not found"]

    sold_df = pd.read_csv(sold_path)
    full_df = pd.read_csv(full_path)

    print(f"\nSold Players file: {len(sold_df)} records")
    print(f"Full List file: {len(full_df)} records")

    full_sold = full_df[full_df["Bid"] == "Sold"].copy()
    print(f"Sold in Full List: {len(full_sold)} records")

    sold_names = set(sold_df["Players"].str.strip().str.lower())
    full_names = set(full_sold["Players"].str.strip().str.lower())

    only_in_sold = sold_names - full_names
    only_in_full = full_names - sold_names

    print(f"\nPlayers only in Sold file: {len(only_in_sold)}")
    if only_in_sold:
        for name in sorted(only_in_sold)[:10]:
            print(f"  - {name}")
        if len(only_in_sold) > 10:
            print(f"  ... and {len(only_in_sold) - 10} more")
        issues.append(f"2022: {len(only_in_sold)} players only in Sold file")

    print(f"\nPlayers only in Full List (marked Sold): {len(only_in_full)}")
    if only_in_full:
        for name in sorted(only_in_full)[:10]:
            print(f"  - {name}")
        if len(only_in_full) > 10:
            print(f"  ... and {len(only_in_full) - 10} more")
        issues.append(f"2022: {len(only_in_full)} players only in Full List")

    print("\n--- Price Comparison ---")
    sold_df["name_clean"] = sold_df["Players"].str.strip().str.lower()
    sold_df["price_lakh"] = sold_df["Price Paid"].apply(parse_indian_price)

    full_sold["name_clean"] = full_sold["Players"].str.strip().str.lower()
    full_sold["price_lakh"] = full_sold["Price Paid"] / 100000

    merged = sold_df.merge(
        full_sold[["name_clean", "price_lakh"]],
        on="name_clean",
        suffixes=("_sold", "_full")
    )

    merged["price_diff"] = abs(merged["price_lakh_sold"] - merged["price_lakh_full"])
    price_mismatches = merged[merged["price_diff"] > 0.1]

    print(f"Price mismatches found: {len(price_mismatches)}")
    if len(price_mismatches) > 0:
        print("\nPrice discrepancies:")
        for _, row in price_mismatches.head(10).iterrows():
            print(f"  {row['Players']}: Sold={row['price_lakh_sold']:.0f}L vs Full={row['price_lakh_full']:.0f}L")
        issues.append(f"2022: {len(price_mismatches)} price mismatches between sources")

    return issues


def verify_year_completeness():
    """Check data completeness by year against known auction sizes."""
    print("\n" + "=" * 60)
    print("YEAR-BY-YEAR DATA COMPLETENESS CHECK")
    print("=" * 60)

    expected_minimums = {
        2008: 70,
        2009: 17,
        2010: 11,
        2011: 100,
        2012: 25,
        2013: 30,
        2014: 120,
        2015: 50,
        2016: 80,
        2017: 50,
        2018: 150,
        2019: 50,
        2020: 50,
        2021: 130,
        2022: 190,
        2023: 70,
        2024: 300,
        2025: 200,
        2026: 70,
    }

    auction_path = AUCTION_DIR / "auction_all_years.csv"
    if not auction_path.exists():
        print("ERROR: auction_all_years.csv not found. Run assemble_auction_data.py first.")
        return ["auction_all_years.csv not found"]

    df = pd.read_csv(auction_path)
    year_counts = df.groupby("year").size()

    issues = []
    print(f"\n{'Year':<6} {'Actual':<8} {'Expected Min':<12} {'Status':<10} {'Source':<20}")
    print("-" * 60)

    for year in sorted(df["year"].unique()):
        actual = year_counts.get(year, 0)
        expected = expected_minimums.get(year, 50)
        source = df[df["year"] == year]["source"].iloc[0] if actual > 0 else "N/A"

        if actual < expected * 0.5:
            status = "CRITICAL"
            issues.append(f"{year}: Only {actual} records (expected min {expected})")
        elif actual < expected:
            status = "LOW"
            issues.append(f"{year}: {actual} records below expected {expected}")
        else:
            status = "OK"

        print(f"{year:<6} {actual:<8} {expected:<12} {status:<10} {source:<20}")

    return issues


def verify_data_quality():
    """Check for data quality issues like malformed names."""
    print("\n" + "=" * 60)
    print("DATA QUALITY CHECKS")
    print("=" * 60)

    auction_path = AUCTION_DIR / "auction_all_years.csv"
    if not auction_path.exists():
        return ["auction_all_years.csv not found"]

    df = pd.read_csv(auction_path)
    issues = []

    country_prefixes = [
        "New Zealand ", "Trinidad and Tobago ", "Barbados ",
        "South Africa ", "India ", "England ", "Australia ",
        "Sri Lanka ", "West Indies ", "Bangladesh ", "Afghanistan ",
        "Zimbabwe ", "Pakistan ", "Ireland ", "Scotland ", "Netherlands "
    ]

    names_with_prefix = df[df["player_name"].apply(
        lambda x: any(str(x).startswith(prefix) for prefix in country_prefixes)
    )]
    if len(names_with_prefix) > 0:
        print(f"\nNames with country prefixes: {len(names_with_prefix)}")
        for _, row in names_with_prefix.iterrows():
            print(f"  {row['year']}: {row['player_name']}")
        issues.append(f"{len(names_with_prefix)} names have country prefixes")

    names_with_symbols = df[df["player_name"].str.contains(r"[†*]", regex=True, na=False)]
    if len(names_with_symbols) > 0:
        print(f"\nNames with special symbols (†, *): {len(names_with_symbols)}")
        for _, row in names_with_symbols.iterrows():
            print(f"  {row['year']}: {row['player_name']}")
        issues.append(f"{len(names_with_symbols)} names have special symbols")

    short_names = df[df["player_name"].str.len() < 3]
    if len(short_names) > 0:
        print(f"\nVery short names (< 3 chars): {len(short_names)}")
        for _, row in short_names.iterrows():
            print(f"  {row['year']}: {row['player_name']}")
        issues.append(f"{len(short_names)} very short names")

    unknown_teams = df[df["team"] == "Unknown"]
    if len(unknown_teams) > 0:
        year_unknown = unknown_teams.groupby("year").size()
        print(f"\nRecords with Unknown team: {len(unknown_teams)}")
        print(year_unknown.to_string())
        issues.append(f"{len(unknown_teams)} records have Unknown team")

    missing_prices = df[df["final_price_lakh"].isna()]
    if len(missing_prices) > 0:
        print(f"\nRecords with missing final price: {len(missing_prices)}")
        issues.append(f"{len(missing_prices)} records have missing final price")

    return issues


def detect_similar_names_within_year(threshold=85):
    """
    Detect potential duplicate players within the same year
    using fuzzy name matching.
    """
    print("\n" + "=" * 60)
    print("SIMILAR NAME DETECTION WITHIN SAME YEAR")
    print("=" * 60)

    auction_path = AUCTION_DIR / "auction_all_years.csv"
    if not auction_path.exists():
        return ["auction_all_years.csv not found"]

    df = pd.read_csv(auction_path)
    issues = []

    def normalize_for_match(name):
        if pd.isna(name):
            return ""
        name = str(name).strip().lower()
        name = re.sub(r"\bMoh[ao]mm?[ae]d\b", "mohammed", name, flags=re.IGNORECASE)
        name = re.sub(r"\bMohd\b", "mohammed", name, flags=re.IGNORECASE)
        name = name.replace(".", " ").replace("-", " ").replace("'", "")
        name = " ".join(name.split())
        return name

    def get_last_name(name):
        parts = name.split()
        return parts[-1] if parts else ""

    total_pairs = 0
    all_pairs = []

    for year in sorted(df["year"].unique()):
        year_df = df[df["year"] == year].copy()
        if len(year_df) < 2:
            continue

        names = year_df["player_name"].tolist()
        names_norm = [normalize_for_match(n) for n in names]

        scores = cdist(names_norm, names_norm, scorer=fuzz.token_sort_ratio, workers=-1)

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if scores[i][j] >= threshold:
                    last_i = get_last_name(names_norm[i])
                    last_j = get_last_name(names_norm[j])

                    is_likely_same = False
                    if last_i == last_j or fuzz.ratio(last_i, last_j) >= 90:
                        if scores[i][j] >= 88:
                            is_likely_same = True

                    if not is_likely_same:
                        continue

                    all_pairs.append({
                        "year": year,
                        "name1": names[i],
                        "name2": names[j],
                        "score": scores[i][j]
                    })
                    total_pairs += 1

    print(f"\nFound {total_pairs} similar name pairs (threshold={threshold}):")

    if total_pairs > 0:
        pairs_df = pd.DataFrame(all_pairs)
        pairs_df = pairs_df.sort_values(["year", "score"], ascending=[True, False])

        for _, row in pairs_df.iterrows():
            print(f"  {row['year']}: {row['name1']} <-> {row['name2']} ({row['score']:.0f}%)")

        if total_pairs > 0:
            issues.append(f"{total_pairs} similar name pairs detected within same year")
    else:
        print("  None found - deduplication appears complete!")

    return issues


def verify_player_registry():
    """Verify player registry integrity."""
    print("\n" + "=" * 60)
    print("PLAYER REGISTRY VERIFICATION")
    print("=" * 60)

    registry_path = AUCTION_DIR / "player_registry.csv"
    auction_path = AUCTION_DIR / "auction_all_years.csv"
    issues = []

    if not registry_path.exists():
        print("WARNING: player_registry.csv not found")
        return ["player_registry.csv not found"]

    if not auction_path.exists():
        return ["auction_all_years.csv not found"]

    registry = pd.read_csv(registry_path)
    auction = pd.read_csv(auction_path)

    print(f"\nRegistry: {len(registry)} unique players")
    print(f"Auction records: {len(auction)}")

    if "player_id" not in auction.columns:
        issues.append("player_id column missing from auction_all_years.csv")
        print("WARNING: player_id column missing")
    else:
        missing_id = auction[auction["player_id"].isna()]
        if len(missing_id) > 0:
            print(f"WARNING: {len(missing_id)} records missing player_id")
            issues.append(f"{len(missing_id)} records missing player_id")

    multi_alias = registry[registry["aliases"].str.contains(r"\|", na=False)]
    print(f"\nPlayers with multiple name variations: {len(multi_alias)}")

    if len(multi_alias) > 0:
        print("\nSample multi-alias players:")
        for _, row in multi_alias.head(10).iterrows():
            aliases = row["aliases"].split("|")
            print(f"  {row['player_id']}: {row['canonical_name']}")
            print(f"         Aliases: {', '.join(aliases)}")

    if "player_id" in auction.columns:
        player_auction_counts = auction.groupby("player_id").size()
        multi_auction = player_auction_counts[player_auction_counts > 5]
        print(f"\nPlayers auctioned >5 times: {len(multi_auction)}")

        for pid in multi_auction.head(5).index:
            records = auction[auction["player_id"] == pid]
            canonical = registry[registry["player_id"] == pid]["canonical_name"].iloc[0]
            years = sorted(records["year"].unique())
            print(f"  {pid}: {canonical} - {len(years)} auctions ({min(years)}-{max(years)})")

    return issues


def verify_top_deals():
    """Spot check top deals each year for sanity."""
    print("\n" + "=" * 60)
    print("TOP DEALS VERIFICATION (Sanity Check)")
    print("=" * 60)

    auction_path = AUCTION_DIR / "auction_all_years.csv"
    if not auction_path.exists():
        return []

    df = pd.read_csv(auction_path)
    df["final_price_lakh"] = pd.to_numeric(df["final_price_lakh"], errors="coerce")

    known_top_deals = {
        2008: [("MS Dhoni", 900, 1000)],
        2011: [("Gautam Gambhir", 1400, 1500)],
        2014: [("Yuvraj Singh", 1300, 1500)],
        2015: [("Yuvraj Singh", 1500, 1700)],
        2018: [("Ben Stokes", 1200, 1300)],
        2022: [("Ishan Kishan", 1500, 1600)],
        2024: [("Mitchell Starc", 2450, 2500)],
    }

    issues = []
    for year, expected_deals in known_top_deals.items():
        year_df = df[df["year"] == year]
        if year_df.empty:
            continue

        top_player = year_df.nlargest(1, "final_price_lakh").iloc[0]
        print(f"\n{year} Top Deal: {top_player['player_name']} @ ₹{top_player['final_price_lakh']:.0f}L")

        for name, min_price, max_price in expected_deals:
            player_row = year_df[year_df["player_name"].str.contains(name.split()[0], case=False, na=False)]
            if player_row.empty:
                print(f"  WARNING: {name} not found in {year} data")
                issues.append(f"{year}: Expected player {name} not found")
            else:
                price = player_row["final_price_lakh"].iloc[0]
                if not (min_price <= price <= max_price):
                    print(f"  WARNING: {name} price {price:.0f}L outside expected range [{min_price}, {max_price}]")
                    issues.append(f"{year}: {name} price mismatch")
                else:
                    print(f"  OK: {name} @ ₹{price:.0f}L (expected {min_price}-{max_price}L)")

    return issues


def generate_report(all_issues):
    """Generate markdown verification report."""
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = DIAGNOSTICS_DIR / "verification_report.md"

    with open(report_path, "w") as f:
        f.write("# IPL Auction Data Verification Report\n\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if not all_issues:
            f.write("## Status: PASSED\n\n")
            f.write("No critical issues found.\n")
        else:
            f.write("## Status: ISSUES FOUND\n\n")
            f.write("### Issues Summary\n\n")
            for issue in all_issues:
                f.write(f"- {issue}\n")

        auction_path = AUCTION_DIR / "auction_all_years.csv"
        if auction_path.exists():
            df = pd.read_csv(auction_path)
            f.write("\n### Data Summary\n\n")
            f.write(f"- Total records: {len(df)}\n")
            f.write(f"- Years covered: {df['year'].min()} - {df['year'].max()}\n")
            f.write(f"- Unique players: {df['player_name'].nunique()}\n")

            f.write("\n### Records by Year\n\n")
            f.write("| Year | Records | Source |\n")
            f.write("|------|---------|--------|\n")
            for year in sorted(df["year"].unique()):
                year_df = df[df["year"] == year]
                source = year_df["source"].iloc[0]
                f.write(f"| {year} | {len(year_df)} | {source} |\n")

    print(f"\nReport saved to: {report_path}")
    return report_path


def main():
    all_issues = []

    issues = verify_2022_sources()
    all_issues.extend(issues)

    issues = verify_year_completeness()
    all_issues.extend(issues)

    issues = verify_data_quality()
    all_issues.extend(issues)

    issues = verify_top_deals()
    all_issues.extend(issues)

    issues = detect_similar_names_within_year(threshold=85)
    all_issues.extend(issues)

    issues = verify_player_registry()
    all_issues.extend(issues)

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    if all_issues:
        print(f"\nFound {len(all_issues)} issues:")
        for issue in all_issues:
            print(f"  - {issue}")
    else:
        print("\nNo critical issues found!")

    report_path = generate_report(all_issues)

    return len(all_issues)


if __name__ == "__main__":
    exit_code = main()
    exit(0)
