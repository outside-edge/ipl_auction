#!/usr/bin/env python3
"""
Create a player name mapping between auction data and performance data.
Handles name variations like "Virat Kohli" vs "V Kohli".
"""

import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


def normalize_name(name):
    """Normalize player name for matching."""
    if pd.isna(name):
        return ""
    name = str(name).strip().lower()
    name = name.replace(".", " ").replace("-", " ")
    name = " ".join(name.split())
    return name


def get_initials_last(name):
    """Convert 'Virat Kohli' to 'v kohli' or 'MS Dhoni' to 'ms dhoni'."""
    parts = name.split()
    if len(parts) == 1:
        return name
    last = parts[-1]
    first_initials = " ".join([p[0] if len(p) > 0 else "" for p in parts[:-1]])
    return f"{first_initials} {last}"


def get_first_last(name):
    """Get first name and last name."""
    parts = name.split()
    if len(parts) == 1:
        return name, ""
    return parts[0], parts[-1]


def similarity(s1, s2):
    """Calculate string similarity."""
    return SequenceMatcher(None, s1, s2).ratio()


def create_name_mapping(auction_names, perf_names):
    """Create mapping between auction names and performance names."""
    auction_norm = {n: normalize_name(n) for n in auction_names}
    perf_norm = {n: normalize_name(n) for n in perf_names}

    mapping = {}
    unmatched = []

    for auc_name, auc_norm in auction_norm.items():
        if auc_norm in perf_norm.values():
            for perf_name, pn in perf_norm.items():
                if pn == auc_norm:
                    mapping[auc_name] = perf_name
                    break
            continue

        auc_init = get_initials_last(auc_norm)
        for perf_name, perf_n in perf_norm.items():
            if auc_init == perf_n or perf_n == auc_init:
                mapping[auc_name] = perf_name
                break
        else:
            auc_first, auc_last = get_first_last(auc_norm)
            best_match = None
            best_score = 0

            for perf_name, perf_n in perf_norm.items():
                perf_first, perf_last = get_first_last(perf_n)

                if auc_last == perf_last:
                    if auc_first and perf_first and auc_first[0] == perf_first[0]:
                        if best_score < 0.9:
                            best_match = perf_name
                            best_score = 0.9
                    elif auc_first == perf_first:
                        if best_score < 0.95:
                            best_match = perf_name
                            best_score = 0.95

                score = similarity(auc_norm, perf_n)
                if score > best_score and score > 0.8:
                    best_match = perf_name
                    best_score = score

            if best_match:
                mapping[auc_name] = best_match
            else:
                unmatched.append(auc_name)

    return mapping, unmatched


def build_master_mapping(auction, perf):
    """Build mapping from auction names to performance names by year."""
    all_mappings = []

    for year in auction["year"].unique():
        auc_year = auction[auction["year"] == year]["player_name"].unique()
        perf_year = perf[perf["season"] == year]["player"].unique()

        mapping, unmatched = create_name_mapping(auc_year, perf_year)

        for auc_name, perf_name in mapping.items():
            all_mappings.append({
                "year": year,
                "auction_name": auc_name,
                "perf_name": perf_name
            })

    mapping_df = pd.DataFrame(all_mappings)
    return mapping_df


def build_war_mapping(auction, war):
    """Build mapping from auction names to WAR player names by year."""
    all_mappings = []

    for year in auction["year"].unique():
        auc_year = auction[auction["year"] == year]["player_name"].unique()
        war_year = war[war["season"] == year]["player"].unique()

        mapping, unmatched = create_name_mapping(auc_year, war_year)

        for auc_name, war_name in mapping.items():
            all_mappings.append({
                "year": year,
                "auction_name": auc_name,
                "war_name": war_name
            })

    mapping_df = pd.DataFrame(all_mappings)
    return mapping_df


def merge_auction_performance():
    """Merge auction and performance data using name mapping."""
    print("Loading data...")
    auction = pd.read_csv(DATA_DIR / "auction_all_years.csv")
    perf = pd.read_csv(DATA_DIR / "player_season_stats.csv")

    war_path = DATA_DIR / "player_season_war.csv"
    if war_path.exists():
        war = pd.read_csv(war_path)
        print(f"  Loaded WAR data: {len(war)} player-seasons")
    else:
        war = None
        print("  WAR data not found, skipping WAR merge")

    print("Building name mappings...")
    mapping_df = build_master_mapping(auction, perf)
    print(f"  Created {len(mapping_df)} performance name mappings")

    auction_mapped = auction.merge(
        mapping_df,
        left_on=["year", "player_name"],
        right_on=["year", "auction_name"],
        how="left"
    )

    auction_mapped["perf_name"] = auction_mapped["perf_name"].fillna(auction_mapped["player_name"])

    merged = auction_mapped.merge(
        perf,
        left_on=["year", "perf_name"],
        right_on=["season", "player"],
        how="left"
    )

    drop_cols = ["auction_name", "perf_name", "season", "player"]
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

    if war is not None:
        print("Building WAR name mappings...")
        war_mapping_df = build_war_mapping(auction, war)
        print(f"  Created {len(war_mapping_df)} WAR name mappings")

        merged = merged.merge(
            war_mapping_df,
            left_on=["year", "player_name"],
            right_on=["year", "auction_name"],
            how="left"
        )
        merged["war_name"] = merged["war_name"].fillna(merged["player_name"])

        war_cols = ["batting_war", "bowling_war", "total_war"]
        merged = merged.merge(
            war[["season", "player"] + war_cols],
            left_on=["year", "war_name"],
            right_on=["season", "player"],
            how="left",
            suffixes=("", "_war")
        )

        drop_cols = ["auction_name", "war_name", "season_war", "player_war"]
        merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

        war_match_rate = merged["total_war"].notna().sum() / len(merged) * 100
        print(f"\nWAR match rate: {war_match_rate:.1f}%")
        print(f"WAR matched: {merged['total_war'].notna().sum()} / {len(merged)}")

    match_rate = merged["runs"].notna().sum() / len(merged) * 100
    print(f"\nPerformance match rate: {match_rate:.1f}%")
    print(f"Matched: {merged['runs'].notna().sum()} / {len(merged)}")

    output_path = DATA_DIR / "auction_with_performance.csv"
    merged.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    return merged


def main():
    merged = merge_auction_performance()

    merged["final_price_lakh"] = pd.to_numeric(merged["final_price_lakh"], errors="coerce")

    print("\n=== Sample Merged Data ===")
    cols = ["year", "player_name", "team_x", "final_price_lakh", "runs", "wickets", "batting_avg", "economy"]
    if "total_war" in merged.columns:
        cols.append("total_war")
    sample = merged[merged["runs"].notna()][cols].head(20)
    print(sample.to_string(index=False))

    print("\n=== High-Value Players with Performance ===")
    cols = ["year", "player_name", "final_price_lakh", "runs", "wickets", "batting_avg"]
    if "total_war" in merged.columns:
        cols.append("total_war")
    high_value = merged[
        (merged["final_price_lakh"] > 1000) & (merged["runs"].notna())
    ][cols].head(15)
    print(high_value.to_string(index=False))

    print("\n=== Still Unmatched (sample) ===")
    unmatched = merged[merged["runs"].isna()][
        ["year", "player_name", "team_x", "final_price_lakh"]
    ].head(10)
    print(unmatched.to_string(index=False))


if __name__ == "__main__":
    main()
