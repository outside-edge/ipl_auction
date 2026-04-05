#!/usr/bin/env python3
"""
Create a player name mapping between auction data and performance data.
Handles name variations like "Virat Kohli" vs "V Kohli".

Uses a multi-stage matching approach:
1. Exact match (normalized)
2. Initials-last name match ("Virat Kohli" -> "V Kohli")
3. Last-name-first initial match
4. Manual alias table lookup
5. Fuzzy matching with rapidfuzz
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rapidfuzz import fuzz, process

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


def normalize_name(name):
    """Normalize player name for matching."""
    if pd.isna(name):
        return ""
    name = str(name).strip().lower()
    name = name.replace(".", " ").replace("-", " ").replace("'", "")
    name = " ".join(name.split())
    return name


def get_initials_last(name):
    """Convert 'Virat Kohli' to 'v kohli' or 'MS Dhoni' to 'ms dhoni'."""
    parts = name.split()
    if len(parts) == 1:
        return name
    last = parts[-1]
    first_initials = "".join([p[0] if len(p) > 0 else "" for p in parts[:-1]])
    return f"{first_initials} {last}"


def get_last_name(name):
    """Get last name from a normalized name."""
    parts = name.split()
    if len(parts) == 0:
        return ""
    return parts[-1]


def get_first_initial(name):
    """Get first initial from a normalized name."""
    parts = name.split()
    if len(parts) == 0:
        return ""
    return parts[0][0] if parts[0] else ""


def convert_full_to_initial_format(name):
    """Convert 'virat kohli' to 'v kohli' format."""
    parts = name.split()
    if len(parts) < 2:
        return name
    first_initial = parts[0][0] if parts[0] else ""
    middle_initials = "".join([p[0] for p in parts[1:-1]])
    last = parts[-1]
    if middle_initials:
        return f"{first_initial}{middle_initials} {last}"
    return f"{first_initial} {last}"


def load_alias_table():
    """Load manual alias table from CSV if it exists."""
    alias_path = DATA_DIR / "name_aliases.csv"
    if not alias_path.exists():
        return {}

    df = pd.read_csv(alias_path)
    aliases = {}
    for _, row in df.iterrows():
        auction_name = normalize_name(row["auction_name"])
        perf_name = row["performance_name"]
        year = row.get("year", None)
        if pd.notna(year):
            aliases[(auction_name, int(year))] = perf_name
        else:
            aliases[(auction_name, None)] = perf_name
    return aliases


def create_name_mapping(auction_names, perf_names, alias_table=None, year=None):
    """
    Create mapping between auction names and performance names.

    Uses a multi-stage matching approach:
    1. Exact match (normalized)
    2. Check alias table
    3. Initials-last name match
    4. Last-name + first-initial match
    5. Fuzzy matching with rapidfuzz
    """
    if alias_table is None:
        alias_table = {}

    auction_norm = {n: normalize_name(n) for n in auction_names}
    perf_norm = {n: normalize_name(n) for n in perf_names}

    perf_norm_reverse = {v: k for k, v in perf_norm.items()}

    perf_initials = {n: get_initials_last(normalize_name(n)) for n in perf_names}
    perf_initials_reverse = {}
    for perf_name, init in perf_initials.items():
        if init not in perf_initials_reverse:
            perf_initials_reverse[init] = perf_name

    mapping = {}
    unmatched = []

    for auc_name, auc_norm in auction_norm.items():
        matched = False

        if auc_norm in perf_norm_reverse:
            mapping[auc_name] = perf_norm_reverse[auc_norm]
            continue

        alias_key_year = (auc_norm, year)
        alias_key_global = (auc_norm, None)
        if alias_key_year in alias_table:
            mapping[auc_name] = alias_table[alias_key_year]
            continue
        if alias_key_global in alias_table:
            mapping[auc_name] = alias_table[alias_key_global]
            continue

        auc_init = convert_full_to_initial_format(auc_norm)
        for perf_name, perf_n in perf_norm.items():
            if auc_init == perf_n:
                mapping[auc_name] = perf_name
                matched = True
                break
        if matched:
            continue

        if auc_init in perf_initials_reverse:
            mapping[auc_name] = perf_initials_reverse[auc_init]
            continue

        auc_last = get_last_name(auc_norm)
        auc_first_init = get_first_initial(auc_norm)
        if auc_last and auc_first_init:
            for perf_name, perf_n in perf_norm.items():
                perf_last = get_last_name(perf_n)
                perf_first_init = get_first_initial(perf_n)
                if auc_last == perf_last and auc_first_init == perf_first_init:
                    mapping[auc_name] = perf_name
                    matched = True
                    break
        if matched:
            continue

        perf_names_list = list(perf_names)
        if perf_names_list:
            result = process.extractOne(
                auc_norm,
                [normalize_name(n) for n in perf_names_list],
                scorer=fuzz.token_sort_ratio
            )
            if result and result[1] >= 85:
                best_idx = [normalize_name(n) for n in perf_names_list].index(result[0])
                mapping[auc_name] = perf_names_list[best_idx]
                continue

        unmatched.append(auc_name)

    return mapping, unmatched


def build_master_mapping(auction, perf, alias_table=None):
    """Build mapping from auction names to performance names by year."""
    if alias_table is None:
        alias_table = load_alias_table()

    all_mappings = []
    all_unmatched = []

    for year in sorted(auction["year"].unique()):
        auc_year = auction[auction["year"] == year]["player_name"].unique()
        perf_year = perf[perf["season"] == year]["player"].unique()

        mapping, unmatched = create_name_mapping(auc_year, perf_year, alias_table, year)

        for auc_name, perf_name in mapping.items():
            all_mappings.append({
                "year": year,
                "auction_name": auc_name,
                "perf_name": perf_name
            })

        for auc_name in unmatched:
            all_unmatched.append({
                "year": year,
                "auction_name": auc_name
            })

    mapping_df = pd.DataFrame(all_mappings)
    unmatched_df = pd.DataFrame(all_unmatched)

    return mapping_df, unmatched_df


def build_war_mapping(auction, war, alias_table=None):
    """Build mapping from auction names to WAR player names by year."""
    if alias_table is None:
        alias_table = load_alias_table()

    all_mappings = []

    for year in auction["year"].unique():
        auc_year = auction[auction["year"] == year]["player_name"].unique()
        war_year = war[war["season"] == year]["player"].unique()

        mapping, _ = create_name_mapping(auc_year, war_year, alias_table, year)

        for auc_name, war_name in mapping.items():
            all_mappings.append({
                "year": year,
                "auction_name": auc_name,
                "war_name": war_name
            })

    mapping_df = pd.DataFrame(all_mappings)
    return mapping_df


def generate_alias_suggestions(auction, perf):
    """
    Generate suggested aliases for unmatched players by finding close matches
    in performance data.
    """
    alias_table = load_alias_table()
    _, unmatched_df = build_master_mapping(auction, perf, alias_table)

    suggestions = []

    for _, row in unmatched_df.iterrows():
        year = row["year"]
        auc_name = row["auction_name"]
        auc_norm = normalize_name(auc_name)

        perf_year = perf[perf["season"] == year]["player"].unique()
        perf_names_list = list(perf_year)

        if not perf_names_list:
            continue

        results = process.extract(
            auc_norm,
            [normalize_name(n) for n in perf_names_list],
            scorer=fuzz.token_sort_ratio,
            limit=3
        )

        for result in results:
            if result[1] >= 50:
                idx = [normalize_name(n) for n in perf_names_list].index(result[0])
                suggestions.append({
                    "year": year,
                    "auction_name": auc_name,
                    "suggested_perf_name": perf_names_list[idx],
                    "similarity": result[1]
                })

    return pd.DataFrame(suggestions)


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

    alias_table = load_alias_table()
    if alias_table:
        print(f"  Loaded {len(alias_table)} manual aliases")

    print("Building name mappings...")
    mapping_df, unmatched_df = build_master_mapping(auction, perf, alias_table)
    print(f"  Created {len(mapping_df)} performance name mappings")
    print(f"  Unmatched: {len(unmatched_df)} player-years")

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
        war_mapping_df = build_war_mapping(auction, war, alias_table)
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

    if len(unmatched_df) > 0:
        unmatched_path = DATA_DIR / "analysis/unmatched_players.csv"
        unmatched_df.to_csv(unmatched_path, index=False)
        print(f"Saved unmatched list to {unmatched_path}")

    return merged


def main():
    merged = merge_auction_performance()

    merged["final_price_lakh"] = pd.to_numeric(merged["final_price_lakh"], errors="coerce")

    print("\n=== Match Rate by Year ===")
    year_stats = merged.groupby("year").apply(
        lambda x: pd.Series({
            "total": len(x),
            "matched": x["runs"].notna().sum(),
            "rate": x["runs"].notna().sum() / len(x) * 100
        })
    )
    print(year_stats.to_string())

    print("\n=== Sample Merged Data ===")
    cols = ["year", "player_name", "team_x", "final_price_lakh", "runs", "wickets", "batting_avg", "economy"]
    if "total_war" in merged.columns:
        cols.append("total_war")
    cols_available = [c for c in cols if c in merged.columns]
    sample = merged[merged["runs"].notna()][cols_available].head(20)
    print(sample.to_string(index=False))

    print("\n=== High-Value Players with Performance ===")
    cols = ["year", "player_name", "final_price_lakh", "runs", "wickets", "batting_avg"]
    if "total_war" in merged.columns:
        cols.append("total_war")
    cols_available = [c for c in cols if c in merged.columns]
    high_value = merged[
        (merged["final_price_lakh"] > 1000) & (merged["runs"].notna())
    ][cols_available].head(15)
    print(high_value.to_string(index=False))

    print("\n=== Still Unmatched (high value sample) ===")
    unmatched = merged[merged["runs"].isna()][
        ["year", "player_name", "team_x", "final_price_lakh"]
    ].sort_values("final_price_lakh", ascending=False).head(15)
    print(unmatched.to_string(index=False))


if __name__ == "__main__":
    main()
