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

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import process
from rapidfuzz.distance import JaroWinkler

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from shared.names import (
    normalize_name,
    get_initials_last,
    get_last_name,
    names_compatible,
    convert_full_to_initial_format,
)
from shared.inflation import adjust_prices_for_inflation
from shared.constants import SUFFIX_AUCTION, SUFFIX_IPL, COL_HAS_IPL_HISTORY
from shared.io import save_dataset, load_dataset

DATA_DIR = BASE_DIR / "data"
ACQUISITIONS_DIR = DATA_DIR / "acquisitions"
PERF_DIR = DATA_DIR / "perf" / "ipl"
ANALYSIS_DIR = DATA_DIR / "analysis"
JOINED_DIR = ANALYSIS_DIR / "joined"
DIAGNOSTICS_DIR = ANALYSIS_DIR / "diagnostics"


def load_alias_table():
    """Load manual alias table from CSV if it exists."""
    alias_path = ACQUISITIONS_DIR / "name_aliases.csv"
    if not alias_path.exists():
        return {}

    df = pd.read_csv(alias_path)
    aliases = {}
    for _, row in df.iterrows():
        auction_name = normalize_name(row["auction_name"])
        perf_name = normalize_name(row["performance_name"])
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

    mapping = {}
    unmatched = []

    for auc_name, auc_norm in auction_norm.items():
        matched = False

        if auc_norm in perf_norm_reverse:
            mapping[auc_name] = perf_norm_reverse[auc_norm]
            continue

        alias_key_year = (auc_norm, year)
        alias_key_global = (auc_norm, None)
        alias_perf_norm = None
        if alias_key_year in alias_table:
            alias_perf_norm = alias_table[alias_key_year]
        elif alias_key_global in alias_table:
            alias_perf_norm = alias_table[alias_key_global]
        if alias_perf_norm and alias_perf_norm in perf_norm_reverse:
            mapping[auc_name] = perf_norm_reverse[alias_perf_norm]
            continue

        auc_init = convert_full_to_initial_format(auc_norm)
        for perf_name, perf_n in perf_norm.items():
            if auc_init == perf_n:
                mapping[auc_name] = perf_name
                matched = True
                break
        if matched:
            continue

        candidates = [
            pn for pn, pnorm in perf_norm.items()
            if get_initials_last(pnorm) == auc_init
            and names_compatible(auc_norm, pnorm)
        ]
        if len(candidates) == 1:
            mapping[auc_name] = candidates[0]
            continue

        auc_last = get_last_name(auc_norm)
        perf_names_list = list(perf_names)
        if perf_names_list:
            result = process.extractOne(
                auc_norm,
                [normalize_name(n) for n in perf_names_list],
                scorer=JaroWinkler.normalized_similarity
            )
            if result:
                best_idx = [normalize_name(n) for n in perf_names_list].index(result[0])
                perf_match = perf_names_list[best_idx]
                perf_last = get_last_name(normalize_name(perf_match))
                same_last = auc_last == perf_last
                if result[1] >= 0.95 or (result[1] >= 0.85 and same_last):
                    mapping[auc_name] = perf_match
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
            scorer=JaroWinkler.normalized_similarity,
            limit=3
        )

        for result in results:
            if result[1] >= 0.70:
                idx = [normalize_name(n) for n in perf_names_list].index(result[0])
                suggestions.append({
                    "year": year,
                    "auction_name": auc_name,
                    "suggested_perf_name": perf_names_list[idx],
                    "similarity": result[1]
                })

    return pd.DataFrame(suggestions)


def load_non_auction_acquisitions():
    """Load the non-auction acquisitions file if it exists."""
    non_auction_path = ACQUISITIONS_DIR / "non_auction_acquisitions.csv"
    if non_auction_path.exists():
        df = pd.read_csv(non_auction_path)
        non_auction_set = set()
        for _, row in df.iterrows():
            non_auction_set.add((int(row["year"]), normalize_name(row["player"])))
        return non_auction_set
    return set()


def find_perf_without_auction(perf, auction, mapping_df):
    """
    Find players in performance data who have no corresponding auction record.
    These are typically icon players, retained players, mid-season replacements, etc.
    Excludes players documented in non_auction_acquisitions.csv.
    """
    unmatched = []

    non_auction = load_non_auction_acquisitions()

    auction_mapped_names = set()
    for _, row in mapping_df.iterrows():
        auction_mapped_names.add((int(row["year"]), row["perf_name"]))

    for year in sorted(perf["season"].unique()):
        year = int(year)
        perf_year = set(perf[perf["season"] == year]["player"].unique())
        auction_year = set(auction[auction["year"] == year]["player_name"].unique())

        mapped_perf_names = {pn for (y, pn) in auction_mapped_names if y == year}
        all_auction_linked = auction_year | mapped_perf_names

        for player in perf_year:
            player_norm = normalize_name(player)
            is_matched = False
            for auc_name in all_auction_linked:
                if normalize_name(auc_name) == player_norm:
                    is_matched = True
                    break
            if player in mapped_perf_names:
                is_matched = True
            if (year, player_norm) in non_auction:
                is_matched = True
            if not is_matched:
                unmatched.append({"year": year, "player": player})

    return pd.DataFrame(unmatched)


def merge_auction_performance():
    """Merge auction and performance data using name mapping."""
    print("Loading data...")
    auction = load_dataset(ACQUISITIONS_DIR / "auction_all_years")
    perf = load_dataset(PERF_DIR / "player_season_stats")

    try:
        war = load_dataset(PERF_DIR / "player_season_war")
        print(f"  Loaded WAR data: {len(war)} player-seasons")
    except FileNotFoundError:
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
        how="left",
        suffixes=(SUFFIX_AUCTION, SUFFIX_IPL),
    )

    drop_cols = ["auction_name", "perf_name", "season", "player"]
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

    merged[COL_HAS_IPL_HISTORY] = merged["runs"].notna().astype(int)

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

    print("\nApplying inflation adjustment...")
    merged = adjust_prices_for_inflation(merged)
    print(f"  Added inflation-adjusted price columns")

    JOINED_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = save_dataset(merged, JOINED_DIR / "auction_with_performance", format="parquet")
    print(f"\nSaved to {output_path}")
    print(f"  Unmatched auction players: {len(unmatched_df)}")

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
        }),
        include_groups=False
    )
    print(year_stats.to_string())

    print("\n=== Sample Merged Data ===")
    team_col = "team_auction" if "team_auction" in merged.columns else "team"
    cols = ["year", "player_name", team_col, "final_price_lakh", "runs", "wickets", "batting_avg", "economy"]
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
    team_col = "team_auction" if "team_auction" in merged.columns else "team"
    unmatched_cols = ["year", "player_name", team_col, "final_price_lakh"]
    if "acquisition_type" in merged.columns:
        unmatched_cols.append("acquisition_type")
    unmatched = merged[merged["runs"].isna()][
        [c for c in unmatched_cols if c in merged.columns]
    ].sort_values("final_price_lakh", ascending=False).head(15)
    print(unmatched.to_string(index=False))

    if "acquisition_type" in merged.columns:
        print("\n=== Match Rate by Acquisition Type ===")
        acq_stats = merged.groupby("acquisition_type").apply(
            lambda x: pd.Series({
                "total": len(x),
                "matched": x["runs"].notna().sum(),
                "rate": x["runs"].notna().sum() / len(x) * 100
            }),
            include_groups=False
        )
        print(acq_stats.to_string())


if __name__ == "__main__":
    main()
