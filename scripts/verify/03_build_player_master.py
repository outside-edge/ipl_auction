#!/usr/bin/env python3
"""
Build unified player master table linking IPL and T20I players via Cricsheet IDs.

Uses Cricsheet's 8-character hex player IDs as the canonical global identifier.
Extracts registry from IPL JSON files and merges with existing player_registry.csv.

Output: data/analysis/joined/player_master.csv
"""

import json
import sys
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from shared.names import normalize_name
from shared.io import save_dataset

DATA_DIR = BASE_DIR / "data"
ACQUISITIONS_DIR = DATA_DIR / "acquisitions"
PERF_SOURCES_DIR = DATA_DIR / "perf" / "sources"
JOINED_DIR = DATA_DIR / "analysis" / "joined"
DIAGNOSTICS_DIR = DATA_DIR / "analysis" / "diagnostics"
IPL_JSON_DIR = PERF_SOURCES_DIR / "kaggle" / "ipl-dataset" / "json" / "ipl_match"
T20I_DIR = DATA_DIR / "perf" / "t20i"


def extract_ipl_registry():
    """Extract player registry from IPL JSON files."""
    print("Extracting IPL player registry from JSON files...")

    registry = {}
    match_count = 0

    for json_file in IPL_JSON_DIR.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            match_count += 1
            people = data.get("info", {}).get("registry", {}).get("people", {})
            for name, cricsheet_id in people.items():
                if name not in registry:
                    registry[name] = cricsheet_id

        except (json.JSONDecodeError, KeyError):
            continue

    print(f"  Processed {match_count} IPL matches")
    print(f"  Found {len(registry)} unique IPL players with Cricsheet IDs")

    registry_df = pd.DataFrame([
        {"ipl_name": name, "cricsheet_id": cid}
        for name, cid in registry.items()
    ])

    return registry_df


def load_existing_registry():
    """Load existing player_registry.csv."""
    registry_path = ACQUISITIONS_DIR / "player_registry.csv"
    registry = pd.read_csv(registry_path)
    print(f"Loaded existing registry: {len(registry)} players")
    return registry


def load_t20i_registry():
    """Load T20I player registry."""
    t20i_registry_path = T20I_DIR / "registry.csv"
    if t20i_registry_path.exists():
        t20i_registry = pd.read_csv(t20i_registry_path)
        print(f"Loaded T20I registry: {len(t20i_registry)} players")
        return t20i_registry
    return pd.DataFrame(columns=["player_name", "cricsheet_id"])


def match_ipl_to_registry(ipl_registry, existing_registry):
    """Match IPL players (with Cricsheet IDs) to existing registry (with player_id)."""
    print("\nMatching IPL players to existing registry...")

    if len(ipl_registry) == 0:
        print("  No IPL registry data available - skipping matching")
        return pd.DataFrame(columns=["cricsheet_id", "player_id", "canonical_name", "ipl_name", "match_type", "match_score"]), pd.DataFrame(columns=["cricsheet_id", "ipl_name"])

    existing_registry["canonical_norm"] = existing_registry["canonical_name"].apply(normalize_name)

    existing_registry["all_aliases"] = existing_registry.apply(
        lambda row: [normalize_name(a) for a in str(row.get("aliases", "")).split("|") if a],
        axis=1
    )

    ipl_registry["ipl_norm"] = ipl_registry["ipl_name"].apply(normalize_name)

    matches = []
    unmatched_ipl = []

    for _, ipl_row in ipl_registry.iterrows():
        ipl_name = ipl_row["ipl_name"]
        ipl_norm = ipl_row["ipl_norm"]
        cricsheet_id = ipl_row["cricsheet_id"]

        best_match = None
        best_score = 0
        match_type = None

        for _, reg_row in existing_registry.iterrows():
            canon_norm = reg_row["canonical_norm"]

            if ipl_norm == canon_norm:
                best_match = reg_row
                best_score = 100
                match_type = "exact"
                break

            if ipl_norm in reg_row["all_aliases"]:
                best_match = reg_row
                best_score = 100
                match_type = "alias"
                break

        if best_match is None:
            for _, reg_row in existing_registry.iterrows():
                canon_norm = reg_row["canonical_norm"]

                score = fuzz.WRatio(ipl_norm, canon_norm)
                if score > best_score and score >= 90:
                    best_match = reg_row
                    best_score = score
                    match_type = "fuzzy"

                for alias in reg_row["all_aliases"]:
                    score = fuzz.WRatio(ipl_norm, alias)
                    if score > best_score and score >= 90:
                        best_match = reg_row
                        best_score = score
                        match_type = "fuzzy_alias"

        if best_match is not None:
            matches.append({
                "cricsheet_id": cricsheet_id,
                "player_id": best_match["player_id"],
                "canonical_name": best_match["canonical_name"],
                "ipl_name": ipl_name,
                "match_type": match_type,
                "match_score": best_score,
            })
        else:
            unmatched_ipl.append({
                "cricsheet_id": cricsheet_id,
                "ipl_name": ipl_name,
            })

    matches_df = pd.DataFrame(matches)
    unmatched_df = pd.DataFrame(unmatched_ipl)

    print(f"  Matched: {len(matches_df)} players")
    print(f"  Unmatched IPL players: {len(unmatched_df)}")
    print(f"  Match types: {matches_df['match_type'].value_counts().to_dict()}")

    return matches_df, unmatched_df


def build_player_master(matches_df, unmatched_ipl_df, t20i_registry, existing_registry):
    """Build the final player master table."""
    print("\nBuilding player master table...")

    if len(matches_df) == 0:
        print("  No IPL-Cricsheet matches available - building from existing registry")
        player_master = existing_registry[["player_id", "canonical_name"]].copy()
        player_master["cricsheet_id"] = None
        player_master["has_ipl"] = True
        player_master["has_t20i"] = False
        return player_master

    player_master = matches_df[["cricsheet_id", "player_id", "canonical_name"]].copy()

    t20i_ids = set(t20i_registry["cricsheet_id"]) if len(t20i_registry) > 0 else set()
    ipl_ids = set(player_master["cricsheet_id"])
    t20i_only = t20i_ids - ipl_ids

    print(f"  Players in both IPL and T20I: {len(ipl_ids.intersection(t20i_ids))}")
    print(f"  T20I-only players (not in IPL): {len(t20i_only)}")

    player_master["has_ipl"] = True
    player_master["has_t20i"] = player_master["cricsheet_id"].isin(t20i_ids)

    return player_master


def verify_player_master(player_master):
    """Verify player master table quality."""
    print("\n" + "=" * 60)
    print("PLAYER MASTER VERIFICATION")
    print("=" * 60)

    print(f"\nTotal players: {len(player_master)}")
    print(f"With IPL: {player_master['has_ipl'].sum()}")
    print(f"With T20I: {player_master['has_t20i'].sum()}")
    print(f"Both IPL and T20I: {(player_master['has_ipl'] & player_master['has_t20i']).sum()}")

    dup_cricsheet = player_master["cricsheet_id"].duplicated()
    dup_player_id = player_master["player_id"].duplicated()

    print(f"\nDuplicate Cricsheet IDs: {dup_cricsheet.sum()}")
    print(f"Duplicate player_ids: {dup_player_id.sum()}")

    if dup_cricsheet.sum() > 0:
        print("  Sample duplicates:")
        dups = player_master[dup_cricsheet]["cricsheet_id"].head(5)
        for cid in dups:
            print(f"    {cid}: {player_master[player_master['cricsheet_id'] == cid]['canonical_name'].tolist()}")


def main():
    print("=" * 60)
    print("Building Player Master Table")
    print("=" * 60)

    ipl_registry = extract_ipl_registry()
    existing_registry = load_existing_registry()
    t20i_registry = load_t20i_registry()

    matches_df, unmatched_df = match_ipl_to_registry(ipl_registry, existing_registry)

    player_master = build_player_master(matches_df, unmatched_df, t20i_registry, existing_registry)

    verify_player_master(player_master)

    JOINED_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = save_dataset(player_master, JOINED_DIR / "player_master", format="parquet")
    print(f"\nSaved to {output_path}")

    if len(ipl_registry) > 0:
        ipl_registry_path = JOINED_DIR / "ipl_cricsheet_registry.csv"
        ipl_registry.to_csv(ipl_registry_path, index=False)
        print(f"Saved IPL Cricsheet registry to {ipl_registry_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    return player_master


if __name__ == "__main__":
    main()
