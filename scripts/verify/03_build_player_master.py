#!/usr/bin/env python3
"""
Build unified player master table linking IPL and T20I players via Cricsheet IDs.

Uses Cricsheet's 8-character hex player IDs as the canonical global identifier.
Extracts registry from IPL JSON files and merges with existing player_registry.csv.

Uses preclink for high-precision 1:1 record linkage.

Output: data/analysis/joined/player_master.parquet
"""

import json
import sys
from pathlib import Path

import pandas as pd
from preclink import Pipeline, StringComparison

BASE_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))

from shared.io import save_dataset  # noqa: E402
from shared.names import convert_full_to_initial_format, normalize_name  # noqa: E402

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

    registry_df = pd.DataFrame(
        [{"ipl_name": name, "cricsheet_id": cid} for name, cid in registry.items()]
    )

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


def get_last_name(name: str) -> str:
    """Extract last name from normalized name."""
    parts = name.split()
    return parts[-1] if parts else ""


def match_ipl_to_registry_preclink(ipl_registry, existing_registry):
    """Match IPL players to existing registry using preclink for 1:1 assignment."""
    print("\nMatching IPL players to existing registry using preclink...")

    if len(ipl_registry) == 0:
        print("  No IPL registry data available - skipping matching")
        empty_matches = pd.DataFrame(
            columns=[
                "cricsheet_id",
                "player_id",
                "canonical_name",
                "ipl_name",
                "match_type",
                "match_score",
            ]
        )
        empty_unmatched = pd.DataFrame(columns=["cricsheet_id", "ipl_name"])
        return empty_matches, empty_unmatched

    left = existing_registry[["player_id", "canonical_name"]].copy()
    left["name_norm"] = left["canonical_name"].apply(normalize_name)
    left["name_init"] = left["name_norm"].apply(convert_full_to_initial_format)
    left["last_name"] = left["name_norm"].apply(get_last_name)

    right = ipl_registry[["cricsheet_id", "ipl_name"]].copy()
    right["name_norm"] = right["ipl_name"].apply(normalize_name)
    right["last_name"] = right["name_norm"].apply(get_last_name)

    pipeline = (
        Pipeline()
        .block(on="last_name")
        .score(
            [
                StringComparison("name_norm", algorithm="jaro_winkler", weight=1.5),
                StringComparison("last_name", algorithm="jaro_winkler", weight=1.0),
            ]
        )
        .filter(min_score=0.80)
        .decide(method="greedy")
        .build()
    )

    print("  Running preclink pipeline (high threshold: 0.80)...")
    result = pipeline.link(left, right)
    matches_high = result.matches.copy()
    print(f"  High-confidence matches: {len(matches_high)}")

    matched_left = set(matches_high["left_index"]) if len(matches_high) > 0 else set()
    matched_right = set(matches_high["right_index"]) if len(matches_high) > 0 else set()

    left_remaining = left[~left.index.isin(matched_left)].copy()
    right_remaining = right[~right.index.isin(matched_right)].copy()

    if len(left_remaining) > 0 and len(right_remaining) > 0:
        left_remaining["name_compare"] = left_remaining["name_init"]
        right_remaining["name_compare"] = right_remaining["name_norm"]

        pipeline_init = (
            Pipeline()
            .block(on="last_name")
            .score(
                [
                    StringComparison(
                        "name_compare", algorithm="jaro_winkler", weight=2.0
                    ),
                    StringComparison("last_name", algorithm="jaro_winkler", weight=1.0),
                ]
            )
            .filter(min_score=0.88)
            .decide(method="greedy")
            .build()
        )

        print("  Running initial-format matching pass...")
        result_init = pipeline_init.link(left_remaining, right_remaining)
        matches_init = result_init.matches.copy()
        print(f"  Initial-format matches: {len(matches_init)}")

        matched_left.update(matches_init["left_index"])
        matched_right.update(matches_init["right_index"])

    all_matches = []

    if len(matches_high) > 0:
        for _, row in matches_high.iterrows():
            left_idx = row["left_index"]
            right_idx = row["right_index"]
            all_matches.append(
                {
                    "cricsheet_id": right.loc[right_idx, "cricsheet_id"],
                    "player_id": left.loc[left_idx, "player_id"],
                    "canonical_name": left.loc[left_idx, "canonical_name"],
                    "ipl_name": right.loc[right_idx, "ipl_name"],
                    "match_type": "preclink_high",
                    "match_score": row["score"],
                }
            )

    if len(left_remaining) > 0 and len(right_remaining) > 0 and len(matches_init) > 0:
        for _, row in matches_init.iterrows():
            left_idx = row["left_index"]
            right_idx = row["right_index"]
            all_matches.append(
                {
                    "cricsheet_id": right_remaining.loc[right_idx, "cricsheet_id"],
                    "player_id": left_remaining.loc[left_idx, "player_id"],
                    "canonical_name": left_remaining.loc[left_idx, "canonical_name"],
                    "ipl_name": right_remaining.loc[right_idx, "ipl_name"],
                    "match_type": "preclink_init",
                    "match_score": row["score"],
                }
            )

    matches_df = pd.DataFrame(all_matches)

    unmatched_right_idx = set(right.index) - matched_right
    unmatched_ipl = [
        {
            "cricsheet_id": right.loc[idx, "cricsheet_id"],
            "ipl_name": right.loc[idx, "ipl_name"],
        }
        for idx in unmatched_right_idx
    ]
    unmatched_df = pd.DataFrame(unmatched_ipl)

    print(f"\n  Total matched: {len(matches_df)} players")
    print(f"  Unmatched IPL players: {len(unmatched_df)}")
    if len(matches_df) > 0:
        print(f"  Match types: {matches_df['match_type'].value_counts().to_dict()}")

    return matches_df, unmatched_df


def build_player_master(matches_df, t20i_registry, existing_registry):
    """Build the final player master table."""
    print("\nBuilding player master table...")

    if len(matches_df) == 0:
        print("  No IPL-Cricsheet matches available - building from existing registry")
        player_master = existing_registry[["player_id", "canonical_name"]].copy()
        player_master["cricsheet_id"] = None
        player_master["has_ipl"] = True
        player_master["has_t20i"] = False
        return player_master

    matched_pids = set(matches_df["player_id"])
    all_registry_pids = set(existing_registry["player_id"])
    unmatched_pids = all_registry_pids - matched_pids

    matched_part = matches_df[["player_id", "canonical_name", "cricsheet_id"]].copy()

    unmatched_part = existing_registry[
        existing_registry["player_id"].isin(unmatched_pids)
    ][["player_id", "canonical_name"]].copy()
    unmatched_part["cricsheet_id"] = None

    player_master = pd.concat([matched_part, unmatched_part], ignore_index=True)

    t20i_ids = set(t20i_registry["cricsheet_id"]) if len(t20i_registry) > 0 else set()
    player_master["has_ipl"] = True
    player_master["has_t20i"] = player_master["cricsheet_id"].isin(t20i_ids)

    matched_with_t20i = (
        player_master["cricsheet_id"].notna() & player_master["has_t20i"]
    )
    print(f"  Total registry players: {len(player_master)}")
    print(f"  With Cricsheet ID: {player_master['cricsheet_id'].notna().sum()}")
    print(f"  With T20I data: {matched_with_t20i.sum()}")

    return player_master


def verify_player_master(player_master):
    """Verify player master table quality."""
    print("\n" + "=" * 60)
    print("PLAYER MASTER VERIFICATION")
    print("=" * 60)

    print(f"\nTotal players: {len(player_master)}")
    print(f"With Cricsheet ID: {player_master['cricsheet_id'].notna().sum()}")
    print(f"Without Cricsheet ID: {player_master['cricsheet_id'].isna().sum()}")
    print(f"With T20I: {player_master['has_t20i'].sum()}")

    has_id = player_master["cricsheet_id"].notna()
    dup_cricsheet = player_master[has_id]["cricsheet_id"].duplicated()
    dup_player_id = player_master["player_id"].duplicated()

    print(f"\nDuplicate Cricsheet IDs: {dup_cricsheet.sum()}")
    print(f"Duplicate player_ids: {dup_player_id.sum()}")

    if dup_cricsheet.sum() > 0:
        print("  Sample duplicate Cricsheet IDs:")
        dups = player_master[has_id][dup_cricsheet]["cricsheet_id"].head(5)
        for cid in dups:
            names = player_master[player_master["cricsheet_id"] == cid][
                "canonical_name"
            ].tolist()
            print(f"    {cid}: {names}")


def main():
    print("=" * 60)
    print("Building Player Master Table")
    print("=" * 60)

    ipl_registry = extract_ipl_registry()
    existing_registry = load_existing_registry()
    t20i_registry = load_t20i_registry()

    matches_df, unmatched_df = match_ipl_to_registry_preclink(
        ipl_registry, existing_registry
    )

    player_master = build_player_master(matches_df, t20i_registry, existing_registry)

    verify_player_master(player_master)

    JOINED_DIR.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = save_dataset(
        player_master, JOINED_DIR / "player_master", format="parquet"
    )
    print(f"\nSaved player master to {output_path}")

    if len(ipl_registry) > 0:
        ipl_registry_path = JOINED_DIR / "ipl_cricsheet_registry.csv"
        ipl_registry.to_csv(ipl_registry_path, index=False)
        print(f"Saved IPL Cricsheet registry to {ipl_registry_path}")

    if len(unmatched_df) > 0:
        unmatched_path = DIAGNOSTICS_DIR / "unmatched_ipl_players.csv"
        unmatched_df.to_csv(unmatched_path, index=False)
        print(f"Saved unmatched IPL players to {unmatched_path}")

    if len(matches_df) > 0:
        matches_path = DIAGNOSTICS_DIR / "cricsheet_matches.csv"
        matches_df.to_csv(matches_path, index=False)
        print(f"Saved match details to {matches_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    return player_master


if __name__ == "__main__":
    main()
