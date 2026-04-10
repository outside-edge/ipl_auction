#!/usr/bin/env python3
"""
Build a canonical player registry from IPL auction data.

Extracts all unique player names across all years, clusters them by similarity
to identify the same real-world player, and assigns a unique player_id.

Output: data/player_registry.csv with columns:
- player_id: Unique identifier (P001, P002, etc.)
- canonical_name: The preferred/most common name for this player
- aliases: Pipe-separated list of all name variations seen
"""

import re
from pathlib import Path

import pandas as pd
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cdist

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
ACQUISITIONS_DIR = DATA_DIR / "acquisitions"


def normalize_name_for_clustering(name):
    """Normalize name for similarity matching."""
    if pd.isna(name):
        return ""
    name = str(name).strip().lower()
    name = re.sub(r"\bMoh[ao]mm?[ae]d\b", "mohammed", name, flags=re.IGNORECASE)
    name = re.sub(r"\bMohd\b", "mohammed", name, flags=re.IGNORECASE)
    name = name.replace(".", " ").replace("-", " ").replace("'", "")
    name = " ".join(name.split())
    return name


def get_name_parts(name):
    """Extract first and last name parts for comparison."""
    parts = name.split()
    if len(parts) == 0:
        return "", ""
    if len(parts) == 1:
        return parts[0], parts[0]
    return parts[0], parts[-1]


def cluster_similar_names(names, threshold=88):
    """
    Cluster similar names together using fuzzy matching.

    Returns a list of clusters, where each cluster is a set of similar names.
    """
    if not names:
        return []

    names = list(set(names))
    names_normalized = [normalize_name_for_clustering(n) for n in names]

    scores = cdist(
        names_normalized,
        names_normalized,
        scorer=JaroWinkler.normalized_similarity,
        workers=-1,
    )

    visited = set()
    clusters = []

    for i, name in enumerate(names):
        if i in visited:
            continue

        cluster = {name}
        visited.add(i)

        for j in range(len(names)):
            if j in visited:
                continue

            if scores[i][j] >= threshold / 100:
                _, last_i = get_name_parts(names_normalized[i])
                _, last_j = get_name_parts(names_normalized[j])

                if len(last_i) >= 3 and len(last_j) >= 3:
                    if last_i[:3] != last_j[:3]:
                        continue

                cluster.add(names[j])
                visited.add(j)

        clusters.append(cluster)

    return clusters


def select_canonical_name(names, name_counts):
    """
    Select the best canonical name from a cluster.

    Prefers:
    1. Most frequently appearing name
    2. Longest name (more complete)
    3. Alphabetically first (for consistency)
    """
    if not names:
        return ""

    scored = []
    for name in names:
        count = name_counts.get(name, 0)
        length = len(name)
        scored.append((count, length, name))

    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return scored[0][2]


def build_registry():
    """Build the player registry from auction data."""
    auction_path = ACQUISITIONS_DIR / "auction_all_years.csv"
    if not auction_path.exists():
        print(f"ERROR: {auction_path} not found. Run assemble_auction_data.py first.")
        return None

    print("Loading auction data...")
    df = pd.read_csv(auction_path)
    print(f"  Loaded {len(df)} auction records")

    name_counts = df["player_name"].value_counts().to_dict()
    unique_names = list(df["player_name"].unique())
    print(f"  Found {len(unique_names)} unique names")

    print("\nClustering similar names...")
    clusters = cluster_similar_names(unique_names, threshold=88)
    print(f"  Created {len(clusters)} player clusters")

    multi_alias = [c for c in clusters if len(c) > 1]
    print(f"  Clusters with multiple aliases: {len(multi_alias)}")

    print("\nBuilding registry...")
    registry = []
    for i, cluster in enumerate(clusters):
        player_id = f"P{i+1:04d}"
        canonical = select_canonical_name(cluster, name_counts)
        aliases = "|".join(sorted(cluster))

        registry.append(
            {"player_id": player_id, "canonical_name": canonical, "aliases": aliases}
        )

    registry_df = pd.DataFrame(registry)
    registry_df = registry_df.sort_values("canonical_name").reset_index(drop=True)

    for i, _ in enumerate(registry_df.itertuples()):
        registry_df.loc[i, "player_id"] = f"P{i+1:04d}"

    output_path = ACQUISITIONS_DIR / "player_registry.csv"
    registry_df.to_csv(output_path, index=False)
    print(f"\nSaved registry to {output_path}")
    print(f"  Total players: {len(registry_df)}")

    print("\n=== Sample multi-alias players ===")
    multi = registry_df[registry_df["aliases"].str.contains(r"\|")]
    for _, row in multi.head(20).iterrows():
        print(f"  {row['player_id']}: {row['canonical_name']}")
        print(f"         Aliases: {row['aliases']}")

    return registry_df


def load_registry():
    """Load existing player registry."""
    registry_path = ACQUISITIONS_DIR / "player_registry.csv"
    if not registry_path.exists():
        return None
    return pd.read_csv(registry_path)


def lookup_player_id(name, registry_df):
    """Look up player_id for a given name."""
    if registry_df is None:
        return None

    for _, row in registry_df.iterrows():
        aliases = row["aliases"].split("|")
        if name in aliases:
            return row["player_id"]
    return None


def get_player_history(player_id, auction_df, registry_df):
    """Get all auction records for a player across years."""
    if registry_df is None:
        return pd.DataFrame()

    row = registry_df[registry_df["player_id"] == player_id]
    if row.empty:
        return pd.DataFrame()

    aliases = row["aliases"].iloc[0].split("|")
    return auction_df[auction_df["player_name"].isin(aliases)]


if __name__ == "__main__":
    build_registry()
