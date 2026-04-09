#!/usr/bin/env python3
"""
Download T20 International ball-by-ball data from Cricsheet.

Downloads JSON files for all men's T20I matches and extracts player registry
information for cross-dataset player linking.

Output:
    data/t20i/raw/ - Raw JSON files
    data/t20i/registry.csv - Player name to Cricsheet ID mapping
"""

import json
import zipfile
from io import BytesIO
from pathlib import Path

import requests

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
T20I_DIR = DATA_DIR / "perf" / "t20i"
RAW_DIR = T20I_DIR / "raw"

CRICSHEET_T20I_URL = "https://cricsheet.org/downloads/t20s_male_json.zip"


def download_and_extract():
    """Download T20I JSON zip from Cricsheet and extract."""
    print("Downloading T20I data from Cricsheet...")
    print(f"  URL: {CRICSHEET_T20I_URL}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    response = requests.get(CRICSHEET_T20I_URL, timeout=120)
    response.raise_for_status()

    print(f"  Downloaded {len(response.content) / 1024 / 1024:.1f} MB")

    with zipfile.ZipFile(BytesIO(response.content)) as zf:
        json_files = [f for f in zf.namelist() if f.endswith(".json")]
        print(f"  Found {len(json_files)} JSON files")

        for i, filename in enumerate(json_files):
            if (i + 1) % 500 == 0:
                print(f"  Extracted {i + 1}/{len(json_files)} files...")

            with zf.open(filename) as src:
                content = src.read()
                outpath = RAW_DIR / Path(filename).name
                outpath.write_bytes(content)

    print(f"  Extracted to {RAW_DIR}")
    return len(json_files)


def extract_registry():
    """Extract player registry (name -> Cricsheet ID) from all T20I JSON files."""
    print("\nExtracting player registry...")

    registry = {}
    match_count = 0
    t20i_match_count = 0

    for json_file in RAW_DIR.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)

            match_count += 1

            info = data.get("info", {})
            match_type = info.get("match_type", "")
            event = info.get("event", {})
            event_name = event.get("name", "") if isinstance(event, dict) else ""

            if match_type == "T20" and "International" not in event_name and "IPL" not in event_name:
                t20i_match_count += 1
            elif "T20I" in str(info.get("match_type_number", "")):
                t20i_match_count += 1

            people = data.get("info", {}).get("registry", {}).get("people", {})
            for name, cricsheet_id in people.items():
                if name not in registry:
                    registry[name] = cricsheet_id
                elif registry[name] != cricsheet_id:
                    print(f"  Warning: {name} has multiple IDs: {registry[name]} vs {cricsheet_id}")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Error reading {json_file}: {e}")
            continue

    print(f"  Processed {match_count} match files")
    print(f"  Found {len(registry)} unique players")

    import pandas as pd

    registry_df = pd.DataFrame(
        [{"player_name": name, "cricsheet_id": cid} for name, cid in registry.items()]
    )
    registry_df = registry_df.sort_values("player_name")

    registry_path = T20I_DIR / "registry.csv"
    registry_df.to_csv(registry_path, index=False)
    print(f"  Saved registry to {registry_path}")

    return registry_df


def main():
    print("=" * 60)
    print("Cricsheet T20I Data Download")
    print("=" * 60)

    n_files = download_and_extract()
    registry = extract_registry()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Downloaded: {n_files} match files")
    print(f"Registry: {len(registry)} players")


if __name__ == "__main__":
    main()
