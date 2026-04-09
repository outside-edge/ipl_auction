#!/usr/bin/env python3
"""
Process T20I ball-by-ball data from Cricsheet JSON to CSV format matching IPL schema.

Converts Cricsheet JSON structure to match the existing Kaggle IPL format:
- Ball_By_Ball_Match_Data.csv schema
- Match_Info.csv schema

This allows reuse of existing compute_war.py and process_deliveries.py logic.

Input: data/t20i/raw/*.json (Cricsheet format)
Output: data/t20i/deliveries.csv, data/t20i/matches.csv
"""

import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
T20I_DIR = DATA_DIR / "perf" / "t20i"
RAW_DIR = T20I_DIR / "raw"


def parse_match_info(data, match_id):
    """Parse match info from Cricsheet JSON."""
    info = data.get("info", {})

    dates = info.get("dates", [])
    match_date = dates[0] if dates else None

    teams = info.get("teams", [])
    team1 = teams[0] if len(teams) > 0 else None
    team2 = teams[1] if len(teams) > 1 else None

    toss = info.get("toss", {})
    toss_winner = toss.get("winner")
    toss_decision = toss.get("decision")

    outcome = info.get("outcome", {})
    winner = outcome.get("winner")
    result = "Win" if winner else "No Result"

    pom = info.get("player_of_match", [])
    player_of_match = pom[0] if pom else None

    players = info.get("players", {})
    team1_players = ", ".join(players.get(team1, [])) if team1 else ""
    team2_players = ", ".join(players.get(team2, [])) if team2 else ""

    return {
        "match_number": match_id,
        "team1": team1,
        "team2": team2,
        "match_date": match_date,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "result": result,
        "eliminator": "NA",
        "winner": winner,
        "player_of_match": player_of_match,
        "venue": info.get("venue"),
        "city": info.get("city"),
        "team1_players": team1_players,
        "team2_players": team2_players,
        "match_type": info.get("match_type"),
        "gender": info.get("gender"),
        "event_name": info.get("event", {}).get("name") if isinstance(info.get("event"), dict) else info.get("event"),
    }


def parse_deliveries(data, match_id):
    """Parse ball-by-ball deliveries from Cricsheet JSON."""
    info = data.get("info", {})
    innings_data = data.get("innings", [])

    info.get("teams", [])

    deliveries = []

    for innings_idx, innings in enumerate(innings_data):
        team = innings.get("team")
        overs_data = innings.get("overs", [])

        for over_data in overs_data:
            over_num = over_data.get("over", 0)

            for ball_idx, delivery in enumerate(over_data.get("deliveries", [])):
                batter = delivery.get("batter")
                bowler = delivery.get("bowler")
                non_striker = delivery.get("non_striker")

                runs = delivery.get("runs", {})
                batsman_run = runs.get("batter", 0)
                extras_run = runs.get("extras", 0)
                total_run = runs.get("total", 0)

                extras = delivery.get("extras", {})
                extra_type = None
                if "wides" in extras:
                    extra_type = "wides"
                elif "noballs" in extras:
                    extra_type = "noballs"
                elif "byes" in extras:
                    extra_type = "byes"
                elif "legbyes" in extras:
                    extra_type = "legbyes"
                elif "penalty" in extras:
                    extra_type = "penalty"

                wickets = delivery.get("wickets", [])
                is_wicket = 1 if wickets else 0
                player_out = "NA"
                kind = "NA"
                fielders_involved = "NA"

                if wickets:
                    w = wickets[0]
                    player_out = w.get("player_out", "NA")
                    kind = w.get("kind", "NA")
                    fielders = w.get("fielders", [])
                    if fielders:
                        fielder_names = [f.get("name", "") for f in fielders if isinstance(f, dict)]
                        if not fielder_names:
                            fielder_names = [str(f) for f in fielders if isinstance(f, str)]
                        fielders_involved = ", ".join(fielder_names) if fielder_names else "NA"

                deliveries.append({
                    "ID": match_id,
                    "Innings": innings_idx + 1,
                    "Overs": over_num,
                    "BallNumber": ball_idx + 1,
                    "Batter": batter,
                    "Bowler": bowler,
                    "NonStriker": non_striker,
                    "ExtraType": extra_type if extra_type else "",
                    "BatsmanRun": batsman_run,
                    "ExtrasRun": extras_run,
                    "TotalRun": total_run,
                    "IsWicketDelivery": is_wicket,
                    "PlayerOut": player_out,
                    "Kind": kind,
                    "FieldersInvolved": fielders_involved,
                    "BattingTeam": team,
                })

    return deliveries


def filter_t20i_only(matches_df, deliveries_df):
    """Filter to only T20 International matches (exclude domestic leagues)."""
    domestic_leagues = [
        "Indian Premier League", "IPL",
        "Big Bash League", "BBL",
        "Pakistan Super League", "PSL",
        "Caribbean Premier League", "CPL",
        "Hundred", "The Hundred",
        "SA20",
        "Lanka Premier League", "LPL",
        "Bangladesh Premier League", "BPL",
        "Abu Dhabi T10",
        "Super Smash",
        "Vitality Blast", "T20 Blast",
        "ILT20",
        "Major League Cricket", "MLC",
    ]

    def is_t20i(row):
        event = str(row.get("event_name", "") or "").lower()
        match_type = str(row.get("match_type", "") or "").lower()

        for league in domestic_leagues:
            if league.lower() in event:
                return False

        if "international" in event or "t20i" in match_type:
            return True

        teams = [str(row.get("team1", "") or ""), str(row.get("team2", "") or "")]
        national_teams = ["India", "Australia", "England", "Pakistan", "South Africa",
                         "New Zealand", "West Indies", "Sri Lanka", "Bangladesh",
                         "Afghanistan", "Zimbabwe", "Ireland", "Scotland", "Netherlands",
                         "Nepal", "Oman", "UAE", "USA", "Canada", "Namibia", "PNG"]

        team_matches = sum(1 for t in teams if any(nt.lower() in t.lower() for nt in national_teams))
        if team_matches >= 2:
            return True

        return False

    t20i_mask = matches_df.apply(is_t20i, axis=1)
    t20i_matches = matches_df[t20i_mask].copy()
    t20i_match_ids = set(t20i_matches["match_number"])
    t20i_deliveries = deliveries_df[deliveries_df["ID"].isin(t20i_match_ids)].copy()

    return t20i_matches, t20i_deliveries


def main():
    print("=" * 60)
    print("Process T20I Deliveries (Cricsheet JSON -> CSV)")
    print("=" * 60)

    json_files = list(RAW_DIR.glob("*.json"))
    print(f"Found {len(json_files)} JSON files")

    all_matches = []
    all_deliveries = []
    errors = []

    for i, json_file in enumerate(json_files):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(json_files)} files...")

        try:
            with open(json_file) as f:
                data = json.load(f)

            match_id = int(json_file.stem)

            match_info = parse_match_info(data, match_id)
            all_matches.append(match_info)

            deliveries = parse_deliveries(data, match_id)
            all_deliveries.extend(deliveries)

        except Exception as e:
            errors.append((json_file.name, str(e)))
            continue

    if errors:
        print(f"\n  Errors: {len(errors)} files")
        for fname, err in errors[:5]:
            print(f"    {fname}: {err}")

    print(f"\n  Total matches parsed: {len(all_matches)}")
    print(f"  Total deliveries: {len(all_deliveries)}")

    matches_df = pd.DataFrame(all_matches)
    deliveries_df = pd.DataFrame(all_deliveries)

    print("\nFiltering to T20I only (excluding domestic leagues)...")
    t20i_matches, t20i_deliveries = filter_t20i_only(matches_df, deliveries_df)
    print(f"  T20I matches: {len(t20i_matches)}")
    print(f"  T20I deliveries: {len(t20i_deliveries)}")

    matches_path = T20I_DIR / "matches.csv"
    t20i_matches.to_csv(matches_path, index=False)
    print(f"\nSaved matches to {matches_path}")

    deliveries_path = T20I_DIR / "deliveries.csv"
    t20i_deliveries.to_csv(deliveries_path, index=False)
    print(f"Saved deliveries to {deliveries_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if len(t20i_matches) > 0:
        years = pd.to_datetime(t20i_matches["match_date"]).dt.year
        print(f"  Years covered: {years.min()} - {years.max()}")
        print(f"  Matches per year:")
        for year, count in years.value_counts().sort_index().items():
            print(f"    {year}: {count}")

    print("\n  Top teams by matches:")
    team_counts = pd.concat([t20i_matches["team1"], t20i_matches["team2"]]).value_counts()
    for team, count in team_counts.head(10).items():
        print(f"    {team}: {count}")


if __name__ == "__main__":
    main()
