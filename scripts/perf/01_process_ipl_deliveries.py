#!/usr/bin/env python3
"""
Process ball-by-ball IPL data to create match-level and season-level player statistics.

Input: data/kaggle/ipl-dataset/csv/Ball_By_Ball_Match_Data.csv
       data/kaggle/ipl-dataset/csv/Match_Info.csv

Output: data/player_season_stats.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
PERF_DIR = DATA_DIR / "perf" / "ipl"
SOURCES_DIR = DATA_DIR / "perf" / "sources"


def load_data():
    """Load ball-by-ball and match info data."""
    bbb_path = SOURCES_DIR / "kaggle/ipl-dataset/Ball_By_Ball_Match_Data.csv"
    match_path = SOURCES_DIR / "kaggle/ipl-dataset/Match_Info.csv"

    print("Loading ball-by-ball data...")
    bbb = pd.read_csv(bbb_path)
    print(f"  {len(bbb):,} deliveries loaded")

    print("Loading match info...")
    matches = pd.read_csv(match_path)
    matches["match_date"] = pd.to_datetime(matches["match_date"])
    matches["season"] = matches["match_date"].dt.year
    print(f"  {len(matches):,} matches loaded")

    bbb = bbb.merge(
        matches[["match_number", "season", "winner", "team1", "team2"]],
        left_on="ID",
        right_on="match_number",
        how="left"
    )

    return bbb, matches


def compute_batting_stats(bbb):
    """Compute match-level batting statistics for each player."""
    print("Computing batting statistics...")

    bat = bbb.groupby(["ID", "season", "Batter", "BattingTeam"]).agg(
        runs=("BatsmanRun", "sum"),
        balls=("BatsmanRun", "count"),
        fours=("BatsmanRun", lambda x: (x == 4).sum()),
        sixes=("BatsmanRun", lambda x: (x == 6).sum()),
        dots=("BatsmanRun", lambda x: (x == 0).sum()),
    ).reset_index()

    dismissals = bbb[bbb["IsWicketDelivery"] == 1].groupby(
        ["ID", "PlayerOut"]
    ).size().reset_index(name="times_out")

    bat = bat.merge(
        dismissals,
        left_on=["ID", "Batter"],
        right_on=["ID", "PlayerOut"],
        how="left"
    )
    bat["times_out"] = bat["times_out"].fillna(0).astype(int)
    bat["is_out"] = bat["times_out"] > 0

    bat["strike_rate"] = np.where(
        bat["balls"] > 0,
        (bat["runs"] / bat["balls"]) * 100,
        0
    )

    bat = bat.rename(columns={"Batter": "player", "BattingTeam": "team"})
    bat = bat[["ID", "season", "player", "team", "runs", "balls", "fours", "sixes", "dots", "is_out", "strike_rate"]]

    print(f"  {len(bat):,} batting innings computed")
    return bat


def compute_bowling_stats(bbb):
    """Compute match-level bowling statistics for each player."""
    print("Computing bowling statistics...")

    bowl = bbb.groupby(["ID", "season", "Bowler"]).agg(
        balls=("TotalRun", "count"),
        runs_conceded=("TotalRun", "sum"),
        wides=("ExtraType", lambda x: (x == "wides").sum()),
        no_balls=("ExtraType", lambda x: (x == "noballs").sum()),
        dots=("TotalRun", lambda x: (x == 0).sum()),
        fours_conceded=("BatsmanRun", lambda x: (x == 4).sum()),
        sixes_conceded=("BatsmanRun", lambda x: (x == 6).sum()),
    ).reset_index()

    wickets = bbb[bbb["IsWicketDelivery"] == 1].copy()
    wickets = wickets[~wickets["Kind"].isin(["run out", "retired hurt", "retired out", "obstructing the field"])]
    wicket_counts = wickets.groupby(["ID", "Bowler"]).size().reset_index(name="wickets")

    bowl = bowl.merge(wicket_counts, on=["ID", "Bowler"], how="left")
    bowl["wickets"] = bowl["wickets"].fillna(0).astype(int)

    bowl["legal_balls"] = bowl["balls"] - bowl["wides"] - bowl["no_balls"]
    bowl["overs"] = bowl["legal_balls"] // 6 + (bowl["legal_balls"] % 6) / 10

    bowl["economy"] = np.where(
        bowl["overs"] > 0,
        bowl["runs_conceded"] / (bowl["legal_balls"] / 6),
        0
    )

    bowl = bowl.rename(columns={"Bowler": "player"})
    bowl = bowl[["ID", "season", "player", "overs", "legal_balls", "runs_conceded", "wickets", "dots", "economy", "wides", "no_balls", "fours_conceded", "sixes_conceded"]]

    print(f"  {len(bowl):,} bowling innings computed")
    return bowl


def compute_fielding_stats(bbb):
    """Compute match-level fielding statistics."""
    print("Computing fielding statistics...")

    wickets = bbb[bbb["IsWicketDelivery"] == 1].copy()

    def extract_fielders(row):
        if pd.isna(row["FieldersInvolved"]) or row["FieldersInvolved"] == "NA":
            return []
        return [f.strip() for f in str(row["FieldersInvolved"]).split(",")]

    fielding_records = []
    for _, row in wickets.iterrows():
        fielders = extract_fielders(row)
        kind = row["Kind"]
        for fielder in fielders:
            if fielder:
                fielding_records.append({
                    "ID": row["ID"],
                    "season": row["season"],
                    "player": fielder,
                    "kind": kind
                })

    if not fielding_records:
        print("  No fielding records found")
        return pd.DataFrame(columns=["ID", "season", "player", "catches", "run_outs", "stumpings"])

    field_df = pd.DataFrame(fielding_records)

    field_agg = field_df.groupby(["ID", "season", "player"]).agg(
        catches=("kind", lambda x: (x == "caught").sum()),
        run_outs=("kind", lambda x: (x == "run out").sum()),
        stumpings=("kind", lambda x: (x == "stumped").sum()),
    ).reset_index()

    print(f"  {len(field_agg):,} fielding records computed")
    return field_agg


def aggregate_to_season(bat_match, bowl_match, field_match, matches):
    """Aggregate match-level stats to season-level."""
    print("Aggregating to season level...")

    matches_played = bat_match.groupby(["season", "player"]).agg(
        matches_played=("ID", "nunique"),
        team=("team", lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
    ).reset_index()

    bat_season = bat_match.groupby(["season", "player"]).agg(
        bat_innings=("ID", "count"),
        runs=("runs", "sum"),
        balls_faced=("balls", "sum"),
        fours=("fours", "sum"),
        sixes=("sixes", "sum"),
        bat_dots=("dots", "sum"),
        dismissals=("is_out", "sum"),
        highest_score=("runs", "max"),
    ).reset_index()

    bat_season["not_outs"] = bat_season["bat_innings"] - bat_season["dismissals"]
    bat_season["fifties"] = bat_match.groupby(["season", "player"]).apply(
        lambda x: ((x["runs"] >= 50) & (x["runs"] < 100)).sum(), include_groups=False
    ).reset_index(drop=True)
    bat_season["hundreds"] = bat_match.groupby(["season", "player"]).apply(
        lambda x: (x["runs"] >= 100).sum(), include_groups=False
    ).reset_index(drop=True)
    bat_season["batting_avg"] = np.where(
        bat_season["dismissals"] > 0,
        bat_season["runs"] / bat_season["dismissals"],
        bat_season["runs"]
    )
    bat_season["batting_sr"] = np.where(
        bat_season["balls_faced"] > 0,
        (bat_season["runs"] / bat_season["balls_faced"]) * 100,
        0
    )

    bowl_season = bowl_match.groupby(["season", "player"]).agg(
        bowl_innings=("ID", "count"),
        overs=("overs", "sum"),
        balls_bowled=("legal_balls", "sum"),
        runs_conceded=("runs_conceded", "sum"),
        wickets=("wickets", "sum"),
        bowl_dots=("dots", "sum"),
        wides=("wides", "sum"),
        no_balls=("no_balls", "sum"),
        fours_conceded=("fours_conceded", "sum"),
        sixes_conceded=("sixes_conceded", "sum"),
    ).reset_index()

    bowl_season["bowling_avg"] = np.where(
        bowl_season["wickets"] > 0,
        bowl_season["runs_conceded"] / bowl_season["wickets"],
        np.nan
    )
    bowl_season["economy"] = np.where(
        bowl_season["balls_bowled"] > 0,
        bowl_season["runs_conceded"] / (bowl_season["balls_bowled"] / 6),
        np.nan
    )
    bowl_season["bowling_sr"] = np.where(
        bowl_season["wickets"] > 0,
        bowl_season["balls_bowled"] / bowl_season["wickets"],
        np.nan
    )

    best_bowling = bowl_match.loc[
        bowl_match.groupby(["season", "player"])["wickets"].idxmax()
    ][["season", "player", "wickets", "runs_conceded"]].copy()
    best_bowling["best_figures"] = best_bowling["wickets"].astype(str) + "/" + best_bowling["runs_conceded"].astype(str)
    best_bowling = best_bowling[["season", "player", "best_figures"]]

    bowl_season["four_wickets"] = bowl_match.groupby(["season", "player"]).apply(
        lambda x: (x["wickets"] >= 4).sum(), include_groups=False
    ).reset_index(drop=True)
    bowl_season["five_wickets"] = bowl_match.groupby(["season", "player"]).apply(
        lambda x: (x["wickets"] >= 5).sum(), include_groups=False
    ).reset_index(drop=True)

    bowl_season = bowl_season.merge(best_bowling, on=["season", "player"], how="left")

    if not field_match.empty:
        field_season = field_match.groupby(["season", "player"]).agg(
            catches=("catches", "sum"),
            run_outs=("run_outs", "sum"),
            stumpings=("stumpings", "sum"),
        ).reset_index()
    else:
        field_season = pd.DataFrame(columns=["season", "player", "catches", "run_outs", "stumpings"])

    season_stats = matches_played.merge(bat_season, on=["season", "player"], how="outer")
    season_stats = season_stats.merge(bowl_season, on=["season", "player"], how="outer")
    season_stats = season_stats.merge(field_season, on=["season", "player"], how="outer")

    season_stats["matches_played"] = season_stats["matches_played"].fillna(
        season_stats.groupby(["season", "player"])["bat_innings"].transform("first")
    )

    numeric_cols = season_stats.select_dtypes(include=[np.number]).columns
    season_stats[numeric_cols] = season_stats[numeric_cols].fillna(0)

    season_stats = season_stats.sort_values(["season", "runs"], ascending=[True, False])

    print(f"  {len(season_stats):,} player-season records created")
    return season_stats


def main():
    bbb, matches = load_data()

    bat_match = compute_batting_stats(bbb)
    bowl_match = compute_bowling_stats(bbb)
    field_match = compute_fielding_stats(bbb)

    season_stats = aggregate_to_season(bat_match, bowl_match, field_match, matches)

    output_path = PERF_DIR / "player_season_stats.csv"
    season_stats.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    print("\n=== Summary by Season ===")
    summary = season_stats.groupby("season").agg({
        "player": "count",
        "runs": "sum",
        "wickets": "sum",
        "catches": "sum"
    }).rename(columns={"player": "players"})
    print(summary)

    print("\n=== Top Run Scorers (All Time) ===")
    top_batters = season_stats.groupby("player").agg({
        "runs": "sum",
        "matches_played": "sum"
    }).sort_values("runs", ascending=False).head(10)
    print(top_batters)

    print("\n=== Verification ===")
    kohli_2016 = season_stats[(season_stats["player"] == "V Kohli") & (season_stats["season"] == 2016)]
    if not kohli_2016.empty:
        print(f"Virat Kohli 2016: {kohli_2016['runs'].values[0]} runs (expected ~973)")


if __name__ == "__main__":
    main()
