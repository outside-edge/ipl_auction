#!/usr/bin/env python3
"""
Scrape IPL 2026 auction data from Wikipedia.

The ESPN Cricinfo page blocks automated requests, so we use Wikipedia
as the data source for the 2026 mini auction held December 16, 2025.

Output: data/raw/auction_2026.csv
"""

import subprocess
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"


def fetch_wikipedia_page():
    """Fetch Wikipedia page using curl."""
    url = "https://en.wikipedia.org/wiki/List_of_2026_Indian_Premier_League_personnel_changes"
    result = subprocess.run(
        [
            "curl",
            "-s",
            "-L",
            url,
            "-H",
            "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ],
        capture_output=True,
        text=True,
    )
    return result.stdout


def parse_auction_tables(html):
    """Parse auction tables from Wikipedia HTML."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("Installing beautifulsoup4...")
        subprocess.run(["pip", "install", "beautifulsoup4"], check=True)
        from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table", class_="wikitable")

    auction_players = []
    team_keywords = [
        "Super Kings",
        "Capitals",
        "Indians",
        "Knight Riders",
        "Challengers",
        "Royals",
        "Sunrisers",
        "Titans",
        "Giants",
        "Kings",
    ]

    for table_idx in range(13, 23):
        if table_idx >= len(tables):
            break
        table = tables[table_idx]
        rows = table.find_all("tr")

        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) < 4:
                continue

            cell_texts = [c.get_text(strip=True) for c in cells]

            if not cell_texts[1] or cell_texts[1] == "—N/a":
                continue

            try:
                name = cell_texts[1]
                country = cell_texts[2]

                team = None
                price_lakh = None
                role = None

                for i, text in enumerate(cell_texts):
                    if any(t in text for t in team_keywords):
                        if team is None:
                            team = text
                            for j in range(max(0, i - 3), min(len(cell_texts), i + 4)):
                                if cell_texts[j].isdigit():
                                    if j > i:
                                        price_lakh = int(cell_texts[j])
                                        break
                            break
                    if text in ["Batter", "Bowler", "All-rounder", "Wicket-keeper"]:
                        role = text

                if price_lakh is None:
                    for text in reversed(cell_texts[:-1]):
                        if text.isdigit():
                            price_lakh = int(text)
                            break

                if team and name:
                    auction_players.append(
                        {
                            "name": name,
                            "team": team,
                            "country": country,
                            "price_lakh": price_lakh,
                            "role": role,
                        }
                    )
            except Exception as e:
                print(f"Error parsing row: {e}")
                continue

    seen_names = set()
    unique_players = []
    for p in auction_players:
        if p["name"] not in seen_names:
            seen_names.add(p["name"])
            unique_players.append(p)

    return unique_players


def standardize_team_name(team):
    """Standardize team names to abbreviations."""
    mapping = {
        "Chennai Super Kings": "CSK",
        "Delhi Capitals": "DC",
        "Mumbai Indians": "MI",
        "Kolkata Knight Riders": "KKR",
        "Royal Challengers Bengaluru": "RCB",
        "Royal Challengers Bangalore": "RCB",
        "Rajasthan Royals": "RR",
        "Sunrisers Hyderabad": "SRH",
        "Punjab Kings": "PBKS",
        "Lucknow Super Giants": "LSG",
        "Gujarat Titans": "GT",
    }
    for full_name, abbr in mapping.items():
        if full_name in team:
            return abbr
    return team[:3].upper()


def standardize_role(role):
    """Map Wikipedia roles to our standardized roles."""
    if role is None:
        return "Unknown"
    role_map = {
        "Batter": "Batsman",
        "Bowler": "Bowler",
        "All-rounder": "All-Rounder",
        "Wicket-keeper": "Wicket-Keeper",
    }
    return role_map.get(role, "Unknown")


def standardize_nationality(country):
    """Standardize nationality to Indian/Overseas."""
    if country and "India" in country:
        return "Indian"
    return "Overseas"


def main():
    print("Fetching Wikipedia page...")
    html = fetch_wikipedia_page()

    if not html or "Access Denied" in html:
        print("Failed to fetch Wikipedia page")
        return

    print("Parsing auction tables...")
    players = parse_auction_tables(html)
    print(f"Found {len(players)} auction players")

    if len(players) != 77:
        print(f"Warning: Expected 77 players, got {len(players)}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(players)
    df["year"] = 2026
    df["team"] = df["team"].apply(standardize_team_name)
    df["role"] = df["role"].apply(standardize_role)
    df["nationality"] = df["country"].apply(standardize_nationality)
    df["status"] = "SOLD"
    df["source"] = "wikipedia_2026"

    df = df.rename(
        columns={
            "name": "player_name",
            "price_lakh": "final_price_lakh",
        }
    )

    output_cols = [
        "year",
        "player_name",
        "team",
        "final_price_lakh",
        "role",
        "nationality",
        "status",
        "source",
    ]
    df = df[output_cols]

    output_path = RAW_DIR / "auction_2026.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    print("\n=== Summary ===")
    print(f"Total players: {len(df)}")
    print(f"Total spent: ₹{df['final_price_lakh'].sum()/100:.2f} Cr")
    print(f"Top buy: {df.loc[df['final_price_lakh'].idxmax(), 'player_name']} "
          f"at ₹{df['final_price_lakh'].max()/100:.2f} Cr")

    print("\n=== Team-wise spending (Cr) ===")
    team_spending = df.groupby("team")["final_price_lakh"].sum() / 100
    print(team_spending.sort_values(ascending=False).to_string())

    print("\n=== Top 10 buys ===")
    top10 = df.nlargest(10, "final_price_lakh")[
        ["player_name", "team", "final_price_lakh", "nationality"]
    ].copy()
    top10["price_cr"] = top10["final_price_lakh"] / 100
    for _, row in top10.iterrows():
        print(f"  {row['player_name']}: ₹{row['price_cr']:.2f} Cr to {row['team']}")


if __name__ == "__main__":
    main()
