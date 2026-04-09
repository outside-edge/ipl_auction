#!/usr/bin/env python3
"""
Assemble IPL auction data from multiple sources into a consolidated dataset.

Sources:
1. IPLPlayerAuctionData.csv (Kaggle): 2013-2022
2. ipl_auction_wikipedia.xlsx: 2009-2015 (using 2009-2012 portion)
3. 2022 Auction folder: 2022
4. 2023 Auction folder: 2023
5. IPL 2024 SOLD PLAYER DATA ANALYSIS.csv: 2024
6. ipl_dataset.csv: 2025
7. Manual entry for 2008 inaugural auction

Output: data/auction/auction_all_years.csv
"""

import re

import numpy as np
import pandas as pd
from pathlib import Path
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cdist

BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
SOURCES_DIR = DATA_DIR / "sources"
AUCTION_DIR = DATA_DIR / "auction"
PERF_DIR = DATA_DIR / "perf"


def normalize_player_name(name):
    """
    Normalize common spelling variations in player names.

    Handles:
    - Mohammed/Mohammad/Mohd variations
    - Dots and extra spaces
    - Common misspellings
    """
    if pd.isna(name):
        return name

    name = str(name).strip()

    name = re.sub(r"\bMoh[ao]mm?[ae]d\b", "Mohammed", name, flags=re.IGNORECASE)
    name = re.sub(r"\bMohd\b", "Mohammed", name, flags=re.IGNORECASE)

    name = name.replace(".", " ")
    name = " ".join(name.split())

    return name

COUNTRY_PREFIXES = [
    "New Zealand ", "Trinidad and Tobago ", "Barbados ",
    "South Africa ", "India ", "England ", "Australia ",
    "Sri Lanka ", "West Indies ", "Bangladesh ", "Afghanistan ",
    "Zimbabwe ", "Pakistan ", "Ireland ", "Scotland ", "Netherlands ",
    "Kenya ", "Nepal ", "USA ", "UAE ", "Hong Kong ", "Canada ",
    "Namibia ", "Oman ", "Papua New Guinea ",
]


def clean_wikipedia_name(name):
    """
    Clean player names from Wikipedia data that may have country prefixes
    and special symbols like † or *.

    Examples:
        "New Zealand Shane Bond†" -> "Shane Bond"
        "Trinidad and Tobago Kieron Pollard†" -> "Kieron Pollard"
        "India Mohammad Kaif" -> "Mohammad Kaif"
    """
    if pd.isna(name):
        return name

    name = str(name).strip()

    for prefix in COUNTRY_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    name = name.rstrip("†*").strip()

    return name


def validate_player_name(name):
    """
    Validate player name and return issues if any.
    Returns a list of issues or empty list if valid.
    """
    issues = []
    if pd.isna(name):
        return ["Name is missing"]

    name = str(name)

    for prefix in COUNTRY_PREFIXES:
        if name.startswith(prefix):
            issues.append(f"Has country prefix: {prefix.strip()}")
            break

    if "†" in name or "*" in name:
        issues.append("Has special symbols (†, *)")

    if len(name) < 3:
        issues.append("Name too short (< 3 chars)")

    if any(c.isdigit() for c in name):
        issues.append("Name contains numbers")

    return issues


def parse_indian_price(price_str):
    """
    Parse Indian price format (₹2,00,00,000 or 20000000) to lakhs.
    Returns price in lakhs.
    """
    if pd.isna(price_str):
        return np.nan

    price_str = str(price_str).strip()

    price_str = price_str.replace("₹", "").replace(",", "").replace(" ", "")

    try:
        value = float(price_str)
        if value > 1_000_000:
            return value / 100_000
        elif value > 100:
            return value
        else:
            return value * 100
    except ValueError:
        return np.nan


def standardize_team_name(team):
    """Standardize team abbreviations."""
    if pd.isna(team):
        return "Unknown"

    team = str(team).strip().upper()

    mapping = {
        "CHENNAI SUPER KINGS": "CSK",
        "CHENNAI SUPER KINGS'": "CSK",
        "DELHI CAPITALS": "DC",
        "DELHI DAREDEVILS": "DC",
        "MUMBAI INDIANS": "MI",
        "KOLKATA KNIGHT RIDERS": "KKR",
        "ROYAL CHALLENGERS BANGALORE": "RCB",
        "ROYAL CHALLENGERS BENGALURU": "RCB",
        "RAJASTHAN ROYALS": "RR",
        "SUNRISERS HYDERABAD": "SRH",
        "KINGS XI PUNJAB": "PBKS",
        "PUNJAB KINGS": "PBKS",
        "LUCKNOW SUPER GIANTS": "LSG",
        "GUJARAT TITANS": "GT",
        "DECCAN CHARGERS": "DCH",
        "PUNE WARRIORS INDIA": "PWI",
        "RISING PUNE SUPERGIANT": "RPS",
        "RISING PUNE SUPERGIANTS": "RPS",
        "GUJARAT LIONS": "GL",
        "KOCHI TUSKERS KERALA": "KTK",
        "PK": "PBKS",
        "PBKS": "PBKS",
        "KXIP": "PBKS",
    }

    for full_name, abbr in mapping.items():
        if full_name in team or team == abbr:
            return abbr

    if len(team) <= 4:
        return team
    return team[:4]


def standardize_role(role):
    """Standardize player roles."""
    if pd.isna(role):
        return "Unknown"

    role = str(role).strip().upper()

    if "ALL" in role or "AR" == role:
        return "All-Rounder"
    elif "BOWL" in role:
        return "Bowler"
    elif "BAT" in role and "WK" not in role and "WICKET" not in role:
        return "Batsman"
    elif "WICKET" in role or "WK" in role or "KEEPER" in role:
        return "Wicket-Keeper"
    else:
        return role.title()


def standardize_nationality(nat):
    """Standardize nationality to Indian/Overseas."""
    if pd.isna(nat):
        return "Unknown"

    nat = str(nat).strip().upper()

    if "INDIA" in nat or nat == "INDIAN":
        return "Indian"
    elif nat in ["OVERSEAS", "FOREIGN"]:
        return "Overseas"
    elif nat in ["F", "O"]:
        return "Overseas"
    else:
        return "Overseas"


def load_kaggle_main():
    """Load main Kaggle dataset (2013-2022)."""
    path = SOURCES_DIR / "kaggle/iplauctiondata/IPLPlayerAuctionData.csv"
    if not path.exists():
        print(f"Warning: {path} not found")
        return pd.DataFrame()

    df = pd.read_csv(path)

    df = df.dropna(subset=["Year"])

    df_std = pd.DataFrame({
        "year": df["Year"].astype(int),
        "player_name": df["Player"].str.strip(),
        "team": df["Team"].apply(standardize_team_name),
        "final_price_lakh": df["Amount"] / 100_000,
        "role": df["Role"].apply(standardize_role),
        "nationality": df["Player Origin"].apply(standardize_nationality),
        "status": "SOLD",
        "source": "kaggle_main"
    })

    return df_std


def load_wikipedia_excel():
    """Load Wikipedia scraped Excel data (2009-2015)."""
    path = DATA_DIR / "sources/ipl_auction_wikipedia.xlsx"
    if not path.exists():
        print(f"Warning: {path} not found")
        return pd.DataFrame()

    df = pd.read_excel(path)

    def safe_float(val):
        if pd.isna(val):
            return np.nan
        try:
            return float(str(val).replace(",", "").strip())
        except ValueError:
            return np.nan

    def get_price_lakh(row):
        if pd.notna(row["auction_price"]):
            return parse_indian_price(row["auction_price"])
        elif pd.notna(row["dollar_auc_price"]):
            usd_price = safe_float(row["dollar_auc_price"])
            if pd.isna(usd_price):
                return np.nan
            year = row["year"]
            usd_to_inr = {
                2009: 48, 2010: 45, 2011: 45, 2012: 50,
                2013: 55, 2014: 60, 2015: 62
            }
            rate = usd_to_inr.get(year, 50)
            return (usd_price * rate) / 100_000
        return np.nan

    def get_base_price_lakh(row):
        if pd.notna(row["reserve_price"]):
            return parse_indian_price(row["reserve_price"])
        elif pd.notna(row["dollar_res_price"]):
            usd_price = safe_float(row["dollar_res_price"])
            if pd.isna(usd_price):
                return np.nan
            year = row["year"]
            usd_to_inr = {
                2009: 48, 2010: 45, 2011: 45, 2012: 50,
                2013: 55, 2014: 60, 2015: 62
            }
            rate = usd_to_inr.get(year, 50)
            return (usd_price * rate) / 100_000
        return np.nan

    df_std = pd.DataFrame({
        "year": df["year"].astype(int),
        "player_name": df["name"].str.strip().apply(clean_wikipedia_name),
        "team": df["this_year_team"].apply(standardize_team_name),
        "base_price_lakh": df.apply(get_base_price_lakh, axis=1),
        "final_price_lakh": df.apply(get_price_lakh, axis=1),
        "nationality": df["country"].apply(standardize_nationality),
        "status": "SOLD",
        "source": "wikipedia_excel"
    })

    df_std = df_std.dropna(subset=["year"])
    df_std["year"] = df_std["year"].astype(int)

    invalid_names = df_std[df_std["player_name"].apply(lambda x: len(validate_player_name(x)) > 0)]
    if len(invalid_names) > 0:
        print(f"  WARNING: {len(invalid_names)} names still have validation issues after cleaning")
        for _, row in invalid_names.head(5).iterrows():
            issues = validate_player_name(row["player_name"])
            print(f"    {row['year']}: {row['player_name']} - {', '.join(issues)}")

    team_corrections_2009 = {
        "Andrew Flintoff": "CSK",
        "Kevin Pietersen": "RCB",
        "JP Duminy": "MI",
        "Tyron Henderson": "RR",
        "Mashrafe Mortaza": "KKR",
        "Ravi Bopara": "PBKS",
        "Shaun Tait": "RR",
        "Owais Shah": "DC",
        "Paul Collingwood": "DC",
        "Jesse Ryder": "RCB",
        "Fidel Edwards": "DCH",
        "Jerome Taylor": "PBKS",
        "Kyle Mills": "PBKS",
        "Thilan Thushara": "CSK",
        "Dwayne Smith": "DCH",
        "Mohammad Ashraful": "KKR",
        "George Bailey": "CSK",
    }

    for player, team in team_corrections_2009.items():
        mask = (df_std["year"] == 2009) & (df_std["player_name"] == player)
        df_std.loc[mask, "team"] = team

    price_corrections = {
        (2011, "Gautam Gambhir"): 1490.0,
        (2011, "Yusuf Pathan"): 945.0,
        (2011, "Robin Uthappa"): 900.0,
        (2012, "Ravindra Jadeja"): 978.0,
    }
    for (year, player), price in price_corrections.items():
        mask = (df_std["year"] == year) & (df_std["player_name"] == player)
        df_std.loc[mask, "final_price_lakh"] = price

    return df_std


def load_auction_2021():
    """Load 2021 auction data."""
    path = DATA_DIR / "sources/auction_2021.csv"
    if not path.exists():
        print(f"Warning: {path} not found")
        return pd.DataFrame()

    df = pd.read_csv(path)

    df_std = pd.DataFrame({
        "year": 2021,
        "player_name": df["Name"].str.strip(),
        "team": df["Team"].apply(standardize_team_name),
        "base_price_lakh": df["Base_Price(Lakh)"],
        "final_price_lakh": df["Final_Price(Lakh)"],
        "role": df["Role"].apply(standardize_role),
        "nationality": df["Country"].apply(standardize_nationality),
        "status": df["Status"],
        "source": "auction_2021"
    })

    return df_std


def load_auction_2022():
    """Load 2022 auction data."""
    path = SOURCES_DIR / "2022_auction/IPL_2022_Sold_Players.csv"
    if not path.exists():
        print(f"Warning: {path} not found")
        return pd.DataFrame()

    df = pd.read_csv(path)

    df_std = pd.DataFrame({
        "year": 2022,
        "player_name": df["Players"].str.strip(),
        "team": df["Team"].apply(standardize_team_name),
        "final_price_lakh": df["Price Paid"].apply(parse_indian_price),
        "role": df["Type"].apply(standardize_role),
        "nationality": df["Nationality"].apply(standardize_nationality),
        "status": "SOLD",
        "source": "auction_2022"
    })

    return df_std


def load_auction_2023():
    """Load 2023 auction data."""
    path = SOURCES_DIR / "2023_auction/IPL_2023_Auction_Sold.csv"
    if not path.exists():
        print(f"Warning: {path} not found")
        return pd.DataFrame()

    df = pd.read_csv(path)

    df_sold = df[df["Auction_Price"] > 0].copy()

    df_std = pd.DataFrame({
        "year": 2023,
        "player_name": (df_sold["First Name"].fillna("") + " " + df_sold["Surname"].fillna("")).str.strip(),
        "team": df_sold["TEAM"].apply(standardize_team_name),
        "base_price_lakh": df_sold["Reserve_Price"],
        "final_price_lakh": df_sold["Auction_Price"],
        "role": df_sold["Specialism"].apply(standardize_role),
        "nationality": df_sold["Country"].apply(standardize_nationality),
        "status": "SOLD",
        "source": "auction_2023"
    })

    return df_std


def load_auction_2024():
    """Load 2024 auction data."""
    path = SOURCES_DIR / "kaggle/IPL-2024-SOLD-PLAYER-DATA-ANALYSIS/IPL 2024 SOLD PLAYER DATA ANALYSIS.csv"
    if not path.exists():
        print(f"Warning: {path} not found")
        return pd.DataFrame()

    df = pd.read_csv(path)

    def parse_2024_price(p):
        if pd.isna(p):
            return np.nan
        p_str = str(p).replace(",", "").replace(" ", "").strip()
        try:
            return float(p_str) / 100_000
        except ValueError:
            return np.nan

    df_std = pd.DataFrame({
        "year": 2024,
        "player_name": df["PLAYERS"].str.strip(),
        "team": df["TEAM"].apply(standardize_team_name),
        "final_price_lakh": df["PRICE"].apply(parse_2024_price),
        "role": df["TYPE"].apply(standardize_role),
        "nationality": df["NATIONALITY"].apply(standardize_nationality),
        "status": "SOLD",
        "source": "auction_2024"
    })

    return df_std


def load_auction_2025():
    """Load 2025 auction data."""
    path = SOURCES_DIR / "kaggle/IPL-Data-viz/frontend/public/ipl_dataset.csv"
    if not path.exists():
        print(f"Warning: {path} not found")
        return pd.DataFrame()

    df = pd.read_csv(path)

    df["Sold_numeric"] = pd.to_numeric(df["Sold"], errors="coerce")
    df_sold = df[df["Sold_numeric"].notna() & (df["Sold_numeric"] > 0)].copy()

    df_std = pd.DataFrame({
        "year": 2025,
        "player_name": df_sold["Players"].str.strip(),
        "team": df_sold["Team"].apply(standardize_team_name),
        "base_price_lakh": pd.to_numeric(df_sold["Base"].replace("-", np.nan), errors="coerce") * 100,
        "final_price_lakh": df_sold["Sold_numeric"] * 100,
        "role": df_sold["Type"].apply(standardize_role),
        "status": "SOLD",
        "source": "auction_2025"
    })

    return df_std


def load_auction_2026():
    """Load 2026 auction data from scraped Wikipedia source."""
    path = SOURCES_DIR / "scraped/auction_2026.csv"
    if not path.exists():
        print(f"Warning: {path} not found. Run scrape_auction_2026.py first.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    return df


def load_retained_players():
    """Load retained players data for mega auction years."""
    path = AUCTION_DIR / "retained_players.csv"
    if not path.exists():
        print(f"Warning: {path} not found")
        return pd.DataFrame()

    df = pd.read_csv(path)

    df_std = pd.DataFrame({
        "year": df["year"].astype(int),
        "player_name": df["player_name"].str.strip(),
        "team": df["team"].apply(standardize_team_name),
        "final_price_lakh": df["retention_price_lakh"],
        "status": "RETAINED",
        "source": "retained_" + df["source"].astype(str),
        "acquisition_type": "retained"
    })

    return df_std


def infer_retained_players(auction_df, perf_df):
    """
    Infer retained players for non-mega auction years based on performance data.

    For years where a player:
    1. Appears in performance data
    2. Was with the same team in the previous year
    3. Does NOT appear in auction data for that year

    They were likely retained (stayed with team between auctions).

    Mega auction years (2008, 2011, 2014, 2018, 2022, 2025) are handled separately
    with official retention data.

    Uses performance data player names directly since auction data may not have
    matching entries for players who were retained between auctions.
    """
    mega_auction_years = {2008, 2011, 2014, 2018, 2022, 2025}
    perf_years = set(perf_df["season"].unique())
    non_mega_years = [y for y in sorted(perf_years) if y not in mega_auction_years and y > 2008]

    existing_player_years = set()
    for _, row in auction_df.iterrows():
        key = (int(row["year"]), row["player_name"].lower() if pd.notna(row["player_name"]) else "")
        existing_player_years.add(key)

    perf_df = perf_df.copy()
    perf_df["player_norm"] = perf_df["player"].str.lower().str.strip()

    inferred = []

    for year in non_mega_years:
        if year not in perf_years:
            continue
        prev_year = year - 1
        if prev_year not in perf_years:
            continue

        perf_year = perf_df[perf_df["season"] == year][["player", "player_norm", "team"]].drop_duplicates()
        perf_prev = perf_df[perf_df["season"] == prev_year][["player", "player_norm", "team"]].drop_duplicates()

        prev_team_map = dict(zip(perf_prev["player_norm"], perf_prev["team"]))

        for _, row in perf_year.iterrows():
            player = row["player"]
            player_norm = row["player_norm"]
            team = row["team"]

            if (year, player_norm) in existing_player_years:
                continue

            prev_team = prev_team_map.get(player_norm)
            if prev_team is None:
                continue

            prev_team_std = standardize_team_name(prev_team)
            curr_team_std = standardize_team_name(team)

            if prev_team_std == curr_team_std:
                inferred.append({
                    "year": year,
                    "player_name": player,
                    "team": curr_team_std,
                    "final_price_lakh": np.nan,
                    "status": "RETAINED",
                    "source": "inferred_retention",
                    "acquisition_type": "retained"
                })
                existing_player_years.add((year, player_norm))

    return pd.DataFrame(inferred)


def create_2008_data():
    """
    Create 2008 inaugural auction data from known sources.
    The 2008 IPL auction was held on Feb 20, 2008.
    Data compiled from multiple historical sources.
    """
    data_2008 = [
        {"player_name": "MS Dhoni", "team": "CSK", "final_price_lakh": 950, "role": "Wicket-Keeper", "nationality": "Indian"},
        {"player_name": "Andrew Symonds", "team": "DCH", "final_price_lakh": 650, "role": "All-Rounder", "nationality": "Overseas"},
        {"player_name": "Sanath Jayasuriya", "team": "MI", "final_price_lakh": 575, "role": "All-Rounder", "nationality": "Overseas"},
        {"player_name": "Ishant Sharma", "team": "KKR", "final_price_lakh": 550, "role": "Bowler", "nationality": "Indian"},
        {"player_name": "Mahela Jayawardene", "team": "DC", "final_price_lakh": 525, "role": "Batsman", "nationality": "Overseas"},
        {"player_name": "RP Singh", "team": "DCH", "final_price_lakh": 375, "role": "Bowler", "nationality": "Indian"},
        {"player_name": "Irfan Pathan", "team": "PBKS", "final_price_lakh": 350, "role": "All-Rounder", "nationality": "Indian"},
        {"player_name": "Robin Uthappa", "team": "MI", "final_price_lakh": 300, "role": "Batsman", "nationality": "Indian"},
        {"player_name": "S Sreesanth", "team": "PBKS", "final_price_lakh": 275, "role": "Bowler", "nationality": "Indian"},
        {"player_name": "Gautam Gambhir", "team": "DC", "final_price_lakh": 275, "role": "Batsman", "nationality": "Indian"},
        {"player_name": "Yusuf Pathan", "team": "RR", "final_price_lakh": 215, "role": "All-Rounder", "nationality": "Indian"},
        {"player_name": "Praveen Kumar", "team": "RCB", "final_price_lakh": 200, "role": "Bowler", "nationality": "Indian"},
        {"player_name": "Shikhar Dhawan", "team": "DC", "final_price_lakh": 200, "role": "Batsman", "nationality": "Indian"},
        {"player_name": "Munaf Patel", "team": "RR", "final_price_lakh": 175, "role": "Bowler", "nationality": "Indian"},
        {"player_name": "Harbhajan Singh", "team": "MI", "final_price_lakh": 175, "role": "Bowler", "nationality": "Indian"},
        {"player_name": "Piyush Chawla", "team": "PBKS", "final_price_lakh": 175, "role": "Bowler", "nationality": "Indian"},
        {"player_name": "Pragyan Ojha", "team": "DCH", "final_price_lakh": 175, "role": "Bowler", "nationality": "Indian"},
        {"player_name": "Ravindra Jadeja", "team": "RR", "final_price_lakh": 175, "role": "All-Rounder", "nationality": "Indian"},
        {"player_name": "Daniel Vettori", "team": "DC", "final_price_lakh": 150, "role": "All-Rounder", "nationality": "Overseas"},
        {"player_name": "Mohammad Kaif", "team": "RR", "final_price_lakh": 150, "role": "Batsman", "nationality": "Indian"},
        {"player_name": "Rohit Sharma", "team": "DCH", "final_price_lakh": 148, "role": "Batsman", "nationality": "Indian"},
        {"player_name": "Glenn McGrath", "team": "DC", "final_price_lakh": 147.5, "role": "Bowler", "nationality": "Overseas"},
        {"player_name": "Graeme Smith", "team": "RR", "final_price_lakh": 125, "role": "Batsman", "nationality": "Overseas"},
        {"player_name": "Sourav Ganguly", "team": "KKR", "final_price_lakh": 125, "role": "Batsman", "nationality": "Indian"},
        {"player_name": "Rahul Dravid", "team": "RCB", "final_price_lakh": 125, "role": "Batsman", "nationality": "Indian"},
        {"player_name": "Jacques Kallis", "team": "RCB", "final_price_lakh": 125, "role": "All-Rounder", "nationality": "Overseas"},
        {"player_name": "Anil Kumble", "team": "RCB", "final_price_lakh": 125, "role": "Bowler", "nationality": "Indian"},
        {"player_name": "VVS Laxman", "team": "DCH", "final_price_lakh": 125, "role": "Batsman", "nationality": "Indian"},
        {"player_name": "Virender Sehwag", "team": "DC", "final_price_lakh": 125, "role": "Batsman", "nationality": "Indian"},
        {"player_name": "Shane Warne", "team": "RR", "final_price_lakh": 125, "role": "All-Rounder", "nationality": "Overseas"},
        {"player_name": "Brett Lee", "team": "KKR", "final_price_lakh": 125, "role": "Bowler", "nationality": "Overseas"},
        {"player_name": "Matthew Hayden", "team": "CSK", "final_price_lakh": 125, "role": "Batsman", "nationality": "Overseas"},
        {"player_name": "Muttiah Muralitharan", "team": "CSK", "final_price_lakh": 125, "role": "Bowler", "nationality": "Overseas"},
        {"player_name": "Stephen Fleming", "team": "CSK", "final_price_lakh": 125, "role": "Batsman", "nationality": "Overseas"},
        {"player_name": "Michael Hussey", "team": "CSK", "final_price_lakh": 125, "role": "All-Rounder", "nationality": "Overseas"},
        {"player_name": "Adam Gilchrist", "team": "DCH", "final_price_lakh": 125, "role": "Wicket-Keeper", "nationality": "Overseas"},
        {"player_name": "Sachin Tendulkar", "team": "MI", "final_price_lakh": 125, "role": "Batsman", "nationality": "Indian"},
        {"player_name": "Lasith Malinga", "team": "MI", "final_price_lakh": 40, "role": "Bowler", "nationality": "Overseas"},
        {"player_name": "Shaun Pollock", "team": "MI", "final_price_lakh": 125, "role": "All-Rounder", "nationality": "Overseas"},
        {"player_name": "Yuvraj Singh", "team": "PBKS", "final_price_lakh": 125, "role": "All-Rounder", "nationality": "Indian"},
        {"player_name": "Chris Gayle", "team": "KKR", "final_price_lakh": 125, "role": "Batsman", "nationality": "Overseas"},
        {"player_name": "Ricky Ponting", "team": "KKR", "final_price_lakh": 40, "role": "Batsman", "nationality": "Overseas"},
        {"player_name": "Brendon McCullum", "team": "KKR", "final_price_lakh": 30, "role": "Wicket-Keeper", "nationality": "Overseas"},
        {"player_name": "Zaheer Khan", "team": "RCB", "final_price_lakh": 200, "role": "Bowler", "nationality": "Indian"},
        {"player_name": "Suresh Raina", "team": "CSK", "final_price_lakh": 125, "role": "Batsman", "nationality": "Indian"},
        {"player_name": "Parthiv Patel", "team": "CSK", "final_price_lakh": 80, "role": "Wicket-Keeper", "nationality": "Indian"},
        {"player_name": "Dinesh Karthik", "team": "DC", "final_price_lakh": 100, "role": "Wicket-Keeper", "nationality": "Indian"},
        {"player_name": "Ajit Agarkar", "team": "KKR", "final_price_lakh": 100, "role": "All-Rounder", "nationality": "Indian"},
        {"player_name": "Ashish Nehra", "team": "MI", "final_price_lakh": 100, "role": "Bowler", "nationality": "Indian"},
        {"player_name": "Shaun Marsh", "team": "PBKS", "final_price_lakh": 60, "role": "Batsman", "nationality": "Overseas"},
        {"player_name": "Shane Watson", "team": "RR", "final_price_lakh": 62.5, "role": "All-Rounder", "nationality": "Overseas"},
        {"player_name": "Cameron White", "team": "RCB", "final_price_lakh": 75, "role": "All-Rounder", "nationality": "Overseas"},
        {"player_name": "Wasim Jaffer", "team": "RCB", "final_price_lakh": 100, "role": "Batsman", "nationality": "Indian"},
        {"player_name": "Virat Kohli", "team": "RCB", "final_price_lakh": 30, "role": "Batsman", "nationality": "Indian"},
        {"player_name": "Ross Taylor", "team": "RCB", "final_price_lakh": 50, "role": "Batsman", "nationality": "Overseas"},
        {"player_name": "David Hussey", "team": "KKR", "final_price_lakh": 62.5, "role": "All-Rounder", "nationality": "Overseas"},
        {"player_name": "Murali Kartik", "team": "KKR", "final_price_lakh": 125, "role": "Bowler", "nationality": "Indian"},
        {"player_name": "Wriddhiman Saha", "team": "KKR", "final_price_lakh": 10, "role": "Wicket-Keeper", "nationality": "Indian"},
        {"player_name": "Albie Morkel", "team": "CSK", "final_price_lakh": 30, "role": "All-Rounder", "nationality": "Overseas"},
        {"player_name": "Manish Pandey", "team": "MI", "final_price_lakh": 12.5, "role": "Batsman", "nationality": "Indian"},
        {"player_name": "Ambati Rayudu", "team": "MI", "final_price_lakh": 30, "role": "Batsman", "nationality": "Indian"},
        {"player_name": "Joginder Sharma", "team": "CSK", "final_price_lakh": 125, "role": "All-Rounder", "nationality": "Indian"},
        {"player_name": "Saurabh Tiwary", "team": "DCH", "final_price_lakh": 20, "role": "Batsman", "nationality": "Indian"},
        {"player_name": "Scott Styris", "team": "DCH", "final_price_lakh": 100, "role": "All-Rounder", "nationality": "Overseas"},
        {"player_name": "Subramaniam Badrinath", "team": "CSK", "final_price_lakh": 40, "role": "Batsman", "nationality": "Indian"},
        {"player_name": "James Hopes", "team": "PBKS", "final_price_lakh": 75, "role": "All-Rounder", "nationality": "Overseas"},
        {"player_name": "VRV Singh", "team": "PBKS", "final_price_lakh": 125, "role": "Bowler", "nationality": "Indian"},
        {"player_name": "Ramesh Powar", "team": "PBKS", "final_price_lakh": 100, "role": "Bowler", "nationality": "Indian"},
        {"player_name": "Simon Katich", "team": "DCH", "final_price_lakh": 75, "role": "Batsman", "nationality": "Overseas"},
        {"player_name": "Herschelle Gibbs", "team": "DCH", "final_price_lakh": 75, "role": "Batsman", "nationality": "Overseas"},
        {"player_name": "Shahid Afridi", "team": "DCH", "final_price_lakh": 27.5, "role": "All-Rounder", "nationality": "Overseas"},
        {"player_name": "Kumar Sangakkara", "team": "PBKS", "final_price_lakh": 125, "role": "Wicket-Keeper", "nationality": "Overseas"},
        {"player_name": "Shoaib Malik", "team": "DC", "final_price_lakh": 125, "role": "All-Rounder", "nationality": "Overseas"},
        {"player_name": "Tillakaratne Dilshan", "team": "DC", "final_price_lakh": 40, "role": "Batsman", "nationality": "Overseas"},
        {"player_name": "AB de Villiers", "team": "DC", "final_price_lakh": 100, "role": "Batsman", "nationality": "Overseas"},
        {"player_name": "Misbah-ul-Haq", "team": "RCB", "final_price_lakh": 50, "role": "Batsman", "nationality": "Overseas"},
        {"player_name": "Dale Steyn", "team": "RCB", "final_price_lakh": 57.5, "role": "Bowler", "nationality": "Overseas"},
        {"player_name": "Mark Boucher", "team": "RCB", "final_price_lakh": 75, "role": "Wicket-Keeper", "nationality": "Overseas"},
        {"player_name": "Mohammad Asif", "team": "DC", "final_price_lakh": 125, "role": "Bowler", "nationality": "Overseas"},
        {"player_name": "Sohail Tanvir", "team": "RR", "final_price_lakh": 50, "role": "Bowler", "nationality": "Overseas"},
    ]

    df = pd.DataFrame(data_2008)
    df["year"] = 2008
    df["status"] = "SOLD"
    df["source"] = "manual_2008"

    return df


def deduplicate_by_source_priority(df):
    """
    Remove duplicates preferring certain sources over others.
    Priority: auction_YYYY > kaggle_main > wikipedia_excel
    """
    source_priority = {
        "auction_2021": 1,
        "auction_2022": 1,
        "auction_2023": 1,
        "auction_2024": 1,
        "auction_2025": 1,
        "wikipedia_2026": 1,
        "manual_2008": 1,
        "kaggle_main": 2,
        "wikipedia_excel": 3,
    }

    df["source_priority"] = df["source"].map(source_priority).fillna(4)

    df = df.sort_values(["year", "player_name", "source_priority"])
    df = df.drop_duplicates(subset=["year", "player_name"], keep="first")
    df = df.drop(columns=["source_priority"])

    return df


def deduplicate_fuzzy_same_year(df, threshold=85):
    """
    Remove near-duplicate player names within the same year.

    Uses fuzzy matching to identify spelling variations like:
    - "Nicholas Pooran" vs "Nicolas Pooran"
    - "Mohammed Azharuddeen" vs "Mohammed Azharudeen"
    - "KC Cariappa" vs "K C Cariappa"

    Keeps the record from the higher-priority source, or the first one if same source.
    """
    source_priority = {
        "auction_2021": 1,
        "auction_2022": 1,
        "auction_2023": 1,
        "auction_2024": 1,
        "auction_2025": 1,
        "wikipedia_2026": 1,
        "manual_2008": 1,
        "kaggle_main": 2,
        "wikipedia_excel": 3,
    }

    def normalize_for_match(name):
        if pd.isna(name):
            return ""
        name = str(name).strip().lower()
        name = re.sub(r"\bMoh[ao]mm?[ae]d\b", "mohammed", name, flags=re.IGNORECASE)
        name = re.sub(r"\bMohd\b", "mohammed", name, flags=re.IGNORECASE)
        name = name.replace(".", "").replace("-", "").replace("'", "").replace(" ", "")
        return name

    def normalize_for_fuzzy(name):
        if pd.isna(name):
            return ""
        name = str(name).strip().lower()
        name = re.sub(r"\bMoh[ao]mm?[ae]d\b", "mohammed", name, flags=re.IGNORECASE)
        name = re.sub(r"\bMohd\b", "mohammed", name, flags=re.IGNORECASE)
        name = name.replace(".", " ").replace("-", " ").replace("'", "")
        name = " ".join(name.split())
        return name

    def get_last_name(name):
        parts = name.split()
        return parts[-1] if parts else ""

    rows_to_drop = set()

    for year in df["year"].unique():
        year_df = df[df["year"] == year].copy()
        if len(year_df) < 2:
            continue

        names = year_df["player_name"].tolist()
        indices = year_df.index.tolist()
        names_norm = [normalize_for_match(n) for n in names]
        names_fuzzy = [normalize_for_fuzzy(n) for n in names]

        scores = cdist(names_fuzzy, names_fuzzy, scorer=JaroWinkler.normalized_similarity, workers=-1)

        for i in range(len(names)):
            if indices[i] in rows_to_drop:
                continue
            for j in range(i + 1, len(names)):
                if indices[j] in rows_to_drop:
                    continue

                is_duplicate = False

                if names_norm[i] == names_norm[j]:
                    is_duplicate = True
                else:
                    last_i = get_last_name(names_fuzzy[i])
                    last_j = get_last_name(names_fuzzy[j])
                    first_i = names_fuzzy[i].split()[0] if names_fuzzy[i].split() else ""
                    first_j = names_fuzzy[j].split()[0] if names_fuzzy[j].split() else ""

                    last_match = last_i == last_j or JaroWinkler.normalized_similarity(last_i, last_j) >= 0.90

                    if last_match:
                        first_equiv = (
                            first_i == first_j
                            or NAME_EQUIVALENTS.get(first_i) == first_j
                            or NAME_EQUIVALENTS.get(first_j) == first_i
                        )
                        if first_equiv:
                            is_duplicate = True
                        elif scores[i][j] >= 0.88:
                            is_duplicate = True

                if is_duplicate:
                    src_i = year_df.loc[indices[i], "source"]
                    src_j = year_df.loc[indices[j], "source"]
                    pri_i = source_priority.get(src_i, 4)
                    pri_j = source_priority.get(src_j, 4)

                    if pri_i < pri_j:
                        rows_to_drop.add(indices[j])
                    elif pri_j < pri_i:
                        rows_to_drop.add(indices[i])
                    else:
                        rows_to_drop.add(indices[j])

    if rows_to_drop:
        dropped_names = df.loc[list(rows_to_drop), ["year", "player_name", "source"]]
        print(f"  Removing {len(rows_to_drop)} fuzzy duplicates:")
        for _, row in dropped_names.iterrows():
            print(f"    {row['year']}: {row['player_name']} ({row['source']})")

    df = df.drop(index=list(rows_to_drop))
    return df


KNOWN_SAME_PLAYER = [
    {"Ben Stokes", "Benjamin Stokes"},
    {"Rashid Khan", "Rashid Khan Arman"},
    {"Nicholas Pooran", "Nicolas Pooran", "N Pooran"},
    {"Chris Morris", "Christopher Morris"},
    {"Thisara Perera", "Thissara Perera"},
    {"N Tilak Varma", "Tilak Varma"},
    {"Hari Nishanth", "C Hari Nishaanth"},
    {"Suyash Prabhudessai", "Suyash Prabhudesai"},
    {"Mohammed Azharuddeen", "Mohammed Azharudeen"},
    {"Dan Christian", "Daniel Christian"},
]

NAME_EQUIVALENTS = {
    "chris": "christopher",
    "christopher": "chris",
    "ben": "benjamin",
    "benjamin": "ben",
    "dan": "daniel",
    "daniel": "dan",
    "nick": "nicholas",
    "nicholas": "nick",
    "mike": "michael",
    "michael": "mike",
}


def build_player_registry(df):
    """
    Build a player registry from auction data and assign player_id.

    Returns:
    - registry_df: DataFrame with player_id, canonical_name, aliases
    - name_to_id: dict mapping each name to its player_id
    """
    from rapidfuzz.process import cdist as rf_cdist

    def normalize_for_clustering(name):
        if pd.isna(name):
            return ""
        name = str(name).strip().lower()
        name = re.sub(r"\bMoh[ao]mm?[ae]d\b", "mohammed", name, flags=re.IGNORECASE)
        name = re.sub(r"\bMohd\b", "mohammed", name, flags=re.IGNORECASE)
        name = name.replace(".", " ").replace("-", " ").replace("'", "")
        name = " ".join(name.split())
        return name

    def get_last_name(name):
        parts = name.split()
        return parts[-1] if parts else ""

    def get_first_name(name):
        parts = name.split()
        return parts[0] if parts else ""

    def names_are_similar(name1, name2, norm1, norm2, score):
        """Check if two names represent the same player."""
        if score < 0.85:
            return False

        last1 = get_last_name(norm1)
        last2 = get_last_name(norm2)

        if len(last1) >= 3 and len(last2) >= 3:
            if last1 != last2:
                return False

        first1 = get_first_name(norm1)
        first2 = get_first_name(norm2)
        if first1 and first2 and first1[0] != first2[0]:
            return False

        return True

    def cluster_names(names, threshold=85):
        if not names:
            return []
        names = list(set(names))
        names_norm = [normalize_for_clustering(n) for n in names]
        scores = rf_cdist(names_norm, names_norm, scorer=JaroWinkler.normalized_similarity, workers=-1)

        visited = set()
        clusters = []

        for i in range(len(names)):
            if i in visited:
                continue
            cluster = {names[i]}
            visited.add(i)

            for j in range(len(names)):
                if j in visited:
                    continue
                if names_are_similar(names[i], names[j], names_norm[i], names_norm[j], scores[i][j]):
                    cluster.add(names[j])
                    visited.add(j)

            clusters.append(cluster)
        return clusters

    def select_canonical(names, name_counts):
        if not names:
            return ""
        scored = []
        for name in names:
            count = name_counts.get(name, 0)
            length = len(name)
            scored.append((count, length, name))
        scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
        return scored[0][2]

    name_counts = df["player_name"].value_counts().to_dict()
    unique_names = list(df["player_name"].unique())

    clusters = cluster_names(unique_names)

    name_to_cluster_idx = {}
    for idx, cluster in enumerate(clusters):
        for name in cluster:
            name_to_cluster_idx[name] = idx

    for known_set in KNOWN_SAME_PLAYER:
        matching_indices = set()
        for name in known_set:
            if name in name_to_cluster_idx:
                matching_indices.add(name_to_cluster_idx[name])

        if len(matching_indices) > 1:
            indices_list = sorted(matching_indices)
            target_idx = indices_list[0]
            for idx in indices_list[1:]:
                clusters[target_idx] = clusters[target_idx].union(clusters[idx])
                clusters[idx] = set()
                for name in clusters[target_idx]:
                    name_to_cluster_idx[name] = target_idx

    clusters = [c for c in clusters if len(c) > 0]

    registry = []
    name_to_id = {}

    for i, cluster in enumerate(sorted(clusters, key=lambda c: min(c))):
        player_id = f"P{i+1:04d}"
        canonical = select_canonical(cluster, name_counts)
        aliases = "|".join(sorted(cluster))

        registry.append({
            "player_id": player_id,
            "canonical_name": canonical,
            "aliases": aliases
        })

        for name in cluster:
            name_to_id[name] = player_id

    registry_df = pd.DataFrame(registry)
    return registry_df, name_to_id


def assign_player_ids(df, name_to_id):
    """Assign player_id to each record based on player name."""
    df["player_id"] = df["player_name"].map(name_to_id)
    return df


def main():
    print("Loading data sources...")

    dfs = []

    print("  Loading 2008 manual data...")
    df_2008 = create_2008_data()
    if not df_2008.empty:
        dfs.append(df_2008)
        print(f"    2008: {len(df_2008)} records")

    print("  Loading Wikipedia Excel (2009-2012)...")
    df_wiki = load_wikipedia_excel()
    if not df_wiki.empty:
        df_wiki_2009_2012 = df_wiki[df_wiki["year"].between(2009, 2012)]
        dfs.append(df_wiki_2009_2012)
        print(f"    2009-2012: {len(df_wiki_2009_2012)} records")

    print("  Loading Kaggle main dataset (2013-2022)...")
    df_kaggle = load_kaggle_main()
    if not df_kaggle.empty:
        dfs.append(df_kaggle)
        print(f"    2013-2022: {len(df_kaggle)} records")

    print("  Loading 2021 auction data...")
    df_2021 = load_auction_2021()
    if not df_2021.empty:
        dfs.append(df_2021)
        print(f"    2021: {len(df_2021)} records")

    print("  Loading 2022 auction data...")
    df_2022 = load_auction_2022()
    if not df_2022.empty:
        dfs.append(df_2022)
        print(f"    2022: {len(df_2022)} records")

    print("  Loading 2023 auction data...")
    df_2023 = load_auction_2023()
    if not df_2023.empty:
        dfs.append(df_2023)
        print(f"    2023: {len(df_2023)} records")

    print("  Loading 2024 auction data...")
    df_2024 = load_auction_2024()
    if not df_2024.empty:
        dfs.append(df_2024)
        print(f"    2024: {len(df_2024)} records")

    print("  Loading 2025 auction data...")
    df_2025 = load_auction_2025()
    if not df_2025.empty:
        dfs.append(df_2025)
        print(f"    2025: {len(df_2025)} records")

    print("  Loading 2026 auction data...")
    df_2026 = load_auction_2026()
    if not df_2026.empty:
        dfs.append(df_2026)
        print(f"    2026: {len(df_2026)} records")

    print("\nCombining all auction sources...")
    df_all = pd.concat(dfs, ignore_index=True)
    df_all["acquisition_type"] = "auction"
    print(f"  Total auction records before deduplication: {len(df_all)} records")

    print("\nNormalizing player names...")
    df_all["player_name"] = df_all["player_name"].apply(normalize_player_name)

    print("\nDeduplicating by source priority (exact matches)...")
    df_all = deduplicate_by_source_priority(df_all)
    print(f"  Total after exact deduplication: {len(df_all)} records")

    print("\nDeduplicating fuzzy same-year duplicates...")
    df_all = deduplicate_fuzzy_same_year(df_all, threshold=88)
    print(f"  Total after fuzzy deduplication: {len(df_all)} records")

    print("\nLoading retained players for mega auction years...")
    df_retained = load_retained_players()
    if not df_retained.empty:
        df_retained["player_name"] = df_retained["player_name"].apply(normalize_player_name)
        auction_player_years = set(zip(df_all["year"], df_all["player_name"].str.lower()))
        df_retained["_key"] = list(zip(df_retained["year"], df_retained["player_name"].str.lower()))
        df_retained_new = df_retained[~df_retained["_key"].isin(auction_player_years)].drop(columns=["_key"])
        print(f"  Retained players loaded: {len(df_retained)}")
        print(f"  New retained records (not in auction): {len(df_retained_new)}")
        df_all = pd.concat([df_all, df_retained_new], ignore_index=True)
        print(f"  Total after adding retained players: {len(df_all)} records")

    print("\nInferring retained players for non-mega auction years...")
    perf_path = PERF_DIR / "player_season_stats.csv"
    if perf_path.exists():
        perf_df = pd.read_csv(perf_path)
        df_inferred = infer_retained_players(df_all, perf_df)
        if not df_inferred.empty:
            df_inferred["player_name"] = df_inferred["player_name"].apply(normalize_player_name)
            existing_keys = set(zip(df_all["year"], df_all["player_name"].str.lower()))
            df_inferred["_key"] = list(zip(df_inferred["year"], df_inferred["player_name"].str.lower()))
            df_inferred_new = df_inferred[~df_inferred["_key"].isin(existing_keys)].drop(columns=["_key"])
            print(f"  Inferred retained players: {len(df_inferred)}")
            print(f"  New inferred records: {len(df_inferred_new)}")
            df_all = pd.concat([df_all, df_inferred_new], ignore_index=True)
            print(f"  Total after adding inferred retentions: {len(df_all)} records")
    else:
        print("  Skipping - player_season_stats.csv not found")

    print("\nBuilding player registry and assigning player IDs...")
    registry_df, name_to_id = build_player_registry(df_all)
    df_all = assign_player_ids(df_all, name_to_id)
    print(f"  Unique players: {len(registry_df)}")

    registry_path = AUCTION_DIR / "player_registry.csv"
    registry_df.to_csv(registry_path, index=False)
    print(f"  Saved registry to {registry_path}")

    df_all = df_all.sort_values(["year", "final_price_lakh"], ascending=[True, False])

    cols = [
        "year", "player_id", "player_name", "team", "base_price_lakh", "final_price_lakh",
        "role", "nationality", "status", "acquisition_type", "source"
    ]
    for col in cols:
        if col not in df_all.columns:
            df_all[col] = np.nan
    df_all = df_all[cols]

    output_path = AUCTION_DIR / "auction_all_years.csv"
    df_all.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")

    print("\n=== Summary by Year ===")
    df_all["final_price_lakh"] = pd.to_numeric(df_all["final_price_lakh"], errors="coerce")
    summary = df_all.groupby("year").agg(
        players=("player_name", "count"),
        total_spent_lakh=("final_price_lakh", "sum"),
        max_price_lakh=("final_price_lakh", "max")
    ).round(0)
    print(summary)

    print("\n=== Verification: Top 5 deals each year ===")
    for year in sorted(df_all["year"].unique()):
        year_df = df_all[df_all["year"] == year].head(3)
        print(f"\n{year}:")
        for _, row in year_df.iterrows():
            print(f"  {row['player_name']} ({row['player_id']}): ₹{row['final_price_lakh']:.0f}L to {row['team']}")

    print("\n=== Sample multi-auction players ===")
    player_counts = df_all.groupby("player_id").size()
    multi_auction = player_counts[player_counts > 3].index.tolist()[:10]
    for pid in multi_auction:
        player_records = df_all[df_all["player_id"] == pid]
        canonical = registry_df[registry_df["player_id"] == pid]["canonical_name"].iloc[0]
        years = sorted(player_records["year"].unique())
        print(f"  {pid} ({canonical}): {len(years)} auctions - years {min(years)}-{max(years)}")


if __name__ == "__main__":
    main()
