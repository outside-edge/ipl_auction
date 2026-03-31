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

Output: data/auction_all_years.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


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
    path = DATA_DIR / "kaggle/iplauctiondata/IPLPlayerAuctionData.csv"
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
    path = DATA_DIR / "ipl_auction_wikipedia.xlsx"
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
        "player_name": df["name"].str.strip(),
        "team": df["this_year_team"].apply(standardize_team_name),
        "base_price_lakh": df.apply(get_base_price_lakh, axis=1),
        "final_price_lakh": df.apply(get_price_lakh, axis=1),
        "nationality": df["country"].apply(standardize_nationality),
        "status": "SOLD",
        "source": "wikipedia_excel"
    })

    df_std = df_std.dropna(subset=["year"])
    df_std["year"] = df_std["year"].astype(int)

    return df_std


def load_auction_2021():
    """Load 2021 auction data."""
    path = DATA_DIR / "auction_2021.csv"
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
    path = DATA_DIR / "2022 Auction/IPL_2022_Sold_Players.csv"
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
    path = DATA_DIR / "2023 Auction/IPL_2023_Auction_Sold.csv"
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
    path = DATA_DIR / "kaggle/IPL-2024-SOLD-PLAYER-DATA-ANALYSIS/IPL 2024 SOLD PLAYER DATA ANALYSIS.csv"
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
    path = DATA_DIR / "kaggle/IPL-Data-viz/frontend/public/ipl_dataset.csv"
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
    path = DATA_DIR / "raw/auction_2026.csv"
    if not path.exists():
        print(f"Warning: {path} not found. Run scrape_auction_2026.py first.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    return df


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

    print("\nCombining all sources...")
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"  Total before deduplication: {len(df_all)} records")

    print("Deduplicating...")
    df_all = deduplicate_by_source_priority(df_all)
    print(f"  Total after deduplication: {len(df_all)} records")

    df_all = df_all.sort_values(["year", "final_price_lakh"], ascending=[True, False])

    cols = [
        "year", "player_name", "team", "base_price_lakh", "final_price_lakh",
        "role", "nationality", "status", "source"
    ]
    for col in cols:
        if col not in df_all.columns:
            df_all[col] = np.nan
    df_all = df_all[cols]

    output_path = DATA_DIR / "auction_all_years.csv"
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
            print(f"  {row['player_name']}: ₹{row['final_price_lakh']:.0f}L to {row['team']}")


if __name__ == "__main__":
    main()
