#!/usr/bin/env python3
"""
Adjust IPL auction prices for inflation using India CPI data.

Converts all prices to constant 2024 ₹ for comparability across years.
CPI data source: World Bank / RBI

Output: data/analysis/auction_inflation_adjusted.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

INDIA_CPI = {
    2008: 74.2,
    2009: 82.2,
    2010: 92.0,
    2011: 100.4,
    2012: 110.0,
    2013: 120.6,
    2014: 128.4,
    2015: 134.9,
    2016: 141.6,
    2017: 146.4,
    2018: 151.7,
    2019: 156.4,
    2020: 166.5,
    2021: 175.3,
    2022: 187.0,
    2023: 197.2,
    2024: 206.0,
    2025: 214.0,
}


def compute_inflation_factors(base_year=2024):
    """Compute inflation adjustment factors relative to base year."""
    base_cpi = INDIA_CPI[base_year]
    factors = {year: base_cpi / cpi for year, cpi in INDIA_CPI.items()}
    return factors


def adjust_auction_prices():
    """Load auction data and adjust prices for inflation."""
    print("Loading auction data...")
    df = pd.read_csv(DATA_DIR / "auction_with_performance.csv")

    print("Computing inflation factors...")
    factors = compute_inflation_factors(base_year=2024)

    df["inflation_factor"] = df["year"].map(factors)

    df["final_price_lakh"] = pd.to_numeric(df["final_price_lakh"], errors="coerce")
    df["base_price_lakh"] = pd.to_numeric(df["base_price_lakh"], errors="coerce")

    df["price_2024_lakh"] = df["final_price_lakh"] * df["inflation_factor"]
    df["base_price_2024_lakh"] = df["base_price_lakh"] * df["inflation_factor"]

    df["price_2024_cr"] = df["price_2024_lakh"] / 100
    df["price_nominal_cr"] = df["final_price_lakh"] / 100

    return df


def create_cpi_reference():
    """Create a reference CSV with CPI data."""
    cpi_df = pd.DataFrame([
        {"year": year, "cpi": cpi, "inflation_factor_2024": compute_inflation_factors(2024)[year]}
        for year, cpi in INDIA_CPI.items()
    ])
    cpi_df.to_csv(DATA_DIR / "india_cpi.csv", index=False)
    print(f"Saved CPI reference to {DATA_DIR / 'india_cpi.csv'}")
    return cpi_df


def main():
    df = adjust_auction_prices()

    create_cpi_reference()

    output_path = DATA_DIR / "analysis/auction_inflation_adjusted.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved inflation-adjusted data to {output_path}")

    print("\n=== Inflation Adjustment Summary ===")
    summary = df.groupby("year").agg({
        "final_price_lakh": "sum",
        "price_2024_lakh": "sum",
        "inflation_factor": "first",
        "player_name": "count"
    }).rename(columns={"player_name": "n_players"})
    summary["nominal_cr"] = summary["final_price_lakh"] / 100
    summary["real_2024_cr"] = summary["price_2024_lakh"] / 100
    print(summary[["n_players", "nominal_cr", "real_2024_cr", "inflation_factor"]].round(1))

    print("\n=== Top 10 Deals (Inflation-Adjusted) ===")
    top_deals = df.nlargest(10, "price_2024_lakh")[
        ["year", "player_name", "team_x", "price_nominal_cr", "price_2024_cr"]
    ]
    print(top_deals.to_string(index=False))

    print("\n=== Verification ===")
    print("MS Dhoni 2008 (nominal ₹9.5 Cr):")
    dhoni_2008 = df[(df["player_name"] == "MS Dhoni") & (df["year"] == 2008)]
    if not dhoni_2008.empty:
        nominal = dhoni_2008["price_nominal_cr"].values[0]
        real = dhoni_2008["price_2024_cr"].values[0]
        print(f"  Nominal: ₹{nominal:.2f} Cr")
        print(f"  In 2024 ₹: ₹{real:.2f} Cr")
        print(f"  Inflation factor: {dhoni_2008['inflation_factor'].values[0]:.2f}x")


if __name__ == "__main__":
    main()
