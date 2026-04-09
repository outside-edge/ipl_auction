#!/usr/bin/env python3
"""
Shared utilities for analysis scripts.
"""

import pandas as pd

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
    2026: 222.0,
}


def compute_inflation_factors(base_year=2024):
    """Compute inflation adjustment factors relative to base year."""
    base_cpi = INDIA_CPI[base_year]
    return {year: base_cpi / cpi for year, cpi in INDIA_CPI.items()}


def adjust_prices_for_inflation(df, base_year=2024):
    """Add inflation-adjusted price columns to dataframe."""
    factors = compute_inflation_factors(base_year)
    df = df.copy()
    df["inflation_factor"] = df["year"].map(factors)
    df["final_price_lakh"] = pd.to_numeric(df["final_price_lakh"], errors="coerce")
    df["base_price_lakh"] = pd.to_numeric(df.get("base_price_lakh", 0), errors="coerce")
    df["price_2024_lakh"] = df["final_price_lakh"] * df["inflation_factor"]
    df["base_price_2024_lakh"] = df["base_price_lakh"] * df["inflation_factor"]
    df["price_2024_cr"] = df["price_2024_lakh"] / 100
    df["price_nominal_cr"] = df["final_price_lakh"] / 100
    return df
