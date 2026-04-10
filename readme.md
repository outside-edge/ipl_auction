# IPL Auction Analysis: Where Do the Crores Go?

## The Big Picture

IPL teams spent **₹717 Cr** on disappointing players from 2012-2025. Using ball-by-ball data to compute WAR (Wins Above Replacement), we identify which purchases wasted the most money.

## Top Disappointments (by Wasted Money)

<!-- TABLE:disappointments:start -->
**Total wasted: ₹687 Cr** across 376 disappointing players

| Year | Player | Team | Paid | Predicted | Actual | Wasted |
| --- | --- | --- | --- | --- | --- | --- |
| 2014 | Gautam Gambhir | KKR | ₹12.5Cr | 13.9 | 3.3 | ₹8.6Cr |
| 2025 | Ravi Bishnoi | LSG | ₹11.0Cr | 9.7 | -0.9 | ₹8.6Cr |
| 2025 | Shivam Dube | CSK | ₹12.0Cr | 8.9 | -1.3 | ₹8.3Cr |
| 2025 | Tilak Varma | MI | ₹8.0Cr | 13.0 | 2.9 | ₹8.0Cr |
| 2014 | Chris Gayle | RCB | ₹7.5Cr | 15.1 | 0.4 | ₹7.5Cr |
| 2018 | Chris Morris | DC | ₹11.0Cr | 10.1 | 0.9 | ₹7.5Cr |
| 2020 | Glenn Maxwell | PBKS | ₹10.8Cr | 10.4 | 1.3 | ₹7.5Cr |
| 2025 | Sanju Samson | RR | ₹18.0Cr | 13.0 | 4.0 | ₹7.3Cr |
| 2014 | Virat Kohli | RCB | ₹12.5Cr | 15.0 | 6.3 | ₹7.1Cr |
| 2015 | Yuvraj Singh | DC | ₹16.0Cr | 11.3 | 2.6 | ₹7.1Cr |
| 2014 | Shane Watson | RR | ₹12.5Cr | 16.9 | 8.2 | ₹7.0Cr |
| 2022 | Mohammed Siraj | RCB | ₹7.0Cr | 9.8 | -4.6 | ₹7.0Cr |
| 2018 | Manish Pandey | SRH | ₹11.0Cr | 8.4 | 0.0 | ₹6.8Cr |
| 2014 | Shikhar Dhawan | SRH | ₹12.5Cr | 12.6 | 4.6 | ₹6.5Cr |
| 2022 | Anrich Nortje | DC | ₹6.5Cr | 7.7 | -1.1 | ₹6.5Cr |
<!-- TABLE:disappointments:end -->

---

## Team Efficiency

<!-- TABLE:team_efficiency:start -->
| Team | N | Spent | WAR | WAR/Cr |
| --- | --- | --- | --- | --- |
| KKR | 74 | ₹281Cr | 369 | 1.31 |
| SRH | 78 | ₹311Cr | 383 | 1.23 |
| PBKS | 76 | ₹285Cr | 351 | 1.23 |
| RCB | 69 | ₹300Cr | 350 | 1.17 |
| CSK | 60 | ₹313Cr | 364 | 1.16 |
| RR | 72 | ₹310Cr | 337 | 1.09 |
| DC | 83 | ₹348Cr | 367 | 1.06 |
| MI | 68 | ₹330Cr | 322 | 0.97 |
| LSG | 25 | ₹148Cr | 142 | 0.96 |
<!-- TABLE:team_efficiency:end -->

---

## Backtest Results

How well can we predict player performance at auction time using only prior data?

<!-- TABLE:backtest_summary:start -->
| Year | N | R² | Rank ρ | RMSE |
| --- | --- | --- | --- | --- |
| 2012 | 10 | -0.23 | 0.56 | 6.2 |
| 2013 | 14 | -1.27 | -0.27 | 6.4 |
| 2014 | 125 | -0.05 | 0.39 | 6.2 |
| 2015 | 17 | -0.11 | 0.14 | 4.8 |
| 2016 | 43 | -0.01 | 0.16 | 4.9 |
| 2017 | 24 | -0.31 | -0.06 | 5.5 |
| 2018 | 116 | 0.15 | 0.32 | 6.4 |
| 2019 | 22 | 0.19 | 0.40 | 4.6 |
| 2020 | 28 | -0.17 | 0.26 | 5.8 |
| 2021 | 31 | -0.00 | 0.42 | 4.8 |
| 2022 | 150 | 0.12 | 0.37 | 6.6 |
| 2023 | 33 | 0.03 | 0.27 | 7.2 |
| 2025 | 43 | 0.28 | 0.54 | 6.0 |
<!-- TABLE:backtest_summary:end -->

---

## Methodology

### WAR Calculation

From 3,137 player-season records of ball-by-ball data:

- **Batting WAR** = (actual runs − replacement runs) / 8
- **Bowling WAR** = (replacement runs − actual runs conceded) / 8
- Replacement level: 15th percentile (batting), 80th percentile (bowling)

### Wasted Money Calculation

For each player:
1. Predict WAR using only info available at auction (lagged IPL WAR, T20I history)
2. Compare to actual WAR
3. **Wasted = (predicted - actual) × price_per_WAR**, capped at price paid

Implied market price: **₹0.88 Cr per WAR**

### Prediction Model

XGBoost with expanding window training:
- Features: lag1/2/3 IPL WAR, T20I WAR, career totals, trends
- Training: all prior auction+outcome data
- Mean rank correlation: **0.25** (13 years evaluated)

---

## Repository Structure

```
ipl_auction/
├── data/
│   ├── acquisitions/         # Auction price data
│   └── perf/                 # IPL/T20I WAR by season
├── scripts/
│   ├── auction/              # Scraping, assembly
│   ├── perf/                 # Performance data processing
│   ├── prediction/           # WAR prediction, backtest
│   ├── retrospective/        # Hedonic regression
│   └── shared/               # Utilities
├── tabs/                     # Output tables (CSV)
│   ├── disappointments.csv   # Sorted by wasted money
│   ├── team_efficiency.csv
│   └── retroactive_*.csv
└── .github/workflows/        # Auto-update README
```

## Quick Start

```bash
# Full pipeline
make all

# Just analysis
python3 scripts/prediction/06_comprehensive_backtest.py
python3 scripts/prediction/07_economic_analysis.py

# Update README tables
python3 scripts/update_readme_tables.py
```

## Data Sources

| Source | Coverage |
|--------|----------|
| Kaggle, Wikipedia, IPL official | Auction prices 2008-2026 |
| Kaggle ball-by-ball | Match data 2008-2025 |
| World Bank/RBI | CPI for inflation adjustment |

## Limitations

1. **Missing years**: 2024 auction data not available
2. **Selection bias**: ~19% of auctioned players never fielded
3. **Private information**: Teams have scouting data we don't

## References

- Rosen, S. (1974). Hedonic prices and implicit markets. *JPE*.
- Scully, G. W. (1974). Pay and performance in MLB. *AER*.
