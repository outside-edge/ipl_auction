# IPL Auction Analysis: Where Do the Crores Go?

## The Big Picture

IPL teams spent **₹717 Cr** on disappointing players from 2012-2025. Using ball-by-ball data to compute WAR (Wins Above Replacement), we identify which purchases wasted the most money.

## Top Disappointments (by Wasted Money)

<!-- TABLE:disappointments:start -->
**Total wasted: ₹717 Cr** across 369 disappointing players

| Year | Player | Team | Paid | Predicted | Actual | Wasted |
| --- | --- | --- | --- | --- | --- | --- |
| 2025 | Ravi Bishnoi | LSG | ₹11.0Cr | 10.9 | -0.9 | ₹10.4Cr |
| 2014 | Gautam Gambhir | KKR | ₹12.5Cr | 14.1 | 3.3 | ₹9.5Cr |
| 2025 | Shivam Dube | CSK | ₹12.0Cr | 8.9 | -1.3 | ₹9.0Cr |
| 2025 | Tilak Varma | MI | ₹8.0Cr | 14.6 | 2.9 | ₹8.0Cr |
| 2015 | Yuvraj Singh | DC | ₹16.0Cr | 11.2 | 2.6 | ₹7.6Cr |
| 2014 | Chris Gayle | RCB | ₹7.5Cr | 15.4 | 0.4 | ₹7.5Cr |
| 2014 | Shane Watson | RR | ₹12.5Cr | 16.6 | 8.2 | ₹7.4Cr |
| 2018 | Chris Morris | DC | ₹11.0Cr | 9.2 | 0.9 | ₹7.4Cr |
| 2014 | Virat Kohli | RCB | ₹12.5Cr | 14.5 | 6.3 | ₹7.3Cr |
| 2018 | Chris Woakes | RCB | ₹7.4Cr | 6.6 | -1.7 | ₹7.3Cr |
| 2020 | Glenn Maxwell | PBKS | ₹10.8Cr | 9.3 | 1.3 | ₹7.1Cr |
| 2018 | Manish Pandey | SRH | ₹11.0Cr | 8.1 | 0.0 | ₹7.1Cr |
| 2022 | Mohammed Siraj | RCB | ₹7.0Cr | 7.1 | -4.6 | ₹7.0Cr |
| 2014 | Dale Steyn | SRH | ₹9.5Cr | 15.7 | 7.8 | ₹6.9Cr |
| 2025 | Sanju Samson | RR | ₹18.0Cr | 11.6 | 4.0 | ₹6.8Cr |
<!-- TABLE:disappointments:end -->

---

## Team Efficiency

<!-- TABLE:team_efficiency:start -->
| Team | N | Spent | WAR | WAR/Cr |
| --- | --- | --- | --- | --- |
| KKR | 34 | ₹170Cr | 221 | 1.30 |
| PBKS | 26 | ₹141Cr | 168 | 1.19 |
| DC | 30 | ₹173Cr | 180 | 1.04 |
| RCB | 27 | ₹189Cr | 185 | 0.98 |
| LSG | 24 | ₹147Cr | 142 | 0.96 |
| RR | 35 | ₹225Cr | 217 | 0.96 |
| SRH | 34 | ₹204Cr | 196 | 0.96 |
| MI | 37 | ₹249Cr | 230 | 0.92 |
| CSK | 31 | ₹238Cr | 217 | 0.91 |
<!-- TABLE:team_efficiency:end -->

---

## Backtest Results

How well can we predict player performance at auction time using only prior data?

<!-- TABLE:backtest_summary:start -->
| Year | N | R² | Rank ρ | RMSE |
| --- | --- | --- | --- | --- |
| 2012 | 10 | -0.47 | 0.35 | 6.8 |
| 2013 | 14 | -1.19 | -0.25 | 6.3 |
| 2014 | 125 | -0.05 | 0.39 | 6.2 |
| 2015 | 17 | -0.21 | 0.14 | 5.1 |
| 2016 | 43 | 0.05 | 0.24 | 4.8 |
| 2017 | 24 | -0.27 | 0.01 | 5.4 |
| 2018 | 116 | 0.16 | 0.31 | 6.4 |
| 2019 | 22 | 0.21 | 0.33 | 4.6 |
| 2020 | 28 | -0.14 | 0.24 | 5.7 |
| 2021 | 31 | 0.07 | 0.38 | 4.6 |
| 2022 | 150 | 0.17 | 0.40 | 6.4 |
| 2023 | 33 | -0.10 | 0.13 | 7.7 |
| 2025 | 43 | 0.26 | 0.52 | 6.1 |
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
