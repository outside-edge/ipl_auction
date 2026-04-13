# IPL Auction Analysis: Where Do the Crores Go?

## The Big Picture

IPL teams spent **₹687 Cr** on disappointing players from 2012-2025. Using ball-by-ball data to compute WAR (Wins Above Replacement), we identify which purchases wasted the most money.

## Top Disappointments (by Wasted Money)

<!-- TABLE:disappointments:start -->
**Total wasted: ₹728 Cr** across 315 disappointing players

| Year | Player | Team | Paid | Predicted | Actual | Wasted |
| --- | --- | --- | --- | --- | --- | --- |
| 2015 | Yuvraj Singh | DC | ₹16.0Cr | 7.3 | -5.3 | ₹15.5Cr |
| 2018 | Axar Patel | PBKS | ₹12.5Cr | 5.9 | -3.1 | ₹11.1Cr |
| 2014 | Shane Watson | RR | ₹12.5Cr | 9.9 | 0.9 | ₹11.1Cr |
| 2014 | Rohit Sharma | MI | ₹12.5Cr | 5.7 | -3.3 | ₹11.0Cr |
| 2022 | Ravindra Jadeja | CSK | ₹16.0Cr | 7.5 | -1.4 | ₹10.9Cr |
| 2018 | Benjamin Stokes | RR | ₹12.5Cr | 9.5 | 1.5 | ₹9.9Cr |
| 2018 | Chris Morris | DC | ₹11.0Cr | 8.5 | 0.6 | ₹9.8Cr |
| 2014 | Shikhar Dhawan | SRH | ₹12.5Cr | 3.4 | -4.3 | ₹9.4Cr |
| 2023 | Sam Curran | PBKS | ₹18.5Cr | 6.3 | -1.2 | ₹9.3Cr |
| 2022 | Kane Williamson | SRH | ₹14.0Cr | -2.7 | -9.9 | ₹8.9Cr |
| 2018 | Rohit Sharma | MI | ₹15.0Cr | 1.2 | -6.0 | ₹8.9Cr |
| 2018 | Manish Pandey | SRH | ₹11.0Cr | -2.2 | -9.2 | ₹8.6Cr |
| 2022 | Virat Kohli | RCB | ₹15.0Cr | -2.8 | -9.6 | ₹8.3Cr |
| 2022 | Venkatesh Iyer | KKR | ₹8.0Cr | 2.4 | -7.5 | ₹8.0Cr |
| 2014 | Gautam Gambhir | KKR | ₹12.5Cr | 0.3 | -5.9 | ₹7.7Cr |
<!-- TABLE:disappointments:end -->

---

## Team Efficiency

<!-- TABLE:team_efficiency:start -->
| Team | N | Spent | WAR | WAR/Cr |
| --- | --- | --- | --- | --- |
| KKR | 67 | ₹235Cr | 350 | 1.49 |
| SRH | 73 | ₹236Cr | 302 | 1.28 |
| RR | 66 | ₹231Cr | 293 | 1.27 |
| RCB | 66 | ₹263Cr | 332 | 1.26 |
| CSK | 55 | ₹248Cr | 293 | 1.18 |
| PBKS | 73 | ₹275Cr | 324 | 1.18 |
| LSG | 20 | ₹97Cr | 105 | 1.09 |
| DC | 78 | ₹303Cr | 309 | 1.02 |
| MI | 63 | ₹255Cr | 238 | 0.93 |
<!-- TABLE:team_efficiency:end -->

---

## Backtest Results

How well can we predict player performance at auction time using only prior data?

<!-- TABLE:backtest_summary:start -->
| Year | N | R² | Rank ρ | RMSE |
| --- | --- | --- | --- | --- |
| 2012 | 9 | 0.65 | 0.90 | 4.2 |
| 2013 | 14 | -0.03 | 0.36 | 7.4 |
| 2014 | 124 | -0.00 | 0.43 | 5.4 |
| 2015 | 17 | -0.28 | 0.30 | 5.1 |
| 2016 | 43 | 0.23 | 0.54 | 5.6 |
| 2017 | 24 | -0.25 | 0.29 | 6.9 |
| 2018 | 116 | 0.22 | 0.54 | 5.0 |
| 2019 | 21 | -0.02 | 0.14 | 5.0 |
| 2020 | 27 | -0.11 | 0.22 | 4.6 |
| 2021 | 31 | 0.16 | 0.46 | 3.7 |
| 2022 | 150 | 0.27 | 0.50 | 5.4 |
| 2023 | 33 | -0.25 | 0.16 | 6.8 |
<!-- TABLE:backtest_summary:end -->

---

## Methodology

### WAR Calculation

From 3,137 player-season records of ball-by-ball data:

**Batting WAR** = (runs - expected_runs - dismissal_penalty) / RUNS_PER_WIN
- Expected runs = balls_faced × replacement_strike_rate
- Dismissal penalty = dismissals × RUNS_PER_DISMISSAL (6.0)

**Bowling WAR** = (replacement_runs - runs_conceded + wicket_bonus) / RUNS_PER_WIN
- Replacement runs = overs × replacement_economy
- Wicket bonus = wickets × RUNS_PER_WICKET (6.0)

**Replacement levels**: 15th percentile SR (batting), 80th percentile economy (bowling)

### Empirically Estimated Constants

We estimate WAR constants directly from IPL data using year-specific (contemporaneous) estimation:

| Constant | IPL Mean | Range | Derivation |
|----------|----------|-------|------------|
| RUNS_PER_WICKET | 5.8 | 5.0-7.4 | OLS regression: innings_runs ~ wickets_lost + overs_batted |
| RUNS_PER_WIN | 16.0 | 11-20 | Mean victory margin per season |

Constants are estimated separately for each IPL season to capture year-specific game dynamics (e.g., 2023 had higher runs per wicket than 2022). The overall pooled IPL estimates are RUNS_PER_WICKET=5.72 and RUNS_PER_WIN=16.04.

**Note on symmetry**: RUNS_PER_DISMISSAL and RUNS_PER_WICKET are conceptually distinct (batting opportunity cost vs. bowling value), but the innings-level regression provides a single estimate for the marginal effect of a wicket on team total. We apply this symmetrically: a dismissed batter costs their team potential runs; a bowler taking a wicket saves their team runs. This is a modeling choice that simplifies the framework while remaining empirically grounded.

### Wasted Money Calculation

For each player:
1. Predict WAR using only info available at auction (lagged IPL WAR, T20I history)
2. Compare to actual WAR
3. **Wasted = (predicted - actual) × price_per_WAR**, capped at price paid

Implied market price: **₹1.01 Cr per WAR**

### Prediction Model

XGBoost with expanding window training:
- Features: lag1/2/3 IPL WAR, T20I WAR, career totals, trends
- Training: all prior auction+outcome data
- Mean rank correlation: **0.28** (13 years evaluated)

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
